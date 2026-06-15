import os
import torch
import wandb
import pytorch_lightning as pl
from ..eval.loading_model import get_best_checkpoint
from ..utils import prepare_data_and_model, set_seed_for_repeat_and_ensemble

def create_run_name(config, match_string_logger):
    pretrain = config.checkpoint_path is None
    run_name = (
        f"{config.experiment_name}/"
        f"{'pretrain' if pretrain else 'finetune'}_"
        f"{match_string_logger}"
    )
    return run_name


def apply_repeat_config(config, repeat_idx: int):
    """Apply the same per-repeat config logic used by training.

    This mutates `config` in-place and returns (repeat_match, run_string).

    Mirrors the logic previously embedded in `train_model`:
      - split_seed = base_seed + i
      - repeat_match = f"ncosmo{max_trainval_cosmos}_{i}" if enabled else f"_{i}"
      - run_string  = f"ncosmo{max_trainval_cosmos}_{i}" (used for logger/run naming)
      - config.pretrained_band_match_string = repeat_match
    """
    match_num_cosmo = getattr(config, "match_num_cosmo", True)
    base_seed = getattr(config, "split_seed", 42)

    config.split_seed = base_seed + repeat_idx

    num_trainval_cosmos = getattr(config, "max_trainval_cosmos", None)
    if num_trainval_cosmos is not None and match_num_cosmo:
        repeat_match = f"ncosmo{num_trainval_cosmos}_{repeat_idx}"
    else:
        repeat_match = f"_{repeat_idx}"

    run_string = f"ncosmo{num_trainval_cosmos}_{repeat_idx}"

    config.pretrained_band_match_string = repeat_match
    return repeat_match, run_string


def maybe_resolve_repeat_checkpoint(config, original_checkpoint_path, repeat_match: str):
    """If config is in finetune mode, resolve best checkpoint for this repeat."""
    pretrain = original_checkpoint_path is None
    if (not pretrain) and original_checkpoint_path is not None:
        best_checkpoint, _ = get_best_checkpoint(original_checkpoint_path, repeat_match)
        config.checkpoint_path = best_checkpoint[0] if best_checkpoint else None
    else:
        config.checkpoint_path = None


def is_rank_zero() -> bool:
    """True only on global rank 0 (works for torchrun / srun / Lightning)."""
    return int(os.environ.get("LOCAL_RANK", 0)) == 0

def fit_model(
    model,
    epochs,
    wandb_logger,
    train_loader,
    val_loader,
    experiment_name,
    run_name,
    base_path,
    accumulate_grad_batches=1
):
    # Accelerator / strategy
    num_gpus = torch.cuda.device_count()
    accelerator = "gpu" if num_gpus > 0 else "cpu"
    devices = num_gpus if num_gpus > 0 else 1
    strategy = "ddp" if num_gpus > 1 else "auto"

    # Expose distributed flag on model
    is_distributed = strategy == "ddp"
    model.is_distributed = is_distributed
    if hasattr(model, "hparams"):
        model.hparams.is_distributed = is_distributed

    monitor_string = f"val_{model.loss_name}"

    # Checkpointing (rank-safe, logger-independent)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor_string,
        dirpath=f"{base_path}/checkpoints/{run_name}",
        filename=f"checkpoint-{{epoch:02d}}-{{{monitor_string}:.4f}}",
        save_top_k=3,
        mode="min",
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    # Precision: when ml_perf.amp scopes bf16 to the map encoder (model.model.amp_encoder),
    # Lightning must run fp32 — a SECOND whole-forward autocast (precision='bf16-mixed') would
    # push the flow's rational-quadratic spline to bf16 and crash at its index_put. The scoped
    # autocast already gives the bf16 tensor-core/bandwidth win on the encoder (the 90% cost).
    amp_encoder_on = bool(getattr(getattr(model, "model", None), "amp_encoder", False))
    precision = "32" if amp_encoder_on else "bf16-mixed"

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=wandb_logger,          # None on non-zero ranks
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        gradient_clip_val=0.5,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    trainer.fit(model, train_loader, val_loader)

def train_model(config):
    """Train a model, optionally over multiple repeats and/or an ensemble.

    Repeat behaviour:
      - match strings (repeat_match) are bound ONLY to repeat idx (and ncosmo if enabled)
      - split_seed is set ONLY by repeat idx

    Ensemble behaviour:
      - ensemble members share split_seed (repeat-bound)
      - ensemble_seed is set per member, and only reshuffles train/val ordering
        within the selected trainval_cosmos subset.

    Notes
    -----
    Repeat selection:
      - if `config.repeat_indices` is set (list/tuple of ints), iterate those indices
      - else iterate `range(config.repeats)`
    """
    from copy import copy

    original_checkpoint_path = getattr(config, "checkpoint_path", None)

    num_gpus = torch.cuda.device_count()
    config.is_distributed = num_gpus > 1

    repeat_indices_cfg = getattr(config, "repeat_indices", None)
    if repeat_indices_cfg is not None:
        if not isinstance(repeat_indices_cfg, (list, tuple)):
            raise TypeError(
                f"config.repeat_indices must be a list/tuple of ints or None, got {type(repeat_indices_cfg)}"
            )
        repeat_indices = [int(x) for x in repeat_indices_cfg]
    else:
        n_repeats = int(getattr(config, "repeats", 1) or 1)
        repeat_indices = list(range(n_repeats))

    if any(i < 0 for i in repeat_indices):
        raise ValueError(f"Repeat indices must be non-negative, got {repeat_indices}")

    ensemble_repeats = int(getattr(config, "ensemble_repeats", 1) or 1)

    base_seed = int(getattr(config, "split_seed", 42))

    for i in repeat_indices:
        # Work on a per-repeat copy so we don't accumulate mutations.
        cfg_repeat = copy(config)
        cfg_repeat.split_seed = base_seed
        cfg_repeat.ensemble_seed = None

        # Apply repeat match behaviour (also sets cfg_repeat.split_seed = base_seed + i)
        repeat_match, repeat_run_string = apply_repeat_config(cfg_repeat, i)

        # Resolve checkpoint ONCE per repeat, bound to repeat_match
        maybe_resolve_repeat_checkpoint(cfg_repeat, original_checkpoint_path, repeat_match)

        for j in range(ensemble_repeats):
            cfg = copy(cfg_repeat)
            # Ensure repeat seed is the only thing affecting split_seed
            cfg.split_seed = base_seed
            set_seed_for_repeat_and_ensemble(cfg, repeat_idx=i, ensemble_idx=j)

            # Run naming: include _ens{j} only for logging/checkpoint dirs
            run_string = (
                f"{repeat_run_string}_ens{j}" if ensemble_repeats > 1 else repeat_run_string
            )

            print(
                f"[Repeat {i} | Ensemble {j}] split_seed={cfg.split_seed} ensemble_seed={getattr(cfg, 'ensemble_seed', None)}",
                flush=True,
            )
            print("Will try to use checkpoint:", cfg.checkpoint_path, flush=True)

            loaders, model, _ = prepare_data_and_model(cfg)
            train_loader, val_loader, _ = loaders

            run_name = create_run_name(cfg, run_string)

            wandb_logger = None
            if is_rank_zero():
                wandb.finish()
                wandb_logger = pl.loggers.WandbLogger(
                    project=cfg.project,
                    group=cfg.experiment_name,
                    name=run_name,
                    log_model=False,
                    reinit=True,
                )
            accumulate_grad_batches = getattr(cfg, "accumulate_grad_batches", 1)
            fit_model(
                model=model,
                epochs=cfg.epochs,
                wandb_logger=wandb_logger,
                train_loader=train_loader,
                val_loader=val_loader,
                experiment_name=cfg.experiment_name,
                run_name=run_name,
                base_path=cfg.base_path,
                accumulate_grad_batches=accumulate_grad_batches,
            )
            if wandb_logger is not None:
                wandb.finish()

def parse_repeat_and_ensemble_from_match(match_string: str):
    """Parse repeat/ensemble indices from a match/run string.

    Expected patterns:
      - '..._<repeat>'
      - '..._<repeat>_ens<ens>'

    Returns (repeat_idx, ensemble_idx). Ensemble defaults to 0.
    """
    import re

    m = re.search(r"_(\d+)(?:_ens(\d+))?$", str(match_string))
    if not m:
        raise ValueError(f"Could not parse repeat/ensemble idx from '{match_string}'")
    repeat_idx = int(m.group(1))
    ensemble_idx = int(m.group(2)) if m.group(2) is not None else 0
    return repeat_idx, ensemble_idx