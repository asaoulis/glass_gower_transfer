import os
import torch
import wandb
import pytorch_lightning as pl
from ..eval.utils import get_best_checkpoint
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
        precision="bf16-mixed",
    )

    trainer.fit(model, train_loader, val_loader)

def train_model(config):
    """Train a model, optionally over multiple repeats and/or an ensemble.

    Outer loop (`repeats`) preserves the existing behavior:
      - split_seed = base_seed + i
      - repeat_match / pretrained_band_match_string are bound to repeat idx `i`
      - checkpoint resolution uses ONLY the repeat match string

    If `config.ensemble_repeats` is set (>1), we train that many ensemble members
    *per outer repeat*. Ensemble members:
      - do NOT change repeat_match behavior
      - DO get a distinct seed so data splits differ between ensemble members
      - DO get a distinct run_string/run_name for logging + checkpoint directories
    """
    # Determine whether this is a fresh training run or a fine-tune-from-checkpoint
    original_checkpoint_path = getattr(config, "checkpoint_path", None)

    # Number of GPUs controls distributed flag on the config (used downstream)
    num_gpus = torch.cuda.device_count()
    config.is_distributed = num_gpus > 1

    # Number of repeats (default 1 if missing)
    repeats = getattr(config, "repeats", 1)

    # Ensemble size per repeat (default 1)
    ensemble_repeats = int(getattr(config, "ensemble_repeats", 1) or 1)

    # Cache values we mutate so we can restore afterwards
    original_split_seed = getattr(config, "split_seed", 42)
    original_pretrained_band_match = getattr(config, "pretrained_band_match_string", None)

    for i in range(repeats):
        # Apply the original per-repeat logic (also sets split_seed)
        repeat_match, repeat_run_string = apply_repeat_config(config, i)
        repeat_base_seed = config.split_seed

        # Resolve checkpoint ONCE per repeat (bound to repeat idx)
        maybe_resolve_repeat_checkpoint(config, original_checkpoint_path, repeat_match)

        for j in range(ensemble_repeats):
            config.split_seed = set_seed_for_repeat_and_ensemble(config, repeat_idx=i, ensemble_idx=j)
            # Distinguish run names/checkpoint dirs by ensemble member
            if ensemble_repeats > 1:
                run_string = f"{repeat_run_string}_ens{j}"
            else:
                run_string = repeat_run_string

            print(
                f"[Repeat {i} | Ensemble {j}] split_seed={config.split_seed}",
                flush=True,
            )
            print("Will try to use checkpoint:", config.checkpoint_path, flush=True)

            # Prepare data and model with the updated config (including split_seed)
            loaders, model, _ = prepare_data_and_model(config)
            train_loader, val_loader, _ = loaders

            match_string_logger = run_string
            run_name = create_run_name(config, match_string_logger)

            # WandB: rank-0 only, Lightning-managed
            wandb_logger = None
            if is_rank_zero():
                wandb.finish()
                wandb_logger = pl.loggers.WandbLogger(
                    project=config.project,
                    group=config.experiment_name,
                    name=run_name,
                    log_model=False,
                    reinit=True,   # ← IMPORTANT
                )

            fit_model(
                model=model,
                epochs=config.epochs,
                wandb_logger=wandb_logger,
                train_loader=train_loader,
                val_loader=val_loader,
                experiment_name=config.experiment_name,
                run_name=run_name,
                base_path=config.base_path,
            )
            if wandb_logger is not None:
                wandb.finish()

        # restore
            config.split_seed = original_split_seed
        config.checkpoint_path = original_checkpoint_path
        config.pretrained_band_match_string = original_pretrained_band_match

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