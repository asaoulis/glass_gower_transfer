import os
import torch
import wandb
import pytorch_lightning as pl
from ..eval.utils import get_best_checkpoint
from ..utils import prepare_data_and_model

def create_run_name(config, match_string_logger):
    pretrain = config.checkpoint_path is None
    run_name = (
        f"{config.experiment_name}/"
        f"{'pretrain' if pretrain else 'finetune'}_"
        f"{config.model_type}_{match_string_logger}"
    )
    return run_name

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
        dirpath=f"{base_path}/checkpoints/{experiment_name}/{run_name}",
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
    """Train a model, optionally over multiple repeats.

    For each repeat `i` we:
      - set a deterministic split seed = 42 + i so data splits differ
      - construct a repeat-specific match string `ncosmo{ncosmo}_{i}`
      - optionally resolve a checkpoint using that match string
      - set `pretrained_band_match_string` likewise for partial-model loading
    """
    # Determine whether this is a fresh training run or a fine-tune-from-checkpoint
    pretrain = config.checkpoint_path is None
    original_checkpoint_path = config.checkpoint_path

    # Number of GPUs controls distributed flag on the config (used downstream)
    num_gpus = torch.cuda.device_count()
    config.is_distributed = num_gpus > 1

    # Number of repeats (default 1 if missing)
    repeats = getattr(config, "repeats", 1)
    match_num_cosmo = getattr(config, "match_num_cosmo", True)
    # Cache original split_seed and pretrained_band_match_string so we can restore afterwards
    base_seed = getattr(config, "split_seed", 42)
    original_pretrained_band_match = getattr(config, "pretrained_band_match_string", None)

    for i in range(repeats):
        # Per-repeat split seed: 42 + repeat index (or base_seed + i if base_seed was overridden)
        config.split_seed = base_seed + i
        # Per-repeat cosmology count (may be None)
        num_trainval_cosmos = getattr(config, "max_trainval_cosmos", None)
        if num_trainval_cosmos is not None and match_num_cosmo:
            repeat_match = f"ncosmo{num_trainval_cosmos}_{i}"
        else:
            repeat_match = f"_{i}"
        run_string = f"ncosmo{num_trainval_cosmos}_{i}"


        config.pretrained_band_match_string = repeat_match

        if not pretrain and original_checkpoint_path is not None:
            best_checkpoint, _ = get_best_checkpoint(
                original_checkpoint_path,
                repeat_match,
            )
            config.checkpoint_path = best_checkpoint[0]
        else:
            config.checkpoint_path = None

        print(f"[Repeat {i}] split_seed={config.split_seed}", flush=True)
        print("Will try to use checkpoint:", config.checkpoint_path, flush=True)

        # Prepare data and model with the updated config (including split_seed and band match string)
        loaders, model, _ = prepare_data_and_model(config)
        train_loader, val_loader, _ = loaders

        match_string_logger = run_string
        run_name = create_run_name(config, match_string_logger)

        # WandB: rank-0 only, Lightning-managed
        wandb_logger = None
        if is_rank_zero():
            wandb_logger = pl.loggers.WandbLogger(
                project=config.project,
                group=config.experiment_name,
                name=run_name,
                log_model=False,
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
        config.checkpoint_path = original_checkpoint_path
        config.pretrained_band_match_string = original_pretrained_band_match