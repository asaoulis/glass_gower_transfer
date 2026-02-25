import os
import torch
import wandb
import pytorch_lightning as pl
from ..eval.utils import get_best_checkpoint
from ..utils import prepare_data_and_model

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
    pretrain = config.checkpoint_path is None
    original_checkpoint_path = config.checkpoint_path

    if not pretrain:
        best_checkpoints, _ = get_best_checkpoint(
            config.checkpoint_path, config.match_string
        )

    num_gpus = torch.cuda.device_count()
    config.is_distributed = num_gpus > 1

    for i in range(config.repeats):
        config.checkpoint_path = best_checkpoints[i] if not pretrain else None
        print("Will try to use checkpoint:", config.checkpoint_path, flush=True)

        loaders, model, _ = prepare_data_and_model(config)
        train_loader, val_loader, _ = loaders

        num_trainval_cosmos = getattr(config, "max_trainval_cosmos", None)
        match_string_logger = config.match_string or ""

        run_name = (
            f"{config.experiment_name}/"
            f"{'pretrain' if pretrain else 'finetune'}_"
            f"{config.model_type}_{match_string_logger}_"
            f"ncosmo{num_trainval_cosmos}_{i}"
        )

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

        # Reset for next repeat
        config.checkpoint_path = original_checkpoint_path