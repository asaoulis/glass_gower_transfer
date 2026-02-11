import torch
import wandb
import pytorch_lightning as pl
import os
from ..eval.utils import find_best_checkpoint, get_best_checkpoint
from ..utils import prepare_data_and_model

def fit_model(model, epochs, logger, train_loader, val_loader, experiment_name, base_path):
    # Determine available GPUs and accelerator/strategy
    num_gpus = torch.cuda.device_count()
    accelerator = "gpu" if num_gpus > 0 else "cpu"
    devices = num_gpus if num_gpus > 0 else 1
    strategy = "ddp" if num_gpus > 1 else "auto"

    monitor_string = f"val_{model.loss_name}"
    # get logger name
    wandb_logger_name = logger.name if logger else "no_logger" 
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=f"{monitor_string}",
        dirpath=f"{base_path}/checkpoints/{experiment_name}/run_{wandb_logger_name}",
        filename=f"checkpoint-{{epoch:02d}}-{{{monitor_string}:.4f}}",
        save_top_k=3,
        mode="min",
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        logger=pl.loggers.WandbLogger() if logger else None,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=0.5,     
        precision="bf16"
    )
    
    trainer.fit(model, train_loader, val_loader)


def train_model(config):
    # Scale batch size by the number of available GPUs (if attribute exists)
    pretrain = config.checkpoint_path is None
    original_checkpoint_path = config.checkpoint_path
    if not pretrain:
        best_checkpoints, _ = get_best_checkpoint(config.checkpoint_path, config.match_string)
    for i in range(config.repeats):
        config.checkpoint_path = best_checkpoints[i] if not pretrain else None
        loaders, model, _ = prepare_data_and_model(config)
        train_loader, val_loader, _ = loaders

        # Use configured max_trainval_cosmos directly for logging (may be None)
        num_trainval_cosmos = getattr(config, 'max_trainval_cosmos', None)

        match_string_logger = config.match_string if config.match_string else ""

        logger = wandb.init(
            project=config.project,
            group=config.experiment_name,
            name=(
                f"{config.experiment_name}/"
                f"{'pretrain' if pretrain else 'finetune'}_"
                f"{config.model_type}_{match_string_logger}_"
                f"ncosmo{num_trainval_cosmos}_{i}"
            ),
            reinit=True,
        )
        fit_model(model, config.epochs, logger, train_loader, val_loader, config.experiment_name, config.base_path)
        config.checkpoint_path = original_checkpoint_path  # Reset checkpoint path for next iteration if needed
