import torch
import wandb
import pytorch_lightning as pl
import os
from ..eval.utils import find_best_checkpoint, get_best_checkpoint
from ..utils import prepare_data_and_model

def fit_model(model, epochs, logger, train_loader, val_loader, experiment_name):
    monitor_string = f"val_{model.loss_name}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=f"{monitor_string}",
        dirpath=f"/share/gpu0/asaoulis/cmd/checkpoints/{experiment_name}/run_{wandb.run.name}",
        filename=f"checkpoint-{{epoch:02d}}-{{{monitor_string}:.4f}}",
        save_top_k=5,
        mode="min",
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        logger=pl.loggers.WandbLogger() if logger else None,
        callbacks=[checkpoint_callback, lr_monitor],
    )
    
    trainer.fit(model, train_loader, val_loader)


def train_model(config):
    pretrain = config.checkpoint_path is None
    if not pretrain:
        best_checkpoints, _ = get_best_checkpoint(config.checkpoint_path, config.match_string)
    for i in range(config.repeats):
        config.checkpoint_path = best_checkpoints[i] if not pretrain else None
        train_loader, val_loader, model, _ = prepare_data_and_model(config)
        match_string_logger = config.match_string if config.match_string else ""
        logger = wandb.init(
            project="camels-nbody-illustris-paper-SB",
            group=config.experiment_name,
            name=f"{'pretrain' if pretrain else 'finetune'}_{config.dataset_name}_{config.dataset_suite}_{config.scheduler_type}_{config.lr}_{match_string_logger}_ds{config.dataset_size}_{i}",
            reinit=True
        )

        fit_model(model, config.epochs, logger, train_loader, val_loader, config.experiment_name)

