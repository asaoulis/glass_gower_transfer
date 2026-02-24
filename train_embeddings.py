import sys
from typing import List, Dict

import torch
import pytorch_lightning as pl
import wandb

from config.default import get_default_config
from config.experiments import experiments

from src.ml.utils import prepare_data_parameters, build_model
from src.ml.models.lightning_modules import NDELightningModule
from src.ml.eval.utils import load_best_model_and_build_posterior, evaluate_best_checkpoint


class IdentityEmbedding(torch.nn.Module):
    """Simple identity embedding that returns its input as-is.

    This is used so that we can reuse the existing NDELightningModule API,
    which expects an `embedding_net` that maps conditioning data to a
    fixed-size vector. Here, the conditioning data will already be a
    concatenated embedding tensor.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x

    def compress(self, x):
        """For compatibility with encoders that expose `.compress()`.

        Returns (mu, logvar) style tuple, but for plain identity we treat
        the input as `mu` and return zeros for `logvar`.
        """
        mu = x
        logvar = torch.zeros_like(mu)
        return mu, logvar


class EmbeddingDataset(torch.utils.data.Dataset):
    """Dataset of precomputed embeddings and corresponding cosmological params."""

    def __init__(self, embeddings: torch.Tensor, cosmo: torch.Tensor):
        assert embeddings.shape[0] == cosmo.shape[0]
        self.embeddings = embeddings
        self.cosmo = cosmo

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return self.embeddings[idx], self.cosmo[idx]


def _build_config_for_experiment(name: str):
    """Create a config object for a given experiment name.

    Mirrors the logic in train.py: all non-list values are set
    directly, while max_trainval_cosmos is handled separately so
    it can be either scalar or list in experiments.py.
    """
    if name not in experiments:
        raise ValueError(f"Experiment '{name}' not found in config.experiments.experiments.")

    cfg = get_default_config()
    cfg.experiment_name = name
    exp_cfg = experiments[name]

    # Set all non-list values directly on the config, but skip max_trainval_cosmos
    for k, v in exp_cfg.items():
        if k == "max_trainval_cosmos":
            continue
        setattr(cfg, k, v)

    return cfg


def load_pretrained_models(exp_names: List[str]):
    """Build models for a list of experiment names using their best checkpoints.

    Uses `load_best_model_and_build_posterior`, which encapsulates the
    checkpoint directory layout and filename logic.

    Returns:
        models: list of trained NDE models
        cfgs:   list of per-model configs
    """
    models = []
    cfgs = []
    dataset_quantities = set()
    # need to return dataset_quantities as a list
    for name in exp_names:
        cfg = _build_config_for_experiment(name)
        result = load_best_model_and_build_posterior(cfg)
        if result is None:
            raise RuntimeError(f"Failed to load best model for experiment '{name}'.")
        model, _ = result
        model.eval()
        models.append(model)
        dataset_quantities.update(getattr(cfg, "dataset_quantities", []))
    dataset_quantities = list(dataset_quantities)
    return models, dataset_quantities


@torch.no_grad()
def compute_embeddings(models: List[torch.nn.Module], loader: torch.utils.data.DataLoader):
    """Compute concatenated embeddings and corresponding cosmology vectors for one loader."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [m.to(device) for m in models]

    all_z = []
    all_theta = []
    for batch_idx, (data_dict, theta) in enumerate(loader):
        # Move batch to device once
        if isinstance(data_dict, dict):
            data_dict_dev = {k: v.to(device) for k, v in data_dict.items()}
        else:
            data_dict_dev = data_dict.to(device)
        theta_dev = theta.to(device)

        # Run all models on this batch and concatenate embeddings feature-wise
        zs_batch = []
        theta_ref = None
        for m in models:
            encoder = m.embedding_net
            encoder.only_return_mu = True
            z_m = encoder.get_representation(data_dict_dev)
            zs_batch.append(z_m.detach())

            # Extract cosmology from this model for sanity check
            theta_m = theta_dev  # all models see same loader, so theta is shared
            if theta_ref is None:
                theta_ref = theta_m
            else:
                if theta_ref.shape != theta_m.shape or not torch.allclose(theta_ref, theta_m):
                    raise RuntimeError(
                        f"Cosmo vectors differ between models at batch {batch_idx}; "
                        f"shapes {theta_ref.shape} vs {theta_m.shape}."
                    )

        z_cat_batch = torch.cat(zs_batch, dim=-1).cpu()
        all_z.append(z_cat_batch)
        all_theta.append(theta_ref.cpu())

    z_cat = torch.cat(all_z, dim=0)
    theta_cat = torch.cat(all_theta, dim=0)
    return z_cat, theta_cat


def build_embedding_dataloaders(train_loader, val_loader, test_loader, models: List[torch.nn.Module]):
    """Use the *same* train/val/test loaders that were built with prepare_data_parameters
    (and thus respect max_trainval_cosmos) to construct embedding datasets.
    """
    train_z, train_theta = compute_embeddings(models, train_loader)
    val_z, val_theta = compute_embeddings(models, val_loader)
    test_z, test_theta = compute_embeddings(models, test_loader)

    train_ds = EmbeddingDataset(train_z, train_theta)
    val_ds = EmbeddingDataset(val_z, val_theta)
    test_ds = EmbeddingDataset(test_z, test_theta)

    batch_size = getattr(train_loader, "batch_size", 128)
    num_workers = getattr(train_loader, "num_workers", 0)

    train_emb_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_emb_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_emb_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_emb_loader, val_emb_loader, test_emb_loader


def fit_nde_on_embeddings(emb_dim: int, train_loader, val_loader, test_loader, base_cfg):
    """Build and train an NDELightningModule on precomputed embeddings.

    Uses an IdentityEmbedding as the encoder and otherwise reuses the
    flow configuration from `base_cfg`.
    """
    from src.ml.models.lightning_modules import NDELightningModule

    conditioning_dim = emb_dim
    inference_dim = len(base_cfg.cosmo_param_names)

    embedding_net = IdentityEmbedding(emb_dim).to("cuda" if torch.cuda.is_available() else "cpu")

    model = NDELightningModule(
        embedding_net,
        conditioning_dim=conditioning_dim,
        inference_dim=inference_dim,
        lr=base_cfg.lr,
        flow_type=base_cfg.flow_type,
        scheduler_type=base_cfg.scheduler_type,
        element_names=["Omega", "sigma8"],
        test_dataloader=test_loader,
        optimizer_kwargs=base_cfg.optimizer_kwargs,
        num_extra_blocks=base_cfg.extra_blocks,
        freeze_CNN=False,
        scheduler_kwargs=base_cfg.scheduler_kwargs,
        flow_kwargs=base_cfg.flow_kwargs,
    )

    # Standard Lightning trainer setup (similar to fit_model)
    num_gpus = torch.cuda.device_count()
    accelerator = "gpu" if num_gpus > 0 else "cpu"
    devices = num_gpus if num_gpus > 0 else 1
    strategy = "ddp" if num_gpus > 1 else "auto"

    monitor_string = f"val_{model.loss_name}"
    num_trainval_cosmos = getattr(base_cfg, 'max_trainval_cosmos', None)
    match_string_logger = base_cfg.match_string if base_cfg.match_string else ""

    wandb_logger = wandb.init(
        project=getattr(base_cfg, "project", "emb-nde"),
        group="embeddings_nde",
        name=(
            f"{base_cfg.experiment_name}/"
            f"{base_cfg.model_type}_{match_string_logger}_"
            f"ncosmo{num_trainval_cosmos}"
        ),       
        reinit=True,
    )

    wandb_logger_name = wandb_logger.name if wandb_logger else "no_logger" 
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=f"{monitor_string}",
        dirpath=f"{base_cfg.base_path}/checkpoints/{base_cfg.experiment_name}/run_{wandb_logger_name}",
        filename=f"checkpoint-{{epoch:02d}}-{{{monitor_string}:.4f}}",
        save_top_k=3,
        mode="min",
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=base_cfg.epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        logger=pl.loggers.WandbLogger() if wandb_logger else None,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=0.5,
        precision="bf16",
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train_embeddings.py <target_experiment_name> <source_exp1> [<source_exp2> ...]")
        sys.exit(1)

    target_experiment = sys.argv[1]
    source_experiments = sys.argv[2:]

    # Load target experiment config and apply the same max_trainval_cosmos logic as in train.py
    if target_experiment not in experiments:
        raise ValueError(f"Experiment '{target_experiment}' not found in config.experiments.experiments.")

    target_exp_cfg = experiments[target_experiment]

    # Helper to build a cfg with all non-list values applied and a given max_trainval_cosmos
    def _build_target_cfg(n_cosmo=None):
        cfg = get_default_config()
        cfg.experiment_name = target_experiment
        for k, v in target_exp_cfg.items():
            if k == "max_trainval_cosmos":
                continue
            setattr(cfg, k, v)
        if n_cosmo is not None:
            cfg.max_trainval_cosmos = int(n_cosmo)
            cfg.match_string = f"ncosmo{n_cosmo}"
        return cfg

    max_tv = target_exp_cfg.get("max_trainval_cosmos", None)

    def _run_single(target_cfg_local):
        # Load source models (pretrained representation providers)
        models, dataset_quantities = load_pretrained_models(source_experiments)
        # loop over cfgs and sum data patterns
        target_cfg_local.dataset_quantities = dataset_quantities
        target_cfg_local.test_shape_noise_idx = [0]
        # Use *target* cfg to build dataloaders with correct max_trainval_cosmos and scalers
        scalers, train_loader, val_loader, test_loader = prepare_data_parameters(target_cfg_local)

        # Build embedding dataloaders on top of those loaders
        train_emb_loader, val_emb_loader, test_emb_loader = build_embedding_dataloaders(
            train_loader, val_loader, test_loader, models
        )

        # Embedding dimension is last dimension of one batch from train_emb_loader
        sample_batch = next(iter(train_emb_loader))[0]
        emb_dim = sample_batch.shape[-1]

        fit_nde_on_embeddings(emb_dim, train_emb_loader, val_emb_loader, test_emb_loader, target_cfg_local)

        # Custom model builder for evaluation: build embedding-based NDE from checkpoint
        def emb_model_builder(cfg, loader):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Identity embedding with the same embedding dimension
            embedding_net = IdentityEmbedding(emb_dim).to(device)

            conditioning_dim = emb_dim
            inference_dim = len(cfg.cosmo_param_names)

            model = NDELightningModule(
                embedding_net,
                conditioning_dim=conditioning_dim,
                inference_dim=inference_dim,
                lr=cfg.lr,
                flow_type=cfg.flow_type,
                scheduler_type=cfg.scheduler_type,
                element_names=["Omega", "sigma8"],
                test_dataloader=loader,
                optimizer_kwargs=cfg.optimizer_kwargs,
                num_extra_blocks=cfg.extra_blocks,
                freeze_CNN=False,
                scheduler_kwargs=cfg.scheduler_kwargs,
                flow_kwargs=cfg.flow_kwargs,
            )

            # If a checkpoint_path is set on cfg, load the weights
            checkpoint_path = getattr(cfg, "checkpoint_path", None)
            if checkpoint_path:
                ckpt = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(ckpt["state_dict"], strict=False)

            model.to(device)
            model.eval()
            return model

        evaluate_best_checkpoint(
            target_cfg_local,
            test_emb_loader,
            scalers["cosmo"],
            reference_samples=None,
            model_builder=emb_model_builder,
        )

    if isinstance(max_tv, (list, tuple)):
        for n_cosmo in max_tv:
            print(f"Running embedding experiment '{target_experiment}' with max_trainval_cosmos={n_cosmo}")
            cfg_copy = _build_target_cfg(n_cosmo)
            _run_single(cfg_copy)
    else:
        cfg_single = _build_target_cfg(max_tv)
        print(f"Running embedding experiment '{target_experiment}'")
        _run_single(cfg_single)
