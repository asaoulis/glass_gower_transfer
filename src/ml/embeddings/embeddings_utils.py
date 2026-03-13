import os
import sys
from typing import List, Optional, Tuple

import torch
import pytorch_lightning as pl
import wandb

from config.default import get_default_config
from config.experiments import experiments

from ..models.lightning_modules import NDELightningModule, LikelihoodNDELightningModule
from ..eval.utils import load_best_model_and_build_posterior
from ..data.scaling import BaseScaler, PerDimStandardScaler
from ..utils import _build_cosmo_preset_scaler
from ..data.constants import COSMO_PARAM_PRESET_MINMAX


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
    """Dataset of precomputed embeddings and corresponding cosmological params.

    Embeddings are scaled per-feature using PerDimStandardScaler.
    Cosmological parameters are scaled with a preset MinMax scaler
    constructed via _build_cosmo_preset_scaler and COSMO_PARAM_PRESET_MINMAX.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        cosmo: torch.Tensor,
        emb_scaler: Optional[BaseScaler] = None,
        cosmo_scaler: Optional[BaseScaler] = None,
    ):
        assert embeddings.shape[0] == cosmo.shape[0]
        self.embeddings = embeddings
        self.cosmo = cosmo

        # Fit or store embedding scaler (per-variable standard scaling)
        if emb_scaler is None:
            emb_scaler = PerDimStandardScaler()
            emb_scaler.fit(self.embeddings)
        self.emb_scaler = emb_scaler

        # Store cosmology scaler (preset MinMax)
        self.cosmo_scaler = cosmo_scaler

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        z = self.embeddings[idx]
        theta = self.cosmo[idx]

        z_scaled = self.emb_scaler.transform(z)
        theta_scaled = self.cosmo_scaler.transform(theta) if self.cosmo_scaler is not None else theta
        return z_scaled, theta_scaled


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

    # Default inference mode for embeddings is 'npe' (posterior).
    if not hasattr(cfg, "inference_mode"):
        cfg.inference_mode = "npe"

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
            z_m = encoder(data_dict_dev)
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


def _get_embedding_cache_paths(base_cfg, wandb_run_name: str) -> Tuple[str, str, str]:
    """Return file paths for cached embedding datasets under checkpoint dir."""
    base_dir = f"{base_cfg.base_path}/checkpoints/{base_cfg.experiment_name}/{base_cfg.experiment_name}/{wandb_run_name}"
    train_path = f"{base_dir}/emb_train.pt"
    val_path = f"{base_dir}/emb_val.pt"
    test_path = f"{base_dir}/emb_test.pt"
    return train_path, val_path, test_path


def _save_embedding_cache(
    path: str,
    z: torch.Tensor,
    theta: torch.Tensor,
    emb_scaler: BaseScaler,
    cosmo_scaler: Optional[BaseScaler],
):
    """Save raw embeddings, cosmology vectors and scaler parameters.

    We only cache unscaled z/theta plus the scaler statistics so that
    scaling can still be applied consistently at load time.
    """
    state = {
        "z": z.cpu(),
        "theta": theta.cpu(),
        # Embedding PerDimStandardScaler stats
        "emb_mean": getattr(emb_scaler, "mean", None),
        "emb_std": getattr(emb_scaler, "std", None),
        # Cosmology scaler preset min/max and parameter order (if used)
        "cosmo_min": getattr(cosmo_scaler, "min", None) if cosmo_scaler is not None else None,
        "cosmo_max": getattr(cosmo_scaler, "max", None) if cosmo_scaler is not None else None,
        "cosmo_param_names": getattr(cosmo_scaler, "parameter_names", None) if cosmo_scaler is not None else None,
    }
    torch.save(state, path)


def _load_embedding_cache(path: str) -> Tuple[torch.Tensor, torch.Tensor, Optional[BaseScaler], Optional[BaseScaler]]:
    """Load raw embeddings/cosmo and reconstruct scaler objects if stats exist."""
    ckpt = torch.load(path, map_location="cpu")
    z = ckpt["z"]
    theta = ckpt["theta"]

    emb_scaler: Optional[BaseScaler] = None
    if ckpt.get("emb_mean") is not None and ckpt.get("emb_std") is not None:
        emb_scaler = PerDimStandardScaler()
        emb_scaler.mean = ckpt["emb_mean"]
        emb_scaler.std = ckpt["emb_std"]

    cosmo_scaler: Optional[BaseScaler] = None
    if ckpt.get("cosmo_min") is not None and ckpt.get("cosmo_max") is not None:
        # Rebuild preset scaler using stored min/max and parameter names
        param_names = ckpt.get("cosmo_param_names")
        if param_names is not None:
            from ..data.scaling import MinMaxScaler
            cosmo_scaler = MinMaxScaler(param_names)
            cosmo_scaler.min = ckpt["cosmo_min"]
            cosmo_scaler.max = ckpt["cosmo_max"]

    return z, theta, emb_scaler, cosmo_scaler


def build_embedding_dataloaders(
    train_loader,
    val_loader,
    test_loader,
    models: List[torch.nn.Module],
    base_cfg=None,
    wandb_run_name: Optional[str] = None,
):
    """Construct *scaled* embedding dataloaders from existing loaders and models.

    1) Embeddings are scaled per variable using PerDimStandardScaler (shared implementation).
    2) Cosmological parameters are scaled using preset MinMax bounds
       from COSMO_PARAM_PRESET_MINMAX via the shared _build_cosmo_preset_scaler.
    3) Raw embeddings + scaler parameters are cached under the checkpoint path,
       and reused if present.
    """

    # Build cosmology preset scaler from config if available
    cosmo_scaler: Optional[BaseScaler] = None
    cosmo_param_names = []
    if base_cfg is not None and hasattr(base_cfg, "cosmo_param_names"):
        cosmo_param_names = list(base_cfg.cosmo_param_names)
        cosmo_scaler = _build_cosmo_preset_scaler(COSMO_PARAM_PRESET_MINMAX, cosmo_param_names)
    cosmo_scaler = None
    # Try to load cached embeddings
    train_z = val_z = test_z = None
    train_theta = val_theta = test_theta = None
    emb_scaler: Optional[BaseScaler] = None
    cache_used = False

    if base_cfg is not None and wandb_run_name is not None:
        train_path, val_path, test_path = _get_embedding_cache_paths(base_cfg, wandb_run_name)
        if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
            train_z, train_theta, emb_scaler, cached_cosmo_scaler = _load_embedding_cache(train_path)
            val_z, val_theta, _, _ = _load_embedding_cache(val_path)
            test_z, test_theta, _, _ = _load_embedding_cache(test_path)
            # Prefer scaler from cache if present; otherwise use freshly built preset
            if cached_cosmo_scaler is not None:
                cosmo_scaler = cached_cosmo_scaler
            cache_used = True

    if not cache_used:
        # Compute embeddings from scratch
        train_z, train_theta = compute_embeddings(models, train_loader)
        val_z, val_theta = compute_embeddings(models, val_loader)
        test_z, test_theta = compute_embeddings(models, test_loader)

        # Global per-variable standard scaler on training embeddings
        emb_scaler = PerDimStandardScaler()
        emb_scaler.fit(train_z)

        # Save caches if possible
        if base_cfg is not None and wandb_run_name is not None:
            print("Saving embedding caches to disk for future reuse...")
            print("Saving to paths:")
            print(" Train:", train_path)
            train_path, val_path, test_path = _get_embedding_cache_paths(base_cfg, wandb_run_name)
            os.makedirs(os.path.dirname(train_path), exist_ok=True)
            _save_embedding_cache(train_path, train_z, train_theta, emb_scaler, cosmo_scaler)
            _save_embedding_cache(val_path, val_z, val_theta, emb_scaler, cosmo_scaler)
            _save_embedding_cache(test_path, test_z, test_theta, emb_scaler, cosmo_scaler)

    # Build datasets using tensors and scalers
    train_ds = EmbeddingDataset(train_z, train_theta, emb_scaler=emb_scaler, cosmo_scaler=cosmo_scaler)
    val_ds = EmbeddingDataset(val_z, val_theta, emb_scaler=train_ds.emb_scaler, cosmo_scaler=cosmo_scaler)
    test_ds = EmbeddingDataset(test_z, test_theta, emb_scaler=train_ds.emb_scaler, cosmo_scaler=cosmo_scaler)

    batch_size = getattr(train_loader, "batch_size", 128)
    num_workers = getattr(train_loader, "num_workers", 0)

    train_emb_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_emb_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_emb_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=num_workers)

    return train_emb_loader, val_emb_loader, test_emb_loader


def fit_nde_on_embeddings(emb_dim: int, train_loader, val_loader, test_loader, base_cfg):
    """Build and train an NDE Lightning module on precomputed embeddings.

    Uses an IdentityEmbedding as the encoder and otherwise reuses the
    flow configuration from `base_cfg`.

    A config flag ``inference_mode`` controls whether we run in
    NPE (posterior) or NLE (likelihood) mode. Allowed values:
      * 'npe' (default): NDELightningModule, modelling p(theta | z).
      * 'nle'         : LikelihoodNDELightningModule, modelling p(z | theta).
    """
    conditioning_dim = emb_dim
    inference_dim = len(base_cfg.cosmo_param_names)

    embedding_net = IdentityEmbedding(emb_dim).to("cuda" if torch.cuda.is_available() else "cpu")

    # Decide which LightningModule to use based on config.
    inference_mode = getattr(base_cfg, "inference_mode", "npe").lower()
    if inference_mode not in {"npe", "nle"}:
        raise ValueError(f"Unknown inference_mode '{inference_mode}', expected 'npe' or 'nle'.")

    LightningCls = NDELightningModule if inference_mode == "npe" else LikelihoodNDELightningModule
    if inference_mode == "nle":
        # switch conditioning and inference dims for likelihood mode, since the "data" is now the embedding and the "parameters" are the cosmology
        conditioning_dim, inference_dim = inference_dim, conditioning_dim
    model = LightningCls(
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

    # Include inference_mode in run name for clarity
    wandb_logger = wandb.init(
        project=getattr(base_cfg, "project", "emb-nde"),
        group=f"embeddings_nde_{inference_mode}",
        name=(
            f"{base_cfg.experiment_name}/"
            f"{base_cfg.model_type}_{match_string_logger}_"
            f"ncosmo{num_trainval_cosmos}_{inference_mode}"
        ),
        reinit=True,
    )

    wandb_logger_name = wandb_logger.name if wandb_logger else "no_logger"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=f"{monitor_string}",
        dirpath=f"{base_cfg.base_path}/checkpoints/{base_cfg.experiment_name}/{wandb_logger_name}",
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
        gradient_clip_val=0.5, # NEED PRECISION HIGH
        precision="32",
    )

    trainer.fit(model, train_loader, val_loader)
