import os
import sys
from typing import Dict, List, Optional, Tuple

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

    Embeddings are optionally scaled per-feature using PerDimStandardScaler.
    Cosmological parameters are optionally scaled with a preset MinMax scaler.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        cosmo: torch.Tensor,
        emb_scaler: Optional[BaseScaler] = None,
        cosmo_scaler: Optional[BaseScaler] = None,
        *,
        scale_embeddings: bool = True,
    ):
        assert embeddings.shape[0] == cosmo.shape[0]
        self.embeddings = embeddings
        self.cosmo = cosmo
        self.scale_embeddings = bool(scale_embeddings)

        # Fit or store embedding scaler (per-variable standard scaling)
        if self.scale_embeddings:
            if emb_scaler is None:
                emb_scaler = PerDimStandardScaler()
                emb_scaler.fit(self.embeddings)
        else:
            emb_scaler = None
        self.emb_scaler = emb_scaler

        # Store cosmology scaler (preset MinMax)
        self.cosmo_scaler = cosmo_scaler

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        z = self.embeddings[idx]
        theta = self.cosmo[idx]

        z_scaled = self.emb_scaler.transform(z) if self.emb_scaler is not None else z
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


def _as_ncosmo_list(v) -> List[Optional[int]]:
    """Normalize max_trainval_cosmos-like values to a list.

    Accepts None, scalar, list/tuple.
    """
    if v is None:
        return [None]
    if isinstance(v, (list, tuple)):
        return [int(x) if x is not None else None for x in v]
    return [int(v)]


def get_max_trainval_cosmos_grid(target_exp_name: str, source_exp_names: List[str]) -> List[Optional[int]]:
    """Return the list of max_trainval_cosmos values to iterate over.

    Mode is controlled by target experiment's `split_on_source_experiments` flag.

    - If split_on_source_experiments=False (default): iterate over the target experiment's
      max_trainval_cosmos (or [None] if unset).
    - If split_on_source_experiments=True: iterate over the *intersection* of the source
      experiments' max_trainval_cosmos values (or [None] if none of the sources specify it).

    This keeps behavior consistent and avoids silently training embeddings on sources with
    different numbers of cosmologies.
    """
    if target_exp_name not in experiments:
        raise ValueError(f"Experiment '{target_exp_name}' not found in config.experiments.experiments.")

    target_cfg = experiments[target_exp_name]
    split_on_source = bool(target_cfg.get("split_on_source_experiments", False))

    if not split_on_source:
        return _as_ncosmo_list(target_cfg.get("max_trainval_cosmos", None))

    # Source-based mode
    sets = []
    for s in source_exp_names:
        if s not in experiments:
            raise ValueError(f"Source experiment '{s}' not found in config.experiments.experiments.")
        v = experiments[s].get("max_trainval_cosmos", None)
        if v is None:
            continue
        sets.append(set(x for x in _as_ncosmo_list(v) if x is not None))

    if not sets:
        return [None]

    common = set.intersection(*sets)
    if not common:
        raise ValueError(
            "split_on_source_experiments=True but no common max_trainval_cosmos values were found across source experiments. "
            "Set compatible 'max_trainval_cosmos' lists in config.experiments for all sources."
        )
    return sorted(common)


def load_pretrained_models(
    exp_names: List[str],
    cfg_overrides: Optional[Dict[str, object]] = None,
    repeat_idx: Optional[int] = None,
    match_string: Optional[str] = None,
    match_num_cosmo: bool = False,
) -> Tuple[List[torch.nn.Module], List[str], List[str]]:
    """Build models for a list of experiment names using their best checkpoints.

    Uses `load_best_model_and_build_posterior`, which encapsulates the
    checkpoint directory layout and filename logic.

    Args:
        exp_names: list of experiment names.
        cfg_overrides: optional mapping {experiment_name: cfg}. When provided,
            the cfg is used instead of building one from config.experiments.
            This is used to support split_on_source_experiments, where we want
            to set max_trainval_cosmos on the *source* experiments.

    Returns:
        models: list of trained NDE models
        dataset_quantities: merged dataset quantities across experiments
        checkpoint_paths: list of checkpoint paths
    """
    models = []
    dataset_quantities = set()
    checkpoint_paths = []

    # Prefer explicit repeat-bound match string (e.g. 'ncosmo30_0').
    # Fallback to legacy suffix matching by repeat index.
    # if match_string is not None and str(match_string):
    #     ds_string_match = str(match_string)
    # else:
    ds_string_match = f"_{repeat_idx}" if repeat_idx is not None else ""

    for name in exp_names:
        if cfg_overrides is not None and name in cfg_overrides:
            cfg = cfg_overrides[name]
        else:
            cfg = _build_config_for_experiment(name)

        if match_string is not None and str(match_string) and match_num_cosmo:
            ds_string_match = str(match_string)
        else:
            ds_string_match = f"None_{repeat_idx}" if repeat_idx is not None else ""
        print("Searching for checkpoint with match string:", ds_string_match)
        result = load_best_model_and_build_posterior(cfg, ds_string_match=ds_string_match)
        if result is None:
            raise RuntimeError(f"Failed to load best model for experiment '{name}'.")
        model, _, checkpoint_path = result
        model.eval()
        # set use_KL to false to avoid issues with missing encoder components when loading the full NDELightningModule
        if hasattr(model, "use_KL"):
            model.use_KL = False
        models.append(model)
        dataset_quantities.update(getattr(cfg, "dataset_quantities", []))
        checkpoint_paths.append(checkpoint_path)

    dataset_quantities = list(dataset_quantities)
    return models, dataset_quantities, checkpoint_paths


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
    print(f"Computed embeddings for {z_cat.shape[0]} samples with dimension {z_cat.shape[1]}.")
    return z_cat, theta_cat


def format_run_match(match_string: str, ensemble_idx: int = 0, ensemble_repeats: int = 1) -> str:
    """Return the run-name match tag used for logging/checkpoints.

    Baseline behaviour: returns match_string.
    Ensemble behaviour: appends '_ens{ensemble_idx}' when ensemble_repeats>1.

    Note: this must NOT change match_string semantics (repeat-bound).
    """
    ensemble_repeats = int(ensemble_repeats or 1)
    if ensemble_repeats > 1:
        return f"{match_string}_ens{int(ensemble_idx)}"
    return match_string


def _get_embedding_cache_paths(base_cfg, wandb_run_name: str) -> Tuple[str, str, str]:
    """Return file paths for cached embedding datasets under checkpoint dir.

    wandb_run_name should already include any ensemble suffix if applicable.
    """
    base_dir = f"{base_cfg.base_path}/checkpoints/{wandb_run_name}/datasets"
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
    use_cache_if_exists=False,
):
    """Construct *scaled* (or optionally unscaled) embedding dataloaders from existing loaders and models.

    Embeddings scaling is controlled by config flag `scale_embeddings` (default True).
    """

    scale_embeddings = bool(getattr(base_cfg, "scale_embeddings", True))

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
        if use_cache_if_exists:
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

        # Global per-variable standard scaler on training embeddings (optional)
        if scale_embeddings:
            emb_scaler = PerDimStandardScaler()
            emb_scaler.fit(train_z)
        else:
            emb_scaler = None

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
    train_ds = EmbeddingDataset(
        train_z,
        train_theta,
        emb_scaler=emb_scaler,
        cosmo_scaler=cosmo_scaler,
        scale_embeddings=scale_embeddings,
    )
    val_ds = EmbeddingDataset(
        val_z,
        val_theta,
        emb_scaler=train_ds.emb_scaler,
        cosmo_scaler=cosmo_scaler,
        scale_embeddings=scale_embeddings,
    )
    test_ds = EmbeddingDataset(
        test_z,
        test_theta,
        emb_scaler=train_ds.emb_scaler,
        cosmo_scaler=cosmo_scaler,
        scale_embeddings=scale_embeddings,
    )

    batch_size = getattr(train_loader, "batch_size", 128)
    num_workers = getattr(train_loader, "num_workers", 0)

    train_emb_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_emb_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_emb_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=num_workers)

    return train_emb_loader, val_emb_loader, test_emb_loader


def build_nde_on_embeddings(
    *,
    emb_dim: int,
    base_cfg,
    test_loader=None,
    device: Optional[torch.device] = None,
):
    """Construct an (untrained) NDE Lightning module that operates on embeddings.

    This is the shared model-construction logic used by both training code and
    notebooks that want to load a checkpoint without re-implementing the setup.

    Args:
        emb_dim: Dimension of the (concatenated) embedding vector.
        base_cfg: Experiment config (expects the usual NDE/flow fields).
        test_loader: Optional dataloader assigned to the Lightning module.
        device: Optional torch.device. Defaults to cuda if available.

    Returns:
        model: An instance of NDELightningModule or LikelihoodNDELightningModule.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conditioning_dim = emb_dim
    inference_dim = len(base_cfg.cosmo_param_names)

    embedding_net = IdentityEmbedding(emb_dim).to(device)

    inference_mode = getattr(base_cfg, "inference_mode", "npe").lower()
    if inference_mode not in {"npe", "nle"}:
        raise ValueError(f"Unknown inference_mode '{inference_mode}', expected 'npe' or 'nle'.")

    LightningCls = NDELightningModule if inference_mode == "npe" else LikelihoodNDELightningModule
    if inference_mode == "nle":
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

    model.to(device)
    model.eval()
    return model


def fit_nde_on_embeddings(emb_dim: int, train_loader, val_loader, test_loader, base_cfg, run_name):
    """Build and train an NDE Lightning module on precomputed embeddings.

    Uses an IdentityEmbedding as the encoder and otherwise reuses the
    flow configuration from `base_cfg`.

    A config flag ``inference_mode`` controls whether we run in
    NPE (posterior) or NLE (likelihood) mode. Allowed values:
      * 'npe' (default): NDELightningModule, modelling p(theta | z).
      * 'nle'         : LikelihoodNDELightningModule, modelling p(z | theta).
    """

    model = build_nde_on_embeddings(emb_dim=emb_dim, base_cfg=base_cfg, test_loader=test_loader)

    if getattr(base_cfg, 'load_pretrained_flow', False):
        pretrained_band_ckpt = getattr(base_cfg, 'pretrained_band_ckpt_path', None)
        if pretrained_band_ckpt is None:
            raise ValueError("Config flag 'load_pretrained_flow' is True but 'pretrained_band_ckpt_path' is not set.")
        model._load_pretrained_flow(pretrained_band_ckpt, freeze=False)

    # Standard Lightning trainer setup (similar to fit_model)
    num_gpus = torch.cuda.device_count()
    accelerator = "gpu" if num_gpus > 0 else "cpu"
    devices = num_gpus if num_gpus > 0 else 1
    strategy = "ddp" if num_gpus > 1 else "auto"

    monitor_string = f"val_{model.loss_name}"
    num_trainval_cosmos = getattr(base_cfg, 'max_trainval_cosmos', None)
    match_string_logger = base_cfg.match_string if base_cfg.match_string else ""

    # Include inference_mode in run name for clarity
    inference_mode = getattr(base_cfg, "inference_mode", "npe").lower()
    wandb_logger = wandb.init(
        project=getattr(base_cfg, "project", "emb-nde"),
        group=f"embeddings_nde_{inference_mode}",
        name=(run_name),
        reinit=True,
    )

    wandb_logger_name = wandb_logger.name if wandb_logger else "no_logger"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=f"{monitor_string}",
        dirpath=f"{base_cfg.base_path}/checkpoints/{wandb_logger_name}",
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
