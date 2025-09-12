import numpy as np
from typing import Dict, List, Sequence, Tuple, Union, Optional

import torch

from .models.compressors import _MODEL_BUILDERS
from .models.lightning_modules import NDELightningModule
from .models.kids_inference_architectures import KIDS_MODEL_BUILDERS

# Centralized dataloader builder
from .data.data import build_dataloaders, build_nested_keys_from_quantities
from .data.data_loading import unpack_data, load_cosmo_params
from .data.scaling import DataScaler

# Merge model registries (compressors + kids-specific architectures)
MODEL_BUILDERS = {**_MODEL_BUILDERS, **KIDS_MODEL_BUILDERS}


class DataDictScalerTransform:
    """
    Applies per-key standard scaling to entries in the data dict.
    Uses means/stds fitted on the training split.
    """
    def __init__(self, key_scalers: Dict[str, DataScaler]):
        self.key_scalers = key_scalers or {}

    def __call__(self, data: Dict[str, Union[np.ndarray, torch.Tensor]]):
        out = {}
        for k, v in data.items():
            scaler = self.key_scalers.get(k)
            if scaler is None or scaler.std is None:
                out[k] = v
                continue
            mean = float(scaler.mean)
            std = float(scaler.std) if scaler.std != 0 else 1.0
            out[k] = (v - mean) / std
        return out


class TransformingDataset(torch.utils.data.Dataset):
    """
    Wraps a base dataset and applies transforms to (data_dict, cosmo_vector).
    """
    def __init__(
        self,
        base_ds: torch.utils.data.Dataset,
        data_transform: Optional[DataDictScalerTransform] = None,
        cosmo_scaler: Optional[DataScaler] = None,
    ):
        self.base_ds = base_ds
        self.data_transform = data_transform
        self.cosmo_scaler = cosmo_scaler

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        data, cosmo = self.base_ds[idx]
        if self.data_transform is not None:
            data = self.data_transform(data)
        if self.cosmo_scaler is not None and self.cosmo_scaler.min is not None and self.cosmo_scaler.max is not None:
            min_v = torch.as_tensor(self.cosmo_scaler.min, dtype=cosmo.dtype, device=cosmo.device)
            max_v = torch.as_tensor(self.cosmo_scaler.max, dtype=cosmo.dtype, device=cosmo.device)
            denom = torch.clamp(max_v - min_v, min=1e-12)
            cosmo = (cosmo - min_v) / denom
        return data, cosmo


def _fit_data_key_scalers_from_paths(
    train_paths: Sequence[str],
    nested_keys: Dict[str, Tuple[str, ...]],
    keys_to_scale: Optional[Sequence[str]] = None,
) -> Dict[str, DataScaler]:
    key_scalers: Dict[str, DataScaler] = {}
    if keys_to_scale is None:
        keys_to_scale = list(nested_keys.keys())

    for key in keys_to_scale:
        if key not in nested_keys:
            continue
        vals: List[np.ndarray] = []
        single_key = {key: nested_keys[key]}
        for p in train_paths:
            data, _ = unpack_data(p, single_key, [], as_torch=False, dtype=np.float32, stack_groups=False)
            arr = data[key]
            vals.append(arr.reshape(-1))
        if not vals:
            continue
        stacked = np.concatenate(vals, axis=0)
        scaler = DataScaler()
        scaler.fit_standard(stacked)
        if scaler.std == 0:
            scaler.std = 1.0
        key_scalers[key] = scaler
    return key_scalers


def _fit_cosmo_minmax_scaler_from_paths(train_paths: Sequence[str], cosmo_params: Sequence[str]) -> Optional[DataScaler]:
    if not cosmo_params:
        return None
    rows: List[np.ndarray] = []
    for p in train_paths:
        vec = load_cosmo_params(p, list(cosmo_params), as_torch=False, dtype=np.float32)
        rows.append(np.asarray(vec))
    if not rows:
        return None
    X = np.stack(rows, axis=0)
    scaler = DataScaler()
    scaler.fit_minmax(X)
    span = scaler.max - scaler.min
    span[span == 0] = 1.0
    scaler.max = scaler.min + span
    return scaler


def _wrap_loader_with_transforms(loader: torch.utils.data.DataLoader, data_transform, cosmo_scaler, shuffle=True):
    base_ds = loader.dataset
    wrapped_ds = TransformingDataset(base_ds, data_transform=data_transform, cosmo_scaler=cosmo_scaler)

    new_loader = torch.utils.data.DataLoader(
        wrapped_ds,
        batch_size=loader.batch_size,
        shuffle=shuffle,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        persistent_workers=getattr(loader, 'persistent_workers', False),
        drop_last=getattr(loader, 'drop_last', False),
        collate_fn=getattr(loader, 'collate_fn', None),
    )
    return new_loader


def prepare_data_parameters(config):
    """
    Build train/val/test DataLoaders using data.build_dataloaders, then apply
    optional scaling transforms configured by config.scaler_options.
    """
    # Resolve nested_keys from either dataset_quantities helper or explicit mapping
    if getattr(config, 'dataset_quantities', None):
        nested_keys = build_nested_keys_from_quantities(list(config.dataset_quantities))
    else:
        nested_keys = dict(getattr(config, 'dataset_nested_keys', {}))

    # Build base dataloaders via the central entrypoint
    train_loader, val_loader, test_loader = build_dataloaders(
        config.data_patterns,
        nested_keys,
        list(getattr(config, 'cosmo_param_names', [])),
        batch_size=getattr(config, 'batch_size', 4),
        val_batch_size=getattr(config, 'val_batch_size', None),
        test_batch_size=getattr(config, 'test_batch_size', None),
        shuffle_train=getattr(config, 'shuffle_train', True),
        num_workers=getattr(config, 'num_workers', 0),
        pin_memory=getattr(config, 'pin_memory', False),
        persistent_workers=None,
        train_frac=getattr(config, 'train_frac', 0.8),
        val_frac=getattr(config, 'val_frac', 0.1),
        test_frac=getattr(config, 'test_frac', 0.1),
        seed=getattr(config, 'split_seed', 42),
        as_torch=True,
        dtype=np.float32,
        stack_groups=getattr(config, 'stack_groups', False),
    )

    # Fit scalers from the training split
    scaler_options = getattr(config, 'scaler_options', None) or {}
    train_ds = train_loader.dataset
    train_paths = list(getattr(train_ds, 'paths', []))
    # Use explicit nested_keys resolved above (safer than introspection)
    cosmo_params = list(getattr(config, 'cosmo_param_names', []))

    data_keys_to_scale = None
    if 'data' in scaler_options and isinstance(scaler_options['data'], dict):
        data_keys_to_scale = scaler_options['data'].get('keys')

    key_scalers = _fit_data_key_scalers_from_paths(train_paths, nested_keys, keys_to_scale=data_keys_to_scale)

    cosmo_scaler = None
    if 'cosmo' in scaler_options:
        cosmo_scaler = _fit_cosmo_minmax_scaler_from_paths(train_paths, cosmo_params)

    # Build transforms and wrap loaders
    data_transform = DataDictScalerTransform(key_scalers)
    train_loader = _wrap_loader_with_transforms(train_loader, data_transform, cosmo_scaler)
    val_loader = _wrap_loader_with_transforms(val_loader, data_transform, cosmo_scaler, shuffle=False)
    test_loader = _wrap_loader_with_transforms(test_loader, data_transform, cosmo_scaler, shuffle=False)

    scalers = {
        'data': key_scalers,
        'cosmo': cosmo_scaler,
    }
    return scalers, train_loader, val_loader, test_loader


def prepare_data_and_model(config, data_parameters=None):
    # Build data (and scalers) if not provided
    if data_parameters is None:
        scalers, train_loader, val_loader, test_loader = prepare_data_parameters(config)
    else:
        scalers, train_loader, val_loader, test_loader = data_parameters

    # Select the correct backbone dynamically (merged registry)
    embedding_model = MODEL_BUILDERS[config.model_type](config.latent_dim).to(device='cuda')

    # Derive a reasonable warmup if not explicitly provided
    base_sched_kwargs = dict(getattr(config, 'scheduler_kwargs', {}) or {})
    if 'warmup' not in base_sched_kwargs:
        try:
            est_warmup = min(len(train_loader), 1000)
        except Exception:
            est_warmup = 250
        base_sched_kwargs['warmup'] = est_warmup
    batch = next(iter(train_loader))
    model = NDELightningModule(
        embedding_model,
        conditioning_dim=config.latent_dim,
        inference_dim = len(config.cosmo_param_names),
        lr=config.lr,
        scheduler_type=config.scheduler_type,
        element_names=["Omega", "sigma8"],
        test_dataloader=None,
        optimizer_kwargs=config.optimizer_kwargs,
        num_extra_blocks=config.extra_blocks,
        checkpoint_path=config.checkpoint_path,
        freeze_CNN=config.freeze_cnn,
        scheduler_kwargs=base_sched_kwargs,
    )

    return train_loader, val_loader, model, scalers
