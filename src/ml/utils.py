import numpy as np
from typing import Dict, List, Sequence, Tuple, Union, Optional

import torch

from .models.compressors import _MODEL_BUILDERS
from .models.lightning_modules import NDELightningModule, KLDRegularisedNDELightningModule, EnsembleNDELightningModule
from .models.kids_inference_architectures import KIDS_MODEL_BUILDERS
from .eval.loading_model import find_best_checkpoint, get_best_checkpoint

# Centralized dataloader builder
from .data.data import build_dataloaders, build_nested_keys_from_quantities
from .data.data_loading import unpack_data, load_cosmo_params
# Use new abstracted scalers
from .data.scaling import BaseScaler, MinMaxScaler, StandardScaler, LogNormalScaler

# Merge model registries (compressors + kids-specific architectures)
MODEL_BUILDERS = {**_MODEL_BUILDERS, **KIDS_MODEL_BUILDERS}

N_BINS = 6
def _infer_channels_per_map_from_quantities(dataset_quantities: Optional[Sequence[str]]) -> Optional[int]:
    """Infer channels_per_map from dataset_quantities.

    - If only E maps are present (E_north/E_south), use 3.
    - If B maps are also present, use 6.
    - If no map quantities are present, return None.
    """
    if not dataset_quantities:
        return None
    print("Dataset quantities provided:", dataset_quantities)
    qs = set(dataset_quantities)
    has_e = ("E_north" in qs) or ("E_south" in qs)
    has_b = ("B_north" in qs) or ("B_south" in qs)

    if not has_e and not has_b:
        return None
    if has_b:
        return 2*N_BINS
    return N_BINS


class DataDictScalerTransform:
    """
    Applies per-key scaling to entries in the data dict using provided scaler objects
    that implement transform() / inverse_transform().
    """
    def __init__(self, key_scalers: Dict[str, BaseScaler]):
        self.key_scalers = key_scalers or {}

    def __call__(self, data: Dict[str, Union[np.ndarray, torch.Tensor]]):
        out = {}
        for k, v in data.items():
            scaler = self.key_scalers.get(k)
            if scaler is None:
                out[k] = v
            else:
                out[k] = scaler.transform(v)
        return out


class TransformingDataset(torch.utils.data.Dataset):
    """
    Wraps a base dataset and applies transforms to (data_dict, cosmo_vector).
    """
    def __init__(
        self,
        base_ds: torch.utils.data.Dataset,
        data_transform: Optional[DataDictScalerTransform] = None,
        cosmo_scaler: Optional[BaseScaler] = None,
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
        if self.cosmo_scaler is not None:
            cosmo = self.cosmo_scaler.transform(cosmo)
        return data, cosmo


def _fit_data_key_scalers_from_paths(
    train_paths: Sequence[str],
    nested_keys: Dict[str, Tuple[str, ...]],
    keys_to_scale: Optional[Sequence[str]] = None,
    max_obs : int = 1_000,
) -> Dict[str, BaseScaler]:
    key_scalers: Dict[str, BaseScaler] = {}
    if keys_to_scale is None:
        keys_to_scale = list(nested_keys.keys())

    for key in keys_to_scale:
        if key not in nested_keys:
            continue
        vals: List[np.ndarray] = []
        single_key = {key: nested_keys[key]}
        # shuffle
        np.random.shuffle(train_paths)
        for p in train_paths[:max_obs]:
            data, _ = unpack_data(p, single_key, [], as_torch=False, dtype=np.float32, stack_groups=False)
            arr = data[key]
            vals.append(arr.reshape(-1))
        if not vals:
            continue
        stacked = np.concatenate(vals, axis=0)

        # Choose scaler: ensure 'bandpowers' uses LogNormalScaler
        if "bandpowers" in key.lower():
            scaler: BaseScaler = LogNormalScaler()
        else:
            scaler = StandardScaler()
        scaler.fit(stacked)
        key_scalers[key] = scaler
    return key_scalers


def _fit_cosmo_minmax_scaler_from_paths(train_paths: Sequence[str], cosmo_params: Sequence[str]) -> Optional[BaseScaler]:
    if not cosmo_params:
        return None
    rows: List[np.ndarray] = []
    for p in train_paths:
        vec = load_cosmo_params(p, list(cosmo_params), as_torch=False, dtype=np.float32)[0]
        rows.append(np.asarray(vec))
    if not rows:
        return None
    X = np.stack(rows, axis=0)
    scaler = MinMaxScaler(cosmo_params)
    scaler.fit(X)
    return scaler

def _build_cosmo_preset_scaler(preset_minmax: Dict[str, Tuple[float, float]], cosmo_params: Sequence[str]) -> Optional[BaseScaler]: 
    if not cosmo_params:
        return None
    mins = []
    maxs = []
    for p in cosmo_params:
        if p not in preset_minmax:
            raise ValueError(f"Cosmological parameter '{p}' not found in preset min/max dictionary.")
        min_v, max_v = preset_minmax[p]
        mins.append(min_v)
        maxs.append(max_v)
    scaler = MinMaxScaler(cosmo_params)
    scaler.min = np.array(mins, dtype=np.float32)
    scaler.max = np.array(maxs, dtype=np.float32)
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

    # Optional limit on the number of cosmologies used for train+val
    max_trainval_cosmos = getattr(config, 'max_trainval_cosmos', None)

    # Optional shape-noise repeat indices for test set filtering
    test_shape_noise_idx = getattr(config, 'test_shape_noise_idx', None)

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
        augment_eb_patches=getattr(config, 'augment_eb_patches', True),
        max_trainval_cosmos=max_trainval_cosmos,
        test_shape_noise_idx=test_shape_noise_idx,
    )

    # Print dataset lengths for visibility
    print(f"Train dataset length: {len(train_loader.dataset)}")
    print(f"Val dataset length:   {len(val_loader.dataset)}")
    print(f"Test dataset length:  {len(test_loader.dataset)}")
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
        scaling_type = scaler_options['cosmo']['type']
        if scaling_type == 'minmax':
            cosmo_scaler = _fit_cosmo_minmax_scaler_from_paths(train_paths, cosmo_params)
        elif scaling_type == "preset":
            from .data.constants import COSMO_PARAM_PRESET_MINMAX
            cosmo_scaler = _build_cosmo_preset_scaler(COSMO_PARAM_PRESET_MINMAX, cosmo_params)
        else:
            raise ValueError(f"Unsupported cosmo scaler type '{scaling_type}' specified in config.scaler_options['cosmo']['type']")

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

    # Build model
    model = build_model(config, test_dataloader=test_loader)

    return (train_loader, val_loader, test_loader), model, scalers

def build_model(config, test_dataloader=None):

    redundancy_dim = getattr(config, 'redundancy_dim', 0)
    use_KL_loss = getattr(config, 'use_KL_loss', False)

    # latent_dim is always the dimension of mu; encoder may output 2*latent_dim when use_kl
    latent_dim = getattr(config, 'latent_dim', None)
    if latent_dim is None:
        raise ValueError("config.latent_dim must be set")

    # Build encoder / embedding model with KL behaviour controlled by use_KL_loss
    model_kwargs = {**config.model_kwargs.to_dict(), 'redundancy_dim': redundancy_dim}

    # If channels_per_map not explicitly set, infer it from dataset_quantities
    if 'channels_per_map' not in model_kwargs:
        ch = _infer_channels_per_map_from_quantities(getattr(config, 'dataset_quantities', None))
        if ch is not None:
            model_kwargs['input_channels'] = ch
            print("Inferred channels_per_map from dataset_quantities: setting input_channels to", ch)
    print(model_kwargs)
    # Kids encoders honour use_kl; legacy compressors just ignore it via **model_kwargs
    if config.model_type in KIDS_MODEL_BUILDERS:
        model_kwargs = {**model_kwargs, 'use_kl': use_KL_loss}

    # num_outputs passed to builders is the latent_dim (mu dimension)
    embedding_model = MODEL_BUILDERS[config.model_type](latent_dim, **model_kwargs).to(device='cuda')

    # Effective conditioning dimension seen by the flow is latent_dim (+ optional redundancy)
    conditioning_dim = latent_dim + redundancy_dim

    # Derive a reasonable warmup if not explicitly provided
    base_sched_kwargs = dict(getattr(config, 'scheduler_kwargs', {}) or {})
    if 'warmup' not in base_sched_kwargs:
        est_warmup = 1000
        base_sched_kwargs['warmup'] = est_warmup

    num_flow_heads = getattr(config, 'num_flow_heads', 1)

    # Choose correct LightningModule: ensemble vs single-flow, with optional KL
    if num_flow_heads > 1:
        LightningModule = EnsembleNDELightningModule
        lm_extra_kwargs = {"num_flows": num_flow_heads}
    else:
        LightningModule = KLDRegularisedNDELightningModule if use_KL_loss else NDELightningModule
        lm_extra_kwargs = {}

    # Construct LightningModule without any pretrained-loading kwargs
    model = LightningModule(
        embedding_model,
        conditioning_dim=conditioning_dim,
        inference_dim=len(config.cosmo_param_names),
        lr=config.lr,
        flow_type=config.flow_type,
        scheduler_type=config.scheduler_type,
        element_names=["Omega", "sigma8"],
        test_dataloader=test_dataloader,
        optimizer_kwargs=config.optimizer_kwargs,
        num_extra_blocks=config.extra_blocks,
        freeze_CNN=config.freeze_cnn,
        scheduler_kwargs=base_sched_kwargs,
        flow_kwargs=config.flow_kwargs,
        **lm_extra_kwargs,
    )

    # --------------------------------------------------------
    # Explicitly load optional pretrained components
    # --------------------------------------------------------
    band_module_name = None
    backbone_module_name = None

    checkpoint_path = getattr(config, 'checkpoint_path', None)
    if checkpoint_path:
        model.load_from_checkpoint(checkpoint_path)
        print("Loaded full model state from checkpoint:", checkpoint_path)
    else:
        pretrained_band_ckpt_folder = getattr(config, 'pretrained_band_ckpt_path', None)
        if pretrained_band_ckpt_folder is not None:
            pretrained_band_ckpts, _ = get_best_checkpoint(pretrained_band_ckpt_folder, config.pretrained_band_match_string)  # sanity check that folder and checkpoint exist
            freeze_band = getattr(config, 'freeze_band', False)
            band_prefix = getattr(config, 'band_prefix', 'band_encoder.')
            pretrained_band_ckpt = pretrained_band_ckpts[0]  # TODO: fix this
            band_module_name = model._load_pretrained_band_encoder(
                pretrained_band_ckpt,
                freeze=freeze_band,
                band_prefix=band_prefix,
            )

            if getattr(config, 'load_pretrained_flow', False):
                flow_prefix = getattr(config, 'flow_prefix', 'model.flow.')
                model._load_pretrained_flow(pretrained_band_ckpt, freeze=False, flow_prefix=flow_prefix)

        pretrained_backbone_ckpt = getattr(config, 'pretrained_backbone_ckpt_path', None)
        if pretrained_backbone_ckpt is not None:
            freeze_backbone = getattr(config, 'freeze_backbone', False)
            backbone_prefix = getattr(config, 'backbone_prefix', 'shared_cnn.backbone.')
            target_prefix = getattr(config, 'target_backbone_prefix', '')
            backbone_module_name = model._load_pretrained_cnn_backbone(
                pretrained_backbone_ckpt,
                freeze=freeze_backbone,
                backbone_prefix=backbone_prefix,
                target_prefix=target_prefix,
            )

    # After we know which submodules were actually used for pretrained
    # loading, configure optional per-module learning rates, if present
    # in the config. These are scalars; we build the dicts that
    # BaseLightningModule._build_param_groups expects.
    band_lr = getattr(config, 'pretrained_band_lr', None)
    if band_lr is not None and band_module_name is not None:
        model.pretrained_band_lrs = {band_module_name: float(band_lr)}

    backbone_lr = getattr(config, 'pretrained_backbone_lr', None)
    if backbone_lr is not None and backbone_module_name is not None:
        model.pretrained_backbone_lrs = {backbone_module_name: float(backbone_lr)}

    return model

def load_best_checkpoint_model(config, run_folder, test_loader=None):
    """Find the best checkpoint in a run folder and load its model.
    
    Args:
        config: Configuration object containing experiment settings
        run_folder: Path to the run folder containing checkpoints
        data_parameters: Optional data parameters for model preparation
    
    Returns:
        tuple: (model, best_checkpoint_path, best_val_loss) or (None, None, None) if no checkpoint found
    """
    best_checkpoint, best_val_loss = find_best_checkpoint(run_folder)
    if not best_checkpoint:
        return None, None, None

    config.checkpoint_path = best_checkpoint
    model= build_model(config, test_dataloader=test_loader)
    model.to("cuda")
    model.eval()
    return model, best_checkpoint, best_val_loss