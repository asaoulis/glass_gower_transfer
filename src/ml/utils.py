"""Data, model and checkpoint plumbing shared by training and evaluation.

prepare_data_parameters builds the dataloaders and fitted scalers from a config. build_model builds
the encoder, the flow head and the LightningModule and loads the pretrained submodules.
build_ensemble_model_from_checkpoints rebuilds a trained ensemble for evaluation.
"""
import numpy as np
from typing import Dict, List, Sequence, Tuple, Union, Optional

import torch
import os

from .models.deprecated.compressors import _MODEL_BUILDERS
from .models.lightning_modules import NDELightningModule, KLDRegularisedNDELightningModule, EnsembleNDELightningModule, EnsembleLikelihoodNDELightningModule, RegressionLightningModule, LikelihoodNDELightningModule, JointVMIMNLELightningModule
from .models.kids_inference_architectures import KIDS_MODEL_BUILDERS
from .eval.loading_model import find_best_checkpoint, get_best_checkpoint

from .data.data_loaders import build_dataloaders, build_nested_keys_from_quantities
from .data.side_inputs import build_side_input_fn, side_input_config
from .data.data_loading import unpack_data, load_cosmo_params
from .data.scaling import BaseScaler, MinMaxScaler, StandardScaler, LogNormalScaler, PerDimStandardScaler

MODEL_BUILDERS = {**_MODEL_BUILDERS, **KIDS_MODEL_BUILDERS}

N_BINS = 6
# E/B map quantity names, as in data_augmentations._EB_QUANTITY_RE.
import re as _re
_E_MAP_RE = _re.compile(r"^E(?:_.+)?_(north|south)$")
_B_MAP_RE = _re.compile(r"^B(?:_.+)?_(north|south)$")


def _infer_channels_per_map_from_quantities(
    dataset_quantities: Optional[Sequence[str]],
    n_bins: int = N_BINS,
) -> Optional[int]:
    """Infer channels_per_map from dataset_quantities: n_bins for E only, 2 * n_bins for E+B.

    Returns None without map quantities. n_bins comes from config.n_tomo_bins.
    """
    if not dataset_quantities:
        return None
    print("Dataset quantities provided:", dataset_quantities)
    has_e = any(_E_MAP_RE.match(q) for q in dataset_quantities)
    has_b = any(_B_MAP_RE.match(q) for q in dataset_quantities)

    if not has_e and not has_b:
        return None
    if has_b:
        return 2*n_bins
    return n_bins


class DataDictScalerTransform:
    """Apply per-key scalers (transform / inverse_transform) to entries of the data dict."""
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
    """Wrap a base dataset and apply transforms to (data_dict, cosmo_vector)."""
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
        item = self.base_ds[idx]
        # Items are (data, cosmo) or (data, cosmo, cosmo_id). The cosmo_id passes through unscaled.
        if len(item) == 3:
            data, cosmo, cid = item
        else:
            data, cosmo = item
            cid = None
        if self.data_transform is not None:
            data = self.data_transform(data)
        if self.cosmo_scaler is not None:
            cosmo = self.cosmo_scaler.transform(cosmo)
        if cid is not None:
            return data, cosmo, cid
        return data, cosmo


def _resolve_scaler_fit_max_obs(config):
    """Scaler-fit file cap: an int, or a per-key dict built from the optional by-key override.

    scaler_fit_max_obs is the cap for every key, and scaler_fit_max_obs_by_key overrides it for
    named keys.
    """
    default = int(getattr(config, 'scaler_fit_max_obs', 1000) or 1000)
    by_key = getattr(config, 'scaler_fit_max_obs_by_key', None)
    if not by_key:
        return default
    out = {k: int(v) for k, v in dict(by_key).items()}
    out["default"] = default
    return out


def _fit_data_key_scalers_from_paths(
    train_paths: Sequence[str],
    nested_keys: Dict[str, Tuple[str, ...]],
    keys_to_scale: Optional[Sequence[str]] = None,
    max_obs=1_000,
    seed: int = 0,
    per_dim_keys: Optional[Sequence[str]] = None,
) -> Dict[str, BaseScaler]:
    """Fit one scaler per data key on a subsample of `train_paths`.

    Each file is flattened and a scalar scaler is fitted (LogNormalScaler for band powers,
    StandardScaler otherwise). Keys in `per_dim_keys` are stacked as (N, D) rows and get a
    PerDimStandardScaler."""
    key_scalers: Dict[str, BaseScaler] = {}
    per_dim_keys = set(per_dim_keys or ())
    if keys_to_scale is None:
        keys_to_scale = list(nested_keys.keys())

    # max_obs: an int, or {key: cap, 'default': cap}. A frozen encoder must see the scaler it was
    # trained with, so the bandpower cap is kept apart from the map cap.
    def _cap_for(key: str) -> int:
        if isinstance(max_obs, dict):
            v = max_obs.get(key, max_obs.get("default", 1_000))
        else:
            v = max_obs
        return int(v or 1_000)

    rng = np.random.default_rng(seed)

    for key in keys_to_scale:
        if key not in nested_keys:
            continue
        vals: List[np.ndarray] = []
        single_key = {key: nested_keys[key]}
        order = rng.permutation(len(train_paths))[:_cap_for(key)]
        for p in [train_paths[i] for i in order]:
            # Skip corrupt files or ones missing the group, as H5CosmoDataset does.
            try:
                data, _ = unpack_data(p, single_key, [], as_torch=False, dtype=np.float32, stack_groups=False)
                arr = data[key]
            except (OSError, KeyError) as e:
                import warnings
                warnings.warn(f"[scaler-fit] skipping unreadable file ({type(e).__name__}): {p}")
                continue
            vals.append(arr.reshape(1, -1) if key in per_dim_keys else arr.reshape(-1))
        if not vals:
            continue
        stacked = np.concatenate(vals, axis=0)

        if key in per_dim_keys:
            scaler = PerDimStandardScaler()
        elif "bandpowers" in key.lower():
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

    common = dict(
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        persistent_workers=getattr(loader, 'persistent_workers', False),
        collate_fn=getattr(loader, 'collate_fn', None),
    )

    return torch.utils.data.DataLoader(
        wrapped_ds,
        batch_size=loader.batch_size,
        shuffle=shuffle,
        drop_last=getattr(loader, 'drop_last', False),
        **common,
    )


def prepare_data_parameters(config):
    """Build the train/val/test DataLoaders and apply the scalers of config.scaler_options.

    Returns (scalers, train_loader, val_loader, test_loader).
    """
    if getattr(config, 'dataset_quantities', None):
        nested_keys = build_nested_keys_from_quantities(
            list(config.dataset_quantities),
            eb_variant=getattr(config, 'eb_map_variant', None),
        )
    else:
        nested_keys = dict(getattr(config, 'dataset_nested_keys', {}))

    # Optional limit on the number of cosmologies used for train+val
    max_trainval_cosmos = getattr(config, 'max_trainval_cosmos', None)

    # Optional shape-noise repeat indices for test set filtering
    test_shape_noise_idx = getattr(config, 'test_shape_noise_idx', None)

    # Build base dataloaders via the central entrypoint
    cosmo_param_names = list(getattr(config, 'cosmo_param_names', []))
    selection_cosmo_params = getattr(config, 'train_val_selection_cosmo_params', None)
    if selection_cosmo_params is None:
        selection_cosmo_params = cosmo_param_names if len(cosmo_param_names) > 0 else None
    train_loader, val_loader, test_loader = build_dataloaders(
        config.data_patterns,
        nested_keys,
        cosmo_param_names,
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
        augment_eb_ra_roll=getattr(config, 'augment_eb_ra_roll', False),
        max_trainval_cosmos=max_trainval_cosmos,
        selection_strategy=getattr(config, 'train_val_selection_strategy', 'random'),
        selection_cosmo_params=selection_cosmo_params,
        stratified_bins=getattr(config, 'train_val_stratified_bins', 'auto'),
        test_shape_noise_idx=test_shape_noise_idx,
        ensemble_seed=getattr(config, 'ensemble_seed', None),
        N_extra_test_cosmologies=getattr(config, 'N_extra_test_cosmologies', None),
        fixed_test_sim_ids=getattr(config, 'fixed_test_sim_ids', None),
        N_test_cosmologies=getattr(config, 'N_test_cosmologies', None),
        # Oracle conditioning (config.side_input type 'true_theta'), else None.
        side_input_fn=build_side_input_fn(config),
        preload=bool(getattr(config, 'preload_data', False)),
        exclude_files=getattr(config, 'exclude_files', None),
        include_files_regex=getattr(config, 'include_files_regex', None),
    )

    print(f"Train dataset length: {len(train_loader.dataset)}")
    print(f"Val dataset length:   {len(val_loader.dataset)}")
    print(f"Test dataset length:  {len(test_loader.dataset)}")
    scaler_options = getattr(config, 'scaler_options', None) or {}
    train_ds = train_loader.dataset
    val_ds = val_loader.dataset
    train_paths = list(getattr(train_ds, 'paths', []))
    val_paths = list(getattr(val_ds, 'paths', []))
    cosmo_params = cosmo_param_names

    # Single-model runs fit scalers on train only, ensemble runs on train+val.
    is_ensemble_run = int(getattr(config, 'ensemble_repeats', 1) or 1) > 1
    scaler_fit_paths = train_paths if not is_ensemble_run else (train_paths + val_paths)

    data_keys_to_scale = None
    if 'data' in scaler_options and isinstance(scaler_options['data'], dict):
        data_keys_to_scale = scaler_options['data'].get('keys')
    key_scalers = _fit_data_key_scalers_from_paths(
        scaler_fit_paths,
        nested_keys,
        keys_to_scale=data_keys_to_scale,
        # Whole files are concatenated per key, so the cap bounds memory.
        max_obs=_resolve_scaler_fit_max_obs(config),
        seed=int(getattr(config, 'scaler_fit_seed', 0) or 0),
        # scaler_options may be a ConfigDict, not a dict subclass.
        per_dim_keys=(getattr(scaler_options, 'get', lambda *_: None)('data') or {}).get('per_dim_keys'),
    )

    cosmo_scaler = None
    if 'cosmo' in scaler_options:
        scaling_type = scaler_options['cosmo']['type']
        if scaling_type == 'minmax':
            cosmo_scaler = _fit_cosmo_minmax_scaler_from_paths(scaler_fit_paths, cosmo_params)
        elif scaling_type == "preset":
            from .data.constants import COSMO_PARAM_PRESET_MINMAX
            # Optional per-config box overrides of the global preset.
            preset = dict(COSMO_PARAM_PRESET_MINMAX)
            overrides = scaler_options['cosmo'].get('preset_overrides') or {}
            preset.update({k: tuple(v) for k, v in overrides.items()})
            cosmo_scaler = _build_cosmo_preset_scaler(preset, cosmo_params)
        else:
            raise ValueError(f"Unsupported cosmo scaler type '{scaling_type}' specified in config.scaler_options['cosmo']['type']")

    data_transform = DataDictScalerTransform(key_scalers)
    # Optional per-sample E/B noise normalisation, applied before the data scalers.
    eb_noise_norm = getattr(config, 'eb_noise_norm', None)
    if eb_noise_norm:
        from .data.data_augmentations import EBNoiseNormTransform, ChainedDataTransform
        data_transform = ChainedDataTransform(
            [EBNoiseNormTransform(eb_noise_norm), data_transform])

    train_loader = _wrap_loader_with_transforms(train_loader, data_transform, cosmo_scaler)
    val_loader = _wrap_loader_with_transforms(val_loader, data_transform, cosmo_scaler, shuffle=False)
    test_loader = _wrap_loader_with_transforms(test_loader, data_transform, cosmo_scaler, shuffle=False)

    scalers = {
        'data': key_scalers,
        'cosmo': cosmo_scaler,
    }
    return scalers, train_loader, val_loader, test_loader


def prepare_data_and_model(config, data_parameters=None):
    if data_parameters is None:
        scalers, train_loader, val_loader, test_loader = prepare_data_parameters(config)
    else:
        scalers, train_loader, val_loader, test_loader = data_parameters

    model = build_model(config, test_dataloader=test_loader)

    return (train_loader, val_loader, test_loader), model, scalers

def _attach_side_input(embedding_model, side_cfg, config, model_kwargs):
    """Attach config.side_input to the encoder; returns the extra conditioning dim."""
    from .models.side_input import DataKeySideInput, FrozenEncoderSideInput

    if not hasattr(embedding_model, "attach_side_input"):
        raise ValueError(f"side_input is only supported for kids_hybrid_bandpowers_maps, not {config.model_type}")
    if side_cfg["type"] == "true_theta":
        module = DataKeySideInput(len(side_cfg["params"]))
    else:
        # The trained encoder's architecture without KL, built from config.latent_dim.
        side_latent = int(side_cfg.get("latent_dim", config.latent_dim))
        side_kwargs = {**model_kwargs, 'use_kl': False} if config.model_type in KIDS_MODEL_BUILDERS else model_kwargs
        side_encoder = MODEL_BUILDERS[config.model_type](side_latent, **side_kwargs)
        side_dim = int(side_kwargs.get('hybrid_output_dim') or side_latent)
        module = FrozenEncoderSideInput(side_encoder, side_dim)
    module = module.to(next(embedding_model.parameters()).device)
    embedding_model.attach_side_input(module)
    print(f"[build_model] side_input {side_cfg['type']} -> conditioning dim +{module.dim}")
    return module.dim


def build_model(config, test_dataloader=None):

    redundancy_dim = getattr(config, 'redundancy_dim', 0)
    use_KL_loss = getattr(config, 'use_KL_loss', False)
    is_distributed = getattr(config, 'is_distributed', False)

    training_mode = getattr(config, 'training_mode', 'nde')

    # latent_dim is the dimension of mu. The encoder emits 2 * latent_dim with use_kl.
    latent_dim = getattr(config, 'latent_dim', None)
    if latent_dim is None:
        raise ValueError("config.latent_dim must be set")

    model_kwargs = {**config.model_kwargs.to_dict(), 'redundancy_dim': redundancy_dim}

    if 'channels_per_map' not in model_kwargs:
        ch = _infer_channels_per_map_from_quantities(
            getattr(config, 'dataset_quantities', None),
            n_bins=int(getattr(config, 'n_tomo_bins', N_BINS) or N_BINS),
        )
        if ch is not None:
            model_kwargs['input_channels'] = ch
            print("Inferred channels_per_map from dataset_quantities: setting input_channels to", ch)
    print(model_kwargs)
    if config.model_type in KIDS_MODEL_BUILDERS:
        model_kwargs = {**model_kwargs, 'use_kl': use_KL_loss}

    embedding_model = MODEL_BUILDERS[config.model_type](latent_dim, **model_kwargs).to("cuda" if torch.cuda.is_available() else "cpu")

    hybrid_output_dim = model_kwargs.get('hybrid_output_dim', None)
    if hybrid_output_dim is not None:
        print(f"Using hybrid_output_dim={hybrid_output_dim} instead of latent_dim={latent_dim} for encoder output dimension.")
        latent_dim = hybrid_output_dim  # override latent_dim for encoder output when hybrid_output_dim is set

    side_cfg = side_input_config(config)
    side_dim = 0
    if side_cfg is not None:
        side_dim = _attach_side_input(embedding_model, side_cfg, config, model_kwargs)

    conditioning_dim = latent_dim + redundancy_dim + side_dim
    inference_dim = len(config.cosmo_param_names)
    inference_mode = getattr(config, 'inference_mode', 'npe')

    is_joint = str(inference_mode).lower() in {"joint", "npe+nle", "nle+npe"}

    if inference_mode == 'nle':
        conditioning_dim, inference_dim = inference_dim, conditioning_dim

    base_sched_kwargs = dict(getattr(config, 'scheduler_kwargs', {}) or {})
    if 'warmup' not in base_sched_kwargs:
        est_warmup = 1000
        base_sched_kwargs['warmup'] = est_warmup

    num_flow_heads = getattr(config, 'num_flow_heads', 1)

    lm_extra_kwargs = {}
    # Map-only auxiliary VMIM head, NPE-family modules only.
    patch_aux_weight = float(getattr(config, 'patch_aux_weight', 0.0) or 0.0)
    if patch_aux_weight > 0.0:
        lm_extra_kwargs['patch_aux_weight'] = patch_aux_weight
        lm_extra_kwargs['patch_aux_flow_kwargs'] = dict(
            getattr(config, 'patch_aux_flow_kwargs', {}) or {})
    if training_mode == 'regression':
        LightningModule = RegressionLightningModule
    elif is_joint:
        LightningModule = JointVMIMNLELightningModule
        lm_extra_kwargs = {
            "nle_weighting": float(getattr(config, "nle_weighting", 1.0)),
        }
    elif num_flow_heads > 1:
        LightningModule = EnsembleNDELightningModule
        lm_extra_kwargs = {"num_flows": num_flow_heads}
    elif use_KL_loss:
        LightningModule = KLDRegularisedNDELightningModule
    elif inference_mode == 'nle':
        LightningModule = LikelihoodNDELightningModule  # NLE is just a special case of NDE with swapped conditioning/inference
    else:
        LightningModule = NDELightningModule

    model = LightningModule(
        embedding_model,
        conditioning_dim=conditioning_dim,
        inference_dim=inference_dim,
        lr=config.lr,
        flow_type=config.flow_type,
        scheduler_type=config.scheduler_type,
        element_names=config.cosmo_param_names,
        test_dataloader=test_dataloader,
        optimizer_kwargs=config.optimizer_kwargs,
        num_extra_blocks=config.extra_blocks,
        freeze_CNN=config.freeze_cnn,
        scheduler_kwargs=base_sched_kwargs,
        flow_kwargs=config.flow_kwargs,
        is_distributed=is_distributed,
        **lm_extra_kwargs,
    )

    # --- optional pretrained components ---
    band_module_name = None
    backbone_module_name = None

    checkpoint_path = getattr(config, 'checkpoint_path', None)
    pretrained_band_ckpt_folder = getattr(config, 'pretrained_band_ckpt_path', None)

    # Load order: the embedding_net load runs last and overwrites a separately loaded band unless
    # band_load_after_embedding is set.
    band_load_after_embedding = getattr(config, 'band_load_after_embedding', False)

    def _do_band_load():
        # Returns the band module name, or None.
        pretrained_band_ckpts, _ = get_best_checkpoint(pretrained_band_ckpt_folder, config.pretrained_band_match_string)  # sanity check that folder and checkpoint exist
        freeze_band = getattr(config, 'freeze_band', False)
        pretrained_band_ckpt = pretrained_band_ckpts[0]  # TODO: fix this
        return model._load_pretrained_band_encoder(
            pretrained_band_ckpt,
            freeze=freeze_band,
            band_prefix="model.embedding_net.",
        )

    if checkpoint_path:
        model.load_from_checkpoint(checkpoint_path)
        print("Loaded full model state from checkpoint:", checkpoint_path)
        # A whole-model load supplies the band too, so freeze_band is honoured here (the band-load
        # branch is skipped).
        if getattr(config, 'freeze_band', False):
            band_module_name = model._freeze_band_encoder()
    else:
        if pretrained_band_ckpt_folder is not None and not band_load_after_embedding:
            band_module_name = _do_band_load()

    # checkpoint_path restores a trained model, so the warm-start loads below (flow, backbone,
    # embedding) must not run after it.
    if checkpoint_path and getattr(config, 'load_pretrained_flow', False):
        # Print which warm starts were skipped.
        print("[build_model] checkpoint_path set -> SKIPPING the pretrained-flow warm start so "
              "the loaded checkpoint's weights survive.")
    if (not checkpoint_path) and getattr(config, 'load_pretrained_flow', False):
        # Joint module has two flows (NPE and NLE heads) that may be pretrained separately.
        if is_joint:
            npe_folder = getattr(config, 'pretrained_npe_flow_ckpt_path', None) or getattr(config, 'pretrained_flow_ckpt_path', None) or pretrained_band_ckpt_folder
            nle_folder = getattr(config, 'pretrained_nle_flow_ckpt_path', None) or getattr(config, 'pretrained_flow_ckpt_path', None) or pretrained_band_ckpt_folder

            if npe_folder is not None:
                npe_ckpts, _ = get_best_checkpoint(npe_folder, config.pretrained_band_match_string)
                if npe_ckpts:
                    npe_ckpt = npe_ckpts[0]
                    npe_prefix = getattr(config, 'npe_flow_prefix', getattr(config, 'flow_prefix', 'model.flow.'))
                    freeze_npe = getattr(config, 'freeze_npe_flow', getattr(config, 'freeze_flow', False))
                    if hasattr(model, '_load_pretrained_npe_head'):
                        model._load_pretrained_npe_head(npe_ckpt, freeze=freeze_npe, flow_prefix=npe_prefix)
                    else:
                        raise AttributeError("Joint model missing _load_pretrained_npe_head")

            if nle_folder is not None:
                nle_ckpts, _ = get_best_checkpoint(nle_folder, config.pretrained_band_match_string)
                if nle_ckpts:
                    nle_ckpt = nle_ckpts[0]
                    nle_prefix = getattr(config, 'nle_flow_prefix', getattr(config, 'flow_prefix', 'model.flow.'))
                    freeze_nle = getattr(config, 'freeze_nle_flow', getattr(config, 'freeze_flow', False))
                    if hasattr(model, '_load_pretrained_nle_head'):
                        model._load_pretrained_nle_head(nle_ckpt, freeze=freeze_nle, flow_prefix=nle_prefix)
                    else:
                        raise AttributeError("Joint model missing _load_pretrained_nle_head")
        else:
            pretrained_flow_ckpt_folder = getattr(config, 'pretrained_flow_ckpt_path', pretrained_band_ckpt_folder)
            pretrained_flow_ckpts, _ = get_best_checkpoint(pretrained_flow_ckpt_folder, config.pretrained_band_match_string)  # sanity check that folder and checkpoint exist
            pretrained_flow_ckpt = pretrained_flow_ckpts[0]  # TODO: fix this

            flow_prefix = getattr(config, 'flow_prefix', 'model.flow.')
            model._load_pretrained_flow(pretrained_flow_ckpt, freeze=False, flow_prefix=flow_prefix)

    pretrained_backbone_ckpt = getattr(config, 'pretrained_backbone_ckpt_path', None)
    if checkpoint_path and pretrained_backbone_ckpt is not None:
        # No freeze re-application: only evaluation sets checkpoint_path, where freezing is moot.
        print("[build_model] checkpoint_path set -> SKIPPING the pretrained-backbone warm start so "
              "the loaded checkpoint's weights survive.")
        pretrained_backbone_ckpt = None
    if pretrained_backbone_ckpt is not None:
        freeze_backbone = getattr(config, 'freeze_backbone', False)
        backbone_prefix = getattr(config, 'backbone_prefix', 'shared_cnn.backbone.')
        target_prefix = getattr(config, 'target_backbone_prefix', '')
        if os.path.isdir(pretrained_backbone_ckpt):
            # A directory resolves to its best checkpoint for this repeat, as the band loader does.
            backbone_ckpts, _ = get_best_checkpoint(
                pretrained_backbone_ckpt, config.pretrained_band_match_string)
            pretrained_backbone_ckpt = backbone_ckpts[0]
        pretrained_band_ckpts, _ = get_best_checkpoint(pretrained_band_ckpt_folder, config.pretrained_band_match_string)  # sanity check that folder and checkpoint exist
        backbone_module_name = model._load_pretrained_cnn_backbone(
            pretrained_backbone_ckpt,
            freeze=freeze_backbone,
            backbone_prefix=backbone_prefix,
            target_prefix=target_prefix,
        )
    pretrained_embedding_ckpt_folder = getattr(config, 'pretrained_embedding_ckpt_path', None)
    if checkpoint_path and pretrained_embedding_ckpt_folder is not None:
        print("[build_model] checkpoint_path set -> SKIPPING the pretrained-embedding warm start so "
              "the loaded checkpoint's weights survive.")
        pretrained_embedding_ckpt_folder = None
    if pretrained_embedding_ckpt_folder is not None:
        pretrained_embedding_ckpts, _ = get_best_checkpoint(pretrained_embedding_ckpt_folder, config.pretrained_band_match_string)  # sanity check that folder and checkpoint exist
        pretrained_embedding_ckpt = pretrained_embedding_ckpts[0]  # TODO: fix this
        freeze_embedding_net = getattr(config, 'freeze_embedding_net', False)
        patch_module_name = model._load_pretrained_embedding_net(
            pretrained_embedding_ckpt,
            freeze=freeze_embedding_net,
        )

    # Frozen side encoder: loaded after the embedding_net warm start, skipped under checkpoint_path.
    if side_cfg is not None and side_cfg["type"] == "frozen_encoder":
        if checkpoint_path:
            print("[build_model] checkpoint_path set -> SKIPPING the side-encoder load so the "
                  "loaded checkpoint's weights survive.")
        elif not side_cfg.get("ckpt_path"):
            # Only the local smoke gate sets allow_random_init (it never loads cluster checkpoints).
            if not side_cfg.get("allow_random_init"):
                raise ValueError("side_input frozen_encoder needs ckpt_path (the trained encoder folder)")
            print("[build_model] side-input encoder left RANDOM (allow_random_init; smoke only)")
        else:
            side_ckpts, _ = get_best_checkpoint(side_cfg["ckpt_path"], config.pretrained_band_match_string)
            side_ckpt = side_ckpts[0]
            print(f"[build_model] side-input encoder <- {side_ckpt}")
            src_state = torch.load(side_ckpt, map_location="cpu")
            src_state = src_state.get("state_dict", src_state)
            from .models.lightning.utils import load_partial_weights
            side_mod = model.model.embedding_net.side_input_module
            diag = load_partial_weights(side_mod.encoder, src_state, prefix="model.embedding_net.",
                                        freeze=True, verbose=True)
            # The whole side encoder must come from the checkpoint, never a partial load.
            if not diag["loaded"] or diag["missing"] or diag["skipped_shape"] or diag.get("skipped_missing"):
                raise RuntimeError(f"side-input encoder load incomplete from {side_ckpt}: loaded "
                                   f"{diag['loaded']}, missing {len(diag['missing'])}, shape "
                                   f"{len(diag['skipped_shape'])}, unmatched {len(diag.get('skipped_missing') or [])}")
            side_mod.freeze()

    # Deferred band load, after the embedding_net load, so the requested band survives.
    if (not checkpoint_path) and pretrained_band_ckpt_folder is not None and band_load_after_embedding:
        print("[build_model] band_load_after_embedding=True -> loading the band encoder AFTER "
              "the embedding_net load so it is not overwritten.")
        band_module_name = _do_band_load()

    band_lr = getattr(config, 'pretrained_band_lr', None)
    if band_lr is not None and band_module_name is not None:
        model.pretrained_band_lrs = {band_module_name: float(band_lr)}

    backbone_lr = getattr(config, 'pretrained_backbone_lr', None)
    if backbone_lr is not None and backbone_module_name is not None:
        model.pretrained_backbone_lrs = {backbone_module_name: float(backbone_lr)}

    # Optionally freeze the hybrid's patch branch after every load, with its dropout zeroed.
    if getattr(config, 'freeze_patch_encoder', False):
        patch_enc = getattr(model.embedding_net, 'patch_encoder', None)
        if patch_enc is None:
            raise ValueError("freeze_patch_encoder=True but embedding_net has no patch_encoder")
        n = 0
        for p in patch_enc.parameters():
            p.requires_grad = False
            n += 1
        for m in patch_enc.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
        print(f"[build_model] freeze_patch_encoder: froze embedding_net.patch_encoder ({n} param tensors, dropout off)")

    _apply_ml_perf(model, config)

    return model


def _apply_ml_perf(model, config):
    """Apply the config.ml_perf training-speed options (tf32, amp, compile), all off when unset."""
    ml_perf = getattr(config, 'ml_perf', None)
    if not ml_perf:
        return
    ml_perf = dict(ml_perf.to_dict() if hasattr(ml_perf, 'to_dict') else ml_perf)

    if ml_perf.get('tf32'):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Encoder-scoped bf16 autocast. The flow stays fp32: whole-forward bf16 crashes the spline.
    if ml_perf.get('amp'):
        inner = getattr(model, 'model', None)
        if inner is not None and hasattr(inner, 'amp_encoder'):
            inner.amp_encoder = True
            print("[ml_perf] AMP encoder autocast (bf16) ON; flow kept fp32")
        else:
            print("[ml_perf] WARNING amp requested but model.model has no amp_encoder hook")

    # Compile the CNN backbone only: compiling the PoolProj head breaks inductor.
    compile_mode = ml_perf.get('compile', 'none')
    if compile_mode and compile_mode != 'none':
        inner = getattr(model, 'model', model)
        target = None
        for name, mod in inner.named_modules():
            if name.endswith('shared_cnn.backbone'):
                target = (name, mod)
                break
        if target is None:  # fallback: any '*.backbone'
            for name, mod in inner.named_modules():
                if name.endswith('.backbone'):
                    target = (name, mod)
                    break
        if target is not None:
            name, mod = target
            parent = inner
            parts = name.split('.')
            for a in parts[:-1]:
                parent = getattr(parent, a)
            # 'default' mode: reduce-overhead needs static input addresses.
            mode = 'default' if compile_mode in (True, 'backbone', 'default') else compile_mode
            setattr(parent, parts[-1], torch.compile(mod, mode=mode))
            print(f"[ml_perf] torch.compile(mode={mode}) on model.model.{name}")
        else:
            print("[ml_perf] WARNING compile requested but no '*.backbone' module found")

def load_best_checkpoint_model(config, run_folder, test_loader=None):
    """Find the best checkpoint in a run folder and load its model.

    Returns (model, best_checkpoint_path, best_val_loss), or (None, None, None) without a checkpoint.
    """
    best_checkpoint, best_val_loss = find_best_checkpoint(run_folder)
    if not best_checkpoint:
        return None, None, None

    config.checkpoint_path = best_checkpoint
    model= build_model(config, test_dataloader=test_loader)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, best_checkpoint, best_val_loss

def is_ensemble_eval_active(cfg) -> bool:
    """True when config requests an evaluation-time flow ensemble."""
    return int(getattr(cfg, "ensemble_repeats", 1) or 1) > 1


def build_ensemble_model_from_checkpoints(
    cfg,
    test_loader,
    *,
    match_string: str,
    member_test_loaders: Optional[Sequence[torch.utils.data.DataLoader]] = None,
    model_builder=None,
):
    """Rebuild an evaluation-time ensemble from the member checkpoints under the run folder.

    Each member is rebuilt with its own scalers and test loader, using the training seed logic,
    unless member_test_loaders supplies them. test_loader is ignored. Raises when a member is
    missing unless allow_partial_ensemble is set.
    """
    from copy import copy
    if model_builder is None:
        model_builder = build_model

    n_ens = int(getattr(cfg, "ensemble_repeats", 1) or 1)
    if n_ens <= 1:
        return None

    experiment_path = f"{cfg.base_path}/checkpoints/{cfg.experiment_name}/"
    if not os.path.exists(experiment_path):
        print("Experiment path does not exist:", experiment_path, flush=True)
        return None

    # Determine repeat idx from match_string suffix
    try:
        repeat_idx = int(str(match_string).split("_")[-1])
    except Exception:
        # Fallback: try regex via helper
        from .models.utils import parse_repeat_and_ensemble_from_match
        repeat_idx, _ = parse_repeat_and_ensemble_from_match(match_string)

    base_seed = 42 #int(getattr(cfg, "split_seed", 42))

    members = []
    member_checkpoints = []
    resolved_member_test_loaders = []
    missing_members = []

    for j in range(n_ens):
        member_match = f"{match_string}_ens{j}"
        best_ckpts, _ = get_best_checkpoint(experiment_path, member_match)
        best_ckpt = best_ckpts[0] if best_ckpts else None
        if best_ckpt is None:
            print(
                f"[build_ensemble_model_from_checkpoints] No checkpoint found for member '{member_match}'",
                flush=True,
            )
            missing_members.append(member_match)
            continue  # Skip this member but keep going for others
        # Log the exact checkpoint file resolved for each member: the best-checkpoint choice
        # changes if files are added or rotated.
        print(f"[build_ensemble_model_from_checkpoints] member {j}: {best_ckpt}", flush=True)
        member_checkpoints.append(str(best_ckpt))

        cfg_j = copy(cfg)
        cfg_j.checkpoint_path = best_ckpt

        # Reset to base, then apply repeat for split_seed and member for ensemble_seed.
        cfg_j.split_seed = base_seed
        set_seed_for_repeat_and_ensemble(cfg_j, repeat_idx=repeat_idx, ensemble_idx=j)

        test_loader_j = None
        if member_test_loaders is not None and j < len(member_test_loaders):
            test_loader_j = member_test_loaders[j]

        if test_loader_j is None:
            _, _, _, test_loader_j = prepare_data_parameters(cfg_j)

        resolved_member_test_loaders.append(test_loader_j)

        member = model_builder(cfg_j, test_dataloader=test_loader_j)
        member.to("cuda" if torch.cuda.is_available() else "cpu")
        member.eval()
        members.append(member)
    if len(members) == 0:
        print("No ensemble members were successfully loaded. Returning None.", flush=True)
        return None
    # A short ensemble must not be scored as the full one: raise unless allow_partial_ensemble is set.
    if missing_members and not bool(getattr(cfg, "allow_partial_ensemble", False)):
        raise RuntimeError(
            f"[build_ensemble_model_from_checkpoints] resolved only {len(members)}/{n_ens} ensemble "
            f"members for '{cfg.experiment_name}' match '{match_string}'; missing "
            f"{missing_members}. A short ensemble is not comparable to the ens{n_ens} numbers it "
            "would be filed against. Set `allow_partial_ensemble: True` on the config to override."
        )
    inference_mode = str(getattr(cfg, "inference_mode", "npe")).lower()
    EnsembleCls = EnsembleLikelihoodNDELightningModule if inference_mode == "nle" else EnsembleNDELightningModule
    model = EnsembleCls(members)
    model.member_checkpoints = member_checkpoints   # provenance: exactly which files were scored
    # use the first member loader for theta0s extraction; others are used internally
    model.test_dataloader = resolved_member_test_loaders[0]
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model

def set_seed_for_repeat_and_ensemble(config, repeat_idx: int, ensemble_idx: int = 0):
    """Set repeat-bound split_seed and optional ensemble_seed on config.

    split_seed (base + repeat_idx) sets which cosmologies are selected and the train/val/test
    split. ensemble_seed only reshuffles the selected trainval cosmologies between train and val.
    Returns (split_seed, ensemble_seed).
    """
    base_seed = int(getattr(config, "split_seed", 42))
    config.split_seed = base_seed + int(repeat_idx)

    n_ens = int(getattr(config, "ensemble_repeats", 1) or 1)
    config.ensemble_seed = int(ensemble_idx) if n_ens > 1 else None

    return config.split_seed, config.ensemble_seed