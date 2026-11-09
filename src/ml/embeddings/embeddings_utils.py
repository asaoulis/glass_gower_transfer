"""Embeddings pipeline helpers.

load_pretrained_models resolves source experiments to frozen encoders, build_embedding_dataloaders
caches and serves their embeddings (with provenance and the per-source whitener), and
build_nde_on_embeddings and fit_nde_on_embeddings build and train the flow on them.
"""
import hashlib
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import pytorch_lightning as pl
import wandb

from config.default import get_default_config
from config.experiments import experiments

from ..models.lightning_modules import (
    NDELightningModule,
    LikelihoodNDELightningModule,
    EnsembleLikelihoodNDELightningModule,
)
from ..eval.utils import load_best_model_and_build_posterior
from ..eval.loading_model import get_best_checkpoint
from ..tmpdir import redirect_tempdir
from ..data.scaling import BaseScaler, PerDimStandardScaler, WhitenPCAScaler
from ..utils import _build_cosmo_preset_scaler
from ..data.constants import COSMO_PARAM_PRESET_MINMAX

# Filename for the persisted per-source whitener, colocated with a run's emb_{train,val,test}.pt.
WHITENER_FILENAME = "whitener.pt"


class IdentityEmbedding(torch.nn.Module):
    """Identity embedding, so NDELightningModule can take precomputed embeddings as its input."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x

    def compress(self, x):
        """The `.compress()` interface: returns (mu, logvar) = (x, zeros)."""
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

        if self.scale_embeddings:
            if emb_scaler is None:
                emb_scaler = PerDimStandardScaler()
                emb_scaler.fit(self.embeddings)
        else:
            emb_scaler = None
        self.emb_scaler = emb_scaler

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
    """Create a config object for a given experiment name, as train.py does, skipping max_trainval_cosmos."""
    if name not in experiments:
        raise ValueError(f"Experiment '{name}' not found in config.experiments.experiments.")

    cfg = get_default_config()
    cfg.experiment_name = name
    exp_cfg = experiments[name]

    for k, v in exp_cfg.items():
        if k == "max_trainval_cosmos":
            continue
        setattr(cfg, k, v)

    # Default inference mode for embeddings is 'npe' (posterior).
    if not hasattr(cfg, "inference_mode"):
        cfg.inference_mode = "npe"

    return cfg


def _as_ncosmo_list(v) -> List[Optional[int]]:
    """Normalise a max_trainval_cosmos value (None, scalar, list or tuple) to a list."""
    if v is None:
        return [None]
    if isinstance(v, (list, tuple)):
        return [int(x) if x is not None else None for x in v]
    return [int(v)]


def get_max_trainval_cosmos_grid(target_exp_name: str, source_exp_names: List[str]) -> List[Optional[int]]:
    """Return the list of max_trainval_cosmos values to iterate over.

    The target experiment's max_trainval_cosmos by default, or with split_on_source_experiments
    the intersection of the sources' values. [None] when unset.
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
    per_source_match_strings: Optional[Sequence[str]] = None,
) -> Tuple[List[torch.nn.Module], List[str], List[str]]:
    """Build models for a list of experiment names using their best checkpoints.

    cfg_overrides maps experiment name to a cfg used instead of config.experiments.
    per_source_match_strings gives one run match per source, which stacks several repeats of one
    experiment. Returns (models, ordered merged dataset_quantities, checkpoint_paths).
    """
    if per_source_match_strings is not None and len(per_source_match_strings) != len(exp_names):
        raise ValueError(
            f"per_source_match_strings has {len(per_source_match_strings)} entries but there are "
            f"{len(exp_names)} source experiments; they must correspond one-to-one."
        )
    models = []
    # Ordered, not a set: the scaler subsample is drawn in key order.
    dataset_quantities = {}
    checkpoint_paths = []

    ds_string_match = f"_{repeat_idx}" if repeat_idx is not None else ""

    for src_i, name in enumerate(exp_names):
        if cfg_overrides is not None and name in cfg_overrides:
            cfg = cfg_overrides[name]
        else:
            cfg = _build_config_for_experiment(name)

        if per_source_match_strings is not None:
            # An explicit per-source override wins over the shared and repeat-derived matches.
            ds_string_match = str(per_source_match_strings[src_i])
        elif match_string is not None and str(match_string) and match_num_cosmo:
            ds_string_match = str(match_string)
        else:
            ds_string_match = f"None_{repeat_idx}" if repeat_idx is not None else ""
        print("Searching for checkpoint with match string:", ds_string_match)
        result = load_best_model_and_build_posterior(cfg, ds_string_match=ds_string_match)
        if result is None:
            raise RuntimeError(f"Failed to load best model for experiment '{name}'.")
        model, _, checkpoint_path = result
        model.eval()
        # use_KL off, since the loaded module may lack the KL encoder components.
        if hasattr(model, "use_KL"):
            model.use_KL = False
        models.append(model)
        dataset_quantities.update(dict.fromkeys(getattr(cfg, "dataset_quantities", [])))
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
        if isinstance(data_dict, dict):
            data_dict_dev = {k: v.to(device) for k, v in data_dict.items()}
        else:
            data_dict_dev = data_dict.to(device)
        theta_dev = theta.to(device)

        # Concatenate every model's embedding feature-wise.
        zs_batch = []
        theta_ref = None
        for m in models:
            encoder = m.embedding_net
            if getattr(encoder, "embedding_cut", None) == "hybrid_pre_head":
                z_m = encoder.get_frozen_features(data_dict_dev)
            else:
                encoder.only_return_mu = True
                z_m = encoder(data_dict_dev)
            zs_batch.append(z_m.detach())

            theta_m = theta_dev
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
    """Return the cached embedding dataset paths under the checkpoint dir.

    wandb_run_name already carries any ensemble suffix.
    """
    base_dir = f"{base_cfg.base_path}/checkpoints/{wandb_run_name}/datasets"
    train_path = f"{base_dir}/emb_train.pt"
    val_path = f"{base_dir}/emb_val.pt"
    test_path = f"{base_dir}/emb_test.pt"
    return train_path, val_path, test_path


def _encoder_fingerprint(model) -> Optional[str]:
    """Content hash of the assembled embedding_net weights.

    Keys are sorted, tensors moved to CPU, floating tensors cast to float32 and integer buffers
    hashed in their own dtype, so the hash is the same across devices and precisions.
    """
    net = getattr(model, "embedding_net", None)
    if net is None:
        return None
    try:
        h = hashlib.sha256()
        for k in sorted(net.state_dict().keys()):
            v = net.state_dict()[k]
            if not torch.is_tensor(v):
                h.update(f"{k}|scalar|{v!r}".encode())
                continue
            t = v.detach().cpu()
            if t.is_floating_point():
                kind, t = "f32", t.to(torch.float32)
            else:
                kind = str(t.dtype)
            t = t.contiguous()
            h.update(f"{k}|{kind}|{tuple(t.shape)}|".encode())
            h.update(t.numpy().tobytes())
        return h.hexdigest()[:16]
    except Exception as exc:
        print(f"[emb-cache] could not fingerprint an encoder: {exc}", flush=True)
        return None


def _build_cache_provenance(base_cfg, source_checkpoints=None, source_models=None) -> dict:
    """Everything that must match for a cached emb_*.pt to still be valid for this run.

    The cache key holds the target experiment, repeat and member, but not the source encoders,
    data patterns or theta box, so those are recorded here.
    """
    return {
        "data_patterns": _as_jsonable(getattr(base_cfg, "data_patterns", None)),
        "cosmo_param_names": _as_jsonable(getattr(base_cfg, "cosmo_param_names", None)),
        "split_seed": _as_jsonable(getattr(base_cfg, "split_seed", None)),
        "max_trainval_cosmos": _as_jsonable(getattr(base_cfg, "max_trainval_cosmos", None)),
        "test_shape_noise_idx": _as_jsonable(getattr(base_cfg, "test_shape_noise_idx", None)),
        "source_checkpoints": _as_jsonable(source_checkpoints),
        "source_encoder_fingerprints": (
            [_encoder_fingerprint(m) for m in source_models] if source_models else None
        ),
    }


def _as_jsonable(v):
    if isinstance(v, (list, tuple)):
        return [_as_jsonable(x) for x in v]
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


def _check_cache_provenance(path: str, stored: Optional[dict], expected: dict):
    """Compare a cache's recorded provenance with this run's. Raise on a real mismatch.

    A cache with no provenance (stored is None) only warns. The source keys are compared only when
    both sides recorded them.
    """
    if stored is None:
        print(
            f"[emb-cache] ⚠️  LEGACY CACHE (no provenance recorded): {path}\n"
            f"[emb-cache]     cannot verify it was built by this run's encoder / data / theta box. "
            f"Expected: {expected}",
            flush=True,
        )
        return
    mismatches = []
    for key, want in expected.items():
        if key in ("source_checkpoints", "source_encoder_fingerprints") and (
            want is None or stored.get(key) is None
        ):
            continue  # only comparable when both sides recorded it
        got = stored.get(key)
        if got != want:
            mismatches.append(f"  {key}: cache={got!r} != run={want!r}")
    if mismatches:
        raise RuntimeError(
            f"[emb-cache] STALE CACHE at {path} — it was NOT built for this run:\n"
            + "\n".join(mismatches)
            + "\n[emb-cache] Cached embeddings from a different encoder / fidelity / theta box "
              "live in a different coordinate frame from the persisted whitener and the trained "
              "flow, so every log-prob and posterior from them is meaningless. Delete the cache "
              "to force a recompute, or point the run at the matching source."
        )
    print(f"[emb-cache] provenance OK: {path}", flush=True)


def _save_embedding_cache(
    path: str,
    z: torch.Tensor,
    theta: torch.Tensor,
    emb_scaler: BaseScaler,
    cosmo_scaler: Optional[BaseScaler],
    provenance: Optional[dict] = None,
):
    """Save unscaled embeddings and cosmology vectors with the scaler statistics and provenance."""
    state = {
        "z": z.cpu(),
        "theta": theta.cpu(),
        "emb_mean": getattr(emb_scaler, "mean", None),
        "emb_std": getattr(emb_scaler, "std", None),
        "cosmo_min": getattr(cosmo_scaler, "min", None) if cosmo_scaler is not None else None,
        "cosmo_max": getattr(cosmo_scaler, "max", None) if cosmo_scaler is not None else None,
        "cosmo_param_names": getattr(cosmo_scaler, "parameter_names", None) if cosmo_scaler is not None else None,
        "provenance": provenance,
    }
    torch.save(state, path)


def _load_embedding_cache(path: str):
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
        param_names = ckpt.get("cosmo_param_names")
        if param_names is not None:
            from ..data.scaling import MinMaxScaler
            cosmo_scaler = MinMaxScaler(param_names)
            cosmo_scaler.min = ckpt["cosmo_min"]
            cosmo_scaler.max = ckpt["cosmo_max"]

    return z, theta, emb_scaler, cosmo_scaler, ckpt.get("provenance")


def resolve_whitener_path(
    pretrained_ckpt_path_or_dir: Optional[str],
    repeat_match: Optional[str] = None,
) -> Optional[str]:
    """Resolve the persisted whitener.pt for a pretrain run, for reuse at finetune and eval.

    For a checkpoint file it is <dirname>/datasets/whitener.pt. For a directory, the best checkpoint
    for repeat_match is found with get_best_checkpoint first. Returns None if nothing resolves.
    """
    if not pretrained_ckpt_path_or_dir:
        return None
    p = str(pretrained_ckpt_path_or_dir)

    if os.path.isfile(p):
        run_dir = os.path.dirname(p)
    elif os.path.isdir(p):
        best, _ = get_best_checkpoint(p, repeat_match or "")
        if not best:
            return None
        run_dir = os.path.dirname(best[0])
    else:
        # A non-existent path is treated as a file path.
        run_dir = os.path.dirname(p)

    candidate = os.path.join(run_dir, "datasets", WHITENER_FILENAME)
    return candidate if os.path.exists(candidate) else None


def fit_and_persist_whitener(
    z_train: torch.Tensor,
    k: int,
    save_path: str,
    *,
    fit_source_experiment: Optional[str] = None,
    fit_repeat_match: Optional[str] = None,
) -> WhitenPCAScaler:
    """Fit a WhitenPCAScaler on the pretrain train split and persist it, once.

    An existing save_path is loaded, not refit. This is the only place a whitener is fit.
    """
    k = int(k)
    if os.path.exists(save_path):
        wh = WhitenPCAScaler.load(save_path)
        if int(wh.k) != k:
            raise RuntimeError(
                f"[whiten] Existing whitener at {save_path} has k={wh.k} but config asks k={k}. "
                "Refusing to overwrite; use a distinct pretrain experiment for a different k."
            )
        print(f"[whiten] Reusing existing (fit-once) whitener k={wh.k} from {save_path}")
        return wh

    wh = WhitenPCAScaler(k=k)
    wh.fit(z_train, k=k)
    wh.fit_source_experiment = fit_source_experiment
    wh.fit_repeat_match = fit_repeat_match
    wh.save(save_path)
    print(
        f"[whiten] Fit whitener k={k} on {wh.fit_n_train_samples} rows (input_dim={wh.input_dim}), "
        f"source={fit_source_experiment} match={fit_repeat_match} -> {save_path}"
    )
    print(
        "[whiten] explained-variance ratio (top-k): "
        f"{[round(float(v), 4) for v in wh.explained_variance_ratio]}"
    )
    return wh


def build_embedding_dataloaders(
    train_loader,
    val_loader,
    test_loader,
    models: List[torch.nn.Module],
    base_cfg=None,
    wandb_run_name: Optional[str] = None,
    use_cache_if_exists=False,
    *,
    whiten_cfg: Optional[dict] = None,
    whitener_run_name: Optional[str] = None,
    source_checkpoints=None,
    is_pretrain_source: bool = False,
    pretrained_ckpt_path_or_dir: Optional[str] = None,
    repeat_match: Optional[str] = None,
    test_only: bool = False,
):
    """Construct scaled (or unscaled) embedding dataloaders from existing loaders and models.

    Returns (train, val, test) embedding loaders. config.scale_embeddings sets the scaling.
    whiten_cfg ({"k": 6}) replaces it with a whitener: a pretrain source fits and persists one,
    a finetune or eval run loads the pretrain's and raises if none resolves. test_only skips the
    train and val encoder passes unless a scaler must be fit on train embeddings.
    """

    scale_embeddings = bool(getattr(base_cfg, "scale_embeddings", True))
    # Train embeddings are needed only to fit a PerDim scaler or a pretrain-source whitener.
    needs_train_fit = (
        not test_only
        or (whiten_cfg is None and scale_embeddings)
        or (whiten_cfg is not None and is_pretrain_source)
    )
    if test_only and not needs_train_fit:
        print(
            "[build_embedding_dataloaders] test_only: skipping train/val embedding computes "
            "(no scaler fit required — whitener resolved or scaling off).",
            flush=True,
        )

    cosmo_scaler: Optional[BaseScaler] = None
    cosmo_param_names = []
    if base_cfg is not None and hasattr(base_cfg, "cosmo_param_names"):
        cosmo_param_names = list(base_cfg.cosmo_param_names)
        preset = dict(COSMO_PARAM_PRESET_MINMAX)
        _scaler_opts = getattr(base_cfg, "scaler_options", None) or {}
        _overrides = (_scaler_opts.get("cosmo") or {}).get("preset_overrides") or {}
        preset.update({k: tuple(v) for k, v in _overrides.items()})
        # Validation only: theta is already scaled by the raw loader, so the scaler is discarded.
        _build_cosmo_preset_scaler(preset, cosmo_param_names)
    cosmo_scaler = None
    train_z = val_z = test_z = None
    train_theta = val_theta = test_theta = None
    emb_scaler: Optional[BaseScaler] = None
    cache_used = False

    if base_cfg is not None and wandb_run_name is not None:
        train_path, val_path, test_path = _get_embedding_cache_paths(base_cfg, wandb_run_name)
        if use_cache_if_exists:
            if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
                expected_prov = _build_cache_provenance(base_cfg, source_checkpoints, models)
                train_z, train_theta, emb_scaler, cached_cosmo_scaler, prov = _load_embedding_cache(train_path)
                _check_cache_provenance(train_path, prov, expected_prov)
                val_z, val_theta, _, _, prov_v = _load_embedding_cache(val_path)
                _check_cache_provenance(val_path, prov_v, expected_prov)
                test_z, test_theta, _, _, prov_t = _load_embedding_cache(test_path)
                _check_cache_provenance(test_path, prov_t, expected_prov)
                print(
                    f"[emb-cache] HIT {train_path} (train {tuple(train_z.shape)}, "
                    f"val {tuple(val_z.shape)}, test {tuple(test_z.shape)})",
                    flush=True,
                )
                if cached_cosmo_scaler is not None:
                    if cosmo_scaler is not None:
                        import numpy as _np
                        same = (
                            list(getattr(cached_cosmo_scaler, "parameter_names", []) or [])
                            == list(getattr(cosmo_scaler, "parameter_names", []) or [])
                            and _np.allclose(_np.asarray(cached_cosmo_scaler.min, dtype=float),
                                             _np.asarray(cosmo_scaler.min, dtype=float))
                            and _np.allclose(_np.asarray(cached_cosmo_scaler.max, dtype=float),
                                             _np.asarray(cosmo_scaler.max, dtype=float))
                        )
                        if not same:
                            raise RuntimeError(
                                "[emb-cache] the cached theta scaler disagrees with the one this "
                                f"config builds ({train_path}).\n  cache names="
                                f"{getattr(cached_cosmo_scaler, 'parameter_names', None)}\n  run   names="
                                f"{getattr(cosmo_scaler, 'parameter_names', None)}\n"
                                "Using either one would scale theta into a box the flow was not "
                                "trained on. Delete the cache to force a recompute."
                            )
                    cosmo_scaler = cached_cosmo_scaler
                cache_used = True

    if not cache_used:
        if needs_train_fit:
            train_z, train_theta = compute_embeddings(models, train_loader)
        if not test_only:
            val_z, val_theta = compute_embeddings(models, val_loader)
        test_z, test_theta = compute_embeddings(models, test_loader)

        if scale_embeddings:
            emb_scaler = PerDimStandardScaler()
            emb_scaler.fit(train_z)
        else:
            emb_scaler = None

        if not test_only and base_cfg is not None and wandb_run_name is not None:
            print("Saving embedding caches to disk for future reuse...")
            print("Saving to paths:")
            print(" Train:", train_path)
            train_path, val_path, test_path = _get_embedding_cache_paths(base_cfg, wandb_run_name)
            os.makedirs(os.path.dirname(train_path), exist_ok=True)
            _prov = _build_cache_provenance(base_cfg, source_checkpoints, models)
            _save_embedding_cache(train_path, train_z, train_theta, emb_scaler, cosmo_scaler, _prov)
            _save_embedding_cache(val_path, val_z, val_theta, emb_scaler, cosmo_scaler, _prov)
            _save_embedding_cache(test_path, test_z, test_theta, emb_scaler, cosmo_scaler, _prov)

    # --- per-source whitening and PCA truncation ---
    if whiten_cfg is not None:
        if "k" not in whiten_cfg:
            raise ValueError(f"whiten_embeddings must contain 'k'; got {whiten_cfg!r}.")
        k = int(whiten_cfg["k"])
        if scale_embeddings:
            print(
                "[whiten] NOTE: scale_embeddings=True is IGNORED because whiten_embeddings is set "
                "(the whitener already standardises per-dim)."
            )

        if is_pretrain_source:
            if wandb_run_name is None:
                raise RuntimeError(
                    "[whiten] pretrain-source whitening requires wandb_run_name to derive the "
                    "persist path (datasets/whitener.pt)."
                )
            # The whitener lives in the run's own folder, never in the shared cache.
            w_name = whitener_run_name or wandb_run_name
            w_train_path, _, _ = _get_embedding_cache_paths(base_cfg, w_name)
            whitener_path = os.path.join(os.path.dirname(w_train_path), WHITENER_FILENAME)
            os.makedirs(os.path.dirname(whitener_path), exist_ok=True)
            emb_scaler = fit_and_persist_whitener(
                train_z,
                k,
                whitener_path,
                fit_source_experiment=getattr(base_cfg, "experiment_name", None),
                fit_repeat_match=repeat_match,
            )
        else:
            whitener_path = resolve_whitener_path(pretrained_ckpt_path_or_dir, repeat_match)
            if whitener_path is None:
                raise RuntimeError(
                    "[whiten] whiten_embeddings is enabled for a finetune/eval run but no persisted "
                    f"whitener.pt could be resolved from '{pretrained_ckpt_path_or_dir}' "
                    f"(repeat_match={repeat_match!r}). Refusing to refit downstream (Finding C3) — "
                    "ensure the matching *pretrain* run persisted datasets/whitener.pt."
                )
            emb_scaler = WhitenPCAScaler.load(whitener_path)
            if int(emb_scaler.k) != k:
                raise RuntimeError(
                    f"[whiten] resolved whitener k={emb_scaler.k} != config k={k} at {whitener_path}."
                )
            print(
                f"[whiten] Reusing pretrain whitener k={emb_scaler.k} (fit on "
                f"{emb_scaler.fit_n_train_samples} rows, source={emb_scaler.fit_source_experiment}) "
                f"from {whitener_path}"
            )
        # The whitener always applies, whatever scale_embeddings says.
        scale_embeddings = True

    train_ds = val_ds = None
    if train_z is not None:
        train_ds = EmbeddingDataset(
            train_z,
            train_theta,
            emb_scaler=emb_scaler,
            cosmo_scaler=cosmo_scaler,
            scale_embeddings=scale_embeddings,
        )
        emb_scaler = train_ds.emb_scaler
    elif scale_embeddings and emb_scaler is None:
        raise RuntimeError(
            "[build_embedding_dataloaders] test_only skipped the train embeddings but the "
            "embedding scaler needs a train fit — needs_train_fit logic is out of sync."
        )
    if val_z is not None:
        val_ds = EmbeddingDataset(
            val_z,
            val_theta,
            emb_scaler=emb_scaler,
            cosmo_scaler=cosmo_scaler,
            scale_embeddings=scale_embeddings,
        )
    test_ds = EmbeddingDataset(
        test_z,
        test_theta,
        emb_scaler=emb_scaler,
        cosmo_scaler=cosmo_scaler,
        scale_embeddings=scale_embeddings,
    )

    # Carry the H5 file list onto the unshuffled, freshly computed datasets for the sample dumps' ids.
    if not cache_used:
        from ..eval.utils import _resolve_test_paths as _resolve_loader_paths

        for ds, src_loader in ((val_ds, val_loader), (test_ds, test_loader)):
            if ds is None:
                continue
            ordered = isinstance(getattr(src_loader, "sampler", None), torch.utils.data.SequentialSampler)
            src_paths = _resolve_loader_paths(src_loader)
            if ordered and src_paths is not None and len(src_paths) == len(ds):
                ds.paths = list(src_paths)

    batch_size = getattr(train_loader, "batch_size", 128)
    num_workers = getattr(train_loader, "num_workers", 0)
    # The test batch is the joblib MCMC work unit downstream (config.emb_test_batch_size, default 32).
    test_batch_size = int(getattr(base_cfg, "emb_test_batch_size", None) or 32)

    train_emb_loader = (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        if train_ds is not None else None
    )
    val_emb_loader = (
        torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if val_ds is not None else None
    )
    test_emb_loader = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return train_emb_loader, val_emb_loader, test_emb_loader


class HybridFeaturesHead(torch.nn.Module):
    """Trainable head over cached frozen hybrid features.

    Input layout: [band_mu(band_dim) ++ N/S pre-head map features], as from
    KidsHybridBandpowersMaps.get_frozen_features (embedding_cut='hybrid_pre_head'). The stack is
    map head MLP -> optional patch LayerNorm -> fusion head.
    """

    def __init__(
        self,
        emb_dim: int,
        band_dim: int = 8,
        dim_patch: int = 8,
        head_hidden: int = 256,
        hybrid_output_dim: int = 8,
        hybrid_head_hidden: Optional[int] = None,
        patch_norm: Optional[str] = None,
    ):
        super().__init__()
        self.band_dim = int(band_dim)
        feat_dim = int(emb_dim) - self.band_dim
        if feat_dim <= 0:
            raise ValueError(f"emb_dim {emb_dim} too small for band_dim {band_dim}")
        self.head = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, head_hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(head_hidden, dim_patch),
        )
        if patch_norm is None:
            self.patch_norm_layer = None
        elif patch_norm == "layernorm":
            self.patch_norm_layer = torch.nn.LayerNorm(dim_patch, elementwise_affine=False)
        else:
            raise ValueError(f"Unknown patch_norm '{patch_norm}'")
        fusion_in = self.band_dim + int(dim_patch)
        if hybrid_head_hidden:
            self.hybrid_head = torch.nn.Sequential(
                torch.nn.Linear(fusion_in, int(hybrid_head_hidden)),
                torch.nn.GELU(),
                torch.nn.Linear(int(hybrid_head_hidden), hybrid_output_dim),
            )
        else:
            self.hybrid_head = torch.nn.Linear(fusion_in, hybrid_output_dim)
        self.output_dim = int(hybrid_output_dim)

    def forward(self, x):
        band_mu = x[..., : self.band_dim]
        patch_mu = self.head(x[..., self.band_dim:])
        if self.patch_norm_layer is not None:
            patch_mu = self.patch_norm_layer(patch_mu)
        return self.hybrid_head(torch.cat([band_mu, patch_mu], dim=-1))

    def compress(self, x):
        return self.forward(x)


def build_nde_on_embeddings(
    *,
    emb_dim: int,
    base_cfg,
    test_loader=None,
    device: Optional[torch.device] = None,
):
    """Construct an untrained NDE Lightning module that operates on embeddings.

    emb_dim is the width of the concatenated embedding. base_cfg supplies the flow fields and
    inference_mode. Returns an NDELightningModule (npe) or LikelihoodNDELightningModule (nle).
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conditioning_dim = emb_dim
    inference_dim = len(base_cfg.cosmo_param_names)

    if getattr(base_cfg, "embedding_head_type", None) == "hybrid_features":
        head_kwargs = dict(getattr(base_cfg, "embedding_head_kwargs", {}) or {})
        embedding_net = HybridFeaturesHead(emb_dim=emb_dim, **head_kwargs).to(device)
        conditioning_dim = embedding_net.output_dim
    else:
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


def load_embeddings_checkpoint(model, checkpoint_path, device, *, tag: str = ""):
    """Restore a trained NDE-on-embeddings checkpoint, reporting what actually loaded.

    Loads with strict=False, and raises when the checkpoint matched no key or any model.flow.*
    tensor is missing, since the density head would then be random.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    target_keys = set(model.state_dict().keys())
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    matched = len(target_keys) - len(missing)

    label = f"[emb-ckpt]{(' ' + tag) if tag else ''}"
    print(
        f"{label} {checkpoint_path} -> loaded {matched}/{len(target_keys)} keys "
        f"(missing {len(missing)}, unexpected {len(unexpected)})",
        flush=True,
    )
    if missing:
        print(f"{label}   missing[:8]: {missing[:8]}", flush=True)
    if unexpected:
        print(f"{label}   unexpected[:8]: {unexpected[:8]}", flush=True)

    if matched == 0:
        raise RuntimeError(
            f"{label} checkpoint '{checkpoint_path}' matched NONE of the model's "
            f"{len(target_keys)} parameters — the model is entirely randomly initialised. "
            "Refusing to evaluate."
        )
    missing_flow = [k for k in missing if k.startswith("model.flow.")]
    if missing_flow:
        raise RuntimeError(
            f"{label} checkpoint '{checkpoint_path}' is missing {len(missing_flow)} "
            f"'model.flow.*' tensors (e.g. {missing_flow[:5]}), so the density head is partly or "
            "wholly RANDOM. Any log-prob / posterior from this model is meaningless. "
            "Refusing to evaluate."
        )
    return model


def _parse_best_val_from_ckpt_name(ckpt_path: Optional[str]) -> Optional[float]:
    """Extract the recorded best val_log_prob (NLL) from a checkpoint filename."""
    if not ckpt_path:
        return None
    m = re.search(r"val_log_prob=(-?\d+(?:\.\d+)?)", os.path.basename(str(ckpt_path)))
    return float(m.group(1)) if m else None


@torch.no_grad()
def _compute_val_nll(model, val_loader) -> float:
    """Mean validation NLL of `model` on `val_loader` (matches the module's validation_step loss)."""
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    # NPE and NLE take their flow arguments in opposite order: forward(theta, cond=z) vs forward(z, cond=theta).
    nle_style = isinstance(
        model, (LikelihoodNDELightningModule, EnsembleLikelihoodNDELightningModule)
    )
    total, n = 0.0, 0
    for batch in val_loader:
        x, theta = batch
        x = x.to(device)
        theta = theta.to(device)
        preds = model.forward(x, cond=theta) if nle_style else model.forward(theta, cond=x)
        loss = model.compute_loss(preds, theta)
        bs = int(x.shape[0])
        total += float(loss) * bs
        n += bs
    return total / max(n, 1)


def _warmstart_regression_guard(model, val_loader, base_cfg, pretrained_ckpt_path: str):
    """Guard (c): abort if the finetune ep0 val NLL is a scratch-level gap above the pretrain best.

    Genuine warm starts land about 2-5 nats above the pretrain best, scratch runs 15 or more.
    The threshold is config.whiten_warmstart_max_gap_nats (default 12).
    """
    max_gap = getattr(base_cfg, "whiten_warmstart_max_gap_nats", 12.0)
    if max_gap is None or max_gap <= 0:
        print("[whiten][guard-c] warm-start regression guard disabled (whiten_warmstart_max_gap_nats).")
        return
    pretrain_best = _parse_best_val_from_ckpt_name(pretrained_ckpt_path)
    ep0 = _compute_val_nll(model, val_loader)
    if pretrain_best is None:
        print(
            f"[whiten][guard-c] could not parse pretrain best val from '{pretrained_ckpt_path}'; "
            f"skipping guard (finetune ep0 val NLL={ep0:.3f})."
        )
        return
    gap = ep0 - pretrain_best
    print(
        f"[whiten][guard-c] finetune ep0 val NLL={ep0:.3f}, pretrain best val={pretrain_best:.3f}, "
        f"gap={gap:.3f} nats (threshold {max_gap})."
    )
    if gap > max_gap:
        raise RuntimeError(
            f"[whiten][guard-c] Warm-start regression: finetune ep0 val NLL ({ep0:.3f}) is {gap:.3f} "
            f"nats above the pretrain best ({pretrain_best:.3f}), exceeding the scratch-signature "
            f"threshold of {max_gap} nats. The warm start is NOT taking effect (likely a "
            "mis-resolved/mismatched whitener or flow). Raise whiten_warmstart_max_gap_nats to "
            "override if this gap is genuinely expected."
        )


def fit_nde_on_embeddings(emb_dim: int, train_loader, val_loader, test_loader, base_cfg, run_name):
    """Build and train an NDE Lightning module on precomputed embeddings.

    The encoder is an IdentityEmbedding and the flow comes from base_cfg. config.inference_mode
    picks 'npe' (default, p(theta | z)) or 'nle' (p(z | theta)).
    """

    model = build_nde_on_embeddings(emb_dim=emb_dim, base_cfg=base_cfg, test_loader=test_loader)

    whiten_on = getattr(base_cfg, "whiten_embeddings", None) is not None
    if getattr(base_cfg, 'load_pretrained_flow', False):
        pretrained_band_ckpt = getattr(base_cfg, 'pretrained_band_ckpt_path', None)
        if pretrained_band_ckpt is None:
            raise ValueError("Config flag 'load_pretrained_flow' is True but 'pretrained_band_ckpt_path' is not set.")
        # Guard (a): with whitening on, a wrong-k flow raises instead of leaving random layers.
        model._load_pretrained_flow(pretrained_band_ckpt, freeze=False, error_on_mismatch=whiten_on)
        # Guard (c): warm-start regression check (whitening runs only).
        if whiten_on:
            _warmstart_regression_guard(model, val_loader, base_cfg, pretrained_band_ckpt)

    num_gpus = torch.cuda.device_count()
    accelerator = "gpu" if num_gpus > 0 else "cpu"
    devices = num_gpus if num_gpus > 0 else 1
    strategy = "ddp" if num_gpus > 1 else "auto"

    monitor_string = f"val_{model.loss_name}"
    num_trainval_cosmos = getattr(base_cfg, 'max_trainval_cosmos', None)
    match_string_logger = base_cfg.match_string if base_cfg.match_string else ""

    inference_mode = getattr(base_cfg, "inference_mode", "npe").lower()
    wandb_logger = wandb.init(
        project=getattr(base_cfg, "project", "emb-nde"),
        group=f"embeddings_nde_{inference_mode}",
        name=(run_name),
        reinit=True,
    )

    wandb_logger_name = wandb_logger.name if wandb_logger else "no_logger"
    # This path builds its own Trainer, so it needs fit_model's temp-dir redirect too.
    redirect_tempdir(base_cfg.base_path)
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
        gradient_clip_val=0.5,
        precision="32",
    )

    trainer.fit(model, train_loader, val_loader)
