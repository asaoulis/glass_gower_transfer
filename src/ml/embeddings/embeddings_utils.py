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
from ..data.scaling import BaseScaler, PerDimStandardScaler, WhitenPCAScaler
from ..utils import _build_cosmo_preset_scaler
from ..data.constants import COSMO_PARAM_PRESET_MINMAX

# Filename for the persisted per-source whitener, colocated with a run's emb_{train,val,test}.pt.
WHITENER_FILENAME = "whitener.pt"


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
    per_source_match_strings: Optional[Sequence[str]] = None,
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
        per_source_match_strings: optional per-source run-match overrides, one per entry of
            `exp_names`. Without it every source resolves with the SAME match string, which is
            right when the sources are distinct experiments but makes it impossible to stack
            several REPEATS of one experiment (they share a checkpoint dir and differ only in the
            run subdir, e.g. `pretrain_ncosmoNone_0` .. `_4`). Supplying
            ["None_0", ..., "None_4"] alongside the same name five times stacks those five
            encoders. Default None preserves the previous behaviour exactly.

    Returns:
        models: list of trained NDE models
        dataset_quantities: merged dataset quantities across experiments
        checkpoint_paths: list of checkpoint paths
    """
    if per_source_match_strings is not None and len(per_source_match_strings) != len(exp_names):
        raise ValueError(
            f"per_source_match_strings has {len(per_source_match_strings)} entries but there are "
            f"{len(exp_names)} source experiments; they must correspond one-to-one."
        )
    models = []
    dataset_quantities = set()
    checkpoint_paths = []

    # Prefer explicit repeat-bound match string (e.g. 'ncosmo30_0').
    # Fallback to legacy suffix matching by repeat index.
    # if match_string is not None and str(match_string):
    #     ds_string_match = str(match_string)
    # else:
    ds_string_match = f"_{repeat_idx}" if repeat_idx is not None else ""

    for src_i, name in enumerate(exp_names):
        if cfg_overrides is not None and name in cfg_overrides:
            cfg = cfg_overrides[name]
        else:
            cfg = _build_config_for_experiment(name)

        if per_source_match_strings is not None:
            # Explicit per-source override wins over both the shared match_string and the
            # repeat-derived default — this is the only way to bind several REPEATS of one
            # experiment as separate sources.
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
            # Phase 2b: optional alternative cut point (e.g. 'hybrid_pre_head' caches the frozen
            # band_mu ++ pre-head map features instead of the final z). Default unchanged.
            if getattr(encoder, "embedding_cut", None) == "hybrid_pre_head":
                z_m = encoder.get_frozen_features(data_dict_dev)
            else:
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


def resolve_whitener_path(
    pretrained_ckpt_path_or_dir: Optional[str],
    repeat_match: Optional[str] = None,
) -> Optional[str]:
    """Resolve the persisted `whitener.pt` for a pretrain run, for reuse at finetune/eval.

    - If `pretrained_ckpt_path_or_dir` is a checkpoint FILE (as resolved by `train_embedding_run`),
      the whitener lives at `<dirname>/datasets/whitener.pt` (same run folder as the flow ckpt).
    - If it is a DIRECTORY (as passed by the eval-only loader), locate the best matching checkpoint
      via `get_best_checkpoint(dir, repeat_match)` — the SAME machinery used to resolve the flow —
      then take that run folder's `datasets/whitener.pt`.

    Returns None if nothing resolves (the caller decides whether that is fatal).
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
        # Non-existent path string: best-effort treat as a file path and use its parent.
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
    """Fit a WhitenPCAScaler on the pretrain train split and persist it (idempotent: fit ONCE).

    If `save_path` already exists it is loaded, not refit (the pretrain split is deterministic, so a
    re-run reproduces identical stats; not refitting is the stronger, cheaper guarantee). This is the
    single place a whitener is ever *fit* — finetune/eval only ever load it (Finding C3).
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
    is_pretrain_source: bool = False,
    pretrained_ckpt_path_or_dir: Optional[str] = None,
    repeat_match: Optional[str] = None,
    test_only: bool = False,
):
    """Construct *scaled* (or optionally unscaled) embedding dataloaders from existing loaders and models.

    Embeddings scaling is controlled by config flag `scale_embeddings` (default True).

    When `whiten_cfg` (e.g. {"k": 6}) is provided, per-source whitening + optional PCA truncation
    REPLACES `scale_embeddings` for the embedding scaler:
      - pretrain source (`is_pretrain_source=True`): fit the whitener ONCE on this run's train split
        and persist it next to the emb cache (`datasets/whitener.pt`).
      - finetune/eval (`is_pretrain_source=False`): RESOLVE and REUSE the pretrain's persisted
        whitener; never refit (Finding C3). Hard-fails if it cannot be resolved.

    When `test_only` (eval-time sampling: only the TEST loader is consumed downstream), the
    expensive train/val encoder passes are SKIPPED unless the embedding scaler genuinely has to
    be fit on train embeddings (scale_embeddings without a resolved whitener, or a
    pretrain-source whitener fit). For the production whitened-NLE configs this cuts the work
    per ensemble member from ~10k encoder forwards to just the test points.
    """

    scale_embeddings = bool(getattr(base_cfg, "scale_embeddings", True))
    # Train embeddings are only needed to FIT something: a PerDim scaler (scale_embeddings,
    # no whitener) or a pretrain-source whitener. A resolved (finetune/eval) whitener needs no fit.
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

    # Build cosmology preset scaler from config if available
    cosmo_scaler: Optional[BaseScaler] = None
    cosmo_param_names = []
    if base_cfg is not None and hasattr(base_cfg, "cosmo_param_names"):
        cosmo_param_names = list(base_cfg.cosmo_param_names)
        # Honour the SAME per-config box overrides as the training path (src/ml/utils.py:305-312).
        # A config may legitimately supply boxes for parameters absent from the global preset — the
        # 15-param vector's `b_g_bin1..6` are exactly that (the BGP campaign deliberately keeps them
        # as overrides rather than mutating the shared constant). Without this merge the lookup below
        # raises on them, which is a FALSE failure: the boxes were supplied, just not where this
        # looked. Empty/absent overrides => the global preset, byte-identical to before.
        preset = dict(COSMO_PARAM_PRESET_MINMAX)
        _scaler_opts = getattr(base_cfg, "scaler_options", None) or {}
        _overrides = (_scaler_opts.get("cosmo") or {}).get("preset_overrides") or {}
        preset.update({k: tuple(v) for k, v in _overrides.items()})
        # Kept for its VALIDATION value only — it fails fast when a cosmo parameter has no box.
        # The scaler itself is deliberately discarded on the next line: theta reaching this point
        # has ALREADY been min-max scaled into the prior box by the raw loader's
        # TransformingDataset, so applying a second cosmo scaler here would double-scale it.
        _build_cosmo_preset_scaler(preset, cosmo_param_names)
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
        if needs_train_fit:
            train_z, train_theta = compute_embeddings(models, train_loader)
        if not test_only:
            val_z, val_theta = compute_embeddings(models, val_loader)
        test_z, test_theta = compute_embeddings(models, test_loader)

        # Global per-variable standard scaler on training embeddings (optional)
        if scale_embeddings:
            emb_scaler = PerDimStandardScaler()
            emb_scaler.fit(train_z)
        else:
            emb_scaler = None

        # Save caches if possible
        if not test_only and base_cfg is not None and wandb_run_name is not None:
            print("Saving embedding caches to disk for future reuse...")
            print("Saving to paths:")
            print(" Train:", train_path)
            train_path, val_path, test_path = _get_embedding_cache_paths(base_cfg, wandb_run_name)
            os.makedirs(os.path.dirname(train_path), exist_ok=True)
            _save_embedding_cache(train_path, train_z, train_theta, emb_scaler, cosmo_scaler)
            _save_embedding_cache(val_path, val_z, val_theta, emb_scaler, cosmo_scaler)
            _save_embedding_cache(test_path, test_z, test_theta, emb_scaler, cosmo_scaler)

    # --- Per-source whitening / PCA truncation (research Finding C3 fix) ------------------------
    # Both the cache-hit and fresh-compute paths converge here with raw train_z/val_z/test_z, so a
    # single insertion point covers pretrain (fit+persist) and finetune/eval (resolve+reuse).
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
            # The whitener must live in the RUN's own folder, not in a shared embedding cache:
            # finetune/eval resolve it via `resolve_whitener_path(<flow ckpt>)`, i.e.
            # `<ckpt dir>/datasets/whitener.pt`. With no `embedding_cache_name` the two coincide
            # (wandb_run_name == source_run_name == "<experiment>/<run>"), so this is a no-op for
            # every existing row. When a cache IS shared across head variants, pinning the whitener
            # to the run keeps Stage-B able to find it AND stops two different k's from colliding on
            # one whitener.pt.
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
        # Force the EmbeddingDataset to APPLY the whitener regardless of the scale_embeddings flag
        # (which the production NLE configs leave False).
        scale_embeddings = True

    # Build datasets using tensors and scalers
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

    # Carry the underlying H5 file list onto the embedding datasets so downstream sample
    # dumps can tag each test point with its sim/aug id (_resolve_test_paths looks for
    # `.paths`). Embedding row i comes from loader item i (sequential compute, shuffle
    # irrelevant to the mapping only for the UNSHUFFLED val/test loaders — the train loader
    # shuffles, so only attach where the loader preserves order). Skip when embeddings came
    # from the on-disk cache: the cached rows may predate the current split, so positional
    # alignment cannot be trusted.
    if not cache_used:
        from ..eval.utils import _resolve_test_paths as _resolve_loader_paths

        for ds, src_loader in ((val_ds, val_loader), (test_ds, test_loader)):
            if ds is None:
                continue
            ordered = isinstance(getattr(src_loader, "sampler", None), torch.utils.data.SequentialSampler)
            # Walks .dataset AND TransformingDataset's .base_ds (the H5 test loader is wrapped).
            src_paths = _resolve_loader_paths(src_loader)
            if ordered and src_paths is not None and len(src_paths) == len(ds):
                ds.paths = list(src_paths)

    batch_size = getattr(train_loader, "batch_size", 128)
    num_workers = getattr(train_loader, "num_workers", 0)
    # Test batch size doubles as the joblib MCMC work unit (one job per batch downstream);
    # opt-in override via config, default = the historical 32.
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
    """Trainable head over cached frozen hybrid features (Phase 2b T2).

    Input vector layout = [band_mu(band_dim) ++ N/S pre-head map features] as produced by
    KidsHybridBandpowersMaps.get_frozen_features (embedding_cut='hybrid_pre_head'). Reproduces
    the trainable hybrid head stack: map head MLP -> optional patch LayerNorm -> fusion head.
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

    # Phase 2b: optionally train a small head over the cached features instead of feeding the
    # raw embedding vector straight into the flow. Default (None) preserves prior behaviour.
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
    # NPE and NLE pass their flow arguments in OPPOSITE orders, so this helper must dispatch
    # on the module type to really match "the module's validation_step loss" as documented:
    #   NPE  models p(theta | z): forward(theta, cond=z)   -- npe.py:391-393
    #   NLE  models p(z | theta): forward(z,     cond=theta) -- nle.py:36-38
    # LikelihoodNDELightningModule SUBCLASSES NDELightningModule, so the NLE test must come
    # first. The ensemble modules subclass pl.LightningModule directly (they are not in that
    # hierarchy) and so are named explicitly -- ensemble_nle.py:93-100 vs ensemble_npe.py:63-71.
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

    Local calibration (research task): genuine warm starts land ~2-5 nats above the pretrain best;
    scratch / mis-framed-input runs land >=15 nats above. Threshold defaults to ~12 nats.
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

    Uses an IdentityEmbedding as the encoder and otherwise reuses the
    flow configuration from `base_cfg`.

    A config flag ``inference_mode`` controls whether we run in
    NPE (posterior) or NLE (likelihood) mode. Allowed values:
      * 'npe' (default): NDELightningModule, modelling p(theta | z).
      * 'nle'         : LikelihoodNDELightningModule, modelling p(z | theta).
    """

    model = build_nde_on_embeddings(emb_dim=emb_dim, base_cfg=base_cfg, test_loader=test_loader)

    whiten_on = getattr(base_cfg, "whiten_embeddings", None) is not None
    if getattr(base_cfg, 'load_pretrained_flow', False):
        pretrained_band_ckpt = getattr(base_cfg, 'pretrained_band_ckpt_path', None)
        if pretrained_band_ckpt is None:
            raise ValueError("Config flag 'load_pretrained_flow' is True but 'pretrained_band_ckpt_path' is not set.")
        # Guard (a): when whitening is on, a shape-mismatched (wrong-k) flow must fail HARD rather
        # than silently leave layers randomly initialised.
        model._load_pretrained_flow(pretrained_band_ckpt, freeze=False, error_on_mismatch=whiten_on)
        # Guard (c): warm-start regression check (whitening runs only).
        if whiten_on:
            _warmstart_regression_guard(model, val_loader, base_cfg, pretrained_band_ckpt)

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
