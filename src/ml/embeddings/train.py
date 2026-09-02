"""Reusable training helpers for the embeddings pipeline.

This module factors out the orchestration code that was previously embedded
in the top-level `train_embeddings.py` script.

Design goals:
- Keep functions small and composable.
- Avoid duplication with existing helpers in `embeddings_utils.py`.
- Leave the heavyweight model/dataloader logic in `embeddings_utils.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence
import re

import torch

from config.default import get_default_config
from config.experiments import experiments

from ..models.utils import create_run_name
from ..models.utils import parse_repeat_and_ensemble_from_match
from ..utils import prepare_data_parameters
from ..utils import build_ensemble_model_from_checkpoints, is_ensemble_eval_active
from ..eval.loading_model import get_best_checkpoint
from ..utils import set_seed_for_repeat_and_ensemble

NUM_SAMPLES = 8000
NUM_WARMUP = 500

@dataclass(frozen=True)
class EmbeddingTrainArgs:
    """High-level inputs for an embeddings run."""

    target_experiment: str
    source_experiments: Sequence[str]


@dataclass(frozen=True)
class LoadedEmbeddingArtifacts:
    """Container returned by `load_embedding_model_with_dataloader`.

    This keeps the call site simple while still exposing everything downstream
    code usually needs for sampling/evaluation.
    """

    model: object
    scalers: Dict[str, object]
    test_loader: object
    config: object
    checkpoint_path: Optional[str]


def build_cfg_from_experiment_dict(exp_name: str, exp_dict: dict, *, n_cosmo: Optional[int] = None):
    """Build a config object and optionally set `max_trainval_cosmos`."""

    cfg = get_default_config()
    cfg.experiment_name = exp_name

    for k, v in exp_dict.items():
        if k == "max_trainval_cosmos":
            continue
        setattr(cfg, k, v)

    if n_cosmo is not None:
        cfg.max_trainval_cosmos = int(n_cosmo)

    # Default inference mode for embeddings is 'npe' unless experiment overrides it.
    if not hasattr(cfg, "inference_mode"):
        cfg.inference_mode = exp_dict.get("inference_mode", "npe")

    return cfg


def format_ncosmo_tag(n_cosmo: Optional[int]) -> str:
    return "ncosmoNone" if n_cosmo is None else f"ncosmo{int(n_cosmo)}"


def _extract_ncosmo_from_match(match_string: str) -> Optional[int]:
    """Extract ncosmo value from a match string.

    Accepts both `ncosmo123` and the common typo `ncosm123`.
    """

    m = re.search(r"ncosmo?(\d+)", str(match_string))
    if m is None:
        return None
    return int(m.group(1))


def _select_best_checkpoint_for_match(cfg, match_string: str) -> Optional[str]:
    """Return the best checkpoint path under the experiment folder for `match_string`."""

    experiment_path = f"{cfg.base_path}/checkpoints/{cfg.experiment_name}"
    best_ckpts, val_losses = get_best_checkpoint(experiment_path, match_string)
    if not best_ckpts:
        return None
    best_idx = min(range(len(best_ckpts)), key=lambda i: val_losses[i]) if val_losses else 0
    return best_ckpts[best_idx]


def _try_parse_repeat_idx(match_string: str) -> Optional[int]:
    """Best-effort extraction of repeat index from match string suffix."""

    try:
        repeat_idx, _ = parse_repeat_and_ensemble_from_match(match_string)
        return int(repeat_idx)
    except Exception:
        return None


def _build_embedding_test_loader_for_cfg(
    cfg,
    source_models,
    *,
    whiten_cfg=None,
    is_pretrain_source: bool = False,
    pretrained_ckpt_path_or_dir=None,
    repeat_match=None,
):
    """Build scaled embedding test loader (and scalers) for one config instance.

    Only the TEST embedding loader is consumed downstream, so `test_only=True` lets
    build_embedding_dataloaders skip the (expensive, otherwise-discarded) train/val encoder
    passes whenever no embedding-scaler fit is required (e.g. resolved whitener)."""

    from .embeddings_utils import build_embedding_dataloaders

    scalers, train_loader, val_loader, test_loader = prepare_data_parameters(cfg)
    _, _, test_emb_loader = build_embedding_dataloaders(
        train_loader,
        val_loader,
        test_loader,
        source_models,
        base_cfg=cfg,
        wandb_run_name=None,
        use_cache_if_exists=True,
        whiten_cfg=whiten_cfg,
        is_pretrain_source=is_pretrain_source,
        pretrained_ckpt_path_or_dir=pretrained_ckpt_path_or_dir,
        repeat_match=repeat_match,
        test_only=True,
    )
    return scalers, test_emb_loader


def _build_embeddings_model_from_cfg_checkpoint(cfg, test_dataloader=None):
    """Build NDE-on-embeddings model and load cfg.checkpoint_path when provided."""

    from .embeddings_utils import build_nde_on_embeddings, load_embeddings_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if test_dataloader is None:
        raise ValueError("Embeddings model builder requires a test_dataloader.")

    emb_dim = next(iter(test_dataloader))[0].shape[-1]
    model = build_nde_on_embeddings(emb_dim=emb_dim, base_cfg=cfg, test_loader=test_dataloader, device=device)

    checkpoint_path = getattr(cfg, "checkpoint_path", None)
    if checkpoint_path:
        load_embeddings_checkpoint(model, checkpoint_path, device,
                                   tag=str(getattr(cfg, "experiment_name", "")))

    model.to(device)
    model.eval()
    model.test_dataloader = test_dataloader
    return model


def load_embedding_model_with_dataloader(
    experiment_name: str,
    match_string: str,
    *,
    source_experiments: Sequence[str],
    config_overrides: Optional[Dict[str, object]] = None,
) -> LoadedEmbeddingArtifacts:
    """Load one trained embeddings model and its matching embedding test loader.

    This function is embeddings-only and reuses existing helpers for:
    - source model loading,
    - embedding dataloader construction,
    - NDE-on-embeddings model construction,
    - best-checkpoint selection for the provided match string.
    """

    if experiment_name not in experiments:
        raise ValueError(f"Experiment '{experiment_name}' not found in config.experiments.experiments.")
    if not source_experiments:
        raise ValueError("source_experiments must be a non-empty sequence for embeddings model loading.")

    exp_dict = experiments[experiment_name]
    n_cosmo = _extract_ncosmo_from_match(match_string)

    cfg = build_cfg_from_experiment_dict(experiment_name, exp_dict, n_cosmo=n_cosmo)
    cfg.match_string = str(match_string)
    cfg.test_shape_noise_idx = [0]
    if config_overrides:
        for key, value in config_overrides.items():
            setattr(cfg, key, value)

    repeat_idx = _try_parse_repeat_idx(match_string)
    source_cfg_overrides = build_source_cfg_overrides(source_experiments, n_cosmo=n_cosmo)

    # Whitening context (Finding C3): resolve+reuse the pretrain's persisted whitener at eval time.
    # cfg.pretrained_band_ckpt_path is the (unresolved) pretrain checkpoint ROOT dir here — the
    # dir-branch of resolve_whitener_path handles that via get_best_checkpoint(dir, repeat_match).
    whiten_cfg = getattr(cfg, "whiten_embeddings", None)
    whiten_is_pretrain_source = (exp_dict.get("pretrained_band_ckpt_path") is None)
    whiten_ckpt_dir = getattr(cfg, "pretrained_band_ckpt_path", None)
    whiten_repeat_match = f"None_{repeat_idx}" if repeat_idx is not None else None

    from .embeddings_utils import load_pretrained_models
    match_num_cosmo = getattr(cfg, "match_num_cosmo", False)
    if not match_num_cosmo:
        pretrained_models_match_string = "None_" + match_string.split("_")[1]  # e.g. "ncosmo30_0" -> "_0"
    else:
        pretrained_models_match_string = match_string  # use full match_string for loading sources if match_num_cosmo is True
    source_models, dataset_quantities, _ = load_pretrained_models(
        list(source_experiments),
        cfg_overrides=source_cfg_overrides,
        repeat_idx=repeat_idx,
        match_string=pretrained_models_match_string,
        per_source_match_strings=getattr(cfg, "source_match_strings", None),
    )
    cfg.dataset_quantities = dataset_quantities

    if is_ensemble_eval_active(cfg):
        if repeat_idx is None:
            raise ValueError(
                f"Ensemble loading requires repeat-bound match_string (e.g. ncosmo30_0), got '{match_string}'."
            )

        n_ens = int(getattr(cfg, "ensemble_repeats", 1) or 1)
        member_test_loaders = []
        scalers = None

        for j in range(n_ens):
            cfg_j = build_cfg_from_experiment_dict(experiment_name, exp_dict, n_cosmo=n_cosmo)
            cfg_j.match_string = str(pretrained_models_match_string)
            cfg_j.test_shape_noise_idx = [0]
            if config_overrides:
                for key, value in config_overrides.items():
                    setattr(cfg_j, key, value)
            cfg_j.dataset_quantities = dataset_quantities

            cfg_j.split_seed = 42
            set_seed_for_repeat_and_ensemble(cfg_j, repeat_idx=repeat_idx, ensemble_idx=j)

            scalers_j, test_emb_loader_j = _build_embedding_test_loader_for_cfg(
                cfg_j,
                source_models,
                whiten_cfg=whiten_cfg,
                is_pretrain_source=whiten_is_pretrain_source,
                pretrained_ckpt_path_or_dir=whiten_ckpt_dir,
                repeat_match=whiten_repeat_match,
            )
            if scalers is None:
                scalers = scalers_j
            member_test_loaders.append(test_emb_loader_j)

        # NB: member checkpoints live in the TARGET experiment's run folders
        # (…{match_string}_ens{j}…), so search with the target run match. The
        # "None_"-form (pretrained_models_match_string) is ONLY for resolving the
        # SOURCE-encoder checkpoints when match_num_cosmo=False — using it here finds
        # no members (folders are e.g. pretrain_ncosmo300_0_ens0_<source>).
        model = build_ensemble_model_from_checkpoints(
            cfg,
            test_loader=None,
            match_string=str(match_string),
            member_test_loaders=member_test_loaders,
            model_builder=_build_embeddings_model_from_cfg_checkpoint,
        )
        if model is None:
            raise RuntimeError(
                f"Failed to build embeddings ensemble for experiment '{experiment_name}' and match '{match_string}'."
            )

        checkpoint_path = None
        test_emb_loader = member_test_loaders[0]
    else:
        scalers, test_emb_loader = _build_embedding_test_loader_for_cfg(
            cfg,
            source_models,
            whiten_cfg=whiten_cfg,
            is_pretrain_source=whiten_is_pretrain_source,
            pretrained_ckpt_path_or_dir=whiten_ckpt_dir,
            repeat_match=whiten_repeat_match,
        )
        # Same as the ensemble branch: the target experiment's own run folders are named
        # by the full run match string, not the source-encoder "None_" form.
        checkpoint_path = _select_best_checkpoint_for_match(cfg, str(match_string))
        if checkpoint_path is None:
            raise RuntimeError(
                f"No checkpoint found for embeddings experiment '{experiment_name}' and match '{match_string}'."
            )
        cfg.checkpoint_path = checkpoint_path
        model = _build_embeddings_model_from_cfg_checkpoint(cfg, test_dataloader=test_emb_loader)

    return LoadedEmbeddingArtifacts(
        model=model,
        scalers=scalers,
        test_loader=test_emb_loader,
        config=cfg,
        checkpoint_path=checkpoint_path,
    )


def build_source_cfg_overrides(source_experiments: Sequence[str], *, n_cosmo: Optional[int]) -> Optional[Dict[str, object]]:
    """Build per-source cfg overrides for source-based `max_trainval_cosmos` iteration."""

    if n_cosmo is None:
        return None

    overrides: Dict[str, object] = {}
    for name in source_experiments:
        if name not in experiments:
            raise ValueError(f"Source experiment '{name}' not found in config.experiments.experiments.")
        overrides[name] = build_cfg_from_experiment_dict(name, experiments[name], n_cosmo=n_cosmo)
    return overrides


def train_embedding_run(
    *,
    target_cfg,
    source_experiments: Sequence[str],
    source_cfg_overrides: Optional[Dict[str, object]],
    run_name: str,
    repeat_idx: int,
    run_evaluation: bool = True,
):
    """One end-to-end run: load sources, build embedding loaders, train flow on embeddings.

    Returns an evaluation context dict that can be reused by the caller to run
    deferred evaluation (e.g. once after all ensemble members have finished).
    """

    from .embeddings_utils import fit_nde_on_embeddings  # local import to keep module light

    # Optional evaluation helpers (kept here to mirror prior script behavior).
    from ..eval.utils import evaluate_best_checkpoint
    from .embeddings_utils import build_embedding_dataloaders, load_pretrained_models

    source_experiments_strings = "_".join(source_experiments)
    source_run_name = f"{run_name}_{source_experiments_strings}"

    models, dataset_quantities, checkpoint_paths = load_pretrained_models(
        list(source_experiments),
        cfg_overrides=source_cfg_overrides,
        repeat_idx=repeat_idx,
        match_string=getattr(target_cfg, "match_string", None),
        match_num_cosmo=getattr(target_cfg, "match_num_cosmo", False),
        # Opt-in per-source run binding, so several REPEATS of one source experiment can be
        # stacked into a single wider embedding (see load_pretrained_models). Absent => unchanged.
        per_source_match_strings=getattr(target_cfg, "source_match_strings", None),
    )

    # Phase 2b: optional non-default embedding cut point, tagged onto the source encoders so
    # compute_embeddings picks it up (see embeddings_utils.compute_embeddings).
    _embedding_cut = getattr(target_cfg, "embedding_cut", None)
    if _embedding_cut:
        for _m in models:
            _m.embedding_net.embedding_cut = _embedding_cut

    # Ensure downstream code has the dataset quantities from sources.
    target_cfg.dataset_quantities = dataset_quantities
    target_cfg.test_shape_noise_idx = [0]

    do_run_training = getattr(target_cfg, "run_training", True)

    # Whitening context (Finding C3). Capture pretrain-vs-finetune BEFORE the resolution below
    # mutates pretrained_band_ckpt_path in place: a pretrain source has it None here (it then gets
    # pointed at the source-encoder ckpt, which is NOT a flow warm start); a finetune has it set to
    # the pretrain checkpoint dir (resolved to a file below).
    whiten_cfg = getattr(target_cfg, "whiten_embeddings", None)
    whiten_is_pretrain_source = target_cfg.pretrained_band_ckpt_path is None
    whiten_repeat_match = f"None_{repeat_idx}"

    # Default pretrained flow checkpoint path to the first source checkpoint.
    if target_cfg.pretrained_band_ckpt_path is None:
        target_cfg.pretrained_band_ckpt_path = checkpoint_paths[0]
    else:
        # get best checkpoint for this repeat based on match_string logic
        repeat_match = whiten_repeat_match
        best_checkpoint, _ = get_best_checkpoint(target_cfg.pretrained_band_ckpt_path, repeat_match)
        target_cfg.pretrained_band_ckpt_path = best_checkpoint[0] if best_checkpoint else None
    scalers, train_loader, val_loader, test_loader = prepare_data_parameters(target_cfg)

    train_emb_loader, val_emb_loader, test_emb_loader = build_embedding_dataloaders(
        train_loader,
        val_loader,
        test_loader,
        models,
        base_cfg=target_cfg,
        # Phase 2b: embedding_cache_name lets several head-variant targets SHARE one cache
        # (frozen-trunk features are variant-independent); reuse_embedding_cache opts into
        # reusing it during training runs. Defaults preserve prior behaviour.
        wandb_run_name=(getattr(target_cfg, "embedding_cache_name", None) or source_run_name),
        use_cache_if_exists=((not do_run_training)
                             or bool(getattr(target_cfg, "reuse_embedding_cache", False))),
        whiten_cfg=whiten_cfg,
        # Persist/resolve the whitener under the RUN's own folder even when the embedding cache is
        # shared via `embedding_cache_name` — that is where finetune/eval look for it.
        whitener_run_name=source_run_name,
        # Record WHICH source-encoder checkpoints produced z, so a cache reused under a fixed key
        # (which `run_training: False` always does) can be checked instead of trusted.
        source_checkpoints=list(checkpoint_paths or []),
        is_pretrain_source=whiten_is_pretrain_source,
        pretrained_ckpt_path_or_dir=target_cfg.pretrained_band_ckpt_path,
        repeat_match=whiten_repeat_match,
    )


    emb_dim = next(iter(train_emb_loader))[0].shape[-1]
    if do_run_training:
        fit_nde_on_embeddings(emb_dim, train_emb_loader, val_emb_loader, test_emb_loader, target_cfg, source_run_name)

    # Keeping the ad-hoc evaluation hook (commented) as in the original script.
    # If you want to enable it, ensure `target_cfg.checkpoint_path` points to the trained
    # embeddings flow checkpoint before calling.
    #
    def emb_model_builder(cfg, test_dataloader = None):
        from .embeddings_utils import build_nde_on_embeddings, load_embeddings_checkpoint
        import torch
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_nde_on_embeddings(emb_dim=emb_dim, base_cfg=cfg, test_loader=test_dataloader, device=device)
    
        checkpoint_path = getattr(cfg, "checkpoint_path", None)
        if checkpoint_path:
            load_embeddings_checkpoint(model, checkpoint_path, device,
                                       tag=str(getattr(cfg, "experiment_name", "")))
    
        model.to(device)
        model.eval()
        return model

    eval_context = {
        "test_loader": test_emb_loader,
        "param_scaler": scalers["cosmo"],
        "model_builder": emb_model_builder,
    }

    if run_evaluation:
        evaluate_best_checkpoint(
            target_cfg,
            test_emb_loader,
            scalers["cosmo"],
            reference_samples=None,
            model_builder=emb_model_builder,
            num_chains=1,
            num_samples=NUM_SAMPLES,
            warmup_steps=NUM_WARMUP,
        )

    return eval_context


def train_embeddings_experiment(
    args: EmbeddingTrainArgs,
    *,
    repeats: Optional[int] = None,
):
    """Train embeddings NDE(s) for a (target, sources) set.

    Now supports optional evaluation-time ensembles created by training multiple
    independently-seeded members per repeat (config.ensemble_repeats).

    Match string behaviour remains *bound to the repeat idx* (and n_cosmo when
    applicable): the ensemble member index only affects run naming and split_seed.

    Repeat selection:
      - if `target_cfg.repeat_indices` is set (list/tuple of ints), iterate those indices
      - else iterate `range(repeats)` where repeats is either the function arg or config.repeats
    """

    target_experiment = args.target_experiment
    source_experiments = list(args.source_experiments)

    from .embeddings_utils import get_max_trainval_cosmos_grid

    if target_experiment not in experiments:
        raise ValueError(f"Experiment '{target_experiment}' not found in config.experiments.experiments.")

    target_exp_dict = experiments[target_experiment]
    split_on_source = bool(target_exp_dict.get("split_on_source_experiments", False))

    # Grid is either target-based or source-intersection based.
    ncosmo_grid = get_max_trainval_cosmos_grid(target_experiment, source_experiments)

    for n_cosmo in ncosmo_grid:
        # Build target cfg: in source-split mode, do NOT apply max_trainval_cosmos to the target.
        target_n_cosmo = None if split_on_source else n_cosmo
        base_target_cfg = build_cfg_from_experiment_dict(target_experiment, target_exp_dict, n_cosmo=target_n_cosmo)

        # Config-driven evaluation on/off switch. Pre-training runs set `run_evaluation: False`
        # to skip the (expensive MCMC) post-training evaluation; fine-tuning runs leave it True.
        # This gates BOTH the single-run eval (in train_embedding_run) and the ensemble deferred
        # eval below, so eval can be fully disabled regardless of ensemble_repeats.
        cfg_run_evaluation = bool(getattr(base_target_cfg, "run_evaluation", True))

        n_cosmo_tag = format_ncosmo_tag(n_cosmo)

        # Overrides are only needed in source-split mode.
        source_cfg_overrides = build_source_cfg_overrides(source_experiments, n_cosmo=n_cosmo if split_on_source else None)

        # Determine repeat indices (repeat_indices overrides repeats)
        repeat_indices_cfg = getattr(base_target_cfg, "repeat_indices", None)
        if repeat_indices_cfg is not None:
            if not isinstance(repeat_indices_cfg, (list, tuple)):
                raise TypeError(
                    f"repeat_indices must be a list/tuple of ints or None, got {type(repeat_indices_cfg)}"
                )
            repeat_indices = [int(x) for x in repeat_indices_cfg]
        else:
            repeats_cfg = repeats if repeats is not None else getattr(base_target_cfg, "repeats", 1)
            n_repeats = int(repeats_cfg or 1)
            repeat_indices = list(range(n_repeats))

        if any(i < 0 for i in repeat_indices):
            raise ValueError(f"Repeat indices must be non-negative, got {repeat_indices}")

        base_seed = int(getattr(base_target_cfg, "split_seed", 42))
        ensemble_repeats = int(getattr(base_target_cfg, "ensemble_repeats", 1) or 1)

        for i in repeat_indices:
            # Base per-repeat cfg copy
            cfg_repeat = build_cfg_from_experiment_dict(target_experiment, target_exp_dict, n_cosmo=target_n_cosmo)
            cfg_repeat.split_seed = base_seed

            # Run matching (repeat-bound)
            match_string = f"{n_cosmo_tag}_{i}"
            cfg_repeat.match_string = match_string

            deferred_eval_context = None
            ensemble_member_test_loaders = []

            for j in range(ensemble_repeats):
                # Per-member cfg (seed differs; match_string stays repeat-bound)
                cfg = build_cfg_from_experiment_dict(target_experiment, target_exp_dict, n_cosmo=target_n_cosmo)
                cfg.match_string = match_string
                cfg.split_seed = base_seed
                set_seed_for_repeat_and_ensemble(cfg, repeat_idx=i, ensemble_idx=j)

                # Run naming: include ensemble suffix only if >1
                run_match = match_string if ensemble_repeats <= 1 else f"{match_string}_ens{j}"
                run_name = create_run_name(cfg, run_match)

                deferred_eval_context = train_embedding_run(
                    target_cfg=cfg,
                    source_experiments=source_experiments,
                    source_cfg_overrides=source_cfg_overrides,
                    run_name=run_name,
                    repeat_idx=i,
                    run_evaluation=(ensemble_repeats <= 1) and cfg_run_evaluation,
                )

                if deferred_eval_context is not None:
                    ensemble_member_test_loaders.append(deferred_eval_context.get("test_loader"))

            if ensemble_repeats > 1 and cfg_run_evaluation:
                if deferred_eval_context is None:
                    raise RuntimeError(
                        f"Missing deferred evaluation context for repeat {i} of experiment '{target_experiment}'."
                    )

                # Evaluate once, after all ensemble members for this repeat have
                # completed training and checkpoints are available.
                from ..eval.utils import evaluate_best_checkpoint

                evaluate_best_checkpoint(
                    cfg_repeat,
                    deferred_eval_context["test_loader"],
                    deferred_eval_context["param_scaler"],
                    reference_samples=None,
                    model_builder=deferred_eval_context["model_builder"],
                    ensemble_member_test_loaders=ensemble_member_test_loaders,
                    num_chains=1,
                    num_samples=NUM_SAMPLES,
                    warmup_steps=NUM_WARMUP,
                )

