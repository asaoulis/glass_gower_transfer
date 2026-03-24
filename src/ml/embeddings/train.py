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
from typing import Dict, Iterable, List, Optional, Sequence

from config.default import get_default_config
from config.experiments import experiments

from .embeddings_utils import (
    build_embedding_dataloaders,
    get_max_trainval_cosmos_grid,
    load_pretrained_models,
)

from ..models.utils import create_run_name
from ..utils import prepare_data_parameters
from ..eval.loading_model import find_best_checkpoint, get_best_checkpoint



@dataclass(frozen=True)
class EmbeddingTrainArgs:
    """High-level inputs for an embeddings run."""

    target_experiment: str
    source_experiments: Sequence[str]


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
):
    """One end-to-end run: load sources, build embedding loaders, train flow on embeddings."""

    from .embeddings_utils import fit_nde_on_embeddings  # local import to keep module light

    # Optional evaluation helpers (kept here to mirror prior script behavior).
    from ..eval.utils import evaluate_best_checkpoint

    source_experiments_strings = "_".join(source_experiments)
    source_run_name = f"{run_name}_{source_experiments_strings}"

    models, dataset_quantities, checkpoint_paths = load_pretrained_models(
        list(source_experiments), cfg_overrides=source_cfg_overrides
    )

    # Ensure downstream code has the dataset quantities from sources.
    target_cfg.dataset_quantities = dataset_quantities
    target_cfg.test_shape_noise_idx = [0]

    # Default pretrained flow checkpoint path to the first source checkpoint.
    if target_cfg.pretrained_band_ckpt_path is None:
        target_cfg.pretrained_band_ckpt_path = checkpoint_paths[0]
    else:
        # get best checkpoint for this repeat based on match_string logic
        # repeat_match = f"{n_cosmo_tag}_{i}" if n_cosmo_tag else f"_{i}"
        best_checkpoint, _ = get_best_checkpoint(target_cfg.pretrained_band_ckpt_path, "")
        target_cfg.pretrained_band_ckpt_path = best_checkpoint[0] if best_checkpoint else None

    scalers, train_loader, val_loader, test_loader = prepare_data_parameters(target_cfg)

    train_emb_loader, val_emb_loader, test_emb_loader = build_embedding_dataloaders(
        train_loader,
        val_loader,
        test_loader,
        models,
        base_cfg=target_cfg,
        wandb_run_name=source_run_name,
    )

    emb_dim = next(iter(train_emb_loader))[0].shape[-1]
    fit_nde_on_embeddings(emb_dim, train_emb_loader, val_emb_loader, test_emb_loader, target_cfg, source_run_name)

    # Keeping the ad-hoc evaluation hook (commented) as in the original script.
    # If you want to enable it, ensure `target_cfg.checkpoint_path` points to the trained
    # embeddings flow checkpoint before calling.
    #
    def emb_model_builder(cfg, loader):
        from .embeddings_utils import build_nde_on_embeddings
        import torch
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_nde_on_embeddings(emb_dim=emb_dim, base_cfg=cfg, test_loader=loader, device=device)
    
        checkpoint_path = getattr(cfg, "checkpoint_path", None)
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["state_dict"], strict=False)
    
        model.to(device)
        model.eval()
        return model
    
    evaluate_best_checkpoint(
        target_cfg,
        test_emb_loader,
        scalers["cosmo"],
        reference_samples=None,
        model_builder=emb_model_builder,
        num_chains=1,
        num_samples=1000,
        warmup_steps=200
    )


def train_embeddings_experiment(
    args: EmbeddingTrainArgs,
    *,
    repeats: Optional[int] = None,
):
    """Train embeddings NDE(s) for a (target, sources) set.

    Iteration over `max_trainval_cosmos` is controlled by the target experiment's
    `split_on_source_experiments` flag (see `get_max_trainval_cosmos_grid`).

    The per-repeat seed logic is handled here.
    """

    target_experiment = args.target_experiment
    source_experiments = list(args.source_experiments)

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

        n_cosmo_tag = format_ncosmo_tag(n_cosmo)

        # Overrides are only needed in source-split mode.
        source_cfg_overrides = build_source_cfg_overrides(source_experiments, n_cosmo=n_cosmo if split_on_source else None)

        n_repeats = int(repeats) if repeats is not None else int(getattr(base_target_cfg, "repeats", 1))
        base_seed = int(getattr(base_target_cfg, "split_seed", 42))

        for i in range(n_repeats):
            # Per-repeat cfg copy
            cfg = build_cfg_from_experiment_dict(target_experiment, target_exp_dict, n_cosmo=target_n_cosmo)
            cfg.split_seed = base_seed + i

            # Run naming
            match_string = f"{n_cosmo_tag}_{i}"
            cfg.match_string = match_string
            run_name = create_run_name(cfg, match_string)

            train_embedding_run(
                target_cfg=cfg,
                source_experiments=source_experiments,
                source_cfg_overrides=source_cfg_overrides,
                run_name=run_name,
            )

