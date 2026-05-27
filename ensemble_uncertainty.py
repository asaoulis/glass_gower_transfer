import argparse
from copy import copy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from config.ablations import ablation_experiments
from config.default import get_default_config
from config.experiments import experiments
from src.ml.eval import diag_gaussian_symmetric_kl, median_heuristic_sigma2, mmd_disagreement_score
from src.ml.eval.utils import build_gower_prior, load_best_model_and_build_posterior
from src.ml.models.sampling import align_indices_by_path, get_dataset_paths
from src.ml.models.utils import apply_repeat_config
from src.ml.utils import prepare_data_parameters


def _as_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _collect_member_outputs(model, *, num_samples: int, batch_size: int, prior, warmup_steps: int, num_chains: int):
    """Collect theta0s, representation, z and samples for a single ensemble member."""
    # Delegate to the model method so sampling/passing uses the posterior path.
    return model.get_representations_and_samples(
        num_samples=num_samples,
        batch_size=batch_size,
        prior=prior,
        warmup_steps=warmup_steps,
        num_chains=num_chains,
    )


def load_config(experiment_name: str):
    """Load config in a way consistent with train.py (mirrors eval.py)."""
    experiment_config = experiments[experiment_name]

    config = get_default_config()
    config.experiment_name = experiment_name

    for key, val in experiment_config.items():
        if key == "max_trainval_cosmos":
            continue
        setattr(config, key, val)

    return config, experiment_config


def parse_args():
    parser = argparse.ArgumentParser(description="Compute deep-ensemble uncertainty/OOD scores.")
    parser.add_argument("--experiment", required=True, type=str, help="Experiment name key from config.experiments")
    parser.add_argument("--output", default="data/ensemble_uncertainty", type=Path, help="Output directory")
    parser.add_argument(
        "--data-patterns",
        default=None,
        type=str,
        help="Optional override for config.data_patterns (glob) to point at a different dataset.",
    )
    parser.add_argument("--num-samples", default=5000, type=int, help="Posterior samples per test example per ensemble member")
    parser.add_argument("--batch-size", default=8, type=int, help="Batch size used for posterior sampling given z")
    parser.add_argument("--seed", default=0, type=int, help="Random seed for kernel bandwidth subsampling")
    parser.add_argument("--save-raw", action="store_true", help="Also save raw intermediate arrays (can be large)")
    parser.add_argument("--warmup-steps", default=500, type=int, help="MCMC warmup (only for NLE/joint models that use MCMC)")
    parser.add_argument("--num-chains", default=1, type=int, help="MCMC chains (only for NLE/joint models that use MCMC)")
    return parser.parse_args()


def main():
    # Combine experiments and ablations into a single dict (mirrors eval.py)
    experiments.update(ablation_experiments)

    args = parse_args()
    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    config, experiment_config = load_config(args.experiment)

    max_tv = experiment_config.get("max_trainval_cosmos", None)
    repeats = int(getattr(config, "repeats", 1) or 1)

    # We only support the non-ensemble training case (one run per repeat).
    if int(getattr(config, "ensemble_repeats", 1) or 1) > 1:
        raise ValueError(
            "This script expects models NOT trained with evaluation-time ensembles (ensemble_repeats must be 1)."
        )

    if isinstance(max_tv, (list, tuple)):
        configs_to_run = []
        for n_cosmo in max_tv:
            cfg = get_default_config()
            cfg.experiment_name = config.experiment_name
            cfg.test_shape_noise_idx = [0]
            for key, val in experiment_config.items():
                if key == "max_trainval_cosmos":
                    continue
                setattr(cfg, key, val)
            cfg.max_trainval_cosmos = int(n_cosmo)
            cfg.match_num_cosmo = True
            configs_to_run.append((f"ncosmo{int(n_cosmo)}", cfg))
    else:
        cfg = config
        cfg.test_shape_noise_idx = [0]
        if max_tv is not None:
            cfg.max_trainval_cosmos = int(max_tv)
        configs_to_run = [("default", cfg)]

    for tag, cfg_base in configs_to_run:
        # ----------------------------
        # 1) Load one best model per repeat
        # ----------------------------
        model_names: List[str] = []
        models = []
        test_loaders = {}
        scalers = {}
        checkpoint_paths = {}

        for r in range(repeats):
            cfg_r = copy(cfg_base)
            repeat_match, _ = apply_repeat_config(cfg_r, r)
            cfg_r.match_string = repeat_match

            if args.data_patterns is not None:
                cfg_r.data_patterns = str(args.data_patterns)

            rep_scalers, _, _, test_loader = prepare_data_parameters(cfg_r)

            model, _, ckpt_path = load_best_model_and_build_posterior(
                cfg_r,
                ds_string_match=repeat_match,
                data_parameters=test_loader,
            )
            if model is None:
                raise RuntimeError(f"Failed to load best model for repeat {r} match='{repeat_match}'")

            # Ensure model uses the same loader used for encoding/sampling.
            model.test_dataloader = test_loader

            name = f"repeat{r}"
            model_names.append(name)
            models.append(model)
            test_loaders[name] = test_loader
            scalers[name] = rep_scalers
            checkpoint_paths[name] = ckpt_path

        # Align by underlying file paths if available.
        # (If not, we assume identical ordering across repeats.)
        base_name = model_names[0]
        base_paths = get_dataset_paths(test_loaders[base_name])
        if base_paths is None:
            base_paths = [str(i) for i in range(len(test_loaders[base_name].dataset))]

        # ----------------------------
        # 2) Collect intermediate outputs per repeat
        # ----------------------------
        theta0_ref = None
        reps_by_model: Dict[str, torch.Tensor] = {}
        z_by_model: Dict[str, torch.Tensor] = {}
        theta_by_model: Dict[str, torch.Tensor] = {}
        samples_by_model: Dict[str, torch.Tensor] = {}

        prior = build_gower_prior(cfg_base.cosmo_param_names)

        for name, model in zip(model_names, models):
            # The LightningModule method iterates `model.test_dataloader`.
            # Attach the loader we prepared for this repeat.
            model.test_dataloader = test_loaders[name]
            out = _collect_member_outputs(
                model,
                num_samples=int(args.num_samples),
                batch_size=int(args.batch_size),
                prior=prior,
                warmup_steps=int(args.warmup_steps),
                num_chains=int(args.num_chains),
            )

            theta0s = out.get("theta0s")
            reps_t = out.get("representation")
            z_t = out.get("z")
            samples_t = out.get("samples")

            if theta0s is None or reps_t is None or z_t is None or samples_t is None:
                raise RuntimeError("Model did not return required keys from get_representations_and_samples")

            reps_by_model[name] = reps_t
            z_by_model[name] = z_t
            theta_by_model[name] = theta0s
            samples_by_model[name] = samples_t

            if theta0_ref is None:
                theta0_ref = theta0s

        # If datasets expose paths, reorder all repeats onto the base ordering.
        if get_dataset_paths(test_loaders[base_name]) is not None:
            all_indices = list(range(len(base_paths)))
            mapping = align_indices_by_path(test_loaders, all_indices, base_name=base_name)
            for name in model_names:
                idx = torch.as_tensor(mapping[name], dtype=torch.long)
                reps_by_model[name] = reps_by_model[name].index_select(0, idx)
                z_by_model[name] = z_by_model[name].index_select(0, idx)
                theta_by_model[name] = theta_by_model[name].index_select(0, idx)
            theta0_ref = theta_by_model[base_name]

        # stack to [K, N, D]
        reps_stack = torch.stack([reps_by_model[n] for n in model_names], dim=0).cpu().numpy()
        z_stack = torch.stack([z_by_model[n] for n in model_names], dim=0).cpu().numpy()

        # per-example kernels work on [N, K, D]
        reps_nkd = np.transpose(reps_stack, (1, 0, 2))
        z_nkd = np.transpose(z_stack, (1, 0, 2))

        # bandwidth selection: use all member outputs pooled as samples
        reps_pooled = reps_stack.reshape(-1, reps_stack.shape[-1])
        z_pooled = z_stack.reshape(-1, z_stack.shape[-1])
        sigma2_rep = median_heuristic_sigma2(reps_pooled, seed=args.seed)
        sigma2_z = median_heuristic_sigma2(z_pooled, seed=args.seed + 1)

        rep_mmd = mmd_disagreement_score(reps_nkd, sigma2=sigma2_rep)
        z_mmd = mmd_disagreement_score(z_nkd, sigma2=sigma2_z)

        # ----------------------------
        # 3) Compute posterior KL disagreement from samples
        # ----------------------------
        # compute diag-Gaussian KL on scaled samples
        # -> per model: mu/var over samples dimension
        mu = []
        var = []
        for name in model_names:
            s = samples_by_model[name].cpu().numpy()  # [S,N,D]
            mu.append(s.mean(axis=0))
            var.append(s.var(axis=0, ddof=0))
        mu = np.stack(mu, axis=0)   # [K,N,D]
        var = np.stack(var, axis=0) # [K,N,D]

        kl_score = diag_gaussian_symmetric_kl(mu, var)

        # ----------------------------
        # 4) Save outputs
        # ----------------------------
        out_path = out_dir / f"ensemble_uncertainty_{cfg_base.experiment_name}_{tag}.npz"
        payload = {
            "experiment": cfg_base.experiment_name,
            "tag": tag,
            "model_names": np.array(model_names, dtype=object),
            "checkpoint_paths": np.array([checkpoint_paths[n] for n in model_names], dtype=object),
            "paths": np.array(base_paths, dtype=object),
            "theta0": theta0_ref.detach().cpu().numpy() if theta0_ref is not None else None,
            "rep_mmd": rep_mmd,
            "z_mmd": z_mmd,
            "kl_score": kl_score,
            "sigma2_rep": float(sigma2_rep),
            "sigma2_z": float(sigma2_z),
            "num_samples": int(args.num_samples),
        }

        if args.save_raw:
            payload["reprs"] = reps_stack
            payload["z"] = z_stack
            # samples are huge; store as float16 to reduce disk.
            payload["samples"] = np.stack([samples_by_model[n].cpu().numpy().astype(np.float16) for n in model_names], axis=0)

        np.savez_compressed(out_path, **payload)
        print(f"Saved ensemble uncertainty results to {out_path}")


if __name__ == "__main__":
    main()
