import argparse
import os

import torch

param_names = ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"]
FIXED_PARAMS = None
FIXED_PARAMS_SPACE = "physical"

def _build_fixed_parameters_list(
    fixed_params,
    param_names,
    *,
    space: str = "physical",
):
    from src.ml.embeddings.embeddings_utils import COSMO_PARAM_PRESET_MINMAX, _build_cosmo_preset_scaler

    if fixed_params is None:
        return None
    if not isinstance(fixed_params, dict):
        raise TypeError("FIXED_PARAMS must be a dict of {param_name: value} or None")

    fixed_parameters = []

    if space not in {"scaled", "physical"}:
        raise ValueError("FIXED_PARAMS_SPACE must be 'scaled' or 'physical'")

    preset_scaler = None
    if space == "physical":
        preset_scaler = _build_cosmo_preset_scaler(COSMO_PARAM_PRESET_MINMAX, param_names)
        if preset_scaler is None or getattr(preset_scaler, "min", None) is None or getattr(preset_scaler, "max", None) is None:
            raise ValueError("Could not build preset scaler for physical->scaled conversion")

    for name, value in fixed_params.items():
        if name not in param_names:
            raise ValueError(f"Unknown fixed parameter '{name}'. Expected one of: {param_names}")
        idx = param_names.index(name)

        if space == "physical":
            span = float(preset_scaler.max[idx] - preset_scaler.min[idx])
            if span <= 0:
                raise ValueError(f"Invalid preset scaling span for '{name}'")
            value = (float(value) - float(preset_scaler.min[idx])) / span

        fixed_parameters.append((idx, float(value)))

    return fixed_parameters

# (prior mode, output suffix) pairs run BACK TO BACK against each model in one job.
# The gatekeeper's `sample-submit` passes no arguments, so anything configurable has to live in
# the file and be `sync`ed; running the priors as a list (rather than one PRIOR_MODE per job)
# means the expensive part -- building a 9-member NLE ensemble and its cached test loader -- is
# paid ONCE for all three sample sets. Suffixes match what the plotting notebooks load:
#   longsamples = Gower empirical prior (apples-to-apples with the published NPE chains)
#   FINAL       = KiDS-Legacy analytic S8 prior, wCDM
#   LCDM        = same analytic prior with w0 fixed to -1
PRIOR_RUNS = [
    ("gower", "longsamples"),
    ("kids_s8_analytic", "FINAL"),
    ("LCDM_fixed_w0", "LCDM"),
]

def _build_prior(PRIOR_MODE):
    from src.ml.embeddings.embeddings_utils import COSMO_PARAM_PRESET_MINMAX, _build_cosmo_preset_scaler
    from src.ml.eval.utils import build_gower_prior, build_s8_analytic_prior

    if PRIOR_MODE == "gower":
        prior = build_gower_prior(param_names)
        fixed_parameters = None
    elif PRIOR_MODE == "kids_s8_analytic":
        print("Using flat analytic prior on S8")
        scaler = _build_cosmo_preset_scaler(COSMO_PARAM_PRESET_MINMAX, param_names)
        prior = build_s8_analytic_prior(param_names, scaler, return_restricted=False)
        fixed_parameters = None
        print("prior", prior, type(prior))
    elif PRIOR_MODE == "LCDM_fixed_w0":
        print("Using LCDM analytic prior on S8")
        scaler = _build_cosmo_preset_scaler(COSMO_PARAM_PRESET_MINMAX, param_names)
        prior = build_s8_analytic_prior(param_names, scaler, return_restricted=False)
        fixed_parameters = _build_fixed_parameters_list(
            {"w0": -1.0},
            param_names,
            space=FIXED_PARAMS_SPACE,
        )
    else:
        raise ValueError(f"Unknown PRIOR_MODE: {PRIOR_MODE!r}")

    return prior, fixed_parameters

def _build_output_path(outpath, config_name, output_suffix):
    suffix = f"_{output_suffix}" if output_suffix else ""
    return os.path.join(outpath, f"{config_name}{suffix}.tch")

def _run_generation(output_suffix: str):
    from config.default import get_default_config
    from config.experiments import experiments
    from src.ml.embeddings.train import load_embedding_model_with_dataloader
    from src.ml.eval.utils import load_best_model_and_build_posterior
    from src.ml.utils import build_ensemble_model_from_checkpoints, is_ensemble_eval_active, prepare_data_parameters

    # Under MODELS_ROOT (not the repo) so the gatekeeper can `send` these: `view`/`send` are
    # confined to MODELS_ROOT, and the old repo-local dumps were unreachable from outside.
    outpath = "/share/gpu5/asaoulis/transfer_models/checkpoints/saved_samples"
    os.makedirs(outpath, exist_ok=True)

    prior_runs = list(PRIOR_RUNS)
    if output_suffix is not None:
        if len(prior_runs) != 1:
            raise ValueError(
                "--output-suffix only makes sense when PRIOR_RUNS has exactly one entry; "
                f"it has {len(prior_runs)}. Edit PRIOR_RUNS instead."
            )
        prior_runs = [(prior_runs[0][0], output_suffix)]

    experiment_names = [
        # 2-tuple: regular/ensemble path
        # ("hybrid_patches_16_9param", "ncosmo400_"),
        # ("finetune_hybrid_16_9param", "ncosmo150_"),
        # ("finetune_hybrid_16_9param_ensemble_stratify", "ncosmo60_0"),
        # ("finetune_hybrid_16_9param_ensemble_stratify", "ncosmo60_1"),
        # ("finetune_hybrid_16_9param_ensemble_stratify", "ncosmo60_2"),

        # ("finetune_direct_9param_nle_anaprior_longsamples", "ncosmo80_0", ["glass_hybrid_patches_16_9param"]),
        # ("finetune_direct_9param_nle_anaprior_longsamples", "ncosmo120_2", ["glass_hybrid_patches_16_9param"]),
        # ("finetune_direct_9param_nle_anaprior_longsamples", "ncosmo120_1", ["glass_hybrid_patches_16_9param"]),
        # ("finetune_direct_9param_nle_anaprior_longsamples", "ncosmo120_0", ["glass_hybrid_patches_16_9param"]),

        # ("finetune_direct_9param_nle_anaprior_longsamples", "ncosmo200_2", ["glass_hybrid_patches_16_9param"]),
        # ("finetune_9param_nle_anaprior_ensemble_stratify", "ncosmo100_2", ["glass_hybrid_patches_16_9param"]),
        # ("finetune_9param_nle_anaprior_ensemble_stratify", "ncosmo100_0", ["glass_hybrid_patches_16_9param"]),
        # Whitened (k=8) transfer NLE ensemble, N=60, repeat 0 -- r0 is the best-calibrated of the
        # three repeats at N=60 (full 9-param TARP 0.01003 vs 0.01195 / 0.01012) and also the best
        # FoM (75.87). Runs cache-only off the published chain's embeddings; raw gower_mocks is gone.
        ("finetune_9param_nle_ensemble_white8_v2", "ncosmo60_0", ["glass_hybrid_patches_16_9param"]),
        # ("finetune_9param_nle_anaprior_ensemble_stratify", "ncosmo100_1", ["glass_hybrid_patches_16_9param"]),

        # ("finetune_9param_nle_anaprior_ensemble_stratify", "ncosmo80_2", ["glass_hybrid_patches_16_9param"]),
        # ("finetune_9param_nle_anaprior_ensemble_stratify", "ncosmo60_1", ["glass_hybrid_patches_16_9param"]),
        # ("direct_9param_nle_anaprior_large", "ncosmo400_0", ["hybrid_patches_16_9param"]),
        # 3-tuple: embeddings path (third element = source_experiments)
        # ("finetune_direct_embeddings_9param_nle", "ncosmo200_0", ["hybrid_patches_16_9param"]),
    ]

    model_configs = {}
    samples_dict = {}

    for experiment_entry in experiment_names:
        if len(experiment_entry) == 3:
            experiment_name, match_string, source_experiments = experiment_entry
        else:
            experiment_name, match_string = experiment_entry
            source_experiments = None

        config_name = f"{experiment_name}_{match_string}"

        pending = []
        for prior_mode, suffix in prior_runs:
            output_path = _build_output_path(outpath, config_name, suffix)
            if os.path.exists(output_path):
                print(f"Already on disk, skipping: {output_path}")
                continue
            pending.append((prior_mode, suffix, output_path))

        if not pending:
            print(f"All prior runs already present for {config_name}; nothing to do.")
            continue

        if len(experiment_entry) == 3:
            art = load_embedding_model_with_dataloader(
                experiment_name=experiment_name,
                match_string=match_string,
                source_experiments=source_experiments,
                config_overrides={
                    "test_shape_noise_idx": [0, 0],
                    "N_extra_test_cosmologies": 130,
                },
            )

            model = art.model
            scalers = art.scalers
            test_loader = art.test_loader
            config = art.config
        else:
            experiment_config = experiments[experiment_name]

            ncosmo_in_match = None
            for part in match_string.split("_"):
                if part.startswith("ncosmo"):
                    try:
                        ncosmo_in_match = int(part.replace("ncosmo", ""))
                    except ValueError:
                        pass

            print(
                f"Experiment '{experiment_name}' has match_string='{match_string}' "
                f"and ncosmo_in_match={ncosmo_in_match}"
            )

            config = get_default_config()
            config.experiment_name = experiment_name
            config.test_shape_noise_idx = [0, 0]
            for key, val in experiment_config.items():
                if key == "max_trainval_cosmos":
                    continue
                setattr(config, key, val)

            config.match_string = match_string
            config.N_extra_test_cosmologies = 130
            config.max_trainval_cosmos = ncosmo_in_match

            if is_ensemble_eval_active(config):
                model = build_ensemble_model_from_checkpoints(
                    config,
                    test_loader=None,
                    match_string=match_string,
                )
                if model is None:
                    raise RuntimeError(
                        f"Failed to build ensemble model for {experiment_name} match_string='{match_string}'"
                    )
                scalers, _, _, test_loader = prepare_data_parameters(config)
            else:
                model, _, _ = load_best_model_and_build_posterior(
                    config, ds_string_match=match_string
                )
                scalers, _, _, test_loader = prepare_data_parameters(config)

        model_configs[config_name] = {
            "model": model,
            "scalers": scalers,
            "test_loader": test_loader,
            "inputs": config.dataset_quantities,
        }

        num_samples = 10000
        model.test_dataloader = test_loader

        for prior_mode, suffix, output_path in pending:
            print(f"\n=== {config_name}: prior={prior_mode} -> {os.path.basename(output_path)} ===")
            prior, fixed_parameters = _build_prior(prior_mode)
            theta0s, samples = model.generate_samples(
                prior=prior,
                fixed_parameters=fixed_parameters,
                num_samples=num_samples,
                # One loky worker per requested core. The test set is 35 batches, so 36
                # workers finish it in ONE wave; 26 would need two, doubling the wall
                # clock for no gain (same lesson as the B2 ensemble evals).
                num_jobs=36,
                num_chains=1,
                show_progress_bars=True,
                warmup_steps=500,
            )
            samples_dict[f"{config_name}_{suffix}"] = (samples, theta0s)
            torch.save((samples, theta0s), output_path)
            print(f"Saved samples to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate posterior samples for configured experiments.")
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Legacy override of the output suffix. Only valid when PRIOR_RUNS has exactly one "
             "entry; otherwise edit PRIOR_RUNS (which carries one suffix per prior).",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    _run_generation(args.output_suffix)

if __name__ == "__main__":
    main()