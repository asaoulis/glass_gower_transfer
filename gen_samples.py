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

PRIOR_MODE = "LCDM_fixed_w0"  # "gower" | "kids_s8_analytic" | "LCDM_fixed_w0"

def _build_prior():
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

    prior, fixed_parameters = _build_prior()
    outpath = "data/saved_samples"

    experiment_names = [
        # 2-tuple: regular/ensemble path
        # ("hybrid_patches_16_9param", "ncosmo400_"),
        # ("finetune_hybrid_16_9param", "ncosmo150_"),
        # ("finetune_hybrid_16_9param_ensemble_stratify", "ncosmo60_0"),
        # ("finetune_hybrid_16_9param_ensemble_stratify", "ncosmo60_1"),
        # ("finetune_hybrid_16_9param_ensemble_stratify", "ncosmo60_2"),

        # ("finetune_direct_9param_nle_anaprior_longsamples", "ncosmo80_0", ["glass_hybrid_patches_16_9param"]),
        ("finetune_direct_9param_nle_anaprior_longsamples", "ncosmo120_2", ["glass_hybrid_patches_16_9param"]),
        ("finetune_direct_9param_nle_anaprior_longsamples", "ncosmo120_1", ["glass_hybrid_patches_16_9param"]),
        ("finetune_direct_9param_nle_anaprior_longsamples", "ncosmo120_0", ["glass_hybrid_patches_16_9param"]),

        # ("finetune_direct_9param_nle_anaprior_longsamples", "ncosmo200_2", ["glass_hybrid_patches_16_9param"]),
        # ("finetune_9param_nle_anaprior_ensemble_stratify", "ncosmo100_2", ["glass_hybrid_patches_16_9param"]),
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
        output_path = _build_output_path(outpath, config_name, output_suffix)

        if os.path.exists(output_path):
            samples, theta0s = torch.load(output_path)
            print(f"Loaded existing samples from {output_path}")
            samples_dict[config_name] = (samples, theta0s)
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
        theta0s, samples = model.generate_samples(
            prior=prior,
            fixed_parameters=fixed_parameters,
            num_samples=num_samples,
            num_jobs=26,
            num_chains=1,
            show_progress_bars=True,
            warmup_steps=500,
        )
        samples_dict[config_name] = (samples, theta0s)
        torch.save((samples, theta0s), output_path)
        print(f"Saved samples to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate posterior samples for configured experiments.")
    parser.add_argument(
        "--output-suffix",
        default="LCDM",
        help="Optional suffix appended to saved sample files. Use an empty string for no suffix.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    _run_generation(args.output_suffix)

if __name__ == "__main__":
    main()