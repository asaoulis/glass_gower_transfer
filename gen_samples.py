import argparse
import os

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

PRIOR_MODE = "kids_s8_analytic"  # "gower" | "kids_s8_analytic" | "LCDM_fixed_w0"

def _build_prior():
    from src.ml.embeddings.embeddings_utils import COSMO_PARAM_PRESET_MINMAX, _build_cosmo_preset_scaler
    from src.ml.eval.utils import build_gower_prior, build_s8_analytic_prior

    if PRIOR_MODE == "gower":
        prior = build_gower_prior(param_names)
        fixed_parameters = None
    elif PRIOR_MODE == "kids_s8_analytic":
        print("Using flat analytic prior on S8")
        scaler = _build_cosmo_preset_scaler(COSMO_PARAM_PRESET_MINMAX, param_names)
        # return_restricted defaults to True when (a_ia, b_ia) are present: the NLA-M IA block
        # is an unbounded Gaussian in scaled space, and NLE MCMC must stay inside the [0,1]^D
        # box the flow was trained on.
        prior = build_s8_analytic_prior(param_names, scaler)
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

def _build_output_path(base_path, experiment_name, match_string, output_suffix,
                       prior_tag=None):
    """Sample dump path: under checkpoints/<exp>/ on MODELS_ROOT so `fetch --exp` pulls it.
    Encodes the prior in the name so different-prior runs coexist (eval npz schema).

    prior_tag overrides PRIOR_MODE in the filename. It exists because NOT every model class
    honours `prior`: NLE samples by MCMC against the explicit prior, but
    NPELightningModule.generate_samples builds its posterior with no prior and draws straight from
    q(theta|x) — i.e. NPE dumps are under the SIMULATION prior whatever PRIOR_MODE says. Tagging
    those `trainprior` keeps the filename from asserting a prior that was never applied."""
    suffix = f"_{output_suffix or SAMPLE_SUFFIX}" if (output_suffix or SAMPLE_SUFFIX) else ""
    tag = prior_tag or PRIOR_MODE
    return os.path.join(
        base_path, "checkpoints", experiment_name,
        f"samples_{tag}_{match_string}{suffix}.npz",
    )

# Eval-time loader overrides for the 3-tuple (embeddings) path:
#  - test_shape_noise_idx [0,[0,1]] -> rot0, trailing noise in {0,1}: with the gower store's
#    out{0,1} x rot{0..4} x _{0..3} layout this keeps out{0,1}_rot0_{0,1} = 4 noise
#    variants/cosmology (2 outer x 2 inner, same footprint rotation)
#  - N_test_cosmologies 40      -> trim the 200 fixed-test cosmologies to the first 40 by sim_id
#                                  (train/val + scalers stay identical to training) => N = 160
#  - emb_test_batch_size 6      -> ceil(160/6) = 27 MCMC joblib jobs (one per batch): a single
#    wave on 30 CPUs with smaller per-job batches (wall ~ batch size for vectorised slice
#    sampling), ~25% faster than batch 8 / 20 jobs
#
# 2026-08-04 (M4b-EARLY preview): trimmed to 15 cosmologies x 4 noise = 60 inference points and
# batch 2 => 30 joblib jobs = EXACTLY one wave on the sample job's 30 CPUs. Wall is set by the
# BATCH size (each job samples its batch serially), not the point count, so 60 points at batch 2
# costs ~1/3 the wall of 160 points at batch 6 while still giving 60 representative posteriors.
CONFIG_OVERRIDES = {
    "test_shape_noise_idx": [0, [0, 1]],
    "N_test_cosmologies": 7,
    "emb_test_batch_size": 2,
}
NUM_JOBS = 30

# ⏱️ FAST-PREVIEW SIZING (2026-08-05). The wall of an NLE sampling job is set by the MCMC CHAIN
# LENGTH, not the number of test points: each joblib task runs ~(num_samples + warmup) * num_chains
# steps at ~4.4 it/s, so 20 000 samples = 41 100 steps ≈ 2 h 36 m PER TASK, and tasks beyond
# NUM_JOBS queue into a second wave. Measured live on job 1341694.
#   20k samples, 15 cosmologies (120 pts / 2 = 60 tasks over 30 workers = 2 waves) ⇒ ~5 h 15 m
#    4k samples,  7 cosmologies ( 56 pts / 2 = 28 tasks over 30 workers = 1 wave)  ⇒ ~35 min
# 4 000 samples is ample for corner plots. Dumps carry the SAMPLE_SUFFIX below so a fast run and a
# full run coexist instead of one clobbering the other.
NUM_SAMPLES = 4000
SAMPLE_SUFFIX = "n4k"

def _run_generation(output_suffix: str):
    from config.default import get_default_config
    from config.experiments import experiments
    from config.ablations import ablation_experiments
    from config.kids_legacy import kids_legacy_experiments
    from config.kids_legacy_novd import kids_legacy_novd_experiments
    from src.ml.embeddings.train import load_embedding_model_with_dataloader
    from src.ml.eval.utils import load_best_model_and_build_posterior
    from src.ml.utils import build_ensemble_model_from_checkpoints, is_ensemble_eval_active, prepare_data_parameters

    # Same merge as train_embeddings.py / eval.py: the registry modules share one dict object,
    # so updating here also makes the KiDS-Legacy names resolvable inside
    # load_embedding_model_with_dataloader (which imports the same `experiments`).
    experiments.update(ablation_experiments)
    experiments.update(kids_legacy_experiments)
    experiments.update(kids_legacy_novd_experiments)  # NO-VD production suite configs

    prior, fixed_parameters = _build_prior()

    experiment_names = [
        # 3-tuple: embeddings path (third element = source_experiments).
        # M4b-EARLY preview: the 5-member no-VD NLE ensemble on the r4 foundation. The match
        # string is `ncosmoNone_4` because that config leaves max_trainval_cosmos at the default
        # None (run_string = f"ncosmo{max_trainval_cosmos}_{repeat}"), unlike the ncosmo300_*
        # production runs below.
        ("gower_nle_finetune_nla_m_novd_z8_r4_ens5_early", "ncosmoNone_4",
         ["kids_legacy_hybrid_nla_m_novd_z8_resnet"]),
        # M3-EARLY preview: the 5-member NPE ensemble on the SAME store + foundation. 2-tuple =
        # direct (non-embeddings) path. NPE ignores `prior` (see _build_output_path), so this dump
        # is under the SIMULATION prior and is tagged `trainprior`, not `kids_s8_analytic`.
        #
        # ⚠️ GATED OFF UNTIL ITS TRAINING FINISHES — re-enable then. This is NOT cosmetic:
        # build_ensemble_model_from_checkpoints (src/ml/utils.py) `continue`s past a member whose
        # checkpoint is missing and only returns None when ZERO members load. So sampling this
        # entry mid-training would silently build a 1- or 2-member "ensemble", write it under the
        # 5-member filename, and then the os.path.exists skip above would stop the correct
        # 5-member dump from ever being generated. Uncomment once all five
        # finetune_ncosmoNone_4_ens{0..4} checkpoints exist.
        # ("gower_npe_finetune_nla_m_novd_z8_r4_ens5_early", "ncosmoNone_4"),
        # Previous (VD-era) production main-variate NLE ensembles (z8, 9 members), repeats 0..4 —
        # already sampled 2026-07-08; kept for reference, skipped automatically when the npz exists.
        # *[(f"gower_nle_finetune_nla_m_z8_r{r}_ens9", f"ncosmo300_{r}",
        #    ["kids_legacy_hybrid_nla_m_lmin50_fwhm4_z8"]) for r in range(5)],
    ]
    # Model classes that do NOT honour an explicit prior -> tag their dumps `trainprior`.
    NO_PRIOR_SUPPORT = {"gower_npe_finetune_nla_m_novd_z8_r4_ens5_early"}

    from src.ml.eval.utils import _resolve_test_paths, _save_posterior_samples

    base_path = get_default_config().base_path

    for experiment_entry in experiment_names:
        if len(experiment_entry) == 3:
            experiment_name, match_string, source_experiments = experiment_entry
        else:
            experiment_name, match_string = experiment_entry
            source_experiments = None

        config_name = f"{experiment_name}_{match_string}"
        output_path = _build_output_path(
            base_path, experiment_name, match_string, output_suffix,
            prior_tag="trainprior" if experiment_name in NO_PRIOR_SUPPORT else None,
        )

        if os.path.exists(output_path):
            print(f"Samples already exist at {output_path}; skipping {config_name}")
            continue

        if len(experiment_entry) == 3:
            art = load_embedding_model_with_dataloader(
                experiment_name=experiment_name,
                match_string=match_string,
                source_experiments=source_experiments,
                config_overrides=dict(CONFIG_OVERRIDES),
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
            for key, val in experiment_config.items():
                if key == "max_trainval_cosmos":
                    continue
                setattr(config, key, val)

            config.match_string = match_string
            config.max_trainval_cosmos = ncosmo_in_match
            for key, val in CONFIG_OVERRIDES.items():
                setattr(config, key, val)

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

        num_samples = NUM_SAMPLES
        model.test_dataloader = test_loader
        theta0s, samples = model.generate_samples(
            prior=prior,
            fixed_parameters=fixed_parameters,
            num_samples=num_samples,
            num_jobs=NUM_JOBS,
            num_chains=1,
            show_progress_bars=True,
            warmup_steps=500,
        )
        # Eval npz schema (samples/theta0s/test_files/sim_ids/aug_ids) so downstream
        # cross-model comparisons can match points by test-file basename.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        _save_posterior_samples(output_path, theta0s, samples, _resolve_test_paths(test_loader))

def parse_args():
    parser = argparse.ArgumentParser(description="Generate posterior samples for configured experiments.")
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix appended to saved sample files. Use an empty string for no suffix.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    _run_generation(args.output_suffix)

if __name__ == "__main__":
    main()