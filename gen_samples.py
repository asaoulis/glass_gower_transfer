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
    # The analytic-prior dumps feed the wCDM-vs-LCDM figure, not the coverage figure this run is
    # for; re-enable them when that figure needs the new chains. The on-disk skip makes a later
    # re-submit resume rather than redo.
    # ("kids_s8_analytic", "FINAL"),
    # ("LCDM_fixed_w0", "LCDM"),
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

# Identical `sample` jobs self-partition across nodes: each claims a different prior instead of
# one job walking all three back to back. Three priors x ~1 wave of 35 batches is ~3x faster on
# three nodes than on one, and the priors are completely independent (same checkpoints, different
# prior object), so there is nothing to serialise.
#
# The claim is an atomically-created directory. A job starts at an offset derived from its SLURM
# job id, so concurrently-submitted jobs pick different priors immediately, then walks the rest of
# the list -- so ONE job still completes all three, just sequentially. Restart-safe: a prior whose
# .tch already exists is skipped outright, and a claim older than CLAIM_STALE_HOURS with no output
# is treated as abandoned (its job died) and retaken.
CLAIM_STALE_HOURS = 12.0


def _claim_dir(outpath, config_name, suffix):
    return os.path.join(outpath, "_claims", f"{config_name}_{suffix}")


def _claim_job_is_alive(claim_dir) -> bool:
    """Is the SLURM job that wrote this claim still in the queue?

    Age alone is a poor staleness test: a cancelled job holds its cell for CLAIM_STALE_HOURS even
    though it will never finish it. Asking squeue makes an abandoned claim reclaimable at once.
    Unknown/unreadable => treat as alive and let the age rule decide.

    The subtlety that matters: `squeue -j <id>` exits NON-ZERO for a job SLURM has purged, with
    "Invalid job id specified" on stderr. That is precisely the "this job is gone" case, so it
    must not be lumped in with "squeue is broken" -- doing so makes a cancelled job's claim
    immortal, which is exactly what stranded the ensemble cell on 2026-08-19 (job 1345263 was
    cancelled, its claim survived, and the sibling that should have retaken it exited with
    nothing to do). Any OTHER non-zero exit still means "do not steal on a guess".
    """
    import subprocess

    try:
        with open(os.path.join(claim_dir, "jobid")) as f:
            jid = f.read().strip()
        if not jid.isdigit():
            return True
        out = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%T"],
                             capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            err = (out.stderr or "").lower()
            if "invalid job id" in err or "invalid job" in err:
                return False                 # SLURM has purged it: definitely not running
            return True                      # squeue itself failed -- do not steal on a guess
        return bool(out.stdout.strip())
    except Exception:
        return True


def _try_claim(outpath, config_name, suffix):
    """Atomically claim (config, prior). True if this process owns the work."""
    import time

    d = _claim_dir(outpath, config_name, suffix)
    os.makedirs(os.path.dirname(d), exist_ok=True)
    try:
        os.mkdir(d)
    except FileExistsError:
        age_h = (time.time() - os.path.getmtime(d)) / 3600.0
        alive = _claim_job_is_alive(d)
        if alive and age_h < CLAIM_STALE_HOURS:
            print(f"  [claim] {suffix}: held by a live job ({age_h:.1f} h old), skipping")
            return False
        why = "job no longer in the queue" if not alive else f"stale ({age_h:.1f} h, no output)"
        print(f"  [claim] {suffix}: {why} -- retaking")
        os.utime(d, None)
    with open(os.path.join(d, "jobid"), "w") as f:
        f.write(str(os.environ.get("SLURM_JOB_ID", "local")) + "\n")
    return True


def _rotate_for_this_job(items):
    """`items` rotated by SLURM job id, so sibling jobs start on different work.

    Identical `sample` submissions self-partition: with the claim directory below, N jobs cover N
    different (model, prior) cells concurrently instead of one job walking them back to back.
    """
    items = list(items)
    if not items:
        return items
    try:
        offset = int(os.environ.get("SLURM_JOB_ID", "0")) % len(items)
    except ValueError:
        offset = 0
    return items[offset:] + items[:offset]


def _prior_runs_for_this_job():
    """PRIOR_RUNS rotated by SLURM job id, so sibling jobs start on different priors."""
    return _rotate_for_this_job(PRIOR_RUNS)


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

    prior_runs = _prior_runs_for_this_job()
    print(f"[claim] this job will try priors in order: {[m for m, _ in prior_runs]}")
    if output_suffix is not None:
        if len(prior_runs) != 1:
            raise ValueError(
                "--output-suffix only makes sense when PRIOR_RUNS has exactly one entry; "
                f"it has {len(prior_runs)}. Edit PRIOR_RUNS instead."
            )
        prior_runs = [(prior_runs[0][0], output_suffix)]

    # Entries are (experiment, match_string[, source_experiments[, opts]]).
    # `opts` (4th element, embeddings path only): {"overrides": {...extra config...},
    # "suffix": "..."} -- the suffix REPLACES the prior's, which is what lets the same
    # (experiment, match) be dumped twice against two different test sets without colliding.
    HOLDOUT190 = {"path": "config/fixed_test_sets/nle_coverage_holdout190.json",
                  "align_to_native": True}
    experiment_names = [
        # --- the NLE coverage figure: 190 held-out cosmologies x 4 augmentations ---------------
        # Both whitened chains are cache-only, so their own emb_test.pt is a 59-cosmology set.
        # `holdout_test_spec` swaps in the paper's 190-cosmology test set, assembled from the
        # N=530 cache and affine-aligned into each model's own encoder frame
        # (src/ml/embeddings/holdout_testset.py). Same 760 rows for both models, so the two
        # coverage curves are computed on identical observations.
        ("finetune_9param_nle_ensemble_white8_v2", "ncosmo60_0", ["glass_hybrid_patches_16_9param"],
         {"overrides": {"holdout_test_spec": HOLDOUT190}, "suffix": "holdout190"}),
        ("finetune_9param_nle_white8_v2", "ncosmo100_0", ["glass_hybrid_patches_16_9param"],
         {"overrides": {"holdout_test_spec": HOLDOUT190}, "suffix": "holdout190"}),
        # Cross-check: the single chain on its OWN cached test set (59 cosmologies, no swap, no
        # frame map). Its ensemble twin already has this dump, so the pair lets the figure show
        # that the swap did not move the coverage curve. Cheap -- one flow, not nine.
        ("finetune_9param_nle_white8_v2", "ncosmo100_0", ["glass_hybrid_patches_16_9param"],
         {"suffix": "longsamples"}),
    ]

    model_configs = {}
    samples_dict = {}

    for experiment_entry in _rotate_for_this_job(experiment_names):
        opts = {}
        if len(experiment_entry) == 4:
            experiment_name, match_string, source_experiments, opts = experiment_entry
        elif len(experiment_entry) == 3:
            experiment_name, match_string, source_experiments = experiment_entry
        else:
            experiment_name, match_string = experiment_entry
            source_experiments = None
        is_embeddings_entry = source_experiments is not None

        config_name = f"{experiment_name}_{match_string}"

        pending = []
        for prior_mode, prior_suffix in prior_runs:
            suffix = opts.get("suffix", prior_suffix)
            output_path = _build_output_path(outpath, config_name, suffix)
            if os.path.exists(output_path):
                print(f"Already on disk, skipping: {output_path}")
                continue
            if not _try_claim(outpath, config_name, suffix):
                continue
            pending.append((prior_mode, suffix, output_path))

        if not pending:
            print(f"All prior runs already present for {config_name}; nothing to do.")
            continue

        if is_embeddings_entry:
            # NOTE: these two overrides are INERT under `embeddings_cache_only` -- the cached
            # tensors are the dataset. Reaching a different test set is what `holdout_test_spec`
            # (passed via opts["overrides"]) is for.
            overrides = {"test_shape_noise_idx": [0, 0], "N_extra_test_cosmologies": 130}
            overrides.update(opts.get("overrides", {}))
            art = load_embedding_model_with_dataloader(
                experiment_name=experiment_name,
                match_string=match_string,
                source_experiments=source_experiments,
                config_overrides=overrides,
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
            # Re-check: a sibling job may have finished this prior while we built the model.
            if os.path.exists(output_path):
                print(f"Completed by another job, skipping: {output_path}")
                continue
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