"""Checkpoint evaluation entrypoint.

Two modes:
- ``list``    — the standard evaluation: loop over experiment names, rebuild config/dataloaders
                exactly as training did (same match_string/repeat logic), and write
                evaluation_results.json / tarp_credible_intervals.json per run (ensemble
                variants at the experiment level). This is how the NPE ensembles are evaluated
                in-distribution.
- ``misspec`` — the model-misspecification driver (src/ml/eval/misspec.py): ONE trained model
                evaluated on the TEST split of every Gower variate dataset with the ORIGINAL
                training scalers injected; outputs under checkpoints/<exp>/misspec/<variate>/.

CLI (all optional):
    python eval.py [--mode {list,misspec}] [--experiments EXP ...] [--repeat-indices I ...]
                   [--misspec-base EXP] [--num-samples N]

NOTE: until the gatekeeper's eval-submit passes CLI args through (requires a
bootstrap_install.sh redeploy), the cluster job runs a bare ``python eval.py`` and DEFAULT_MODE
below decides what that does. Flip DEFAULT_MODE back to "list" once args flow end-to-end.
"""
import argparse

from config.default import get_default_config
from config.experiments import experiments
from config.ablations import ablation_experiments
from config.kids_legacy import kids_legacy_experiments
from config.kids_legacy_counts import kids_legacy_counts_experiments
from config.kids_legacy_novd import kids_legacy_novd_experiments
from src.ml.eval.utils import evaluate_best_checkpoint
from copy import copy

experiments.update(ablation_experiments)  # Combine experiments and ablations into a single dict
experiments.update(kids_legacy_experiments)  # KiDS-Legacy NLA-M configs
experiments.update(kids_legacy_counts_experiments)  # counts-normalisation rerun configs
experiments.update(kids_legacy_novd_experiments)  # NO-VD production suite configs

# What a bare `python eval.py` runs. Explicit --mode wins. The gatekeeper passes eval.py CLI
# args through (redeployed 2026-07-08), so submissions should say --mode explicitly:
#   run_remote.py eval --args "--mode misspec --repeat-indices 0 1 2 3 4"
DEFAULT_MODE = "list"

# Standard-mode default experiment list (used when --experiments is not given).
DEFAULT_EXPERIMENTS = [
    # PRODUCTION eval (run on v100 as jobs finish; evaluate_best_checkpoint skips repeats/runs with no
    # checkpoint yet, so re-running is safe): main-variate Gower NPE ens9 (all 5 repeats) + the GLASS
    # sub-variate encoder-finetunes (NPE-style compressors). Each writes evaluation_results.json /
    # ensemble_evaluation_results_*.json + the P0 posterior_samples.npz / ensemble_posterior_samples_*.npz.
    "gower_npe_finetune_nla_m_z8",
    "glass_encoder_finetune_nla_z_z8",
    "glass_encoder_finetune_nla_z8",
    "glass_encoder_finetune_no_vd_z8",
    # counts-normalisation rerun (M3e): main-variate Gower NPE ens9 (all 5 repeats).
    "gower_npe_finetune_nla_m_counts_z8",
]


def load_config(experiment_name: str):
    """Load config in a way consistent with train.py."""
    experiment_config = experiments[experiment_name]

    config = get_default_config()
    config.experiment_name = experiment_name

    # Set all non-list values directly on the config, but skip max_trainval_cosmos
    for key, val in experiment_config.items():
        if key == "max_trainval_cosmos":
            continue
        setattr(config, key, val)

    return config, experiment_config


from src.ml.utils import prepare_data_parameters
from src.ml.models.utils import apply_repeat_config


def run_standard_eval(experiment_names, repeat_indices_override=None):
    """The standard experiment-list evaluation (in-distribution, per repeat)."""
    for experiment_name in experiment_names:
        if experiment_name not in experiments:
            print(f"Experiment '{experiment_name}' not found in config.experiments, skipping.")
            continue

        config, experiment_config = load_config(experiment_name)

        # Handle max_trainval_cosmos similarly to train.py
        max_tv = experiment_config.get("max_trainval_cosmos", None)

        # Keep evaluation consistent with training: evaluate across repeats. Production configs use
        # `repeat_indices` (not `repeats`); fall back to range(repeats) for legacy configs.
        repeats = getattr(config, "repeats", 1)
        repeat_idxs = list(
            repeat_indices_override
            if repeat_indices_override is not None
            else (getattr(config, "repeat_indices", None) or range(repeats))
        )

        if isinstance(max_tv, (list, tuple)):
            # Multiple cosmos: evaluate each separately
            for n_cosmo in max_tv:
                cfg = get_default_config()
                cfg.experiment_name = config.experiment_name
                cfg.test_shape_noise_idx = [0]

                for key, val in experiment_config.items():
                    if key == "max_trainval_cosmos":
                        continue
                    setattr(cfg, key, val)

                cfg.max_trainval_cosmos = int(n_cosmo)
                cfg.match_num_cosmo = True  # Ensure match_string includes n_cosmo
                for i in repeat_idxs:
                    cfg_copy = copy(cfg)  # Avoid mutating cfg across repeats
                    # Apply the exact repeat match_string logic used by train_model
                    repeat_match, _ = apply_repeat_config(cfg_copy, i)
                    cfg_copy.match_string = repeat_match

                    print(
                        f"Evaluating '{experiment_name}' ncosmo={n_cosmo} repeat={i} match_string={cfg_copy.match_string}",
                        flush=True,
                    )

                    scalers, _, _, test_loader = prepare_data_parameters(cfg_copy)
                    evaluate_best_checkpoint(cfg_copy, test_loader, scalers["cosmo"])
        else:
            # Single or no max_trainval_cosmos
            config.test_shape_noise_idx = [0]
            if max_tv is not None:
                config.max_trainval_cosmos = int(max_tv)

            for i in repeat_idxs:
                repeat_match, _ = apply_repeat_config(config, i)
                config.match_string = repeat_match

                print(
                    f"Evaluating '{experiment_name}' max_trainval_cosmos={getattr(config, 'max_trainval_cosmos', None)} repeat={i} match_string={config.match_string}",
                    flush=True,
                )

                scalers, _, _, test_loader = prepare_data_parameters(config)
                evaluate_best_checkpoint(config, test_loader, scalers["cosmo"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate trained checkpoints.")
    parser.add_argument("--mode", choices=["list", "misspec"], default=None,
                        help=f"evaluation mode (default: {DEFAULT_MODE})")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="list mode: experiment names (default: the DEFAULT_EXPERIMENTS list)")
    parser.add_argument("--repeat-indices", type=int, nargs="+", default=None,
                        help="restrict to these repeat indices (both modes; misspec >1 repeat "
                             "also computes the cross-repeat disagreement statistic)")
    parser.add_argument("--misspec-base", default="gower_npe_finetune_nla_m_z8",
                        help="misspec mode: the base experiment to evaluate on all variates")
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="misspec mode: posterior samples per test point")
    parser.add_argument("--variates", default=None, choices=["gower", "glass_pretrain"],
                        help="misspec mode: which variate-dataset set to evaluate (default gower)")
    parser.add_argument("--max-test-files", type=int, default=None,
                        help="misspec mode: cap each variate's test set to ~this many mocks "
                             "(whole cosmologies, sorted by sim_id)")
    args = parser.parse_args(argv)

    mode = args.mode or DEFAULT_MODE
    if mode == "misspec":
        from src.ml.eval.misspec import run_misspecification_eval

        run_misspecification_eval(
            base_experiment=args.misspec_base,
            repeat_indices=args.repeat_indices or (0,),
            num_samples=args.num_samples,
            variate_set=args.variates,
            max_test_files=args.max_test_files,
        )
    else:
        run_standard_eval(args.experiments or DEFAULT_EXPERIMENTS, args.repeat_indices)


if __name__ == "__main__":
    main()
