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
import os
import argparse

from config.default import get_default_config
from config.experiments import experiments
from config.ablations import ablation_experiments
from config.kids_legacy import kids_legacy_experiments
from config.kids_legacy_counts import kids_legacy_counts_experiments
from config.kids_legacy_novd import kids_legacy_novd_experiments
from config.kids_legacy_dn import kids_legacy_dn_experiments
from config.kids_legacy_bgp import kids_legacy_bgp_experiments
from config.archive.legacy_bgp_stack5 import bgp_stack5_experiments
from src.ml.eval.utils import evaluate_best_checkpoint
from src.ml.eval.misspec import VARIATE_SETS
from copy import copy

# Selectable variate-dataset sets for --mode misspec (defined in src/ml/eval/misspec.py):
# gower / glass_pretrain (VD-on era) and gower_novd / glass_pretrain_novd (the NO-VD suite).
_VARIATE_SET_NAMES = sorted(VARIATE_SETS)

experiments.update(ablation_experiments)  # Combine experiments and ablations into a single dict
experiments.update(kids_legacy_experiments)  # KiDS-Legacy NLA-M configs
experiments.update(kids_legacy_counts_experiments)  # counts-normalisation rerun configs
experiments.update(kids_legacy_novd_experiments)  # NO-VD production suite configs
experiments.update(kids_legacy_dn_experiments)  # dual-normalisation arm-comparison suite
experiments.update(kids_legacy_bgp_experiments)  # BGP campaign (galaxy-bias prior marginalised)
experiments.update(bgp_stack5_experiments)  # ARCHIVED stacked-ensemble ablation (closed; do not extend)

# What a bare `python eval.py` runs. Explicit --mode wins. The gatekeeper passes eval.py CLI
# args through (redeployed 2026-07-08), so submissions should say --mode explicitly:
#   run_remote.py eval --args "--mode misspec --repeat-indices 0 1 2 3 4"
DEFAULT_MODE = "list"

# Standard-mode default experiment list (used when --experiments is not given).
#
# ⚠️ Run eval on **a100 or l40s — NEVER v100**: the FoM credible-interval step OOMs a 16 GiB v100 at
# ens9 even after 72ec561 moved the quantile computation to CPU. (An earlier version of this comment
# said "run on v100"; that is wrong and will kill the job.)
# evaluate_best_checkpoint skips repeats/runs with no checkpoint yet, so re-running is safe and
# entries may be listed before their models exist.
# ARCHIVED list from the NO-VD / counts campaigns. Kept verbatim so those rows can be re-run by
# pointing DEFAULT_EXPERIMENTS back at it (or via --experiments); NOT evaluated by default any more,
# because a bare `python eval.py` on the cluster walks this whole list and each row costs real time.
_ARCHIVED_EXPERIMENTS = [
    # PRODUCTION eval: main-variate Gower NPE ens9 (all 5 repeats) + the GLASS sub-variate
    # encoder-finetunes (NPE-style compressors). Each writes evaluation_results.json /
    # ensemble_evaluation_results_*.json + the P0 posterior_samples.npz / ensemble_posterior_samples_*.npz.
    "gower_npe_finetune_nla_m_z8",
    "glass_encoder_finetune_nla_z_z8",
    "glass_encoder_finetune_nla_z8",
    "glass_encoder_finetune_no_vd_z8",
    # counts-normalisation rerun (M3e): main-variate Gower NPE ens9 (all 5 repeats).
    "gower_npe_finetune_nla_m_counts_z8",
    # === NO-VD production suite (the current MAIN analysis variate) =============================
    # M3e: main-variate Gower NPE ens9. M5a: the two GLASS sub-variate encoder-finetunes.
    "gower_npe_finetune_nla_m_novd_z8",
    "glass_encoder_finetune_nla_novd_z8",
    "glass_encoder_finetune_nla_z_novd_z8",
]


# === CURRENT CAMPAIGN: the M15 compression-ceiling suite + its GLASS-side references =============
# `evaluate_best_checkpoint` skips rows with no checkpoint yet, so the M15 entries can sit here
# before their models exist and the same list can be re-run as the arms land.
DEFAULT_EXPERIMENTS = [
    # --- REFERENCES: what the EXISTING compressors already achieve on the GLASS test split -------
    # These have never been evaluated (TALLY_evals.md has no NPE row for either), which is why the
    # variates' GLASS-side constraining power was unknown. `kids_legacy_band_nla_m_bgp` is the
    # BAND-ONLY model -- the pure 2-point reference the M15 hybrids have to beat.
    # ⚠️ theta differs across these rows (8 / 9 / 9 params), so FoM is comparable only on a SHARED
    # subset -- use FoM(omega_m,sigma_8) and the per-parameter CI68 widths, never the all-parameter
    # FoM, which is a d-dimensional volume ratio.
    "kids_legacy_band_nla_m_bgp",
    "glass_encoder_finetune_nla_bgp_z8",
    "glass_encoder_finetune_nla_z_bgp_z8",
    # --- M15: the 2-parameter compression-ceiling suite (user directive 2026-09-01) --------------
    # 4 arms x 3 repeats on the `nla` GLASS store, theta = (omega_m, sigma_8) only. M15a is the
    # 2-pt ceiling; the three hybrids test whether the maps add anything on top of it, whether the
    # band must be variate-matched, and whether the map warm start is load-bearing. Compare the
    # arms to EACH OTHER on FoM(omega_m,sigma_8) and the CI68 widths -- all four share one store,
    # one architecture and one split per repeat, so the comparison is clean.
    "m15a_band_nla_p2",
    "m15b_hybrid_nla_p2_newband_warmmap",
    "m15c_hybrid_nla_p2_oldband_warmmap",
    "m15d_hybrid_nla_p2_newband_scratchmap",
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
    parser.add_argument("--mode",
                        choices=["list", "misspec", "ebdiff", "summaries", "nle-external", "recover-scalers",
                                 "observe-build", "obs-reference", "obs-score"],
                        default=None,
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
    parser.add_argument("--variates", default=None, choices=_VARIATE_SET_NAMES,
                        help="misspec mode: which variate-dataset set to evaluate (default gower)")
    parser.add_argument("--variate-names", nargs="+", default=None,
                        help="misspec mode: restrict the chosen set to these variate names "
                             "(e.g. glass_nla_m glass_gb0p5 glass_gb1p5)")
    parser.add_argument("--max-test-files", type=int, default=None,
                        help="misspec mode: cap each variate's test set to ~this many mocks "
                             "(whole cosmologies, sorted by sim_id)")
    parser.add_argument("--eb-variant", default=None,
                        help="E/B group tag for --mode ebdiff (omit for bare-E pre-baked stores)")
    parser.add_argument("--test-id-source", choices=["heldout", "shared", "all"], default="heldout",
                        help="misspec mode: 'heldout' (default) evaluates each variate on the "
                             "model's held-out test cosmologies only; 'shared' evaluates EVERY "
                             "variate (reference included) on the cosmologies the OOD variates "
                             "have on disk — many more events, matched across variates, but no "
                             "held-out guarantee. Writes to misspec_shared/ so both can coexist.")
    parser.add_argument("--variate-glob", nargs="+", default=None, metavar="NAME=GLOB",
                        help="summaries mode: ad-hoc variate(s) as NAME=GLOB (e.g. a local store or a "
                             "real-data file), used INSTEAD of --variates when given")
    parser.add_argument("--max-train-files", type=int, default=20000,
                        help="summaries mode: size of the random train-split subset encoded as the "
                             "reference cloud (per repeat)")
    parser.add_argument("--data-patterns", default=None,
                        help="summaries mode: override the base experiment's data_patterns (LOCAL "
                             "smoke/self-tests only — the split, scalers and train cloud follow it)")
    parser.add_argument("--base-path", default=None,
                        help="summaries mode: override config.base_path (LOCAL self-tests: a dir whose "
                             "checkpoints/ holds the fetched ml-checkpoints tree)")
    parser.add_argument("--data-tag", default=None,
                        help="--mode nle-external: names the OUTPUT dir under checkpoints/<exp>/"
                             "external/<tag>/. Defaults to --data-store. Decoupled from the glob so "
                             "a real-observation run can be tagged e.g. kids_legacy_dr5.")
    parser.add_argument("--prior-mode", default="gower",
                        choices=["gower", "kids_s8_analytic", "LCDM_fixed_w0"],
                        help="--mode nle-external: an NLE posterior is only defined WITH its prior, "
                             "so this is recorded in the output filename. The real-data run uses "
                             "kids_s8_analytic.")
    parser.add_argument("--num-chains", type=int, default=1,
                        help="--mode nle-external: MCMC chains per observation. num_samples is the "
                             "TOTAL per observation and is split across chains, so raising this "
                             "shortens each chain -- the parallelism knob for the N=1 case.")
    parser.add_argument("--num-jobs", type=int, default=None,
                        help="--mode nle-external: joblib pool size (batches sampled in parallel).")
    parser.add_argument("--emb-batch-size", type=int, default=64,
                        help="--mode nle-external: events per sampling work item. Smaller => more "
                             "batches => more cores busy when the event count is small.")
    parser.add_argument("--repro-check", action="store_true",
                        help="--mode nle-external: run the REPRODUCTION GATE instead of an eval -- "
                             "score the model's own test set through the external path and check it "
                             "reproduces the production dump (file set, theta alignment, raw z vs "
                             "the cached embeddings, against the measured scaler-refit floor).")
    parser.add_argument("--max-events", type=int, default=200,
                        help="--mode recover-scalers: events used to fit the input frame. 6 scalar "
                             "params against n_events x z_dim residuals, so a few hundred is "
                             "already hugely overdetermined; start small to prove convergence.")
    parser.add_argument("--recover-steps", type=int, default=40,
                        help="--mode recover-scalers: LBFGS max_iter.")
    parser.add_argument("--members", type=int, nargs="+", default=None,
                        help="--mode recover-scalers: ensemble member indices (default: all). Each "
                             "member had its OWN stochastic scaler fit, so each needs its own frame.")
    parser.add_argument("--no-save", action="store_true",
                        help="--mode recover-scalers: fit and report but do not write scalers.pt")
    parser.add_argument("--external-dry-run", action="store_true",
                        help="--mode nle-external: resolve the pipeline and stop before sampling.")
    parser.add_argument("--data-store", default=None,
                        help="summaries mode: like --data-patterns but a bare dataset DIR NAME under the "
                             "gpu5 datasets root (cluster-safe: the gatekeeper forbids '/' and '*')")
    parser.add_argument("--fixed-test-ids", default=None,
                        help="summaries mode: override config.fixed_test_sim_ids (lock-file path) so a "
                             "GLASS-trained encoder can be split like the Gower chain")
    parser.add_argument("--max-trainval-cosmos", type=int, default=None,
                        help="summaries mode: override config.max_trainval_cosmos (train-cloud size in "
                             "cosmologies; checkpoint match string is NOT affected)")
    parser.add_argument("--no-ood", action="store_true",
                        help="summaries mode: only dump summaries, skip the in-job OOD scoring")
    parser.add_argument("--ood-k", type=int, default=10, help="summaries mode: kNN k for the OOD scores")
    parser.add_argument("--ood-n-perm", type=int, default=200,
                        help="summaries mode: permutations for the MMD two-sample p-value")
    # Unblinding front end (src/observation): catalogue -> observation -> baked per-arm stores.
    # Fail-soft: a bug in the (independent) observation package must never break the other modes.
    try:
        from src.observation.cluster_modes import add_observe_args, add_reference_args, add_score_args
        add_observe_args(parser)
        add_reference_args(parser)
        add_score_args(parser)
    except Exception as _ex:  # pragma: no cover
        print(f"[eval] WARNING: observe-build args unavailable ({type(_ex).__name__}: {_ex})")
    args = parser.parse_args(argv)

    mode = args.mode or DEFAULT_MODE
    if mode == "observe-build":
        from src.observation.cluster_modes import run_observe_build
        return run_observe_build(args)
    if mode == "obs-reference":
        from src.observation.cluster_modes import run_obs_reference
        return run_obs_reference(args)
    if mode == "obs-score":
        from src.observation.cluster_modes import run_obs_score
        return run_obs_score(args)
    if mode == "ebdiff":
        # Difference-map forensics on the paired b_g stores: no model, no training. Decides
        # whether the surviving b_g channel is signal-sector (red, coherent with the map) or
        # noise-sector (white). Writes under MODELS_ROOT so `fetch` can pull the JSON.
        from src.ml.eval.ebdiff import run_ebdiff_analysis
        from config.default import get_default_config

        out_root = os.path.join(get_default_config().base_path, "checkpoints",
                                "ebdiff_analysis", args.variates or "unknown")
        run_ebdiff_analysis(
            variate_set=args.variates,
            eb_variant=args.eb_variant,
            max_files=args.max_test_files or 120,
            out_root=out_root,
        )
    elif mode == "summaries":
        # Summary-space OOD diagnostic: dump the compressor's z for train / ID-heldout / each variate
        # (per repeat, per ensemble member) and score them (src/ml/eval/ood.py). Cheap: one encoder
        # forward per mock, no posterior sampling.
        from src.ml.eval.summaries import run_summary_extraction

        adhoc = None
        if args.variate_glob:
            adhoc = []
            for spec in args.variate_glob:
                name, _, glob_ = spec.partition("=")
                if not name or not glob_:
                    raise SystemExit(f"--variate-glob expects NAME=GLOB, got {spec!r}")
                adhoc.append({"name": name, "patterns": glob_, "exclude_params": []})
        data_patterns = args.data_patterns
        if args.data_store:
            from src.ml.eval.misspec import _GPU5
            data_patterns = f"{_GPU5}/{args.data_store}/output_*.h5"
        run_summary_extraction(
            base_experiment=args.misspec_base,
            repeat_indices=args.repeat_indices or (0,),
            variate_set=args.variates,
            variate_names=args.variate_names,
            variates=adhoc,
            max_test_files=args.max_test_files,
            test_id_source=args.test_id_source,
            max_train_files=args.max_train_files,
            data_patterns_override=data_patterns,
            run_ood=not args.no_ood,
            ood_k=args.ood_k,
            ood_n_perm=args.ood_n_perm,
            base_path_override=args.base_path,
            fixed_test_ids_override=args.fixed_test_ids,
            max_trainval_cosmos_override=args.max_trainval_cosmos,
        )
    elif mode == "nle-external":
        # Score a trained Stage-B NLE ensemble on an EXTERNAL dataset (a variate store, or a single
        # real observation) with the original training scalers + whitener injected, never refit.
        # See src/ml/eval/nle_external.py and EVAL_RECIPES.md.
        from src.ml.eval.nle_external import run_external_nle_eval

        if not args.experiments:
            raise SystemExit("--mode nle-external requires --experiments <stage_b_experiment>")
        if args.repro_check:
            # The anti-refit proof: run over the model's OWN test set and compare with what the
            # production eval already produced. No sampling, no external store needed.
            from src.ml.eval.nle_external import run_reproduction_check
            for exp in args.experiments:
                for r in (args.repeat_indices or (0,)):
                    run_reproduction_check(exp, r, batch_size=args.emb_batch_size)
            return 0
        if not args.data_store:
            raise SystemExit("--mode nle-external requires --data-store <gpu5 store name>")
        for exp in args.experiments:
            for r in (args.repeat_indices or (0,)):
                from src.ml.eval.nle_external import match_string_for
                match = match_string_for(exp, r)
                run_external_nle_eval(
                    exp, match,
                    store=args.data_store,
                    tag=args.data_tag or args.data_store,
                    prior_mode=args.prior_mode,
                    num_samples=args.num_samples,
                    num_chains=args.num_chains,
                    num_jobs=args.num_jobs,
                    batch_size=args.emb_batch_size,
                    dry_run=args.external_dry_run,
                )
    elif mode == "recover-scalers":
        # Recover the training-time input frame of an already-trained run and persist it per
        # ensemble member. See src/ml/eval/scaler_recovery.py for why this is necessary and why
        # matching z (rather than the "true" scalers) is the correct objective.
        from src.ml.eval.nle_external import run_scaler_recovery

        if not args.experiments:
            raise SystemExit("--mode recover-scalers requires --experiments <stage_b_experiment>")
        for exp in args.experiments:
            for r in (args.repeat_indices or (0,)):
                run_scaler_recovery(
                    exp, r,
                    members=args.members,
                    max_events=args.max_events,
                    steps=args.recover_steps,
                    batch_size=args.emb_batch_size,
                    save=not args.no_save,
                )
        return 0
    elif mode == "misspec":
        from src.ml.eval.misspec import run_misspecification_eval
        if args.test_id_source == "all":
            raise SystemExit("--test-id-source all is only implemented for --mode summaries")

        run_misspecification_eval(
            base_experiment=args.misspec_base,
            repeat_indices=args.repeat_indices or (0,),
            num_samples=args.num_samples,
            variate_set=args.variates,
            variate_names=args.variate_names,
            max_test_files=args.max_test_files,
            test_id_source=args.test_id_source,
        )
    else:
        run_standard_eval(args.experiments or DEFAULT_EXPERIMENTS, args.repeat_indices)


if __name__ == "__main__":
    main()
