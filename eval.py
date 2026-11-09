"""Evaluation entry point: python eval.py --mode <mode> [options].

mode list evaluates the best checkpoint of each named experiment in distribution. misspec evaluates
one model on the variate stores. The other modes pool repeats, score external data and dump
posteriors and summaries (EVAL_RECIPES.md documents each).
"""
import os
import argparse

from config.default import get_default_config
from config.experiments import experiments
from config.paper_2026 import paper_2026_experiments
from config.kids_legacy import kids_legacy_experiments
from config.kids_legacy_novd import kids_legacy_novd_experiments
from config.euclid import euclid_experiments
from config.kids_legacy_bgp import kids_legacy_bgp_experiments
from config.kids_legacy_vdq import kids_legacy_vdq_experiments
from config.kids_legacy_hos import kids_legacy_hos_experiments
from src.ml.eval.utils import evaluate_best_checkpoint
from src.ml.eval.misspec import VARIATE_SETS
from copy import copy

_VARIATE_SET_NAMES = sorted(VARIATE_SETS)

experiments.update(paper_2026_experiments)
experiments.update(kids_legacy_experiments)
experiments.update(kids_legacy_novd_experiments)
experiments.update(euclid_experiments)
experiments.update(kids_legacy_bgp_experiments)
experiments.update(kids_legacy_vdq_experiments)
experiments.update(kids_legacy_hos_experiments)

# A bare `python eval.py` runs DEFAULT_MODE.
DEFAULT_MODE = "list"

# evaluate_best_checkpoint skips runs with no checkpoint, so rows may be listed before their models exist.
DEFAULT_EXPERIMENTS = [
    # theta differs across these rows (8 / 9 / 9 parameters): compare FoM(omega_m, sigma_8) and
    # CI68 widths, never the all-parameter FoM.
    "kids_legacy_band_nla_m_bgp",
    "glass_encoder_finetune_nla_bgp_z8",
    "glass_encoder_finetune_nla_z_bgp_z8",
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

        max_tv = experiment_config.get("max_trainval_cosmos", None)

        # production configs use repeat_indices, legacy configs fall back to range(repeats)
        repeats = getattr(config, "repeats", 1)
        repeat_idxs = list(
            repeat_indices_override
            if repeat_indices_override is not None
            else (getattr(config, "repeat_indices", None) or range(repeats))
        )

        if isinstance(max_tv, (list, tuple)):
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
                for i in repeat_idxs:
                    cfg_copy = copy(cfg)
                    repeat_match, _ = apply_repeat_config(cfg_copy, i)
                    cfg_copy.match_string = repeat_match

                    print(
                        f"Evaluating '{experiment_name}' ncosmo={n_cosmo} repeat={i} match_string={cfg_copy.match_string}",
                        flush=True,
                    )

                    scalers, _, _, test_loader = prepare_data_parameters(cfg_copy)
                    evaluate_best_checkpoint(cfg_copy, test_loader, scalers["cosmo"])
        else:
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
                        choices=["list", "misspec", "summaries", "nle-external", "recover-scalers",
                                 "observe-build", "observe-strip", "obs-reference", "obs-score", "pool",
                                 "nle-loglik", "hos-bake", "hos-train"],
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
    parser.add_argument("--test-shape-noise-idx", default=None,
                        help="misspec mode: 'all' disables the per-cosmology augmentation subsample.")
    parser.add_argument("--test-id-source", choices=["heldout", "shared", "all"], default="heldout",
                        help="misspec mode: evaluate each variate on 'heldout' (default) or 'shared' test cosmologies.")
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
                        help="nle-external mode: output tag under checkpoints/<exp>/external/. Default: --data-store.")
    # --- --mode pool: tokens are limited to [A-Za-z0-9][A-Za-z0-9_.-]* or --[a-z-]* by the launcher ---
    parser.add_argument("--pool-arm", default=None,
                        help="--mode pool: arm name (src/ml/eval/arms.py) -- resolves to one "
                             "experiment per repeat. Alternative to listing --experiments.")
    parser.add_argument("--pool-draws", type=int, default=None,
                        help="pool mode: subsample each member to this many draws before concatenating. "
                             "Default: concatenate all.")
    parser.add_argument("--pool-check", action="store_true",
                        help="--mode pool: run the gates (pool-of-one identity + reproduction of "
                             "the production metrics) instead of producing a pooled posterior.")
    parser.add_argument("--prior-mode", default="gower",
                        choices=["gower", "kids_s8_analytic", "LCDM_fixed_w0"],
                        help="--mode nle-external: an NLE posterior is only defined WITH its prior, "
                             "so this is recorded in the output filename. The real-data run uses "
                             "kids_s8_analytic.")
    parser.add_argument("--num-chains", type=int, default=1,
                        help="nle-external mode: MCMC chains per observation (num_samples is split across them).")
    parser.add_argument("--num-jobs", type=int, default=None,
                        help="--mode nle-external: joblib pool size (work items in parallel).")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="nle-external mode: warmup sweeps discarded per chain.")
    parser.add_argument("--mcmc-workers", type=int, default=None,
                        help="nle-external mode: shards per event batch. Splits one observation's sample "
                             "budget over processes.")
    parser.add_argument("--mcmc-threads", type=int, default=None,
                        help="nle-external mode: torch threads per sampling worker (default 1).")
    parser.add_argument("--mcmc-seed", type=int, default=None,
                        help="--mode nle-external: base seed for the samplers (numpy global RNG "
                             "+ torch); shard i gets seed+i. Default None = unseeded, as before.")
    parser.add_argument("--emb-batch-size", type=int, default=64,
                        help="--mode nle-external: events per sampling work item. Smaller => more "
                             "batches => more cores busy when the event count is small.")
    parser.add_argument("--repro-check", action="store_true",
                        help="nle-external mode: reproduction gate. Scores the model's own test set "
                             "through the external path.")
    parser.add_argument("--max-events", type=int, default=200,
                        help="recover-scalers mode: number of events used to fit the input frame.")
    parser.add_argument("--members", type=int, nargs="+", default=None,
                        help="--mode recover-scalers: ensemble member indices (default: all). Each "
                             "member had its OWN stochastic scaler fit, so each needs its own frame.")
    parser.add_argument("--verify-frames", action="store_true",
                        help="--mode recover-scalers: do NOT fit; verify every member's persisted frame on "
                             "held-out test events through the sampler's embedding path, on this node")
    parser.add_argument("--require-member-frames", action="store_true",
                        help="nle-external mode: refuse to score unless every ensemble member has a "
                             "persisted input frame.")
    # --- --mode nle-loglik: NLE log L(t_e | theta_b) at every mock block, for the PPC weights ----
    parser.add_argument("--ppc-theta-store", default=None,
                        help="--mode nle-loglik: store whose (sim,outer,rot) blocks supply the theta "
                             "grid (default: the experiment's own data store)")
    parser.add_argument("--ppc-null", choices=["lock", "none"], default="lock",
                        help="--mode nle-loglik: also score the experiment's test-lock events (inner 0) "
                             "as fake observations for the PPC null")
    parser.add_argument("--ppc-max-null", type=int, default=None,
                        help="--mode nle-loglik: thin the null events to about this many (strided)")
    parser.add_argument("--ppc-nodes", type=int, default=128,
                        help="--mode nle-loglik: quadrature nodes for the cosmology marginal (0 = off)")
    parser.add_argument("--ppc-phi-grid", type=int, default=2048,
                        help="--mode nle-loglik: dense (Om,s8,w0) Sobol grid size for the emulator route (0 = off)")
    parser.add_argument("--ppc-summaries-only", action="store_true",
                        help="--mode nle-loglik: no likelihoods; write the arm's summaries t(x) for the observation "
                             "and EVERY theta-store file (summary-space PPC replicates)")
    parser.add_argument("--ppc-threads", type=int, default=None,
                        help="--mode nle-loglik: torch threads (and cosmo_dict reader threads)")
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
    parser.add_argument("--ood-k", type=int, default=10, help="summaries mode: kNN k for the OOD scores")
    # --mode hos-bake: per-mock HOS sidecars of a baked store, written under {base_path}/checkpoints/hos_sidecars/<store>/.
    parser.add_argument("--hos-workers", type=int, default=30, help="--mode hos-bake: worker processes")
    parser.add_argument("--hos-shard", type=int, default=None, help="--mode hos-bake: bake only this shard index")
    parser.add_argument("--hos-nshards", type=int, default=20, help="--mode hos-bake: shard count (sim_id %% n)")
    parser.add_argument("--hos-limit", type=int, default=None, help="--mode hos-bake: first N files only (test)")
    parser.add_argument("--hos-variant", choices=["r1ext", "x"], default="r1ext",
                        help="--mode hos-bake: 'x' also bakes the hos_x superset (src/ml/data/hos_x.py) into "
                             "hos_sidecars/<store>_hosx/ (the default r1ext bake is unchanged)")
    # --mode hos-train: run the named CPU training waves of config/kids_legacy_hos.py:HOS_TRAIN_WAVES.
    parser.add_argument("--hos-parallel", type=int, default=None, help="--mode hos-train: concurrent tasks")
    parser.add_argument("--hos-cores-per", type=int, default=None, help="--mode hos-train: cpus per task")
    # Observation modes live in src/observation. Imported lazily so a bug there cannot break the other modes.
    try:
        from src.observation.cluster_modes import (add_observe_args, add_reference_args, add_score_args,
                                                   add_strip_args)
        add_observe_args(parser)
        add_strip_args(parser)
        add_reference_args(parser)
        add_score_args(parser)
    except Exception as _ex:
        print(f"[eval] WARNING: observe-build args unavailable ({type(_ex).__name__}: {_ex})")
    args = parser.parse_args(argv)

    mode = args.mode or DEFAULT_MODE
    if mode == "observe-build":
        from src.observation.cluster_modes import run_observe_build
        return run_observe_build(args)
    if mode == "observe-strip":
        from src.observation.cluster_modes import run_obs_strip
        return run_obs_strip(args)
    if mode == "obs-reference":
        from src.observation.cluster_modes import run_obs_reference
        return run_obs_reference(args)
    if mode == "obs-score":
        from src.observation.cluster_modes import run_obs_score
        return run_obs_score(args)
    if mode == "hos-bake":
        if not args.data_store:
            raise SystemExit("--mode hos-bake requires --data-store <gpu5 store name>")
        from config.default import get_default_config as _default_config
        from src.ml.data.hos import run_bake
        from src.ml.eval.misspec import _GPU5
        out_dir = os.path.join(_default_config().base_path, "checkpoints", "hos_sidecars",
                               args.data_store + ("_hosx" if args.hos_variant == "x" else ""))
        counts = run_bake(f"{_GPU5}/{args.data_store}/output_*.h5", out_dir, workers=args.hos_workers,
                          shard=args.hos_shard, nshards=args.hos_nshards, limit=args.hos_limit,
                          log=lambda m: print(m, flush=True), variant=args.hos_variant)
        return 0 if counts["bad"] == 0 else 1
    if mode == "hos-train":
        from config.default import get_default_config as _default_config
        from config.kids_legacy_hos import HOS_TRAIN_WAVES
        from src.ml.parallel_train import run_wave
        import time
        if not args.experiments:
            raise SystemExit(f"--mode hos-train requires --experiments <wave> (one of {sorted(HOS_TRAIN_WAVES)})")
        tasks = [t for w in args.experiments for t in HOS_TRAIN_WAVES[w]
                 if args.repeat_indices is None or t["repeat"] in args.repeat_indices]
        tag = "_".join(args.experiments) + "_" + os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d%H%M%S"))
        log_dir = os.path.join(_default_config().base_path, "checkpoints", "hos_ptrain_logs", tag)
        rc = run_wave(tasks, repo=os.path.dirname(os.path.abspath(__file__)), log_dir=log_dir,
                      parallel=args.hos_parallel, cores_per=args.hos_cores_per, log=lambda m: print(m, flush=True))
        return 0 if all(v == 0 for v in rc.values()) else 1
    if mode == "pool":
        from src.ml.eval.pool_mode import run_pool_mode
        return run_pool_mode(args)
    if mode == "summaries":
        # --mode summaries: dump the compressor's z for train, held-out and each variate, then score
        # them (src/ml/eval/ood.py).
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
            ood_k=args.ood_k,
            base_path_override=args.base_path,
            fixed_test_ids_override=args.fixed_test_ids,
            max_trainval_cosmos_override=args.max_trainval_cosmos,
        )
    elif mode == "nle-external":
        # --mode nle-external: score a trained Stage-B ensemble on an external store with the
        # original scalers injected, never refit.
        from src.ml.eval.nle_external import run_external_nle_eval

        if not args.experiments:
            raise SystemExit("--mode nle-external requires --experiments <stage_b_experiment>")
        if args.repro_check:
            # --repro-check: run over the model's own test set and compare with the production eval dump.
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
                    warmup_steps=args.warmup_steps,
                    mcmc_workers=args.mcmc_workers,
                    mcmc_threads=args.mcmc_threads,
                    mcmc_seed=args.mcmc_seed,
                    batch_size=args.emb_batch_size,
                    dry_run=args.external_dry_run,
                    require_member_frames=args.require_member_frames,
                )
    elif mode == "nle-loglik":
        # --mode nle-loglik: PPC importance weights (ensemble NLE log-likelihood) of the observation
        # and the null events.
        from src.ml.eval.ppc_loglik import run_ppc_loglik
        from src.ml.eval.nle_external import match_string_for

        if not args.experiments:
            raise SystemExit("--mode nle-loglik requires --experiments <stage_b_experiment>")
        for exp in args.experiments:
            for r in (args.repeat_indices or (0,)):
                run_ppc_loglik(
                    exp, match_string_for(exp, r),
                    theta_store=args.ppc_theta_store,
                    obs_store=args.data_store,
                    tag=args.data_tag or "A2",
                    null=args.ppc_null,
                    max_null_events=args.ppc_max_null,
                    batch_size=args.emb_batch_size,
                    require_member_frames=args.require_member_frames,
                    n_threads=args.ppc_threads,
                    n_nodes=args.ppc_nodes,
                    phi_grid=args.ppc_phi_grid,
                    dry_run=args.external_dry_run,
                    summaries_only=args.ppc_summaries_only,
                )
        return 0
    elif mode == "recover-scalers":
        # --mode recover-scalers: recover each member's training-time input frame and persist it per
        # member (src/ml/eval/scaler_recovery.py).
        from src.ml.eval.nle_external import run_frame_verification, run_scaler_recovery

        if not args.experiments:
            raise SystemExit("--mode recover-scalers requires --experiments <stage_b_experiment>")
        if args.verify_frames:
            for exp in args.experiments:
                for r in (args.repeat_indices or (0,)):
                    run_frame_verification(exp, r, batch_size=args.emb_batch_size)
            return 0
        for exp in args.experiments:
            for r in (args.repeat_indices or (0,)):
                run_scaler_recovery(
                    exp, r,
                    members=args.members,
                    max_events=args.max_events,
                    batch_size=args.emb_batch_size,
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
            **({"test_shape_noise_idx": None}
               if str(args.test_shape_noise_idx).lower() == "all" else {}),
        )
    else:
        run_standard_eval(args.experiments or DEFAULT_EXPERIMENTS, args.repeat_indices)


if __name__ == "__main__":
    main()
