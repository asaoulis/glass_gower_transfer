#!/usr/bin/env python
"""Blind sampling driver: run every production NLE arm x repeat x prior on one or more observation
stores on the cluster, and pull the raw posteriors straight into the BLIND store.

    PYTHONPATH=. python scripts/sample_observation.py plan   --labels A B C --store-prefix obs_kids [--priors kids_s8_analytic LCDM_fixed_w0] [--repeats 0 1 2 3 4]
    PYTHONPATH=. python scripts/sample_observation.py submit --labels A --store-prefix obs_gate0b --arms nla_m --repeats 0 --num-samples 2000 [--dry-run]
    PYTHONPATH=. python scripts/sample_observation.py fetch  --labels A --store-prefix obs_gate0b [--arms ...] [--repeats ...]

`plan` prints the job matrix and the exact `run_remote.py eval --cpu --args ...` lines (nothing
submitted). `submit` runs them (one job per (arm, repeat); ALL labels' stores can be scored in one
job only if they share a store -- here one store per label, so one job per label as well).
`fetch` pulls checkpoints/<exp>/external/<label>/ into BLIND_ROOT/<label>/<exp>/ (the guard allows
`run_remote.py fetch`; nothing here opens a sample file).

    PYTHONPATH=. python scripts/sample_observation.py pool   --labels A B C --store-prefix obs_kids [--arms ...] [--repeats ...] [--priors ...]
    PYTHONPATH=. python scripts/sample_observation.py fetch  --labels A --store-prefix obs_kids --what pooled

`pool` submits ONE cheap CPU job per (label, arm, prior): `eval.py --mode pool --pool-arm <arm>
--repeat-indices <reps> --data-tag <label> --prior-mode <prior>` (HANDOFF_pooling.md §3.2) -- the
equal-weight seed mixture q_pool = (1/R) sum_r q_r, i.e. the concatenated draws. THE FINAL
POSTERIORS ARE THE POOLED ONES (user, 2026-09-08); the per-repeat dumps stay as the seed-spread
diagnostic. `fetch --what pooled` pulls checkpoints/pooled/<arm>/external/<label>/ into
BLIND_ROOT/<label>/pooled_<arm>/ (`--what both` = repeats + pooled).

Arm table (ASSESSMENT_hf.md, 2026-09-07) -> Stage-B experiment name per repeat r:
  nla_m        gower_nle_finetune_nla_m_bgp_z8_r{r}_ens9                 (headline; notebook arm)
  nla_m_nobgp  gower_nle_finetune_nla_m_z8_r{r}_ens9
  nla          gower_nle_finetune_nla_bgp_z8_hf_r{r}_ens9_e150
  nla_z        gower_nle_finetune_nla_z_bgp_z8_hf_r{r}_ens9_e150
  vd           gower_nle_finetune_nla_m_vd_bgp_z8_hf_r{r}_ens9_e150
  k2           gower_nle_finetune_nla_m_bgpk2_z16_k5_hf_r{r}_ens9_e150   (repeats 0-3 only)
  band         gower_nle_finetune_band_nla_m_bgp_k8_r{r}_ens9_e150       (M16, 2-pt only;
               repeats 0,2,3,4 -- r1 joins when job 1359261's eval lands)
Each arm reads the sc8a1 baked observation store `<store-prefix>_<label>_sc8a1` EXCEPT nla_m_nobgp,
whose training store `gower_mocks_nla_m_f16_fwhm4_lmin56_lcut1400` was baked with
`--eb-variant fwhm4_lmin56_lcut1400 --keep-variant-tag` and NO `--noise-norm` (prebake log in
.claude/runs/training-runs/vicreg-nle-first-test/plan.md; config `_GOWER_EB_VARIANT_FWHM4`, no
`eb_noise_norm`), i.e. the `a0_tagged` bake: it reads `<store-prefix>_<label>_a0_tagged`.

Cost: ~14 h wall per (arm, repeat, label) at 25k samples on ONE core (the historical path; Stage-B
burn-in dominates). With the N=1 sharding (--ncpu 64 --mcmc-workers 60 --num-jobs 60 on CORES64,
or --ncpu 40 --mcmc-workers 36 --num-jobs 36 on CORES40) the same run is ~17 min at warmup 500
(SAMPLING_PROFILE_N1.md). One label x 29 (arm,repeat) x 2 priors = 58 jobs. Use --num-samples 2000
and --repeats 0 for a pilot.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN_REMOTE = REPO / ".claude" / "cluster" / "run_remote.py"

sys.path.insert(0, str(REPO))
from src.ml.eval.arms import ARMS, DEFAULT_PRIORS  # noqa: E402  -- the ONE arm table (eval --mode pool reads the same)


def jobs(args):
    for label in args.labels:
        for arm in args.arms:
            exp_t, bake, reps = ARMS[arm]
            for r in args.repeats:
                if r not in reps:
                    continue
                for prior in args.priors:
                    store = (args.store_override or (f"{args.store_prefix}_{label}_{bake}" if args.per_label_store
                                                    else f"{args.store_prefix}_{bake}"))
                    yield {"label": label, "arm": arm, "repeat": r, "prior": prior, "experiment": exp_t.format(r=r),
                           "store": store}


def shard_seed(arm, repeat, prior):
    """Per (arm, repeat, prior) base seed; shard k uses base+k (ensemble_nle.generate_samples)."""
    return 1000 * (list(ARMS).index(arm) + 1) + 100 * list(DEFAULT_PRIORS).index(prior) + 10 * int(repeat) if prior in DEFAULT_PRIORS \
        else 1000 * (list(ARMS).index(arm) + 1) + 10 * int(repeat) + 7


def eval_args(j, args):
    toks = ["--mode", "nle-external", "--experiments", j["experiment"], "--repeat-indices", str(j["repeat"]),
            "--data-store", j["store"], "--data-tag", j["label"], "--prior-mode", j["prior"],
            "--num-samples", str(args.num_samples), "--num-chains", str(args.num_chains)]
    if args.num_jobs:
        toks += ["--num-jobs", str(args.num_jobs)]
    # N=1 sampling knobs (SAMPLING_PROFILE_N1.md): shard ONE observation's sample budget over
    # --mcmc-workers processes (each with --num-chains chains; pooled = one process with K*C chains,
    # identical in distribution), 1 torch thread each, seeded per shard. 4.7x like-for-like on a
    # 64-core node at 25k draws; keep warmup 500 for the blind run.
    if args.mcmc_workers:
        # Deterministic but REPEAT- and ARM-dependent shard seed (DECISIONS P-6): a constant seed
        # would give every pooled member the same MC initialisation noise. --mcmc-seed overrides.
        seed = args.mcmc_seed if args.mcmc_seed is not None else shard_seed(j["arm"], j["repeat"], j["prior"])
        toks += ["--mcmc-workers", str(args.mcmc_workers), "--mcmc-threads", str(args.mcmc_threads),
                 "--mcmc-seed", str(seed)]
    if args.warmup_steps is not None:
        toks += ["--warmup-steps", str(args.warmup_steps)]
    if args.emb_batch_size is not None:
        toks += ["--emb-batch-size", str(args.emb_batch_size)]   # multi-event stores: items = n_batches x mcmc_workers
    return " ".join(toks)


def cmd_plan(args):
    js = list(jobs(args))
    print(f"{len(js)} sampling jobs:")
    for j in js:
        line = (f"python {RUN_REMOTE} eval --cpu --partition {args.partition} --ncpu {args.ncpu} --wall_h {args.wall_h} "
                f"--mem-gb {args.mem_gb} --args {shlex.quote(eval_args(j, args))}")
        print(f"  [{j['label']} {j['arm']} r{j['repeat']} {j['prior']}] {line}")
    return 0


def cmd_submit(args):
    js = list(jobs(args))
    for j in js:
        cmd = [sys.executable, str(RUN_REMOTE)] + (["--dry-run"] if args.dry_run else []) + [
            "eval", "--cpu", "--partition", args.partition, "--ncpu", str(args.ncpu), "--wall_h", str(args.wall_h),
            "--mem-gb", str(args.mem_gb), "--args", eval_args(j, args)]
        print(f"[{j['label']} {j['arm']} r{j['repeat']} {j['prior']}] " + " ".join(shlex.quote(c) for c in cmd[2:]))
        subprocess.check_call(cmd)
    return 0


def pool_jobs(args):
    """(label, arm, prior, repeats-that-exist-and-were-requested) -- pooling needs >= 2 members."""
    for label in args.labels:
        for arm in args.arms:
            _exp_t, _bake, reps_avail = ARMS[arm]
            reps = [r for r in args.repeats if r in reps_avail]
            if len(reps) < 2:
                print(f"[{label} {arm}] SKIP pool: {len(reps)} member(s) requested/available (need >= 2)")
                continue
            for prior in args.priors:
                yield {"label": label, "arm": arm, "prior": prior, "repeats": reps}


def pool_args(j):
    return " ".join(["--mode", "pool", "--pool-arm", j["arm"], "--repeat-indices", *map(str, j["repeats"]),
                     "--data-tag", j["label"], "--prior-mode", j["prior"]])


def cmd_pool(args):
    js = list(pool_jobs(args))
    for j in js:
        cmd = [sys.executable, str(RUN_REMOTE)] + (["--dry-run"] if args.dry_run else []) + [
            "eval", "--cpu", "--partition", args.partition, "--ncpu", "8", "--wall_h", "1",
            "--mem-gb", "64", "--args", pool_args(j)]
        print(f"[{j['label']} {j['arm']} {j['prior']} pool r{j['repeats']}] " + " ".join(shlex.quote(c) for c in cmd[2:]))
        subprocess.check_call(cmd)
    print(f"{len(js)} pool jobs submitted (seconds of work each; wall/mem are slack)")
    return 0


def cmd_fetch(args):
    from src.blind import BLIND_ROOT
    root = Path(args.blind_root or BLIND_ROOT)
    seen = set()
    if args.what in ("repeats", "both"):
        for j in jobs(args):
            key = (j["label"], j["experiment"])
            if key in seen:
                continue
            seen.add(key)
            out_dir = root / j["label"] / j["experiment"]
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(RUN_REMOTE)] + (["--dry-run"] if args.dry_run else []) + [
                "fetch", "--exp", j["experiment"], "--rel", f"external/{j['label']}", "--out_dir", str(out_dir)]
            print(" ".join(shlex.quote(c) for c in cmd[2:]))
            subprocess.check_call(cmd)
    if args.what in ("pooled", "both"):
        for label in args.labels:
            for arm in args.arms:
                out_dir = root / label / f"pooled_{arm}"
                out_dir.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, str(RUN_REMOTE)] + (["--dry-run"] if args.dry_run else []) + [
                    "fetch", "--exp", "pooled", "--rel", f"{arm}/external/{label}", "--out_dir", str(out_dir)]
                print(" ".join(shlex.quote(c) for c in cmd[2:]))
                try:
                    subprocess.check_call(cmd)
                except subprocess.CalledProcessError:
                    print(f"[{label} {arm}] no pooled dump on the cluster yet (run `pool` first)")
    print("fetched into the blind store (raw files are never opened here)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("plan", cmd_plan), ("submit", cmd_submit), ("fetch", cmd_fetch), ("pool", cmd_pool)):
        s = sub.add_parser(name)
        s.add_argument("--labels", nargs="+", required=True)
        s.add_argument("--store-prefix", required=True, help="observation store prefix; store = <prefix>_<label>_<bake>")
        s.add_argument("--per-label-store", action="store_true", default=True)
        s.add_argument("--store-override", default=None, help="use this exact store name for every job (pilot runs)")
        s.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
        s.add_argument("--repeats", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        s.add_argument("--priors", nargs="+", default=list(DEFAULT_PRIORS))
        s.add_argument("--num-samples", type=int, default=25000)
        s.add_argument("--num-chains", type=int, default=8)
        s.add_argument("--num-jobs", type=int, default=None, help="joblib work items in parallel (aim ~ ncpu with --mcmc-workers)")
        s.add_argument("--mcmc-workers", type=int, default=None, help="shards per event (N=1 knob); e.g. 60 on CORES64, 36 on CORES40")
        s.add_argument("--mcmc-threads", type=int, default=1)
        s.add_argument("--mcmc-seed", type=int, default=None, help="override the per-(arm,repeat,prior) shard seed")
        s.add_argument("--warmup-steps", type=int, default=None, help="MCMC warmup sweeps per chain (eval.py default 500)")
        s.add_argument("--emb-batch-size", type=int, default=None,
                       help="events per work item (eval.py default 64); for a 160-event matched store use ~8 so items ~ ncpu")
        s.add_argument("--partition", default="CORES64")
        s.add_argument("--ncpu", type=int, default=16)
        s.add_argument("--wall_h", type=float, default=24)
        s.add_argument("--mem-gb", type=int, default=64)
        s.add_argument("--blind-root", default=None)
        s.add_argument("--what", default="both", choices=["repeats", "pooled", "both"], help="fetch: which family")
        s.add_argument("--dry-run", action="store_true")
        s.set_defaults(func=fn)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
