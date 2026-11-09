#!/usr/bin/env python
"""Blind sampling driver: run every production NLE arm x repeat x prior on observation stores on the cluster.

Subcommands: plan prints the job matrix, submit runs it, pool submits the pooled-posterior jobs and
fetch pulls the raw posteriors straight into the blind store. Nothing here opens a sample file.
The final posteriors are the pooled ones. The arm table is src/ml/eval/arms.py.
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
from src.ml.eval.arms import ACTIVE_ARMS, ARMS, DEFAULT_PRIORS, arm_experiment, assert_active  # noqa: E402


def jobs(args):
    for label in args.labels:
        for arm in args.arms:
            assert_active(arm)          # a DELAYED arm still names a no-VD chain
            exp_t, bake, reps = ARMS[arm]
            for r in args.repeats:
                if r not in reps:
                    continue
                for prior in args.priors:
                    store = (args.store_override or (f"{args.store_prefix}_{label}_{bake}" if args.per_label_store
                                                    else f"{args.store_prefix}_{bake}"))
                    yield {"label": label, "arm": arm, "repeat": r, "prior": prior, "experiment": arm_experiment(arm, r),
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
    if args.mcmc_workers:
        # The seed depends on arm and repeat, so pooled members do not share MC initialisation noise.
        seed = args.mcmc_seed if args.mcmc_seed is not None else shard_seed(j["arm"], j["repeat"], j["prior"])
        toks += ["--mcmc-workers", str(args.mcmc_workers), "--mcmc-threads", str(args.mcmc_threads),
                 "--mcmc-seed", str(seed)]
    if args.warmup_steps is not None:
        toks += ["--warmup-steps", str(args.warmup_steps)]
    if args.emb_batch_size is not None:
        toks += ["--emb-batch-size", str(args.emb_batch_size)]
    if args.require_member_frames:
        toks += ["--require-member-frames"]   # refuse members without their own persisted input frame
    return " ".join(toks)


def cmd_plan(args):
    js = list(jobs(args))
    print(f"{len(js)} sampling jobs:")
    for j in js:
        line = (f"python {RUN_REMOTE} eval --cpu --partition {args.partition} --ncpu {args.ncpu} --wall_h {args.wall_h:g} "
                f"--mem-gb {args.mem_gb} --args {shlex.quote(eval_args(j, args))}")
        print(f"  [{j['label']} {j['arm']} r{j['repeat']} {j['prior']}] {line}")
    return 0


def cmd_submit(args):
    js = list(jobs(args))
    for j in js:
        cmd = [sys.executable, str(RUN_REMOTE)] + (["--dry-run"] if args.dry_run else []) + [
            "eval", "--cpu", "--partition", args.partition, "--ncpu", str(args.ncpu), "--wall_h", f"{args.wall_h:g}",
            "--mem-gb", str(args.mem_gb), "--args", eval_args(j, args)]
        print(f"[{j['label']} {j['arm']} r{j['repeat']} {j['prior']}] " + " ".join(shlex.quote(c) for c in cmd[2:]))
        subprocess.check_call(cmd)
    return 0


def pool_jobs(args):
    """(label, arm, prior, repeats-that-exist-and-were-requested) -- pooling needs >= 2 members."""
    for label in args.labels:
        for arm in args.arms:
            assert_active(arm)
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
    from src.blind.paths import BLIND_ROOT
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
        s.add_argument("--arms", nargs="+", default=list(ACTIVE_ARMS), choices=list(ARMS))
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
        s.add_argument("--require-member-frames", action="store_true",
                       help="pass --require-member-frames to eval (refuse members lacking a persisted scalers.pt)")
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
