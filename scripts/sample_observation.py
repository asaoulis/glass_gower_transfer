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

Arm table (ASSESSMENT_hf.md, 2026-09-07) -> Stage-B experiment name per repeat r:
  nla_m        gower_nle_finetune_nla_m_bgp_z8_r{r}_ens9                 (headline; notebook arm)
  nla_m_nobgp  gower_nle_finetune_nla_m_z8_r{r}_ens9
  nla          gower_nle_finetune_nla_bgp_z8_hf_r{r}_ens9_e150
  nla_z        gower_nle_finetune_nla_z_bgp_z8_hf_r{r}_ens9_e150
  vd           gower_nle_finetune_nla_m_vd_bgp_z8_hf_r{r}_ens9_e150
  k2           gower_nle_finetune_nla_m_bgpk2_z16_k5_hf_r{r}_ens9_e150   (repeats 0-3 only)
Each arm reads the sc8a1 baked observation store `<store-prefix>_<label>_sc8a1` EXCEPT nla_m_nobgp,
which was trained on the plain-counts product and reads `<store-prefix>_<label>_a1` (F9: verify
the bake flags of that arm's training store before trusting its posterior).

Cost: ~14 h wall per (arm, repeat, label) at 25k samples on CPU (Stage-B burn-in dominates; see
memory nle-sampling-burnin); 3 labels x 29 (arm,repeat) = 87 jobs. Use --num-samples 2000 and
--repeats 0 for the GATE 3b pilot.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN_REMOTE = REPO / ".claude" / "cluster" / "run_remote.py"

ARMS = {
    "nla_m": ("gower_nle_finetune_nla_m_bgp_z8_r{r}_ens9", "sc8a1", (0, 1, 2, 3, 4)),
    "nla_m_nobgp": ("gower_nle_finetune_nla_m_z8_r{r}_ens9", "a1", (0, 1, 2, 3, 4)),
    "nla": ("gower_nle_finetune_nla_bgp_z8_hf_r{r}_ens9_e150", "sc8a1", (0, 1, 2, 3, 4)),
    "nla_z": ("gower_nle_finetune_nla_z_bgp_z8_hf_r{r}_ens9_e150", "sc8a1", (0, 1, 2, 3, 4)),
    "vd": ("gower_nle_finetune_nla_m_vd_bgp_z8_hf_r{r}_ens9_e150", "sc8a1", (0, 1, 2, 3, 4)),
    "k2": ("gower_nle_finetune_nla_m_bgpk2_z16_k5_hf_r{r}_ens9_e150", "sc8a1", (0, 1, 2, 3)),
}
DEFAULT_PRIORS = ("kids_s8_analytic", "LCDM_fixed_w0")


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


def eval_args(j, args):
    toks = ["--mode", "nle-external", "--experiments", j["experiment"], "--repeat-indices", str(j["repeat"]),
            "--data-store", j["store"], "--data-tag", j["label"], "--prior-mode", j["prior"],
            "--num-samples", str(args.num_samples), "--num-chains", str(args.num_chains)]
    if args.num_jobs:
        toks += ["--num-jobs", str(args.num_jobs)]
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


def cmd_fetch(args):
    from src.blind import BLIND_ROOT
    seen = set()
    for j in jobs(args):
        key = (j["label"], j["experiment"])
        if key in seen:
            continue
        seen.add(key)
        out_dir = Path(args.blind_root or BLIND_ROOT) / j["label"] / j["experiment"]
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(RUN_REMOTE)] + (["--dry-run"] if args.dry_run else []) + [
            "fetch", "--exp", j["experiment"], "--rel", f"external/{j['label']}", "--out_dir", str(out_dir)]
        print(" ".join(shlex.quote(c) for c in cmd[2:]))
        subprocess.check_call(cmd)
    print("fetched into the blind store (raw files are never opened here)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("plan", cmd_plan), ("submit", cmd_submit), ("fetch", cmd_fetch)):
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
        s.add_argument("--num-jobs", type=int, default=None)
        s.add_argument("--partition", default="CORES64")
        s.add_argument("--ncpu", type=int, default=16)
        s.add_argument("--wall_h", type=float, default=24)
        s.add_argument("--mem-gb", type=int, default=64)
        s.add_argument("--blind-root", default=None)
        s.add_argument("--dry-run", action="store_true")
        s.set_defaults(func=fn)
    args = ap.parse_args(argv)
    sys.path.insert(0, str(REPO))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
