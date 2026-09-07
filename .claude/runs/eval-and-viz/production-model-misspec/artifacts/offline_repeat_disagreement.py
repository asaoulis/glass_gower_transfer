"""Offline cross-repeat ensemble-disagreement from fetched misspec outputs.

When the misspec repeats run as SEPARATE cluster jobs (zombie-GPU workaround), the in-job
cross-repeat disagreement step never fires. This runner reproduces it exactly from the fetched
per-repeat posterior npz files: per variate, per repeat, compute per-event posterior moments,
align by test-file basename, and score with mean pairwise symmetric diag-Gaussian KL
(src.ml.eval.misspec._compute_repeat_disagreement — the ensemble_uncertainty.py formulation).

Also emits the calibration-vs-disagreement summary: per variate, cal errors (mean ± std across
repeats, from the per-repeat results JSONs) alongside kl_mean — the misspecification-statistic
comparison — and a scatter plot.

Usage:
    python offline_repeat_disagreement.py \
        --root ml-checkpoints/kids_legacy_hybrid_nla_m_lmin50_fwhm4_z8/misspec \
        --matches _0 _1 _2 _3 _4 --out-prefix glass_misspec
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="misspec/ dir holding <variate>/ subdirs")
    ap.add_argument("--matches", nargs="+", required=True, help="repeat match strings, e.g. _0 _1")
    ap.add_argument("--out-prefix", default="misspec_offline")
    args = ap.parse_args()

    from src.ml.eval.misspec import _compute_repeat_disagreement

    variates = sorted({os.path.basename(os.path.dirname(p))
                       for p in glob.glob(os.path.join(args.root, "*", "misspec_evaluation_results_*.json"))})
    table = {}
    for name in variates:
        vdir = os.path.join(args.root, name)
        per_repeat, cals_full, cals_sub, dropped = {}, [], [], []
        for m in args.matches:
            npz = os.path.join(vdir, f"misspec_posterior_samples_{m}.npz")
            js = os.path.join(vdir, f"misspec_evaluation_results_{m}.json")
            if not os.path.exists(npz):
                print(f"[offline] {name} {m}: npz missing — skipping this repeat")
                continue
            with np.load(npz, allow_pickle=True) as f:
                s = f["samples"]
                files = ([str(x) for x in f["test_files"]] if "test_files" in f
                         else [str(i) for i in range(s.shape[1])])
            per_repeat[m] = {
                "mu": s.mean(axis=0).astype(np.float32),
                "var": s.var(axis=0).astype(np.float32),
                "test_files": files,
            }
            if os.path.exists(js):
                with open(js) as fh:
                    payload = json.load(fh)
                tarp = payload["metrics"]["tarp"]
                cals_full.append(tarp["full"]["calibration_error"])
                sub = tarp["subsets"].get("sigma_8__omega_m__w0", {})
                cals_sub.append(sub.get("calibration_error", float("nan")))
                dropped.append(payload.get("n_dropped_nonfinite", 0))
        if len(per_repeat) < 2:
            print(f"[offline] {name}: <2 repeats with samples — no disagreement")
            continue
        dis = _compute_repeat_disagreement(name, per_repeat, vdir)
        table[name] = {
            "n_repeats": len(per_repeat),
            "cal_full_mean": float(np.nanmean(cals_full)), "cal_full_std": float(np.nanstd(cals_full)),
            "cal_sub_mean": float(np.nanmean(cals_sub)), "cal_sub_std": float(np.nanstd(cals_sub)),
            "kl_mean": dis["kl_mean"], "kl_median": dis["kl_median"], "kl_p90": dis["kl_p90"],
            "dropped_mean": float(np.mean(dropped)) if dropped else 0.0,
        }

    print("\n=== calibration vs cross-repeat disagreement ===")
    print(f"{'variate':>14} | {'cal_full (m±s)':>18} | {'cal_om_s8_w0 (m±s)':>20} | "
          f"{'kl_mean':>8} {'kl_med':>8} {'kl_p90':>8} | drop")
    for name, r in sorted(table.items(), key=lambda kv: kv[1]["kl_mean"]):
        print(f"{name:>14} | {r['cal_full_mean']:8.4f} ±{r['cal_full_std']:6.4f} | "
              f"{r['cal_sub_mean']:9.4f} ±{r['cal_sub_std']:7.4f} | "
              f"{r['kl_mean']:8.3f} {r['kl_median']:8.3f} {r['kl_p90']:8.3f} | {r['dropped_mean']:.0f}")

    with open(f"{args.out_prefix}_cal_vs_disagreement.json", "w") as f:
        json.dump(table, f, indent=4)

    fig, ax = plt.subplots(figsize=(6.4, 5))
    for name, r in table.items():
        ax.errorbar(r["kl_mean"], r["cal_sub_mean"], yerr=r["cal_sub_std"],
                    fmt="o", ms=8, capsize=3, label=name)
        ax.annotate(name, (r["kl_mean"], r["cal_sub_mean"]), textcoords="offset points",
                    xytext=(7, 5), fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("cross-repeat posterior disagreement (mean pairwise sym. KL)")
    ax.set_ylabel(r"TARP calibration error $\{\Omega_m, \sigma_8, w_0\}$ (mean over repeats)")
    ax.set_title("Miscalibration vs repeat-ensemble spread")
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_cal_vs_disagreement.png", dpi=160)
    print(f"\nwrote {args.out_prefix}_cal_vs_disagreement.{{json,png}}")


if __name__ == "__main__":
    main()
