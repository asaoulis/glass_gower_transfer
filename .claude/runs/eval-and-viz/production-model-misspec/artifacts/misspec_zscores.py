"""Z-score diagnostics + matched-N calibration floor for the misspec eval.

Per variate, from misspec_posterior_samples_<match>.npz (SCALED space; z-scores are
affine-invariant so identical in physical space):
  z[event, param] = (theta0 - posterior_mean) / posterior_std
- coherent-bias check (advisor check #1): mean z per param, same-sign fraction
- z histograms vs N(0,1)
Matched-N floor (advisor check #2): recompute the EXACT calibration_error pipeline on random
subsamples of the in-dist nla_m events at the OOD variates' N.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.getcwd())  # repo-root imports (src.ml...) when run by path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PARAMS = ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"]
KEY3 = ["omega_m", "sigma_8", "w0"]

# Ordering comes from the ONE shared style module so every figure and table in this task
# agrees; the local copy this script used to carry listed only the GLASS ladder names and
# would have dumped all six gower_* variates into an alphabetical tail.
try:
    from gower_style import ORDER
except ImportError:  # running from the repo root rather than artifacts/
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gower_style import ORDER


def load(root, match):
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "*", f"misspec_posterior_samples_{match}.npz"))):
        name = os.path.basename(os.path.dirname(d))
        with np.load(d, allow_pickle=True) as z:
            out[name] = {"theta0s": z["theta0s"], "samples": z["samples"]}
        jp = os.path.join(os.path.dirname(d), f"misspec_evaluation_results_{match}.json")
        if os.path.exists(jp):
            with open(jp) as f:
                out[name]["json"] = json.load(f)
    return out


def zscores(theta0s, samples):
    mu = samples.mean(axis=0)          # [N, D]
    sd = samples.std(axis=0)
    return (theta0s - mu) / np.maximum(sd, 1e-12)


def calibration_error(samples, theta0s):
    """The exact production pipeline: get_tarp_coverage bootstrap-25 -> rank-hist MSE."""
    from src.ml.eval.tarp import get_tarp_coverage
    from src.ml.eval.evaluate_models import TARPDiagnostics

    # npz samples are already [S, N, D] — exactly get_tarp_coverage's expected layout.
    cov = get_tarp_coverage(
        samples.copy(), theta0s.copy(),
        bootstrap=True, num_bootstrap=25, seed=None, num_alpha_bins=100,
    )
    return TARPDiagnostics._calibration_error_from_coverage(cov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--match", default="ncosmo300_0")
    ap.add_argument("--out-prefix", default="misspec")
    ap.add_argument("--floor-reps", type=int, default=8)
    ap.add_argument("--indist", default="nla_m")
    args = ap.parse_args()

    data = load(args.root, args.match)
    names = [n for n in ORDER if n in data] + sorted(set(data) - set(ORDER))

    # ---------------- z-score table + coherence ----------------
    print("\n=== Z-SCORE SUMMARY (per param: mean z +- sem | std z | same-sign frac) ===")
    ztabs = {}
    for name in names:
        th, s = data[name]["theta0s"], data[name]["samples"]
        z = zscores(th, s)
        ztabs[name] = z
        n = z.shape[0]
        print(f"\n--- {name} (N={n}) ---")
        for j, p in enumerate(PARAMS):
            col = z[:, j]
            col = col[np.isfinite(col)]
            if col.size == 0:
                print(f"  {p:>8}: (no truth on disk)")
                continue
            sem = col.std() / np.sqrt(col.size)
            same = max((col > 0).mean(), (col < 0).mean())
            print(f"  {p:>8}: mean z = {col.mean():+6.2f} +- {sem:4.2f} | std z = {col.std():5.2f} "
                  f"| same-sign = {same:4.0%}")

    # ---------------- z histograms ----------------
    fig, axes = plt.subplots(len(names), len(PARAMS), figsize=(2.1 * len(PARAMS), 1.9 * len(names)),
                             sharex=True)
    xg = np.linspace(-6, 6, 200)
    ref = np.exp(-xg ** 2 / 2) / np.sqrt(2 * np.pi)
    for i, name in enumerate(names):
        z = ztabs[name]
        for j, p in enumerate(PARAMS):
            ax = axes[i, j]
            col = z[:, j]
            col = col[np.isfinite(col)]
            if col.size:
                ax.hist(np.clip(col, -6, 6), bins=40, range=(-6, 6), density=True,
                        color="tab:blue", alpha=0.75)
                ax.plot(xg, ref, "k-", lw=0.8)
                ax.axvline(0, color="k", lw=0.5, ls=":")
                m = col.mean()
                ax.axvline(np.clip(m, -6, 6), color="tab:red", lw=1.2)
                ax.set_title(f"{name}  {p}\nmean z={m:+.2f}", fontsize=7)
            else:
                ax.set_axis_off()
            ax.set_yticks([])
            ax.tick_params(labelsize=6)
    fig.suptitle("Per-event standardized residuals z=(θ₀−μ_post)/σ_post — red = mean, black = N(0,1)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    zpath = f"{args.out_prefix}_zscores.png"
    fig.savefig(zpath, dpi=150)
    print(f"\nwrote {zpath}")

    # ---------------- matched-N in-dist calibration floor ----------------
    if args.indist in data:
        th0, s0 = data[args.indist]["theta0s"], data[args.indist]["samples"]
        n0 = th0.shape[0]
        print(f"\n=== MATCHED-N IN-DIST FLOOR ({args.indist}, N={n0}; exact pipeline, "
              f"{args.floor_reps} random subsamples) ===")
        rng = np.random.default_rng(0)
        idx3 = [PARAMS.index(p) for p in KEY3]
        targets = sorted({data[n]["theta0s"].shape[0] for n in names if n != args.indist} | {n0})
        for N in targets:
            fulls, subs = [], []
            reps = 1 if N == n0 else args.floor_reps
            for _ in range(reps):
                sel = rng.choice(n0, size=min(N, n0), replace=False)
                fulls.append(calibration_error(s0[:, sel, :], th0[sel]))
                subs.append(calibration_error(s0[:, sel, :][:, :, idx3], th0[sel][:, idx3]))
            print(f"  N={N:4d}: cal_full(9p) = {np.mean(fulls):.4f} +- {np.std(fulls):.4f} | "
                  f"cal_om_s8_w0 = {np.mean(subs):.4f} +- {np.std(subs):.4f}")

    # ---------------- drop accounting ----------------
    print("\n=== EVENT ACCOUNTING ===")
    for name in names:
        j = data[name].get("json", {})
        print(f"  {name:>6}: n_test={j.get('n_test_files', '?')} "
              f"dropped_nonfinite={j.get('n_dropped_nonfinite', 'n/a (run-1 output)')} "
              f"missing={j.get('missing_params')} excluded={j.get('excluded_params')}")


if __name__ == "__main__":
    main()
