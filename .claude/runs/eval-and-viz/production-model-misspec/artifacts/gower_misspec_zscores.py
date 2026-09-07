"""Z-score diagnostics + matched-N calibration floor for the misspec eval.

Per variate, from misspec_posterior_samples_<match>.npz (SCALED space; z-scores are
affine-invariant so identical in physical space):
  z[event, param] = (theta0 - posterior_mean) / posterior_std
- coherent-bias check (advisor check #1): mean z per param, same-sign fraction
- z histograms vs N(0,1)
Matched-N floor (advisor check #2): recompute the EXACT calibration_error pipeline on random
subsamples of the in-dist nla_m events at the OOD variates' N.

The floor is matched on CLUSTER STRUCTURE, not on row count. The rows are augmentations
(footprint rotation x shape-noise) of a much smaller set of cosmologies: the in-dist set is
1590 rows over 199 cosmologies (8 each), while `gower_gb1p0`/`gower_gb1p3` are 80 rows over
only **40 cosmologies x 2 rows**. Drawing 80 random in-dist ROWS would touch ~80 different
cosmologies and so carry ~8x the effective sample of the thing it is meant to calibrate,
producing a floor band far too narrow and making the gb points look more significant than
they are. Instead we draw the same NUMBER OF COSMOLOGIES and the same ROWS PER COSMOLOGY as
the target variate.
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
            out[name] = {"theta0s": z["theta0s"], "samples": z["samples"],
                         "sim_ids": z["sim_ids"] if "sim_ids" in z.files else None,
                         "aug_ids": z["aug_ids"] if "aug_ids" in z.files else None}
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
    ap.add_argument("--floor-reps", type=int, default=200,
                    help="draws for the SMALL cluster shapes, where the floor actually matters")
    ap.add_argument("--summary-json", default=None,
                    help="gower_misspec_cal_vs_disagreement.json - supplies the observed "
                         "5-repeat cal_full_mean per variate for the floor comparison")
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
        sim0 = data[args.indist]["sim_ids"]
        n0 = th0.shape[0]
        if sim0 is None:
            print("\n!! in-dist npz has no sim_ids - cannot build a cluster-matched floor.")
            return
        idx_by_cos0 = {c: np.flatnonzero(sim0 == c) for c in np.unique(sim0)}
        cos0 = np.array(sorted(idx_by_cos0))
        print(f"\n=== CLUSTER-MATCHED IN-DIST FLOOR ({args.indist}: {n0} rows / "
              f"{len(cos0)} cosmologies; exact pipeline, {args.floor_reps} draws) ===")
        print("    each draw takes the target's OWN (n_cosmologies x rows-per-cosmology),")
        print("    NOT n random rows -- see the module docstring for why.")
        rng = np.random.default_rng(0)
        idx3 = [PARAMS.index(p) for p in KEY3]

        # group targets by cluster shape so gb1p0 and gb1p3 (both 40x2) share one floor
        shapes = {}
        for n in names:
            if n == args.indist or data[n]["sim_ids"] is None:
                continue
            sv = data[n]["sim_ids"]
            ncos = len(np.unique(sv))
            rpc = int(round(len(sv) / max(ncos, 1)))
            shapes.setdefault((ncos, rpc), []).append(n)

        floors = {}
        for (ncos, rpc), who in sorted(shapes.items()):
            if ncos > len(cos0):
                print(f"  [{ncos} x {rpc}] SKIP - more cosmologies than the in-dist set has.")
                continue
            # cost scales with n_rows; the large shapes' floor is essentially the in-dist
            # value itself, so spend the draws on the small shapes where the floor is the
            # whole point.
            reps = args.floor_reps if ncos * rpc <= 200 else max(4, args.floor_reps // 40)
            fulls, subs = [], []
            for _ in range(reps):
                pick = rng.choice(cos0, size=ncos, replace=False)
                sel = np.concatenate([
                    rng.choice(idx_by_cos0[c], size=min(rpc, len(idx_by_cos0[c])), replace=False)
                    for c in pick])
                fulls.append(calibration_error(s0[:, sel, :], th0[sel]))
                subs.append(calibration_error(s0[:, sel, :][:, :, idx3], th0[sel][:, idx3]))
            f_m, f_s = float(np.mean(fulls)), float(np.std(fulls))
            s_m, s_s = float(np.mean(subs)), float(np.std(subs))
            floors[f"{ncos}x{rpc}"] = {
                "n_cosmologies": ncos, "rows_per_cosmology": rpc, "n_rows": ncos * rpc,
                "variates": who, "reps": reps,
                "cal_full_mean": f_m, "cal_full_sd": f_s,
                "cal_full_p95": float(np.percentile(fulls, 95)),
                "cal3_mean": s_m, "cal3_sd": s_s,
                "cal3_p95": float(np.percentile(subs, 95)),
            }
            print(f"  [{ncos:3d} cos x {rpc} rows = {ncos * rpc:4d}] matches {','.join(who)} "
                  f"({reps} draws)")
            print(f"      cal_full(9p)  = {f_m:.4f} +- {f_s:.4f}  (95th pct {np.percentile(fulls, 95):.4f})")
            print(f"      cal_om_s8_w0  = {s_m:.4f} +- {s_s:.4f}  (95th pct {np.percentile(subs, 95):.4f})")

        # the ROW-matched floor the earlier version computed, kept to show the difference
        for (ncos, rpc), who in sorted(shapes.items()):
            N = ncos * rpc
            if N >= n0:
                continue
            nrep = args.floor_reps if N <= 200 else max(4, args.floor_reps // 40)
            fulls = [calibration_error(s0[:, sel, :], th0[sel])
                     for sel in (rng.choice(n0, size=N, replace=False)
                                 for _ in range(nrep))]
            print(f"  [row-matched N={N}, WRONG but shown for contrast] "
                  f"cal_full = {np.mean(fulls):.4f} +- {np.std(fulls):.4f}")

        fp = f"{args.out_prefix}_matched_floor.json"
        with open(fp, "w") as f:
            json.dump(floors, f, indent=2)
        print(f"  wrote {fp}")

        # where the real variates sit relative to their own floor
        summ = {}
        if args.summary_json and os.path.exists(args.summary_json):
            with open(args.summary_json) as f:
                summ = json.load(f)
        if summ:
            print("\n  --- variate vs its CLUSTER-MATCHED floor (cal_full, 5-repeat mean) ---")
            for (ncos, rpc), who in sorted(shapes.items()):
                key = f"{ncos}x{rpc}"
                if key not in floors:
                    continue
                fl = floors[key]
                for n in who:
                    obs = (summ.get(n) or {}).get("cal_full_mean")
                    if obs is None:
                        continue
                    sd = fl["cal_full_sd"] or 1e-12
                    above = obs > fl["cal_full_p95"]
                    # a z built from a few-draw floor sd is meaningless once obs is orders
                    # of magnitude above the floor -- report the ratio there instead
                    zc = (obs - fl["cal_full_mean"]) / sd
                    mag = f"{zc:+.1f} sd" if abs(zc) <= 20 else \
                          f"{obs / fl['cal_full_mean']:.0f}x the floor"
                    print(f"    {n:16s}: obs={obs:.4f} vs floor[{key}] "
                          f"{fl['cal_full_mean']:.4f} +- {fl['cal_full_sd']:.4f} "
                          f"(95th {fl['cal_full_p95']:.4f})  ->  {mag}  "
                          f"{'ABOVE the floor' if above else 'WITHIN the floor -- not a detection'}")
        else:
            print("\n  (pass --summary-json gower_misspec_cal_vs_disagreement.json "
                  "for the variate-vs-floor comparison)")

    # ---------------- drop accounting ----------------
    print("\n=== EVENT ACCOUNTING ===")
    for name in names:
        j = data[name].get("json", {})
        print(f"  {name:>6}: n_test={j.get('n_test_files', '?')} "
              f"dropped_nonfinite={j.get('n_dropped_nonfinite', 'n/a (run-1 output)')} "
              f"missing={j.get('missing_params')} excluded={j.get('excluded_params')}")


if __name__ == "__main__":
    main()
