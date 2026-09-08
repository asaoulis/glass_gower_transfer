#!/usr/bin/env python
"""Tier-1 data-check report for one or more observation labels against the nla_m reference cloud.

    PYTHONPATH=. python scripts/tier1_report.py --ref-baked reference_baked.npz --ref-raw reference_raw.npz \
        --obs A=/path/obsA_store/output_9001_out0_rot0_0.h5 [B=...] --out-dir <dir> \
        [--obs-eb-variant sc8_fwhm4_lmin56_lcut1400 --obs-noise-norm rand]   # if the obs file is RAW (not baked)
        [--exclude-truthkey <_truthkey/observation_A_truthkey.json> ...]    # mock-as-real: drop its own sim_id
        [--alpha 0.01] [--seed 0]

Writes tier1_results.json (all statistics), tier1_verdict.json (pre-registered pass/fail at alpha),
and figures: bandpower robust-z residuals (21 panels), BB bandpowers vs the mock cloud, and the
kNN score histograms with the observation marked. Blind-safe: nothing here reads a cosmology, and
the truth-key sidecar is read ONLY for its sim_id, which is never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402


def _figures(res, out_dir: Path, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "twopoint" in res:
        z = np.asarray(res["twopoint"]["robust_z"])          # [M,21,B]
        labs = res["twopoint"]["spectrum_labels"]
        fig, axes = plt.subplots(3, 7, figsize=(18, 7), sharey=True)
        for s, ax in enumerate(axes.ravel()):
            for m, lab in enumerate(labels):
                ax.plot(z[m, s], marker="o", ms=3, lw=1, label=lab)
            ax.axhspan(-2, 2, color="0.9")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"EE {labs[s]}", fontsize=9)
            ax.set_ylim(-5, 5)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle("Tier-1a: bandpower robust z vs the nla_m mock cloud (per band)")
        fig.tight_layout()
        fig.savefig(out_dir / "tier1_bandpower_residuals.png", dpi=130)
        plt.close(fig)

    if "bmodes" in res:
        z = np.asarray(res["bmodes"]["robust_z"])
        labs = res["bmodes"]["spectrum_labels"]
        fig, axes = plt.subplots(3, 7, figsize=(18, 7), sharey=True)
        for s, ax in enumerate(axes.ravel()):
            for m, lab in enumerate(labels):
                ax.plot(z[m, s], marker="o", ms=3, lw=1, label=lab)
            ax.axhspan(-2, 2, color="0.9")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"BB {labs[s]}", fontsize=9)
            ax.set_ylim(-5, 5)
        pte = res["bmodes"]["pte_empirical"]
        fig.suptitle("Tier-1b: BB bandpower robust z vs the mock-BB cloud; PTE(chi2 vs mock mean) = "
                     + ", ".join(f"{l}: {p:.3f}" for l, p in zip(labels, pte)))
        fig.tight_layout()
        fig.savefig(out_dir / "tier1_bmode_residuals.png", dpi=130)
        plt.close(fig)

    panels = []
    if "twopoint" in res:
        panels.append(("2-pt kNN (whitened bandpowers)", res["twopoint"]["knn"]))
    if "emap" in res:
        panels.append(("E-map kNN (full)", res["emap"]["full"]["knn"]))
        for k, v in res["emap"].items():
            if k.startswith("pca"):
                panels.append((f"E-map kNN ({k})", v["knn"]))
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.6))
        axes = np.atleast_1d(axes)
        for ax, (title, blk) in zip(axes, panels):
            null = np.asarray(blk["null_scores"])
            ax.hist(null, bins=40, color="0.8", label="held-out mocks (null)")
            for m, lab in enumerate(labels):
                ax.axvline(blk["obs"][m], lw=2, label=f"{lab} (p={blk['p'][m]:.3f})")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("mean k-NN distance")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "tier1_knn_scores.png", dpi=130)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-baked", default=None)
    ap.add_argument("--ref-raw", default=None)
    ap.add_argument("--obs", nargs="+", required=True, metavar="LABEL=PATH")
    ap.add_argument("--obs-eb-variant", default=None, help="E group variant if the obs files are RAW (not baked)")
    ap.add_argument("--obs-noise-norm", default="rand")
    ap.add_argument("--exclude-truthkey", nargs="*", default=[],
                    help="mock-as-real: truth-key sidecars whose sim_id must be dropped from the cloud (never printed)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from src.observation.reference import extract_observation, load_reference
    from src.observation.checks.tier1 import Cloud, run_tier1, verdict

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels, paths = [], []
    for tok in args.obs:
        lab, p = tok.split("=", 1)
        labels.append(lab)
        paths.append(p)
    tmp = out_dir / "_obs_vectors.npz"
    extract_observation(paths, str(tmp), eb_variant=args.obs_eb_variant,
                        noise_norm=(args.obs_noise_norm if args.obs_eb_variant else None),
                        want=("bandpowers", "bb", "emap"))
    ov = load_reference(str(tmp))
    obs = {k: ov[k] for k in ("bandpowers", "bb", "emap") if k in ov}
    cloud = Cloud.from_npz(args.ref_baked, args.ref_raw)
    excl = []
    for tk in args.exclude_truthkey:
        with open(tk) as fh:
            excl.append(int(json.load(fh)["sim_id"]))
    res = run_tier1(cloud, obs, labels, seed=args.seed, exclude_sim_ids=excl)
    v = verdict(res, alpha=args.alpha)
    res["verdict"] = v
    res["alpha"] = args.alpha
    with open(out_dir / "tier1_results.json", "w") as fh:
        json.dump(res, fh, default=float)
    with open(out_dir / "tier1_verdict.json", "w") as fh:
        json.dump({"alpha": args.alpha, "verdict": v,
                   "summary": {l: ("PASS" if not f else "FAIL: " + ",".join(f)) for l, f in v.items()}}, fh, indent=2)
    _figures(res, out_dir, labels)
    print("Tier-1 verdict (alpha=%.3g):" % args.alpha)
    for l, f in v.items():
        print(f"  {l}: {'PASS' if not f else 'FAIL ' + ','.join(f)}")
    for key, name in (("twopoint", "2pt"), ("emap", "E-map")):
        if key in res:
            blk = res[key] if key == "twopoint" else res[key]["full"]
            print(f"  {name}: kNN p={['%.3f' % p for p in blk['knn']['p']]} Mahalanobis p={['%.3f' % p for p in blk['mahalanobis']['p']]}")
    if "bmodes" in res:
        print(f"  B-null: PTE={['%.3f' % p for p in res['bmodes']['pte_empirical']]} (vs zero: {['%.3f' % p for p in res['bmodes']['pte_vs_zero_empirical']]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
