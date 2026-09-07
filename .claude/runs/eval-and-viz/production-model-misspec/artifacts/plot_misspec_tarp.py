"""TARP coverage plots for the model-misspecification eval.

Reads the fetched per-variate `misspec_tarp_credible_intervals_<match>.json` files and draws
expected-coverage-probability (ECP) vs credibility-level curves: one curve per variate,
in-distribution (nla_m) highlighted as the reference, y=x = perfect calibration.
Two panels: (a) full available-params TARP, (b) the {omega_m, sigma_8, w0} subset.

Usage:
    python plot_misspec_tarp.py --root ml-checkpoints/gower_npe_finetune_nla_m_z8/misspec \
        --match ncosmo300_0 --out misspec_tarp_coverage.png
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gower_style import COLORS, LABELS, IN_DIST, order_variates  # shared palette
SUBSET_KEY = "sigma_8__omega_m__w0"


def _coverage_xy(node):
    """(alpha, ecp_mean, ecp_std) from a _summarize_coverage-style dict."""
    alpha = np.asarray(node["credible_intervals"], dtype=float)
    ecp = np.asarray(node.get("ecp_bootstrap", node.get("ecp")), dtype=float)
    if ecp.ndim == 1:
        return alpha, ecp, np.zeros_like(ecp)
    return alpha, ecp.mean(axis=0), ecp.std(axis=0)


def load_variates(root, match):
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "*", f"misspec_tarp_credible_intervals_{match}.json"))):
        with open(path) as f:
            payload = json.load(f)
        name = payload.get("variate") or os.path.basename(os.path.dirname(path))
        out[name] = payload
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="misspec/ dir holding <variate>/ subdirs")
    ap.add_argument("--match", default="ncosmo300_0")
    ap.add_argument("--out", default="misspec_tarp_coverage.png")
    ap.add_argument("--title", default="NPE misspecification test — gower_npe_finetune_nla_m_z8 r0, fixed test cosmologies")
    ap.add_argument("--results-root", default=None,
                    help="optional: same tree's misspec_evaluation_results jsons for cal errors")
    args = ap.parse_args()

    variates = load_variates(args.root, args.match)
    if not variates:
        raise SystemExit(f"no misspec_tarp_credible_intervals_{args.match}.json under {args.root}")

    # optional calibration errors for the legend
    cal = {}
    for name in variates:
        rp = os.path.join(args.root, name, f"misspec_evaluation_results_{args.match}.json")
        if os.path.exists(rp):
            with open(rp) as f:
                m = json.load(f)["metrics"]["tarp"]
            cal[name] = {
                "full": m["full"]["calibration_error"],
                "subset": m["subsets"].get(SUBSET_KEY, {}).get("calibration_error"),
            }

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    panels = [
        ("full", "TARP — all available params", axes[0]),
        ("subset", r"TARP — $\{\Omega_m,\ \sigma_8,\ w_0\}$", axes[1]),
    ]
    order = order_variates(list(variates))
    for kind, title, ax in panels:
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="perfect calibration")
        for name in order:
            payload = variates[name]
            tarp = payload.get("tarp", payload)
            node = tarp["full"] if kind == "full" else tarp.get("subsets", {}).get(SUBSET_KEY)
            if node is None:
                continue
            alpha, mean, std = _coverage_xy(node)
            color = COLORS.get(name, None)
            is_ref = name in IN_DIST
            label = LABELS.get(name, name)
            if kind == "full" and payload.get("available_params"):
                label += f" [{len(payload['available_params'])}p]"
            c = cal.get(name, {}).get(kind if kind == "full" else "subset")
            if c is not None:
                label += f"  cal={c:.3f}"
            ax.plot(alpha, mean, color=color, lw=2.4 if is_ref else 1.6,
                    zorder=5 if is_ref else 3, label=label)
            ax.fill_between(alpha, mean - std, mean + std, color=color, alpha=0.18, lw=0)
        ax.set_title(title)
        ax.set_xlabel("credibility level")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("expected coverage probability")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
