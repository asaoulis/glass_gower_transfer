"""Per-event A_IA vs misspecification signal, for the a_ia-misspec variates (nla, nla_z).

Panel per variate: scatter of TRUE A_IA (physical, reconstructed from the scaled theta0s via
the nla_m preset box) vs |z| of A_IA (|theta0 - mu_post| / sigma_post, affine-invariant so
computed in scaled space). Overlays: binned running mean +- 1sigma band, the NLA-M training
support [4.48, 7] shaded, E|z| = sqrt(2/pi) reference for a calibrated posterior, and the
in-dist nla_m cloud as the calibrated baseline inside the support.

Optional --kl-mode: x = TRUE A_IA, y = per-event cross-repeat disagreement (kl_score from
misspec_repeat_disagreement_*.npz, joined on test-file basename) — the direct OOD proxy from
the repeat ensemble (no reverse-engineering from z-scores). Use once multi-repeat outputs land.

NOTE: events whose posterior was intractable to sample (n_dropped_nonfinite; ~25-28% for
nla/nla_z, plausibly the most extreme A_IA) are ABSENT from the npz and hence from this plot —
the visible trend is a lower bound on the far-OOD behaviour.
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AIA_BOX = (4.48, 7.0)          # nla_m preset min/max — the training support
PARAMS = ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"]
J_AIA = PARAMS.index("a_ia")
from gower_style import COLORS  # shared palette
CAL_REF = np.sqrt(2 / np.pi)   # E|z| for a calibrated (unit-normal) residual


def load_events(root, name, match):
    p = os.path.join(root, name, f"misspec_posterior_samples_{match}.npz")
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=True) as f:
        th, s = f["theta0s"], f["samples"]
        files = [str(x) for x in f["test_files"]] if "test_files" in f else None
    mu, sd = s.mean(axis=0), s.std(axis=0)
    z = (th - mu) / np.maximum(sd, 1e-12)
    aia_phys = th[:, J_AIA] * (AIA_BOX[1] - AIA_BOX[0]) + AIA_BOX[0]
    return {"aia": aia_phys, "z_aia": z[:, J_AIA], "files": files}


def load_kl(root, name):
    hits = sorted(glob.glob(os.path.join(root, name, "misspec_repeat_disagreement_*.npz")))
    if not hits:
        return None
    with np.load(hits[-1], allow_pickle=True) as f:
        return {str(x): float(k) for x, k in zip(f["test_files"], f["kl_score"])}


def binned_stats(x, y, nbins=18):
    edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
    edges = np.unique(edges)
    centers, means, stds = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x <= hi)
        if m.sum() < 5:
            continue
        centers.append(0.5 * (lo + hi))
        means.append(y[m].mean())
        stds.append(y[m].std())
    return np.array(centers), np.array(means), np.array(stds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--match", default="ncosmo300_0")
    ap.add_argument("--variates", nargs="+", default=["nla", "nla_z"])
    ap.add_argument("--indist", default="nla_m")
    ap.add_argument("--out", default="misspec_aia_vs_z.png")
    ap.add_argument("--kl-mode", action="store_true",
                    help="y = per-event cross-repeat KL disagreement instead of |z_aia|")
    args = ap.parse_args()

    ref = load_events(args.root, args.indist, args.match)
    fig, axes = plt.subplots(1, len(args.variates), figsize=(6.0 * len(args.variates), 4.8),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, args.variates):
        d = load_events(args.root, name, args.match)
        if d is None:
            ax.set_title(f"{name} (no data)")
            continue
        if args.kl_mode:
            klmap = load_kl(args.root, name)
            if klmap is None or d["files"] is None:
                ax.set_title(f"{name} (no disagreement npz yet)")
                continue
            y = np.array([klmap.get(f, np.nan) for f in d["files"]])
            keep = np.isfinite(y)
            x, y = d["aia"][keep], y[keep]
            ylabel = "cross-repeat disagreement (sym. KL)"
            ax.set_yscale("log")
        else:
            x, y = d["aia"], np.abs(d["z_aia"])
            ylabel = r"$|z_{A_{\rm IA}}|$"

        ax.axvspan(*AIA_BOX, color="0.85", zorder=0,
                   label=f"NLA-M training support [{AIA_BOX[0]}, {AIA_BOX[1]}]")
        c = COLORS.get(name, "tab:orange")
        ax.scatter(x, y, s=9, alpha=0.35, color=c, lw=0, zorder=2)
        bc, bm, bs = binned_stats(x, y)
        ax.plot(bc, bm, color=c, lw=2.4, zorder=4, label=f"{name}: binned mean")
        ax.fill_between(bc, bm - bs, bm + bs, color=c, alpha=0.18, lw=0, zorder=3)

        if not args.kl_mode and ref is not None:
            ax.scatter(ref["aia"], np.abs(ref["z_aia"]), s=7, alpha=0.25, color="k",
                       lw=0, zorder=2, label=f"{args.indist} (in-dist)")
            ax.axhline(CAL_REF, color="k", ls="--", lw=1.0, alpha=0.6,
                       label=r"calibrated $\mathbb{E}|z|=\sqrt{2/\pi}$")

        ax.set_xlabel(r"true $A_{\rm IA}$")
        ax.set_title(name)
        ax.legend(fontsize=8, loc="upper right", frameon=False)
    axes[0].set_ylabel(ylabel)
    kind = "cross-repeat OOD score" if args.kl_mode else "posterior residual"
    fig.suptitle(f"Per-event $A_{{\\rm IA}}$ vs {kind} — intractable events "
                 "(~25-28%) excluded (lower bound at extreme $A_{{\\rm IA}}$)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out, dpi=170)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
