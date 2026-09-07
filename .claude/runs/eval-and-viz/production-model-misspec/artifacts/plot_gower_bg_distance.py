"""Figures for the galaxy-bias severity axis.

Panel (a) severity distribution: how far each mock's per-bin galaxy bias sits from the
    kappa=1 NLA-M training prior, d = sqrt(sum_i z_i^2) over the six tomographic bins, for the
    in-distribution store and the kappa=2 store, against the chi_6 null.
Panel (b) which BINS leave support: the per-bin |z| distribution. The training prior is
    truncated at +-3 sigma, so |z| > 3 means the mock is outside the support the network ever
    saw in that bin.
Panel (c) PILOT ONLY -- detector response against severity for the handful of test mocks whose
    b_g could be recovered. This is NOT the b_g analogue of `gower_aia_vs_kl.png`; that figure
    needs per-event b_g for all 1592 test mocks and the gatekeeper read is bounded at 200
    records (~21 test mocks, since b_g is drawn per (sim, outer, rot) block and the test set is
    rot0 only). The panel is drawn so the reader can see there is nothing to see at this n,
    and is labelled accordingly. Do not quote a trend from it.
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MU = np.array([1.0181, 1.0698, 1.1302, 1.2427, 1.3739, 1.4805])
SIGMA = np.array([0.1801, 0.1491, 0.1252, 0.0951, 0.0960, 0.0985])
TRUNC = 3.0
C_IND, C_K2 = "black", "#56B4E9"


def load_records(p):
    with open(p) as f:
        return json.load(f)["records"]


def severity(recs):
    B = np.array([[r[f"b_g_bin{i}"] for i in range(1, 7)] for r in recs], float)
    Z = (B - MU) / SIGMA
    return B, Z, np.sqrt((Z ** 2).sum(1)), np.abs(Z).max(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indist", required=True)
    ap.add_argument("--kappa2", required=True)
    ap.add_argument("--out", default="gower_bg_distance.png")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    ri, rk = load_records(args.indist), load_records(args.kappa2)
    Bi, Zi, di, mi = severity(ri)
    Bk, Zk, dk, mk = severity(rk)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # ---- (a) severity distribution --------------------------------------------------
    ax = axes[0]
    bins = np.linspace(0, 11, 45)
    ax.hist(di, bins=bins, density=True, color=C_IND, alpha=0.40, label=f"in-distribution ($\\kappa=1$), n={len(di)}")
    ax.hist(dk, bins=bins, density=True, color=C_K2, alpha=0.60, label=f"$\\kappa=2$ prior, n={len(dk)}")
    x = np.linspace(0.01, 11, 400)
    ax.plot(x, chi.pdf(x, df=6), "k--", lw=1.4, label=r"$\chi_6$ null (training prior)")
    ax.axvline(np.median(di), color=C_IND, lw=1.2, ls=":")
    ax.axvline(np.median(dk), color=C_K2, lw=1.6, ls=":")
    ax.set_xlabel(r"galaxy-bias severity  $d=\sqrt{\sum_i z_i^2}$   (6 tomographic bins)")
    ax.set_ylabel("density")
    ax.set_title("(a) how far $b_g$ sits from the training prior")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    ax.text(0.97, 0.55, f"median  {np.median(di):.2f} $\\to$ {np.median(dk):.2f}",
            transform=ax.transAxes, ha="right", fontsize=9)

    # ---- (b) per-bin |z|: which bins leave support -----------------------------------
    ax = axes[1]
    pos = np.arange(1, 7)
    for Z, c, lab, off in ((Zi, C_IND, "in-distribution", -0.17), (Zk, C_K2, r"$\kappa=2$", +0.17)):
        q = np.abs(Z)
        med = np.median(q, axis=0)
        lo, hi = np.percentile(q, 10, axis=0), np.percentile(q, 90, axis=0)
        ax.errorbar(pos + off, med, yerr=[med - lo, hi - med], fmt="o", color=c,
                    capsize=3, lw=1.6, ms=5, label=lab)
    ax.axhline(TRUNC, color="crimson", lw=1.4, ls="--",
               label=r"training support edge ($|z|=3$)")
    ax.set_xticks(pos)
    ax.set_xlabel("tomographic bin")
    ax.set_ylabel(r"$|z_i|$  against the training prior")
    ax.set_title("(b) which bins leave support (median, 10–90%)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fo_i, fo_k = float(np.mean(mi > TRUNC)), float(np.mean(mk > TRUNC))
    ax.set_ylim(0, 4.3)
    ax.text(0.03, 0.97, f"ANY bin outside support:   in-dist {fo_i:.1%}   $\\kappa=2$ {fo_k:.1%}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(fc="white", ec="0.7", lw=0.6, pad=2.5))

    # ---- (c) pilot: detector vs severity ---------------------------------------------
    ax = axes[2]
    test_k = [r for r in rk if r.get("rot") == 0 and r.get("cat") in (0, 1)]
    if test_k:
        EXP = ("/home/alex/work/glass_gower_transfer/ml-checkpoints/"
               "gower_npe_finetune_nla_m_bgp_z8_ens1")
        tag = "_".join(f"ncosmo300_{i}" for i in range(5))
        with np.load(f"{EXP}/misspec/gower_gbk2/misspec_repeat_disagreement_{tag}.npz",
                     allow_pickle=True) as f:
            klmap = dict(zip(f["test_files"].tolist(), np.asarray(f["kl_score"], float)))
        _, _, dt, _ = severity(test_k)
        kl = np.array([klmap.get(r["file"], np.nan) for r in test_k])
        k = np.isfinite(kl)
        ax.scatter(dt[k], kl[k], s=34, color=C_K2, edgecolor="k", lw=0.4, zorder=3)
        ax.axhline(float(np.median(list(klmap.values()))), color="0.4", ls="-", lw=1.2,
                   label=r"$\kappa=2$ suite median KL")
        ax.axvline(TRUNC, color="crimson", lw=1.2, ls="--")
        if k.sum() >= 8:
            from scipy.stats import spearmanr
            rho = spearmanr(dt[k], kl[k]).statistic
            ax.text(0.04, 0.93, rf"Spearman $\rho={rho:+.2f}$   (n={int(k.sum())})",
                    transform=ax.transAxes, fontsize=9, va="top")
        ax.set_yscale("log")
    ax.set_xlabel(r"galaxy-bias severity $d$")
    ax.set_ylabel("per-event cross-repeat KL")
    ax.set_title("(c) PILOT ONLY — not a measurement", color="crimson")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(alpha=0.25, lw=0.5)
    ax.text(0.03, 0.06,
            "needs per-event $b_g$ for all 1592 test mocks;\n"
            "gatekeeper read caps at 200 records",
            transform=ax.transAxes, ha="left", fontsize=8, color="crimson",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=2.0))

    fig.suptitle(r"Galaxy-bias severity: $\kappa=2$ is far outside training support, "
                 r"yet no detector separates it (KL 0.537, kNN 0.499)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}")
    print(f"  in-dist  median d={np.median(di):.2f}  any-bin-outside={fo_i:.1%}")
    print(f"  kappa=2  median d={np.median(dk):.2f}  any-bin-outside={fo_k:.1%}")
    print(f"  panel (c) pilot n={len(test_k)} test mocks -- labelled as not a measurement")


if __name__ == "__main__":
    main()
