"""Combined misspecification figure: dataset-level calibration AND per-event residuals vs the
cross-repeat OOD score, on a shared (log) x axis.

- LEFT y (log):  dataset-averaged TARP calibration error over {omega_m, sigma_8, w0}
                 (mean ± std across repeats), one big marker per variate at x = mean KL.
- RIGHT y (log): per-event Mahalanobis distance of the truth under the 3-D {omega_m, sigma_8,
                 w0} posterior (averaged over repeats, aligned by test file), scattered at
                 x = that event's cross-repeat KL. Faint dots + a shaded 90% KDE region per
                 variate (same colour as its dataset marker).

Reads the per-variate misspec_repeat_disagreement_*.npz (per-event KL + aligned file list),
the per-repeat posterior npz (for Mahalanobis), and the cal-vs-disagreement JSON produced by
offline_repeat_disagreement.py.
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

PARAMS = ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"]
IDX3 = [PARAMS.index(p) for p in ["omega_m", "sigma_8", "w0"]]
CHI3_MEAN = 2 * np.sqrt(2 / np.pi)  # E[d], d ~ chi(3): 2*Gamma(2)/Gamma(1.5) = 2*sqrt(2/pi)
from gower_style import COLORS, LABELS, order_variates  # shared palette


def _mahalanobis_from_moments(path, order, n_out):
    """Mahalanobis from the COMPACT moments npz (mean + full cov), or None if unusable.

    Preferred over the raw sample dump: the moments file is ~KB where the samples are
    hundreds of MB per variate, so the whole figure is reproducible from a small fetch.
    Requires the `cov` field (added to _save_posterior_moments 2026-09-04); moments npz
    written before that carry only the diagonal `std`, so we fall back to the samples.
    """
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=True) as f:
        if "cov" not in f.files:
            return None                      # pre-`cov` npz: diagonal only, cannot do 3-D
        th, mean, cov = f["theta0"], f["mean"], f["cov"]
        files = [str(x) for x in f["test_files"]]
    sel = [i for i, fl in enumerate(files) if fl in order]
    if not sel:
        return None
    pos = [order[files[i]] for i in sel]
    diff = th[sel][:, IDX3] - mean[sel][:, IDX3]                 # [n, 3]
    c = cov[sel][:, IDX3][:, :, IDX3].astype(float)              # [n, 3, 3]
    d = np.sqrt(np.einsum("ni,nij,nj->n", diff, np.linalg.inv(c), diff))
    v = np.full(n_out, np.nan)
    v[pos] = d
    return v


def per_event_mahalanobis(root, name, matches, common_files):
    """Mean-over-repeats Mahalanobis distance of theta0 under the 3-D subset posterior.

    Reads the compact moments npz when it carries `cov`, else recomputes from the full
    posterior samples. Both paths compute the identical statistic.
    """
    d_sum = np.zeros(len(common_files))
    n_rep = 0
    order = {f: i for i, f in enumerate(common_files)}
    for m in matches:
        v = _mahalanobis_from_moments(
            os.path.join(root, name, f"misspec_posterior_moments_{m}.npz"),
            order, len(common_files))
        if v is None:
            p = os.path.join(root, name, f"misspec_posterior_samples_{m}.npz")
            if not os.path.exists(p):
                continue
            with np.load(p, allow_pickle=True) as f:
                th, s = f["theta0s"], f["samples"]
                files = [str(x) for x in f["test_files"]]
            sel = [i for i, fl in enumerate(files) if fl in order]
            pos = [order[files[i]] for i in sel]
            sub = s[:, sel, :][:, :, IDX3]                      # [S, n, 3]
            diff = th[sel][:, IDX3] - sub.mean(axis=0)          # [n, 3]
            # per-event 3x3 covariance + Mahalanobis
            c = np.einsum("sni,snj->nij", sub - sub.mean(0), sub - sub.mean(0)) / (sub.shape[0] - 1)
            d = np.sqrt(np.einsum("ni,nij,nj->n", diff, np.linalg.inv(c), diff))
            v = np.full(len(common_files), np.nan)
            v[pos] = d
        d_sum += np.nan_to_num(v)
        n_rep += 1
    return d_sum / max(n_rep, 1)


def kde_region(ax, x, y, color, frac=0.90):
    """Shade the (log-log) KDE region containing `frac` of the points."""
    lx, ly = np.log10(x), np.log10(y)
    k = np.isfinite(lx) & np.isfinite(ly)
    lx, ly = lx[k], ly[k]
    kde = gaussian_kde(np.vstack([lx, ly]))
    dens = kde(np.vstack([lx, ly]))
    level = np.quantile(dens, 1 - frac)
    gx = np.linspace(lx.min() - 0.4, lx.max() + 0.4, 160)
    gy = np.linspace(ly.min() - 0.4, ly.max() + 0.4, 160)
    GX, GY = np.meshgrid(gx, gy)
    GZ = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    ax.contourf(10 ** GX, 10 ** GY, GZ, levels=[level, GZ.max()], colors=[color], alpha=0.16)
    ax.contour(10 ** GX, 10 ** GY, GZ, levels=[level], colors=[color], linewidths=1.0, alpha=0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--summary-json", required=True,
                    help="the *_cal_vs_disagreement.json from offline_repeat_disagreement.py")
    ap.add_argument("--matches", nargs="+", default=["_0", "_1", "_2", "_3", "_4"])
    ap.add_argument("--out", default="misspec_combined.png")
    ap.add_argument("--title", default="Misspecification vs repeat-ensemble OOD score — "
                                       "GLASS foundation on GLASS variates")
    args = ap.parse_args()

    with open(args.summary_json) as f:
        table = json.load(f)

    fig, axL = plt.subplots(figsize=(9.0, 6.2))
    axR = axL.twinx()

    indist_cal = next((r["cal_sub_mean"] for n, r in table.items()
                       if n in ("glass_nla_m", "nla_m")), 8e-3)

    for name, r in table.items():
        color = COLORS.get(name, "tab:gray")
        kl = None
        # per-event cloud (right axis)
        kd = sorted(glob.glob(os.path.join(args.root, name, "misspec_repeat_disagreement_*.npz")))
        if kd:
            with np.load(kd[-1], allow_pickle=True) as f:
                kl = np.asarray(f["kl_score"], dtype=float)
                files = [str(x) for x in f["test_files"]]
            d = per_event_mahalanobis(args.root, name, args.matches, files)
            k = np.isfinite(kl) & np.isfinite(d) & (kl > 0) & (d > 0)
            axR.scatter(kl[k], d[k], s=6, alpha=0.12, color=color, lw=0, zorder=2)
            kde_region(axR, kl[k], d[k], color)
        # dataset-level marker (left axis); x-spread = the 2-sigma quantiles of the per-event
        # KL distribution (2.275% / 97.725%)
        if kl is not None and np.isfinite(kl).sum():
            lo, hi = np.nanpercentile(kl, [2.275, 97.725])
            xerr = [[max(r["kl_mean"] - lo, 0.0)], [max(hi - r["kl_mean"], 0.0)]]
        else:
            xerr = None
        axL.errorbar(r["kl_mean"], r["cal_sub_mean"], yerr=r["cal_sub_std"], xerr=xerr,
                     fmt="D", ms=11, mec="white", mew=1.2, color=color, capsize=4,
                     elinewidth=1.6, zorder=10)

    axR.axhline(CHI3_MEAN, color="k", ls=":", lw=1.1, alpha=0.7)
    axR.text(0.985, CHI3_MEAN * 1.08, r"calibrated $\mathbb{E}[d_{\rm Mah}]$ ($\chi_3$)",
             transform=axR.get_yaxis_transform(), ha="right", fontsize=8, alpha=0.8)

    axL.set_xscale("log"); axL.set_yscale("log"); axR.set_yscale("log")

    # Align the two log y-axes via two anchors: the calibrated chi_3 mean sits at the in-dist
    # diamond's calibration error, and Mahalanobis = 10 sits at calibration error = 5 (so the
    # OOD diamonds land on the tail of their event clouds). log10(cal) = a*log10(d) + b.
    a = (np.log10(5.0) - np.log10(indist_cal)) / (np.log10(10.0) - np.log10(CHI3_MEAN))
    b = np.log10(5.0) - a * np.log10(10.0)
    L0, L1 = 1e-3, 20.0
    axL.set_ylim(L0, L1)
    axR.set_ylim(10 ** ((np.log10(L0) - b) / a), 10 ** ((np.log10(L1) - b) / a))
    axL.set_xlabel("cross-repeat posterior disagreement (sym. KL)\n"
                   "diamonds: dataset mean KL — dots: per-event KL")
    axL.set_ylabel(r"dataset TARP calibration error $\{\Omega_m,\sigma_8,w_0\}$   (diamonds)")
    axR.set_ylabel(r"per-event Mahalanobis $d(\theta_0\,|\,$3-D posterior$)$   (dots + 90% KDE region)")

    handles = [mlines.Line2D([], [], color=COLORS.get(n, "gray"), marker="D", ls="",
                             ms=9, mec="white", label=LABELS.get(n, n)) for n in table]
    handles.append(mlines.Line2D([], [], color="k", ls=":", label=r"calibrated $\chi_3$ mean"))
    handles.append(mpatches.Patch(facecolor="0.6", alpha=0.25, label="90% of events (KDE)"))
    axL.legend(handles=handles, fontsize=8.5, loc="upper left", frameon=False)
    axL.set_title(args.title, fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=170)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
