#!/usr/bin/env python
"""P6 -- paper figures for the VDQ model-misspecification analysis (F-1 .. F-6).

D7: the 18 pt draw-and-scale notebook convention (`S.context()` default; panels >= 6 in wide,
<= 2 side by side, otherwise stacked), PNG 300 dpi + PDF, saved INSIDE the style context
(`S.save` re-enters it), no suptitles, every string through `S.tex`/`S.label`, `$w$` not
`$w_0$`, the observation = black dashed line drawn last. Ports of production-model-misspec's
plot_misspec_tarp.py, plot_combined_misspec.py, gower_aia_vs_kl.py (+ report detectability
panel), gower_misspec_zscores.py; colours/names from vdq_config.

    python figures.py [--only tarp kl ood aia zscore bg]

Inputs: the local misspec/summaries trees (vdq_config.PACKS) + tables.py json
($OUT_MISSPEC/tables/) + meanp_knn.py npz. Variates = whatever is present locally.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K  # noqa: E402
import vdq_config as C  # noqa: E402

from src.viz import style as S  # noqa: E402

PNG_DPI = 300
PACK_ORDER = ("field", "band")
PANEL_W, PANEL_H = 7.0, 5.2
CHI3_MEAN = 2 * np.sqrt(2 / np.pi)       # E[d] for d ~ chi_3
OBS_LABEL = r"KiDS-Legacy"


def _variates(pk, need_summaries=False):
    v = C.present(pk)
    if need_summaries:
        v = [x for x in v if x in C.present(pk, "summaries")]
    return v


def _tables(pk, kind):
    p = os.path.join(C.OUT_MISSPEC, "tables", f"{pk['key']}_{kind}.json")
    return K.load_json(p) if os.path.exists(p) else None


def _save(fig, name):
    os.makedirs(C.FIG_DIR, exist_ok=True)
    out = S.save(fig, os.path.join(C.FIG_DIR, name), png_dpi=PNG_DPI)
    print("[fig] wrote", ", ".join(out), flush=True)


def _obs_line(ax, x, vertical=True):
    kw = dict(color=C.OBS_COLOUR, ls=C.OBS_LS, lw=2.4, zorder=20)
    (ax.axvline if vertical else ax.axhline)(x, **kw)


def _lw(v):
    return 3.0 if C.is_in_dist(v) else 2.0


# =============================================================================================
# F-1 TARP coverage
# =============================================================================================
def _ecp(pk, v, subset):
    curves, alpha, nparams = [], None, None
    for j in range(len(pk["repeats"])):
        t = K.tarp_ci(pk, v, j)
        node = t["tarp"]["full"] if not subset else t["tarp"]["subsets"].get(K.SUBSET_KEY)
        if node is None:
            return None
        alpha = np.asarray(node["credible_intervals"], float)
        e = np.asarray(node.get("ecp_bootstrap", node.get("ecp")), float)
        curves.append(e.mean(0) if e.ndim == 2 else e)
        nparams = len(t.get("available_params", []))
    c = np.stack(curves)
    return alpha, c.mean(0), c.std(0), nparams


def fig_tarp(subset):
    with S.context():
        fig, axes = plt.subplots(2, 2, figsize=(2 * PANEL_W, PANEL_H * 1.45), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.06, "wspace": 0.22})
        for col, key in enumerate(PACK_ORDER):
            pk = C.PACKS[key]
            ax, axr = axes[0, col], axes[1, col]
            ax.plot([0, 1], [0, 1], color=S.role_colour("reference"), ls="--", lw=1.2, zorder=1)
            axr.axhline(0, color=S.role_colour("reference"), ls="--", lw=1.2, zorder=1)
            for v in _variates(pk):
                r = _ecp(pk, v, subset)
                if r is None:
                    continue
                a, m, s, npar = r
                c = C.colour(v)
                z = 10 if C.is_in_dist(v) else 5
                ax.plot(a, m, color=c, lw=_lw(v), zorder=z, label=S.tex(C.name(v)))
                ax.fill_between(a, m - s, m + s, color=c, alpha=0.18, lw=0, zorder=z - 1)
                axr.plot(a, m - a, color=c, lw=_lw(v), zorder=z)
                axr.fill_between(a, m - a - s, m - a + s, color=c, alpha=0.18, lw=0, zorder=z - 1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(S.tex(pk["title"]), loc="left")
            S.panel_label(ax, "(a)" if col == 0 else "(b)", x=0.04, y=0.96)
            axr.set_xlabel(S.tex("credibility level"))
            if col == 0:
                ax.set_ylabel(S.tex("expected coverage"))
                axr.set_ylabel(S.tex("ECP $-$ level"))
        lim = max(0.05, max(float(np.nanmax(np.abs(a_.get_ylim()))) for a_ in axes[1]))
        for a_ in axes[1]:
            a_.set_ylim(-lim, lim)
        S.legend(axes[0, 1], loc="lower right", fontsize="small")
        _save(fig, "fig_tarp_coverage" + ("_om_s8_w" if subset else "_full"))


# =============================================================================================
# F-2 / F-3 dataset calibration + per-event Mahalanobis vs a misspecification score
# =============================================================================================
def _mahalanobis(pk, v, files):
    order = {f: i for i, f in enumerate(files)}
    acc = np.zeros(len(files))
    n = np.zeros(len(files))
    for j in range(len(pk["repeats"])):
        m = K.moments(pk, v, j)
        names = [str(x) for x in m["params"]]
        i3 = [names.index(p) for p in K.KEY3]
        mf = [K.b(x) for x in m["test_files"]]
        sel = [i for i, f in enumerate(mf) if f in order]
        pos = [order[mf[i]] for i in sel]
        diff = m["theta0"][sel][:, i3] - m["mean"][sel][:, i3]
        cv = m["cov"][sel][:, i3][:, :, i3].astype(float)
        d = np.sqrt(np.einsum("ni,nij,nj->n", diff, np.linalg.inv(cv), diff))
        acc[pos] += d
        n[pos] += 1
    return acc / np.maximum(n, 1)


def _kde_region(ax, x, y, color, logx, frac=0.9):
    lx = np.log10(x) if logx else np.asarray(x, float)
    ly = np.log10(y)
    k = np.isfinite(lx) & np.isfinite(ly)
    lx, ly = lx[k], ly[k]
    if lx.size < 10:
        return
    kde = gaussian_kde(np.vstack([lx, ly]))
    level = np.quantile(kde(np.vstack([lx, ly])), 1 - frac)
    # the OOD axis (-log10 p) is bounded below at 0: do not let the KDE spill into x < 0
    # (and it saturates at the null's resolution, -log10(1/(n_null+1)) = 3.2: no KDE beyond it)
    gx = np.linspace(lx.min() - 0.3 if logx else max(lx.min() - 0.3, 0.0), lx.max() + (0.3 if logx else 0.0), 140)
    gy = np.linspace(ly.min() - 0.3, ly.max() + 0.3, 140)
    GX, GY = np.meshgrid(gx, gy)
    GZ = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    X = 10 ** GX if logx else GX
    ax.contourf(X, 10 ** GY, GZ, levels=[level, GZ.max()], colors=[color], alpha=0.16, zorder=2)
    ax.contour(X, 10 ** GY, GZ, levels=[level], colors=[color], linewidths=1.2, alpha=0.8, zorder=3)


def _xscore(pk, v, axis):
    """(files, per-event score) on the requested axis: 'kl' = adopted cross-repeat KL,
    'ood' = -log10 of the obs-identical recalibrated mean-p (in-dist LOCO)."""
    if axis == "kl":
        f, _s, x = K.kl_event(pk, v, pk["kl"])
        return f, x
    d = K.load_npz(pk["meanp_npz"])
    m = d["variate"] == v
    return np.array([K.b(x) for x in d["test_files"][m]]), -np.log10(d["meanp_knn_recal"][m])


def _obs_x(pk, axis):
    o = K.obs(pk)
    if axis == "kl":
        return float(o[f"kl_{pk['kl']}"])
    return float(-np.log10(o["meanp_recalibrated"]))


def fig_axis(axis):
    logx = axis == "kl"
    with S.context():
        fig, axes = plt.subplots(2, 2, figsize=(2 * PANEL_W, 2 * PANEL_H), sharex="col",
                                 gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.06, "wspace": 0.24})
        for col, key in enumerate(PACK_ORDER):
            pk = C.PACKS[key]
            kt = _tables(pk, "klaxis")
            axc, axm = axes[0, col], axes[1, col]
            vs = _variates(pk) if axis == "kl" else [v for v in _variates(pk, True)]
            gb_x = []
            for v in vs:
                c = C.colour(v)
                files, x = _xscore(pk, v, axis)
                d = _mahalanobis(pk, v, files)
                ok = np.isfinite(x) & np.isfinite(d) & (d > 0) & ((x > 0) if logx else True)
                z = 8 if C.is_in_dist(v) else 4
                axm.scatter(x[ok], d[ok], s=7, alpha=0.12, color=c, lw=0, zorder=z, rasterized=True)
                _kde_region(axm, x[ok], d[ok], c, logx)
                # dataset level (upper): cal over (Om, s8, w), mean +- sd over repeats, at the mean score
                if kt and v in kt["variates"]:
                    cal = kt["variates"][v]["cal"]
                    xm = float(np.mean(x[ok]))
                    lo, hi = np.percentile(x[ok], [2.275, 97.725])
                    axc.errorbar(xm, cal["cal3_mean"], yerr=cal["cal3_sd"], xerr=[[max(xm - lo, 0)], [max(hi - xm, 0)]],
                                 fmt="D", ms=11, mec="white", mew=1.2, color=c, ecolor=c, elinewidth=1.6, capsize=4,
                                 zorder=10 if C.is_in_dist(v) else 6)
                    if v in C.GB_PAIR:
                        gb_x += [lo, hi]
            # matched floor for the gb slices (40 cos x 2): mean +- 2 sd and the 95th percentile
            if kt and gb_x:
                fl = next((f for k_, f in kt["floors"].items() if k_.startswith("matched_")), None)
                if fl is not None:
                    axc.fill_between([min(gb_x), max(gb_x)], max(fl["cal3_mean"] - 2 * fl["cal3_sd"], 1e-4),
                                     fl["cal3_mean"] + 2 * fl["cal3_sd"], color=S.role_colour("mock"), alpha=0.45,
                                     lw=0, zorder=1)
            axm.axhline(CHI3_MEAN, color=S.role_colour("reference"), ls=":", lw=1.4, zorder=5)
            _obs_line(axc, _obs_x(pk, axis))
            _obs_line(axm, _obs_x(pk, axis))
            axc.set_yscale("log")
            lo_, hi_ = axc.get_ylim()
            axc.set_ylim(lo_, hi_ * 6)          # headroom so the panel letter clears the top markers
            axm.set_yscale("log")
            if logx:
                axm.set_xscale("log")
            else:
                axm.set_xlim(-0.05, 3.35)
            axc.set_title(S.tex(pk["title"]), loc="left")
            S.panel_label(axc, "(a)" if col == 0 else "(b)", x=0.96, y=0.94, ha="right")
            if col == 0:
                axc.set_ylabel(S.tex(r"TARP cal.\ error ($\Omega_\mathrm{m}, \sigma_8, w$)"))
                axm.set_ylabel(S.tex(r"Mahalanobis $d$ of truth"))
            if axis == "kl":
                axm.set_xlabel(S.tex("cross-repeat KL" + (r" ($\Omega_\mathrm{m}$-$\sigma_8$ full cov.)"
                                                           if pk["kl"] == "om_s8_fullcov" else " (all parameters)")))
            else:
                axm.set_xlabel(S.tex(r"$-\log_{10}$ recalibrated mean-$p$ (kNN)"))
        handles = [Line2D([], [], color=C.colour(v), marker="D", ls="", ms=9, mec="white", label=S.tex(C.name(v)))
                   for v in C.VARIATES if any(v in _variates(C.PACKS[k]) for k in PACK_ORDER)]
        handles += [Line2D([], [], color=C.OBS_COLOUR, ls=C.OBS_LS, lw=2.4, label=S.tex(OBS_LABEL)),
                    Line2D([], [], color=S.role_colour("reference"), ls=":", lw=1.4, label=S.tex(r"calibrated $\langle d\rangle$ ($\chi_3$)")),
                    Patch(facecolor=S.role_colour("mock"), alpha=0.45, label=S.tex(r"matched floor ($40\times2$, $\pm 2\sigma$)"))]
        S.legend(axes[0, 1], handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")
        _save(fig, "fig_misspec_kl" if axis == "kl" else "fig_misspec_ood")


# =============================================================================================
# F-4 A_IA detectability
# =============================================================================================
def _aia_band(pk):
    names, lo, hi = C.scaler_box(pk["exp"])
    th = K.moments(pk, C.IN_DIST, 0)["theta0"]
    ia, ib = names.index("a_ia"), names.index("b_ia")
    A = K.nla_m_effective(th[:, ia] * (hi[ia] - lo[ia]) + lo[ia], th[:, ib] * (hi[ib] - lo[ib]) + lo[ib])
    return np.percentile(A, [5, 95]), float(np.median(A))


def _aia_of(pk, v, files):
    names, lo, hi = C.scaler_box(pk["exp"])
    m = K.moments(pk, v, 0)
    ia = names.index("a_ia")
    amap = dict(zip([K.b(x) for x in m["test_files"]], m["theta0"][:, ia] * (hi[ia] - lo[ia]) + lo[ia]))
    return np.array([amap.get(f, np.nan) for f in files])


def fig_aia(n_boot=300, half=0.3, fpr=0.05):
    rng = np.random.default_rng(0)
    nla = [v for v in C.NLA_FAMILY]
    with S.context():
        fig, axes = plt.subplots(2, 2, figsize=(2 * PANEL_W, 2 * PANEL_H), sharex=True,
                                 gridspec_kw={"hspace": 0.08, "wspace": 0.22})
        grid = np.linspace(-6, 6, 121)
        any_nla = False
        for row, key in enumerate(PACK_ORDER):
            pk = C.PACKS[key]
            (blo, bhi), bmed = _aia_band(pk)
            ax, axp = axes[row, 0], axes[row, 1]
            for a_ in (ax, axp):
                a_.axvspan(blo, bhi, color="0.88", zorder=0)
                a_.axvline(bmed, color="0.55", lw=1.0, zorder=1)
            _f, _s, k_ind = K.kl_event(pk, C.IN_DIST, pk["kl"])
            thr = float(np.quantile(k_ind, 1 - fpr))
            ax.axhline(thr, color=S.role_colour("reference"), ls=":", lw=1.4, zorder=3)
            dmp = K.load_npz(pk["meanp_npz"]) if os.path.exists(pk["meanp_npz"]) else None
            for v in nla:
                if v not in _variates(pk):
                    continue
                any_nla = True
                c = C.colour(v)
                f, sims, k = K.kl_event(pk, v, pk["kl"])
                A = _aia_of(pk, v, f)
                ok = np.isfinite(A) & np.isfinite(k)
                ax.scatter(A[ok], k[ok], s=6, alpha=0.2, color=c, lw=0, zorder=2, rasterized=True)
                med = np.array([np.median(k[ok][np.abs(A[ok] - g) <= half]) if np.sum(np.abs(A[ok] - g) <= half) >= 20
                                else np.nan for g in grid])
                ax.plot(grid, med, color=c, lw=2.4, zorder=4, label=S.tex(C.name(v)))
                # power curves: KL (solid) and obs-identical mean-p (dashed) at the 5 % in-dist false-alarm rate
                flags = {"kl": (k > thr).astype(float)}
                if dmp is not None and v in set(dmp["variate"].tolist()):
                    mm = dmp["variate"] == v
                    pmap = dict(zip([K.b(x) for x in dmp["test_files"][mm]], dmp["meanp_knn_recal"][mm]))
                    flags["meanp"] = np.array([float(pmap.get(x, np.nan) < fpr) for x in f])
                for det, ls in (("kl", "-"), ("meanp", "--")):
                    if det not in flags:
                        continue
                    fl = flags[det]

                    def curve(idx):
                        Aq, fq = A[idx], fl[idx]
                        return np.array([fq[np.abs(Aq - g) <= half].mean() if np.sum(np.abs(Aq - g) <= half) >= 20
                                         else np.nan for g in grid])
                    c0 = curve(np.arange(len(A)))
                    uniq = np.unique(sims)
                    idx_by = {g: np.flatnonzero(sims == g) for g in uniq}
                    cb = np.array([curve(np.concatenate([idx_by[g] for g in rng.choice(uniq, len(uniq))]))
                                   for _ in range(n_boot)])
                    axp.plot(grid, c0, color=c, ls=ls, lw=2.2, zorder=4)
                    axp.fill_between(grid, np.nanpercentile(cb, 2.5, 0), np.nanpercentile(cb, 97.5, 0),
                                     color=c, alpha=0.12, lw=0, zorder=2)
            axp.axhline(fpr, color=S.role_colour("reference"), ls=":", lw=1.2)
            ax.set_yscale("log")
            axp.set_ylim(0, 1)
            ax.set_ylabel(S.tex("cross-repeat KL"))
            axp.set_ylabel(S.tex("detection power"))
            S.panel_label(ax, f"({'ac'[row]}) " + S.tex(pk["title"]), x=0.03, y=0.95)
            S.panel_label(axp, f"({'bd'[row]})", x=0.03, y=0.95)
        for a_ in axes[1]:
            a_.set_xlabel(S.tex(r"true $A_\mathrm{IA}^\mathrm{total}$"))
            a_.set_xlim(-6, 6)
        hs = [Line2D([], [], color=C.colour(v), lw=2.4, label=S.tex(C.name(v))) for v in nla
              if any(v in _variates(C.PACKS[k]) for k in PACK_ORDER)]
        hs += [Line2D([], [], color="0.3", ls="-", lw=2.2, label=S.tex("KL")),
               Line2D([], [], color="0.3", ls="--", lw=2.2, label=S.tex(r"mean-$p$ (kNN)")),
               Patch(facecolor="0.88", label=S.tex("NLA-M training range (5-95%)")),
               Line2D([], [], color=S.role_colour("reference"), ls=":", lw=1.4, label=S.tex("in-dist. 95th pct. / 5% rate"))]
        S.legend(axes[0, 1], handles=hs, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")
        if not any_nla:
            for a_ in axes.ravel():
                a_.text(0.5, 0.5, S.tex("NLA-family variates not yet fetched"), transform=a_.transAxes,
                        ha="center", va="center", color="0.4")
        _save(fig, "fig_aia_detectability")


# =============================================================================================
# F-5 z-score bias
# =============================================================================================
def fig_zscore():
    params = [("omega_m", "o"), ("sigma_8", "s"), ("S8", "D")]
    with S.context():
        fig, axes = plt.subplots(1, 2, figsize=(2 * PANEL_W, PANEL_H * 1.1), sharey=True, sharex=True,
                                 gridspec_kw={"wspace": 0.08})
        for col, key in enumerate(PACK_ORDER):
            pk = C.PACKS[key]
            kt = _tables(pk, "klaxis")
            ax = axes[col]
            vs = [v for v in C.VARIATES if kt and v in kt["variates"]]
            ys = np.arange(len(C.VARIATES))[::-1]
            ymap = dict(zip(C.VARIATES, ys))
            ax.axvline(0, color=S.role_colour("reference"), ls="--", lw=1.2)
            for v in vs:
                z = kt["variates"][v]["z"]
                for k_, (p, mk) in enumerate(params):
                    e = z[p]
                    y = ymap[v] + (k_ - 1) * 0.22
                    st = "median_z" if "median_z" in e else "mean_z"
                    lo, hi = e[f"{st}_ci95"]
                    ax.errorbar(e[st], y, xerr=[[e[st] - lo], [hi - e[st]]], fmt=mk, ms=8,
                                color=C.colour(v), mec=C.colour(v), mew=1.6,
                                mfc=C.colour(v) if p != "omega_m" else "white", elinewidth=1.6, capsize=0)
            ax.set_yticks(ys)
            ax.set_yticklabels([S.tex(C.name(v)) for v in C.VARIATES])
            ax.set_xlabel(S.tex(r"median $z = (\theta_0 - \mu)/\sigma$"))
            ax.set_title(S.tex(pk["title"]), loc="left")
            S.panel_label(ax, "(a)" if col == 0 else "(b)", x=0.03, y=0.98)
        lim = max(abs(x) for a_ in axes for x in a_.get_xlim())
        axes[0].set_xlim(-lim, lim)
        hs = [Line2D([], [], color="0.25", marker=mk, ls="", ms=8, mew=1.6, mfc="white" if p == "omega_m" else "0.25",
                     label=S.label(p)) for p, mk in params]
        S.legend(axes[1], handles=hs, loc="lower right", fontsize="small")
        _save(fig, "fig_zscore_bias")

    # appendix: z histograms vs N(0,1), rows = params, cols = packs
    xg = np.linspace(-6, 6, 300)
    with S.context():
        fig, axes = plt.subplots(3, 2, figsize=(2 * PANEL_W, 3 * PANEL_H * 0.75), sharex=True,
                                 gridspec_kw={"hspace": 0.1, "wspace": 0.15})
        for col, key in enumerate(PACK_ORDER):
            pk = C.PACKS[key]
            rows = K.rows(pk)
            for r_, (p, _mk) in enumerate(params):
                ax = axes[r_, col]
                ax.plot(xg, np.exp(-xg ** 2 / 2) / np.sqrt(2 * np.pi), color=S.role_colour("reference"), ls="--", lw=1.4)
                for v in C.VARIATES:
                    m = rows["variate"] == v
                    if not m.any():
                        continue
                    z = rows[f"z_{p}"][m]
                    z = z[np.isfinite(z)]          # NOT clipped: overflow would pile into spikes at the edges
                    ax.hist(z, bins=50 if z.size > 2000 else 25, range=(-6, 6), density=False, weights=np.full(z.size, 1.0 / (z.size * 12.0 / (50 if z.size > 2000 else 25))), histtype="step", lw=_lw(v), color=C.colour(v))
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(S.label(p))
                if r_ == 0:
                    ax.set_title(S.tex(pk["title"]), loc="left")
            axes[-1, col].set_xlabel(S.tex(r"$z = (\theta_0 - \mu)/\sigma$"))
        hs = [Line2D([], [], color=C.colour(v), lw=_lw(v), label=S.tex(C.name(v))) for v in C.VARIATES
              if any(v in _variates(C.PACKS[k]) for k in PACK_ORDER)]
        hs.append(Line2D([], [], color=S.role_colour("reference"), ls="--", lw=1.4, label=S.tex(r"$\mathcal{N}(0,1)$")))
        S.legend(axes[0, 1], handles=hs, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")
        _save(fig, "fig_zscore_hist")


# =============================================================================================
# F-6 galaxy-bias slices
# =============================================================================================
def fig_bg():
    """(a) paired b_g = 1.3 minus 1.0 shift in units of the in-dist posterior width (both packs);
    (b) kappa=2: detection power vs per-event b_g severity d; (c) kappa=2: mean z vs d."""
    params = ["omega_m", "sigma_8", "w0"]
    pack_col = {"field": S.role_colour("flagship"), "band": S.arm_colour("band")}
    with S.context():
        fig = plt.figure(figsize=(2 * PANEL_W, 2 * PANEL_H))
        gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.26)
        ax = fig.add_subplot(gs[0, 0])
        have = False
        for k_, key in enumerate(PACK_ORDER):
            t = _tables(C.PACKS[key], "bgia")
            if not t or "gb_pair" not in t:
                continue
            have = True
            for i, p in enumerate(params):
                sm = t["gb_pair"]["summary"][p]
                x = i + (k_ - 0.5) * 0.3
                ax.bar(x, sm["dtheta_over_sigma"], width=0.28, color=pack_col[key], alpha=0.85,
                       label=S.tex(C.PACKS[key]["title"]) if i == 0 else None)
                ax.errorbar(x, sm["dtheta_over_sigma"], yerr=sm["dtheta_sd_over_repeats"] / sm["sigma_post_indist"],
                            color="black", lw=1.4, capsize=3)
        ax.axhline(0, color=S.role_colour("reference"), lw=1.0)
        ax.set_xticks(range(len(params)))
        ax.set_xticklabels([S.label(p) for p in params])
        ax.set_ylabel(S.tex(r"$\Delta\bar\theta / \sigma_\mathrm{post}$"))
        ax.set_title(S.tex(r"$b_g = 1.3$ minus $b_g = 1.0$ (paired)"), loc="left", fontsize="medium")
        S.panel_label(ax, "(a)", x=0.03, y=0.96)
        if have:
            S.legend(ax, loc="upper right", fontsize="small")
        # (b) power vs d, (c) mean z_S8 / z_Om vs d -- kappa=2
        axp = fig.add_subplot(gs[0, 1])
        axz = fig.add_subplot(gs[1, :])
        v = "gower_vdq_gbk2"
        for k_, key in enumerate(PACK_ORDER):
            t = _tables(C.PACKS[key], "bgia")
            b = (t or {}).get("bg_severity", {}).get(v)
            if not b or "bins" not in b:
                continue
            xc = np.array([0.5 * (e["d_lo"] + min(e["d_hi"], 8.0)) for e in b["bins"]]) + (k_ - 0.5) * 0.12
            for det, ls, mk in (("kl", "-", "o"), ("meanp", "--", "s")):
                key_ = f"power_{det}"
                if not all(key_ in e for e in b["bins"]):
                    continue
                y = np.array([e[key_] for e in b["bins"]])
                lo = np.array([e[key_ + "_ci95"][0] for e in b["bins"]])
                hi = np.array([e[key_ + "_ci95"][1] for e in b["bins"]])
                axp.errorbar(xc, y, yerr=[y - lo, hi - y], color=pack_col[key], ls=ls, marker=mk, ms=7, lw=2, capsize=3)
            for p, mk, off in (("S8", "D", 0.0), ("omega_m", "o", 0.06)):
                y = np.array([e[f"mean_z_{p}"] for e in b["bins"]])
                lo = np.array([e[f"mean_z_{p}_ci95"][0] for e in b["bins"]])
                hi = np.array([e[f"mean_z_{p}_ci95"][1] for e in b["bins"]])
                axz.errorbar(xc + off, y, yerr=[y - lo, hi - y], color=pack_col[key], marker=mk, ms=8, lw=0,
                             elinewidth=1.8, capsize=3, mfc=pack_col[key] if p == "S8" else "white", mew=1.6)
        for a_ in (axp, axz):
            a_.axvspan(0, 3.55, color="0.9", zorder=0)
            a_.set_xlim(0, 8.2)
        axp.axhline(0.05, color=S.role_colour("reference"), ls=":", lw=1.2)
        axp.set_ylim(0, 1)
        axp.set_xlabel(S.tex(r"$b_g$ severity $d = |z_{b_g}|$ ($\kappa{=}1$ prior)"))
        axp.set_ylabel(S.tex("detection power"))
        axp.set_title(S.tex(r"$b_g$ prior $\kappa = 2$"), loc="left", fontsize="medium")
        S.panel_label(axp, "(b)", x=0.03, y=0.96)
        axz.axhline(0, color=S.role_colour("reference"), ls="--", lw=1.2)
        axz.set_xlabel(S.tex(r"$b_g$ severity $d = |z_{b_g}|$ ($\kappa{=}1$ prior; shaded: within the $\chi_6$ 95th percentile)"))
        axz.set_ylabel(S.tex(r"mean $z$"))
        S.panel_label(axz, "(c)", x=0.015, y=0.95)
        hs = [Line2D([], [], color=pack_col[k], lw=2.4, label=S.tex(C.PACKS[k]["title"])) for k in PACK_ORDER]
        hs += [Line2D([], [], color="0.3", ls="-", marker="o", lw=2, label=S.tex("KL")),
               Line2D([], [], color="0.3", ls="--", marker="s", lw=2, label=S.tex(r"mean-$p$ (kNN)"))]
        S.legend(axp, handles=hs, loc="upper left", bbox_to_anchor=(0.12, 1.0), fontsize="small")
        hz = [Line2D([], [], color="0.3", marker="D", ls="", ms=8, label=S.label("S8")),
              Line2D([], [], color="0.3", marker="o", ls="", ms=8, mfc="white", mew=1.6, label=S.label("omega_m"))]
        yl = axz.get_ylim()
        axz.set_ylim(yl[0], yl[1] + 0.45 * (yl[1] - yl[0]))     # room for the legend above the points
        S.legend(axz, handles=hz + hs[:2], loc="upper center", fontsize="small", ncol=4)
        _save(fig, "fig_bg_slices")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=["tarp", "kl", "ood", "aia", "zscore", "bg"])
    a = ap.parse_args(argv)
    assert S.probe_tex(), "TeX broken"
    if "tarp" in a.only:
        fig_tarp(False)
        fig_tarp(True)
    if "kl" in a.only:
        fig_axis("kl")
    if "ood" in a.only:
        fig_axis("ood")
    if "aia" in a.only:
        fig_aia()
    if "zscore" in a.only:
        fig_zscore()
    if "bg" in a.only:
        fig_bg()
    return 0


if __name__ == "__main__":
    sys.exit(main())
