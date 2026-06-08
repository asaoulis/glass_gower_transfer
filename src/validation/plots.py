"""Headless diagnostic plots for the theory-vs-empirical bandpower comparison.

Lifted (and tidied) from ``sims_analysis.ipynb``.  All plotting is savefig-only
(``Agg`` backend) so it runs on a SLURM CPU node with no display.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config as cfg  # noqa: E402


def _shade_thresholds(ax, ok_frac: float, warn_frac: float):
    """Shade the +/-OK (green) and +/-WARN (amber) bands and the unity line."""
    ax.axhspan(1 - ok_frac, 1 + ok_frac, color="green", alpha=0.10, zorder=0)
    ax.axhspan(1 + ok_frac, 1 + warn_frac, color="orange", alpha=0.08, zorder=0)
    ax.axhspan(1 - warn_frac, 1 - ok_frac, color="orange", alpha=0.08, zorder=0)
    ax.axhline(1.0, color="k", lw=1, ls="--", alpha=0.7, zorder=1)


def plot_bandpowers_loglog(cll_bands, cl_bandpowers, theory_bandpowers, nbins, ylim=None):
    """Log-log overlay of empirical and theory bandpowers in a nbins x nbins grid."""
    fig, ax = plt.subplots(nbins, nbins, figsize=(12, 12), sharex=True, sharey=True)
    for i in range(nbins):
        for j in range(nbins):
            if i < j:
                ax[i, j].axis("off")
                continue
            lbl = f"S{i + 1}-S{j + 1}"
            idx = int(i * (i + 1) / 2 + j)
            if theory_bandpowers is not None:
                ax[i, j].loglog(cll_bands, theory_bandpowers[lbl], lw=2, color="k",
                                label="Theory")
            ax[i, j].loglog(cll_bands, cl_bandpowers[idx], marker="x", ls="none", ms=4,
                            color="cornflowerblue", label="Empirical")
            ax[i, j].text(0.05, 0.95, lbl, transform=ax[i, j].transAxes, fontsize=9,
                          ha="left", va="top")
            if ylim is not None:
                ax[i, j].set_ylim(*ylim)
            if i == nbins - 1:
                ax[i, j].set_xlabel(r"$\ell$")
            if j == 0:
                ax[i, j].set_ylabel(r"$C_\ell$")
    handles, labels = ax[nbins - 1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=11)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    return fig, ax


def plot_ratio_quantiles(cls, ratios, labels, nbins, ratio_ylim=(0.7, 1.3),
                         ok_frac=cfg.OK_FRAC, warn_frac=cfg.WARN_FRAC):
    """Median + 16-84% / 10-90% ratio whiskers per spectrum, with threshold bands."""
    q16, q50, q84 = np.percentile(ratios, [16, 50, 84], axis=0)
    q10, q90 = np.percentile(ratios, [10, 90], axis=0)

    fig, ax = plt.subplots(nbins, nbins, figsize=(15, 15), sharex=True, sharey=True)
    for i in range(nbins):
        for j in range(nbins):
            if i < j:
                ax[i, j].axis("off")
                continue
            idx = int(i * (i + 1) / 2 + j)
            ax[i, j].set_xlim(cls[0] * 0.9, cls[-1] * 1.1)
            _shade_thresholds(ax[i, j], ok_frac, warn_frac)
            ax[i, j].errorbar(cls, q50[idx],
                              yerr=[q50[idx] - q16[idx], q84[idx] - q50[idx]],
                              fmt="x", ms=4, lw=1.5, label="16-84%")
            ax[i, j].errorbar(cls, q50[idx],
                              yerr=[q50[idx] - q10[idx], q90[idx] - q50[idx]],
                              fmt="none", ecolor="k", alpha=0.5, lw=1, label="10-90%")
            ax[i, j].text(0.05, 0.9, labels[idx], transform=ax[i, j].transAxes, fontsize=10)
            ax[i, j].set_xscale("log")
            ax[i, j].set_ylim(*ratio_ylim)
            if i == nbins - 1:
                ax[i, j].set_xlabel(r"$\ell$")
            if j == 0:
                ax[i, j].set_ylabel(r"$C_\ell^{\rm emp}/C_\ell^{\rm th}$")
    plt.tight_layout()
    return fig, ax


def plot_ratio_ensemble_grey(cls, ratios, labels, nbins, alpha=0.05, color="0.3",
                             ratio_ylim=(0.7, 1.3), ok_frac=cfg.OK_FRAC,
                             warn_frac=cfg.WARN_FRAC):
    """Low-alpha grey spaghetti of every realisation's ratio, with threshold bands."""
    fig, ax = plt.subplots(nbins, nbins, figsize=(15, 15), sharex=True, sharey=True)
    for i in range(nbins):
        for j in range(nbins):
            if i < j:
                ax[i, j].axis("off")
                continue
            idx = int(i * (i + 1) / 2 + j)
            _shade_thresholds(ax[i, j], ok_frac, warn_frac)
            for r in ratios[:, idx, :]:
                ax[i, j].plot(cls, r, color=color, alpha=alpha, lw=1)
            ax[i, j].text(0.05, 0.9, labels[idx], transform=ax[i, j].transAxes, fontsize=10)
            ax[i, j].set_xscale("log")
            ax[i, j].set_ylim(*ratio_ylim)
            if i == nbins - 1:
                ax[i, j].set_xlabel(r"$\ell$")
            if j == 0:
                ax[i, j].set_ylabel(r"$C_\ell^{\rm emp}/C_\ell^{\rm th}$")
    plt.tight_layout()
    return fig, ax


def save_all_plots(
    ensemble: dict,
    out_dir: str,
    nbins: int = cfg.NBINS,
    ratio_ylim=(0.7, 1.3),
    ok_frac: float = cfg.OK_FRAC,
    warn_frac: float = cfg.WARN_FRAC,
    example_theory: Optional[dict] = None,
    example_empirical: Optional[Sequence] = None,
    dpi: int = 200,
) -> list:
    """Write the standard plot set into ``out_dir``; return the list of files written."""
    os.makedirs(out_dir, exist_ok=True)
    cls, ratios, labels = ensemble["cls"], ensemble["ratios"], ensemble["labels"]
    written = []

    fig, _ = plot_ratio_quantiles(cls, ratios, labels, nbins, ratio_ylim, ok_frac, warn_frac)
    p = os.path.join(out_dir, "bandpower_ratios_quantiles.png")
    fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig); written.append(p)

    fig, _ = plot_ratio_ensemble_grey(cls, ratios, labels, nbins,
                                      ratio_ylim=ratio_ylim, ok_frac=ok_frac, warn_frac=warn_frac)
    p = os.path.join(out_dir, "bandpower_ratios_ensemble_grey.png")
    fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig); written.append(p)

    if example_theory is not None and example_empirical is not None:
        fig, _ = plot_bandpowers_loglog(cls, example_empirical, example_theory, nbins)
        p = os.path.join(out_dir, "bandpowers_loglog_example.png")
        fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig); written.append(p)

    return written
