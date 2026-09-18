"""The stacked S1xS1 ... S6xS6 bandpower corner: where an observation's 2-pt statistics sit
inside the mock population.

This is the companion to the Tier-1a 21x8 robust-z heat map (figure
``03_tier1a_bandpowers_vs_cloud``). Same data, same cloud, same ordering -- the heat map
compresses each (spectrum, band) to one number and shows the whole grid at a glance; this
shows the actual C_ell with the population spread, so the reader can see *how* the observation
sits rather than only *how far*.

Layout and index convention are taken verbatim from ``src/validation/plots.py``
(``plot_ratio_quantiles`` / ``plot_bandpowers_loglog``), which is the repo's existing 6x6
lower-triangle theory-comparison figure::

    panel (i, j) for i >= j   <->   spectrum index  idx = i*(i+1)/2 + j   <->   label "S{i+1}-S{j+1}"

That is the same flattening as ``cls_results/full/mixed_bandpowers`` (21, 8) and as
``src/observation/checks/tier1.py:spectrum_labels``, so panel, heat-map row and stored
spectrum always refer to the same tomographic pair.

Deliberate deviation from ``plot_bandpowers_loglog``: **not** log-log. Cross-bin EE bandpowers
from a noisy mock cloud go negative at low ell, which a log axis silently drops. We plot
``ell(ell+1) C_ell / 2pi`` on a linear axis (or symlog via ``yscale``), so a negative excursion
is visible rather than missing.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.viz import style

NBINS = 6
N_SPECTRA = 21


def spectrum_index(i: int, j: int) -> int:
    """Flat spectrum index of tomographic panel (i, j), i >= j (0-based)."""
    if j > i:
        raise ValueError(f"lower triangle only: got (i={i}, j={j})")
    return int(i * (i + 1) / 2 + j)


def spectrum_label(i: int, j: int) -> str:
    return f"S{i + 1}-S{j + 1}"


def plot_bandpower_corner(
    ell: Sequence[float],
    cloud_bp: np.ndarray,
    obs_bp: np.ndarray,
    *,
    obs_labels: Sequence[str] | None = None,
    obs_colours: Sequence[str] | None = None,
    nbins: int = NBINS,
    bands: Sequence[float] = (2.5, 16, 84, 97.5),
    scale_ell: bool = True,
    normalise: str = "median",
    yscale: str = "linear",
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (13.0, 11.0),
    palette: str = style.DEFAULT_PALETTE,
    font_size: float = 11.0,
    title: str | None = None,
):
    """Draw the 6x6 lower-triangle bandpower corner.

    Parameters
    ----------
    ell
        The ``nb`` bandpower centres (``cls_results/full/bandpower_ls``).
    cloud_bp
        ``[N, 21, nb]`` mock-population bandpowers (the Tier-1 reference cloud).
    obs_bp
        ``[M, 21, nb]`` observation bandpowers -- one row per label.
    obs_labels
        One legend entry per row of ``obs_bp``.
    bands
        Percentile pairs for the cloud envelope, outermost first. Default draws a
        2.5-97.5 outer band and a 16-84 inner band.
    scale_ell
        Plot ``ell(ell+1) C_ell / 2pi`` instead of raw ``C_ell``. Ignored when
        ``normalise='median'`` (the ratio cancels any ell-only weight).
    normalise
        ``'median'`` (default) plots ``C_ell / median(C_ell^cloud)`` per (spectrum, band).
        This is the readable form for "where does the observation sit in the population":
        the steep ell-dependence divides out, every panel shares one y range, and a
        deviation is legible at low ell instead of being squashed against zero. ``'none'``
        plots the spectra themselves, which looks like the conventional data-vs-theory
        figure but hides the low-ell bands.

    Returns the ``(fig, axes)`` pair; the caller saves it with ``style.save`` INSIDE a
    ``style.context`` (see the style guide -- a savefig after the context exits silently
    re-renders every word in CM Sans).
    """
    import matplotlib.pyplot as plt

    ell = np.asarray(ell, dtype=float)
    cloud_bp = np.asarray(cloud_bp, dtype=float)
    obs_bp = np.atleast_3d(np.asarray(obs_bp, dtype=float))
    if obs_bp.ndim == 2:                       # a single observation passed as [21, nb]
        obs_bp = obs_bp[None]
    if cloud_bp.shape[1] != N_SPECTRA or obs_bp.shape[1] != N_SPECTRA:
        raise ValueError(f"expected {N_SPECTRA} spectra, got cloud {cloud_bp.shape} obs {obs_bp.shape}")
    if len(ell) != cloud_bp.shape[2]:
        raise ValueError(f"{len(ell)} ell values but {cloud_bp.shape[2]} bandpowers")

    if len(bands) % 2:
        raise ValueError("`bands` must be percentile PAIRS")
    pairs = [(bands[k], bands[-1 - k]) for k in range(len(bands) // 2)]

    obs_labels = list(obs_labels) if obs_labels is not None else [f"obs {k}" for k in range(len(obs_bp))]
    if obs_colours is None:
        cyc = style.palette(palette)
        obs_colours = [cyc[k % len(cyc)] for k in range(len(obs_bp))]

    if normalise not in ("median", "none"):
        raise ValueError(f"normalise must be 'median' or 'none', got {normalise!r}")
    if normalise == "median":
        # Per (spectrum, band) cloud median. Tiny |median| would blow the ratio up; those bands
        # carry no information anyway, so guard rather than divide by ~0.
        med = np.median(cloud_bp, axis=0)
        scale = np.where(np.abs(med) > 1e-30, med, np.nan)
        ylab = r"$C_\ell\,/\,\mathrm{median}(C_\ell^{\rm mock})$"
    else:
        w = ell * (ell + 1) / (2 * np.pi) if scale_ell else np.ones_like(ell)
        ylab = (r"$\ell(\ell+1)C_\ell^{EE}/2\pi$" if scale_ell else r"$C_\ell^{EE}$")

    cloud_ref = style.role_colour("reference", palette)

    fig, ax = plt.subplots(nbins, nbins, figsize=figsize, sharex=True,
                           sharey=(normalise == "median"))
    for i in range(nbins):
        for j in range(nbins):
            a = ax[i, j]
            if i < j:
                a.axis("off")
                continue
            s = spectrum_index(i, j)
            if normalise == "median":
                cl = cloud_bp[:, s, :] / scale[s]
                obs_s = obs_bp[:, s, :] / scale[s]
                a.axhline(1.0, color="0.55", lw=0.6, ls=":", zorder=0)
            else:
                cl = cloud_bp[:, s, :] * w
                obs_s = obs_bp[:, s, :] * w
                a.axhline(0.0, color="0.6", lw=0.5, zorder=0)

            # cloud envelope, widest band faintest
            for k, (lo, hi) in enumerate(pairs):
                qlo, qhi = np.percentile(cl, [lo, hi], axis=0)
                a.fill_between(ell, qlo, qhi, color=cloud_ref, lw=0,
                               alpha=0.18 + 0.22 * k,
                               label=(style.tex(f"mock cloud {lo:g}-{hi:g}%") if (i, j) == (nbins - 1, 0) else None))
            a.plot(ell, np.median(cl, axis=0), color=cloud_ref, lw=1.1, ls="--",
                   label=(style.tex("mock median") if (i, j) == (nbins - 1, 0) else None))

            for m in range(len(obs_bp)):
                a.plot(ell, obs_s[m], marker="o", ms=3.4, lw=1.2,
                       color=obs_colours[m],
                       label=(style.tex(obs_labels[m]) if (i, j) == (nbins - 1, 0) else None))

            a.set_xscale("log")
            a.set_yscale(yscale)
            if ylim is not None:
                a.set_ylim(*ylim)
            a.text(0.05, 0.93, style.tex(spectrum_label(i, j)), transform=a.transAxes,
                   fontsize=font_size - 1.5, ha="left", va="top")
            a.tick_params(labelsize=font_size - 2.5)
            if i == nbins - 1:
                a.set_xlabel(r"$\ell$", fontsize=font_size)
            if j == 0:
                a.set_ylabel(ylab, fontsize=font_size - 1)

    handles, labels = ax[nbins - 1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False,
               fontsize=font_size, bbox_to_anchor=(0.98, 0.98))
    if title:
        fig.suptitle(style.tex(title), fontsize=font_size + 2, y=0.995)
    fig.tight_layout(rect=(0, 0, 0.99, 0.97 if title else 1.0))
    return fig, ax
