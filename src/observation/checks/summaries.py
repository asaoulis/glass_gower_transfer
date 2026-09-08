"""Hand-crafted, theta-light summary vector of the E-mode patch maps (the pre-CNN Tier-1 statistic).

Computed per (patch, tomographic bin) on the SAME product the models read (the baked, noise-normed
``E`` patches), so the observation and the reference cloud go through identical arithmetic:

  moments   var, skewness, excess kurtosis                                     (3)
  pdf       quantiles at 5/25/50/75/95 %                                        (5)
  peaks     local-maximum density above +1/+2/+3 sigma_self, local-minimum density
            below -1/-2/-3 sigma_self (3x3 neighbourhood; per 1e3 pixels)      (6)
  power     log10 azimuthally-averaged 2-D power in 6 log-spaced |k| bins       (6)
  xcorr     Pearson r between adjacent tomographic bins (per patch)             (5 / patch)

i.e. 20 x 6 bins x 2 patches + 10 = 250 numbers. Groups are named so a test can use a subset.
Everything is invariant to the *labelling* of the observation and never touches a cosmology.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

GROUPS = ("moments", "pdf", "peaks", "power", "xcorr")
_Q = (0.05, 0.25, 0.5, 0.75, 0.95)
_PEAK_SIG = (1.0, 2.0, 3.0)
_N_KBINS = 6


def _radial_power(x: np.ndarray, nk: int = _N_KBINS) -> np.ndarray:
    """log10 of the azimuthally averaged |FFT|^2 in nk log-spaced |k| bins (k in cycles/pixel)."""
    h, w = x.shape
    f = np.fft.rfft2(x - x.mean())
    p = (f.real ** 2 + f.imag ** 2) / (h * w)
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.rfftfreq(w)[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2)
    kmin = 1.0 / max(h, w)
    edges = np.geomspace(kmin, 0.5, nk + 1)
    out = np.empty(nk)
    for i in range(nk):
        sel = (k >= edges[i]) & (k < edges[i + 1])
        out[i] = np.log10(p[sel].mean() + 1e-30) if sel.any() else np.nan
    return out


def _peak_counts(x: np.ndarray, sig: float) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import maximum_filter, minimum_filter
    mx = maximum_filter(x, size=3, mode="nearest")
    mn = minimum_filter(x, size=3, mode="nearest")
    is_max = (x == mx)
    is_min = (x == mn)
    n = x.size / 1e3
    peaks = np.array([np.sum(is_max & (x > s * sig)) / n for s in _PEAK_SIG])
    voids = np.array([np.sum(is_min & (x < -s * sig)) / n for s in _PEAK_SIG])
    return peaks, voids


def summary_names(nbins: int = 6, patch_names: Sequence[str] = ("north", "south")) -> List[str]:
    names = []
    for p in patch_names:
        for b in range(nbins):
            pre = f"{p}/b{b}/"
            names += [pre + "moments/var", pre + "moments/skew", pre + "moments/kurt"]
            names += [pre + f"pdf/q{int(q * 100):02d}" for q in _Q]
            names += [pre + f"peaks/max_gt{int(s)}s" for s in _PEAK_SIG] + [pre + f"peaks/min_lt{int(s)}s" for s in _PEAK_SIG]
            names += [pre + f"power/k{i}" for i in range(_N_KBINS)]
        names += [f"{p}/xcorr/b{b}b{b + 1}" for b in range(nbins - 1)]
    return names


def emap_summary_vector(patches: Dict[str, np.ndarray], patch_names: Sequence[str] = ("north", "south")) -> np.ndarray:
    """``patches``: {patch_name: (nbins, H, W)} of the noise-normed E maps -> 1-D float64 vector
    ordered as ``summary_names``."""
    vec: List[float] = []
    for p in patch_names:
        arr = np.asarray(patches[p], dtype=np.float64)
        nb = arr.shape[0]
        flat_bins = []
        for b in range(nb):
            x = arr[b]
            xf = x.ravel()
            flat_bins.append(xf)
            mu = xf.mean()
            sd = xf.std()
            d = (xf - mu) / (sd if sd > 0 else 1.0)
            vec += [sd ** 2, float(np.mean(d ** 3)), float(np.mean(d ** 4) - 3.0)]
            vec += list(np.quantile(xf, _Q))
            pk, vd = _peak_counts(x - mu, sd if sd > 0 else 1.0)
            vec += list(pk) + list(vd)
            vec += list(_radial_power(x))
        for b in range(nb - 1):
            a, c = flat_bins[b], flat_bins[b + 1]
            vec.append(float(np.corrcoef(a, c)[0, 1]) if a.std() > 0 and c.std() > 0 else 0.0)
    return np.asarray(vec, dtype=np.float64)


def group_mask(names: Sequence[str], groups: Sequence[str]) -> np.ndarray:
    """Boolean mask selecting the summary entries whose group is in ``groups``."""
    return np.array([any(f"/{g}/" in n for g in groups) for n in names], dtype=bool)
