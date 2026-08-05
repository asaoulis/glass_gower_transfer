"""Candidate estimators: sparse pixel cache -> normalised full-sky complex shear map.

Every candidate is a config dict: {"id", "nside_bin", "norm", <norm params>, "map_post"}.
``accumulate`` turns the cache's NESTED sparse sums into dense RING maps at ``nside_bin``
(exact for any nside <= native: coarser pixels are bit-shifted parents, sums are additive).
``normalise`` implements the candidate normalisation modes — the production ``counts`` branch
reproduces `_apply_normalization('counts')` in src/cosmology/map_shears.py:194 exactly.

The "mean" mode here uses n̄ = ΣN / N_occupied (occupied-pixel mean) rather than the production
Σ(fractional mask)/npix denominator (the mask is not stored with the catalogues). This is a
per-mock CONSTANT rescale — irrelevant for any b_g differential (the mask is identical across
arms), only the absolute amplitude differs from the legacy store. Documented deviation.
"""
from __future__ import annotations

import healpy as hp
import numpy as np


def dense_maps(cache_bin, nside_native, nside_bin, n_rand=0, want_half=False, want_q=False):
    """Dense RING-ordered maps at nside_bin from one bin's sparse NESTED cache arrays."""
    shift = 2 * (int(np.log2(nside_native)) - int(np.log2(nside_bin)))
    if shift < 0:
        raise ValueError("nside_bin must be <= native nside")
    coarse_nest = cache_bin["pix"].astype(np.int64) >> shift
    ring = hp.nest2ring(nside_bin, coarse_nest)
    npix = hp.nside2npix(nside_bin)

    def scatter(values, dtype=np.float64):
        m = np.zeros(npix, dtype=dtype)
        np.add.at(m, ring, values)
        return m

    out = {
        "N": scatter(cache_bin["N"]),
        "S": scatter(cache_bin["S1"]) + 1j * scatter(cache_bin["S2"]),
    }
    if want_q:
        out["Q"] = scatter(cache_bin["Q11"].astype(np.float64)
                           + cache_bin["Q22"].astype(np.float64))
    if want_half:
        out["NA"] = scatter(cache_bin["NA"])
        out["SA"] = scatter(cache_bin["SA1"].astype(np.float64)) \
            + 1j * scatter(cache_bin["SA2"].astype(np.float64))
    for k in range(n_rand):
        out[f"R{k}"] = scatter(cache_bin[f"R{k}r"].astype(np.float64)) \
            + 1j * scatter(cache_bin[f"R{k}i"].astype(np.float64))
    return out


def _sigma2_gal(cache_bin):
    """Per-galaxy per-component variance of ``she`` for this bin (from the Q moments)."""
    n = max(int(cache_bin["N"].sum()), 1)
    q = float(cache_bin["Q11"].astype(np.float64).sum()
              + cache_bin["Q22"].astype(np.float64).sum())
    return q / (2.0 * n)


def normalise(field, counts, cfg, cache_bin=None, nside_bin=None, rng=None):
    """Apply the candidate normalisation to a dense complex sum map. Returns (map, aux dict).

    field/counts are MUTATED-safe copies inside; callers keep their originals.
    """
    mode = cfg.get("norm", "counts")
    m = field.copy()
    valid = counts > 0
    aux = {}

    if mode == "counts":
        m[valid] = m[valid] / counts[valid]
        return m, aux

    if mode == "mean":                       # occupied-pixel mean denominator (see module docstring)
        nbar = counts[valid].mean() if valid.any() else 0.0
        if nbar > 0:
            m[valid] = m[valid] / nbar
        else:
            m[:] = 0.0
        aux["nbar"] = float(nbar)
        return m, aux

    if mode == "alpha":                      # S / (N^a * nbar^(1-a)); a=1 -> counts, a=0 -> mean
        a = float(cfg["alpha"])
        nbar = counts[valid].mean() if valid.any() else 1.0
        m[valid] = m[valid] / (counts[valid] ** a * nbar ** (1.0 - a))
        aux["nbar"] = float(nbar)
        return m, aux

    if mode == "smoothed_counts":            # divide by the count map smoothed at fwhm arcmin
        fwhm = float(cfg["fwhm_arcmin"])
        ns = hp.smoothing(counts.astype(np.float64), fwhm=np.deg2rad(fwhm / 60.0))
        floor = 0.05 * counts[valid].mean() if valid.any() else 1.0
        denom = np.maximum(ns, floor)
        m[valid] = m[valid] / denom[valid]
        aux["nbar"] = float(counts[valid].mean()) if valid.any() else 0.0
        return m, aux

    if mode == "levelled":                   # counts + noise-levelling to the q-quantile count
        q = float(cfg.get("level_quantile", 0.02))
        m[valid] = m[valid] / counts[valid]
        sig2 = _sigma2_gal(cache_bin)
        n_tgt = max(np.quantile(counts[valid], q), 1.0) if valid.any() else 1.0
        var_add = np.zeros_like(counts)
        var_add[valid] = sig2 * np.clip(1.0 / n_tgt - 1.0 / counts[valid], 0.0, None)
        if rng is None:
            rng = np.random.default_rng(0)
        noise = rng.normal(size=m.shape) + 1j * rng.normal(size=m.shape)
        m = m + np.sqrt(var_add) * noise
        m[~valid] = 0.0
        aux["n_target"] = float(n_tgt)
        aux["sigma2_gal"] = float(sig2)
        return m, aux

    if mode == "global_rescale":             # counts, then per-mock/bin analytic amplitude fix
        m[valid] = m[valid] / counts[valid]
        nbar = counts[valid].mean()
        inv_mean = (1.0 / counts[valid]).mean()
        f = 1.0 / np.sqrt(nbar * inv_mean)   # noise var <sig^2/N> -> sig^2/nbar exactly
        m *= f
        aux["rescale"] = float(f)
        aux["nbar"] = float(nbar)
        aux["inv_mean"] = float(inv_mean)
        return m, aux

    if mode == "nn":                         # unnormalised (global-mean) + counts as aux channel
        nbar = counts[valid].mean() if valid.any() else 1.0
        m[valid] = m[valid] / nbar
        aux["counts_map"] = counts.copy()
        aux["nbar"] = float(nbar)
        return m, aux

    raise ValueError(f"unknown norm mode {mode!r}")


# --- candidate register (mirrors artifacts/candidates.md) -------------------------------------
CANDIDATES = {
    "A0_counts":        {"norm": "counts"},
    "A0b_mean":         {"norm": "mean"},
    "A1_wht_rand":      {"norm": "counts", "map_post": "div_rand_std"},
    "A2_nsideHalf":     {"norm": "counts", "nside_bin_factor": 2},   # 1024->512 (smoke 256->128)
    "A2_nsideQuarter":  {"norm": "counts", "nside_bin_factor": 4},
    "A3_smooth4":       {"norm": "smoothed_counts", "fwhm_arcmin": 4.0},
    "A3_smooth8":       {"norm": "smoothed_counts", "fwhm_arcmin": 8.0},
    "A3_smooth16":      {"norm": "smoothed_counts", "fwhm_arcmin": 16.0},
    "A4_levelled":      {"norm": "levelled", "level_quantile": 0.02},
    "A5_alpha0p5":      {"norm": "alpha", "alpha": 0.5},
    "A5_alpha0p75":     {"norm": "alpha", "alpha": 0.75},
    "A5_alpha0p9":      {"norm": "alpha", "alpha": 0.9},
    "A8_rescale":       {"norm": "global_rescale"},
    "A6_nn_counts":     {"norm": "nn"},
    "B1_selfstd":       {"norm": "counts", "map_post": "self_std"},
    "B2_divBstd":       {"norm": "counts", "map_post": "div_B_std"},
    # combinations
    "A3s8_A1":          {"norm": "smoothed_counts", "fwhm_arcmin": 8.0, "map_post": "div_rand_std"},
    "A8_B2":            {"norm": "global_rescale", "map_post": "div_B_std"},
}


def get_candidate(name):
    cfg = dict(CANDIDATES[name])
    cfg["id"] = name
    return cfg
