"""Catalogue -> sparse per-pixel moment cache.

One cache file per mock, per tomographic bin: the sufficient statistics of the pixel-level
estimator, so every candidate normalisation is a cheap function of the cache instead of a
~30M-galaxy replay. Pixel indices are stored in NESTED ordering at the catalogue's native nside
so coarser binnings (nside 512/256) are exact bit-shifts of the same sums.

Stored per bin (sparse, occupied pixels only; ``she`` = (e1-ē1 + i(e2-ē2))/(1+m_i), i.e. the
exact per-galaxy quantity `make_alm_shear_convergence` accumulates):
    pix   int32   NESTED pixel index at native nside
    N     int32   galaxy count
    S1,S2 float64 Σ she (real, imag)                      -> the estimator numerator
    Q11,Q22,Q12 float32  Σ she_re², Σ she_im², Σ she_re·she_im  -> exact per-pixel noise moments
    NA    int32,  SA1,SA2 float32   half-split "A" (even in-bin galaxy index); B = total - A
    R{k}r,R{k}i float32  k=0..n_rand-1: Σ e^{iθ_g}·she  random-rotation noise realisations
                 (θ ~ U[0,2π) per galaxy, deterministic per (cache, k) from the seed sequence)
Bin attrs: e1_mean, e2_mean, n_gal. File attrs: everything from the catalogue file (sim_id,
cat_idx, galaxy_bias, m_bias_for_shear, nside, nside_out, shear_normalization, rng_seed, ...),
plus cache_version and the rand seed material. cosmo_dict group is copied through.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import healpy as hp
import numpy as np

CACHE_VERSION = 1
N_RAND_DEFAULT = 2


def _sparse_moments(pix, she, n_rand, rng):
    """Accumulate per-pixel sufficient statistics for one tomographic bin."""
    upix, inv = np.unique(pix, return_inverse=True)
    nsp = upix.size
    out = {
        "pix": upix.astype(np.int32),
        "N": np.bincount(inv, minlength=nsp).astype(np.int32),
        "S1": np.bincount(inv, weights=she.real, minlength=nsp),
        "S2": np.bincount(inv, weights=she.imag, minlength=nsp),
        "Q11": np.bincount(inv, weights=she.real ** 2, minlength=nsp).astype(np.float32),
        "Q22": np.bincount(inv, weights=she.imag ** 2, minlength=nsp).astype(np.float32),
        "Q12": np.bincount(inv, weights=she.real * she.imag, minlength=nsp).astype(np.float32),
    }
    half_a = np.arange(pix.size) % 2 == 0
    out["NA"] = np.bincount(inv[half_a], minlength=nsp).astype(np.int32)
    out["SA1"] = np.bincount(inv[half_a], weights=she.real[half_a], minlength=nsp).astype(np.float32)
    out["SA2"] = np.bincount(inv[half_a], weights=she.imag[half_a], minlength=nsp).astype(np.float32)
    for k in range(n_rand):
        theta = 2.0 * np.pi * rng.random(pix.size)
        rot = she * np.exp(1j * theta)   # identical to the e1_corr/e2_corr rotation in map_shears.py:253
        out[f"R{k}r"] = np.bincount(inv, weights=rot.real, minlength=nsp).astype(np.float32)
        out[f"R{k}i"] = np.bincount(inv, weights=rot.imag, minlength=nsp).astype(np.float32)
    return out


def build_pixel_cache(cat_path, out_path, n_rand=N_RAND_DEFAULT, seed_extra=0, overwrite=False):
    """Reduce one catalogue file to its sparse pixel cache. Returns the output path."""
    cat_path, out_path = Path(cat_path), Path(out_path)
    if out_path.exists() and not overwrite:
        return out_path

    with h5py.File(cat_path, "r") as f:
        g = f["catalogue"]
        ra = g["RA"][()].astype(np.float64)
        dec = g["DEC"][()].astype(np.float64)
        zbin = g["ZBIN"][()]
        e1 = g["E1"][()].astype(np.float64)
        e2 = g["E2"][()].astype(np.float64)
        attrs = {k: f.attrs[k] for k in f.attrs}
        cosmo = {}
        if "cosmo_dict" in f:
            for k in f["cosmo_dict"]:
                v = f["cosmo_dict"][k][()]
                cosmo[k] = v.decode() if isinstance(v, bytes) else v

    nside = int(attrs["nside"])
    m_bias = np.atleast_1d(np.asarray(attrs["m_bias_for_shear"], dtype=np.float64))
    nbins = int(zbin.max()) + 1 if zbin.size else m_bias.size
    if m_bias.size == 1:
        m_bias = np.repeat(m_bias, nbins)

    # Deterministic rotation seed: unique per (fixed-rng seed, sim, block, cat, b_g arm, extra).
    seed_seq = np.random.SeedSequence([
        int(attrs.get("rng_seed", -1)) & 0x7FFFFFFF,
        int(attrs.get("sim_id", 0)), int(attrs.get("outer_idx", 0)),
        int(attrs.get("rot_idx", 0)), int(attrs.get("cat_idx", 0)),
        int(round(float(attrs.get("galaxy_bias", 1.0)) * 1000)), int(seed_extra),
    ])
    rng = np.random.default_rng(seed_seq)

    pix_nest = hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.h5")
    with h5py.File(tmp, "w") as f:
        f.attrs["cache_version"] = CACHE_VERSION
        f.attrs["n_rand"] = int(n_rand)
        f.attrs["nbins"] = int(nbins)
        f.attrs["seed_extra"] = int(seed_extra)
        for k, v in attrs.items():
            f.attrs[k] = v
        cg = f.create_group("cosmo_dict")
        for k, v in cosmo.items():
            if isinstance(v, str):
                cg.create_dataset(k, data=v, dtype=h5py.string_dtype(encoding="utf-8"))
            else:
                cg.create_dataset(k, data=np.asarray(v))
        for i in range(nbins):
            sel = zbin == i
            gb = f.create_group(f"bin{i}")
            gb.attrs["n_gal"] = int(sel.sum())
            if not np.any(sel):
                gb.attrs["e1_mean"] = 0.0
                gb.attrs["e2_mean"] = 0.0
                continue
            be1, be2 = e1[sel], e2[sel]
            e1m, e2m = float(be1.mean()), float(be2.mean())
            gb.attrs["e1_mean"], gb.attrs["e2_mean"] = e1m, e2m
            she = ((be1 - e1m) + 1j * (be2 - e2m)) / (1.0 + m_bias[i])
            mom = _sparse_moments(pix_nest[sel], she, n_rand, rng)
            for name, arr in mom.items():
                gb.create_dataset(name, data=arr)
    tmp.rename(out_path)
    return out_path


def load_cache(path):
    """Load a cache file into a plain dict: {'attrs': ..., 'cosmo': ..., 'bins': [dict per bin]}."""
    with h5py.File(path, "r") as f:
        attrs = {k: f.attrs[k] for k in f.attrs}
        cosmo = {}
        if "cosmo_dict" in f:
            for k in f["cosmo_dict"]:
                v = f["cosmo_dict"][k][()]
                cosmo[k] = v.decode() if isinstance(v, bytes) else v
        bins = []
        for i in range(int(attrs["nbins"])):
            gb = f[f"bin{i}"]
            b = {k: gb[k][()] for k in gb}
            b.update({k: gb.attrs[k] for k in gb.attrs})
            bins.append(b)
    return {"attrs": attrs, "cosmo": cosmo, "bins": bins}
