"""Replay chain: cache -> candidate estimator -> alms -> filtered E/B maps -> patches + bandpowers.

Downstream of the estimator this calls the PROTECTED production functions directly
(`filter_EB_alms_and_make_maps`, `get_patch_values`, `denoise_shear_cls`, `compute_cl_bandpowers`)
so every candidate shares the exact production filter/patch/2-pt code path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import healpy as hp
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.cosmology.map_shears import filter_EB_alms_and_make_maps          # noqa: E402
from src.cosmology.pixelise_maps import get_patch_values                   # noqa: E402
from src.cosmology.manip_cls import compute_cl_bandpowers, denoise_shear_cls  # noqa: E402
from src.KiDS.simulation_config import named_patches                       # noqa: E402

from . import estimators                                                    # noqa: E402

# (fwhm, lmin, lcut) smoothing variants written by the master sim (master_kids_legacy_simulator.py:241)
EB_SMOOTHING_VARIANTS = [(4.0, 56, 1400), (8.0, 56, 1400), (8.0, 56, 1024)]

PROD_BANDS = {"lower_lscale": 56, "upper_lscale": 1500, "nbands": 8}
SMOKE_BANDS = {"lower_lscale": 56, "upper_lscale": 480, "nbands": 8}


def variant_tag(fwhm, lmin, lcut):
    return (f"fwhm{fwhm:g}" + ("" if lmin is None else f"_lmin{int(lmin)}")
            + ("" if lcut is None else f"_lcut{int(lcut)}"))


def resolve_geometry(attrs):
    nside = int(attrs["nside"])
    geom = {
        "nside": nside,
        "nside_out": int(attrs["nside_out"]),
        "lmax": 2 * nside,
        "patches": dict(named_patches),
    }
    geom.update(PROD_BANDS if nside >= 1024 else SMOKE_BANDS)
    return geom


def effective_nside_bin(cand, nside_native):
    return int(cand.get("nside_bin")
               or nside_native // int(cand.get("nside_bin_factor", 1)))


def _alm_pair(m, lmax):
    almE, almB = hp.sphtfunc.map2alm_spin([np.ascontiguousarray(m.real),
                                           np.ascontiguousarray(m.imag)], spin=2, lmax=lmax)
    return almE, almB


def build_arm(cache, cand, eb_variant, rng=None, n_rand=1, bp_baseline=None):
    """Run one candidate over one mock's cache. Returns the full per-arm product dict.

    bp_baseline: optional precomputed baseline (alm, alm_rand) at native nside for the
    two-branch design (bandpowers stay at native counts when the candidate bins coarser).
    """
    attrs = cache["attrs"]
    geom = resolve_geometry(attrs)
    nbins = int(attrs["nbins"])
    nside_native = geom["nside"]
    nside_bin = effective_nside_bin(cand, nside_native)
    lmax = 2 * nside_bin
    fwhm, lmin, lcut = eb_variant
    if rng is None:
        rng = np.random.default_rng(12345)

    alm, alm_rand, aux_bins, counts_out = [], [], [], []
    alm_size = hp.Alm.getsize(lmax)
    for i in range(nbins):
        cb = cache["bins"][i]
        if int(cb.get("n_gal", 0)) == 0 or "pix" not in cb:
            z = np.zeros(alm_size, dtype=complex)
            alm.append((z, z.copy()))
            alm_rand.append((z.copy(), z.copy()))
            aux_bins.append({})
            counts_out.append(np.zeros(hp.nside2npix(geom["nside_out"])))
            continue
        dm = estimators.dense_maps(cb, nside_native, nside_bin, n_rand=n_rand, want_q=True)
        m, aux = estimators.normalise(dm["S"], dm["N"], cand, cache_bin=cb,
                                      nside_bin=nside_bin, rng=rng)
        r, _ = estimators.normalise(dm["R0"], dm["N"], cand, cache_bin=cb,
                                    nside_bin=nside_bin, rng=rng)
        alm.append(_alm_pair(m, lmax))
        alm_rand.append(_alm_pair(r, lmax))
        aux_bins.append(aux)
        counts_out.append(hp.ud_grade(dm["N"], geom["nside_out"], power=-2))

    E_maps, B_maps = filter_EB_alms_and_make_maps(
        alm_list=alm, nside_out=geom["nside_out"], lmax_out=None,
        fwhm_arcmin=fwhm, taper_start_frac=0.95, lmin=lmin, lcut=lcut)
    randE_maps, randB_maps = filter_EB_alms_and_make_maps(
        alm_list=alm_rand, nside_out=geom["nside_out"], lmax_out=None,
        fwhm_arcmin=fwhm, taper_start_frac=0.95, lmin=lmin, lcut=lcut)

    counts_out = np.stack(counts_out)
    foot = counts_out > 0

    # --- map post-processing (Track B stages), applied to the E channel -----------------------
    post = cand.get("map_post")
    post_factors = np.ones(nbins)
    if post:
        for i in range(nbins):
            fp = foot[i]
            if not fp.any():
                continue
            if post == "self_std":
                mu, sd = E_maps[i][fp].mean(), E_maps[i][fp].std()
                if sd > 0:
                    E_maps[i][fp] = (E_maps[i][fp] - mu) / sd
                    E_maps[i][~fp] = 0.0
                    post_factors[i] = sd
            elif post == "div_B_std":
                sd = B_maps[i][fp].std()
                if sd > 0:
                    E_maps[i] = E_maps[i] / sd
                    post_factors[i] = sd
            elif post == "div_rand_std":
                sd = randE_maps[i][fp].std()
                if sd > 0:
                    E_maps[i] = E_maps[i] / sd
                    post_factors[i] = sd
            else:
                raise ValueError(f"unknown map_post {post!r}")

    patches = list(geom["patches"].values())
    patch_names = list(geom["patches"].keys())
    E_p = get_patch_values(E_maps, patches, geom["nside_out"], 0)
    B_p = get_patch_values(B_maps, patches, geom["nside_out"], 0)
    E_patches = {name: E_p[j] for j, name in enumerate(patch_names)}
    B_patches = {name: B_p[j] for j, name in enumerate(patch_names)}

    # --- bandpower branch ---------------------------------------------------------------------
    if bp_baseline is not None:
        bp_alm, bp_alm_rand, bp_lmax = bp_baseline
    else:
        bp_alm, bp_alm_rand, bp_lmax = alm, alm_rand, lmax
    mixed_cls = denoise_shear_cls(nbins, bp_alm, bp_alm_rand, bp_lmax)
    lo, up, nb = geom["lower_lscale"], geom["upper_lscale"], geom["nbands"]
    up = min(up, bp_lmax)
    mixed_cut = mixed_cls[:, :, :, lo:up + 1]
    cll_bands, bandpowers = compute_cl_bandpowers(mixed_cut, nbins, lo, up, nb)

    return {
        "geom": geom, "candidate": dict(cand), "eb_variant": tuple(eb_variant),
        "alm": alm, "alm_rand": alm_rand, "lmax": lmax,
        "E_maps": E_maps, "B_maps": B_maps, "randE_maps": randE_maps, "randB_maps": randB_maps,
        "counts_out": counts_out, "footprint": foot,
        "E_patches": E_patches, "B_patches": B_patches,
        "post_factors": post_factors, "aux_bins": aux_bins,
        "bandpowers": bandpowers, "bandpower_ls": cll_bands,
        "mixed_cls": mixed_cls,
    }


def baseline_alms(cache, n_rand=1, rng=None):
    """Native-nside counts-normalised (alm, alm_rand, lmax) — the production 2-pt branch."""
    attrs = cache["attrs"]
    geom = resolve_geometry(attrs)
    out = build_arm(cache, {"id": "A0_counts", "norm": "counts"},
                    EB_SMOOTHING_VARIANTS[0], rng=rng, n_rand=n_rand)
    return out["alm"], out["alm_rand"], out["lmax"]


def rel_rms(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.sqrt(np.mean(b ** 2))
    if denom == 0:
        return float(np.sqrt(np.mean((a - b) ** 2)))
    return float(np.sqrt(np.mean((a - b) ** 2)) / denom)


def fidelity_check(replay_out, mock_path, eb_tag=None):
    """Compare a counts-mode replay against the stored output_*.h5. Returns {dataset: rel_rms}."""
    res = {}
    fwhm, lmin, lcut = replay_out["eb_variant"]
    tag = eb_tag or variant_tag(fwhm, lmin, lcut)
    with h5py.File(mock_path, "r") as f:
        pr = f["pixelised_results"]
        e_key = f"E_{tag}" if f"E_{tag}" in pr else "E"
        b_key = f"B_{tag}" if f"B_{tag}" in pr else "B"
        for patch in replay_out["E_patches"]:
            stored_E = pr[e_key][patch][()]
            stored_B = pr[b_key][patch][()]
            res[f"E/{patch}"] = rel_rms(replay_out["E_patches"][patch], stored_E)
            res[f"B/{patch}"] = rel_rms(replay_out["B_patches"][patch], stored_B)
        stored_bp = f["cls_results"]["full"]["mixed_bandpowers"][()]
        res["mixed_bandpowers"] = rel_rms(replay_out["bandpowers"], stored_bp)
    return res
