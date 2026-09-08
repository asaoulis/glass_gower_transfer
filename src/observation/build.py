"""Catalogue -> observation HDF5, by calling master's observable-building sequence verbatim.

Sequence (master_kids_legacy_simulator.py L1262-1457, production 'sc8only' preset):

  1. alm, alm_rand  = make_alm_shear_convergence(cat, m, nbins, nside, lmax, nosh=False, mask,
                       normalization="counts", rng=postproc_rng)
  2. mixed_cls      = denoise_shear_cls(nbins, alm, alm_rand, lmax)             (EE,BB,EB per pair)
     mixed_bandpowers = compute_cl_bandpowers(mixed_cls[..., lmin:lmax+1], nbins, lmin, lmax, nbands)
  3. alm_sc8, alm_rand_sc8 = make_alm_shear_convergence(..., normalization="smoothed_counts",
                       smoothed_counts_fwhm_arcmin=8, rng=postproc_rng.spawn(1)[0])
     for each A3s8 variant: E_sc8_<tag> = filter_EB_alms_and_make_maps(alm_sc8, ...)[0];
                            noise_std_sc8_<tag> = patch_noise_std(filter(alm_rand_sc8)[0], ...)
  4. for each counts variant: E_<tag>, B_<tag> = filter(alm); noise_std_<tag> = patch_noise_std(filter(alm_rand))
  5. patches via get_patch_values(map, patches, nside_out, ang=0); E_sc8_* cast to float16.
  6. save: cls_results/full/{mixed_bandpowers, bandpower_ls, cls[:, :, :2]}, pixelised_results/*,
     cosmo_dict (EMPTY here), + an ``observation/`` provenance group (never cosmological).

Additions over a mock (harmless to every loader, which read only the paths they ask for):
  * ``cls_results/full/bb_bandpowers`` (21, nbands): the BB spectra through the SAME Brown
    binning as the EE ``mixed_bandpowers`` -- the Tier-1 B-mode null-test vector.
  * ``observation/`` group: label, geometry, variants, treatment knobs, catalogue fingerprint,
    per-bin counts, git revision, build timestamp.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np

from .catalogue_io import Catalogue
from .geometry import (A3S8_MAP_DTYPE, TAPER_START_FRAC, Geometry, MapVariants, eb_variant_tag,
                       patch_noise_std)


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=str(Path(__file__).resolve().parents[2]),
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def bb_bandpowers_from_cls(mixed_cut, nbins, lower, upper, nbands):
    """BB bandpowers with the identical binning as compute_cl_bandpowers (which bins index 0 = EE).
    ``mixed_cut`` is (nbins, nbins, >=2, n_ell) already cut to [lower, upper]."""
    from src.cosmology.manip_cls import make_bandpowers
    rows = []
    for i in range(nbins):
        for j in range(nbins):
            if i < j:
                continue
            bp, _, _ = make_bandpowers(lower, upper, cls=np.asarray(mixed_cut[i][j][1], dtype=float),
                                       nbands=nbands)
            rows.append(bp)
    return np.asarray(rows)


def _save_dict(h5group, dictionary):
    """Mirror of src/cosmology/sim_utils.save_results_h5._save_dict (kept local so we control the
    output filename and can add groups)."""
    for key, value in dictionary.items():
        if isinstance(value, dict):
            _save_dict(h5group.create_group(str(key)), value)
        elif isinstance(value, str):
            h5group.create_dataset(str(key), data=value, dtype=h5py.string_dtype(encoding="utf-8"))
        else:
            arr = np.asarray(value)
            if arr.dtype == object:
                arr = arr.astype(np.float64)
            h5group.create_dataset(str(key), data=arr)


def build_observation(cat: Catalogue, m_bias: np.ndarray, *, out_path: str, label: str,
                      geometry: Geometry, variants: Optional[MapVariants] = None,
                      rng_seed: int = 20260908, rng=None, mask=None, verbose: bool = True,
                      extra_provenance: Optional[Dict] = None) -> str:
    """Build ``out_path`` from a loaded catalogue. Returns the written path.

    ``mask``: passed through to make_alm_shear_convergence, which does NOT apply it to the maps
    (it only enters the 'mean' normalisation, unused here) -- so None is safe and is the default.
    ``rng_seed``: seeds the random-rotation noise meter (a pure shape-noise realisation); the
    observable maps themselves are deterministic given the catalogue, and the bandpowers depend on
    it only through the high-ell shape-noise debias. ``rng``: an explicit Generator that overrides
    ``rng_seed`` (the fidelity gate passes the mock's own block stream, see
    ``geometry.master_postproc_rng``).
    """
    from src.cosmology.manip_cls import compute_cl_bandpowers, denoise_shear_cls
    from src.cosmology.map_shears import filter_EB_alms_and_make_maps, make_alm_shear_convergence
    from src.cosmology.pixelise_maps import get_patch_values

    variants = variants or MapVariants()
    g = geometry
    m_bias = np.asarray(m_bias, dtype=float)
    assert m_bias.shape == (g.nbins,), m_bias.shape
    patches = [tuple(p) for p in g.patches]
    patch_names = list(g.patch_names)

    def log(msg):
        if verbose:
            print(msg, flush=True)

    postproc_rng = rng if rng is not None else np.random.default_rng(int(rng_seed))
    rng_note = "explicit generator" if rng is not None else f"default_rng({int(rng_seed)})"

    # 1-2. counts branch -> bandpowers (EE) + BB + per-ell cls
    log(f"[obs:{label}] counts-branch alms (nside={g.nside}, lmax={g.lmax}, n_gal={cat.n_gal:,})")
    alm, alm_rand = make_alm_shear_convergence(
        cat.data, m_bias, g.nbins, g.nside, g.lmax, nosh=False, mask=mask,
        normalization="counts", rng=postproc_rng)
    mixed_cls = denoise_shear_cls(g.nbins, alm, alm_rand, g.lmax)
    mixed_cut = mixed_cls[:, :, :, g.lower_lscale:g.upper_lscale + 1]
    cll_bands, mixed_bandpowers = compute_cl_bandpowers(mixed_cut, g.nbins, g.lower_lscale,
                                                        g.upper_lscale, g.nbands)
    bb_bandpowers = bb_bandpowers_from_cls(mixed_cut, g.nbins, g.lower_lscale, g.upper_lscale, g.nbands)

    # 3. smoothed-counts (A3s8) branch
    alm_sc8 = alm_rand_sc8 = None
    if variants.a3s8:
        sc8_rng = postproc_rng.spawn(1)[0]
        log(f"[obs:{label}] smoothed-counts alms (fwhm={variants.a3s8_fwhm_arcmin:g} arcmin)")
        alm_sc8, alm_rand_sc8 = make_alm_shear_convergence(
            cat.data, m_bias, g.nbins, g.nside, g.lmax, nosh=False, mask=mask,
            normalization="smoothed_counts",
            smoothed_counts_fwhm_arcmin=variants.a3s8_fwhm_arcmin, rng=sc8_rng)

    map_types: Dict[str, np.ndarray] = {}
    noise_std: Dict[str, dict] = {}
    sc8_tags = []
    if alm_sc8 is not None:
        for fwhm_v, lmin_v, lcut_v in variants.a3s8:
            sc8_tag = f"sc8_{eb_variant_tag(fwhm_v, lmin_v, lcut_v)}"
            sc8_tags.append(sc8_tag)
            E_s, _ = filter_EB_alms_and_make_maps(alm_list=alm_sc8, nside_out=g.nside_out, lmax_out=None,
                                                  fwhm_arcmin=fwhm_v, taper_start_frac=TAPER_START_FRAC,
                                                  lmin=lmin_v, lcut=lcut_v)
            map_types[f"E_{sc8_tag}"] = E_s
            Er_s, _ = filter_EB_alms_and_make_maps(alm_list=alm_rand_sc8, nside_out=g.nside_out,
                                                   lmax_out=None, fwhm_arcmin=fwhm_v,
                                                   taper_start_frac=TAPER_START_FRAC,
                                                   lmin=lmin_v, lcut=lcut_v)
            noise_std[sc8_tag] = patch_noise_std(Er_s, patches, g.nside_out, g.ang, patch_names)
            del Er_s
        alm_sc8 = alm_rand_sc8 = None

    # 4. counts-normalised E/B variants (+ noise meters)
    want_ns = (list(variants.eb) if variants.noise_std_for_eb is None
               else [tuple(v) for v in variants.noise_std_for_eb])
    for fwhm_v, lmin_v, lcut_v in variants.eb:
        E_v, B_v = filter_EB_alms_and_make_maps(alm_list=alm, nside_out=g.nside_out, lmax_out=None,
                                                fwhm_arcmin=fwhm_v, taper_start_frac=TAPER_START_FRAC,
                                                lmin=lmin_v, lcut=lcut_v)
        tag = eb_variant_tag(fwhm_v, lmin_v, lcut_v)
        map_types[f"E_{tag}"] = E_v
        map_types[f"B_{tag}"] = B_v
        if (fwhm_v, lmin_v, lcut_v) in want_ns:
            Er_v, _ = filter_EB_alms_and_make_maps(alm_list=alm_rand, nside_out=g.nside_out, lmax_out=None,
                                                   fwhm_arcmin=fwhm_v, taper_start_frac=TAPER_START_FRAC,
                                                   lmin=lmin_v, lcut=lcut_v)
            noise_std[tag] = patch_noise_std(Er_v, patches, g.nside_out, g.ang, patch_names)
            del Er_v

    # 5. patches
    pixelised: Dict[str, dict] = {name: {} for name in map_types}
    for name, cat_map in map_types.items():
        per_patch = get_patch_values(cat_map, patches, g.nside_out, g.ang)
        out_dtype = (A3S8_MAP_DTYPE if name.startswith("E_sc8_") else np.float32)
        for pidx, pname in enumerate(patch_names):
            arr = per_patch[pidx].astype(out_dtype, copy=False)
            if out_dtype is not np.float32 and not np.isfinite(arr).all():
                log(f"[obs:{label}] WARNING non-finite values in {name}/{pname} after "
                    f"{np.dtype(out_dtype).name} cast")
            pixelised[name][pname] = arr
    for tag, per_patch in noise_std.items():
        pixelised[f"noise_std_{tag}"] = per_patch
    prov_strings = {}
    if noise_std:
        prov_strings["noise_std"] = ("std of the filtered random-rotation E map over ALL pixels of each "
                                     "stored patch grid, per tomographic bin; 'all' pools both patches.")
    if sc8_tags:
        prov_strings["a3s8"] = (f"E_sc8_* built by a SECOND make_alm_shear_convergence call with "
                                f"normalization='smoothed_counts', smoothed_counts_fwhm_arcmin="
                                f"{variants.a3s8_fwhm_arcmin:g}.")
        prov_strings["a3s8_variant"] = ",".join(sc8_tags)
    if prov_strings:
        pixelised["_provenance"] = prov_strings

    cls_results = {"full": {"mixed_bandpowers": mixed_bandpowers, "bandpower_ls": cll_bands,
                            "cls": mixed_cls[:, :, :2, :], "bb_bandpowers": bb_bandpowers}}

    # 6. write (NO cosmology anywhere)
    observation_prov = {
        "label": str(label), "built_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_rev": _git_rev(), "rng_seed": int(rng_seed), "rng_note": rng_note, "n_gal": int(cat.n_gal),
        "counts_per_bin": json.dumps(cat.counts_per_bin(g.nbins)),
        "geometry": json.dumps(g.as_dict()), "variants": json.dumps(variants.as_dict()),
        "m_bias_used": np.asarray(m_bias, dtype=float),
        "catalogue_provenance": json.dumps(scrub_provenance(cat.provenance), default=str),
    }
    if extra_provenance:
        observation_prov.update({k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                                 for k, v in extra_provenance.items()})
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path + ".tmp"
    with h5py.File(tmp, "w") as f:
        _save_dict(f.create_group("cls_results"), cls_results)
        _save_dict(f.create_group("pixelised_results"), pixelised)
        f.create_group("cosmo_dict")            # EMPTY by design: an observation has no truth
        _save_dict(f.create_group("observation"), observation_prov)
    os.replace(tmp, out_path)
    with open(os.path.splitext(out_path)[0] + "_provenance.json", "w") as fh:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in observation_prov.items()}, fh, indent=2, default=str)
    # TRUTHKEY sidecar (mock-as-real only): the sim identity, kept OUT of the observation and the
    # store, under a _truthkey/ dir that the blind guard denies. Real data has no sim attrs.
    sim_attrs = cat.provenance.get("sim_attrs", {})
    if sim_attrs:
        tk_dir = Path(out_path).parent / "_truthkey"
        tk_dir.mkdir(exist_ok=True)
        with open(tk_dir / f"observation_{label}_truthkey.json", "w") as fh:
            json.dump({k: sim_attrs[k] for k in TRUTHKEY_KEYS if k in sim_attrs}, fh, indent=2, default=str)
    log(f"[obs:{label}] wrote {out_path}")
    return out_path


_COSMO_LIKE = ("omega_m", "sigma_8", "s8", "s_8", "w0", "h", "ns", "n_s", "ombh2", "a_ia", "b_ia",
               "mnu", "m_nu", "log10_m_eff", "b_g", "cosmo",
               # a SIM catalogue's identity IS its truth (sim_id -> cosmology table), so it is
               # scrubbed too; the gate keeps it in a separate TRUTHKEY sidecar (see build_observation)
               "sim_id", "galaxy_bias", "systematics_model", "outer_idx", "rot_idx", "cat_idx", "rng_seed")
TRUTHKEY_KEYS = ("sim_id", "galaxy_bias", "systematics_model", "outer_idx", "rot_idx", "cat_idx", "rng_seed")


def scrub_provenance(prov: dict) -> dict:
    """Defensive: drop any provenance key that looks cosmological before it is persisted.
    (A sim catalogue's file attrs carry none, but the contract is 'never'.)"""
    def _clean(d):
        if isinstance(d, dict):
            return {k: _clean(v) for k, v in d.items()
                    if not any(str(k).lower().startswith(c) for c in _COSMO_LIKE)}
        return d
    return _clean(prov)
