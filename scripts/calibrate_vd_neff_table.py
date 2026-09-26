#!/usr/bin/env python
"""Derive the A' variable-depth count-contrast table (`vd_contrast_xbar`, `vd_contrast_neff` in
src/KiDS/variable_depth_config.py) from the real KiDS-Legacy catalogue.

Per tomographic bin: smooth the tracer map with `smooth_vd_tracer` (mask-weighted Gaussian, FWHM
`vd_contrast_fwhm_arcmin`), split the galaxies on non-hole pixels into 10 galaxy-equipopulated bins of the
smoothed tracer, and record the mask-weighted mean smoothed tracer and the H12 effective density
(sum w)^2 / sum w^2 / area of each bin. Survey properties only (positions, TOMOBIN, lensfit weights):
blind-safe.

    python scripts/calibrate_vd_neff_table.py            # print paste-ready constants + write JSON
    python scripts/calibrate_vd_neff_table.py --check    # compare with the committed constants
    python scripts/calibrate_vd_neff_table.py --table patch [--check]   # the per-patch table (vd_patch_*)

`--table patch` calibrates KiDS-N and KiDS-S separately on the tail-resolved galaxy quantiles
`vd_patch_quantiles` (1, 2.5, 5, 10, ..., 90, 95, 97.5, 99 %), resolving the shallow and deep tails that
the ten 10 % bins average over.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import healpy as hp
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.KiDS.simulation_config import load_kids_mask
from src.KiDS.tomo import nbins
from src.KiDS.variable_depth_config import (load_vd_maps, n_vardepth_bins, smooth_vd_tracer, vd_contrast_fwhm_arcmin,
                                            vd_patch_dec_split, vd_patch_quantiles)

NSIDE = 1024


def calibrate(catalogue, data_dir, fwhm, quantiles=None, patch=None):
    """(xbar, neff), each (nbins, len(quantiles) - 1). `quantiles` are the galaxy-quantile bin edges
    (default: n_vardepth_bins equipopulated bins); `patch` in {None, 'N', 'S'} restricts galaxies and
    mask pixels to one KiDS patch (split at `vd_patch_dec_split`)."""
    q = np.linspace(0, 1, n_vardepth_bins + 1) if quantiles is None else np.asarray(quantiles)
    nb = len(q) - 1
    mask = load_kids_mask(str(data_dir))
    vd_map = load_vd_maps(data_dir, NSIDE)
    vd_smooth = smooth_vd_tracer(vd_map, mask, fwhm)
    with h5py.File(catalogue) as f:
        pix = hp.ang2pix(NSIDE, f["RAJ2000"][()], f["DECJ2000"][()], lonlat=True)
        tomo = f["TOMOBIN"][()].astype(int) - 1
        w = f["weight"][()]
    in_patch = np.ones(hp.nside2npix(NSIDE), bool)
    if patch is not None:
        north = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True)[1] > vd_patch_dec_split
        in_patch = north if patch == "N" else ~north
    pixarea = hp.nside2pixarea(NSIDE, degrees=True) * 3600.0
    xbar = np.zeros((nbins, nb))
    neff = np.zeros((nbins, nb))
    for i in range(nbins):
        hole = vd_map[i] <= 0
        sel = (tomo == i) & ~hole[pix] & in_patch[pix]
        xs = vd_smooth[i]
        edges = np.quantile(xs[pix[sel]], q)
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        jg = np.clip(np.digitize(xs[pix[sel]], edges) - 1, 0, nb - 1)
        ok = (mask > 0) & ~hole & in_patch
        jp = np.clip(np.digitize(xs[ok], edges) - 1, 0, nb - 1)
        s1 = np.bincount(jg, weights=w[sel], minlength=nb)
        s2 = np.bincount(jg, weights=w[sel] ** 2, minlength=nb)
        area = np.bincount(jp, weights=mask[ok], minlength=nb)
        xbar[i] = np.bincount(jp, weights=(mask * xs)[ok], minlength=nb) / area
        neff[i] = s1 ** 2 / s2 / (area * pixarea)
    return xbar, neff


def _rows(a, indent):
    out = []
    for r in a:
        v = [f"{x:.12g}" for x in r]
        chunks = [", ".join(v[k:k + 5]) for k in range(0, len(v), 5)]
        out.append(indent + "[" + (",\n" + indent + " ").join(chunks) + "],")
    return out


def paste(name, a):
    if a.ndim == 2:
        return "\n".join([f"{name} = np.array(["] + _rows(a, "    ") + ["])"])
    lines = [f"{name} = np.array(["]
    for p in a:
        lines += ["    ["] + _rows(p, "        ") + ["    ],"]
    return "\n".join(lines + ["])"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default="/data/alex/unblinding/real/inputs/catalogue/kids_legacy_7col.h5")
    ap.add_argument("--data-dir", default=str(REPO / "kids-legacy-sbi" / "data"))
    ap.add_argument("--fwhm", type=float, default=vd_contrast_fwhm_arcmin)
    ap.add_argument("--table", choices=["pooled", "patch"], default="pooled",
                    help="pooled: the A' table (10 equipopulated bins, KiDS-N+S together); patch: one table per "
                         "KiDS patch on the tail-resolved quantiles `vd_patch_quantiles`")
    ap.add_argument("--out", default=None, help="JSON output path")
    ap.add_argument("--check", action="store_true", help="compare with the committed constants (rtol 1e-6)")
    a = ap.parse_args()
    if a.table == "pooled":
        xbar, neff = calibrate(a.catalogue, Path(a.data_dir), a.fwhm)
        names = ("vd_contrast_xbar", "vd_contrast_neff")
    else:
        tabs = [calibrate(a.catalogue, Path(a.data_dir), a.fwhm, vd_patch_quantiles, p) for p in ("N", "S")]
        xbar, neff = np.stack([t[0] for t in tabs]), np.stack([t[1] for t in tabs])
        names = ("vd_patch_xbar", "vd_patch_neff")
    print(paste(names[0], xbar))
    print(paste(names[1], neff))
    for i in range(nbins):
        n = neff[..., i, :] if neff.ndim == 3 else neff[i]
        print(f"bin {i + 1}: n_eff {np.round(np.atleast_2d(n)[:, [0, -1]], 3).tolist()}")
    if a.out:
        json.dump({"fwhm_arcmin": a.fwhm, "table": a.table, "xbar": xbar.tolist(), "neff": neff.tolist()},
                  open(a.out, "w"), indent=1)
    if a.check:
        import src.KiDS.variable_depth_config as cfg
        ok = np.allclose(xbar, getattr(cfg, names[0]), rtol=1e-6) and np.allclose(neff, getattr(cfg, names[1]), rtol=1e-6)
        print("check vs committed constants:", "OK" if ok else "MISMATCH")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
