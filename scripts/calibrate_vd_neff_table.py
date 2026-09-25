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
from src.KiDS.variable_depth_config import load_vd_maps, n_vardepth_bins, smooth_vd_tracer, vd_contrast_fwhm_arcmin

NSIDE = 1024


def calibrate(catalogue, data_dir, fwhm):
    mask = load_kids_mask(str(data_dir))
    vd_map = load_vd_maps(data_dir, NSIDE)
    vd_smooth = smooth_vd_tracer(vd_map, mask, fwhm)
    with h5py.File(catalogue) as f:
        pix = hp.ang2pix(NSIDE, f["RAJ2000"][()], f["DECJ2000"][()], lonlat=True)
        tomo = f["TOMOBIN"][()].astype(int) - 1
        w = f["weight"][()]
    pixarea = hp.nside2pixarea(NSIDE, degrees=True) * 3600.0
    xbar = np.zeros((nbins, n_vardepth_bins))
    neff = np.zeros((nbins, n_vardepth_bins))
    for i in range(nbins):
        hole = vd_map[i] <= 0
        sel = (tomo == i) & ~hole[pix]
        xs = vd_smooth[i]
        edges = np.quantile(xs[pix[sel]], np.linspace(0, 1, n_vardepth_bins + 1))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        jg = np.clip(np.digitize(xs[pix[sel]], edges) - 1, 0, n_vardepth_bins - 1)
        ok = (mask > 0) & ~hole
        jp = np.clip(np.digitize(xs[ok], edges) - 1, 0, n_vardepth_bins - 1)
        s1 = np.bincount(jg, weights=w[sel], minlength=n_vardepth_bins)
        s2 = np.bincount(jg, weights=w[sel] ** 2, minlength=n_vardepth_bins)
        area = np.bincount(jp, weights=mask[ok], minlength=n_vardepth_bins)
        xbar[i] = np.bincount(jp, weights=(mask * xs)[ok], minlength=n_vardepth_bins) / area
        neff[i] = s1 ** 2 / s2 / (area * pixarea)
    return xbar, neff


def paste(name, a):
    lines = [f"{name} = np.array(["]
    for r in a:
        v = [f"{x:.12g}" for x in r]
        lines += ["    [" + ", ".join(v[:5]) + ",", "     " + ", ".join(v[5:]) + "],"]
    return "\n".join(lines + ["])"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default="/data/alex/unblinding/real/inputs/catalogue/kids_legacy_7col.h5")
    ap.add_argument("--data-dir", default=str(REPO / "kids-legacy-sbi" / "data"))
    ap.add_argument("--fwhm", type=float, default=vd_contrast_fwhm_arcmin)
    ap.add_argument("--out", default=None, help="JSON output path")
    ap.add_argument("--check", action="store_true", help="compare with the committed constants (rtol 1e-6)")
    a = ap.parse_args()
    xbar, neff = calibrate(a.catalogue, Path(a.data_dir), a.fwhm)
    print(paste("vd_contrast_xbar", xbar))
    print(paste("vd_contrast_neff", neff))
    for i in range(nbins):
        print(f"bin {i + 1}: contrast {neff[i, 0] / neff[i].mean():.3f} -> {neff[i, -1] / neff[i].mean():.3f}")
    if a.out:
        json.dump({"fwhm_arcmin": a.fwhm, "xbar": xbar.tolist(), "neff": neff.tolist()}, open(a.out, "w"), indent=1)
    if a.check:
        from src.KiDS.variable_depth_config import vd_contrast_neff, vd_contrast_xbar
        ok = np.allclose(xbar, vd_contrast_xbar, rtol=1e-6) and np.allclose(neff, vd_contrast_neff, rtol=1e-6)
        print("check vs committed constants:", "OK" if ok else "MISMATCH")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
