"""Build the KiDS-Legacy patch footprint fraction used by the HOS bake (src/ml/data/hos.py).

The KiDS frac masks go to nside 512 and through the simulator's patch projection at rot_angle=0.
Usage: python scripts/build_hos_mask.py [--data-dir kids-legacy-sbi/data] [--out <npz>]
"""
import argparse
from pathlib import Path

import healpy as hp
import numpy as np

from src.KiDS.simulation_config import load_kids_mask, named_patches
from src.cosmology.pixelise_maps import get_patch_values


def build(data_dir):
    m512 = hp.ud_grade(load_kids_mask(Path(data_dir)), 512)
    pv = get_patch_values([m512], list(named_patches.values()), 512, 0)
    return {side: np.asarray(pv[i])[0].astype(np.float32) for i, side in enumerate(named_patches)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="kids-legacy-sbi/data")
    ap.add_argument("--out", default="config/hos_assets/kids_patch_mask_frac_nside512.npz")
    a = ap.parse_args()
    out = build(a.data_dir)
    np.savez_compressed(a.out, **out)
    print({s: (v.shape, int((v > 0.5).sum())) for s, v in out.items()}, "->", a.out)
