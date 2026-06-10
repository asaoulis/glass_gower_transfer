from pathlib import Path

import healpy as hp
import numpy as np


nbins = 6
bias = 1

nside = 1024
n_ell = 20
lmax = 2 * nside
lmin = 0

zmin, zmax = 0.0, 2.0
dx = 200.0  # Mpc/h
n_los_chi = 1000

mask_rotation_angles = [0, 90, 180, 270]
OVERWRITE = False


# Grouped configs used by the KiDS legacy simulators.
SIM_GRID = {
	"nside": nside,
	"lmax": lmax,
}

LOS_GRID = {
	"zmin": zmin,
	"zmax": zmax,
	"dx": dx,
}

CAMB_LIMITS = {
	"mem_limit_gb": 200,
	"timeout_s": 3600 * 4,
}


# Shared, persistent cache of CAMB-computed matter Cls (shells + glass_cls), keyed by sim_id.
# MUST be separate from any per-variate `output_dir` so that different analysis variates
# (e.g. NLA-M, NLA-z, clustering, post-proc) reuse the same expensive CAMB products. The first
# variate to run a given sim_id populates the cache; later variates load it and skip CAMB.
CAMB_CLS_CACHE_DIR = Path("/share/gpu5/asaoulis/camb_cls_cache")

# Base seed for deterministic per-sim_id cosmology sampling. Combined with sim_id it fixes the
# cosmology drawn for each sim_id across runs/variates. Change only to deliberately regenerate a
# different fixed cosmology set (the on-disk cache guard will then flag stale entries).
COSMO_BASE_SEED = 0


# Shape-noise augmentation counts (outer = independent realisations; inner = per-rotation augmentations).
# NOTE: gower_street outer set to 1 for the theory-test runs (outer reloads the N-body backend,
# ~40 min each; inner+mask augmentations are cheap). Restore to 4 for production Gower datasets.
OUTER_NUM_SHAPE_NOISE_REALISATIONS = {
	"gower_street": 4,
	"glass": 4,
}

INNER_NUM_SHAPE_NOISE_REALISATIONS = {
	"gower_street": 4,
	"glass": 1,
}

lower_lscale = 76
upper_lscale = 1500
nbands = 8

named_patches = {
	"south": (12, -31, 90, 11),
	"north": (-178, 0, 112, 10),
}

patches = list(named_patches.values())


def load_kids_mask(data_dir: str | Path) -> np.ndarray:
	data_dir = Path(data_dir)
	return hp.read_map(
		data_dir / "masks" / "KiDS_Legacy_N_healpix_1024_frac_withAstrom.fits"
	) + hp.read_map(
		data_dir / "masks" / "KiDS_Legacy_S_healpix_1024_frac_withAstrom.fits"
	)
