"""Fixed survey and simulation constants for the KiDS-Legacy mocks.

Grids (nside, lmax, shells), tomographic bins, bandpower scales, N/S patches, shape-noise
augmentation counts, galaxy-bias prior presets and the CAMB-Cls cache location.
"""
from pathlib import Path

import healpy as hp
import numpy as np


nbins = 6
bias = 1

# Galaxy-bias prior presets (O3-diag): per block, independent per-bin b_i ~ N(mean_i, kappa * sigma_i),
# truncated at +-3 kappa sigma.
GALAXY_BIAS_PRIOR_MEANS = [1.0181, 1.0698, 1.1302, 1.2427, 1.3739, 1.4805]
GALAXY_BIAS_PRIOR_SIGMAS = [0.1801, 0.1491, 0.1252, 0.0951, 0.0960, 0.0985]
GALAXY_BIAS_PRIORS = {
	"flamingo_pt_diag": {"kappa": 1.0},
	"flamingo_pt_diag_rob": {"kappa": 1.5},
	# kappa=2: a deliberately wide prior so b_g is a broadly sampled nuisance.
	"flamingo_pt_diag_k2": {"kappa": 2.0},
	# kappa=0: b_g pinned at the per-bin means (no scatter, truncation is a no-op).
	"flamingo_pt_diag_mean": {"kappa": 0.0},
}
GALAXY_BIAS_CLIP = (0.3, 2.2)

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


# Shared CAMB-Cls cache keyed by sim_id. Keep it separate from any per-variate output_dir.
CAMB_CLS_CACHE_DIR = Path("/share/gpu5/asaoulis/camb_cls_cache")

# Base seed of the per-sim_id cosmology draw. Changing it regenerates every cosmology.
COSMO_BASE_SEED = 0


# Shape-noise augmentation: outer = independent realisations (reload the N-body shells), inner = per-rotation.
OUTER_NUM_SHAPE_NOISE_REALISATIONS = {
	# external: 1 outer x 5 rot x 1 inner x 4 mask = 20 mocks/cosmology (disjoint rotations).
	"external": 1,
	"gower_street": 4,
	"glass": 4,
}

INNER_NUM_SHAPE_NOISE_REALISATIONS = {
	# inner > 1 multiplies per-rank memory and runs out of memory; production uses 1.
	"external": 1,
	"gower_street": 1,
	"glass": 1,
}

lower_lscale = 56
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
