from pathlib import Path

import healpy as hp
import numpy as np


nbins = 6
bias = 1

# Galaxy-bias prior presets (O3-diag, task simulation-runs/galaxy-bias-priors).
# Per (sim, outer, rot) block draw of an independent per-tomo-bin 6-vector
# b_i ~ N(mean_i, kappa * sigma_i), truncated at +-3*kappa*sigma_i and clipped to
# GALAXY_BIAS_CLIP. Means/sigmas are the Flamingo KiDS-Legacy calibration
# (data/galaxy_bias_priors.txt: columns b and TOTAL). kappa=1 is the calibrated
# width; the _rob preset inflates it for robustness (pending final width decision).
GALAXY_BIAS_PRIOR_MEANS = [1.0181, 1.0698, 1.1302, 1.2427, 1.3739, 1.4805]
GALAXY_BIAS_PRIOR_SIGMAS = [0.1801, 0.1491, 0.1252, 0.0951, 0.0960, 0.0985]
GALAXY_BIAS_PRIORS = {
	"flamingo_pt_diag": {"kappa": 1.0},
	"flamingo_pt_diag_rob": {"kappa": 1.5},
	# kappa=3 — the "galaxy bias as a VARIATE" prior (user, 2026-08-18). Deliberately much wider than
	# a calibrated prior: the point is to make b_g a broadly-sampled nuisance the network must learn to
	# marginalise, not to encode a belief about its value.
	# ⚠️ AT kappa=3 THE CLIP IS ACTIVE AND RESHAPES THE PRIOR. With GALAXY_BIAS_CLIP=(0.3, 2.2):
	#   bin1 (mu 1.018, sigma_3 0.540): nominal +-3sigma range [-0.603, 2.639] -> ~10.4 % of draws clip
	#   bin6 (mu 1.480, sigma_3 0.295): [0.594, 2.367]                          -> ~0.6 % clip
	# (kappa=1 clips 0.00 %, kappa=1.5 clips 0.26 % — so this is new behaviour at kappa=3, not a
	# continuation.) The low-z bins would otherwise reach NEGATIVE bias, which is unphysical, so the
	# clip is doing real work rather than trimming a tail. Consequence: bin1's effective prior is
	# closer to "wide with a pile-up at 0.3" than to a Gaussian, and the realised expansion is < 3x
	# for the low bins. See the variate write-up in
	# .claude/runs/training-runs/production-training-runs/ — the clip width is an OPEN USER DECISION.
	"flamingo_pt_diag_k3": {"kappa": 3.0},
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
	# Prod value is 1: 4 outer x 5 rot x 1 inner x 4 mask = 80 augs/sim (matches the on-disk
	# gower_mocks dataset). inner>1 multiplies per-rank memory and OOMs; the eb85842 cleanup
	# wrongly bumped this 1->4 (=320 augs/sim). Keep at 1 for production Gower datasets.
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
