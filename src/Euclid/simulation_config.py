"""Fixed survey / simulation constants for the Euclid DR3 forecast mocks.

Counterpart of `src/KiDS/simulation_config.py`. Consumed by `master_euclid_simulator.py`.

Survey specification (origin/Euclid README "Euclid DR3 details"):
  * 14 000 deg^2 footprint, a full-RA declination band centred on the equator
  * 13 tomographic bins from `euclid_nzs_dr3.txt`
  * n = 6.2 gal/arcmin^2 TOTAL across the 13 bins (user-confirmed 2026-09-09)
  * sigma_e = 0.26 PER COMPONENT (user-confirmed 2026-09-09; matches the KiDS convention in
    `src/KiDS/systematics.py`, which passes 0.2619-0.3002 per component to
    `glass.shapes.ellipticity_intnorm`)
  * no survey systematics: no IA, no variable depth, no m/c shear bias, no photo-z shifts
"""

from pathlib import Path

import healpy as hp
import numpy as np

# --- Tomography / shape noise ----------------------------------------------------
nbins = 13
bias = 1                      # linear source-galaxy clustering bias b_g
sigma_e_per_component = 0.26
sigma_e = np.full(nbins, sigma_e_per_component)

# Total galaxy number density across ALL bins; the per-bin split comes from the n(z) file
# (`src/Euclid/tomo.py:build_tomo_nz`), which for DR3 is 13 equal-population bins.
N_ARCMIN2_TOTAL = 6.2

# --- Simulation grid -------------------------------------------------------------
nside = 1024
lmax = 2 * nside              # 2048
nside_out = 512               # E-map output resolution (6.87' pixels)

# zmax must cover the n(z) support (euclid_nzs_dr3.txt ends at z = 2.994).
zmin, zmax = 0.0, 3.05
dx = 200.0                    # Mpc/h shell thickness
n_los_chi = 1000

SIM_GRID = {"nside": nside, "lmax": lmax}
LOS_GRID = {"zmin": zmin, "zmax": zmax, "dx": dx}
CAMB_LIMITS = {"mem_limit_gb": 200, "timeout_s": 3600 * 4}

# --- Bandpowers ------------------------------------------------------------------
# Same binning as the current KiDS production arm (src/KiDS/simulation_config.py).
lower_lscale = 56
upper_lscale = 1500
nbands = 8

# --- The single stored E-map variant ---------------------------------------------
# User decision 2026-09-09: bandpowers + ONE E variant only. `fwhm` is a gentle beam taper
# (4' is SUB-PIXEL at nside_out 512, whose pixels are 6.87'); the effective bandlimit is `lcut`.
EB_FWHM_ARCMIN = 4.0
EB_LMIN = None
EB_LCUT = 1400
MAP_DTYPE = np.float16

# --- Augmentation ----------------------------------------------------------------
# The footprint is a full-RA band, so it is EXACTLY invariant under any RA (Z-axis) rotation
# (verified numerically, task euclid/dataset-generation step A4). Latitude rotations cannot be
# undone by a Dec crop (step A2), so there is no field-rotation augmentation at all: one
# identity rotation, one mask angle, one inner shape-noise draw. Independent realisations come
# from the OUTER loop only.
IDENTITY_ROTATION = {"rot": [0, 0, 0], "flip": False, "backend": "identity"}
ROTATION_SPECS = [IDENTITY_ROTATION]
MASK_ROTATION_ANGLES = [0]
INNER_NUM_SHAPE_NOISE_REALISATIONS = 1
OUTER_NUM_SHAPE_NOISE_REALISATIONS = 1

OVERWRITE = False

# --- Cosmology sampling ----------------------------------------------------------
# Same Gower Street empirical prior as the KiDS master (w0 only; wa is NOT sampled).
COSMO_PARAM_NAMES = ["omega_m", "sigma_8", "ombh2", "h", "ns", "w0", "mnu"]
EUCLID_N_JOBS = 50000
COSMO_BASE_SEED = 0

# MUST be distinct from the KiDS cache: the cached-Cls guard keys on the cosmology AND the
# (lmax, zmin, zmax, dx) grid, and Euclid's zmax=3.05 differs from KiDS's 2.0, so sharing the
# KiDS directory would make every lookup a (correct) guard failure.
CAMB_CLS_CACHE_DIR = Path("/share/gpu5/asaoulis/camb_cls_cache_euclid")

# --- Footprint -------------------------------------------------------------------
FOOTPRINT_AREA_DEG2 = 14000.0


def dec_half_width_deg(area_deg2=FOOTPRINT_AREA_DEG2):
    """Half-width (deg) of a symmetric, full-RA Dec band of the given solid angle.

    A band |Dec| <= d spanning all RA subtends Omega = 4*pi*sin(d).
    14 000 deg^2 -> d = 19.8385 deg, f_sky = 0.3394.
    """
    area_sr = area_deg2 * np.deg2rad(1.0) ** 2
    return float(np.rad2deg(np.arcsin(area_sr / (4 * np.pi))))


DEC_HALF_DEG = dec_half_width_deg()
PATCH_HALFWIDTH_DEG = DEC_HALF_DEG / 2.0

# Cartesian crop boxes (lon_centre, lat_centre, lon_range, lat_range) in degrees, in the format
# `src/cosmology/pixelise_maps.py:get_patch_values` expects. The band is split at Dec = 0 into
# two equal halves so the HDF5 read contract `pixelised_results/E/{north,south}` is unchanged.
#
# These are PLAIN plate-carree crops with NO recentring rotation. The origin/Euclid branch's
# `get_recentred_patch_values` attempted to rotate each half onto the equator first; that is
# geometrically impossible for a full-RA band (a Y-axis rotation maps a Dec annulus onto an
# annulus about a TILTED axis) and silently loses ~50 % of the footprint -- see
# `.claude/runs/euclid/dataset-generation/artifacts/rotation_bug.png`. The plain crop costs at
# most cos(19.84 deg) = 0.94 of projection distortion at the band edge, which is milder than the
# distortion already accepted in the KiDS `named_patches`.
named_patches = {
    "north": (180.0, +PATCH_HALFWIDTH_DEG, 360.0, 2 * PATCH_HALFWIDTH_DEG),
    "south": (180.0, -PATCH_HALFWIDTH_DEG, 360.0, 2 * PATCH_HALFWIDTH_DEG),
}
patches = list(named_patches.values())


def build_footprint(mask_nside=nside, dec_half_deg=None):
    """Boolean HEALPix mask of the equatorial band at `mask_nside`."""
    dec_half_deg = DEC_HALF_DEG if dec_half_deg is None else dec_half_deg
    theta_min = np.deg2rad(90.0 - dec_half_deg)
    theta_max = np.deg2rad(90.0 + dec_half_deg)
    footprint = np.zeros(hp.nside2npix(mask_nside), dtype=bool)
    footprint[hp.query_strip(mask_nside, theta_min, theta_max)] = True
    return footprint


def patch_grid_shape(nside_out_=nside_out):
    """(n_lat, n_lon) of one stored half-band patch, matching get_patch_values' arange grid."""
    res = np.degrees(hp.nside2resol(nside_out_))
    n_lat = np.arange(-PATCH_HALFWIDTH_DEG, PATCH_HALFWIDTH_DEG, res).size
    n_lon = np.arange(0.0, 360.0, res).size
    return int(n_lat), int(n_lon)
