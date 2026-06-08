"""KiDS-Legacy variable-depth (VD) constants + input-map loaders.

All numeric tables are copied **verbatim** from the reference VD driver
``kids-legacy-sbi/kids-legacy-sbi/scripts/kids_legacy_sim_vd_looped.py`` (lines ~335-528) so the
VD effect is identically parameterised. This module is deliberately OUTSIDE the protected
``src/cosmology/`` / ``src/KiDS/systematics.py`` tree: it holds only fixed survey constants and
file loaders (no forward-model physics), mirroring ``src/KiDS/simulation_config.py``.

The VD effect is fixed/tied to the KiDS-Legacy parameters, so these are constants (no inference).
"""
from pathlib import Path

import healpy as hp
import numpy as np

from .tomo import nbins, ztomo_label


# Number of variable-depth tracer bins (digitised galaxy-depth levels).
n_vardepth_bins = 10

# Edges (11,) used to digitise the VD tracer map into `n_vardepth_bins` bins.
vd_trace_edges = np.array([
    0.00039586402840050813, 4.333629902071316, 5.41842919221884,
    5.966776946566387,      6.363840923950448, 6.705373252332217,
    7.027559883740965,      7.346859671979755, 7.67948179599049,
    8.142430491846039,     11.324789884409146,
])

# Effective tracer centre per (tomo bin, VD bin): (6, 10).
vd_trace_eff_centre = np.array([
    [2.64789938, 4.95437723, 5.72446037, 6.18416233, 6.5389531,
    6.8664036,  7.18752662, 7.49721572, 7.84279037, 8.41169263],
    [2.6010878,  4.95463555, 5.72353875, 6.17492315, 6.5381264,
    6.8662059,  7.18740554, 7.51358455, 7.9001472,  8.58291616],
    [2.66567886, 4.9882028,  5.71111977, 6.16986798, 6.53738992,
    6.86853353, 7.18556119, 7.50993119, 7.88656406, 8.38171309],
    [2.63998023, 4.97203209, 5.71254017, 6.17169432, 6.53822272,
    6.86750078, 7.19258744, 7.51312043, 7.90339967, 8.50940635],
    [2.75105258, 4.97090594, 5.70153218, 6.1690816,  6.53424077,
    6.87010161, 7.18967823, 7.50957196, 7.89454552, 8.52311655],
    [2.86437597, 4.92491669, 5.6973029,  6.17278079, 6.53739508,
    6.8680397,  7.18974719, 7.51441527, 7.90650554, 8.99924988],
])

# Effective galaxy number density per (VD bin, tomo bin): (10, 6).
n_eff_table = np.array([
    [1.84903444, 1.50820069, 1.58883612, 1.4614307,  1.5544033,  1.27138885],
    [1.55903354, 1.19314162, 1.71511607, 1.47817089, 2.06345788, 1.42060107],
    [1.755447,   1.32978093, 1.9952575,  1.5497238,  1.63370181, 0.90045119],
    [2.28572619, 1.51414118, 1.73943957, 1.39805187, 1.32345546, 0.71152113],
    [2.52706076, 1.57223635, 1.5760765,  1.32843467, 1.24480019, 0.64055796],
    [2.52702181, 1.5744785,  1.57234357, 1.3347428,  1.2063014,  0.65748268],
    [2.33148385, 1.55980374, 1.60864612, 1.41904153, 1.27416397, 0.70289423],
    [2.08357571, 1.70440906, 1.56535879, 1.59006323, 1.20941186, 0.77011492],
    [0.88825301, 2.31658281, 1.47390373, 1.96691002, 1.39756221, 1.08143991],
    [0.29974931, 2.85776421, 0.6418011,  1.61584575, 1.09864345, 3.1430868 ],
])

# Cubic sigma_eps(VD tracer) model coefficients per tomo bin (6,).
a_se = np.array([ 0.000403,  0.000158, -1.9e-05,  0.000275, -0.000428, -5.8e-05])
b_se = np.array([-0.006561, -0.002528,  0.00025, -0.004668,  0.006636,  0.001098])
c_se = np.array([ 0.032342,  0.012569,  0.000116, 0.024811, -0.032343, -0.007647])
d_se = np.array([ 0.232259,  0.252409,  0.284059, 0.221895,  0.329548,  0.319706])

# Per-(tomo, VD bin) multiplicative shear bias mean/uncertainty: (6, 10).
m_bias_vd = np.array([
    [-0.033754, -0.026962, -0.034473, -0.019362, -0.015617, -0.019519,
    -0.012956, -0.027953, -0.022462, -0.040446],
    [-0.020141, -0.016864, -0.026674, -0.025799, -0.025235, -0.0029,
    -0.011385, -0.012112, -0.016647, -0.015067],
    [-0.002508, -0.007223, -0.01488,  -0.008141, -0.01835,  -0.010122,
    -0.024149, -0.002857, -0.007611, -0.013968],
    [ 0.023157,  0.000612,  0.025447,  0.013191,  0.01541,   0.021786,
    0.023982,  0.015922,  0.029325,  0.022082],
    [ 0.02566,   0.03476,   0.031331,  0.029178,  0.028164,  0.029091,
    0.016602,  0.021035,  0.026226,  0.05884 ],
    [ 0.032336,  0.052092,  0.054903,  0.032802,  0.063345,  0.036114,
    0.027017,  0.060289,  0.042254,  0.041393],
])
m_bias_vd_unc = np.array([
    [0.019524, 0.021472, 0.019926, 0.017265, 0.016387, 0.016348,
    0.016989, 0.018111, 0.028343, 0.050147],
    [0.02068,  0.023506, 0.021966, 0.020398, 0.019943, 0.019986,
    0.020069, 0.019282, 0.01693,  0.015682],
    [0.023006, 0.022585, 0.020483, 0.021865, 0.022761, 0.022631,
    0.022427, 0.02281,  0.023472, 0.036736],
    [0.021581, 0.021588, 0.020667, 0.021542, 0.022082, 0.022151,
    0.021611, 0.020624, 0.01909,  0.022047],
    [0.022363, 0.019711, 0.021506, 0.023631, 0.024188, 0.024585,
    0.024031, 0.024637, 0.023461, 0.027449],
    [0.025277, 0.023789, 0.028864, 0.031624, 0.033004, 0.032607,
    0.03178,  0.030784, 0.026866, 0.017471],
])

# Residual-PSF additive shear bias coefficients (alpha) per tomo bin (6,).
alpha_1     = np.array([-0.003, -0.007, -0.003, -0.004, -0.005, -0.004])
alpha_2     = np.array([ 0.013,  0.008,  0.007,  0.008,  0.007,  0.015])
alpha_1_unc = np.array([ 0.015,  0.013,  0.013,  0.013,  0.013,  0.014])
alpha_2_unc = np.array([ 0.011,  0.010,  0.011,  0.011,  0.011,  0.011])


def _zb_label_parts(i: int) -> tuple[str, str, str, str]:
    """Split tomo-bin `i`'s z-edge labels into the ``ZB{z1a}p{z1b}t{z2a}p{z2b}`` parts."""
    z1a, z1b = ztomo_label[i][0].split(".")
    z2a, z2b = ztomo_label[i][1].split(".")
    return z1a, z1b, z2a, z2b


def load_vd_maps(data_dir: str | Path, nside: int) -> np.ndarray:
    """Load the 6 per-tomo VD tracer maps, ud_grade-ing to the run ``nside``.

    Returns an array of shape ``(nbins, hp.nside2npix(nside))``.
    """
    data_dir = Path(data_dir)
    vd_map = np.empty((nbins, hp.nside2npix(nside)))
    for i in range(nbins):
        z1a, z1b, z2a, z2b = _zb_label_parts(i)
        m = hp.read_map(
            data_dir / "vd_maps"
            / f"kids_legacy_ORweight_600_1024_ZB{z1a}p{z1b}t{z2a}p{z2b}_SNcoadd.fits"
        )
        if hp.get_nside(m) != nside:
            m = hp.ud_grade(m, nside)
        vd_map[i] = m
    return vd_map


def load_psf_maps(data_dir: str | Path, nside: int) -> tuple[np.ndarray, np.ndarray]:
    """Load the PSF e1/e2 residual maps, ud_grade-ing to the run ``nside``."""
    data_dir = Path(data_dir)
    psf_bias_map_1 = hp.read_map(data_dir / "psfmaps" / "psf_e1_Map_KiDS-Legacy_1024_ALL.fits")
    psf_bias_map_2 = hp.read_map(data_dir / "psfmaps" / "psf_e2_Map_KiDS-Legacy_1024_ALL.fits")
    if hp.get_nside(psf_bias_map_1) != nside:
        psf_bias_map_1 = hp.ud_grade(psf_bias_map_1, nside)
    if hp.get_nside(psf_bias_map_2) != nside:
        psf_bias_map_2 = hp.ud_grade(psf_bias_map_2, nside)
    return psf_bias_map_1, psf_bias_map_2
