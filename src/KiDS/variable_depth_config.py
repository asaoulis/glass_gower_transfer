"""KiDS-Legacy variable-depth (VD) constants + input-map loaders.

All numeric tables are extracted **mechanically** (via ``ast.literal_eval`` on the upstream
assignment, never retyped) from the reference VD driver
``Kiyam/kids-legacy-sbi @ 4a22578 : scripts/kids_legacy_sim_vd_cluster.py`` so the VD effect is
identically parameterised. ``kids_legacy_sim_vd_looped.py`` and ``kids_legacy_sim_vd_test.py`` at
the same rev carry byte-identical tables (checked), so the choice of source script is immaterial.

This module is deliberately OUTSIDE the protected ``src/cosmology/`` / ``src/KiDS/systematics.py``
tree: it holds only fixed survey constants and file loaders (no forward-model physics), mirroring
``src/KiDS/simulation_config.py``.

The VD effect is fixed/tied to the KiDS-Legacy parameters, so these are constants (no inference).

**2026-08-25 recalibration (upstream PR #2 `variable_depth_sim_v2`, merged in `5aef98d`).**
Re-derived from the new target-galaxy (TG) weights. Relative to the previous tables:

- ``vd_trace_edges`` is now **per-tomographic-bin**, ``(11,) -> (6, 11)``. Consumers must index it
  as ``vd_trace_edges[tomo]`` before digitising.
- ``vd_trace_eff_centre``, ``m_bias_vd``, ``m_bias_vd_unc`` — new values, same shapes.
- The galaxy-count contrast model changed **functional form**: the ``n_eff_table`` (10, 6) lookup +
  ``interp1d`` is replaced by a per-tomo **quadratic** in the tracer, ``a_ngal x^2 + b_ngal x +
  c_ngal``, clipped to ``[0.95, 1.8]`` and divided by ``n_arcmin2[i]``. ``n_eff_table`` is gone.
- The ``sigma_eps`` model dropped from **cubic to quadratic**: ``a_se x^2 + b_se x + c_se``
  (``d_se`` is gone), clip window ``[0.25, 0.34]`` unchanged.
- ``load_vd_maps`` now reads per-tomo SOM-derived maps named by the **integer tomo index**, not by
  the ZB label.

Verified UNCHANGED at the same rev (do not touch): ``alpha_1/2(_unc)``, the N/S additive c-bias
tables, the PSF map filenames, ``m_bias(_unc)``, ``n_arcmin2``, ``sigma_e``.
"""
from pathlib import Path

import healpy as hp
import numpy as np

from .tomo import nbins


# Number of variable-depth tracer bins (digitised galaxy-depth levels).
n_vardepth_bins = 10

# Edges (nbins, 11) used to digitise each tomo bin's VD tracer map into `n_vardepth_bins` bins.
# NOTE: per-tomographic-bin since the 2026-08 recalibration — index as `vd_trace_edges[tomo]`.
vd_trace_edges = np.array([
    [0.0012413758210859523, 4.430003360530595, 5.252902596734906, 5.6069376920719805,
     5.849180679713157, 6.047681462039527, 6.226779225434265, 6.386899380773178,
     6.6054691679437685, 6.82492142049382, 9.61218319807485],
    [0.0073206554886097955, 4.3519703656718915, 5.042847778235911, 5.4514463808894424,
     5.76953791163161, 6.066261652560526, 6.319031718362657, 6.574645469556037,
     6.849620636233027, 7.234391329757189, 9.624202610438527],
    [0.0034045911077663104, 4.006400906296537, 4.6389099075768865, 5.01044403858862,
     5.313712089734094, 5.6105282039491735, 5.86639599359827, 6.1178102435428485,
     6.414026258694012, 6.798519346025607, 8.125587645678067],
    [0.002902690366637731, 4.005754618595313, 4.634405698404758, 5.045160139490447,
     5.386063755542248, 5.719650671923531, 6.035446094206181, 6.3427188086790185,
     6.697604035415804, 7.174785642908927, 8.91069579795505],
    [0.004044487269890585, 3.650608396271394, 4.288082857810785, 4.748061781527252,
     5.143633190655405, 5.503984279926088, 5.850569878425522, 6.200597990602988,
     6.586286201576565, 7.081025789392527, 9.858601952761187],
    [0.0026781022501058942, 3.1465005989290735, 3.8484736431953497, 4.41990867194422,
     4.937602576409741, 5.402087764638201, 5.856581090965961, 6.368597582246347,
     6.974696826475142, 7.732414146464567, 11.473029380376559],
])

# Effective tracer centre per (tomo bin, VD bin): (6, 10).
vd_trace_eff_centre = np.array([
    [3.19053923, 4.91536702, 5.44340465, 5.73704696, 5.94714644, 6.14196638,
     6.30399688, 6.49182295, 6.71424724, 7.12728813],
    [3.17297406, 4.75112099, 5.25041016, 5.61626858, 5.91691688, 6.19630868,
     6.44624204, 6.70763686, 7.01402954, 7.67007619],
    [2.93432037, 4.37055504, 4.83299341, 5.16245183, 5.46605344, 5.74078023,
     5.9918383, 6.25970959, 6.59893062, 7.14381448],
    [2.95117514, 4.3664851, 4.84606198, 5.22062422, 5.55075732, 5.88256008,
     6.18891067, 6.51183721, 6.9249573, 7.54802104],
    [2.72627722, 4.00285001, 4.52452636, 4.94974528, 5.32578628, 5.67697522,
     6.02622737, 6.39228781, 6.81118459, 7.53480551],
    [2.39086186, 3.5116398, 4.1437544, 4.68218203, 5.17117532, 5.62566507,
     6.10448105, 6.66362351, 7.33460146, 8.42028027],
])

# Quadratic n_gal(VD tracer) model coefficients per tomo bin (6,). Replaces the old
# `n_eff_table` (10, 6) + interp1d lookup. Used clipped to [0.95, 1.8] and divided by
# `n_arcmin2[i]` to form the raw count contrast — see `sim_utils.build_variable_depth`.
a_ngal = np.array([-0.001047, -0.001979, -0.000701, -0.004175, -0.004781, -0.001811])
b_ngal = np.array([0.007246, 0.015826, 0.005413, 0.030059, 0.029338, -0.000527])
c_ngal = np.array([1.764172, 1.627612, 1.489635, 1.427104, 1.334675, 1.129347])

# Clip window for the raw count contrast, applied BEFORE the /n_arcmin2 division.
n_contrast_clip = (0.95, 1.8)

# Quadratic sigma_eps(VD tracer) model coefficients per tomo bin (6,). Was cubic before the
# 2026-08 recalibration (the `d_se` constant term is gone).
a_se = np.array([-0.000439, 0.000224, -0.000224, 0.000194, -8.7e-05, 0.000366])
b_se = np.array([0.003677, -0.001357, 0.003897, -0.001368, -0.001988, -0.004989])
c_se = np.array([0.270832, 0.271273, 0.275564, 0.263044, 0.293597, 0.315414])

# Clip window for the raw sigma_eps model, applied before the mask rescaling.
sigma_eps_clip = (0.25, 0.34)

# Per-(tomo, VD bin) multiplicative shear bias mean/uncertainty: (6, 10).
m_bias_vd = np.array([
    [-0.025428, -0.028753, -0.0374, -0.014375, -0.014084, -0.019481,
     -0.007478, -0.032551, -0.026265, -0.016866],
    [-0.011662, -0.012319, -0.027961, -0.011666, -0.010999, -0.02331,
     -0.029309, 0.005671, -0.020734, -0.013564],
    [0.006109, -0.021844, -0.004229, -0.002563, -0.032725, -0.017969,
     -0.001816, 0.001186, -0.015431, -0.021066],
    [0.019529, 0.004984, 0.027373, 0.029931, 0.021429, 0.00141,
     0.023624, 0.022251, 0.021044, 0.027836],
    [0.004625, 0.037177, 0.040486, 0.038934, 0.022625, 0.010299,
     0.029334, 0.041678, 0.015215, 0.058892],
    [0.039822, 0.048569, 0.052208, 0.041547, 0.05653, 0.057191,
     0.035376, 0.049524, 0.025109, 0.03705],
])
m_bias_vd_unc = np.array([
    [0.019623, 0.019732, 0.019666, 0.019647, 0.019711, 0.019743,
     0.019599, 0.019727, 0.019723, 0.019557],
    [0.019549, 0.019707, 0.019765, 0.019656, 0.019638, 0.019621,
     0.019532, 0.019583, 0.01959, 0.019519],
    [0.023366, 0.02339, 0.023481, 0.023351, 0.02327, 0.023261,
     0.023154, 0.023248, 0.023152, 0.023036],
    [0.021358, 0.021391, 0.021276, 0.021349, 0.021317, 0.021391,
     0.021316, 0.021303, 0.021285, 0.021385],
    [0.023294, 0.023332, 0.023354, 0.023357, 0.02332, 0.023368,
     0.023388, 0.023374, 0.023375, 0.023474],
    [0.026307, 0.026361, 0.026551, 0.026549, 0.026657, 0.026753,
     0.026777, 0.026947, 0.02708, 0.027147],
])

# Residual-PSF additive shear bias coefficients (alpha) per tomo bin (6,).
# Unchanged by the 2026-08 recalibration.
alpha_1     = np.array([-0.003, -0.007, -0.003, -0.004, -0.005, -0.004])
alpha_2     = np.array([ 0.013,  0.008,  0.007,  0.008,  0.007,  0.015])
alpha_1_unc = np.array([ 0.015,  0.013,  0.013,  0.013,  0.013,  0.014])
alpha_2_unc = np.array([ 0.011,  0.010,  0.011,  0.011,  0.011,  0.011])


def zb_label(i: int) -> str:
    """The ``ZB{z1}t{z2}`` label for tomo bin `i` as the ``nz_tgweights`` files spell it.

    Those filenames strip trailing zeros in the redshift labels (``0.10 -> "0p1"``,
    ``0.90 -> "0p9"``, ``2.00 -> "2p0"``), unlike the ``nofzs/nz/`` files read by
    ``src/KiDS/tomo.py``, which keep them (``"0p10"``). Round-tripping through ``float``
    reproduces the on-disk spelling; verified against all six files on disk.
    """
    from .tomo import ztomo_label

    z1, z2 = (str(float(e)).replace(".", "p") for e in ztomo_label[i])
    return f"ZB{z1}t{z2}"


def load_vd_maps(data_dir: str | Path, nside: int) -> np.ndarray:
    """Load the 6 per-tomo VD tracer maps, ud_grade-ing to the run ``nside``.

    Since the 2026-08 recalibration these are per-tomo SOM-derived maps indexed by the integer
    tomo index (upstream `80591f5` "Fixed vd_map filename logic"), not by the ZB label.

    Returns an array of shape ``(nbins, hp.nside2npix(nside))``.
    """
    data_dir = Path(data_dir)
    vd_map = np.empty((nbins, hp.nside2npix(nside)))
    for i in range(nbins):
        m = hp.read_map(
            data_dir / "vd_maps"
            / f"KiDSLegacy_ORweight_tomo{i}_1024_eachSOM_SNcoadd.fits"
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
