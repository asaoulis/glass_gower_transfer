"""Constants, cluster-default paths and discrepancy thresholds for the validation.

Everything here is overridable on the :mod:`src.validation.run_validation` CLI; these
are just the sensible production defaults (matched to the KiDS-Legacy simulator config
in ``src/KiDS/simulation_config.py`` and the cluster's hardcoded paths).
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Survey / measurement geometry (mirrors src/KiDS/simulation_config.py) -------
NSIDE = 1024
LMAX = 2 * NSIDE          # 2048
NBINS = 6                 # tomographic bins -> 6*7/2 = 21 spectra
N_SPECTRA = NBINS * (NBINS + 1) // 2
LMIN_CUT = 56
LMAX_CUT = 1500
NBANDS = 8

# Theory recipe (see theory.py): "shell_projection" (Theory_B, matches the sim's discrete
# multi-plane lensing; removes the continuous-vs-discrete ℓ-tilt) or "splined" (Theory_A,
# continuous CAMB SplinedSourceWindow).
THEORY_MODE = "shell_projection"
# CAMB non-linear model for the splined recipe ONLY (shell mode inherits the sim's matter_cls
# settings -> effectively NonLinear_both, and IGNORES this knob). Default flipped to NonLinear_both
# (precision-logbook H9): for source-window lensing, NonLinear_lens governs the CMB lensing potential
# (NOT the source-window matter power), so the old _lens default fed ~linear shear Cls -> high-l
# deficit. _both is the correct choice if the splined path is ever used.
NONLINEAR = "NonLinear_both"
ZMIN, ZMAX = 0.0, 2.0
DX = 200.0
N_LOS_CHI = 1000
SHIFT_NZ = False          # clean theory test: no photo-z shifts

# --- Cluster-default input paths -------------------------------------------------
# The KiDS-Legacy pseudo-Cl mixing matrix (EE/EB/BB blocks, shape (3*(LMAX+1),)^2).
MIXING_MATRIX_PATH = "/share/gpu5/asaoulis/KiDS_Legacy_mixing_matrix_mask.npy"
# Tomographic n(z) + priors live under this data dir (read by calculate_tomo_nz).
DATA_DIR = "/home/asaoulis/projects/glass_transfer/kids-legacy-sbi/data"
# Gower Street cosmology table (only needed for sim-id -> cosmology lookups).
GOWER_CSV_PATH = (
    "/home/asaoulis/projects/glass_transfer/kids-legacy-sbi/data/gower_st/"
    "PKDGRAV3_on_DiRAC_DES_330.csv"
)

# --- Discrepancy thresholds (the core acceptance criteria) -----------------------
# |median(ratio) - 1| <= OK_FRAC          -> OK
#                      <= WARN_FRAC        -> WARNING
#                      >  WARN_FRAC        -> ERROR ("!!!")
# Tightened to the sub-5% precision bar (precision-logbook H11): for the real KiDS-Legacy analysis
# the best-measured bandpowers have sub-percent errors, so 5% is now the ERROR line, not OK.
OK_FRAC = 0.02
WARN_FRAC = 0.05

STATUS_OK = "OK"
STATUS_WARNING = "WARNING"
STATUS_ERROR = "ERROR"

# Exit codes surfaced by the driver (so validate-submit reflects pass/fail in SLURM).
EXIT_PASS = 0
EXIT_WARNINGS = 3
EXIT_ERRORS = 4


@dataclass(frozen=True)
class ValidationConfig:
    """Bundle of the knobs the driver threads through theory + ratios + diagnostics."""

    nside: int = NSIDE
    lmax: int = LMAX
    nbins: int = NBINS
    lmin_cut: int = LMIN_CUT
    lmax_cut: int = LMAX_CUT
    nbands: int = NBANDS
    nonlinear: str = NONLINEAR
    theory_mode: str = THEORY_MODE
    data_dir: str = DATA_DIR
    mixing_matrix_path: str | None = MIXING_MATRIX_PATH
    ok_frac: float = OK_FRAC
    warn_frac: float = WARN_FRAC


def spectrum_labels(nbins: int = NBINS):
    """Return the 21 spectrum labels in the SAME order as ``mixed_bandpowers`` rows.

    Lower triangle, row index ``idx = i*(i+1)/2 + j`` for ``i >= j``; label
    ``S{i+1}-S{j+1}``.  This ordering is the contract with the simulator output.
    """
    labels = [None] * (nbins * (nbins + 1) // 2)
    for i in range(nbins):
        for j in range(nbins):
            if i < j:
                continue
            idx = int(i * (i + 1) / 2 + j)
            labels[idx] = f"S{i + 1}-S{j + 1}"
    return labels
