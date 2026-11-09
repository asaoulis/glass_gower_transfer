"""Constants, cluster-default paths and discrepancy thresholds for the validation.

The production defaults match src/KiDS/simulation_config.py. run_validation overrides them on its CLI.
"""

from __future__ import annotations

# --- survey and measurement geometry (mirrors src/KiDS/simulation_config.py) ---
NSIDE = 1024
LMAX = 2 * NSIDE          # 2048
NBINS = 6                 # tomographic bins -> 6*7/2 = 21 spectra
N_SPECTRA = NBINS * (NBINS + 1) // 2
LMIN_CUT = 56
LMAX_CUT = 1500
NBANDS = 8

# Theory recipe (theory.py): "shell_projection" (the sim's discrete shells) or "splined" (continuous CAMB windows).
THEORY_MODE = "shell_projection"
NONLINEAR = "NonLinear_both"
ZMIN, ZMAX = 0.0, 2.0
DX = 200.0
N_LOS_CHI = 1000
SHIFT_NZ = False          # clean theory test: no photo-z shifts

# --- cluster-default input paths ---
# The KiDS-Legacy pseudo-Cl mixing matrix (EE/EB/BB blocks, shape (3*(LMAX+1),)^2).
MIXING_MATRIX_PATH = "/share/gpu5/asaoulis/KiDS_Legacy_mixing_matrix_mask.npy"
# Tomographic n(z) + priors live under this data dir (read by calculate_tomo_nz).
DATA_DIR = "/home/asaoulis/projects/glass_transfer/kids-legacy-sbi/data"

# --- discrepancy thresholds ---
# |median(ratio) - 1| <= OK_FRAC          -> OK
#                      <= WARN_FRAC        -> WARNING
#                      >  WARN_FRAC        -> ERROR ("!!!")
OK_FRAC = 0.02
WARN_FRAC = 0.05

STATUS_OK = "OK"
STATUS_WARNING = "WARNING"
STATUS_ERROR = "ERROR"

# Exit codes surfaced by the driver (so validate-submit reflects pass/fail in SLURM).
EXIT_PASS = 0
EXIT_WARNINGS = 3
EXIT_ERRORS = 4


def spectrum_labels(nbins: int = NBINS):
    """Return the 21 spectrum labels in the order of the mixed_bandpowers rows.

    Lower triangle, row idx = i*(i+1)/2 + j for i >= j, label S{i+1}-S{j+1}. This order is the
    contract with the simulator output.
    """
    labels = [None] * (nbins * (nbins + 1) // 2)
    for i in range(nbins):
        for j in range(nbins):
            if i < j:
                continue
            idx = int(i * (i + 1) / 2 + j)
            labels[idx] = f"S{i + 1}-S{j + 1}"
    return labels
