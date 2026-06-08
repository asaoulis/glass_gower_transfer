"""Per-cosmology *theory* pseudo-Cl bandpowers for the validation.

This reproduces the recipe used in ``theory_tests_systematics.py`` (and the final,
``NonLinear_lens``-enabled cells of ``sims_analysis.ipynb``):

    build_cosmology(params, include_baryons=False)            # gravity-only Mead2020
      -> CAMB SplinedSourceWindow per tomographic bin (lensing)
      -> get_source_cls_dict(lmax, raw_cl=True)               # EE per W_i x W_j
      -> assemble full (EE, EB=0, BB=0) vector T
      -> pseudo = mixing_matrix @ T                           # KiDS-Legacy MCM
      -> multiply EE block by the pol pixel window pw**2
      -> cut to [lmin_cut, lmax_cut] and bin into `nbands` bandpowers

Resolved decision (NonLinear_lens):
    ``theory_tests_systematics.py`` sets ``pars.NonLinear = "NonLinear_lens"`` before
    ``camb.get_results``.  The saved notebook cell body omitted it, but the later cells
    are explicitly labelled "WITH NonLinear_lens theory now!" — i.e. the kept runs used
    it.  We therefore default to ``nonlinear="NonLinear_lens"`` (the correct non-linear
    lensing convention; it materially shifts the high-ℓ ratio).  It is exposed as a knob
    so an on/off comparison is a one-line change.

Only read-only imports of the protected physics modules are used here.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import healpy as hp
import camb
import glass
import glass.shells

from src.cosmology.parameters import build_cosmology
from src.cosmology.manip_cls import compute_cl_bandpowers
from src.KiDS.tomo import calculate_tomo_nz

from . import config as cfg

CosmoVec = Tuple[Sequence[float], Sequence[str]]


def params_from_cosmo_vec(cosmo_vec: CosmoVec) -> Dict[str, float]:
    """Extract the CAMB cosmology parameters from an ``unpack_data`` cosmo vector.

    ``cosmo_vec`` is ``(values, names)`` as returned by
    ``unpack_data(..., return_names=True)``.  Only the 7 parameters CAMB needs are
    pulled; extra saved keys (e.g. ``a_ia``, ``b_ia``) are ignored.
    """
    values, names = cosmo_vec
    p = dict(zip(names, [float(v) for v in values]))
    return {
        "h": float(p["h"]),
        "ombh2": float(p["ombh2"]),
        "omega_m": float(p["omega_m"]),
        "ns": float(p["ns"]),
        "w0": float(p["w0"]),
        "sigma_8": float(p["sigma_8"]),
        "mnu": float(p.get("mnu", 0.0)),
    }


def load_mixing_matrix(path: str, lmax: int = cfg.LMAX) -> np.ndarray:
    """Load the KiDS mixing matrix and sanity-check its shape (3*(lmax+1) blocks)."""
    mms = np.load(path)
    n = mms.shape[0] // 3
    if n != lmax + 1:
        raise ValueError(
            f"mixing matrix block size {n} != lmax+1 ({lmax + 1}); "
            f"matrix shape {mms.shape} incompatible with lmax={lmax}"
        )
    return mms


def compute_bandpower_theory_from_cosmo_vec(
    cosmo_vec: CosmoVec,
    *,
    nside: int = cfg.NSIDE,
    nbins: int = cfg.NBINS,
    lmin_cut: int = cfg.LMIN_CUT,
    lmax_cut: int = cfg.LMAX_CUT,
    nbands: int = cfg.NBANDS,
    mixing_matrix: Optional[np.ndarray] = None,
    data_dir: str = cfg.DATA_DIR,
    nonlinear: str = cfg.NONLINEAR,
    lmax: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Return ``{ "S{i}-S{j}": bandpowers (nbands,) }`` for one cosmology.

    If ``mixing_matrix`` is given, the full (EE, EB, BB) theory vector is mixed by it
    (pseudo-Cl mode coupling for the KiDS mask) before the pixel-window + ℓ-cut +
    bandpower binning.  Otherwise the EE spectrum is used directly (no mixing).
    """
    if lmax is None:
        lmax = 2 * nside

    params = params_from_cosmo_vec(cosmo_vec)
    cosmo, pars = build_cosmology(params, include_baryons=False)

    # Tomographic n(z) on the comoving line-of-sight grid (clean test: no shift).
    zb = glass.shells.distance_grid(cosmo, cfg.ZMIN, cfg.ZMAX, dx=cfg.DX)
    los_z_integration = np.linspace(zb[0], zb[-1], cfg.N_LOS_CHI)
    tomo_nz = calculate_tomo_nz(data_dir, cfg.N_LOS_CHI, los_z_integration, cfg.SHIFT_NZ)

    # CAMB lensing source spectra (recipe matches theory_tests_systematics.py).
    if nonlinear:
        pars.NonLinear = nonlinear
    pars.Want_CMB = False
    pars.min_l = 2
    pars.set_for_lmax(lmax)
    pars.SourceWindows = [
        camb.sources.SplinedSourceWindow(
            z=los_z_integration, W=tomo_nz[i], source_type="lensing"
        )
        for i in range(len(tomo_nz))
    ]
    results = camb.get_results(pars)
    theory_cls = results.get_source_cls_dict(lmax=lmax, raw_cl=True)

    # Polarisation pixel window (EE uses the pol window, squared).
    _, pw = hp.pixwin(nside, lmax=lmax, pol=True)

    cut_theory_cls: Dict[str, np.ndarray] = {}
    for i in range(nbins):
        for j in range(nbins):
            if i < j:
                continue
            Cl_EE = theory_cls[f"W{i + 1}xW{j + 1}"]
            T = np.zeros(3 * (lmax + 1))
            T[0 * (lmax + 1):1 * (lmax + 1)] = Cl_EE
            # EB and BB blocks are zero for pure lensing.

            pseudo = mixing_matrix @ T if mixing_matrix is not None else T
            pseudo_EE = pw ** 2 * pseudo[0 * (lmax + 1):1 * (lmax + 1)]
            cut_theory_cls[f"S{i + 1}-S{j + 1}"] = pseudo_EE[lmin_cut:lmax_cut + 1]

    bandpower_theory: Dict[str, np.ndarray] = {}
    for i in range(nbins):
        for j in range(nbins):
            if i < j:
                continue
            lbl = f"S{i + 1}-S{j + 1}"
            cls_cut = cut_theory_cls[lbl]
            # reshape to (1,1,1,L) so we can reuse the same binning function.
            _, bp_th = compute_cl_bandpowers(
                cls_cut[np.newaxis, np.newaxis, np.newaxis, :],
                1, lmin_cut, lmax_cut, nbands,
            )
            bandpower_theory[lbl] = bp_th[0, :]

    return bandpower_theory
