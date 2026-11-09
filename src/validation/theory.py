"""Per-cosmology theory pseudo-Cl bandpowers for the validation.

theory_mode "shell_projection" (the default) projects the simulator's own shell matter C_l with
the multi-plane lensing weights. "splined" uses continuous CAMB SplinedSourceWindow windows.
Both share the downstream: the KiDS-Legacy MCM, the pol pixel window pw**2, the ell-cut and the
bandpower binning. The nonlinear knob applies to the splined mode only.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import healpy as hp
import camb
import glass
import glass.shells
import glass.lensing
import joblib
from cosmology import Cosmology

from src.cosmology.parameters import build_cosmology
from src.cosmology.camb_matter_power import get_camb_matter_cls
from src.cosmology.manip_cls import compute_cl_bandpowers
from src.KiDS.tomo import calculate_tomo_nz

from . import config as cfg

CosmoVec = Tuple[Sequence[float], Sequence[str]]

# Disk-backed memo for the slow non-Limber matter_cls (one entry per cosmology+grid+lmax).
_MATTER_CACHE_DIR = os.environ.get("GLASS_MATTER_CACHE_DIR", ".glass_matter_cache")
_MEM = joblib.Memory(_MATTER_CACHE_DIR, verbose=0)

# Test seam: set to ``(ws, matter)`` to bypass the slow CAMB matter_cls in unit tests.
_TEST_MATTER_OVERRIDE: Optional[Tuple[list, list]] = None


def _params_key(params: Dict[str, float]) -> Tuple[Tuple[str, float], ...]:
    """Hashable, jitter-stable cache key for a cosmology parameter dict."""
    return tuple(sorted((k, round(float(v), 10)) for k, v in params.items()))


@_MEM.cache
def _cached_matter_cls(params_key, lmax, zmin, zmax, dx):
    """Compute (and disk-cache) the simulator's shell matter C_l for one cosmology.

    Returns ``(ws, matter)`` exactly as ``get_camb_matter_cls`` does (RadialWindow list +
    the flat list of W_i x W_j spectra in CAMB emission order).
    """
    params = {k: v for k, v in params_key}
    _, pars = build_cosmology(params, include_baryons=False)
    ws, matter = get_camb_matter_cls(pars, lmax, zmin, zmax, dx)
    return ws, [np.asarray(m) for m in matter]


def _matter_tensor(matter: List[np.ndarray], nsh: int, lmax: int) -> np.ndarray:
    """Pack the flat matter_cls list into a dense symmetric (nsh, nsh, lmax+1) tensor.

    glass.ext.camb.matter_cls emits [W{i}xW{j} for i in 1..n for j in i..1], with the inner index
    descending. That order is replayed here so each auto-spectrum lands on the diagonal.
    """
    C = np.zeros((nsh, nsh, lmax + 1))
    idx = 0
    for i in range(nsh):
        for j in range(i, -1, -1):
            cl = np.asarray(matter[idx]); idx += 1
            n = min(len(cl), lmax + 1)
            C[i, j, :n] = cl[:n]
            C[j, i, :n] = cl[:n]
    if idx != len(matter):
        raise ValueError(f"matter_cls length {len(matter)} != n(n+1)/2 for nsh={nsh}")
    return C


def _shell_projection_kappa_cls(
    ws, matter, tomo_nz, los_z, cosmo, lmax: int, nbins: int
) -> Dict[Tuple[int, int], np.ndarray]:
    """Project shell matter C_l onto per-bin convergence C_l^{kk} via multi-plane weights.

    The shell weights are ngal_{t,s} = ∫ n_t(z) W_s(z) dz, as in the simulator, and g_{ts} comes
    from glass.lensing.multi_plane_weights. So C_l^{kk}_{tt'} = sum_{jk} g_{tj} g_{t'k} C_l^{d_j d_k}.
    """
    nsh = len(ws)
    C = _matter_tensor(matter, nsh, lmax)
    g = []
    for t in range(nbins):
        ngal = np.array([
            np.trapezoid(*reversed(glass.shells.restrict(los_z, tomo_nz[t], ws[s])))
            for s in range(nsh)
        ])
        g.append(np.asarray(glass.lensing.multi_plane_weights(ngal, ws, cosmo)))
    out: Dict[Tuple[int, int], np.ndarray] = {}
    for i in range(nbins):
        for j in range(i + 1):
            out[(i, j)] = np.einsum("a,b,abl->l", g[i], g[j], C)
    return out


def params_from_cosmo_vec(cosmo_vec: CosmoVec) -> Dict[str, float]:
    """Extract the CAMB cosmology parameters from an ``unpack_data`` cosmo vector.

    cosmo_vec is (values, names) as returned by unpack_data(..., return_names=True). Only the 7
    parameters CAMB needs are pulled, and other saved keys are ignored.
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


def _source_cls_splined(pars, tomo_nz, los_z, lmax, nbins, nonlinear):
    """Theory_A: continuous CAMB SplinedSourceWindow lensing C_l^{kk} per W_i x W_j."""
    if nonlinear:
        pars.NonLinear = nonlinear
    pars.Want_CMB = False
    pars.min_l = 2
    pars.set_for_lmax(lmax)
    pars.SourceWindows = [
        camb.sources.SplinedSourceWindow(z=los_z, W=tomo_nz[i], source_type="lensing")
        for i in range(nbins)
    ]
    theory_cls = camb.get_results(pars).get_source_cls_dict(lmax=lmax, raw_cl=True)
    return {(i, j): theory_cls[f"W{i + 1}xW{j + 1}"]
            for i in range(nbins) for j in range(i + 1)}


def _source_cls_shell_projection(params, cosmo, tomo_nz, los_z, lmax, nbins):
    """Theory_B: discrete multi-plane shell-projection C_l^{kk} (matches the simulator)."""
    if _TEST_MATTER_OVERRIDE is not None:
        ws, matter = _TEST_MATTER_OVERRIDE
    else:
        ws, matter = _cached_matter_cls(_params_key(params), lmax, cfg.ZMIN, cfg.ZMAX, cfg.DX)
    return _shell_projection_kappa_cls(ws, matter, tomo_nz, los_z, cosmo, lmax, nbins)


_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def warm_matter_cache(cosmo_vecs: Sequence[CosmoVec], *, lmax: Optional[int] = None,
                      nside: int = cfg.NSIDE, n_jobs: int = 1,
                      omp_threads: int = 8) -> int:
    """Populate the matter_cls disk cache, once per unique cosmology, and return their count.

    Run it before the parallel ratio loop, so heavy CAMB jobs do not run concurrently with it.
    With n_jobs > 1 it runs n_jobs CAMB calls, each pinned to omp_threads threads through the
    parent env. Keep n_jobs * omp_threads <= cores.
    """
    if lmax is None:
        lmax = 2 * nside
    seen = {}
    for cv in cosmo_vecs:
        seen.setdefault(_params_key(params_from_cosmo_vec(cv)), params_from_cosmo_vec(cv))
    items = list(seen.values())
    print(f"[validation] warming matter_cls cache for {len(items)} unique cosmologies "
          f"(n_jobs={n_jobs}, omp_threads={omp_threads}) ...", flush=True)

    def _one(p):
        return _cached_matter_cls(_params_key(p), lmax, cfg.ZMIN, cfg.ZMAX, cfg.DX)

    if n_jobs and n_jobs > 1 and len(items) > 1:
        import joblib
        saved = {k: os.environ.get(k) for k in _THREAD_ENV_VARS}
        for k in _THREAD_ENV_VARS:
            os.environ[k] = str(omp_threads)
        try:
            joblib.Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
                joblib.delayed(_one)(p) for p in items)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    else:
        for i, p in enumerate(items):
            print(f"[validation]   cosmo {i + 1}/{len(items)}", flush=True)
            _one(p)
    return len(items)


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
    theory_mode: str = cfg.THEORY_MODE,
    lmax: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Return ``{ "S{i}-S{j}": bandpowers (nbands,) }`` for one cosmology.

    theory_mode is "shell_projection" (the default) or "splined". When mixing_matrix is given, the
    full (EE, EB=0, BB=0) vector is mode-coupled by it. The EE block then gets pw**2, is cut to
    [lmin_cut, lmax_cut] and is binned into nbands bandpowers.
    """
    if lmax is None:
        lmax = 2 * nside

    params = params_from_cosmo_vec(cosmo_vec)
    cosmo, pars = build_cosmology(params, include_baryons=False)

    # Tomographic n(z) on the comoving line-of-sight grid (clean test: no shift).
    zb = glass.shells.distance_grid(cosmo, cfg.ZMIN, cfg.ZMAX, dx=cfg.DX)
    los_z_integration = np.linspace(zb[0], zb[-1], cfg.N_LOS_CHI)
    tomo_nz = calculate_tomo_nz(data_dir, cfg.N_LOS_CHI, los_z_integration, cfg.SHIFT_NZ)

    if theory_mode == "shell_projection":
        source_cls = _source_cls_shell_projection(
            params, cosmo, tomo_nz, los_z_integration, lmax, nbins)
    elif theory_mode == "splined":
        source_cls = _source_cls_splined(
            pars, tomo_nz, los_z_integration, lmax, nbins, nonlinear)
    else:
        raise ValueError(f"unknown theory_mode {theory_mode!r}")

    # kk and shear-EE differ by fl=(l+2)(l-1)/[l(l+1)] ~ 1 for l>=50, omitted in both modes.
    _, pw = hp.pixwin(nside, lmax=lmax, pol=True)

    bandpower_theory: Dict[str, np.ndarray] = {}
    for i in range(nbins):
        for j in range(nbins):
            if i < j:
                continue
            Cl_EE = source_cls[(i, j)]
            T = np.zeros(3 * (lmax + 1))
            T[0 * (lmax + 1):1 * (lmax + 1)] = Cl_EE[:lmax + 1]
            # EB and BB blocks are zero for pure lensing.

            pseudo = mixing_matrix @ T if mixing_matrix is not None else T
            pseudo_EE = pw ** 2 * pseudo[0 * (lmax + 1):1 * (lmax + 1)]
            cls_cut = pseudo_EE[lmin_cut:lmax_cut + 1]
            # compute_cl_bandpowers takes shape (1, 1, 1, L).
            _, bp_th = compute_cl_bandpowers(
                cls_cut[np.newaxis, np.newaxis, np.newaxis, :],
                1, lmin_cut, lmax_cut, nbands,
            )
            bandpower_theory[f"S{i + 1}-S{j + 1}"] = bp_th[0, :]

    return bandpower_theory
