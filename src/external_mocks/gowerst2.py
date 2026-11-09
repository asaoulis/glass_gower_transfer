"""External shell-cube ingestion: GowerSt2 / Flamingo_big (DMO and baryonified).

ShellCubeSimulator subclasses GowerStreetSimulator and reads one stacked delta cube plus a
PKDGRAV3 control.par. prepare_external_backend returns the backend dict for the master script's
`external` simulator type.
"""

from __future__ import annotations

import os
import warnings

import glass
import glass.ext.pkdgrav
import healpy as hp
import numpy as np

from src.cosmology.parameters import build_cosmology
from src.cosmology.sim_utils import sample_ia_params
from src.cosmology.simulators import GowerStreetSimulator

# Our production shell stack stops at z=2 (src/KiDS/simulation_config.py).
DEFAULT_ZMAX = 2.0

# Flamingo_big cosmology in the model keys of build_cosmology / GowerStCosmologies.PARAM_MAP
# (sigma_8 and Omega_b derivations in the trailing notes).
FLAMINGO_BIG = {
    "name": "gowerst2_flamingo",
    "data_dir": "/data/alex/external_mocks/gowerst2_flamingo",
    "cubes": {
        "dmo": "delta_1024.npy",
        "bary": "delta_b_1024.npy",
    },
    "cosmology": {
        "omega_m": 0.305997,   # Omega_b + Omega_cdm + Omega_nu (CLASS class_params)
        "sigma_8": 0.809177,   # see the note above: END-TO-END bandpower validation selects this
        "h": 0.681,            # H0 = 68.1
        "ombh2": 0.022539,     # 0.0486 * 0.681**2
        "ns": 0.967,           # control.par dSpectral
        "w0": -1.0,            # w0_fld
        "mnu": 0.06,           # N_ncdm=1, deg_ncdm=3, m_ncdm=0.02 eV
    },
    "provenance": {
        "box": "1250 Mpc/h, nGrid=1350, iSeed=2000, nSideHealpix=4096 (cubes degraded to 1024)",
        "source": "/global/cfs/cdirs/m5099/GowerSt2/Flamingo_big (NERSC, M. Gatti)",
        "baryonification": "FLAMINGO; applied only below z = 1.0122 (cube rows 0..57); rows "
                           "58..99 are the DMO field. Matches baryons['max_z_halo_catalog'] = 1.",
        "As": 2.099e-09,
        "geometry_check": "z_values.txt chi(z) matches Omega_m=0.306/h=0.681/w0=-1 to 0.06%",
        "sigma_8_check": "anafast shells 62,70 vs Limber at ell 400-600 -> ratio 0.90 -> s8 ~ 0.768",
    },
}


def load_external_shells(data_dir, zmax=DEFAULT_ZMAX):
    """Return (rows, shells): cube row indices and the matching glass RadialWindow list for the
    lightcone in data_dir, ordered by increasing comoving distance."""
    sim = glass.ext.pkdgrav.load(os.path.join(data_dir, "control.par"))
    z = np.asarray(sim.redshifts, dtype=float)

    n_shells = len(z) - 1
    if n_shells != int(sim.parameters["nSteps"]):
        raise ValueError(f"{data_dir}: {n_shells} shells from run.log but nSteps="
                         f"{sim.parameters['nSteps']} in control.par")
    # Ordering is load-bearing: add_window needs increasing comoving distance and will not
    # complain if handed the lightcone backwards.
    if not (abs(z[0]) < 1e-6 and np.all(np.diff(z) > 0)):
        raise ValueError(f"{data_dir}: expected redshifts ascending from 0, got "
                         f"z[0]={z[0]:.6g}, monotonic={bool(np.all(np.diff(z) > 0))}")

    rows, shells = [], []
    for k in range(n_shells):
        zmin, zmax_k = float(z[k]), float(z[k + 1])
        if zmin > zmax:              # parent's rule verbatim (keeps the straddling shell)
            continue
        za = np.linspace(zmin, zmax_k, 100)
        wa = np.ones_like(za)
        shells.append(glass.shells.RadialWindow(za, wa, 0.5 * (zmin + zmax_k)))
        rows.append(k)
    return rows, shells


class ShellCubeSimulator(GowerStreetSimulator):
    """``GowerStreetSimulator`` reading one stacked ``.npy`` delta cube + a PKDGRAV3 ``control.par``.

    Two overrides only; ``__init__`` just records which cube to stream.
    """

    def __init__(self, data_dir, cube_name, *args, zmax=DEFAULT_ZMAX, **kwargs):
        self.cube_name = cube_name
        self.zmax = zmax
        self._checked = False
        super().__init__(data_dir, *args, **kwargs)   # -> self._load_shells(data_dir)

    # -- override 1: where the shells come from -------------------------------------------
    def _load_shells(self, data_dir):
        rows, shells = load_external_shells(data_dir, zmax=self.zmax)
        if self.debug:
            print(f"[external] {len(rows)} shells with z <= {self.zmax} from {data_dir} "
                  f"(rows {rows[0]}..{rows[-1]}); cube={self.cube_name}", flush=True)
        return rows, shells

    # -- override 2: how one shell is read -------------------------------------------------
    def get_matter_fields(self):
        path = os.path.join(self.data_dir, self.cube_name)
        cube = np.load(path, mmap_mode="r")            # lazy: the DMO cube is 10 GB of f64
        if cube.ndim != 2:
            raise ValueError(f"{path}: expected a (n_shells, npix) cube, got shape {cube.shape}")

        for i, row in enumerate(self.steps):
            if row >= cube.shape[0]:
                raise ValueError(f"{path}: needs row {row} but cube has {cube.shape[0]} rows")
            delta = np.asarray(cube[row], dtype=np.float64)

            if hp.get_nside(delta) != self.nside:
                # delta is INTENSIVE -> plain averaging (power=0). The parent uses power=-2
                # because it degrades particle COUNTS; using that here would rescale delta.
                delta = hp.ud_grade(delta, self.nside)

            if not self._checked:
                self._check_is_delta(delta, row)
                self._checked = True

            if self.debug:
                print(f"[external] shell {i} (cube row {row}) zeff={self.ws[i].zeff:.3f} "
                      f"std={delta.std():.4f}", flush=True)
            # NOTE: no `arr / np.mean(arr) - 1` here -- the cube is already delta.
            yield i, delta

    # -- guard ------------------------------------------------------------------------------
    @staticmethod
    def _check_is_delta(delta, row):
        """Fail loudly on a raw-mass cube; warn (once) on the known delta < -1 pixels."""
        mean = float(delta.mean())
        if not np.isfinite(delta).all():
            raise ValueError(f"external shell row {row}: non-finite pixels")
        if abs(mean) > 1e-3:
            raise ValueError(
                f"external shell row {row}: mean = {mean:.6g}, expected ~0. This cube looks like "
                "RAW MASS, not a density contrast -- it needs `arr/mean(arr) - 1` applied first. "
                "Refusing to run rather than silently produce a wrong matter field.")
        lo = float(delta.min())
        if lo < -1.0:
            warnings.warn(
                f"external shell row {row}: min(delta) = {lo:.3f} < -1 "
                f"({np.mean(delta < -1):.3%} of pixels). Expected for this box (the cubes are "
                "counts plus a continuous additive component, not a pure count field). The "
                "convergence is linear in delta so it is unaffected; glass.positions_from_delta "
                "clips the expected counts at zero before the Poisson draw. See SHELL_AUDIT.md.",
                RuntimeWarning, stacklevel=2)


def prepare_external_backend(sim_num, *, rng, spec, prior_ranges, sim_grid, variant):
    """Backend dict for an external box, as prepare_gower_backend. matter is None (shells are
    streamed by ShellCubeSimulator)."""
    nuisance_params = sample_ia_params(prior_ranges, rng)
    param_dict = {**spec["cosmology"], **nuisance_params}

    # DEFAULT_ZMAX (not sim_grid) so these shells are identical to the ones ShellCubeSimulator
    # builds.
    _, shells = load_external_shells(spec["data_dir"], zmax=DEFAULT_ZMAX)
    cosmo, _pars = build_cosmology(param_dict)

    return {
        "param_dict": param_dict,
        "shells": shells,
        "matter": None,
        "cosmo": cosmo,
        "data_dir": spec["data_dir"],
        "cube_name": spec["cubes"][variant],
    }
