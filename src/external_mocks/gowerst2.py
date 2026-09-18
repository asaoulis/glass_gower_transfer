"""External shell-cube ingestion: GowerSt2 / ``Flamingo_big`` (DMO and baryonified).

The whole point of this module is that it is SMALL. ``GowerStreetSimulator`` already implements
exactly the pattern we need -- ``_load_shells()`` builds the ``z <= zmax`` radial-window list and
``get_matter_fields()`` lazily yields ONE shell at a time off disk -- so ingesting an external
lightcone is **two method overrides**. Everything downstream (IA, n(z), mask, Poisson sampling
with source clustering, shape noise, shear bias, pseudo-Cl, patches) is untouched production code.

Nothing here edits protected physics: it subclasses `src.cosmology.simulators` and calls
`src.cosmology.parameters` / `sim_utils`.

Data layout (one directory, both variants, shared ``control.par``)::

    control.par                     PKDGRAV3 parameters   (nSteps=100, achOutName="run", bClass=1)
    run.log                         101 step redshifts, ascending 0 -> 49
    class_processed_Flamingo.hdf5   CLASS cosmology (referenced RELATIVELY by control.par)
    delta_1024.npy    (100, 12582912) f64   gravity-only  delta
    delta_b_1024.npy  (100, 12582912) f32   FLAMINGO-baryonified delta (same phases)

Audited facts that this module depends on (see
``.claude/runs/eval-and-viz/baryonified-unblind-mock/artifacts/SHELL_AUDIT.md``):

* ``sim.redshifts`` is ASCENDING from z=0, and cube row 0 is the NEAREST shell. Feeding
  ``MultiPlaneConvergence.add_window`` in the wrong order does NOT raise -- it silently computes
  the wrong lensing kernel -- so :func:`load_external_shells` asserts the direction.
* The cubes are already ``delta`` (mean ~ 1e-8). The parent's trailing
  ``arr / np.mean(arr) - 1`` therefore MUST NOT be inherited (it would divide by ~0).
* ``delta < -1`` occurs at low z (49% of pixels in the z~0 shell, 4.5e-3 by z=0.54, none above
  z=1.5): the cubes are counts PLUS a continuous additive component, not a pure count field.
  This is harmless for the convergence (linear in delta) and ``glass.positions_from_delta``
  clips the expected counts at zero before the Poisson draw, so it degrades gracefully. We warn
  once rather than raise.
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

#: The one external box we currently ingest. ``cosmology`` is in the model keys that
#: ``build_cosmology`` / ``GowerStCosmologies.PARAM_MAP`` use. See artifacts/COSMOLOGY.md.
#:
#: sigma_8 = 0.809177. This value was challenged, changed to control.par's 0.7622848, and then
#: CHANGED BACK when the end-to-end test rejected it. The history matters, so:
#:
#:   * control.par's `dNormalization` comment says "calculated from sigma_8 = 0.7622848".
#:   * The delta cube's own angular spectra (anafast on shells 62/70 vs Limber, read at
#:     ell = 400-600 where Limber is valid for a thin shell) imply ~0.768, near that comment.
#:   * BUT the full-pipeline bandpower validation -- the mocks this code actually produces,
#:     compared to CAMB/HMcode through the KiDS mixing matrix -- is decisive the other way:
#:         sigma_8 = 0.809177 -> mean |dev| 3%,  per-bin signed dev +0.015..-0.003 (UNBIASED)
#:         sigma_8 = 0.762    -> mean |dev| 17%, per-bin signed dev +0.144..+0.185 (theory low)
#:     implying a best fit near 0.81, consistent with the CLASS file (0.8082) and published
#:     FLAMINGO (0.807).
#:
#: So the shell-level anafast test carries an UNEXPLAINED SYSTEMATIC of order 10% in power and
#: must not be used to set the normalisation; the end-to-end comparison is the one that counts.
#: Do not "fix" this back to 0.762 without re-running src/validation/run_validation.py and
#: showing it improves. The residual ell-tilt at 0.809 (band 0 low by ~11%, high-ell high by
#: ~4%, the latter concentrated in tomographic bin 1) is a SEPARATE unresolved issue -- see
#: artifacts/SIGMA8_INVESTIGATION.md.
#:
#: The rest of the cosmology is verified INDEPENDENTLY and directly: z_values.txt tabulates the
#: chi(z) the run actually used, and it matches these Omega_m / h / w0 to 0.06%. So the
#: background -- the only part that enters the lensing kernel -- is right.
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
    """Return ``(rows, shells)`` for the external lightcone in ``data_dir``.

    ``rows`` are **cube row indices** (not PKDGRAV3 step numbers) and ``shells`` the matching
    ``glass.shells.RadialWindow`` list, so the two stay aligned through the ``z <= zmax`` cut.

    The window construction deliberately mirrors ``GowerStreetSimulator._load_shells`` exactly
    (uniform tophat, 100 samples, ``zeff`` = the midpoint): using the SAME window convention as
    the Gower-Street-1 training cloud is what keeps a window-definition difference from leaking
    into the misspecification we are trying to measure.
    """
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
    """Seam contract for an external box, mirroring ``prepare_gower_backend``.

    ``matter`` is **None**: the shells are streamed by :class:`ShellCubeSimulator`, which the
    master script constructs directly for this simulator type. ``shells`` is still returned
    because the systematics builders need the shell edges.

    The IA draw is taken FIRST, from ``rng``, exactly as ``prepare_gower_backend`` does -- that
    ordering is part of the ``--rng-seed`` contract that pairs the DMO and BARY catalogues.
    """
    nuisance_params = sample_ia_params(prior_ranges, rng)
    param_dict = {**spec["cosmology"], **nuisance_params}

    # DEFAULT_ZMAX (not sim_grid) so these shells are IDENTICAL to the ones ShellCubeSimulator
    # builds; a divergence here would silently mismatch the systematics builders against the
    # windows actually used for the convergence.
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
