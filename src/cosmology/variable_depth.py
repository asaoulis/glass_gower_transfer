"""Variable-depth (VD) look-up classes for galaxy-catalogue sampling.

These three classes are copied **VERBATIM** from the reference implementation
``kids-legacy-sbi/kids-legacy-sbi/kids_legacy_sim/catalogue.py`` (lines 15-176) so the VD effect
is identical to the validated reference. They are pure look-up / sampling helpers (no edits to the
forward model) and depend only on numpy + healpy. Construction of instances lives in
``src/cosmology/sim_utils.py:build_variable_depth``; the constants live in
``src/KiDS/variable_depth_config.py``.
"""
import numpy as np
import healpy as hp

from collections.abc import Callable
from numpy.typing import NDArray


class AngularVariableDepthMask:
    """Per-tomographic-bin variable depth mask varying in the angular direction only.

    Indexing is ``mask[tomo, shell]``; the shell index is accepted for API
    parity with ``AngularLosVariableDepthMask`` but ignored here.
    """

    def __init__(
        self,
        vardepth_map: NDArray[np.float64],
        n_bins: int,
        zbins: list[tuple[float, float]],
    ) -> None:
        self.vardepth_map = vardepth_map
        self.n_bins = n_bins
        self.zbins = zbins

    def check_index(self, index: tuple[int, int]) -> None:
        if not isinstance(index, tuple):
            raise TypeError("Index must be an tuple of two integers")
        if index[0] >= self.n_bins:
            raise ValueError("Leading index cannot exceed number of tomographic bins")
        if index[1] >= len(self.zbins):
            raise ValueError("Trailing index cannot exceed number of shells")

    def __getitem__(self, index: tuple[int, int]) -> NDArray[np.float64]:
        self.check_index(index)
        return self.vardepth_map[index[0]]  # type: ignore[no-any-return]

class AngularLosVariableDepthMask(AngularVariableDepthMask):
    """Per-tomographic-bin variable depth mask varying in both the angular and
    line-of-sight directions.

    ``vardepth_tomo_functions`` should map ``vardepth_map`` values to the
    galaxy-count contrast (VD / no-VD); when omitted, ``vardepth_map`` is
    itself treated as the contrast.  ``dndz_vardepth`` provides per-VD-bin
    n(z)s used to derive the LOS fraction per shell.
    """

    def __init__(  # noqa: PLR0913
        self,
        vardepth_map: NDArray[np.float64],
        n_bins: int,
        zbins: list[tuple[float, float]],
        ztomo: list[tuple[float, float]],
        dndz: NDArray[np.float64],
        z: NDArray[np.float64],
        dndz_vardepth: NDArray[np.float64],
        vardepth_values: NDArray[np.float64],
        vardepth_los_tracer: NDArray[np.float64] | None = None,
        vardepth_tomo_functions: list[
            Callable[[NDArray[np.float64]], NDArray[np.float64]]
        ]
        | None = None,
    ) -> None:
        super().__init__(vardepth_map, n_bins, zbins)
        self.ztomo = ztomo
        self.dndz = dndz
        self.z = z
        self.dndz_vardepth = dndz_vardepth
        self.vardepth_values = vardepth_values
        self.vardepth_los_tracer = vardepth_los_tracer
        self.vardepth_tomo_functions = vardepth_tomo_functions

    def get_los_fraction(self, index: tuple[int, int]) -> NDArray[np.float64]:
        """Per-VD-bin ratio of in-shell galaxy counts (with VD / without VD).

        The integration grid includes the exact shell boundaries ``zb[i]`` and
        ``zb[i+1]`` with ``dndz`` linearly interpolated at those points, so the
        result varies continuously as the shell edges shift (e.g. with
        cosmology). A boolean slice on ``self.z`` would flip individual grid
        points in or out of the shell and produce step-function jumps that
        destabilise finite-difference derivatives.
        """
        z0, z1 = self.zbins[index[1]]
        inside = (self.z > z0) & (self.z < z1)
        z_int = np.concatenate([[z0], self.z[inside], [z1]])

        dndz_on_grid = np.interp(z_int, self.z, self.dndz[index[0]])
        n_gal_in_tomo = np.trapezoid(dndz_on_grid, z_int)

        dndz_vd_on_grid = np.stack(
            [
                np.interp(z_int, self.z, self.dndz_vardepth[index[0]][k])
                for k in range(self.dndz_vardepth.shape[1])
            ]
        )
        n_gal_in_tomo_vardepth = np.trapezoid(dndz_vd_on_grid, z_int, axis=-1)

        return np.divide(  # type: ignore[no-any-return]
            n_gal_in_tomo_vardepth,
            n_gal_in_tomo,
            out=np.ones_like(n_gal_in_tomo_vardepth),
            where=n_gal_in_tomo != 0,
        )

    def __getitem__(self, index: tuple[int, int]) -> NDArray[np.float64]:
        self.check_index(index)

        if self.vardepth_tomo_functions is None:
            angular_vardepth_map = angular_tracer_map = self.vardepth_map[index[0]]
        else:
            angular_vardepth_map = self.vardepth_tomo_functions[index[0]](
                self.vardepth_map[index[0]]
            )
            angular_tracer_map = self.vardepth_map[index[0]]

        los_fraction_vardepth = self.get_los_fraction(index)

        tracer = (angular_tracer_map if self.vardepth_los_tracer is None
                  else self.vardepth_los_tracer)
        los_vardepth_map = np.interp(
            tracer, self.vardepth_values[index[0]], los_fraction_vardepth
        )

        return np.multiply(angular_vardepth_map, los_vardepth_map)  # type: ignore[no-any-return]


class VariableDepthShapeDispersion:
    """Per-pixel intrinsic shape dispersion for variable-depth ellipticity sampling.

    Evaluates ``sigma_eps_var[i](vd_map[i])`` once at construction so each
    galaxy can read its sigma from the precomputed map.  Sampling uses the
    same intrinsic-normal transform as ``glass.shapes.ellipticity_intnorm``
    but is fully vectorised over the per-galaxy sigma array, avoiding the
    Python loop in that function's ``vdmode=True`` path.  ``sigma_eps_var``
    values must lie in ``[0, sqrt(0.5))``.
    """

    def __init__(
        self,
        sigma_eps_var: list[Callable[[NDArray[np.float64]], NDArray[np.float64]]],
        vd_map: NDArray[np.float64],
        nside: int,
    ) -> None:
        self.nside = nside
        self.sigma_maps: NDArray[np.float64] = np.array(
            [sigma_eps_var[i](vd_map[i]) for i in range(len(sigma_eps_var))]
        )

    def sample(
        self,
        tomo: int,
        gal_lon: NDArray[np.float64],
        gal_lat: NDArray[np.float64],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.complex128]:
        """Draw complex ellipticities with per-galaxy sigma read from the precomputed map."""
        if rng is None:
            rng = np.random.default_rng()

        gal_pix = hp.ang2pix(self.nside, gal_lon, gal_lat, lonlat=True)
        sigma = self.sigma_maps[tomo][gal_pix]

        # Vectorised intrinsic-normal transform (cf. glass.shapes.ellipticity_intnorm).
        sigma_eta = sigma * ((8 + 5 * sigma**2) / (2 - 4 * sigma**2)) ** 0.5
        e = rng.standard_normal(2 * len(sigma)).view(np.complex128)
        e *= sigma_eta
        r = np.hypot(e.real, e.imag)
        r_safe = np.where(r > 0, r, 1.0)
        e *= np.where(r > 0, np.tanh(r / 2) / r_safe, 1.0)
        return e
