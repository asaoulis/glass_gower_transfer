"""Euclid DR3 tomographic n(z).

Deliberately separate from `src/KiDS/tomo.py` (which is PROTECTED physics and KiDS-specific):
Euclid reads a single ascii table rather than the KiDS per-bin nofz files, and applies no
photo-z shifts.
"""

import numpy as np


def build_tomo_nz(nz_path, los_z_integration, n_arcmin2_total):
    """Interpolate the Euclid DR3 n(z) table onto the line-of-sight integration grid.

    The file is `z` followed by one column per tomographic bin. Each bin's share of
    `n_arcmin2_total` is its share of the raw (unnormalised) integral of its column, so the
    returned curves integrate to `n_arcmin2_total` in total, in gal/arcmin^2.

    Returns
    -------
    tomo_nz : (nbins, len(los_z_integration)) array
    n_arcmin2 : (nbins,) array -- the per-bin number densities.
    """
    data = np.loadtxt(nz_path)
    z, nz = data[:, 0], data[:, 1:]
    nbins = nz.shape[1]

    raw_integrals = np.trapezoid(nz, z, axis=0)
    n_arcmin2 = n_arcmin2_total * raw_integrals / raw_integrals.sum()

    tomo_nz = np.zeros((nbins, len(los_z_integration)))
    for i in range(nbins):
        interpolated = np.interp(
            los_z_integration, z, n_arcmin2[i] * nz[:, i] / raw_integrals[i]
        )
        tomo_nz[i] = np.clip(interpolated, 0, None)
    return tomo_nz, n_arcmin2
