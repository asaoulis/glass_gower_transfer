import numpy as np
import healpy as hp
from numba import njit

@njit(nogil=True)
def _map_shears_weights(she_map, wht_map, gal_pix, gal_she, gal_wht):
    for i, s, w in zip(gal_pix, gal_she, gal_wht):
        she_map[i] += s
        wht_map[i] += w

@njit(nogil=True)
def _map_shears(she_map, wht_map, gal_pix, gal_she):
    for i, s in zip(gal_pix, gal_she):
        she_map[i] += s
        wht_map[i] += 1

def map_shears(she_map, wht_map, gal_lon, gal_lat, gal_she, gal_wht=None):
    nside = hp.get_nside(she_map)
    gal_pix = hp.ang2pix(nside, gal_lon, gal_lat, lonlat=True)
    if gal_wht is None:
        _map_shears(she_map, wht_map, gal_pix, gal_she)
    else:
        _map_shears_weights(she_map, wht_map, gal_pix, gal_she, gal_wht)

def _cosine_taper(ell, l0, l1):
    """
    Cosine taper: 1 for ell<l0, 0 for ell>l1, cosine roll-off on [l0, l1].
    """
    w = np.ones_like(ell, dtype=float)
    if l1 <= l0:
        # degenerate: step at l0
        w[ell > l0] = 0.0
        return w
    m = (ell >= l0) & (ell <= l1)
    w[ell > l1] = 0.0
    w[m] = 0.5 * (1.0 + np.cos(np.pi * (ell[m] - l0) / float(l1 - l0)))
    return w

def _build_ell_filter(lmax_in, lmax_out, fwhm_arcmin=8.0, taper_start_frac=0.95):
    """
    Build an anti-aliasing filter for alms:
    - optional Gaussian beam (fwhm_arcmin)
    - cosine taper starting at taper_start_frac*lmax_out to lmax_out, zero above.
    """
    ell = np.arange(lmax_in + 1)
    if fwhm_arcmin and fwhm_arcmin > 0:
        beam = hp.gauss_beam(fwhm=np.deg2rad(fwhm_arcmin / 60.0), lmax=lmax_in)
    else:
        beam = np.ones_like(ell, dtype=float)

    taper = np.ones_like(ell, dtype=float)
    if lmax_out < lmax_in:
        l0 = int(max(0, np.floor(taper_start_frac * lmax_out)))
        l1 = int(lmax_out)
        taper = _cosine_taper(ell, l0, l1)

    return beam * taper

def filter_EB_alms_and_make_maps(alm_list, nside_out=512, lmax_out=None, fwhm_arcmin=8.0, taper_start_frac=0.95):
    """
    Apply anti-alias filtering to (E,B) alms per tomographic bin and synthesize
    E and B scalar maps directly at nside_out.

    Parameters
    ----------
    alm_list : list of (almE, almB)
        One (E,B) tuple per tomographic bin as returned by make_alm_shear_convergence.
    nside_out : int
        Target HEALPix resolution for output maps.
    lmax_out : int or None
        Target bandlimit. If None, use min(lmax_in, 3*nside_out-1, 1500).
    fwhm_arcmin : float
        Gaussian smoothing FWHM in arcmin (mitigate near-Nyquist modes).
    taper_start_frac : float
        Fraction of lmax_out at which to start cosine taper.

    Returns
    -------
    E_maps_out, B_maps_out : np.ndarray
        Arrays of shape (nbins, hp.nside2npix(nside_out)) with filtered maps.
    """
    if len(alm_list) == 0:
        return np.array([]), np.array([])

    almE0, _ = alm_list[0]
    lmax_in = hp.Alm.getlmax(almE0.size)
    if lmax_out is None:
        lmax_out = min(lmax_in, 3 * nside_out - 1, 1500)

    fl = _build_ell_filter(lmax_in=lmax_in, lmax_out=lmax_out,
                           fwhm_arcmin=fwhm_arcmin, taper_start_frac=taper_start_frac)

    nbins = len(alm_list)
    npix_out = hp.nside2npix(nside_out)
    E_maps_out = np.zeros((nbins, npix_out), dtype=float)
    B_maps_out = np.zeros((nbins, npix_out), dtype=float)

    for i, (almE, almB) in enumerate(alm_list):
        almE_f = hp.almxfl(almE, fl)
        almB_f = hp.almxfl(almB, fl)
        # Synthesize directly at target resolution and bandlimit
        E_maps_out[i] = hp.alm2map(almE_f, nside=nside_out)
        B_maps_out[i] = hp.alm2map(almB_f, nside=nside_out)

    return E_maps_out, B_maps_out

def make_alm_shear_convergence(
    catalogue,
    m_bias,
    nbins,
    nside,
    lmax,
    mask=None,
    nosh=False,
    return_shear=False,
    normalization="mean",
    n_arcmin2=None,
    rng=None,
):
    """Compute spin-2 shear alms (E/B) per tomographic bin.

    This function maps galaxy ellipticities to HEALPix shear maps and then
    computes spin-2 spherical harmonic coefficients.

    Normalization modes
    -------------------
    - "counts": divide by the *observed* number of galaxies per pixel (per-bin).
    - "mean": divide by the mean counts per pixel (previous behaviour here).
    - "expected": divide by n_arcmin2[i] * pixel_area_arcmin2 (legacy `make_alm`).

    Notes
    -----
    - `mask` is currently not applied to the maps; it is only used for the
      effective pixel count when `normalization="mean"`.
    - If you pass `rng` (np.random.Generator), it is used for the random
      rotations; otherwise, the global `np.random` is used.
    """

    if return_shear:
        all_shear = np.zeros((nbins, hp.nside2npix(nside)), dtype=complex)

    if mask is None:
        npix_norm = hp.nside2npix(nside)
    else:
        npix_norm = np.sum(mask)

    alm, alm_rand = [], []
    ell, emm = hp.Alm.getlm(lmax=lmax)
    alm_size = hp.Alm.getsize(lmax)

    def _apply_normalization(field_map, count_map, bin_idx):
        if normalization == "counts":
            valid = count_map > 0
            field_map[valid] = field_map[valid] / count_map[valid]
            return

        if normalization == "mean":
            denom = float(np.sum(count_map)) / float(npix_norm) if npix_norm else 0.0
            if denom > 0:
                valid = count_map > 0
                field_map[valid] = field_map[valid] / denom
            else:
                field_map[:] = 0.0
            return

        if normalization == "expected":
            if n_arcmin2 is None:
                raise ValueError("n_arcmin2 must be provided when normalization='expected'")
            denom = float(n_arcmin2[bin_idx]) * hp.nside2pixarea(nside, degrees=True) * 3600.0
            if denom <= 0:
                raise ValueError(f"Expected normalization <= 0 for bin {bin_idx}: {denom}")
            valid = count_map > 0
            field_map[valid] = field_map[valid] / denom
            return

        raise ValueError(f"Unknown normalization mode: {normalization!r}")

    for i in range(nbins):
        shear = np.zeros(hp.nside2npix(nside), dtype=complex)
        counts = np.zeros(hp.nside2npix(nside), dtype=int)
        in_bin = catalogue["ZBIN"] == i

        if not np.any(in_bin):
            alm.append((np.zeros(alm_size, dtype=complex), np.zeros(alm_size, dtype=complex)))
            alm_rand.append((np.zeros(alm_size, dtype=complex), np.zeros(alm_size, dtype=complex)))
            if return_shear:
                all_shear[i] = shear
            continue

        e1 = catalogue["E1"][in_bin]
        e2 = catalogue["E2"][in_bin]
        she = (1.0 / (1.0 + m_bias[i])) * ((e1 - np.mean(e1)) + 1j * (e2 - np.mean(e2)))

        map_shears(
            shear,
            counts,
            catalogue["RA"][in_bin],
            catalogue["DEC"][in_bin],
            she,
            gal_wht=None,
        )

        _apply_normalization(shear, counts, i)

        gal_num = int(np.sum(in_bin))
        if rng is None:
            rand_theta = 2.0 * np.pi * np.random.random_sample(gal_num)
        else:
            rand_theta = 2.0 * np.pi * rng.random(gal_num)

        e1_corr = she.real * np.cos(rand_theta) - she.imag * np.sin(rand_theta)
        e2_corr = she.imag * np.cos(rand_theta) + she.real * np.sin(rand_theta)

        rand = np.zeros(hp.nside2npix(nside), dtype=complex)
        rand_counts = np.zeros_like(rand, dtype=int)
        map_shears(
            rand,
            rand_counts,
            catalogue["RA"][in_bin],
            catalogue["DEC"][in_bin],
            e1_corr + 1j * e2_corr,
            gal_wht=None,
        )

        _apply_normalization(rand, rand_counts, i)

        almE, almB = hp.sphtfunc.map2alm_spin([shear.real, shear.imag], spin=2, lmax=lmax)
        almE_rand, almB_rand = hp.sphtfunc.map2alm_spin([rand.real, rand.imag], spin=2, lmax=lmax)

        if nosh:
            factor = np.sqrt((ell * (ell + 1.0)) / ((ell + 2.0) * (ell - 1.0)))
            almE *= factor
            almB *= factor
            almE_rand *= factor
            almB_rand *= factor

        if return_shear:
            all_shear[i] = shear

        alm.append((almE, almB))
        alm_rand.append((almE_rand, almB_rand))

    if return_shear:
        return alm, alm_rand, all_shear
    return alm, alm_rand

import numpy as np
import healpy as hp

def pixel_area_arcmin2(nside):
    # healpy.nside2pixarea returns steradians
    pix_area_sr = hp.nside2pixarea(nside)          # steradians
    return pix_area_sr * (180.0/np.pi * 60.0)**2   # arcmin^2