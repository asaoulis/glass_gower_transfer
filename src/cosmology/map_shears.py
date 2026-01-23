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

def make_alm_shear_convergence(catalogue, m_bias, nbins, nside, lmax, mask = None, nosh=False, return_shear = False):
    if return_shear:
        all_shear = np.zeros((nbins, hp.nside2npix(nside)), dtype=complex)
    if mask is None:
        npix = hp.nside2npix(nside)
    else:
        npix = np.sum(mask)
    alm, alm_rand = [], []

    ell, emm = hp.Alm.getlm(lmax=lmax)

    for i in range(nbins):
        shear = np.zeros(hp.nside2npix(nside), dtype=complex)
        counts = np.zeros(hp.nside2npix(nside), dtype=int)
        in_bin = (catalogue['ZBIN'] == i)

        she = (1/(1+m_bias[i])) * (
            (catalogue['E1'][in_bin] - np.mean(catalogue['E1'][in_bin]))
            + 1j*(catalogue['E2'][in_bin] - np.mean(catalogue['E2'][in_bin]))
        )

        map_shears(shear, counts,
                   catalogue['RA'][in_bin],
                   catalogue['DEC'][in_bin],
                   she, gal_wht=None)

        # shear[i][counts[i] > 0] = shear[i][counts[i] > 0] / counts[i][counts[i] > 0]
        shear[counts > 0] = shear[counts > 0] / (sum(counts) / npix)

        # Make randomized shear field
        gal_num = len(catalogue[in_bin])
        rand_theta = 2*np.pi*np.random.random_sample(gal_num)
        e1_corr = she.real*np.cos(rand_theta) - she.imag*np.sin(rand_theta)
        e2_corr = she.imag*np.cos(rand_theta) + she.real*np.sin(rand_theta)

        rand = np.zeros(hp.nside2npix(nside), dtype=complex)
        _ = np.zeros_like(rand, dtype=int)

        map_shears(rand, _, catalogue['RA'][in_bin], catalogue['DEC'][in_bin],
                   e1_corr + 1j*e2_corr, gal_wht=None)
        # rand[_ > 0] = rand[_ > 0] / _[_ > 0]
        rand[_ > 0] = rand[_ > 0] / (sum(_) / npix)

        # Compute spin-2 alm decomposition
        almE, almB = hp.sphtfunc.map2alm_spin([shear.real, shear.imag],
                                              spin=2, lmax=lmax)
        almE_rand, almB_rand = hp.sphtfunc.map2alm_spin([rand.real, rand.imag],
                                                        spin=2, lmax=lmax)

        if nosh:
            factor = np.sqrt((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1.)))
            almE *= factor
            almB *= factor
            almE_rand *= factor
            almB_rand *= factor

        if return_shear:
            all_shear[i] = shear
        # Save
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

def make_alm_shear_convergence_fixed(catalogue, m_bias, nbins, nside, lmax,
                                     n_arcmin2, nosh=False):
    """
    Modified: normalise maps by EXPECTED number of galaxies PER PIXEL (nbar_pix)
    rather than dividing by the observed counts in each pixel.
    - n_arcmin2: 1D array (length nbins) with galaxy surface density [gal/arcmin^2]
    """
    npix = hp.nside2npix(nside)
    pix_area_arcmin2 = pixel_area_arcmin2(nside)      # scalar (arcmin^2 per pixel)
    nbar_pix_per_bin = n_arcmin2 * pix_area_arcmin2   # shape (nbins,)

    # prepare arrays
    shear_maps = np.zeros((nbins, npix), dtype=complex)   # will hold SUM(g) initially
    counts = np.zeros((nbins, npix), dtype=float)        # will hold counts (float is convenient)

    alm, alm_rand = [], []

    ell, emm = hp.Alm.getlm(lmax=lmax)

    for i in range(nbins):
        in_bin = (catalogue['ZBIN'] == i)
        if not np.any(in_bin):
            # no galaxies in this bin; append empty alms
            alm.append((np.zeros(hp.Alm.getsize(lmax), dtype=complex),
                        np.zeros(hp.Alm.getsize(lmax), dtype=complex)))
            alm_rand.append((np.zeros_like(alm[-1][0]), np.zeros_like(alm[-1][1])))
            continue

        # per-galaxy shears (your bias correction & mean-removal)
        gal_e1 = catalogue['E1'][in_bin]
        gal_e2 = catalogue['E2'][in_bin]
        she = (1.0/(1.0 + m_bias[i])) * ((gal_e1 - np.mean(gal_e1)) + 1j*(gal_e2 - np.mean(gal_e2)))

        # accumulate sums and counts (map_shears adds sums to shear_maps[i], increments counts[i])
        map_shears(shear_maps[i], counts[i],
                   catalogue['RA'][in_bin],
                   catalogue['DEC'][in_bin],
                   she, gal_wht=None)

        # --- NORMALIZE BY EXPECTED PER-PIXEL NUMBER (not by observed counts) ---
        nbar_pix = nbar_pix_per_bin[i]
        if not (nbar_pix > 0):
            raise ValueError("Computed nbar_pix <= 0 for bin %d" % i)

        # convert accumulated sums -> average field scaled to expected mean density
        # -> shear_field = sum(g) / nbar_pix
        shear_maps[i] /= nbar_pix
        counts[i] /= nbar_pix   # this makes 'counts' a density-like map consistent with reference

        # --- Make randomized shear field (same normalization) ---
        gal_num = np.sum(in_bin)
        rand_theta = 2.0*np.pi*np.random.random_sample(gal_num)
        e1_corr = she.real * np.cos(rand_theta) - she.imag * np.sin(rand_theta)
        e2_corr = she.imag * np.cos(rand_theta) + she.real * np.sin(rand_theta)

        rand_map = np.zeros(npix, dtype=complex)
        rand_counts = np.zeros(npix, dtype=float)
        map_shears(rand_map, rand_counts,
                   catalogue['RA'][in_bin],
                   catalogue['DEC'][in_bin],
                   e1_corr + 1j*e2_corr, gal_wht=None)

        rand_map /= nbar_pix
        rand_counts /= nbar_pix

        # --- Spin-2 alm decomposition ---
        almE, almB = hp.sphtfunc.map2alm_spin([shear_maps[i].real, shear_maps[i].imag],
                                              spin=2, lmax=lmax)
        almE_rand, almB_rand = hp.sphtfunc.map2alm_spin([rand_map.real, rand_map.imag],
                                                        spin=2, lmax=lmax)

        if nosh:
            factor = np.sqrt((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1.)))
            almE *= factor
            almB *= factor
            almE_rand *= factor
            almB_rand *= factor

        alm.append((almE, almB))
        alm_rand.append((almE_rand, almB_rand))

    return alm, alm_rand, shear_maps




import numpy as np
import healpy as hp

def pixel_area_arcmin2(nside):
    """Return the HEALPix pixel area in arcmin^2."""
    npix = hp.nside2npix(nside)
    pixel_area_sr = 4*np.pi / npix
    sr_to_arcmin2 = (180/np.pi * 60)**2
    return pixel_area_sr * sr_to_arcmin2


def make_alm_shear_convergence_fixed_mask(
    catalogue,
    m_bias,
    nbins,
    nside,
    lmax,
    n_arcmin2,    # per-bin galaxy density [gal/arcmin^2]
    mask,         # HEALPix mask (0/1 or fractional), shape (npix,)
    nosh=False
):
    """
    Compute shear alm's from a galaxy catalogue, normalising by the
    EXPECTED number of galaxies per pixel (nbar_pix = n_arcmin2 * pixel_area * mask),
    and apply the SAME mask to both the observed and random shear maps.
    """

    # ------------------------------------------------------------------
    # 1. Geometry and expected pixel densities
    # ------------------------------------------------------------------
    npix = hp.nside2npix(nside)
    if len(mask) != npix:
        raise ValueError("mask must have length npix = hp.nside2npix(nside)")

    pix_area = pixel_area_arcmin2(nside)  # arcmin^2 per pixel

    # Build expected number of galaxies per pixel per bin:
    # nbar_pix_map[i,p] = n_arcmin2[i] * pix_area * mask[p]
    nbar_pix_map = np.zeros((nbins, npix))
    for i in range(nbins):
        nbar_pix_map[i] = n_arcmin2[i] * pix_area * mask

    # Prepare arrays
    shear_maps = np.zeros((nbins, npix), dtype=complex)
    counts = np.zeros((nbins, npix), dtype=float)

    alm, alm_rand = [], []
    ell, emm = hp.Alm.getlm(lmax=lmax)


    # ------------------------------------------------------------------
    # 2. Loop over tomographic bins
    # ------------------------------------------------------------------
    for i in range(nbins):
        in_bin = (catalogue['ZBIN'] == i)
        if not np.any(in_bin):
            # Empty bin: append zero alms
            size = hp.Alm.getsize(lmax)
            alm.append((np.zeros(size, complex), np.zeros(size, complex)))
            alm_rand.append((np.zeros(size, complex), np.zeros(size, complex)))
            continue

        # --------------------------------------------------------------
        # 2a. Per-galaxy shear with bias correction & mean removal
        # --------------------------------------------------------------
        e1 = catalogue['E1'][in_bin]
        e2 = catalogue['E2'][in_bin]
        she = (1.0 / (1.0 + m_bias[i])) * (
            (e1 - np.mean(e1)) + 1j*(e2 - np.mean(e2))
        )

        # --------------------------------------------------------------
        # 2b. Accumulate raw sums of shear into map
        # --------------------------------------------------------------
        map_shears(
            shear_maps[i],
            counts[i],
            catalogue['RA'][in_bin],
            catalogue['DEC'][in_bin],
            she,
            gal_wht=None
        )

        # --------------------------------------------------------------
        # 2c. Normalize by expected number of galaxies
        #      nbar_pix_map[i,p] = n_arcmin2[i]*pixel_area*mask[p]
        # --------------------------------------------------------------
        zero_mask = (nbar_pix_map[i] <= 0)
        shear_maps[i][zero_mask] = 0.0
        valid = ~zero_mask
        shear_maps[i][valid] /= nbar_pix_map[i][valid]

        # --------------------------------------------------------------
        # 2d. Apply the mask (explicitly zero masked pixels)
        #      This ensures geometry matches observed selection
        # --------------------------------------------------------------
        shear_maps[i][mask <= 0] = 0.0

        # --------------------------------------------------------------
        # 2e. Build random shear map (noise-only realization),
        #     and apply SAME normalization + mask
        # --------------------------------------------------------------
        gal_num = len(she)
        rand_theta = 2*np.pi*np.random.random_sample(gal_num)
        e1r = she.real * np.cos(rand_theta) - she.imag * np.sin(rand_theta)
        e2r = she.imag * np.cos(rand_theta) + she.real * np.sin(rand_theta)

        rand_map = np.zeros(npix, dtype=complex)
        rand_counts = np.zeros(npix, dtype=float)

        map_shears(
            rand_map,
            rand_counts,
            catalogue['RA'][in_bin],
            catalogue['DEC'][in_bin],
            e1r + 1j*e2r,
            gal_wht=None
        )

        rand_map[zero_mask] = 0.0
        rand_map[valid] /= nbar_pix_map[i][valid]
        rand_map[mask <= 0] = 0.0   # APPLY MASK HERE TOO ✅

        # --------------------------------------------------------------
        # 2f. Spin-2 spherical harmonic transform
        # --------------------------------------------------------------
        almE, almB = hp.sphtfunc.map2alm_spin(
            [shear_maps[i].real, shear_maps[i].imag],
            spin=2,
            lmax=lmax
        )
        almE_rand, almB_rand = hp.sphtfunc.map2alm_spin(
            [rand_map.real, rand_map.imag],
            spin=2,
            lmax=lmax
        )

        # --------------------------------------------------------------
        # 2g. Optional NO-SHANK filter
        # --------------------------------------------------------------
        if nosh:
            with np.errstate(divide='ignore', invalid='ignore'):
                factor = np.sqrt((ell*(ell+1.0))/((ell+2.0)*(ell-1.0)))
            almE *= factor
            almB *= factor
            almE_rand *= factor
            almB_rand *= factor

        # Save results
        alm.append((almE, almB))
        alm_rand.append((almE_rand, almB_rand))

    return alm, alm_rand, shear_maps
