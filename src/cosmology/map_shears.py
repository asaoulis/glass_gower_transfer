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

def make_alm_shear_convergence(catalogue, m_bias, nbins, nside, lmax, nosh=False):
    shear = np.zeros((nbins, hp.nside2npix(nside)), dtype=complex)
    counts = np.zeros_like(shear, dtype=int)

    alm, alm_rand = [], []

    ell, emm = hp.Alm.getlm(lmax=lmax)

    for i in range(nbins):
        in_bin = (catalogue['ZBIN'] == i)

        she = (1/(1+m_bias[i])) * (
            (catalogue['E1'][in_bin] - np.mean(catalogue['E1'][in_bin]))
            + 1j*(catalogue['E2'][in_bin] - np.mean(catalogue['E2'][in_bin]))
        )

        map_shears(shear[i], counts[i],
                   catalogue['RA'][in_bin],
                   catalogue['DEC'][in_bin],
                   she, gal_wht=None)

        shear[i][counts[i] > 0] = shear[i][counts[i] > 0] / counts[i][counts[i] > 0]

        # Make randomized shear field
        gal_num = len(catalogue[in_bin])
        rand_theta = 2*np.pi*np.random.random_sample(gal_num)
        e1_corr = she.real*np.cos(rand_theta) - she.imag*np.sin(rand_theta)
        e2_corr = she.imag*np.cos(rand_theta) + she.real*np.sin(rand_theta)

        rand = np.zeros(hp.nside2npix(nside), dtype=complex)
        _ = np.zeros_like(rand, dtype=int)

        map_shears(rand, _, catalogue['RA'][in_bin], catalogue['DEC'][in_bin],
                   e1_corr + 1j*e2_corr, gal_wht=None)
        rand[_ > 0] = rand[_ > 0] / _[_ > 0]

        # Compute spin-2 alm decomposition
        almE, almB = hp.sphtfunc.map2alm_spin([shear[i].real, shear[i].imag],
                                              spin=2, lmax=lmax)
        almE_rand, almB_rand = hp.sphtfunc.map2alm_spin([rand.real, rand.imag],
                                                        spin=2, lmax=lmax)

        if nosh:
            factor = np.sqrt((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1.)))
            almE *= factor
            almB *= factor
            almE_rand *= factor
            almB_rand *= factor


        # Save
        alm.append((almE, almB))
        alm_rand.append((almE_rand, almB_rand))

    return alm, alm_rand, shear