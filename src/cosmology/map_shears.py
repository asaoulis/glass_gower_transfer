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

def make_alm_shear_convergence(catalogue, m_bias, nbins, nside, lmax, nosh=True, compute_convergence=True):
    shear = np.zeros((nbins, hp.nside2npix(nside)), dtype=complex)
    counts = np.zeros_like(shear, dtype=int)

    alm, alm_rand = [], []
    E_maps, B_maps = [], []

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

        # Clean monopole/dipole
        almE[ell <= 1] = 0.0
        almB[ell <= 1] = 0.0

        # Save
        alm.append((almE, almB))
        alm_rand.append((almE_rand, almB_rand))

        # Back to real-space maps
        if compute_convergence:
            E_maps.append(hp.alm2map(almE, nside=nside, lmax=lmax, pol=False))
            B_maps.append(hp.alm2map(almB, nside=nside, lmax=lmax, pol=False))

    return alm, alm_rand, shear, E_maps, B_maps