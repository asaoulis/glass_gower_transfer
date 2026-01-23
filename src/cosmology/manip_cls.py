import numpy as np
import healpy as hp
import iolaus
from heracles.result import Result
import heracles

def denoise_shear_cls(nbins, alm, alm_rand, lmax):
    shear_cls = np.zeros((nbins, nbins, 3, lmax+1), dtype=np.ndarray)
    rand_cls = np.zeros((nbins, nbins, 3, lmax+1), dtype=np.ndarray)
    shear_cls_noiseless = np.zeros((nbins, nbins, 3, lmax+1), dtype=np.ndarray)

    for i in range(nbins):
        for j in range(i+1):
            shear_cls[i][j] = shear_cls[j][i] = hp.alm2cl(alm[i], alm[j], lmax = lmax)
            rand_cls[i][j] = rand_cls[j][i] = hp.alm2cl(alm_rand[i], alm_rand[j], lmax = lmax)
    
            shear_cls_noiseless[i][j] = shear_cls_noiseless[j][i] = shear_cls[i][j]
            if i == j:
                shear_cls_noiseless[i][j][0] -= np.average(rand_cls[i][j][0][200:], weights= 2*np.arange(200, lmax+1))
                shear_cls_noiseless[i][j][1] -= np.average(rand_cls[i][j][1][200:], weights= 2*np.arange(200, lmax+1))
    
    return shear_cls_noiseless


def coord2hpx(lon, lat, nside, radec=True):
    """
    Convert a catalogue to a HEALPix map of number counts per resolution
    element.

    Parameters
    ----------
    lon, lat : (ndarray, ndarray)
        Coordinates of the sources in degree. If radec=True, assume input is in the icrs
        coordinate system. Otherwise assume input is glon, glat

    nside : int
        HEALPix nside of the target map

    Return
    ------
    hpx_map : ndarray
        HEALPix map of the catalogue number counts in Galactic coordinates

    """

    npix = hp.nside2npix(nside)

    # conver to theta, phi
    theta = np.radians(90. - lat)
    phi = np.radians(lon)

    # convert to HEALPix indices
    indices = hp.ang2pix(nside, theta, phi)

    idx, counts = np.unique(indices, return_counts=True)

    # fill the fullsky map
    hpx_map = np.zeros(npix, dtype=int)
    hpx_map[idx] = counts

    return hpx_map

def cat2mask(catalogue, nbins, nside):
    """
    Create a mask from the catalogue. The mask is a HEALPix map of the number
    of galaxies per pixel.

    Parameters
    ----------
    catalogue : ndarray
        The catalogue to be masked

    nbins : int
        Number of tomographic bins

    Return
    ------
    mask : ndarray
        HEALPix map of the number counts per pixel

    """

    bin_masks = np.zeros((nbins, hp.nside2npix(nside)), dtype=int)

    for i in range(nbins):
        in_bin = (catalogue['ZBIN'] == i)
        bin_masks[i] = coord2hpx(catalogue['RA'][in_bin], catalogue['DEC'][in_bin], nside)
        bin_masks[i][bin_masks[i] > 0] = 1

    return bin_masks


import numpy as np
import healpy as hp

def cat2mask_fractional_2(catalogue, nbins, frac_mask,
                                                         nside=None, return_diagnostics=False):
    """
    Build bin-dependent sampling masks W_p = 1 - exp(-mu_p) using a fractional mask
    already at the target nside.

    Parameters
    ----------
    catalogue : structured array or dict-like
        Fields: 'RA' (deg), 'DEC' (deg), 'ZBIN' (int from 0..nbins-1)
    nbins : int
        number of tomographic bins
    frac_mask : ndarray, length = hp.nside2npix(nside)
        fractional mask values in [0,1] at the desired nside (the "true" mask)
    nside : int, optional
        healpix nside of frac_mask; deduced from length if None
    return_diagnostics : bool
        if True return (masks, mu_maps, rho_bins, fsky_bins)

    Returns
    -------
    masks : ndarray shape (nbins, npix)
        sampling masks in [0,1] for each bin
    (optionally) mu_maps : ndarray (nbins, npix)
        expected counts per pixel mu_p
    rho_bins : ndarray (nbins,)
        surface density used for each bin (gal/steradian)
    fsky_bins : ndarray (nbins,)
        effective f_sky (sum W_p * A_pix / 4pi) for each bin
    """
    if nside is None:
        nside = hp.npix2nside(len(frac_mask))
    npix = hp.nside2npix(nside)
    assert len(frac_mask) == npix, "frac_mask length does not match nside"

    # pixel area (steradian)
    A_pix = 4.0 * np.pi / float(npix)

    masks = np.zeros((nbins, npix), dtype=np.float64)
    mu_maps = np.zeros_like(masks)
    rho_bins = np.zeros(nbins, dtype=np.float64)
    fsky_bins = np.zeros(nbins, dtype=np.float64)

    # effective unmasked area for this fractional mask
    Omega_eff = np.sum(frac_mask) * A_pix  # steradian

    # Precompute pixel indices of catalogue
    ra = np.asarray(catalogue['RA'])
    dec = np.asarray(catalogue['DEC'])
    theta = np.deg2rad(90.0 - dec)
    phi = np.deg2rad(ra)
    pix_cat = hp.ang2pix(nside, theta, phi)

    for ibin in range(nbins):
        sel = (catalogue['ZBIN'] == ibin)
        Nbin = np.count_nonzero(sel)
        if Nbin == 0:
            masks[ibin, :] = 0.0
            mu_maps[ibin, :] = 0.0
            rho_bins[ibin] = 0.0
            fsky_bins[ibin] = 0.0
            continue

        # surface density (gal/steradian) estimated using the fractional mask area
        rho = Nbin / Omega_eff
        rho_bins[ibin] = rho

        # expected counts per pixel (including fractional area f_p)
        mu = rho * A_pix * frac_mask  # vector length npix
        mu_maps[ibin, :] = mu

        # sampling probability (= fractional sampling mask)
        W = 1.0 - np.exp(-mu)

        # enforce zero where the fractional mask is exactly zero
        W[frac_mask <= 0.0] = 0.0

        masks[ibin, :] = W

        # compute f_sky as effective fraction of full sky contributed by sampling mask
        fsky_bins[ibin] = np.sum(W) * A_pix / (4.0 * np.pi)

    if return_diagnostics:
        return masks, mu_maps, rho_bins, fsky_bins
    return masks



def maskcls(masks, lmax, nbins):
    """
    Compute the masked power spectrum of the catalogue

    Parameters
    ----------
    masks : ndarray
        array of binary mask HEALPix maps

    Return
    ------
    masked_cls : ndarray
        cls of the realised mask for each tomographic bin pair

    """

    masked_cls = np.zeros((nbins, nbins, lmax+1), dtype=np.ndarray)

    for i in range(nbins):
        for j in range(i+1):
            masked_cls[i][j] = masked_cls[j][i] = hp.sphtfunc.anafast(masks[i], masks[j], lmax=lmax, pol=False)
    
            if i == j:
                masked_cls[i][j] = hp.sphtfunc.anafast(masks[i], lmax=lmax, pol=False)
    inverse_pixwin = hp.sphtfunc.pixwin(hp.get_nside(masks[0]), lmax=lmax, pol=True)[1]**-2
    inverse_pixwin[0], inverse_pixwin[1] = 1.0, 1.0
    masked_cls *= inverse_pixwin[np.newaxis, np.newaxis, :] 
    return masked_cls

def to_heracles_cl(num_bins, shear_cls):
    data_cls = {}
    for i in range(num_bins):
        for j in range(i + 1):
            temp_cl = np.empty((4, shear_cls.shape[-1]))
            temp_cl[0] = shear_cls[i][j][0]
            temp_cl[1] = shear_cls[i][j][1]
            temp_cl[2] = shear_cls[i][j][2]
            temp_cl[3] = shear_cls[i][j][2]
            data_cls[("SHE", "SHE", i + 1, j + 1)] = temp_cl
    return data_cls

def to_heracles_noise_cl(num_bins, cl_noise, lmin, lmax):
    mask_cls = {}
    for i in range(num_bins):
        for j in range(i + 1):
            mask_cls[("WHT", "WHT", i + 1, j + 1)] = cl_noise[i][j][lmin:lmax+1]
    return mask_cls

def to_heracles_mask_cl(num_bins, mask_cls, lmin, lmax):
    cls = {}
    for i in range(num_bins):
        for j in range(i + 1):
            cls_temp = mask_cls[i][j][lmin:lmax+1]
            cls_temp = Result(cls_temp, axis=(0,))
            cls[("WHT", "WHT", i + 1, j + 1)] = cls_temp
    
    return cls
    
def to_heracles_data_cl(num_bins, data_cls):
    cls = {}
    for i in range(num_bins):
        for j in range(i + 1):
            temp_cl = np.empty((2, 2, data_cls.shape[-1]))
            temp_cl[0][0] = data_cls[i][j][0]
            temp_cl[0][1] = data_cls[i][j][1]
            temp_cl[1][0] = data_cls[i][j][2]
            temp_cl[1][1] = data_cls[i][j][2]
            temp_cl = Result(temp_cl, axis=(2,))
            cls[("SHE", "SHE", i + 1, j + 1)] = temp_cl
            
    return cls

def unmix_shear_cl(num_bins, shear_cls, mask_cl, lmin, lmax, patch_hole=True):
    data_cls = to_heracles_data_cl(
        num_bins=6,
        data_cls=shear_cls
    )

    mask_cls = to_heracles_mask_cl(num_bins=num_bins, mask_cls=mask_cl, lmin=lmin, lmax=lmax)
    unmixed_cls = heracles.unmixing.natural_unmixing(data_cls, mask_cls, patch_hole=patch_hole)
    kids_unmixed_cls = heracles_to_kids_cls(num_bins, unmixed_cls, lmax)
    return kids_unmixed_cls

def heracles_to_kids_cls(num_bins, heracles_cls, lmax):
    cls = np.zeros((num_bins, num_bins, 3, lmax+1), dtype=np.ndarray)
    for i in range(num_bins):
        for j in range(i + 1):
            cls[i][j][0] = heracles_cls[("SHE", "SHE", i + 1, j + 1)][0, 0]
            cls[i][j][1] = heracles_cls[("SHE", "SHE", i + 1, j + 1)][1, 1]
            cls[i][j][2] = heracles_cls[("SHE", "SHE", i + 1, j + 1)][1, 0]
            cls[j][i] = cls[i][j]
    
    return cls
    
def make_bandpowers(lower_lscale, upper_lscale, cls, nbands):
    '''
    n is the number of bands
    '''
    ell = np.arange(lower_lscale, upper_lscale+1)
    band_cutoffs = np.logspace(np.log(lower_lscale), np.log(upper_lscale+1), nbands+1, base = np.e)
    bandpowers, centre_ell = np.zeros(nbands), np.zeros(nbands)
    for i in range(nbands):
        temp_l = ell[np.logical_and(ell >= band_cutoffs[i], ell < band_cutoffs[i+1])]
        temp_cls = cls[np.logical_and(ell >= band_cutoffs[i], ell < band_cutoffs[i+1])].copy()
        temp_cls *= temp_l * (temp_l + 1)
        bandpowers[i] = np.sum(temp_cls)/(2*np.pi*( band_cutoffs[i+1] - band_cutoffs[i])) #Use Brown et al. 2005 multipole binning scheme
        centre_ell[i] = band_cutoffs[i] + 0.5*(band_cutoffs[i+1] - band_cutoffs[i])
    return bandpowers, centre_ell, band_cutoffs

def compute_cl_bandpowers(realised_unmixed_shear_cls_cut, nbins, lower_lscale, upper_lscale, nbands):
    theory_bandpowers_stacked = np.array([])

    for i in range(nbins):
        for j in range(0, nbins):
            if i<j:
                pass
            else:
                bandpowers, centre_ell, band_cutoffs = make_bandpowers(lower_lscale, upper_lscale, cls=realised_unmixed_shear_cls_cut[i][j][0], nbands=nbands)
                theory_bandpowers_stacked = np.append(theory_bandpowers_stacked, bandpowers)

    theory_bandpowers = theory_bandpowers_stacked.reshape(int(nbins*(nbins+1)/2), -1)
    return centre_ell, theory_bandpowers

def unmixing_mask_cls(catalogue, nbins, nside, lmax, lmin):
    bin_masks = cat2mask(catalogue, nbins, nside)
    bin_mask_cls = maskcls(bin_masks, lmax=int(1.5 * lmax), nbins=nbins)
    return bin_mask_cls

def process_cls(bin_mask_cls, nbins, nside, alm, alm_rand, lower_lscale, upper_lscale, lmin, lmax, nbands):
    shear_cls_noiseless = denoise_shear_cls(nbins, alm, alm_rand, lmax)
    realised_unmixed_shear_cls = unmix_shear_cl(num_bins = nbins, shear_cls = shear_cls_noiseless, mask_cl = bin_mask_cls, lmin = lmin, lmax = lmax)
    realised_unmixed_shear_cls_cut = realised_unmixed_shear_cls[:, :, :, lower_lscale:upper_lscale+1]
    cll_bands, bandpowers = compute_cl_bandpowers(realised_unmixed_shear_cls_cut, nbins, lower_lscale, upper_lscale, nbands)
    return realised_unmixed_shear_cls, cll_bands, bandpowers
