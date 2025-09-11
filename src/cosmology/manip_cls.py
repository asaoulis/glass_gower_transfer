import numpy as np
import healpy as hp
import iolaus


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


def unmix_shear_cl(num_bins, shear_cls, mask_cl, lmin, lmax):
    B = np.identity(lmax - lmin + 1)

    data_cls = to_heracles_cl(
        num_bins=6,
        shear_cls=shear_cls
    )

    mask_cls = to_heracles_noise_cl(num_bins=num_bins, cl_noise=mask_cl, lmin=lmin, lmax=lmax)
    dnp = iolaus.Naive_Polspice(data_cls, mask_cls, B, patch_hole=True)
    compsep_dnp = iolaus.compsep_cls(dnp)
    
    unmixed_cls = np.zeros(shear_cls.shape, dtype=np.ndarray)
    for i in range(num_bins):
        for j in range(i + 1):
            unmixed_cls[i][j][0] = compsep_dnp[("G_E", "G_E", i + 1, j + 1)]
            unmixed_cls[i][j][1] = compsep_dnp[("G_B", "G_B", i + 1, j + 1)]
            unmixed_cls[i][j][2] = compsep_dnp[("G_E", "G_B", i + 1, j + 1)]
            unmixed_cls[j][i] = unmixed_cls[i][j]
    
    return unmixed_cls
    
def make_bandpowers(lmin, lmax, cls, nbands):
    '''
    n is the number of bands
    '''
    ell = np.arange(lmin, lmax+1)
    band_cutoffs = np.logspace(np.log(lmin), np.log(lmax+1), nbands+1, base = np.e)
    bandpowers, centre_ell = np.zeros(nbands), np.zeros(nbands)
    for i in range(nbands):
        temp_l = ell[np.logical_and(ell >= band_cutoffs[i], ell < band_cutoffs[i+1])]
        temp_cls = cls[np.logical_and(ell >= band_cutoffs[i], ell < band_cutoffs[i+1])]
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
                bandpowers, centre_ell, band_cutoffs = make_bandpowers(lmin=lower_lscale, lmax=upper_lscale, cls=realised_unmixed_shear_cls_cut[i][j][0], nbands=nbands)
                theory_bandpowers_stacked = np.append(theory_bandpowers_stacked, bandpowers)

    theory_bandpowers = theory_bandpowers_stacked.reshape(int(nbins*(nbins+1)/2), -1)
    return centre_ell, theory_bandpowers