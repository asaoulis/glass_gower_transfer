import numpy as np
import healpy as hp

# use the CAMB cosmology that generated the matter power spectra
import camb
from cosmology import Cosmology

from scipy import integrate
from scipy.stats.qmc import Sobol

import levinpower

# GLASS modules: cosmology and everything in the glass namespace
import glass.shells
import glass.points
import glass.shapes
import glass.lensing
import glass.galaxies
import glass.observations

import iolaus

from numba import njit

import time
import os

# Get some constants
alpha = 0.5

# Sobol sequence generator
def draw_sobol_samples(prior: dict, num_samples: int) -> np.ndarray:
    """
    Draw Sobol samples from the prior distribution.
    
    Args:
        prior: The prior distribution object.
        
    Returns:
        A numpy array of Sobol samples.
    """
    # Create a Sobol sequence generator
    nd = len(prior)
    sobol_samples = Sobol(d=nd, scramble=True, seed=100)
    
    # Draw samples from the Sobol sequence
    samples = sobol_samples.random(n=num_samples)
    
    widths = np.array([prior[key][1] - prior[key][0] for key in prior])
    lower = np.array([prior[key][0] for key in prior])
    # Transform the samples according to the prior distribution
    transformed_samples = samples*widths + lower
    
    return transformed_samples

# Get As as a parameter
def s8_to_As(s8,
             h,
             ombh2,
             omch2,
             omega_k,
             mnu,
             w0,
             wa,
             ns,
             fid_As = 2.1e-9, 
             kmax = 1.2, 
             k_per_logint = 5):
        
    Omega_c = omch2/(h**2)
    Omega_b = ombh2/(h**2)
    Omega_m = Omega_c + Omega_b
    alpha = 0.5
    sigma8 = s8/((Omega_m/0.3)**alpha)
    p = camb.CAMBparams(WantTransfer=True, 
                        Want_CMB=False, Want_CMB_lensing=False, DoLensing=False, 
                        NonLinear="NonLinear_none",
                        WantTensors=False, WantVectors=False, WantCls=False, 
                        WantDerivedParameters=False,
                        want_zdrag=False, want_zstar=False)
    p.set_accuracy(DoLateRadTruncation=True)
    p.Transfer.high_precision = False
    p.Transfer.accurate_massive_neutrino_transfers = False
    p.Transfer.kmax = kmax
    p.Transfer.k_per_logint = k_per_logint
    p.Transfer.PK_redshifts = np.array([0.0])

    p.set_cosmology(H0=h*100, ombh2=ombh2, omch2=omch2, omk=omega_k, mnu=mnu)
    p.set_dark_energy(w=w0, wa=wa)
    p.set_initial_power(camb.initialpower.InitialPowerLaw(As=fid_As, ns=ns))

    p.Reion = camb.reionization.TanhReionization()
    p.Reion.Reionization = False

    r = camb.get_results(p)

    fid_sigma8 = r.get_sigma8()[-1]

    As = fid_As*(sigma8/fid_sigma8)**2
    
    return sigma8, As, Omega_c, Omega_b, Omega_m

# Cosmology functions
def linear_extend(x, y, xmin, xmax, nmin, nmax, nfit):
    if xmin < x.min():
        xf = x[:nfit]
        yf = y[:nfit]
        p = np.polyfit(xf, yf, 1)
        xnew = np.linspace(xmin, x.min(), nmin, endpoint=False)
        ynew = np.polyval(p, xnew)
        x = np.concatenate((xnew, x))
        y = np.concatenate((ynew, y))
    if xmax > x.max():
        xf = x[-nfit:]
        yf = y[-nfit:]
        p = np.polyfit(xf, yf, 1)
        xnew = np.linspace(x.max(), xmax, nmax, endpoint=True)
        # skip the first point as it is just the xmax
        xnew = xnew[1:]
        ynew = np.polyval(p, xnew)
        x = np.concatenate((x, xnew))
        y = np.concatenate((y, ynew))
    return x, y

def extrapolate_section(k, z, Pk, kmin, kmax, nmin, nmax, npoint):
    # load current values
    nz = len(z)
    # load other current values
    # extrapolate
    P_out = []
    for i in range(nz):
        Pi = Pk[i, :]
        logk, logp = linear_extend(np.log(k), np.log(Pi), np.log(
            kmin), np.log(kmax), nmin, nmax, npoint)
        P_out.append(np.exp(logp))
    
    return np.exp(logk), np.array(P_out)

def growth_integrand(a, cosmo):
    if a == 0:
        return 0.0
    # For standard cosmologies, (x * E(1/x-1))^-3 approaches x^(3/2) or x^3 near x=0, so integral converges.
    return (a * cosmo.ef(1.0/a - 1.0))**-3.0

def growth_integral_value(a_val, cosmo):
    result, error = integrate.quad(growth_integrand, 0, a_val, args=(cosmo,))
    return result

def linear_growth_factor(z, cosmo):
    integral_at_z0 = growth_integral_value(1.0, cosmo)

    if isinstance(z, (list, np.ndarray)):
        z_arr = np.asarray(z)
        results = np.empty_like(z_arr, dtype=float)
        for i, z_val in enumerate(z_arr):
            if z_val < 0:
                results[i] = np.nan
                continue
            a_val = 1.0 / (1.0 + z_val)
            integral_at_a = growth_integral_value(a_val, cosmo)
            results[i] = (cosmo.ef(z_val) * integral_at_a) / (cosmo.ef(0) * integral_at_z0)
        return results
    else:
        if z < 0:
            return np.nan
        a = 1.0 / (1.0 + z)
        integral_at_a = growth_integral_value(a, cosmo)
        return (cosmo.ef(z) * integral_at_a) / (cosmo.ef(0) * integral_at_z0)
    
def kappa_ia_nla_m(delta, zeff, red_f, cosmo, a_ia, b_ia, log10_M,
                   log10_M_pivot=13.5 #solar masses / h
    ):
    
    # c1 = 5e-14 / cosmo.h**2  # Solar masses per cubic Mpc
    # rho_c1 = c1 * cosmo.rho_c_z(0.0)

    rho_c1 = 0.0134

    prefactor = -a_ia * rho_c1 * cosmo.omega_m
    inverse_linear_growth = 1.0 / linear_growth_factor(zeff, cosmo)
    mass_term = (10**(log10_M)/10**(log10_M_pivot))**(b_ia)

    f_nla = (
        red_f * prefactor * inverse_linear_growth * mass_term
    )
    return delta * f_nla

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

# Glass Kernels
def make_kernels(pars, z_distance, chi_distance, zbins):
    kernels = []
    
    for i in range(len(zbins) - 1):
        z_range = z_distance[((zbins[i + 1] >= z_distance) & (z_distance > zbins[i]))]
        chi_range = chi_distance[((zbins[i + 1] >= z_distance) & (z_distance > zbins[i]))]
        w = pars.DarkEnergy.w + pars.DarkEnergy.wa * np.divide(z_range, z_range + 1)

        # E_of_z      = np.sqrt(config['omega_m']*(1+z_range)**3 + config['omega_k']*(1+z_range)**2 + config['omega_lambda']*(1+z_range)**(3*(1+w)))
        omega_de = 1 + pars.omk - pars.omegam - pars.omeganu
        E_of_z = np.sqrt(
            pars.omegam * (1 + z_range) ** 3
            + pars.omk * (1 + z_range) ** 2
            + omega_de * (1 + z_range) ** (3 * (1 + w))
        )

        k = np.zeros(len(z_distance))
        k[((zbins[i + 1] >= z_distance) & (z_distance > zbins[i]))] = np.divide(
            chi_range**2, E_of_z
        )
        k /= np.trapezoid(k, z_distance)
        kernels.append(k)
     
    return np.array(kernels)

# gen log space ell
def gen_log_space(limit, n):
    result = [1]
    if n>1:
        ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    while len(result)<n:
        next_value = result[-1]*ratio
        if next_value - result[-1] >= 1:
            result.append(next_value)
        else:
            result.append(result[-1]+1)
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    return np.array(list(map(lambda x: round(x)-1, result)), dtype=np.uint64)

# Measure alm from catalogues
def make_alm(catalogue, nbins, nside, m_bias, lmax):
    shear = np.zeros((nbins, hp.nside2npix(nside)), dtype=complex)
    counts = np.zeros_like(shear, dtype=int)

    alm, alm_rand = [], []

    for i in range(nbins):
        in_bin = (catalogue['ZBIN'] == i)

        she = (1/1/(1+m_bias[i])) * ((catalogue['E1'][in_bin] - np.mean(catalogue['E1'][in_bin])) + 1j*(catalogue['E2'][in_bin] - np.mean(catalogue['E2'][in_bin])))

        map_shears(shear[i], counts[i], catalogue['RA'][in_bin],  catalogue['DEC'][in_bin], she, gal_wht=None)

        shear[i][counts[i] > 0] = np.divide(shear[i][counts[i] > 0], counts[i][counts[i] > 0])

        gal_num = len(catalogue[in_bin])
        rand_theta = 2*np.pi*np.random.random_sample(gal_num)

        e1_corr = she.real*np.cos(rand_theta) - she.imag*np.sin(rand_theta)
        e2_corr = she.imag*np.cos(rand_theta) + she.real*np.sin(rand_theta)

        rand = np.zeros(hp.nside2npix(nside), dtype=complex)
        _ = np.zeros_like(rand, dtype=int)

        map_shears(rand, _, catalogue['RA'][in_bin],  catalogue['DEC'][in_bin],  e1_corr + 1j * e2_corr, gal_wht=None)

        rand[_ > 0] = np.divide(rand[_ > 0], _[_ > 0])

        alm.append(hp.sphtfunc.map2alm_spin([shear[i].real, shear[i].imag], spin = 2, lmax = lmax)) #Compute the alms for the shear field
        alm_rand.append(hp.sphtfunc.map2alm_spin([rand.real, rand.imag], spin = 2, lmax = lmax)) #Compute the alms for the random shear field
    
    return alm, alm_rand

def denoise_shear_cls(alm, alm_rand, nbins, lmax):
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

# Catalogue to HEALPix maps
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

# Polspice
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

if __name__ == "__main__":
    s = time.time() # Start timer 
    data_dir = '/home/kiyam/cosmology/kids-legacy-sbi/data'
    output_dir = '/home/kiyam/cosmology/kids-legacy-sbi/sim_output'
    sim_batch_name = 'sobol_batch_1'
    os.makedirs(f'{output_dir}/{sim_batch_name}', exist_ok=True)
    sim_num = 5
    
    seed_num = None # Seed for reproducibility
    shift_nz = True # Whether to shift the n(z) using the dz biases and covariance
    do_variable_depth = False
    
    prior = {'s8': (0.5, 1.0),
                'h': (0.64, 0.82),
                'ns': (0.84, 1.1),
                'ombh2': (0.019, 0.026),
                'omch2': (0.051, 0.255),
                'w0': (-1.0, -1.0),
                'wa': (0.0, 0.0),
                'logT_AGN':(7.3, 8.3),
                'mnu': (0.06, 0.06),
                'a_ia': (4.48, 7.0),
                'b_ia': (0.28, 0.6),}
    
    sobol_samples = np.loadtxt(f'{output_dir}/{sim_batch_name}_params.txt')
    
    sim_params = sobol_samples
    
    # cosmology for the simulation
    h = sim_params[:, list(prior.keys()).index('h')]
    omch2 = sim_params[:, list(prior.keys()).index('omch2')]
    ombh2 = sim_params[:, list(prior.keys()).index('ombh2')]
    s8 = sim_params[:, list(prior.keys()).index('s8')]
    w0 = sim_params[:, list(prior.keys()).index('w0')]
    wa = sim_params[:, list(prior.keys()).index('wa')]
    ns = sim_params[:, list(prior.keys()).index('ns')]
    logT_AGN = sim_params[:, list(prior.keys()).index('logT_AGN')]
    mnu = sim_params[:, list(prior.keys()).index('mnu')]
    a_ia = sim_params[:, list(prior.keys()).index('a_ia')]
    b_ia = sim_params[:, list(prior.keys()).index('b_ia')]
    sims = sim_params[:,-1].astype(int) # Pick the simulation number to run
    
    # Define some constants
    h = h[sim_num]
    omch2 = omch2[sim_num]
    ombh2 = ombh2[sim_num]
    s8 = s8[sim_num]
    w0 = w0[sim_num]
    wa = wa[sim_num]
    ns = ns[sim_num]
    logT_AGN = logT_AGN[sim_num]
    a_ia = a_ia[sim_num]
    b_ia = b_ia[sim_num]
    mnu = mnu[sim_num]
    sim_num = sims[sim_num]
    
    omega_k = 0.0
    
    # Convert s8 to As and get concordant cosmology values
    sigma8, As, Omega_c, Omega_b, Omega_m = s8_to_As(s8, h, ombh2, omch2, omega_k, mnu, w0, wa, ns)
    
    # intrinsic alignments params
    massdep_means = np.loadtxt(f'{data_dir}/priors/massdep_means.txt')
    massdep_cov = np.loadtxt(f'{data_dir}/priors/massdep_cov.txt')
    log10_M_eff_means = massdep_means[2:]
    log10_M_eff_cov = massdep_cov[2:,2:] 
    f_red = np.array([0.15, 0.2, 0.17, 0.24, 0.19, 0.03])
    log10_M_eff = np.random.multivariate_normal(log10_M_eff_means, log10_M_eff_cov, size=1)[0]

    # Save the cosmology ---------------------------------------------------------------------
    cosmo_params = {'h': h,
                    'omch2': omch2,
                    'ombh2': ombh2,
                    's8': s8,
                    'w0': w0,
                    'wa': wa,
                    'ns': ns,
                    'logT_ANG': logT_AGN,
                    'a_ia': a_ia,
                    'b_ia': b_ia,
                    'sigma8': sigma8,
                    'As': As,
                    'Omega_c': Omega_c,
                    'Omega_b': Omega_b,
                    'Omega_m': Omega_m}

    np.save(f'{output_dir}/{sim_batch_name}/cosmo_params_{sim_num}.npy', cosmo_params)
            
    # Set the other simulation parameters ---------------------------------------------------
    
    # basic parameters of the simulation
    nside = 1024
    lmax = 2*nside
    lmin = 0
    
    # Redshfit settings
    zmin = 0.0
    zmid = 2.0
    nz_mid = 50
    zmax = 6.0
    nz = 256
    z_grid = np.linspace(zmin, zmax, nz)

    # Levin settings
    kmax = 500.0
    kmin = 1e10
    nmin = 50
    npoint = 3
    nmax = 200
    
    # CAMB ---------------------------------------------------
    pars = camb.set_params(H0=h*100, 
                           ombh2=ombh2, 
                           omch2=omch2, 
                           HMCode_logT_AGN=logT_AGN,
                           As=As, 
                           ns=ns, 
                           w=w0,
                           wa=wa,
                           halofit_version='mead2020', 
                           neutrino_hierarchy='normal',
                           DoLensing=False,
                           NonLinear=camb.model.NonLinear_both,
                           lmax=3000)

    pars.set_matter_power(redshifts=z_grid, kmax=20.0)

    cosmo = Cosmology.from_camb(pars)
    results = camb.get_results(pars)
    results.calc_power_spectra(pars)
    k, z_grid, pk = results.get_nonlinear_matter_power_spectrum()
    chi_grid = results.angular_diameter_distance(z_grid) * (1+z_grid) # the correpsonding comoving distances 
    extended_k, extended_pk = extrapolate_section(k, z_grid, pk, kmin, kmax, nmin, nmax, npoint)
    
    print("Cosmology and CAMB parameters set up successfully.")
    print(f"It took {time.time() - s:.2f} seconds")
    
    # GLASS setup --------------------------------------------------------------------
    # shells of 200 Mpc in comoving distance spacing
    zb = glass.shells.distance_grid(cosmo, 0.000, 3.1, dx=200.)
    chi_lims = results.angular_diameter_distance(zb) * (1+zb)
    zbins = [(zb[i], zb[i+1]) for i in range(len(zb)-1)]
    
    n_los_chi = 1000 #define the integration limits here
    los_z_integration = np.linspace(zb[0], zb[-1], n_los_chi)
    los_chi_integration = np.asarray(results.angular_diameter_distance(los_z_integration) * (1+los_z_integration))
    ws = glass.shells.tophat_windows(zb)
    
    # LEVIN --------------------------------------------------------------------
    print("Setting up LevinPower...")
    s_levin = time.time()
    
    ell_max = 1600
    ell_min = 2
    n_ell = 1600

    levin_ell_max = 30000
    levin_ell_min = 2
    ell_limber = 30000
    ell_nonlimber = 1600
    max_number_subintervals = 30
    N_nonlimber = 200
    N_limber = 100
    Ninterp = 1000
    n_glass_shells = len(ws)
    
    kernels = make_kernels(pars, los_z_integration, los_chi_integration, zb)
    ell = list(map(int, gen_log_space(ell_max, int(n_ell)) + ell_min))
    
    print("Setting up LevinPower parameters...")
    lp = levinpower.LevinPower(
        False,
        number_count=int(n_glass_shells),
        z_bg=z_grid,
        chi_bg=chi_grid,
        chi_cl=los_chi_integration,
        kernel=kernels.T,
        k_pk=extended_k,
        z_pk=z_grid,
        pk=extended_pk.flatten(),
        boxy=True,
        )
    
    lp.set_parameters(
        ELL_limber=ell_limber,
        ELL_nonlimber=ell_nonlimber,
        max_number_subintervals=max_number_subintervals,
        minell=int(levin_ell_min),
        maxell=int(levin_ell_max),
        N_nonlimber=N_nonlimber,
        N_limber=N_limber,
        Ninterp=Ninterp,
    )

    print("Computing angular power spectra...")
    Cl_gg, Cl_gs, Cl_ss = lp.compute_C_ells(ell)
    Cl_gg = np.array(Cl_gg)
    print(f"LevinPower took {time.time() - s_levin:.2f} seconds")
    
    # Cl reordering --------------------------------------------------------------------
    idx_ls = np.array([[i, j] for i in range(0, n_glass_shells) for j in range(0, i+1)]) #Generates bin permutations such that i>=j
    idx_sl = np.array([[i, j+i] for i in range(0, n_glass_shells) for j in range(0, n_glass_shells-i)]) #Generates bin permuatations such that i<=j
    new_order = [np.where((idx_sl == pair).all(axis=1))[0][0] for pair in idx_ls[:, [1, 0]]]
    Cl_gg_reordered = Cl_gg[new_order] #Reorders Cls s.t. bins are ordered following i>=j
    n_glass_shells -= 1
    
    # Drop the last shell --------------------------------------------------------------------
    ws = ws[:-1]  # Remove the last shell as it is not used in t§he analysis
    counter = 0
    
    # Finish cl reordering --------------------------------------------------------------------
    cl_gg_blocks = {}
    for i in range(0, n_glass_shells):
        for j in range(0, i+1):
            c_ell = Cl_gg_reordered[counter]
            cl_gg_blocks[f'W{i+1}xW{j+1}'] = c_ell
            counter += 1
        
    glass_cls = np.array([cl_gg_blocks[f'W{i}xW{j}'] for i in range(1, n_glass_shells+1) for j in range(i, 0, -1)])
    
    # Save cl's --------------------------------------------------------------------
    np.save(f'{output_dir}/{sim_batch_name}/glass_cls_{sim_num}.npy', glass_cls)
    print(f'Saved glass_cls_{sim_num}.npy')
    
    #! TODO Add in method to load glass_cls from file to skip levin step if possible
    
    # Get GLASS fields --------------------------------------------------------------------
    print("Setting up GLASS fields...")
    glass_cls_discretized = glass.discretized_cls(glass_cls, nside=nside, lmax=lmax, ncorr=1)
    fields = glass.lognormal_fields(ws)
    gls = glass.solve_gaussian_spectra(fields, glass_cls_discretized)
    # creating a numpy random number generator for sampling
    rng = np.random.default_rng(seed=seed_num)
    # generator for lognormal matter fields
    matter = glass.generate(fields, gls, nside, ncorr=1, rng=rng)
    # this will compute the convergence field iteratively
    convergence = glass.lensing.MultiPlaneConvergence(cosmo)
    print('Calculated the convergence field with GLASS')
    
    # KiDS-Legacy no-variable depth properties --------------------------------------------------------------------
    print('Setting up KiDS-Legacy properties...')
    
    # Galaxy density in each tomographic bin
    n_arcmin2 = np.array([1.7698, 1.6494, 1.4974, 1.4578, 1.3451, 1.0682]) # per arcmin^2

    # Instrinsic galaxy shape dispersion per tomographic bin
    sigma_e = np.array([0.2772, 0.2716, 0.2899, 0.2619, 0.2802, 0.3002])

    # Tomographic redshift bins
    nbins = 6

    ztomo = [
        (0.10, 0.42),
        (0.42, 0.58),
        (0.58, 0.71),
        (0.71, 0.90),
        (0.90, 1.14),
        (1.14, 2.00)
    ]

    tomo_nz = np.zeros((nbins, n_los_chi))

    ztomo_label = [
        ('0.10', '0.42'),
        ('0.42', '0.58'),
        ('0.58', '0.71'),
        ('0.71', '0.90'),
        ('0.90', '1.14'),
        ('1.14', '2.00')
    ]
    
    if shift_nz:
        dz_biases = np.loadtxt(f'{data_dir}/nofzs/dz/Nz_biases.txt')
        dz_cov = np.loadtxt(f'{data_dir}/nofzs/dz/Nz_covariance.txt')
        shift_dz_realised = np.random.multivariate_normal(mean=dz_biases, cov=dz_cov, size=1)[0]
        

    for i in range(nbins):
        hdu = np.loadtxt(f'{data_dir}/nofzs/nz/BLINDSHAPES_KIDS_Legacy_NS_shear_noSG_noWeiCut_newCut_blindABC_A1_rmcol_filt_lab_filt_lab_filt_PSF_RAD_calc_filt_ZB{str(ztomo_label[i][0]).split('.')[0]}p{str(ztomo_label[i][0]).split('.')[1]}t{str(ztomo_label[i][1]).split('.')[0]}p{str(ztomo_label[i][1]).split('.')[1]}_calib_goldwt_Nz.ascii').T
        z = hdu[0]
        if shift_nz:
            z_shifted = z - shift_dz_realised[i]
            n_z = np.interp(z, z_shifted, hdu[1])
            zmid = z[:-1] + 0.5*(z[1:] - z[:-1])
            dndz_interpolated = np.interp(los_z_integration, zmid, n_arcmin2[i]*n_z[:-1]/np.trapezoid((n_z[:-1]), zmid))
            tomo_nz[i] = np.clip(dndz_interpolated, 0, None)    
        else:
            zmid = z[:-1] + 0.5*(z[1:] - z[:-1])
            dndz_interpolated = np.interp(los_z_integration, zmid, n_arcmin2[i]*hdu[1][:-1]/np.trapezoid(hdu[1][:-1], zmid))
            tomo_nz[i] = np.clip(dndz_interpolated, 0, None)

    # Read in the KiDS-Legacy mask
    vis = mask = hp.read_map(f'{data_dir}/masks/KiDS_Legacy_N_healpix_1024_frac_withAstrom.fits') + hp.read_map(f'{data_dir}/masks/KiDS_Legacy_S_healpix_1024_frac_withAstrom.fits')

    #Multiplicative shear bias
    m_bias = np.array([-0.022869, -0.015966, -0.011331, 0.019870, 0.029506, 0.044535 ])
    m_bias_unc = np.array([0.005630, 0.005900, 0.007111, 0.006773, 0.007598, 0.008902 ])

    # Additive shear bias
    c_1_bias_north = np.array([3.372, 8.941, 4.523, 4.722, 6.658, 4.224])*1e-4
    c_1_bias_north_unc = np.array([1.528, 1.442, 1.747, 1.713, 1.887, 2.252])*1e-4
    c_2_bias_north = np.array([7.941, 8.852, 4.533, 5.368, 5.532, 10.26])*1e-4
    c_2_bias_north_unc = np.array([1.442, 1.642, 1.777, 1.665, 1.890, 2.400])*1e-4

    c_1_bias_south = np.array([-3.398, -9.536, -4.755, -4.532, -6.117, -3.717])*1e-4
    c_1_bias_south_unc = np.array([1.626, 1.519, 1.835, 1.653, 1.910, 2.151])*1e-4
    c_2_bias_south = np.array([-8.002, -6.026, -4.766, -5.152, -5.082, -9.027])*1e-4
    c_2_bias_south_unc = np.array([1.572, 1.590, 1.731, 1.594, 1.834, 2.282])*1e-4
    
    bias = 1
    
    # Realised shear bias --------------------------------------------------------------------

    m_bias_realised = np.array([float(np.random.normal(m_bias[i], m_bias_unc[i], 1)[0]) for i in range(len(m_bias))])
    c1_bias_north_realised = np.array([float(np.random.normal(c_1_bias_north[i], c_1_bias_north_unc[i], 1)[0]) for i in range(len(c_1_bias_north))])
    c2_bias_north_realised = np.array([float(np.random.normal(c_2_bias_north[i], c_2_bias_north_unc[i], 1)[0]) for i in range(len(c_2_bias_north))])
    c1_bias_south_realised = np.array([float(np.random.normal(c_1_bias_south[i], c_1_bias_south_unc[i], 1)[0]) for i in range(len(c_1_bias_south))])
    c2_bias_south_realised = np.array([float(np.random.normal(c_2_bias_south[i], c_2_bias_south_unc[i], 1)[0]) for i in range(len(c_2_bias_south))])

    # KiDS-Legacy variable depth properties --------------------------------------------------------------------
    # In KiDS-Legacy the galaxy shapes and their photometric redshifts are subject to the same depth 
    # # variations as they were measured from the same observations

    if do_variable_depth:
        print('Setting up variable depth properties...')
        # Define tracer of the variiable depth, i.e. TG weights
        vd_map, vd_z_map = np.empty((nbins, hp.nside2npix(nside))), np.empty((nbins, hp.nside2npix(nside)))

        for i in range(nbins):
            vd_map[i] = vd_z_map[i] = hp.read_map(f'{data_dir}/vd_maps/kids_legacy_ORweight_600_1024_ZB{str(ztomo_label[i][0]).split(".")[0]}p{str(ztomo_label[i][0]).split(".")[1]}t{str(ztomo_label[i][1]).split(".")[0]}p{str(ztomo_label[i][1]).split(".")[1]}_SNcoadd.fits')

        vd_trace_edges = np.array([0.00039586402840050813, 4.333629902071316, 5.41842919221884, 5.966776946566387, 6.363840923950448, 6.705373252332217, 7.027559883740965, 7.346859671979755, 7.67948179599049, 8.142430491846039, 11.324789884409146])

        n_vardepth_bins = len(vd_trace_edges) - 1

        vd_trace_effective_centre = np.array([[2.64789938, 4.95437723, 5.72446037, 6.18416233, 6.5389531 ,
                6.8664036 , 7.18752662, 7.49721572, 7.84279037, 8.41169263],
            [2.6010878 , 4.95463555, 5.72353875, 6.17492315, 6.5381264 ,
                6.8662059 , 7.18740554, 7.51358455, 7.9001472 , 8.58291616],
            [2.66567886, 4.9882028 , 5.71111977, 6.16986798, 6.53738992,
                6.86853353, 7.18556119, 7.50993119, 7.88656406, 8.38171309],
            [2.63998023, 4.97203209, 5.71254017, 6.17169432, 6.53822272,
                6.86750078, 7.19258744, 7.51312043, 7.90339967, 8.50940635],
            [2.75105258, 4.97090594, 5.70153218, 6.1690816 , 6.53424077,
                6.87010161, 7.18967823, 7.50957196, 7.89454552, 8.52311655],
            [2.86437597, 4.92491669, 5.6973029 , 6.17278079, 6.53739508,
                6.8680397 , 7.18974719, 7.51441527, 7.90650554, 8.99924988]])

        # Effective galaxy number density in each tomographic bin per pixel
        a_n_gal = np.array([-0.12025, 0.018645, -0.026478, 0.007749, 0.033379, 0.077332])
        b_n_gal = np.array([1.885729, -0.219395, 0.369554, -0.107936, -0.594714, -1.239654])
        c_n_gal = np.array([-9.122102, 0.754514, -1.57666, 0.465362, 3.191967, 6.103007])
        d_n_gal = np.array([14.971611, 0.63512, 3.622171, 0.797094, -3.461152, -7.923277])

        n_contrast_vd_func = [lambda x: (a_n_gal[i]*x**3 + b_n_gal[i]*x**2 + c_n_gal[i]*x + d_n_gal[i])/n_arcmin2[i] for i in range(nbins)]

        # Redshift distribution in each tomographic bin per vd_trace bin
        dndz_vd = np.zeros((nbins, n_vardepth_bins, n_los_chi))

        for i in range(nbins):
            for j in range(n_vardepth_bins):
                hdu = np.loadtxt(f'{data_dir}/nofzs/nz_tgweights/BLINDSHAPES_KIDS_Legacy_NS_shear_noSG_noWeiCut_newCut_blindABC_A1_rmcol_filt_PSF_RAD_calc_filt_filt_comb_1_ZB{str(ztomo_label[i][0]).split(".")[0]}p{str(ztomo_label[i][0]).split(".")[1]}t{str(ztomo_label[i][1]).split(".")[0]}p{str(ztomo_label[i][1]).split(".")[1]}_calib_goldwt_Nz_recalibrated.ascii').T
                z = hdu[0]
                zmid = z[:-1] + 0.5*(z[1:] - z[:-1])
                dndz_interpolated = np.interp(z_grid, zmid, n_contrast_vd_func[i](vd_trace_effective_centre[i][j])*hdu[1][:-1]/np.trapezoid(hdu[1][:-1], zmid))
                dndz_vd[i][j] = np.clip(dndz_interpolated, 0, None)

        # Intrinsic shape dispersion in each tomographic bin per pixel
        a_sigma_eps = np.array([0.000403, 0.000158, -1.9e-05, 0.000275, -0.000428, -5.8e-05])
        b_sigma_eps = np.array([-0.006561, -0.002528, 0.00025, -0.004668, 0.006636, 0.001098])
        c_sigma_eps = np.array([0.032342, 0.012569, 0.000116, 0.024811, -0.032343, -0.007647])
        d_sigma_eps = np.array([0.232259, 0.252409, 0.284059, 0.221895, 0.329548, 0.319706])

        sigma_eps_var = np.array([lambda x: (a_sigma_eps[i]*x**3 + b_sigma_eps[i]*x**2 + c_sigma_eps[i]*x + d_sigma_eps[i]) for i in range(nbins)])

        # Multiplicative shear bias in each tomographic bin and each vd_trace bin

        m_bias_vd = np.array([[-0.033754, -0.026962, -0.034473, -0.019362, -0.015617, -0.019519,
                -0.012956, -0.027953, -0.022462, -0.040446],
            [-0.020141, -0.016864, -0.026674, -0.025799, -0.025235, -0.0029  ,
                -0.011385, -0.012112, -0.016647, -0.015067],
            [-0.002508, -0.007223, -0.01488 , -0.008141, -0.01835 , -0.010122,
                -0.024149, -0.002857, -0.007611, -0.013968],
            [ 0.023157,  0.000612,  0.025447,  0.013191,  0.01541 ,  0.021786,
                0.023982,  0.015922,  0.029325,  0.022082],
            [ 0.02566 ,  0.03476 ,  0.031331,  0.029178,  0.028164,  0.029091,
                0.016602,  0.021035,  0.026226,  0.05884 ],
            [ 0.032336,  0.052092,  0.054903,  0.032802,  0.063345,  0.036114,
                0.027017,  0.060289,  0.042254,  0.041393]])

        m_bias_vd_unc = np.array([[0.019524, 0.021472, 0.019926, 0.017265, 0.016387, 0.016348,
                0.016989, 0.018111, 0.028343, 0.050147],
            [0.02068 , 0.023506, 0.021966, 0.020398, 0.019943, 0.019986,
                0.020069, 0.019282, 0.01693 , 0.015682],
            [0.023006, 0.022585, 0.020483, 0.021865, 0.022761, 0.022631,
                0.022427, 0.02281 , 0.023472, 0.036736],
            [0.021581, 0.021588, 0.020667, 0.021542, 0.022082, 0.022151,
                0.021611, 0.020624, 0.01909 , 0.022047],
            [0.022363, 0.019711, 0.021506, 0.023631, 0.024188, 0.024585,
                0.024031, 0.024637, 0.023461, 0.027449],
            [0.025277, 0.023789, 0.028864, 0.031624, 0.033004, 0.032607,
                0.03178 , 0.030784, 0.026866, 0.017471]])

        m_bias_vd_realised = np.array([[float(np.random.normal(m_bias_vd[i][j], m_bias_vd_unc[i][j], 1)) for j in range(n_vardepth_bins)] for i in range(nbins)])

        # PSF bias factors in each tomographic bin
        alpha_1 = np.array([-0.003, -0.007, -0.003, -0.004, -0.005, -0.004])
        alpha_1_error = np.array([0.015, 0.013, 0.013, 0.013, 0.013, 0.014])

        alpha_2 = np.array([0.013, 0.008, 0.007, 0.008, 0.007, 0.015])
        alpha_2_error = np.array([0.011, 0.010, 0.011, 0.011, 0.011, 0.011])

        alpha_1_realised = np.array([float(np.random.normal(alpha_1[i], alpha_1_error[i], 1)) for i in range(len(alpha_1))])
        alpha_2_realised = np.array([float(np.random.normal(alpha_2[i], alpha_2_error[i], 1)) for i in range(len(alpha_2))])

        psf_bias_map_1 = hp.read_map(f'{data_dir}/psfmaps/psf_e1_Map_KiDS-Legacy_1024_ALL.fits')

        psf_bias_map_2 = hp.read_map(f'{data_dir}/psfmaps/psf_e2_Map_KiDS-Legacy_1024_ALL.fits')
    
    # Calculate the variable depth mask for GLASS ---------------------------------------------------
    if do_variable_depth:   
        var_depth_mask = glass.observations.angular_los_variable_depth_mask(vd_map,         # variable depth map in the ANGULAR direction for all tomographic bins
                                                                n_bins = nbins,        # number of tomographic bins
                                                                zbins = zbins,         # redshift shell bin edges
                                                                ztomo = ztomo,         # redshift bin edges for each tomographic bin
                                                                dndz = tomo_nz,        # dndz for each tomographic bin
                                                                z = z_grid,            # redshift grid
                                                                dndz_vardepth=dndz_vd, # redshift distribution per tomographic bin per variable depth bin
                                                                vardepth_values=vd_trace_effective_centre,     # variable depth grid
                                                                vardepth_los_tracer=vd_z_map,              # variable depth map in the REDSHIFT direction for all tomographic bins
                                                                vardepth_tomo_functions = n_contrast_vd_func) 
    
    # Make the catalogue ---------------------------------------------------
    print('Simulating the galaxy catalogue...')
    s_catalogue = time.time()
    
    # we will store the catalogue as a structured numpy array, initially empty
    catalogue = np.empty(0, dtype=[('RA', float), ('DEC', float),
                                ('Z_TRUE', float), ('ZBIN', int),
                                ('E1', float), ('E2', float)])

    matter_fields = np.zeros((n_glass_shells, 12*nside**2), dtype=float)
    kappa_fields = np.zeros((n_glass_shells, 12*nside**2), dtype=float)

    # simulate the matter fields in the main loop, and build up the catalogue
    for i, delta_i in enumerate(matter):
        # compute the lensing maps for this shell
        convergence.add_window(delta_i, ws[i])
        kappa_i = convergence.kappa
        
        matter_fields[i] = delta_i
        kappa_fields[i] = kappa_i

        for tomo in range(nbins):
            # the true galaxy distribution in this shell
            z_i, dndz_i = glass.shells.restrict(los_z_integration, tomo_nz[tomo], ws[i])
            
            # integrate dndz to get the total galaxy density in this shell
            ngal = np.trapezoid(dndz_i, z_i)

            z_eff = np.average(los_z_integration, weights=tomo_nz[tomo])

            kappa_i_ia = kappa_ia_nla_m(delta_i, z_eff, f_red[tomo], cosmo, a_ia, b_ia, log10_M_eff[tomo])
            kappa_i += kappa_i_ia

            gamm1_i, gamm2_i = glass.lensing.shear_from_convergence(kappa_i)

            effective_mask = mask
            # generate galaxy positions from the matter density contrast
            for gal_lon, gal_lat, gal_count in glass.points.positions_from_delta(ngal, delta_i, bias, effective_mask, rng=rng):
                # generate random redshifts from the provided nz
                gal_z = glass.galaxies.redshifts_from_nz(gal_count, z_i, dndz_i, rng=rng, warn=False)

                # generate galaxy ellipticities from the chosen distribution
                gal_eps = glass.shapes.ellipticity_intnorm(gal_count, sigma_e[tomo], rng=rng)

                # apply the shear fields to the ellipticities
                gal_she = glass.galaxies.galaxy_shear(gal_lon, gal_lat, gal_eps,
                                                    kappa_i, gamm1_i, gamm2_i)

                # make a catalogue for the new rows
                rows = np.empty(gal_count, dtype=catalogue.dtype)
                rows['RA'] = gal_lon
                rows['DEC'] = gal_lat
                rows['Z_TRUE'] = gal_z
                rows['ZBIN'] = tomo

                is_north = np.absolute(gal_lat) < 15
                is_south = np.absolute(gal_lat) >= 15

                c_bias = np.zeros_like(gal_she)
                c_bias[is_north] = c1_bias_north_realised[tomo] + 1j*c2_bias_north_realised[tomo]
                c_bias[is_south] = c1_bias_south_realised[tomo] + 1j*c2_bias_south_realised[tomo]

                # apply the shear bias

                rows['E1'] = (1+m_bias_realised[tomo])*gal_she.real + c_bias.real
                rows['E2'] = (1+m_bias_realised[tomo])*gal_she.imag + c_bias.imag

                # add the new rows to the catalogue
                catalogue = np.append(catalogue, rows)

    print(f'Total number of galaxies sampled: {len(catalogue):,}')
    print(f'Simulated the galaxy catalogue in {time.time() - s_catalogue:.2f} seconds')
    
    # Get the shear ---------------------------------------------------
    print('Calculating the shear power spectra...')
    alm, alm_rand = make_alm(catalogue, nbins, nside, m_bias, lmax)
    no_mask_shear_cls_noiseless = denoise_shear_cls(alm, alm_rand, nbins, lmax)
    
    shear = np.zeros((nbins, hp.nside2npix(nside)), dtype=complex)
    counts = np.zeros_like(shear, dtype=int)

    alm, alm_rand = [], []

    for i in range(nbins):
        in_bin = (catalogue['ZBIN'] == i)

        she = (1/1/(1+m_bias[i])) * ((catalogue['E1'][in_bin] - np.mean(catalogue['E1'][in_bin])) + 1j*(catalogue['E2'][in_bin] - np.mean(catalogue['E2'][in_bin])))

        map_shears(shear[i], counts[i], catalogue['RA'][in_bin],  catalogue['DEC'][in_bin], she, gal_wht=None)

        shear[i][counts[i] > 0] = np.divide(shear[i][counts[i] > 0], counts[i][counts[i] > 0])

        gal_num = len(catalogue[in_bin])
        rand_theta = 2*np.pi*np.random.random_sample(gal_num)

        e1_corr = she.real*np.cos(rand_theta) - she.imag*np.sin(rand_theta)
        e2_corr = she.imag*np.cos(rand_theta) + she.real*np.sin(rand_theta)

        rand = np.zeros(hp.nside2npix(nside), dtype=complex)
        _ = np.zeros_like(rand, dtype=int)

        map_shears(rand, _, catalogue['RA'][in_bin],  catalogue['DEC'][in_bin],  e1_corr + 1j * e2_corr, gal_wht=None)

        rand[_ > 0] = np.divide(rand[_ > 0], _[_ > 0])

        alm.append(hp.sphtfunc.map2alm_spin([shear[i].real, shear[i].imag], spin = 2, lmax = lmax)) #Compute the alms for the shear field
        alm_rand.append(hp.sphtfunc.map2alm_spin([rand.real, rand.imag], spin = 2, lmax = lmax)) #Compute the alms for the random shear fielda

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
    print('Calculated the shear power spectra')
    
    # Calculate the mask from the realisation and then de-mix ----------------------------------------------------
    print('Calculating the mask from the realisation...')
    bin_masks = cat2mask(catalogue, nbins, nside)
    bin_mask_cls = maskcls(bin_masks, lmax=int(1.5*lmax), nbins=nbins)
    print('De-mixing the shear power spectra...')
    realised_unmixed_shear_cls = unmix_shear_cl(num_bins = 6, shear_cls = shear_cls_noiseless, mask_cl = bin_mask_cls, lmin = lmin, lmax = lmax)
    
    np.save(f'{output_dir}/{sim_batch_name}/uncut_realised_unmixed_shear_cls_{sim_num}.npy', realised_unmixed_shear_cls)
    
    # Scale cuts
    lower_lscale = 76
    upper_lscale = 1500
    realised_unmixed_shear_cls_cut = realised_unmixed_shear_cls[:, :, :, lower_lscale:upper_lscale+1]
    np.save(f'{output_dir}/{sim_batch_name}/cut_realised_unmixed_shear_cls_{sim_num}.npy', realised_unmixed_shear_cls_cut)
    print(f'Saved shear_cls for sim {sim_num}')
       
    # Calculate the bandpowers ----------------------------------------------------
    print('Calculating the bandpowers...')
    # Use 8 bins for the bandpowers, with low ell of 76 - can adjust this
    # Do one with low ell of 100
    # Can also try 10 bins

    theory_bandpowers_stacked = np.array([])

    for i in range(nbins):
        for j in range(0, nbins):
            if i<j:
                pass
            else:
                bandpowers, centre_ell, band_cutoffs = make_bandpowers(lmin=lower_lscale, lmax=upper_lscale, cls=realised_unmixed_shear_cls_cut[i][j][0], nbands=8)
                theory_bandpowers_stacked = np.append(theory_bandpowers_stacked, bandpowers)

    theory_bandpowers = theory_bandpowers_stacked.reshape(int(nbins*(nbins+1)/2), -1)
    
    np.save(f'{output_dir}/{sim_batch_name}/bandpower_centre_ell_{sim_num}.npy', centre_ell)
    np.save(f'{output_dir}/{sim_batch_name}/bandpowers_{sim_num}.npy', theory_bandpowers)
    np.save(f'{output_dir}/{sim_batch_name}/bandpowers_stacked_{sim_num}.npy', theory_bandpowers_stacked)
    print(f'Saved bandpowers for sim {sim_num}')
    
    print(f'Entire simulation took {time.time() - s:.2f} seconds')
    