
# %%
import numpy as np
import healpy as hp

import h5py
import numpy as np
from pathlib import Path
from collections import deque
import gc

import camb
import glass.ext.camb
from cosmology import Cosmology

# use the CAMB cosmology that generated the matter power spectra
import glass
import glass.shells

import time
import os
from src.cosmology import nla, glass_utils, levin, parameters, priors
from src.cosmology.map_shears import map_shears
from src.cosmology.simulators import GowerStreetSimulator, GlassLogNormalSimulator
from src.cosmology.systematics import NLASystematics
from src.cosmology.gower_street import GowerStCosmologies, GowerStDatasetBuilder, GowerStPrior

from src.cosmology.manip_cls import denoise_shear_cls, unmix_shear_cl, cat2mask, maskcls, compute_cl_bandpowers, cat2mask_fractional_2
from src.cosmology.pixelise_maps import get_patch_values

from src.cosmology.map_shears  import make_alm_shear_convergence, filter_EB_alms_and_make_maps
from src.cosmology.mpi import compute_levin_glass_in_child_npz_subproc,load_levin_child_pickle

from src.KiDS.tomo import calculate_tomo_nz
from src.KiDS.rotations import KiDS_PATCH_GOWER_ROTATIONS, KiDS_PATCH_GLASS_ROTATIONS
from src.cosmology.camb_matter_power import get_camb_matter_cls

import matplotlib
import matplotlib.pyplot as plt
# PLOTTING_DIR = Path('./theory_tests_fullsky_gower_ngaltrapz')
PLOTTING_DIR = Path('./theory_tests_kids_camb_no_systematics_heracles')

PLOTTING_DIR.mkdir(parents=True, exist_ok=True)

plt.rc('text', usetex=False)
plt.rc('font', family='sans-serif')
plt.rc('image', interpolation='none')

for theory_test_index in range(10):
    # sim_num = theory_test_index*10 + 192
    sim_num = theory_test_index*10 + 200
    # %%
    data_dir = '/home/asaoulis/projects/glass_transfer/kids-legacy-sbi/data'
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


    # Instrinsic galaxy shape dispersion per tomographic bin
    sigma_e = np.array([0.2772, 0.2716, 0.2899, 0.2619, 0.2802, 0.3002])

    levin_params = dict(
        kmax=500.0,
        kmin=1e10,
        nmin=50,
        npoint=3,
        nmax=200
    )

    zmin, zmax = 0.0, 2.0
    dx = 200.0  # Mpc/h
    n_los_chi = 1000  # define the integration limits here

    # Tomographic redshift bins
    nbins = 6

    bias = 1
    SIMULATOR_TYPE = 'glass'  # 'glass' or 'gower_street'
    nside = 1024
    n_ell = 20
    # lmax = 300
    lmax = 2*nside
    lmin = 0

    NUM_JOBS = 12500
    inner_num_shape_noise_realisations=1
    outer_num_shape_noise_realisations=1
    lower_lscale = 50
    upper_lscale = 1500
    nbands = 20
    named_patches = {
        "south":(12, -31, 90, 11),     # (lon_center, lat_center, lon_range, lat_range)
        "north":(-178, 0, 112, 10)
    }
    patches = list(named_patches.values())
    # rotation_values = [rot for rot in rotation_angles for _ in range(num_shape_noise_realisations)]
    csv_path = '/home/asaoulis/projects/glass_transfer/kids-legacy-sbi/data/gower_st/PKDGRAV3_on_DiRAC_DES_330.csv'


    # %%
    KIDS_SYSTEMATICS = False
    shift_nz = KIDS_SYSTEMATICS # Whether to shift the n(z) using the dz biases and covariance

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

    # Build Gower Street prior for cosmological parameters (before MPI distribution)
    gower_prior = GowerStPrior.from_csv(csv_path)
    NO_ROTATIONS = False
    if NO_ROTATIONS:
        rotation_specs = {
            "rotations": [{"rot": 0, "flip": False, "backend": "pixel"}]
        }
    else:
        if SIMULATOR_TYPE == 'glass':
            rotation_specs = KiDS_PATCH_GLASS_ROTATIONS
        else:
            rotation_specs = KiDS_PATCH_GOWER_ROTATIONS
        # %%
    for rot_idx, rotation_spec in enumerate(rotation_specs[:6]):

        rng = np.random.default_rng()
        # np.random.seed(100)
        omega_k = 0.0

        # intrinsic alignments params
        massdep_means = np.loadtxt(f'{data_dir}/priors/massdep_means.txt')
        massdep_cov = np.loadtxt(f'{data_dir}/priors/massdep_cov.txt')
        log10_M_eff_means = massdep_means[2:]
        log10_M_eff_cov = massdep_cov[2:,2:] 
        f_red = np.array([0.15, 0.2, 0.17, 0.24, 0.19, 0.03])
        log10_M_eff = np.random.multivariate_normal(log10_M_eff_means, log10_M_eff_cov, size=1)[0]

        logT_AGN_realised = np.random.uniform(*prior['logT_AGN'])
        a_ia_realised     = np.random.uniform(*prior['a_ia'])
        b_ia_realised     = np.random.uniform(*prior['b_ia'])
        nuisance_params = {"a_ia": a_ia_realised, "b_ia": b_ia_realised}
        # intrinsic alignments params
        ia_params = dict(
            a_ia = a_ia_realised,
            b_ia = b_ia_realised,
            f_red = np.array([0.15, 0.2, 0.17, 0.24, 0.19, 0.03]),
            log10_M_eff = log10_M_eff,
        )

        extra_params = {
            "logT_AGN": float(logT_AGN_realised),
            "a_ia": float(a_ia_realised),
            "b_ia": float(b_ia_realised),
        }

        if SIMULATOR_TYPE == "glass":

            sampled_cosmo_params = gower_prior.draw_param_dict_sample(rng=rng)
            param_dict = {
                **sampled_cosmo_params,
                **extra_params,
            }
            # Build cosmology and CAMB parameters
            cosmo, pars = parameters.build_cosmology(param_dict)
            shells, glass_cls = get_camb_matter_cls(pars, lmax, zmin, zmax, dx)
            glass_cls_discretized = glass.discretized_cls(glass_cls, nside=nside, lmax=lmax, ncorr=1)
            fields = glass.lognormal_fields(shells)
            gls = glass.solve_gaussian_spectra(fields, glass_cls_discretized)
            matter = glass.generate(fields, gls, nside, ncorr=1, rng=rng)
            cosmo = Cosmology.from_camb(pars)
            print("RUNNING GLASS TESTS")
        else:
            gower_data_dir = '/share/gpu5/asaoulis/gowerstreet'
            gower_street_loader = GowerStCosmologies(gower_data_dir, csv_path)
            # simulator = gower_street_loader.setup_simulator(sim_num, **kwargs)
            # cosmo, pars, param_dict  = gower_street_loader.get_simulation_cosmology(sim_num, nuisance_params)
            param_dict = gower_street_loader.get_params_from_sim_id(sim_num, extra_params=extra_params)
            shells, matter, cosmo = gower_street_loader.load_shells_matter_and_cosmology(sim_num, nside=nside)
            _, pars, _  = gower_street_loader.get_simulation_cosmology(sim_num, nuisance_params)
            cosmo = Cosmology.from_camb(pars)

        USE_KIDS_MASK = True
        if USE_KIDS_MASK:
            mask = hp.read_map(f'{data_dir}/masks/KiDS_Legacy_N_healpix_1024_frac_withAstrom.fits') + hp.read_map(f'{data_dir}/masks/KiDS_Legacy_S_healpix_1024_frac_withAstrom.fits')
        else:
            mask = np.ones(hp.nside2npix(nside))

        
        # vis = mask = full_sky_mask
        #pretty print param_dict``
        # cosmo, pars, param_dict  = gower_street_loader.get_simulation_cosmology(sim_num, nuisance_params)
        print("\n".join([f"{key}: {value}" for key, value in param_dict.items()]))

        # %%
        if USE_KIDS_MASK:
            kids_mms = np.load("/share/gpu5/asaoulis/KiDS_Legacy_mixing_matrix_mask.npy")
            n = kids_mms.shape[0] // 3   # = 2049
            assert n == lmax + 1         # safety check
        else:
            # construct identity matrix of correct size (EE, EB, BB blocks)
            kids_mms = np.eye(3*(lmax+1))

        zb = glass.shells.distance_grid(cosmo, zmin, zmax, dx=dx)

        los_z_integration = np.linspace(zb[0], zb[-1], n_los_chi)
        tomo_nz = calculate_tomo_nz(data_dir, n_los_chi, los_z_integration, shift_nz)

        # %%
        m_bias_realised = np.array([float(np.random.normal(m_bias[i], m_bias_unc[i], 1)) for i in range(len(m_bias))])
        c1_bias_north_realised = np.array([float(np.random.normal(c_1_bias_north[i], c_1_bias_north_unc[i], 1)) for i in range(len(c_1_bias_north))])
        c2_bias_north_realised = np.array([float(np.random.normal(c_2_bias_north[i], c_2_bias_north_unc[i], 1)) for i in range(len(c_2_bias_north))])
        c1_bias_south_realised = np.array([float(np.random.normal(c_1_bias_south[i], c_1_bias_south_unc[i], 1)) for i in range(len(c_1_bias_south))])
        c2_bias_south_realised = np.array([float(np.random.normal(c_2_bias_south[i], c_2_bias_south_unc[i], 1)) for i in range(len(c_2_bias_south))])

        # %%
        if KIDS_SYSTEMATICS:
            systematics = NLASystematics(
                shear_bias={
                    'm_bias': m_bias_realised,
                    'c1_north': c1_bias_north_realised,
                    'c2_north': c2_bias_north_realised,
                    'c1_south': c1_bias_south_realised,
                    'c2_south': c2_bias_south_realised
                },
                nla=ia_params,
                cosmo=cosmo
            )
        else:
            systematics = None
            sigma_e *= 0.0
            m_bias_realised *= 0.0
        num_shape_noise_realisations=1
        kwargs = {
            'cosmo': cosmo,
            'los_z_integration': los_z_integration,
            'tomo_nz': tomo_nz,
            'galaxy_bias': 1.0,
            'sigma_e': sigma_e,
            'mask': mask,
            'nside': nside,
            'nbins': nbins,
            'rng': rng,
            'systematics': systematics,
        }
        # if SIMULATOR_TYPE == 'glass':

        #     simulator = GlassLogNormalSimulator(matter, ws, **kwargs)
        # elif SIMULATOR_TYPE == 'gower_street':
        #     ### OLD
        #     # builder = GowerStDatasetBuilder(csv_path=csv_path, dataset_path=gower_data_dir)
        #     # simulator = builder.setup_simulator(sim_num, **kwargs)
        #     # NEW
            #  simulator = gower_street_loader.setup_simulator(sim_num, **kwargs)
        simulator = GlassLogNormalSimulator(matter, shells, **kwargs)
        catalogues = simulator.run([rotation_spec], num_shape_noise_realisations=inner_num_shape_noise_realisations)

        # %%
        catalogue = np.concatenate(catalogues[0])

        save_sim_num_string = f"{sim_num}_rot{rot_idx}"

        # %%
        from src.cosmology.map_shears  import *

        from src.cosmology.manip_cls import denoise_shear_cls, unmix_shear_cl, cat2mask, maskcls, compute_cl_bandpowers, process_cls, unmixing_mask_cls
        from src.cosmology.pixelise_maps import get_patch_values

        from src.cosmology.map_shears  import make_alm_shear_convergence, filter_EB_alms_and_make_maps

            
        catalogue = np.concatenate(catalogues[0])

        lower_lscale = 50
        upper_lscale = 1500
        nbands = 8
        alm, alm_rand = make_alm_shear_convergence(
            catalogue, m_bias_realised, nbins, nside, lmax, mask=mask, nosh=False
        )
        mask_cls = unmixing_mask_cls(catalogue, nbins, nside, lmax, lmin)

        realised_unmixed_shear_cls, cll_bands, bandpowers = process_cls(mask_cls, nbins, nside, alm, alm_rand, lower_lscale, upper_lscale,lmin, lmax, nbands, )


        nbands = 8
        lmin_cut = 50
        lmax_cut = 1500
        # lmax_cut = 750

        # get the theory cls from CAMB
        pars.NonLinear = "NonLinear_lens"
        pars.Want_CMB = False
        pars.min_l = 2
        pars.set_for_lmax(lmax)
        pars.SourceWindows = [
            camb.sources.SplinedSourceWindow(z=los_z_integration, W=tomo_nz[i], source_type="lensing") for i in range(len(tomo_nz))

        ]
        results = camb.get_results(pars)
        theory_cls = results.get_source_cls_dict(lmax=lmax, raw_cl=True)
        mixed_cls = denoise_shear_cls(nbins, alm, alm_rand, lmax)

        # --- 1) Cut empirical pseudo Cls (already mixed) ---
        # mixed_cls: shape (i,j,2,ell)
        mixed_cut = mixed_cls[:, :, :, lmin_cut:lmax_cut+1]

        # --- 2) Compute bandpowers of empirical E ---
        cll_bands, bandpowers = compute_cl_bandpowers(
            mixed_cut, nbins, lmin_cut, lmax_cut, nbands
        )

        # --- 3) Compute bandpowers of empirical B ---
        b_mode_cls_cut = mixed_cut.copy()
        b_mode_cls_cut[:, :, 0, :] = b_mode_cls_cut[:, :, 1, :]  # put B into slot 0
        _, b_bandpowers = compute_cl_bandpowers(
            b_mode_cls_cut, nbins, lmin_cut, lmax_cut, nbands
        )

        # --- 4) Compute *mixed theory* using kids_mms, then cut ---
        cut_theory_cls = {}
        _, pw = hp.pixwin(nside, lmax=lmax, pol=True)
        for i in range(nbins):
            for j in range(nbins):
                if i < j:
                    continue

                # Build full theory vector
                Cl_EE = theory_cls[f"W{i+1}xW{j+1}"]
                Cl_EB = np.zeros_like(Cl_EE)
                Cl_BB = np.zeros_like(Cl_EE)

                T = np.zeros(3*(lmax+1))
                T[0*(lmax+1):1*(lmax+1)] = Cl_EE
                T[1*(lmax+1):2*(lmax+1)] = Cl_EB
                T[2*(lmax+1):3*(lmax+1)] = Cl_BB
                # kids_mms[:lmax+1, :lmax+1] = mms[i]


                # Apply global mixing
                pseudo = kids_mms @ T

                # Extract EE + pixel window
                pseudo_EE = pw**2 * pseudo[0*(lmax+1):1*(lmax+1)]

                # Apply same ℓ cut as empirical
                cut_theory_cls[f"S{i+1}-S{j+1}"] = pseudo_EE[lmin_cut:lmax_cut+1]

        # --- 5) Bandpower theory on cut ---
        bandpower_theory = {}

        for i in range(nbins):
            for j in range(nbins):
                if i < j:
                    continue

                lbl = f"S{i+1}-S{j+1}"
                cls_cut = cut_theory_cls[lbl]  # shape (Lcut,)

                # reshape to (1,1,1,L) to reuse same function
                _, bp_th = compute_cl_bandpowers(
                    cls_cut[np.newaxis, np.newaxis, np.newaxis, :],
                    1, lmin_cut, lmax_cut, nbands
                )
                bandpower_theory[lbl] = bp_th[0, :]


        fig, ax = plt.subplots(nbins, nbins, figsize=(15,15))

        for i in range(nbins):
            for j in range(nbins):
                if i < j:
                    ax[i, j].axis('off')
                    continue

                lbl = f"S{i+1}-S{j+1}"

                # Empirical E bandpowers
                ax[i, j].plot(cll_bands, bandpowers[int(i*(i+1)/2 + j)], label='Emp E')

                # Theory bandpowers
                ax[i, j].plot(cll_bands, bandpower_theory[lbl], c='black', label='Theory')

                # Empirical B bandpowers
                ax[i, j].plot(cll_bands, b_bandpowers[int(i*(i+1)/2 + j)],
                            c='red', ls='dashed', label='Emp B')

                # Scales / labels
                ax[i, j].set_xscale('log')
                ax[i, j].set_yscale('log')
                ax[i, j].text(cll_bands[1], 0.5*(ax[i, j].get_ylim()[1]), lbl)

                if i == nbins - 1:
                    ax[i, j].set_xlabel(r'$\ell$', fontsize=12)
                else:
                    plt.setp(ax[i, j].get_xticklabels(), visible=False)
                
                if j == 0:
                    ax[i, j].set_ylabel(r'$C_{\ell}$', fontsize=12)
                else:
                    plt.setp(ax[i, j].get_yticklabels(), visible=False)

        plt.tight_layout()
        plt.show()

        out_dir = PLOTTING_DIR / 'mixed_bandpowers'
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f'mixed_sim{save_sim_num_string}.png', dpi=300)
        plt.close()
        fig, ax = plt.subplots(nbins, nbins, figsize=(15,15))

        for i in range(nbins):
            for j in range(nbins):
                if i < j:
                    ax[i, j].axis('off')
                    continue

                lbl = f"S{i+1}-S{j+1}"

                # Empirical E bandpowers
                # ax[i, j].plot(cll_bands, bandpowers[int(i*(i+1)/2 + j)], label='Emp E')

                # Theory bandpowers
                ax[i, j].plot(cll_bands,  bandpowers[int(i*(i+1)/2 + j)]/ bandpower_theory[lbl], c='blue', label='Ratio')
                ax[i, j].plot(cll_bands,  np.ones_like(cll_bands), c='black', linestyle='-', label='Ratio')


                # Empirical B bandpowers
                # ax[i, j].plot(cll_bands, b_bandpowers[int(i*(i+1)/2 + j)],
                #               c='red', ls='dashed', label='Emp B')

                # Scales / labels
                # ax[i, j].set_xscale('log')
                # ax[i, j].set_yscale('symlog')
                ax[i, j].text(cll_bands[1], 0.5*(ax[i, j].get_ylim()[1]), lbl)

                if i == nbins - 1:
                    ax[i, j].set_xlabel(r'$\ell$', fontsize=12)
                else:
                    plt.setp(ax[i, j].get_xticklabels(), visible=False)
                
                if j == 0:
                    ax[i, j].set_ylabel(r'$C_{\ell}$', fontsize=12)
                else:
                    plt.setp(ax[i, j].get_yticklabels(), visible=False)

        plt.tight_layout()
        plt.savefig(out_dir / f'mixed_sim{save_sim_num_string}_bandpowers_ratios.png', dpi=300)
        plt.close()


        nbands = 8
        lmin_cut = 50
        lmax_cut = 1500
        # lmax_cut = 750

        # get the theory cls from CAMB
        pars.NonLinear = "NonLinear_lens"
        pars.Want_CMB = False
        pars.min_l = 2
        pars.set_for_lmax(lmax)
        pars.SourceWindows = [
            camb.sources.SplinedSourceWindow(z=los_z_integration, W=tomo_nz[i], source_type="lensing") for i in range(len(tomo_nz))

        ]
        results = camb.get_results(pars)
        theory_cls = results.get_source_cls_dict(lmax=lmax, raw_cl=True)
        mixed_cls = realised_unmixed_shear_cls

        # --- 1) Cut empirical pseudo Cls (already mixed) ---
        # mixed_cls: shape (i,j,2,ell)
        mixed_cut = mixed_cls[:, :, :, lmin_cut:lmax_cut+1]

        # --- 2) Compute bandpowers of empirical E ---
        cll_bands, bandpowers = compute_cl_bandpowers(
            mixed_cut, nbins, lmin_cut, lmax_cut, nbands
        )

        # --- 3) Compute bandpowers of empirical B ---
        b_mode_cls_cut = mixed_cut.copy()
        b_mode_cls_cut[:, :, 0, :] = b_mode_cls_cut[:, :, 1, :]  # put B into slot 0
        _, b_bandpowers = compute_cl_bandpowers(
            b_mode_cls_cut, nbins, lmin_cut, lmax_cut, nbands
        )

        # --- 4) Compute *mixed theory* using kids_mms, then cut ---
        cut_theory_cls = {}
        _, pw = hp.pixwin(nside, lmax=lmax, pol=True)
        for i in range(nbins):
            for j in range(nbins):
                if i < j:
                    continue

                # Build full theory vector
                Cl_EE = theory_cls[f"W{i+1}xW{j+1}"]
                Cl_EB = np.zeros_like(Cl_EE)
                Cl_BB = np.zeros_like(Cl_EE)

                T = np.zeros(3*(lmax+1))
                T[0*(lmax+1):1*(lmax+1)] = Cl_EE
                T[1*(lmax+1):2*(lmax+1)] = Cl_EB
                T[2*(lmax+1):3*(lmax+1)] = Cl_BB
                # kids_mms[:lmax+1, :lmax+1] = mms[i]


                # Apply global mixing
                pseudo = np.eye(3*(lmax+1)) @ T

                # Extract EE + pixel window
                pseudo_EE = pw**2 * pseudo[0*(lmax+1):1*(lmax+1)]

                # Apply same ℓ cut as empirical
                cut_theory_cls[f"S{i+1}-S{j+1}"] = pseudo_EE[lmin_cut:lmax_cut+1]

        # --- 5) Bandpower theory on cut ---
        bandpower_theory = {}

        for i in range(nbins):
            for j in range(nbins):
                if i < j:
                    continue

                lbl = f"S{i+1}-S{j+1}"
                cls_cut = cut_theory_cls[lbl]  # shape (Lcut,)

                # reshape to (1,1,1,L) to reuse same function
                _, bp_th = compute_cl_bandpowers(
                    cls_cut[np.newaxis, np.newaxis, np.newaxis, :],
                    1, lmin_cut, lmax_cut, nbands
                )
                bandpower_theory[lbl] = bp_th[0, :]


        fig, ax = plt.subplots(nbins, nbins, figsize=(15,15))

        for i in range(nbins):
            for j in range(nbins):
                if i < j:
                    ax[i, j].axis('off')
                    continue

                lbl = f"S{i+1}-S{j+1}"

                # Empirical E bandpowers
                ax[i, j].plot(cll_bands, bandpowers[int(i*(i+1)/2 + j)], label='Emp E')

                # Theory bandpowers
                ax[i, j].plot(cll_bands, bandpower_theory[lbl], c='black', label='Theory')

                # Empirical B bandpowers
                ax[i, j].plot(cll_bands, b_bandpowers[int(i*(i+1)/2 + j)],
                            c='red', ls='dashed', label='Emp B')

                # Scales / labels
                ax[i, j].set_xscale('log')
                ax[i, j].set_yscale('log')
                ax[i, j].text(cll_bands[1], 0.5*(ax[i, j].get_ylim()[1]), lbl)

                if i == nbins - 1:
                    ax[i, j].set_xlabel(r'$\ell$', fontsize=12)
                else:
                    plt.setp(ax[i, j].get_xticklabels(), visible=False)
                
                if j == 0:
                    ax[i, j].set_ylabel(r'$C_{\ell}$', fontsize=12)
                else:
                    plt.setp(ax[i, j].get_yticklabels(), visible=False)

        plt.tight_layout()
        out_dir = PLOTTING_DIR / 'unmixed_bandpowers'
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f'unmixed_sim{save_sim_num_string}.png', dpi=300)
        plt.close()


        fig, ax = plt.subplots(nbins, nbins, figsize=(15,15))

        for i in range(nbins):
            for j in range(nbins):
                if i < j:
                    ax[i, j].axis('off')
                    continue

                lbl = f"S{i+1}-S{j+1}"

                # Empirical E bandpowers
                # ax[i, j].plot(cll_bands, bandpowers[int(i*(i+1)/2 + j)], label='Emp E')

                # Theory bandpowers
                ax[i, j].plot(cll_bands,  bandpowers[int(i*(i+1)/2 + j)]/ bandpower_theory[lbl], c='blue', label='Ratio')
                ax[i, j].plot(cll_bands,  np.ones_like(cll_bands), c='black', linestyle='-', label='Ratio')


                # Empirical B bandpowers
                # ax[i, j].plot(cll_bands, b_bandpowers[int(i*(i+1)/2 + j)],
                #               c='red', ls='dashed', label='Emp B')

                # Scales / labels
                # ax[i, j].set_xscale('log')
                # ax[i, j].set_yscale('symlog')
                ax[i, j].text(cll_bands[1], 0.5*(ax[i, j].get_ylim()[1]), lbl)

                if i == nbins - 1:
                    ax[i, j].set_xlabel(r'$\ell$', fontsize=12)
                else:
                    plt.setp(ax[i, j].get_xticklabels(), visible=False)
                
                if j == 0:
                    ax[i, j].set_ylabel(r'$C_{\ell}$', fontsize=12)
                else:
                    plt.setp(ax[i, j].get_yticklabels(), visible=False)

        plt.tight_layout()
        plt.savefig(out_dir / f'unmixed_sim{save_sim_num_string}_bandpowers_ratios.png', dpi=300)
        plt.close()

        realised_unmixed_shear_cls_cut = realised_unmixed_shear_cls[:, :, :, lower_lscale:upper_lscale+1]
        cll_bands, bandpowers = compute_cl_bandpowers(realised_unmixed_shear_cls_cut, nbins, lower_lscale, upper_lscale, nbands)
        centre_ell = 0.5 * (cll_bands[1:] + cll_bands[:-1])

        matplotlib.rcParams.update({'font.size': 20})

        counter=0
        fig, axs = plt.subplots(nbins, nbins, figsize=(20, 20))
        xmin, xmax = lower_lscale, upper_lscale
        ymin, ymax = 5e-12, 5e-8

        for i in range(0, nbins):
            for j in range(0, nbins):
                if i<j:
                    axs[i, j].axis('off')
                else:
                    x = 10**(np.log10(xmin)) + 0.7*(np.log10(xmax) - np.log10(xmin))
                    y = ymin + 0.3*(ymax - ymin)

                    axs[i, j].plot(np.arange(lower_lscale, upper_lscale+1), realised_unmixed_shear_cls_cut[i][j][1], c='grey', ls = '--')
                    axs[i, j].plot(np.arange(lower_lscale, upper_lscale+1), realised_unmixed_shear_cls_cut[i][j][0], c='C0')
                    axs[i, j].plot(np.arange(0, lmax+1), theory_cls[f'W{i+1}xW{j+1}'], c='black')


                    axs[i, j].set_xscale('log')
                    axs[i, j].set_yscale('log')
                    axs[i, j].set_xlim([xmin, xmax])
                    axs[i, j].set_ylim([ymin, ymax])
                    label = 'S'+str(i+1)+'-S'+str(j+1)
                    axs[i, j].text(x, y, label)

                    if i == nbins - 1:
                        axs[i, j].set_xlabel(r'$\ell$', fontsize=22)
                    else:
                        plt.setp(axs[i, j].get_xticklabels(), visible=False)
                    
                    if j == 0:
                        axs[i, j].set_ylabel(r'$\tilde{C}_{\ell}$', fontsize=22)
                    else:
                        plt.setp(axs[i, j].get_yticklabels(), visible=False)

                    counter += 1

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        out_dir = PLOTTING_DIR / 'shear_cls'
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f'shear_cls_sim{save_sim_num_string}.png', dpi=300)
        plt.close()