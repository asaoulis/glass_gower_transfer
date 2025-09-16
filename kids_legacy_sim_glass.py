import numpy as np
import healpy as hp

import h5py
import numpy as np
from pathlib import Path
from collections import deque
import gc

import camb
import glass.ext.camb
from mpi4py import MPI

# use the CAMB cosmology that generated the matter power spectra
import glass
import glass.shells


def save_results_h5(filename, cat_idx, cls_results, pixelised_results, cosmo_dict):
    filename = Path(filename)
    # if filename has no suffix, leave it; else keep extension:
    if filename.suffix == "":
        outname = filename.with_name(f"{filename.stem}_{cat_idx}")
    else:
        outname = filename.with_name(f"{filename.stem}_{cat_idx}{filename.suffix}")

    outdir = outname.parent
    outdir.mkdir(parents=True, exist_ok=True)

    def _save_dict(h5group, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                subgroup = h5group.create_group(str(key))
                _save_dict(subgroup, value)

            elif isinstance(value, str):
                dt = h5py.string_dtype(encoding="utf-8")
                h5group.create_dataset(str(key), data=value, dtype=dt)

            else:
                arr = np.asarray(value)

                # force cast object arrays to float64
                if arr.dtype == object:
                    try:
                        arr = arr.astype(np.float64)
                    except Exception as e:
                        raise TypeError(
                            f"Cannot cast key '{key}' to float64: {e}\nValue={value}"
                        )

                h5group.create_dataset(str(key), data=arr)


    with h5py.File(outname, "w") as f:
        _save_dict(f.create_group("cls_results"), cls_results)
        _save_dict(f.create_group("pixelised_results"), pixelised_results)
        _save_dict(f.create_group("cosmo_dict"), cosmo_dict)

    print(f"Results saved to {outname}")


import time
import os
from src.cosmology import nla, glass_utils, levin, parameters, priors
from src.cosmology.map_shears import map_shears
from src.cosmology.simulators import GowerStreetSimulator, GlassLogNormalSimulator

from src.cosmology.gower_street import GowerStCosmologies, GowerStDatasetBuilder

from src.cosmology.manip_cls import denoise_shear_cls, unmix_shear_cl, cat2mask, maskcls, compute_cl_bandpowers
from src.cosmology.pixelise_maps import get_patch_values

from src.cosmology.map_shears  import make_alm_shear_convergence, filter_EB_alms_and_make_maps

from src.KiDS.tomo import calculate_tomo_nz

def process_cls(catalogue, nbins, nside, alm, alm_rand, lower_lscale, upper_lscale, nbands):
    shear_cls_noiseless = denoise_shear_cls(nbins, alm, alm_rand, lmax)
    bin_masks = cat2mask(catalogue, nbins, nside)
    bin_mask_cls = maskcls(bin_masks, lmax=int(1.5*lmax), nbins=nbins)
    realised_unmixed_shear_cls = unmix_shear_cl(num_bins = nbins, shear_cls = shear_cls_noiseless, mask_cl = bin_mask_cls, lmin = lmin, lmax = lmax)
    realised_unmixed_shear_cls_cut = realised_unmixed_shear_cls[:, :, :, lower_lscale:upper_lscale+1]
    cll_bands, bandpowers = compute_cl_bandpowers(realised_unmixed_shear_cls_cut, nbins, lower_lscale, upper_lscale, nbands)
    return realised_unmixed_shear_cls, cll_bands, bandpowers


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

# Tomographic redshift bins
nbins = 6

bias = 1
SIMULATOR_TYPE = 'glass'

nside = 1024
n_ell = 20
# lmax = 300
lmax = 2*nside
lmin = 0

NUM_JOBS = 10000
rotation_angles=[0, 90, 180, 270]
inner_num_shape_noise_realisations=1
outer_num_shape_noise_realisations=1
lower_lscale = 76
upper_lscale = 1500
nbands = 20
named_patches = {
    "south":(12, -31, 90, 11),     # (lon_center, lat_center, lon_range, lat_range)
    "north":(-178, 0, 112, 10)
}
patches = list(named_patches.values())
# rotation_values = [rot for rot in rotation_angles for _ in range(num_shape_noise_realisations)]

OUTPUT_DIR = Path('/share/gpu5/asaoulis/transfer_datasets/glass_full_only_mocks')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # class DummyComm:
    #     def Get_rank(self): return 0
    #     def Get_size(self): return 1
    #     def bcast(self, value, root=0): return value
    #     def Scatterv(self, *args, **kwargs): return args[1]  # naive pass-through
    # comm = DummyComm()
    # rank = 0
    # size = 1
    s = time.time() # Start timer 
    sim_batch_name = 'sobol_batch_1'
    
    seed_num = None # Seed for reproducibility
    shift_nz = True # Whether to shift the n(z) using the dz biases and covariance
    
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
    
    # ------------------ distribute sim_samples using Scatterv (robust) ------------------
    if rank == 0:
        sim_samples = np.arange(NUM_JOBS).reshape(-1, 1).astype(np.float64)
        cols = sim_samples.shape[1]
        N = sim_samples.shape[0]
    else:
        sim_samples = None
        cols = None
        N = None

    # share cols & N with all ranks
    cols = comm.bcast(cols, root=0)
    N = comm.bcast(N, root=0)

    # compute counts in rows (how many rows each rank gets)
    if rank == 0:
        counts_rows = [N // size] * size
        rem = N % size
        for i in range(rem):
            counts_rows[i] += 1
    else:
        counts_rows = None

    # broadcast row counts to everyone
    counts_rows = comm.bcast(counts_rows, root=0)

    # compute row displacements (in rows)
    displs_rows = [0] * size
    for i in range(1, size):
        displs_rows[i] = displs_rows[i-1] + counts_rows[i-1]

    # convert row-based counts/displs -> element-based (MPI.DOUBLE units)
    counts = [r * cols for r in counts_rows]
    displs = [d * cols for d in displs_rows]

    # allocate flat recv buffer of length = elements assigned to this rank
    recv_elems = counts_rows[rank] * cols
    recvbuf_flat = np.empty(recv_elems, dtype=np.float64)

    # prepare flattened sendbuf on rank 0
    sendbuf_flat = sim_samples.flatten() if rank == 0 else None

    # Scatterv: send flattened arrays, counts/displs in element units
    comm.Scatterv([sendbuf_flat, counts, displs, MPI.DOUBLE],
                recvbuf_flat,
                root=0)

    # reshape recv buffer to (rows, cols) so code below can index by columns
    if cols > 0:
        recvbuf = recvbuf_flat.reshape(-1, cols)
    else:
        recvbuf = np.empty((0, cols), dtype=np.float64)

    print(f"[rank {rank}] received chunk (shape {recvbuf.shape}):\n{recvbuf}")

    # sims: last column interpreted as integers (works even if recvbuf is empty)
    sims = recvbuf[:, -1].astype(int) if recvbuf.size else np.array([], dtype=int)

    # loop over received sim rows
    for num_sim_this_batch in range(len(recvbuf)):
        sim_num = sims[num_sim_this_batch]

        gower_street_loader = GowerStDatasetBuilder(csv_path, gower_data_dir)

        # ... rest of your per-simulation code ...

        omega_k = 0.0
        
        # intrinsic alignments params
        massdep_means = np.loadtxt(f'{data_dir}/priors/massdep_means.txt')
        massdep_cov = np.loadtxt(f'{data_dir}/priors/massdep_cov.txt')
        log10_M_eff_means = massdep_means[2:]
        log10_M_eff_cov = massdep_cov[2:,2:] 
        f_red = np.array([0.15, 0.2, 0.17, 0.24, 0.19, 0.03])
        log10_M_eff = np.random.multivariate_normal(log10_M_eff_means, log10_M_eff_cov, size=1)[0]

        logT_AGN_realised = 7.8
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
        vis = mask = hp.read_map(f'{data_dir}/masks/KiDS_Legacy_N_healpix_1024_frac_withAstrom.fits') + hp.read_map(f'{data_dir}/masks/KiDS_Legacy_S_healpix_1024_frac_withAstrom.fits')

        cosmo, pars, param_dict  = gower_street_loader.get_simulation_cosmology(sim_num, nuisance_params)
        results = camb.get_results(pars)
        results.calc_power_spectra(pars)
        k, z_grid, pk = results.get_nonlinear_matter_power_spectrum()
        chi_grid = results.angular_diameter_distance(z_grid) * (1+z_grid) # the correpsonding comoving distances 
        extended_k, extended_pk = levin.extrapolate_section(k, z_grid, pk, **levin_params)

        print("Cosmology and CAMB parameters set up successfully.")
        print(f"It took {time.time() - s:.2f} seconds")
        
        # GLASS setup --------------------------------------------------------------------
        # shells of 200 Mpc in comoving distance spacing
        zb = glass.shells.distance_grid(cosmo, 0.000, 3.1, dx=200.)

        n_los_chi = 1000 #define the integration limits here
        los_z_integration = np.linspace(zb[0], zb[-1], n_los_chi)
        los_chi_integration = np.asarray(results.angular_diameter_distance(los_z_integration) * (1+los_z_integration))
        print('Setting up KiDS-Legacy properties...')
        

        tomo_nz = calculate_tomo_nz(data_dir, n_los_chi, los_z_integration, shift_nz)

        # Read in the KiDS-Legacy mask
        # Realised shear bias --------------------------------------------------------------------
        for outer_noise_idx in range(outer_num_shape_noise_realisations):
            rotation_values = [rot for rot in rotation_angles for _ in range(inner_num_shape_noise_realisations)]
            m_bias_realised = np.array([float(np.random.normal(m_bias[i], m_bias_unc[i], 1)) for i in range(len(m_bias))])
            c1_bias_north_realised = np.array([float(np.random.normal(c_1_bias_north[i], c_1_bias_north_unc[i], 1)) for i in range(len(c_1_bias_north))])
            c2_bias_north_realised = np.array([float(np.random.normal(c_2_bias_north[i], c_2_bias_north_unc[i], 1)) for i in range(len(c_2_bias_north))])
            c1_bias_south_realised = np.array([float(np.random.normal(c_1_bias_south[i], c_1_bias_south_unc[i], 1)) for i in range(len(c_1_bias_south))])
            c2_bias_south_realised = np.array([float(np.random.normal(c_2_bias_south[i], c_2_bias_south_unc[i], 1)) for i in range(len(c_2_bias_south))])

            # Make the catalogue ---------------------------------------------------
            # TODO: add GLASS lognormals here
            print('Simulating the galaxy catalogue...')
            s_catalogue = time.time()

            kwargs = {
                'cosmo': cosmo,
                'los_z_integration': los_z_integration,
                'tomo_nz': tomo_nz,
                'galaxy_bias': 1.0,
                'shear_bias_params': {
                    'm_bias': m_bias_realised,
                    'c1_north': c1_bias_north_realised,
                    'c2_north': c2_bias_north_realised,
                    'c1_south': c1_bias_south_realised,
                    'c2_south': c2_bias_south_realised
                },
                'nla_params': ia_params,
                'sigma_e': sigma_e,
                'mask': mask,
                'nside': nside,
                'nbins': nbins,
                'rng': np.random.default_rng()
            }

            if SIMULATOR_TYPE == 'glass':
                ws, lp, ell = levin.setup_levin_power(zb, z_grid, chi_grid, extended_k, extended_pk, results, pars)
                glass_cls, ws, n_glass_shells = glass_utils.compute_glass_cls(lp, ws, ell)
                glass_cls_discretized = glass.discretized_cls(glass_cls, nside=nside, lmax=lmax, ncorr=1)
                fields = glass.lognormal_fields(ws)
                gls = glass.solve_gaussian_spectra(fields, glass_cls_discretized)
                matter = glass.generate(fields, gls, nside, ncorr=1, rng=kwargs['rng'])
                simulator = GlassLogNormalSimulator(matter, ws, **kwargs)
            elif SIMULATOR_TYPE == 'gower':
                simulator = gower_street_loader.setup_simulator(sim_num, **kwargs)

            catalogues = simulator.run(rotation_angles=rotation_angles, num_shape_noise_realisations=inner_num_shape_noise_realisations)

            print(f'Total number of augmentations sampled: {len(catalogues):,}')
            print(f'Simulated the galaxy catalogue in {time.time() - s_catalogue:.2f} seconds')

            print('Calculating the shear power spectra...')

            # Process catalogues one-by-one and free memory immediately
            cat_queue = deque(catalogues)
            del catalogues
            gc.collect()

            cat_idx = 0
            while cat_queue:
                cat_parts = cat_queue.popleft()
                try:
                    catalogue = np.concatenate(cat_parts)
                finally:
                    del cat_parts

                ang = rotation_values[cat_idx]
                cls_results = {cl_type:{} for cl_type in ['full', 'north', 'south']}

                alm, alm_rand, shear = make_alm_shear_convergence(
                    catalogue, m_bias_realised, nbins, nside, lmax, nosh=False
                )

                E, B = filter_EB_alms_and_make_maps(
                    alm_list=alm, nside_out=512, lmax_out=None, fwhm_arcmin=8.0, taper_start_frac=0.95
                )

                realised_unmixed_shear_cls, cll_bands, bandpowers = process_cls(catalogue, nbins, nside, alm, alm_rand, lower_lscale, upper_lscale, nbands, )

                # map_types = {"shear_real": shear.real, "shear_imag": shear.imag, "E":E, "B":B}
                map_types = {"E":E, "B":B}
                pixelised_results = {name:{} for name in map_types.keys()}
                for name, cat_data in map_types.items():
                    pixelised_tomobin_patches = get_patch_values(cat_data, patches, 512, ang)
                    for patch_idx, patch_name in enumerate(named_patches.keys()):
                        pixelised_results[name][patch_name] = pixelised_tomobin_patches[patch_idx]

                cls_results['full'] = {"cls": realised_unmixed_shear_cls, "bandpowers":bandpowers, "bandpower_ls":cll_bands}
  
                # patch_defs = {
                #     "north": (np.abs(catalogue['DEC']) < 15),
                #     "south": (np.abs(catalogue['DEC']) >= 15),
                # }
                # for patch_name, selector in patch_defs.items():
                #     subcat = catalogue[selector]
                #     alm, alm_rand, _ = make_alm_shear_convergence(subcat, m_bias_realised, nbins, nside, lmax, nosh=False)
                #     realised_unmixed_shear_cls, cll_bands, bandpowers = process_cls(subcat, nbins, nside, alm, alm_rand, lower_lscale, upper_lscale, nbands, )
                #     cls_results[patch_name] = {"cls": realised_unmixed_shear_cls, "bandpowers":bandpowers, "bandpower_ls":cll_bands}


                total_idx = outer_noise_idx * outer_num_shape_noise_realisations + cat_idx
                save_results_h5( OUTPUT_DIR / f"output_{sim_num}.h5", total_idx, cls_results, pixelised_results, param_dict)

                # free per-catalogue heavy products
                del catalogue, cls_results, pixelised_results, realised_unmixed_shear_cls, cll_bands, bandpowers, map_types, E, B, alm, alm_rand, shear
                gc.collect()

                cat_idx += 1

            del cat_queue, simulator
            gc.collect()

        print(f'Saved results for sim {sim_num}')
        
        print(f'Entire simulation took {time.time() - s:.2f} seconds')
