import numpy as np
import healpy as hp

import h5py
from pathlib import Path
from collections import deque
import gc
import argparse

import glass.ext.camb
from cosmology import Cosmology

from mpi4py import MPI

# use the CAMB cosmology that generated the matter power spectra
import glass
import glass.shells

import time
from src.cosmology import parameters
from src.cosmology.simulators import GlassLogNormalSimulator
from src.cosmology.systematics import NLASystematics
from src.cosmology.gower_street import GowerStCosmologies, GowerStPrior

from src.cosmology.manip_cls import  process_cls, unmixing_mask_cls, compute_cl_bandpowers, denoise_shear_cls
from src.cosmology.pixelise_maps import get_patch_values

from src.cosmology.map_shears  import make_alm_shear_convergence, filter_EB_alms_and_make_maps
from src.cosmology.camb_matter_power import get_camb_matter_cls
from src.cosmology.mpi_camb import (
    compute_camb_glass_in_child_npz_subproc,
    load_camb_child_pickle,
)
from src.KiDS.tomo import calculate_tomo_nz
from src.KiDS.rotations import KiDS_PATCH_GOWER_ROTATIONS, KiDS_PATCH_GLASS_ROTATIONS

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


def parse_args():
    parser = argparse.ArgumentParser(description="MPI Glass/Gower KiDS simulation")

    # Core switches
    parser.add_argument("--simulator-type", type=str, default="gower_street",
                        choices=["gower_street", "glass"],
                        help="Simulator backend")

    parser.add_argument("--kids-systematics", action="store_true",
                        help="Enable KiDS systematics")

    parser.add_argument("--no-rotations", action="store_true",
                        help="Disable rotations")

    parser.add_argument("--use-kids-mask", action="store_true",
                        help="Use KiDS mask")
    # Paths
    parser.add_argument("--csv-path", type=Path,
                        default=Path("/home/asaoulis/projects/glass_transfer/kids-legacy-sbi/data/gower_st/PKDGRAV3_on_DiRAC_DES_330.csv"))

    parser.add_argument("--gower-data-dir", type=Path,
                        default=Path("/share/gpu5/asaoulis/gowerstreet"))

    parser.add_argument("--output-dir", type=Path,
                        default=Path("/share/gpu5/asaoulis/transfer_datasets/gower_full_only_mocks"))

    return parser.parse_args()


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

massdep_means = np.loadtxt(f'{data_dir}/priors/massdep_means.txt')
massdep_cov = np.loadtxt(f'{data_dir}/priors/massdep_cov.txt')
log10_M_eff_means = massdep_means[2:]
log10_M_eff_cov = massdep_cov[2:,2:] 
f_red = np.array([0.15, 0.2, 0.17, 0.24, 0.19, 0.03])
# Instrinsic galaxy shape dispersion per tomographic bin
sigma_e = np.array([0.2772, 0.2716, 0.2899, 0.2619, 0.2802, 0.3002])

nbins = 6
bias = 1

nside = 1024
n_ell = 20
# lmax = 300
lmax = 2*nside
lmin = 0

zmin, zmax = 0.0, 2.0
dx = 200.0  # Mpc/h
n_los_chi = 1000  # define the integration limits here

outer_num_shape_noise_realisations=4 # gower
outer_num_shape_noise_realisations=1 # glass
inner_num_shape_noise_realisations=1
mask_rotation_angles = [0, 90, 180, 270]

lower_lscale = 76
upper_lscale = 1500
nbands = 8
named_patches = {
    "south":(12, -31, 90, 11),     # (lon_center, lat_center, lon_range, lat_range)
    "north":(-178, 0, 112, 10)
}
patches = list(named_patches.values())

GLASS_N_JOBS = 3600

prior = {
            'logT_AGN':(7.3, 8.3),
            'a_ia': (4.48, 7.0),
            'b_ia': (0.28, 0.6),}

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    s = time.time() # Start timer 
    
    seed_num = None # Seed for reproducibility
    # SET AND PROCESS ARGS

    # ------------------ distribute sim_samples using Scatterv (robust) ------------------
    if rank == 0:
        args = parse_args()
        if args.simulator_type == 'glass':
            sim_samples = np.arange(GLASS_N_JOBS)
        else:
            offset = 193  # <-- starting point
            FINAL_INDEX = 781 + 1
            sim_samples = np.arange(offset, FINAL_INDEX)
            # remove 

        sim_samples = sim_samples.reshape(-1, 1).astype(np.float64)
        cols = sim_samples.shape[1]
        N = sim_samples.shape[0]
    else:
        sim_samples = None
        cols = None
        N = None
        args= None
    args = comm.bcast(args, root=0)
    SIMULATOR_TYPE = args.simulator_type
    KIDS_SYSTEMATICS = args.kids_systematics
    NO_ROTATIONS = args.no_rotations
    USE_KIDS_MASK = args.use_kids_mask
    csv_path = args.csv_path
    gower_data_dir = args.gower_data_dir
    gower_prior = GowerStPrior.from_csv(csv_path)
    if NO_ROTATIONS:
        rotation_specs =  [{"rot": 0, "flip": False, "backend": "pixel"}]
    else:
        if SIMULATOR_TYPE == 'glass':
            rotation_specs = KiDS_PATCH_GLASS_ROTATIONS
        else:
            rotation_specs = KiDS_PATCH_GOWER_ROTATIONS

    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
    try:
        for num_sim_this_batch in range(len(recvbuf)):
            for outer_idx in range(outer_num_shape_noise_realisations):
                for rot_idx, rotation_spec in enumerate(rotation_specs):
                    # if rot_idx == 0 and outer_idx ==0:
                    #     continue
                    rng = np.random.default_rng()

                    sim_num = sims[num_sim_this_batch]
                    # ... rest of your per-simulation code ...
                    if SIMULATOR_TYPE == "glass" and outer_idx == 0 or SIMULATOR_TYPE == "gower_street" :
                        log10_M_eff = np.random.multivariate_normal(log10_M_eff_means, log10_M_eff_cov, size=1)[0]
                        a_ia_realised     = np.random.uniform(*prior['a_ia'])
                        b_ia_realised     = np.random.uniform(*prior['b_ia'])
                        nuisance_params = {"a_ia": a_ia_realised, "b_ia": b_ia_realised}

                        # intrinsic alignments params
                        ia_params = dict(
                            a_ia = a_ia_realised,
                            b_ia = b_ia_realised,
                            f_red = f_red,
                            log10_M_eff = log10_M_eff,
                        )
                    if USE_KIDS_MASK:
                        mask = hp.read_map(f'{data_dir}/masks/KiDS_Legacy_N_healpix_1024_frac_withAstrom.fits') + hp.read_map(f'{data_dir}/masks/KiDS_Legacy_S_healpix_1024_frac_withAstrom.fits')
                    else:
                        mask = np.ones(hp.nside2npix(nside))

                    print("Cosmology and CAMB parameters set up successfully.")
                    print(f"It took {time.time() - s:.2f} seconds")
                    

                    m_bias_realised = np.array([float(np.random.normal(m_bias[i], m_bias_unc[i], 1)) for i in range(len(m_bias))])
                    c1_bias_north_realised = np.array([float(np.random.normal(c_1_bias_north[i], c_1_bias_north_unc[i], 1)) for i in range(len(c_1_bias_north))])
                    c2_bias_north_realised = np.array([float(np.random.normal(c_2_bias_north[i], c_2_bias_north_unc[i], 1)) for i in range(len(c_2_bias_north))])
                    c1_bias_south_realised = np.array([float(np.random.normal(c_1_bias_south[i], c_1_bias_south_unc[i], 1)) for i in range(len(c_1_bias_south))])
                    c2_bias_south_realised = np.array([float(np.random.normal(c_2_bias_south[i], c_2_bias_south_unc[i], 1)) for i in range(len(c_2_bias_south))])

                    s_catalogue = time.time()
                    print("Setting up simulator...")
                    if SIMULATOR_TYPE == "glass":

                        sampled_cosmo_params = gower_prior.draw_param_dict_sample(rng=rng)
                        param_dict = {
                            **sampled_cosmo_params,
                            **nuisance_params,
                        }
                        # only recompute theory on first outer_idx
                        if outer_idx == 0:
                            print("Computing CAMB matter power spectra...")

                            npz_out_path = compute_camb_glass_in_child_npz_subproc(
                                param_dict,
                                lmax,
                                zmin,
                                zmax,
                                dx,
                                mem_limit_gb=200,
                                timeout_s=3600*4,
                                sim_tag=f"sim{sim_num}",
                            )

                            shells, glass_cls = load_camb_child_pickle(npz_out_path, remove_after_load=True)
                            # Build cosmology and CAMB parameters
                            cosmo, pars = parameters.build_cosmology(param_dict)
                            # shells, glass_cls = get_camb_matter_cls(pars, lmax, zmin, zmax, dx)
                        glass_cls_discretized = glass.discretized_cls(glass_cls, nside=nside, lmax=lmax, ncorr=1)
                        fields = glass.lognormal_fields(shells)
                        gls = glass.solve_gaussian_spectra(fields, glass_cls_discretized)
                        matter = glass.generate(fields, gls, nside, ncorr=1, rng=rng)
                        cosmo = Cosmology.from_camb(pars)
                        print(f"Finished computing CAMB matter power spectra afrer {time.time() - s_catalogue:.2f}s.")
                    else:
                        gower_street_loader = GowerStCosmologies(gower_data_dir, csv_path)
                        param_dict = gower_street_loader.get_params_from_sim_id(sim_num, extra_params=nuisance_params)
                        shells, matter, cosmo = gower_street_loader.load_shells_matter_and_cosmology(sim_num, nside=nside)
                        _, pars, _  = gower_street_loader.get_simulation_cosmology(sim_num, nuisance_params)
                        cosmo = Cosmology.from_camb(pars)

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
                        shift_nz = True
                    else:
                        systematics = None
                        sigma_e *= 0.0
                        m_bias_realised *= 0.0
                        shift_nz = False

                    zb = glass.shells.distance_grid(cosmo, zmin, zmax, dx=dx)
                    los_z_integration = np.linspace(zb[0], zb[-1], n_los_chi)
                    tomo_nz = calculate_tomo_nz(data_dir, n_los_chi, los_z_integration, shift_nz)

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
                    print('Simulating the galaxy catalogue...')

                    simulator = GlassLogNormalSimulator(matter, shells, **kwargs)
                    catalogues = simulator.run(rotation_spec, mask_rotation_angles, num_shape_noise_realisations=inner_num_shape_noise_realisations)

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

                        ang = 0
                        cls_results = {cl_type:{} for cl_type in ['full', 'north', 'south']}

                        alm, alm_rand = make_alm_shear_convergence(
                            catalogue, m_bias_realised, nbins, nside, lmax, nosh=False, mask=mask
                        )
                        # mask_cls = unmixing_mask_cls(catalogue, nbins, nside, lmax, lmin, mask=mask)

                        mixed_cls = denoise_shear_cls(nbins, alm, alm_rand, lmax)
                        mixed_cut = mixed_cls[:, :, :, lower_lscale:upper_lscale+1]
                        cll_bands, mixed_bandpowers = compute_cl_bandpowers(
                            mixed_cut, nbins, lower_lscale, upper_lscale, nbands
                        )

                        del catalogue
                        gc.collect()

                        E, B = filter_EB_alms_and_make_maps(
                            alm_list=alm, nside_out=512, lmax_out=None, fwhm_arcmin=8.0, taper_start_frac=0.95
                        )

                        # realised_unmixed_shear_cls, cll_bands, bandpowers = process_cls(mask_cls, nbins, nside, alm, alm_rand, lower_lscale, upper_lscale,lmin, lmax, nbands, )

                        # map_types = {"shear_real": shear.real, "shear_imag": shear.imag, "E":E, "B":B}
                        map_types = {"E":E, "B":B}
                        pixelised_results = {name:{} for name in map_types.keys()}
                        for name, cat_data in map_types.items():
                            pixelised_tomobin_patches = get_patch_values(cat_data, patches, 512, ang)
                            for patch_idx, patch_name in enumerate(named_patches.keys()):
                                pixelised_results[name][patch_name] = pixelised_tomobin_patches[patch_idx]

                        # cls_results['full'] = {"cls": mixed_cls, "mixed_bandpowers":mixed_bandpowers, "bandpower_ls":cll_bands}
                        cls_results['full'] = {"mixed_bandpowers":mixed_bandpowers, "bandpower_ls":cll_bands}

                        # patch_defs = {
                        #     "north": (np.abs(catalogue['DEC']) < 15),
                        #     "south": (np.abs(catalogue['DEC']) >= 15),
                        # }
                        # for patch_name, selector in patch_defs.items():
                        #     subcat = catalogue[selector]
                        #     alm, alm_rand, _ = make_alm_shear_convergence(subcat, m_bias_realised, nbins, nside, lmax, nosh=False)
                        #     realised_unmixed_shear_cls, cll_bands, bandpowers = process_cls(subcat, nbins, nside, alm, alm_rand, lower_lscale, upper_lscale, nbands, )
                        #     cls_results[patch_name] = {"cls": realised_unmixed_shear_cls, "bandpowers":bandpowers, "bandpower_ls":cll_bands}

                        save_string = f"{sim_num}_out{outer_idx}_rot{rot_idx}"
                        total_idx = cat_idx
                        save_results_h5( OUTPUT_DIR / f"output_{save_string}.h5", total_idx, cls_results, pixelised_results, param_dict)

                        # free per-catalogue heavy products
                        del cls_results, pixelised_results, cll_bands, map_types, E, B, alm, alm_rand
                        gc.collect()

                        cat_idx += 1

                    del cat_queue, simulator
                    gc.collect()
                    print(f"Finished single rotation simulation in {time.time() - s_catalogue:.2f} seconds")

                print(f'Saved results for sim {sim_num}')
            
            print(f'Entire simulation took {time.time() - s:.2f} seconds')
    except Exception as e:
        print(f"[rank {rank}] Error processing sims: {e}")
        # Don't re-raise to avoid MPI hang; just log the error and continue