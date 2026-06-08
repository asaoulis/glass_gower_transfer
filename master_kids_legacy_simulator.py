import numpy as np
import healpy as hp
from pathlib import Path
from collections import deque
import gc
import argparse

import glass.ext.camb

from mpi4py import MPI

# use the CAMB cosmology that generated the matter power spectra
import glass
import glass.shells

import time
from src.cosmology.simulators import GlassMatterShellSimulator
from src.cosmology.systematics import NLASystematics
from src.cosmology.gower_street import GowerStCosmologies, GowerStPrior
from src.cosmology.sim_utils import (
    build_systematics,
    build_variable_depth,
    model_shift_nz,
    prepare_glass_backend,
    prepare_gower_backend,
    resolve_systematics_model,
    save_results_h5,
)

from src.cosmology.manip_cls import compute_cl_bandpowers, denoise_shear_cls
from src.cosmology.pixelise_maps import get_patch_values

from src.cosmology.map_shears  import make_alm_shear_convergence, filter_EB_alms_and_make_maps
from src.KiDS.tomo import calculate_tomo_nz
from src.KiDS.rotations import KiDS_PATCH_GOWER_ROTATIONS, KiDS_PATCH_GLASS_ROTATIONS
from src.KiDS.simulation_config import (
    CAMB_LIMITS,
    INNER_NUM_SHAPE_NOISE_REALISATIONS,
    LOS_GRID,
    OVERWRITE,
    OUTER_NUM_SHAPE_NOISE_REALISATIONS,
    SIM_GRID,
    bias,
    dx,
    lmax,
    lmin,
    load_kids_mask,
    lower_lscale,
    mask_rotation_angles,
    nbands,
    nbins,
    n_ell,
    n_los_chi,
    named_patches,
    nside,
    patches,
    upper_lscale,
    zmax,
    zmin,
)
from src.KiDS.systematics import (
    c_1_bias_north,
    c_1_bias_north_unc,
    c_1_bias_south,
    c_1_bias_south_unc,
    c_2_bias_north,
    c_2_bias_north_unc,
    c_2_bias_south,
    c_2_bias_south_unc,
    f_red,
    load_massdep_priors,
    m_bias,
    m_bias_unc,
    sigma_e,
)
from src.KiDS.variable_depth_config import (
    alpha_1,
    alpha_1_unc,
    alpha_2,
    alpha_2_unc,
    load_psf_maps,
    m_bias_vd,
    m_bias_vd_unc,
    n_vardepth_bins,
    vd_trace_edges,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MPI Glass/Gower KiDS simulation")

    # Core switches
    parser.add_argument("--simulator-type", type=str, default="gower_street",
                        choices=["gower_street", "glass", "smoke"],
                        help="Simulator backend ('smoke' = reduced-cost local pre-flight)")

    parser.add_argument("--kids-systematics", action="store_true",
                        help="Enable KiDS systematics")

    parser.add_argument(
        "--systematics-model",
        type=str,
        default="auto",
        choices=["auto", "none", "nla"],
        help="Systematics model to apply (default: auto = nla iff --kids-systematics)",
    )

    parser.add_argument(
        "--variable-depth", action="store_true",
        help="Enable the KiDS-Legacy variable-depth effect (implies full NLA + shear-bias "
             "systematics; resolves the systematics model to 'nla_vd'). Use with --use-kids-mask.",
    )

    parser.add_argument("--no-rotations", action="store_true",
                        help="Disable rotations")

    parser.add_argument("--num-sims", type=int, default=None,
                        help="Cap the number of cosmologies/simulations to the first N "
                             "(default: all). Used for bounded validation/theory-test runs.")

    parser.add_argument("--no-augmentation", action="store_true",
                        help="Produce ONE mock per cosmology: collapse rotations AND the "
                             "outer/inner shape-noise realisations to a single one. For fast "
                             "per-cosmology theory tests (implies no rotations).")

    parser.add_argument("--shear-normalization", type=str, default="counts",
                        choices=["counts", "mean", "expected"],
                        help="Shear-map normalisation in make_alm_shear_convergence "
                             "(default 'counts' = per-pixel observed counts; 'mean' = global "
                             "mean counts/pixel = the pre-c892fc1 behaviour, for comparison).")

    parser.add_argument("--use-kids-mask", action="store_true",
                        help="Use KiDS mask")
    # Paths
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/home/asaoulis/projects/glass_transfer/kids-legacy-sbi/data"))

    parser.add_argument("--csv-path", type=Path,
                        default=Path("/home/asaoulis/projects/glass_transfer/kids-legacy-sbi/data/gower_st/PKDGRAV3_on_DiRAC_DES_330.csv"))

    parser.add_argument("--gower-data-dir", type=Path,
                        default=Path("/share/gpu5/asaoulis/gowerstreet"))

    parser.add_argument("--output-dir", type=Path,
                        default=Path("/share/gpu5/asaoulis/transfer_datasets/gower_full_only_mocks"))

    return parser.parse_args()

GLASS_N_JOBS = 16000

prior = {
            'a_ia': (4.48, 7.0),
            'b_ia': (0.28, 0.6),}


SIM_TYPE_CONFIGS = {
    "glass": {
        "rotation_specs": KiDS_PATCH_GLASS_ROTATIONS,
        "get_sim_samples": lambda: np.arange(GLASS_N_JOBS),
    },
    "gower_street": {
        "rotation_specs": KiDS_PATCH_GOWER_ROTATIONS,
        "get_sim_samples": lambda: np.arange(193, 781 + 1),
    },
    # Reduced-cost local pre-flight: a single sim; rotations/augmentations collapsed below.
    "smoke": {
        "rotation_specs": KiDS_PATCH_GLASS_ROTATIONS,
        "get_sim_samples": lambda: np.arange(1),
    },
}

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

        sim_samples = SIM_TYPE_CONFIGS[args.simulator_type]["get_sim_samples"]()

        if args.num_sims is not None:
            n_keep = min(args.num_sims, len(sim_samples))
            print(f"[rank 0] --num-sims={args.num_sims}: using {n_keep} of "
                  f"{len(sim_samples)} available {args.simulator_type} cosmologies.")
            sim_samples = sim_samples[:n_keep]

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
    SYSTEMATICS_MODEL = resolve_systematics_model(args)
    NO_ROTATIONS = args.no_rotations
    NO_AUGMENTATION = args.no_augmentation
    SHEAR_NORMALIZATION = args.shear_normalization
    USE_KIDS_MASK = args.use_kids_mask
    csv_path = args.csv_path
    gower_data_dir = args.gower_data_dir
    data_dir = args.data_dir
    SMOKE = (SIMULATOR_TYPE == "smoke")
    if SMOKE:
        # Reduced-cost local pre-flight: fixed cosmo + tiny in-process CAMB->GLASS backend,
        # so no Gower prior CSV is needed. Fall back to the in-repo fixtures when the
        # (cluster) default --data-dir does not exist locally.
        from src.smoke_sim import SMOKE_CONFIG, prepare_smoke_backend
        if not data_dir.exists():
            data_dir = Path(__file__).resolve().parent / "kids-legacy-sbi" / "data"
        gower_prior = None
    else:
        gower_prior = GowerStPrior.from_csv(csv_path, drop_first=192)
    log10_M_eff_means, log10_M_eff_cov = load_massdep_priors(data_dir)
    if NO_ROTATIONS:
        rotation_specs =  [{"rot": 0, "flip": False, "backend": "pixel"}]
    else:
        rotation_specs = SIM_TYPE_CONFIGS[SIMULATOR_TYPE]["rotation_specs"]

    if SMOKE:
        # Collapse all augmentation loops to a single realisation and rebind the cost-driving
        # constants (these names are module globals imported from simulation_config; rebinding
        # them here at __main__ scope feeds them to every downstream call). Production path
        # (else) is byte-identical to before.
        rotation_specs = rotation_specs[:1]
        mask_rotation_angles = [0]
        USE_KIDS_MASK = True
        nside = SMOKE_CONFIG["nside"]
        lmax = SMOKE_CONFIG["lmax"]
        zmax = SMOKE_CONFIG["zmax"]
        dx = SMOKE_CONFIG["dx"]
        n_los_chi = SMOKE_CONFIG["n_los_chi"]
        lower_lscale = SMOKE_CONFIG["lower_lscale"]
        upper_lscale = SMOKE_CONFIG["upper_lscale"]
        nbands = SMOKE_CONFIG["nbands"]
        nside_out = SMOKE_CONFIG["nside_out"]
        outer_reps = 1
        inner_reps = 1
    else:
        nside_out = 512
        outer_reps = OUTER_NUM_SHAPE_NOISE_REALISATIONS[SIMULATOR_TYPE]
        inner_reps = INNER_NUM_SHAPE_NOISE_REALISATIONS[SIMULATOR_TYPE]

    if NO_AUGMENTATION and not SMOKE:
        # One mock per cosmology: a single rotation/footprint and a single shape-noise
        # realisation (outer & inner). For fast per-cosmology theory checks.
        rotation_specs = rotation_specs[:1]
        outer_reps = 1
        inner_reps = 1
        if rank == 0:
            print("[rank 0] --no-augmentation: 1 mock/cosmology "
                  "(1 rotation, outer_reps=inner_reps=1).")

    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    backend_states = {
        "glass": {
            "cache": {},
            "cosmo_prior": gower_prior,
            "output_dir": OUTPUT_DIR,
        },
    }
    if not SMOKE:
        # GowerStCosmologies reads cluster data on construction; skip for the local smoke.
        backend_states["gower_street"] = {
            "loader": GowerStCosmologies(gower_data_dir, csv_path),
        }
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
            path_glob_pattern = f"output_{sims[num_sim_this_batch]}*.h5"
            existing_files = list(OUTPUT_DIR.glob(path_glob_pattern))
            if (not OVERWRITE) and len(existing_files) > 0:
                print(f"[rank {rank}] Skipping sim {sims[num_sim_this_batch]} as output files already exist.")
                continue
            for outer_idx in range(outer_reps):
                for rot_idx, rotation_spec in enumerate(rotation_specs):
                    sim_num = sims[num_sim_this_batch]
                    if SIMULATOR_TYPE == "gower_street" and not OVERWRITE:
                        # check if any sim files for this sim+rotidx+outeridx already exist, and skip if so
                        path_glob_pattern = f"output_{sim_num}_out{outer_idx}_rot{rot_idx}*.h5"
                        existing_files = list(OUTPUT_DIR.glob(path_glob_pattern))
                        if len(existing_files) > 0:
                            print(f"[rank {rank}] Skipping sim {sim_num} with rot_idx {rot_idx} and outer_idx {outer_idx} as output files already exist.")
                            continue
                    rng = np.random.default_rng()
                    log10_M_eff = rng.multivariate_normal(log10_M_eff_means, log10_M_eff_cov, size=1)[0]
                    if USE_KIDS_MASK:
                        mask = load_kids_mask(data_dir).copy()
                        if SMOKE and hp.get_nside(mask) != nside:
                            mask = hp.ud_grade(mask, nside)
                    else:
                        mask = np.ones(hp.nside2npix(nside))

                    print("Cosmology and CAMB parameters set up successfully.")
                    print(f"It took {time.time() - s:.2f} seconds")
                    

                    m_bias_realised = np.array([float(rng.normal(m_bias[i], m_bias_unc[i], 1)) for i in range(len(m_bias))])
                    c1_bias_north_realised = np.array([float(rng.normal(c_1_bias_north[i], c_1_bias_north_unc[i], 1)) for i in range(len(c_1_bias_north))])
                    c2_bias_north_realised = np.array([float(rng.normal(c_2_bias_north[i], c_2_bias_north_unc[i], 1)) for i in range(len(c_2_bias_north))])
                    c1_bias_south_realised = np.array([float(rng.normal(c_1_bias_south[i], c_1_bias_south_unc[i], 1)) for i in range(len(c_1_bias_south))])
                    c2_bias_south_realised = np.array([float(rng.normal(c_2_bias_south[i], c_2_bias_south_unc[i], 1)) for i in range(len(c_2_bias_south))])

                    s_catalogue = time.time()
                    print("Setting up simulator...")

                    if SMOKE:
                        backend = prepare_smoke_backend(rng, SMOKE_CONFIG)
                    elif SIMULATOR_TYPE == "glass":
                        backend = prepare_glass_backend(
                            sim_num,
                            rng=rng,
                            output_dir=backend_states["glass"]["output_dir"],
                            cosmo_prior=backend_states["glass"]["cosmo_prior"],
                            prior_ranges=prior,
                            sim_grid=SIM_GRID,
                            los_grid=LOS_GRID,
                            camb_limits=CAMB_LIMITS,
                            cache=backend_states["glass"]["cache"],
                        )
                    else:
                        backend = prepare_gower_backend(
                            sim_num,
                            rng=rng,
                            loader=backend_states["gower_street"]["loader"],
                            prior_ranges=prior,
                            sim_grid=SIM_GRID,
                        )

                    param_dict = backend["param_dict"]
                    shells = backend["shells"]
                    matter = backend["matter"]
                    cosmo = backend["cosmo"]

                    ia_params = {
                        "a_ia": param_dict["a_ia"],
                        "b_ia": param_dict["b_ia"],
                        "f_red": f_red,
                        "log10_M_eff": log10_M_eff,
                    }

                    # n(z) shift depends only on the systematics model; compute it (and tomo_nz)
                    # BEFORE build_systematics so the variable-depth objects can be built from
                    # tomo_nz / los_z_integration / the shell edges.
                    shift_nz = model_shift_nz(SYSTEMATICS_MODEL)
                    zb = glass.shells.distance_grid(cosmo, zmin, zmax, dx=dx)
                    los_z_integration = np.linspace(zb[0], zb[-1], n_los_chi)
                    tomo_nz = calculate_tomo_nz(data_dir, n_los_chi, los_z_integration, shift_nz)
                    if SMOKE:
                        # "Very low n_eff": scale the real n(z) right down to keep the smoke fast.
                        tomo_nz = tomo_nz * SMOKE_CONFIG["n_eff_scale"]

                    vd = None
                    if SYSTEMATICS_MODEL == "nla_vd":
                        # Shell z-edges for the LOS variable-depth fraction. GLASS uses linear
                        # (overlapping) windows -> replicate the reference (zb[k], zb[k+1]) spacing;
                        # Gower uses top-hat windows -> each shell's own (z_near, z_far).
                        if SIMULATOR_TYPE == "gower_street":
                            zb_tuple = [(float(w.za[0]), float(w.za[-1])) for w in shells]
                        else:
                            zb_tuple = [(zb[k], zb[k + 1]) for k in range(len(zb) - 1)]
                        # tomo_nz is scaled down by n_eff_scale in the smoke; the per-VD-bin n(z)
                        # must share that scale (the LOS fraction is dndz_vd/tomo_nz). 1.0 otherwise.
                        dndz_scale = SMOKE_CONFIG["n_eff_scale"] if SMOKE else 1.0
                        var_depth_mask, vd_shapes, vd_map = build_variable_depth(
                            data_dir,
                            mask=mask,
                            tomo_nz=tomo_nz,
                            los_z_integration=los_z_integration,
                            zb_tuple=zb_tuple,
                            nside=nside,
                            sigma_e=sigma_e,
                            dndz_scale=dndz_scale,
                        )
                        # Per-sim realised VD shear biases (mirror m_bias_realised, using rng).
                        m_bias_vd_realised = np.array([
                            [float(rng.normal(m_bias_vd[i][j], m_bias_vd_unc[i][j]))
                             for j in range(n_vardepth_bins)]
                            for i in range(nbins)
                        ])
                        alpha_1_realised = np.array(
                            [float(rng.normal(alpha_1[i], alpha_1_unc[i])) for i in range(nbins)]
                        )
                        alpha_2_realised = np.array(
                            [float(rng.normal(alpha_2[i], alpha_2_unc[i])) for i in range(nbins)]
                        )
                        psf_bias_map_1, psf_bias_map_2 = load_psf_maps(data_dir, nside)
                        vd = {
                            "var_depth_mask": var_depth_mask,
                            "vd_shapes": vd_shapes,
                            "vd_map": vd_map,
                            "vd_trace_edges": vd_trace_edges,
                            "n_vardepth_bins": n_vardepth_bins,
                            "nside": nside,
                            "m_bias_vd_realised": m_bias_vd_realised,
                            "alpha_1_realised": alpha_1_realised,
                            "alpha_2_realised": alpha_2_realised,
                            "psf_bias_map_1": psf_bias_map_1,
                            "psf_bias_map_2": psf_bias_map_2,
                        }

                    systematics, sigma_e_sim, shift_nz = build_systematics(
                        SYSTEMATICS_MODEL,
                        systematics_cls=NLASystematics,
                        cosmo=cosmo,
                        ia_params=ia_params,
                        shear_bias={
                            "m_bias": m_bias_realised,
                            "c1_north": c1_bias_north_realised,
                            "c2_north": c2_bias_north_realised,
                            "c1_south": c1_bias_south_realised,
                            "c2_south": c2_bias_south_realised,
                        },
                        sigma_e_base=sigma_e,
                        vd=vd,
                    )
                    if SYSTEMATICS_MODEL == "none":
                        m_bias_realised *= 0.0

                    kwargs = {
                        'cosmo': cosmo,
                        'los_z_integration': los_z_integration,
                        'tomo_nz': tomo_nz,
                        'galaxy_bias': bias,
                        'sigma_e': sigma_e_sim,
                        'mask': mask,
                        'nside': nside,
                        'nbins': nbins,
                        'rng': rng,
                        'systematics': systematics,
                    }
                    print('Simulating the galaxy catalogue...')

                    simulator = GlassMatterShellSimulator(matter, shells, **kwargs)
                    catalogues = simulator.run(
                        rotation_spec,
                        mask_rotation_angles,
                        num_shape_noise_realisations=inner_reps,
                    )

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
                            catalogue, m_bias, nbins, nside, lmax, nosh=False, mask=mask,
                            normalization=SHEAR_NORMALIZATION,
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
                            alm_list=alm, nside_out=nside_out, lmax_out=None, fwhm_arcmin=8.0, taper_start_frac=0.95
                        )

                        # realised_unmixed_shear_cls, cll_bands, bandpowers = process_cls(mask_cls, nbins, nside, alm, alm_rand, lower_lscale, upper_lscale,lmin, lmax, nbands, )

                        # map_types = {"shear_real": shear.real, "shear_imag": shear.imag, "E":E, "B":B}
                        map_types = {"E":E, "B":B}
                        pixelised_results = {name:{} for name in map_types.keys()}
                        for name, cat_data in map_types.items():
                            pixelised_tomobin_patches = get_patch_values(cat_data, patches, nside_out, ang)
                            for patch_idx, patch_name in enumerate(named_patches.keys()):
                                pixelised_results[name][patch_name] = pixelised_tomobin_patches[patch_idx]

                        # cls_results['full'] = {"cls": mixed_cls, "mixed_bandpowers":mixed_bandpowers, "bandpower_ls":cll_bands}
                        cls_results['full'] = {"mixed_bandpowers":mixed_bandpowers, "bandpower_ls":cll_bands, "cls": mixed_cls[:, :, :2, :]}  # only save EE and BB

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