import numpy as np
import healpy as hp
from pathlib import Path
from collections import deque
import gc
import argparse

import glass.ext.camb

from mpi4py import MPI

import glass
import glass.shells

import time
from src.cosmology.simulators import GlassMatterShellSimulator
from src.cosmology.gower_street import GowerStPrior
from src.cosmology.sim_utils import (
    prepare_glass_backend,
    save_results_h5,
)

from src.cosmology.manip_cls import compute_cl_bandpowers, denoise_shear_cls
from src.cosmology.pixelise_maps import get_recentred_patch_values

from src.cosmology.map_shears import make_alm_shear_convergence, filter_EB_alms_and_make_maps
from src.KiDS.rotations import KiDS_PATCH_GLASS_ROTATIONS

# --------------------------------------------------------------------------
# Euclid DR3 simulation config (README "Euclid DR3 details"). nside/n_arcmin2
# /outer-realisations are also exposed as CLI flags (defaults below) so this
# script can be smoke-tested cheaply without editing the file.
# --------------------------------------------------------------------------
NBINS = 13
GALAXY_BIAS = 1
SIGMA_E = 0.26  # README: ellipticity dispersion

ZMIN, ZMAX, DX = 0.0, 3.05, 200.0  # zmax must cover euclid_nzs_dr3.txt's support (z up to 3.0)
N_LOS_CHI = 1000

CAMB_LIMITS = {"mem_limit_gb": 200, "timeout_s": 3600 * 4}

# The equatorial-band footprint is RA-symmetric, so a longitude rotation of
# the mask is a no-op augmentation-wise (unlike KiDS's RA-limited patches).
MASK_ROTATION_ANGLES = [0]
# The GLASS backend uses no field-level rotation at all (identity), matching
# KiDS_PATCH_GLASS_ROTATIONS - reused directly since it's backend-generic.
ROTATION_SPECS = KiDS_PATCH_GLASS_ROTATIONS
NUM_SHAPE_NOISE_REALISATIONS = 1  # matches KiDS's INNER_NUM_SHAPE_NOISE_REALISATIONS["glass"]

# Cl bandpower binning - reused from src/KiDS/simulation_config.py's values
LOWER_LSCALE = 76
UPPER_LSCALE = 1500
NBANDS = 8

OVERWRITE = False

# Footprint: FOOTPRINT_AREA_DEG2 equatorial Dec-band, split at Dec=0 into two
# equal halves. Each half's center declination is recentred to the equator
# (via get_recentred_patch_values) before being cropped into a flat Cartesian
# patch - unlike KiDS's named_patches, which are fixed lon/lat crops never
# actually recentred (see src/KiDS/simulation_config.py).
FOOTPRINT_AREA_DEG2 = 14000.0


def dec_half_width_deg(area_deg2):
    """Half-width (deg) of a symmetric, full-RA Dec-band with the given area."""
    area_sr = area_deg2 * np.deg2rad(1.0) ** 2
    return np.rad2deg(np.arcsin(area_sr / (4 * np.pi)))


def build_footprint(mask_nside, dec_half_deg):
    theta_min = np.deg2rad(90.0 - dec_half_deg)
    theta_max = np.deg2rad(90.0 + dec_half_deg)
    footprint = np.zeros(hp.nside2npix(mask_nside), dtype=bool)
    footprint[hp.query_strip(mask_nside, theta_min, theta_max)] = True
    return footprint


def build_tomo_nz(nz_path, n_los_chi, los_z_integration, n_arcmin2_total):
    data = np.loadtxt(nz_path)
    z = data[:, 0]
    nz = data[:, 1:]
    nbins = nz.shape[1]

    raw_integrals = np.trapezoid(nz, z, axis=0)
    n_arcmin2 = n_arcmin2_total * raw_integrals / raw_integrals.sum()

    tomo_nz = np.zeros((nbins, n_los_chi))
    for i in range(nbins):
        dndz_interpolated = np.interp(
            los_z_integration, z, n_arcmin2[i] * nz[:, i] / raw_integrals[i]
        )
        tomo_nz[i] = np.clip(dndz_interpolated, 0, None)
    return tomo_nz


def parse_args():
    parser = argparse.ArgumentParser(description="MPI GLASS Euclid DR3 simulation")

    parser.add_argument("--csv-path", type=Path,
                         default=Path("KiDS_data/KiDS_GLASS_priors.csv"))

    parser.add_argument("--nz-path", type=Path, default=Path("KiDS_data/euclid_nzs_dr3.txt"))

    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--num-sims", type=int, default=1000,
                         help="Number of distinct cosmology draws to simulate")

    parser.add_argument("--nside", type=int, default=1024,
                         help="HEALPix nside for the simulation and footprint (lower for a smoke test; "
                              "needs nside >= ~128 so lmax exceeds denoise_shear_cls's hardcoded ell>=200 "
                              "noise-floor slice, src/cosmology/manip_cls.py)")

    parser.add_argument("--n-arcmin2-total", type=float, default=6.2,
                         help="Total galaxy number density, README default 6.2/arcmin^2 (lower for a smoke test)")

    parser.add_argument("--outer-realisations", type=int, default=4,
                         help="Independent GRF/shape-noise draws per sampled cosmology")

    return parser.parse_args()


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    s = time.time()

    # ------------------ distribute sim_samples using Scatterv (robust) ------------------
    if rank == 0:
        args = parse_args()
        sim_samples = np.arange(args.num_sims).reshape(-1, 1).astype(np.float64)
        cols = sim_samples.shape[1]
        N = sim_samples.shape[0]
    else:
        sim_samples = None
        cols = None
        N = None
        args = None
    args = comm.bcast(args, root=0)

    csv_path = args.csv_path
    nz_path = args.nz_path
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    NSIDE = args.nside
    LMAX = 2 * NSIDE
    N_ARCMIN2_TOTAL = args.n_arcmin2_total
    OUTER_REALISATIONS = args.outer_realisations

    SIM_GRID = {"nside": NSIDE, "lmax": LMAX}
    LOS_GRID = {"zmin": ZMIN, "zmax": ZMAX, "dx": DX}

    dec_half_deg = dec_half_width_deg(FOOTPRINT_AREA_DEG2)
    patch_halfwidth_deg = dec_half_deg / 2  # each north/south half's Dec half-width
    patch_center_dec_deg = {"north": patch_halfwidth_deg, "south": -patch_halfwidth_deg}

    gower_prior = GowerStPrior.from_csv(csv_path, drop_first=192)
    nuisance_prior = {"a_ia": (4.48, 7.0), "b_ia": (0.28, 0.6)}

    backend_cache = {}

    footprint = build_footprint(NSIDE, dec_half_deg)

    cols = comm.bcast(cols, root=0)
    N = comm.bcast(N, root=0)

    if rank == 0:
        counts_rows = [N // size] * size
        rem = N % size
        for i in range(rem):
            counts_rows[i] += 1
    else:
        counts_rows = None
    counts_rows = comm.bcast(counts_rows, root=0)

    displs_rows = [0] * size
    for i in range(1, size):
        displs_rows[i] = displs_rows[i - 1] + counts_rows[i - 1]

    counts = [r * cols for r in counts_rows]
    displs = [d * cols for d in displs_rows]

    recv_elems = counts_rows[rank] * cols
    recvbuf_flat = np.empty(recv_elems, dtype=np.float64)
    sendbuf_flat = sim_samples.flatten() if rank == 0 else None

    comm.Scatterv([sendbuf_flat, counts, displs, MPI.DOUBLE], recvbuf_flat, root=0)

    recvbuf = recvbuf_flat.reshape(-1, cols) if cols > 0 else np.empty((0, cols), dtype=np.float64)
    print(f"[rank {rank}] received chunk (shape {recvbuf.shape}):\n{recvbuf}")

    sims = recvbuf[:, -1].astype(int) if recvbuf.size else np.array([], dtype=int)

    try:
        for num_sim_this_batch in range(len(recvbuf)):
            sim_num = sims[num_sim_this_batch]
            path_glob_pattern = f"output_{sim_num}*.h5"
            existing_files = list(OUTPUT_DIR.glob(path_glob_pattern))
            if (not OVERWRITE) and len(existing_files) > 0:
                print(f"[rank {rank}] Skipping sim {sim_num} as output files already exist.")
                continue

            for outer_idx in range(OUTER_REALISATIONS):
                rotation_spec = ROTATION_SPECS[0]
                rng = np.random.default_rng()

                s_catalogue = time.time()
                print("Setting up simulator...")

                # Cosmology (+ a_ia/b_ia nuisance draws) is sampled once per
                # sim_num and cached inside prepare_glass_backend; the outer
                # loop reuses that same cosmology for independent GRF/shape
                # noise realisations.
                backend = prepare_glass_backend(
                    sim_num,
                    rng=rng,
                    output_dir=OUTPUT_DIR,
                    cosmo_prior=gower_prior,
                    prior_ranges=nuisance_prior,
                    sim_grid=SIM_GRID,
                    los_grid=LOS_GRID,
                    camb_limits=CAMB_LIMITS,
                    cache=backend_cache,
                )
                param_dict = backend["param_dict"]
                shells = backend["shells"]
                matter = backend["matter"]
                cosmo = backend["cosmo"]

                zb = glass.shells.distance_grid(cosmo, ZMIN, ZMAX, dx=DX)
                los_z_integration = np.linspace(zb[0], zb[-1], N_LOS_CHI)
                tomo_nz = build_tomo_nz(nz_path, N_LOS_CHI, los_z_integration, N_ARCMIN2_TOTAL)

                kwargs = {
                    "cosmo": cosmo,
                    "los_z_integration": los_z_integration,
                    "tomo_nz": tomo_nz,
                    "galaxy_bias": GALAXY_BIAS,
                    "sigma_e": np.full(NBINS, SIGMA_E),
                    "mask": footprint.astype(float),
                    "nside": NSIDE,
                    "nbins": NBINS,
                    "rng": rng,
                    "systematics": None,
                }
                print("Simulating the galaxy catalogue...")

                simulator = GlassMatterShellSimulator(matter, shells, **kwargs)
                catalogues = simulator.run(
                    rotation_spec,
                    MASK_ROTATION_ANGLES,
                    num_shape_noise_realisations=NUM_SHAPE_NOISE_REALISATIONS,
                )

                print(f"Simulated the galaxy catalogue in {time.time() - s_catalogue:.2f} seconds")
                print("Calculating the shear power spectra...")

                cat_queue = deque(catalogues)
                del catalogues
                gc.collect()

                lmax_maps = simulator.lmax  # 2*NSIDE - 1
                m_bias = np.zeros(NBINS)  # no shear bias applied (systematics=None)

                cat_idx = 0
                while cat_queue:
                    cat_parts = cat_queue.popleft()
                    try:
                        catalogue = np.concatenate(cat_parts)
                    finally:
                        del cat_parts

                    cls_results = {"full": {}}

                    alm, alm_rand = make_alm_shear_convergence(
                        catalogue, m_bias, NBINS, NSIDE, lmax_maps, nosh=False, mask=footprint.astype(float)
                    )
                    mixed_cls = denoise_shear_cls(NBINS, alm, alm_rand, lmax_maps)
                    # LOWER_LSCALE/UPPER_LSCALE are only valid as-is at production
                    # nside=1024 (lmax~2048); clamp upper so a low --nside smoke
                    # test doesn't index past lmax_maps.
                    lower_lscale = LOWER_LSCALE
                    upper_lscale = min(UPPER_LSCALE, lmax_maps - 1)
                    mixed_cut = mixed_cls[:, :, :, lower_lscale:upper_lscale + 1]
                    cll_bands, mixed_bandpowers = compute_cl_bandpowers(
                        mixed_cut, NBINS, lower_lscale, upper_lscale, NBANDS
                    )

                    del catalogue
                    gc.collect()

                    nside_out = min(512, NSIDE)
                    E, B = filter_EB_alms_and_make_maps(
                        alm_list=alm, nside_out=nside_out, lmax_out=None, fwhm_arcmin=8.0, taper_start_frac=0.95
                    )
                    lmax_out = min(lmax_maps, 3 * nside_out - 1, 1500)

                    map_types = {"E": E, "B": B}
                    pixelised_results = {name: {} for name in map_types.keys()}
                    for name, cat_data in map_types.items():
                        for patch_name, center_dec_deg in patch_center_dec_deg.items():
                            pixelised_results[name][patch_name] = get_recentred_patch_values(
                                cat_data, center_dec_deg, patch_halfwidth_deg, nside_out, lmax_out,
                            )

                    cls_results["full"] = {
                        "mixed_bandpowers": mixed_bandpowers,
                        "bandpower_ls": cll_bands,
                        "cls": mixed_cls[:, :, :2, :],  # only save EE and BB
                    }

                    save_string = f"{sim_num}_out{outer_idx}"
                    save_results_h5(
                        OUTPUT_DIR / f"output_{save_string}.h5", cat_idx, cls_results, pixelised_results, param_dict
                    )

                    del cls_results, pixelised_results, map_types, E, B, alm, alm_rand
                    gc.collect()

                    cat_idx += 1

                del cat_queue, simulator
                gc.collect()
                print(f"Finished outer realisation {outer_idx} in {time.time() - s_catalogue:.2f} seconds")

            print(f"Saved results for sim {sim_num}")

        print(f"Entire simulation took {time.time() - s:.2f} seconds")
    except Exception as e:
        print(f"[rank {rank}] Error processing sims: {e}")
        # Don't re-raise to avoid MPI hang; just log the error and continue
