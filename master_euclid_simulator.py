"""MPI GLASS forward simulator for Euclid-DR3-like mocks (14 000 deg^2, 13 tomographic bins).

This is the Euclid sibling of `master_kids_legacy_simulator.py`. It reuses the SAME forward
model (`src/cosmology/`) and the same HDF5 output contract, and differs only in the survey
configuration, which lives in `src/Euclid/simulation_config.py`:

  * footprint: a full-RA declination band of 14 000 deg^2 (|Dec| <= 19.84 deg), not the KiDS mask
  * 13 tomographic bins from `KiDS_data/euclid_nzs_dr3.txt`, n = 6.2 gal/arcmin^2 in total
  * sigma_e = 0.26 per component, identical in every bin
  * NO survey systematics at all: no intrinsic alignments, no variable depth, no m/c shear bias,
    no photo-z shifts (`systematics=None` -> `NoSystematics`)
  * NO field-rotation augmentation (see "Rotations" below)
  * a single stored map product: the E maps at one (fwhm, lcut) variant, in float16

Everything expensive is shared with the KiDS path: the deterministic per-`sim_id` cosmology draw
from the Gower Street prior, the on-disk CAMB matter-Cls cache, the counts-normalised pseudo-Cl
bandpowers, and the idempotent file-resume.

Rotations
---------
The KiDS pipeline augments each cosmology with rotated footprints. That is a no-op here and is
switched off entirely:

  * The band is invariant under ANY RA (Z-axis) rotation, exactly (verified numerically).
  * A declination rotation cannot be undone by a Dec crop: a Y-axis rotation maps a constant-Dec
    annulus onto an annulus about a TILTED axis, so a full-RA band cannot be "recentred" on the
    equator. The origin/Euclid branch's `get_recentred_patch_values` attempted exactly that and
    silently lost ~50 % of the footprint. See
    `.claude/runs/euclid/dataset-generation/artifacts/rotation_bug.png`.

Independent realisations therefore come only from `--outer-reps` (fresh GLASS field + shape
noise per rep), not from rotations.

Memory
------
n = 6.2 gal/arcmin^2 over 14 000 deg^2 is ~3.1e8 galaxies, i.e. ~15 GB for one catalogue and
~30 GB across the `np.concatenate`. The augmentation loops inside `BaseSimulator.run` are all
collapsed to 1 (`mask_rotation_angles=[0]`, `num_shape_noise_realisations=1`) so exactly ONE
catalogue is ever resident. Budget ~40 GB per MPI rank and size `--ntasks-per-node` accordingly.
"""

import argparse
import gc
import time
from collections import deque
from pathlib import Path

import numpy as np
from mpi4py import MPI

import glass
import glass.shells

from src.cosmology.simulators import GlassMatterShellSimulator
from src.cosmology.sim_utils import (
    prepare_glass_backend,
    save_results_h5,
    count_block_files,
    remove_block_outputs,
    sim_is_complete,
)
from src.cosmology.manip_cls import compute_cl_bandpowers, denoise_shear_cls
from src.cosmology.map_shears import make_alm_shear_convergence, filter_EB_alms_and_make_maps
from src.cosmology.pixelise_maps import get_patch_values

from src.Euclid import simulation_config as cfg
from src.Euclid.tomo import build_tomo_nz


def parse_args():
    p = argparse.ArgumentParser(description="MPI GLASS Euclid DR3 mock simulator")

    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--nz-path", type=Path, default=Path("KiDS_data/euclid_nzs_dr3.txt"))
    p.add_argument(
        "--csv-path", type=Path,
        default=Path("/home/asaoulis/projects/glass_transfer/kids-legacy-sbi/data/gower_st/"
                     "PKDGRAV3_on_DiRAC_DES_330.csv"),
        help="Gower Street cosmology table backing the empirical cosmology prior (same as the "
             "KiDS master). w0 only -- wa is NOT sampled.")

    p.add_argument("--num-sims", type=int, default=cfg.EUCLID_N_JOBS,
                   help="Number of distinct cosmologies (sim_ids 0..N-1).")
    p.add_argument("--sim-id-offset", type=int, default=0,
                   help="Start sim_ids at this offset, to extend an existing store without "
                        "recomputing (cosmologies are keyed by sim_id).")
    p.add_argument("--outer-reps", type=int, default=cfg.OUTER_NUM_SHAPE_NOISE_REALISATIONS,
                   help="Independent GLASS field + shape-noise realisations per cosmology.")

    p.add_argument("--nside", type=int, default=cfg.nside)
    p.add_argument("--nside-out", type=int, default=cfg.nside_out)
    p.add_argument("--n-arcmin2-total", type=float, default=cfg.N_ARCMIN2_TOTAL)
    p.add_argument("--sigma-e", type=float, default=cfg.sigma_e_per_component,
                   help="Per-component intrinsic ellipticity dispersion (not the two-component "
                        "total), matching glass.shapes.ellipticity_intnorm's convention.")
    p.add_argument("--galaxy-bias", type=float, default=float(cfg.bias),
                   help="Linear source-galaxy clustering bias b_g fed to positions_from_delta. "
                        "Pass 0 for a clean theory test (no source clustering).")

    p.add_argument("--fwhm-arcmin", type=float, default=cfg.EB_FWHM_ARCMIN)
    p.add_argument("--lcut", type=int, default=cfg.EB_LCUT)
    p.add_argument("--lmin", type=int, default=cfg.EB_LMIN)
    p.add_argument("--map-dtype", default="float16", choices=["float16", "float32", "float64"])
    p.add_argument("--save-b-maps", action="store_true",
                   help="Also store the B maps (doubles the map bytes). Default: E only.")
    p.add_argument("--no-maps", action="store_true",
                   help="Bandpowers only -- skip the map branch entirely (theory validation).")

    p.add_argument("--full-sky", action="store_true",
                   help="VALIDATION ONLY: replace the band mask with full sky, so measured "
                        "pseudo-Cls can be compared to theory with no mask deconvolution.")

    p.add_argument("--camb-cache-dir", type=Path, default=cfg.CAMB_CLS_CACHE_DIR,
                   help="Shared on-disk CAMB matter-Cls cache, keyed by sim_id. MUST be distinct "
                        "from the KiDS cache (different z-grid).")
    p.add_argument("--cosmo-base-seed", type=int, default=cfg.COSMO_BASE_SEED)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Local pre-flight: tiny in-process CAMB/GLASS backend (src/smoke_sim.py) "
                        "at reduced nside with a scaled-down n(z). Plumbing validation only.")
    p.add_argument("--smoke-n-eff-scale", type=float, default=None)
    return p.parse_args()


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    t_start = time.time()

    args = comm.bcast(parse_args() if rank == 0 else None, root=0)

    OUTPUT_DIR = args.output_dir
    if rank == 0:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    SMOKE = args.smoke
    nside = args.nside
    lmax = 2 * nside
    nside_out = min(args.nside_out, nside)
    nbins = cfg.nbins
    lower_lscale, upper_lscale, nbands = cfg.lower_lscale, cfg.upper_lscale, cfg.nbands
    zmin, zmax, dx, n_los_chi = cfg.zmin, cfg.zmax, cfg.dx, cfg.n_los_chi
    map_dtype = np.dtype(args.map_dtype)
    expected_files_per_sim = args.outer_reps   # 1 identity rotation x 1 inner x 1 mask angle

    smoke_cfg = None
    if SMOKE:
        from src.smoke_sim import SMOKE_CONFIG, prepare_smoke_backend
        smoke_cfg = dict(SMOKE_CONFIG)
        # The Euclid n(z) runs to z ~ 3, so the smoke shells must cover it or the high-z bins
        # stay empty and the bandpower matrix is degenerate.
        smoke_cfg.update({"zmax": cfg.zmax, "nbins": nbins})
        if args.smoke_n_eff_scale is not None:
            smoke_cfg["n_eff_scale"] = float(args.smoke_n_eff_scale)
        nside = smoke_cfg["nside"]
        lmax = smoke_cfg["lmax"]
        nside_out = smoke_cfg["nside_out"]
        lower_lscale, upper_lscale = smoke_cfg["lower_lscale"], smoke_cfg["upper_lscale"]
        zmax, dx, n_los_chi = smoke_cfg["zmax"], smoke_cfg["dx"], smoke_cfg["n_los_chi"]

    # Footprint + Cartesian crop boxes. Both are pure geometry (see simulation_config).
    footprint = np.ones(12 * nside ** 2) if args.full_sky else cfg.build_footprint(nside).astype(float)
    patch_names = list(cfg.named_patches.keys())
    patches = [cfg.named_patches[n] for n in patch_names]

    # Deterministic per-sim_id cosmology sampler (Gower Street flow prior). Lazily imported:
    # it pulls in the ML/eval stack, which the smoke path does not need.
    cosmo_sampler = None
    if not SMOKE:
        from src.ml.eval.utils import build_cosmo_param_sampler
        cosmo_sampler = build_cosmo_param_sampler(cfg.COSMO_PARAM_NAMES, csv_path=str(args.csv_path))

    if rank == 0:
        n_lat, n_lon = cfg.patch_grid_shape(nside_out)
        print(f"[rank 0] Euclid mocks -> {OUTPUT_DIR}", flush=True)
        print(f"[rank 0]   band |Dec| <= {cfg.DEC_HALF_DEG:.4f} deg, f_sky = "
              f"{footprint.mean():.4f}, {nbins} bins, n_tot = {args.n_arcmin2_total} /arcmin^2",
              flush=True)
        print(f"[rank 0]   nside {nside}, lmax {lmax}, nside_out {nside_out}, patch grid "
              f"({n_lat}, {n_lon}) per half, map dtype {map_dtype.name}", flush=True)
        print(f"[rank 0]   {args.num_sims} cosmologies x {args.outer_reps} outer reps, "
              f"b_g = {args.galaxy_bias}, {size} MPI ranks", flush=True)

    # ---------------- distribute sim_ids ----------------
    if rank == 0:
        sim_samples = (np.arange(args.num_sims) + args.sim_id_offset).reshape(-1, 1).astype(np.float64)
        N = sim_samples.shape[0]
    else:
        sim_samples, N = None, None
    N = comm.bcast(N, root=0)

    counts_rows = [N // size + (1 if i < N % size else 0) for i in range(size)]
    displs_rows = np.concatenate([[0], np.cumsum(counts_rows)[:-1]]).astype(int)
    recvbuf = np.empty(counts_rows[rank], dtype=np.float64)
    comm.Scatterv(
        [sim_samples.flatten() if rank == 0 else None, counts_rows, displs_rows, MPI.DOUBLE],
        recvbuf, root=0,
    )
    sims = recvbuf.astype(int)
    print(f"[rank {rank}] {sims.size} sims: {sims[:5]}{'...' if sims.size > 5 else ''}", flush=True)

    backend_cache = {}

    for sim_num in sims:
        try:
            if (not args.overwrite) and sim_is_complete(OUTPUT_DIR, sim_num, expected_files_per_sim):
                print(f"[rank {rank}] sim {sim_num} already complete, skipping.", flush=True)
                continue

            for outer_idx in range(args.outer_reps):
                rot_idx = 0
                if not args.overwrite and count_block_files(OUTPUT_DIR, sim_num, outer_idx, rot_idx) >= 1:
                    continue
                remove_block_outputs(OUTPUT_DIR, sim_num, outer_idx, rot_idx)

                t_block = time.time()
                rng = np.random.default_rng()

                if SMOKE:
                    backend = prepare_smoke_backend(rng, smoke_cfg, ia_prior_spec={})
                else:
                    backend = prepare_glass_backend(
                        sim_num,
                        rng=rng,
                        cosmo_sampler=cosmo_sampler,
                        cosmo_base_seed=args.cosmo_base_seed,
                        cls_cache_dir=args.camb_cache_dir,
                        prior_ranges={},          # no IA nuisance parameters for Euclid
                        sim_grid={"nside": nside, "lmax": lmax},
                        los_grid={"zmin": zmin, "zmax": zmax, "dx": dx},
                        camb_limits=cfg.CAMB_LIMITS,
                        cache=backend_cache,
                    )

                param_dict = dict(backend["param_dict"])
                shells, matter, cosmo = backend["shells"], backend["matter"], backend["cosmo"]

                zb = glass.shells.distance_grid(cosmo, zmin, zmax, dx=dx)
                los_z_integration = np.linspace(zb[0], zb[-1], n_los_chi)
                tomo_nz, n_arcmin2 = build_tomo_nz(
                    args.nz_path, los_z_integration, args.n_arcmin2_total
                )
                if SMOKE:
                    tomo_nz = tomo_nz * smoke_cfg["n_eff_scale"]

                param_dict["galaxy_bias"] = float(args.galaxy_bias)

                simulator = GlassMatterShellSimulator(
                    matter, shells,
                    cosmo=cosmo,
                    los_z_integration=los_z_integration,
                    tomo_nz=tomo_nz,
                    galaxy_bias=args.galaxy_bias,
                    sigma_e=np.full(nbins, args.sigma_e),
                    mask=footprint,
                    nside=nside,
                    nbins=nbins,
                    rng=rng,
                    systematics=None,          # -> NoSystematics
                )
                print(f"[rank {rank}] sim {sim_num} out{outer_idx}: sampling catalogue...", flush=True)
                catalogues = simulator.run(
                    cfg.IDENTITY_ROTATION,
                    cfg.MASK_ROTATION_ANGLES,
                    num_shape_noise_realisations=cfg.INNER_NUM_SHAPE_NOISE_REALISATIONS,
                )

                cat_queue = deque(catalogues)
                del catalogues, matter
                gc.collect()

                m_bias = np.zeros(nbins)       # systematics off -> no multiplicative bias
                cat_idx = 0
                while cat_queue:
                    cat_parts = cat_queue.popleft()
                    try:
                        catalogue = np.concatenate(cat_parts)
                    finally:
                        del cat_parts
                    gc.collect()
                    print(f"[rank {rank}] sim {sim_num} out{outer_idx}: "
                          f"{catalogue.shape[0]:,} galaxies", flush=True)

                    alm, alm_rand = make_alm_shear_convergence(
                        catalogue, m_bias, nbins, nside, lmax, nosh=False, mask=footprint,
                        normalization="counts", rng=rng,
                    )
                    mixed_cls = denoise_shear_cls(nbins, alm, alm_rand, lmax)
                    upper = min(upper_lscale, lmax - 1)
                    cll_bands, mixed_bandpowers = compute_cl_bandpowers(
                        mixed_cls[:, :, :, lower_lscale:upper + 1],
                        nbins, lower_lscale, upper, nbands,
                    )
                    del catalogue, alm_rand, mixed_cls
                    gc.collect()

                    pixelised_results = {}
                    if not args.no_maps:
                        E, B = filter_EB_alms_and_make_maps(
                            alm_list=alm, nside_out=nside_out, lmax_out=None,
                            fwhm_arcmin=args.fwhm_arcmin, taper_start_frac=0.95,
                            lmin=args.lmin, lcut=args.lcut,
                        )
                        map_types = {"E": E} if not args.save_b_maps else {"E": E, "B": B}
                        del B
                        for name, maps in map_types.items():
                            crops = get_patch_values(maps, patches, nside_out, 0)
                            pixelised_results[name] = {
                                pn: np.asarray(crops[i]).astype(map_dtype, copy=False)
                                for i, pn in enumerate(patch_names)
                            }
                            for pn, arr in pixelised_results[name].items():
                                if not np.isfinite(arr).all():
                                    print(f"[rank {rank}] WARNING non-finite in {name}/{pn} "
                                          f"after the {map_dtype.name} cast", flush=True)
                            del crops
                        del map_types, E
                    del alm
                    gc.collect()

                    save_results_h5(
                        OUTPUT_DIR / f"output_{sim_num}_out{outer_idx}_rot{rot_idx}.h5",
                        cat_idx,
                        {"full": {"mixed_bandpowers": mixed_bandpowers, "bandpower_ls": cll_bands}},
                        pixelised_results,
                        param_dict,
                    )
                    del pixelised_results, mixed_bandpowers, cll_bands
                    gc.collect()
                    cat_idx += 1

                del cat_queue, simulator, backend
                gc.collect()
                print(f"[rank {rank}] sim {sim_num} out{outer_idx} done in "
                      f"{time.time() - t_block:.1f} s", flush=True)
        except Exception as exc:
            # Per-sim isolation, as in the KiDS master: one bad sim must not abort the rank.
            import traceback
            print(f"[rank {rank}] ERROR on sim {sim_num}: {exc}\n{traceback.format_exc()}",
                  flush=True)

    print(f"[rank {rank}] finished in {time.time() - t_start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
