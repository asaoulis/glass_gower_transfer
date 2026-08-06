import numpy as np
import healpy as hp
import h5py
from pathlib import Path
from collections import deque
import gc
import argparse
import traceback

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
    count_block_files,
    model_shift_nz,
    prepare_glass_backend,
    prepare_gower_backend,
    remove_block_outputs,
    resolve_systematics_model,
    save_results_h5,
    sim_is_complete,
)

from src.cosmology.manip_cls import compute_cl_bandpowers, denoise_shear_cls
from src.cosmology.pixelise_maps import get_patch_values

from src.cosmology.map_shears  import make_alm_shear_convergence, filter_EB_alms_and_make_maps
from src.KiDS.tomo import calculate_tomo_nz
from src.KiDS.rotations import KiDS_PATCH_GOWER_ROTATIONS, KiDS_PATCH_GLASS_ROTATIONS
from src.KiDS.simulation_config import (
    CAMB_CLS_CACHE_DIR,
    CAMB_LIMITS,
    COSMO_BASE_SEED,
    INNER_NUM_SHAPE_NOISE_REALISATIONS,
    LOS_GRID,
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

    parser.add_argument(
        "--ia-model",
        type=str,
        default="nla_m",
        choices=["nla_m", "nla", "nla_z", "tatt"],
        help="Intrinsic-alignment model (default: nla_m). 'nla' = single-amplitude NLA; "
             "'nla_z' = NLA with linear redshift-dependent amplitude; 'tatt' = restricted "
             "TATT / NLA-k (NLA + density weighting). Orthogonal to --systematics-model.",
    )

    parser.add_argument("--no-rotations", action="store_true",
                        help="Disable rotations")

    parser.add_argument("--overwrite", action="store_true",
                        help="Force a clean regen: recompute every sim even if complete outputs "
                             "already exist. Default OFF = resume (skip already-complete sims and "
                             "(outer,rot) blocks, recompute only what is missing/incomplete).")

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
                             "(default 'counts' = per-pixel observed counts, DES-Y3-style: cancels "
                             "the source-clustering (1+b_g*delta) leakage into the maps at first "
                             "order — see .claude/runs/eval-and-viz/investigate-galaxy-bias-issue. "
                             "'mean' = global mean counts/pixel (Hall & Tessore; former default): "
                             "matches the KiDS MCM/theory to ~few %% for bandpowers but is "
                             "first-order sensitive to source clustering; the 'counts' pseudo-Cl "
                             "offset vs the fractional-mask MCM (~1.5x) needs re-validation.)")

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

    # Shared, persistent CAMB-Cls cache (separate from --output-dir) so analysis variates reuse
    # the expensive Cls; and the base seed fixing the per-sim_id cosmology across variates.
    parser.add_argument("--camb-cache-dir", type=Path, default=CAMB_CLS_CACHE_DIR,
                        help="Shared on-disk cache for CAMB-computed matter Cls, keyed by sim_id.")

    parser.add_argument("--cosmo-base-seed", type=int, default=COSMO_BASE_SEED,
                        help="Base seed for deterministic per-sim_id cosmology sampling.")

    parser.add_argument(
        "--galaxy-bias", type=float, default=None,
        help="Override the linear source-galaxy clustering bias b_g fed to "
             "positions_from_delta (default: the simulation_config `bias` constant when "
             "systematics are ON). Only takes effect on the systematics-ON path; the clean "
             "theory-test path (systematics OFF) always uses galaxy_bias=0. Use to generate "
             "clustering variates, e.g. --galaxy-bias 1.5 (strong) / 0.7 (weak).")

    parser.add_argument(
        "--rng-seed", type=int, default=None,
        help="FIXED-RNG (paired-catalogue) mode. Default None = today's behaviour: one fresh, "
             "UNSEEDED generator per (sim, outer, rot) block. When set, every random stream is "
             "derived deterministically from (this seed, sim_id, outer_idx, rot_idx) AND the "
             "matter-field stream is split from the galaxy-sampling stream (see build_block_rngs). "
             "Two runs sharing --rng-seed then produce the SAME matter fields and the SAME "
             "nuisance draws, so a variate that only changes the galaxy sampling (e.g. "
             "--galaxy-bias 0.5 vs 1.5) is PAIRED with its reference: cosmic variance and "
             "nuisance scatter cancel exactly. Note the galaxy catalogues themselves cannot be "
             "identical when b_g differs (positions_from_delta draws a different number of "
             "galaxies) — only the fields and the nuisance realisation are shared.")

    parser.add_argument(
        "--save-catalogues", action="store_true",
        help="ALSO dump each mock's raw galaxy catalogue (RA, DEC, Z_TRUE, ZBIN, E1, E2) to "
             "<output-dir>/catalogues/catalogue_<sim>_out<o>_rot<r>_<i>.h5, so alternative shear "
             "estimators can be replayed offline (e.g. testing which normalisation is hardened "
             "against source clustering). Columns are downcast to float32/int8 (~21 B/galaxy). "
             "EXPENSIVE: a production KiDS-Legacy footprint has ~4e7 galaxies => ~0.9 GB PER MOCK, "
             "so pair this with a small --num-sims and --no-augmentation 1.")

    parser.add_argument(
        "--smoke-n-eff-scale", type=float, default=None,
        help="SMOKE ONLY: override SMOKE_CONFIG['n_eff_scale'] (default None = keep the config "
             "value, 1e-3). The default smoke is pure-shape-noise (~30k galaxies, lambda<<1), so "
             "counts-normalisation physics (1/N convexity, b_g clustering modulation) is absent; "
             "~0.0625 matches nside-256 pixel occupancy to production's ~13-21 gal/pixel "
             "(~1.9M galaxies, ~2 min/mock). No effect on non-smoke runs.")

    parser.add_argument("--outer-reps", type=int, default=None,
                        help="Override the per-sim OUTER shape-noise realisation count (default: the "
                             "OUTER_NUM_SHAPE_NOISE_REALISATIONS[simulator] config value; glass=4, "
                             "gower=4). Used to produce smaller datasets, e.g. --outer-reps 1 on GLASS "
                             "gives 1 outer x rotations x inner x mask files/sim. Ignored under --smoke "
                             "and --no-augmentation, which already force outer_reps=1.")

    parser.add_argument("--gower-sim-set", type=str, default="full",
                        choices=["full", "fixed_test"],
                        help="gower_street sim_id selection: 'full' = np.arange(193,782) (589 ids); "
                             "'fixed_test' = the committed 200-id lock-file "
                             "config/fixed_test_sets/gower_test_ids.json. Ignored for glass/smoke.")

    return parser.parse_args()

GLASS_N_JOBS = 6500

# Cosmology parameters drawn (deterministically) per sim_id and fed to CAMB. These are exactly the
# keys the on-disk Cls cache guard compares (see src/cosmology/mpi_camb.COSMO_GUARD_KEYS).
COSMO_PARAM_NAMES = ["omega_m", "sigma_8", "ombh2", "h", "ns", "w0", "mnu"]

# E/B map smoothing variants saved per mock: a list of (fwhm_arcmin, lmin, lcut) triples. Each
# triple produces one set of pixelised E/B maps, stored under keys
# E_fwhm{fwhm}[_lmin{lmin}][_lcut{lcut}] / B_... . `lmin` / `lcut` are hard top-hats (lower /
# upper ell-cut) applied after the Gaussian beam + cosine taper inside
# filter_EB_alms_and_make_maps: lmin zeros all ell < lmin, lcut zeros all ell > lcut (a
# Jeffrey-et-al-2025-style hard scale band). Either may be None to disable that edge (both None
# -> smoothing-only, keyed E_fwhm{fwhm}). The first entry reproduces the previous production map;
# append triples to ALSO save lighter smoothings / hard-cut band variants, e.g.
# [(8.0, None, None), (6.0, None, 1024), (6.0, 56, 1024)].
EB_SMOOTHING_VARIANTS = [(4.0, 56, 1400,), (8.0, 56, 1400), (8.0, 56, 1024)]

# --- Shear-estimator hardening: the dual-normalisation SUPERSET store -------------------------
# Source-galaxy clustering (b_g) modulates the per-pixel galaxy count N_p, and the counts
# normalisation S_p/N_p therefore leaks it into the map branch as a noise-AMPLITUDE channel
# (a b_g 1.0 -> 1.5 change moves the stored E amplitude by ~+7 sigma). See the measured
# leaderboard in .claude/runs/kids-preparation/improved-shear-processing/artifacts/RESULTS.md.
#
# One generation run writes a superset from which FIVE estimators can be trained with no
# re-simulation:
#   A0_counts   the stored E_<tag> maps, unmodified                      (baseline)
#   A1_wht_rand E_<tag> / noise_std_<tag>                                (DEPLOYED)
#   B1_selfstd  E_<tag> standardised per bin at load time                (eb_noise_norm='self')
#   A3s8        the stored E_sc8_<tag> maps, unmodified
#   A3s8_A1     E_sc8_<tag> / noise_std_sc8_<tag>
# What each arm actually costs, so nobody re-derives this the hard way:
#   B1 is the ONLY arm that is genuinely retrofittable. It is a pure function of the stored E
#      maps (a per-bin standardisation applied by the loader), so it runs on ANY existing store
#      with no regeneration at all -- config knob eb_noise_norm='self'.
#   A1 is NOT retrofittable, despite reading "E / scalar". The scalar is the std of the matched
#      RANDOM-ROTATION noise map -- an independent realisation that is not recoverable from the
#      stored E maps. It costs a regeneration; what it does NOT cost is a second map product.
#   A3s8 is not retrofittable either, and is the expensive one: it replaces the per-pixel
#      DENOMINATOR before the spin-2 SHT.
# Measured per-mock map-stage cost at production geometry (2026-08-06, jobs 1342042/1342078):
#   baseline 120 s | + A1 scalars (1 variant) ~49 s | + the other 2 variants ~98 s
#   | + the whole A3s8 branch ~215 s  =>  480 s as configured here.
# Against ~728 s/mock of amortised shell time that is +6 % (A1 alone) vs +42 % (full store).
# The A3s8 share buys ONE thing: a hedge against SPATIALLY STRUCTURED misspecification (variable
# depth), which a single per-mock scalar cannot absorb. Kept deliberately (user, 2026-08-06). It is carried for robustness option value
# against SPATIALLY STRUCTURED misspecification (variable depth), which a single per-mock
# scalar cannot absorb -- see artifacts/DUAL_STORE_A3S8.md.
#
# The BANDPOWER branch is deliberately untouched: it keeps consuming the primary `counts` alms,
# so the MCM/mask convention and src/validation/ are unaffected (per-observable normalisation).

# EB variant the A3s8 branch is built for, as a (fwhm_arcmin, lmin, lcut) triple that MUST be one
# of EB_SMOOTHING_VARIANTS -- training reads exactly one variant, so only that one is worth the
# +2 MB/mock. None disables the whole second branch (byte-identical legacy output).
A3S8_VARIANT = (4.0, 56, 1400)
# FWHM (arcmin) the count map is smoothed at before being used as the denominator. 8' is harness
# candidate A3_smooth8 / A3s8_A1; do NOT change it without re-running the replay harness.
A3S8_FWHM_ARCMIN = 8.0
# The A3s8 maps are an ML-only derived product (never a science archive) and the loader casts on
# read anyway, so float16 halves the on-disk delta to ~+1.95 MB/mock (+8 %).
A3S8_MAP_DTYPE = np.float16

# Per-(variant, bin, patch) standard deviations of the RANDOM-ROTATION noise map, stored as
# scalars (~288 B/mock) so the A1 rescale is a loader/prebake-side division and stays ablatable.
#   None  -> compute for every EB_SMOOTHING_VARIANTS entry (full optionality, ~+12.5 s/variant)
#   list  -> compute only for these (fwhm, lmin, lcut) triples
#   ()    -> disable (no noise_std groups written; the A1 arm becomes untrainable)
NOISE_STD_VARIANTS = None


def eb_variant_tag(fwhm_v, lmin_v, lcut_v):
    """HDF5 key suffix for one (fwhm, lmin, lcut) smoothing variant."""
    return (f"fwhm{fwhm_v:g}"
            + ("" if lmin_v is None else f"_lmin{int(lmin_v)}")
            + ("" if lcut_v is None else f"_lcut{int(lcut_v)}"))


def patch_noise_std(rand_E_maps, patches, nside_out, ang, patch_names):
    """Per-(patch, bin) std of the filtered random-rotation E map, + the pooled 'all'.

    The random-rotation map is a pure shape-noise realisation of the SAME galaxies with the
    SAME per-pixel counts, so its std is a per-mock meter of the noise amplitude that the
    counts normalisation makes b_g-dependent. Dividing the stored E map by it is the deployed
    A1_wht_rand estimator.

    The std is taken over ALL pixels of each stored patch grid (not the galaxy-occupied
    footprint the offline harness uses) so that it is reproducible from the stored patches
    alone. The two differ by a geometric factor that is fixed across mocks because the mask
    and patch geometry are fixed -- harness candidates A1_patch / A3s8_A1_patch verify this.
    """
    per_patch = get_patch_values(rand_E_maps, patches, nside_out, ang)
    out = {}
    flat = []
    for patch_idx, patch_name in enumerate(patch_names):
        p = np.asarray(per_patch[patch_idx], dtype=np.float64)   # (nbins, H, W)
        out[patch_name] = p.reshape(p.shape[0], -1).std(axis=1)
        flat.append(p.reshape(p.shape[0], -1))
    out["all"] = np.concatenate(flat, axis=1).std(axis=1)
    return out

# Per-IA-model forward-sampling priors. Each entry maps a parameter to a distribution spec
# ("uniform", lo, hi) or ("normal", mu, sigma); sampled per mock by sim_utils.sample_ia_params.
# nla_m: Fortuna et al. 2025 / Wright et al. 2025 (covers their posterior at 5 sigma).
# nla / nla_z / tatt: Wright et al. 2025 priors (A_IA ~ U[-6,6]; B_IA ~ N(-3.7,4.3); b_src ~ U[-0.5,1.5]).
IA_PRIOR_SPECS = {
    'nla_m': {'a_ia': ('uniform', 4.48, 7.0), 'b_ia': ('uniform', 0.28, 0.6)},
    'nla':   {'a_ia': ('uniform', -6.0, 6.0)},
    'nla_z': {'a_ia': ('uniform', -6.0, 6.0), 'b_z': ('normal', -3.7, 4.3)},
    'tatt':  {'a_ia': ('uniform', -6.0, 6.0), 'b_src': ('uniform', -0.5, 1.5)},
}


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

# Fixed seed for the `--gower-sim-set fixed_test --num-sims N (>200)` random top-up (below): the
# 200 fixed-test ids are always kept and the extra (N-200) sim_ids are drawn WITHOUT replacement
# from the Gower complement using this seed, so a resume / re-submit reproduces the SAME cosmology
# set (the sim resumes by skipping complete sim_ids, so the id list must be stable across runs).
GOWER_TOPUP_SEED = 20260710


# --- FIXED-RNG (paired-catalogue) mode -------------------------------------------------------
# Stream tags spawned off the per-block SeedSequence. Order/identity is part of the on-disk
# contract: changing them changes every mock generated with a given --rng-seed.
RNG_STREAM_SAMPLE = 0   # nuisance draws + galaxy sampling / shape noise (Simulator rng)
RNG_STREAM_BACKEND = 1  # IA-prior draws + the GLASS log-normal matter fields
RNG_STREAM_POSTPROC = 2  # random ellipticity rotation for the noise-only alm (map_shears)
RNG_STREAM_LEGACY = 3   # seeds the GLOBAL numpy RNG (see build_block_rngs)


def build_block_rngs(seed, sim_num, outer_idx, rot_idx):
    """Return ``(sample_rng, backend_rng, postproc_rng)`` for one (sim, outer, rot) block.

    With ``seed is None`` (the default) all three are INDEPENDENT UNSEEDED generators, which
    reproduces the historical behaviour bit-for-bit *in distribution* — the old code used one
    shared unseeded generator, but with no seed there is nothing to preserve, and splitting the
    streams is what makes the fixed-seed path meaningful.

    With an integer ``seed`` the three streams are spawned from
    ``SeedSequence([seed, sim_num, outer_idx, rot_idx])``, so they are:

    * **reproducible** — re-running the same block reproduces the same mock byte-for-byte;
    * **independent** — crucially, the matter-field stream (``backend_rng``, consumed lazily by
      ``glass.generate`` inside the shell loop) no longer shares a generator with the galaxy
      sampling (``sample_rng``, consumed by ``positions_from_delta`` / ``redshifts`` /
      ``sample_ellipticity``). The galaxy draws depend on ``galaxy_bias``, so under the old
      single-generator scheme the stream de-synchronised after the first shell and every later
      shell's delta differed between b_g variants. Split, the delta shells are IDENTICAL across
      variants that only change the galaxy sampling => a paired (cosmic-variance-free) test.

    Note ``prepare_glass_backend`` draws the IA nuisance params from ``backend_rng`` *before*
    generating the fields, but only on an in-memory cache MISS. The caller therefore hands it a
    fresh per-block cache in fixed-RNG mode so every block consumes the identical sequence
    (IA draws -> field draws) regardless of resume state or rank layout.

    SIDE EFFECT (seeded mode only): also seeds the **global** legacy numpy RNG from a fourth
    spawned stream. Some physics draws are not reachable through any ``rng`` argument — notably
    the per-mock photo-z shift ``np.random.multivariate_normal`` in ``src/KiDS/tomo.py`` (protected
    code, so it is seeded rather than re-plumbed). That draw perturbs ``tomo_nz`` and therefore
    EVERY downstream product, so leaving it unseeded makes two same-seed runs fully uncorrelated.
    Seeding it here also covers any future global-RNG use on the sim path.
    """
    if seed is None:
        return (np.random.default_rng(), np.random.default_rng(), np.random.default_rng())
    ss = np.random.SeedSequence([int(seed), int(sim_num), int(outer_idx), int(rot_idx)])
    children = ss.spawn(4)
    np.random.seed(int(children[RNG_STREAM_LEGACY].generate_state(1, dtype=np.uint32)[0]))
    return (
        np.random.default_rng(children[RNG_STREAM_SAMPLE]),
        np.random.default_rng(children[RNG_STREAM_BACKEND]),
        np.random.default_rng(children[RNG_STREAM_POSTPROC]),
    )


# --- Raw galaxy-catalogue dumps --------------------------------------------------------------
# Column dtypes for --save-catalogues. The catalogue is the LARGEST object in the pipeline
# (~4e7 galaxies for a KiDS-Legacy footprint at the production n_eff), so the float64 working
# array is downcast on write: 48 B/galaxy -> 21 B/galaxy, i.e. ~0.9 GB per mock instead of ~2 GB.
# float32 keeps ~7 significant digits, far beyond the precision of any of these quantities
# (positions are degrees, ellipticities are O(0.3) with O(1e-3) shear on top).
CATALOGUE_DTYPES = {
    "RA": np.float32, "DEC": np.float32, "Z_TRUE": np.float32,
    "ZBIN": np.int8, "E1": np.float32, "E2": np.float32,
}


def save_catalogue_h5(path, catalogue, cosmo_dict, extra_attrs=None):
    """Write ONE mock's raw galaxy catalogue next to (not inside) its output_*.h5.

    Rationale: the per-mock shear maps are already the *output* of a particular estimator
    (counts-normalised pseudo-Cl / pixelised E-B). To ask whether a DIFFERENT estimator is
    hardened against source clustering, you need the galaxies themselves — position, tomographic
    bin and observed ellipticity — so the map-making can be replayed offline at will.

    Written to a ``catalogues/`` SUBDIR of the output dir so the ``output_*.h5`` globs that drive
    every dataset config are untouched, and so a catalogue run can be deleted independently.

    ``extra_attrs`` records what the catalogue cannot: the realised multiplicative shear bias
    (``make_alm_shear_convergence`` applies the 1/(1+m) de-bias, so replaying the maps needs it),
    the galaxy bias in force, the shear normalisation, and the fixed-RNG seed if any.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("catalogue")
        for col, dt in CATALOGUE_DTYPES.items():
            g.create_dataset(col, data=np.asarray(catalogue[col]).astype(dt, copy=False))
        g.attrs["n_galaxies"] = int(catalogue.shape[0])
        for k, v in (extra_attrs or {}).items():
            if v is None:
                continue
            f.attrs[k] = v
        cg = f.create_group("cosmo_dict")
        for k, v in (cosmo_dict or {}).items():
            if isinstance(v, str):
                cg.create_dataset(k, data=v, dtype=h5py.string_dtype(encoding="utf-8"))
            else:
                cg.create_dataset(k, data=np.asarray(v))
    return path


def gower_fixed_test_with_topup(fixed_ids, num_sims, full_gower_ids, seed=GOWER_TOPUP_SEED):
    """Ordered Gower sim_id array for ``--gower-sim-set fixed_test`` (+ optional random top-up).

    The 200 fixed-test ids come FIRST (so the generic ``--num-sims`` prefix cap applied by the
    caller never drops them). When ``num_sims`` exceeds the fixed count, ``num_sims - len(fixed)``
    extra ids are drawn WITHOUT replacement from ``full_gower_ids`` minus the fixed set, using a
    deterministic ``seed`` so a resume / re-submit reproduces the same set. This gives a
    "200 fixed-test + M random" training suite in one output dir (the fixed ids stay forced into
    the eval test split via ``config.fixed_test_sim_ids``; the extras become train/val cosmologies).

    Args:
        fixed_ids: ordered list of the fixed-test sim_ids (farthest-point order preserved).
        num_sims: the ``--num-sims`` value, or ``None`` (=> just the fixed ids, no top-up).
        full_gower_ids: the full Gower sim_id array (``np.arange(193, 782)``).
        seed: RNG seed for the top-up draw.

    Returns:
        1-D ``float64`` ``np.ndarray`` of sim_ids (fixed first, then the random extras).
    """
    base = np.array(fixed_ids, dtype=np.float64).reshape(-1)
    if num_sims is None or num_sims <= len(fixed_ids):
        return base
    fixed_set = set(int(i) for i in fixed_ids)
    complement = np.array(
        sorted(int(i) for i in np.asarray(full_gower_ids).reshape(-1).tolist() if int(i) not in fixed_set),
        dtype=np.int64,
    )
    n_extra = min(int(num_sims) - len(fixed_ids), len(complement))
    if n_extra < int(num_sims) - len(fixed_ids):
        print(f"[gower-topup] WARNING requested {int(num_sims) - len(fixed_ids)} extra sim_ids but "
              f"only {len(complement)} non-fixed Gower cosmologies exist; capping at {n_extra}.")
    rng = np.random.default_rng(seed)
    extra = rng.choice(complement, size=n_extra, replace=False)
    return np.concatenate([base, extra.astype(np.float64)])

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

        if args.simulator_type == "gower_street" and args.gower_sim_set == "fixed_test":
            # Reduced Gower suite: use exactly the committed 200-id fixed test set instead of the
            # full np.arange(193,782). Reads the repo lock-file (single source of truth; synced to
            # the cluster checkout). Lazy import so the glass/smoke paths don't need the ml stack.
            from src.ml.data.fixed_test_set import load_fixed_test_ids_ordered
            json_path = Path(__file__).resolve().parent / "config" / "fixed_test_sets" / "gower_test_ids.json"
            # Order-preserving load: the lock-file stores the 200 ids in a maximally-separated
            # (farthest-point) order so a --num-sims N prefix stays well-spread across the param
            # space (sorting would cluster the prefix at the low sim_ids). The full-set gower runs
            # (num_sims=None) use all 200 regardless of order; only the prefix cares.
            fixed_ids = load_fixed_test_ids_ordered(str(json_path))
            full_gower_ids = SIM_TYPE_CONFIGS["gower_street"]["get_sim_samples"]()
            sim_samples = gower_fixed_test_with_topup(fixed_ids, args.num_sims, full_gower_ids)
            n_extra = len(sim_samples) - len(fixed_ids)
            print(f"[rank 0] --gower-sim-set fixed_test: {len(fixed_ids)} fixed sim_ids from "
                  f"{json_path.name} (min {min(fixed_ids)}, max {max(fixed_ids)}, first {fixed_ids[0]})"
                  + (f" + {n_extra} random top-up (seed {GOWER_TOPUP_SEED}) = {len(sim_samples)} total"
                     if n_extra > 0 else "."))

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
    IA_MODEL = args.ia_model
    ia_prior_spec = IA_PRIOR_SPECS[IA_MODEL]
    NO_ROTATIONS = args.no_rotations
    NO_AUGMENTATION = args.no_augmentation
    OVERWRITE = args.overwrite  # OFF (default) = resume: skip complete sims/blocks (see loop below)
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
    # Deterministic per-sim_id cosmology sampler (Gower Street flow prior), glass path only.
    # Seeded by (cosmo_base_seed, sim_id) so the same sim_id yields the same cosmology across runs /
    # analysis variates (see prepare_glass_backend). Imported lazily so the heavy ML/eval stack is
    # only required for the glass path.
    if SIMULATOR_TYPE == "glass":
        from src.ml.eval.utils import build_cosmo_param_sampler
        cosmo_sampler = build_cosmo_param_sampler(COSMO_PARAM_NAMES, csv_path=str(csv_path))
    else:
        cosmo_sampler = None
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
        if args.smoke_n_eff_scale is not None:
            # Mutating the module-level dict propagates to every SMOKE_CONFIG consumer
            # (tomo_nz scaling, VD dndz_scale, prepare_smoke_backend). Default None keeps
            # the config value — the sim smoke gate stays byte-identical.
            SMOKE_CONFIG["n_eff_scale"] = float(args.smoke_n_eff_scale)
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
        # --outer-reps overrides the config default (used to shrink datasets, e.g. GLASS outer=1).
        outer_reps = (args.outer_reps if args.outer_reps is not None
                      else OUTER_NUM_SHAPE_NOISE_REALISATIONS[SIMULATOR_TYPE])
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

    # Full augmentation set per (outer,rot) block and per sim, derived from the now-finalised
    # constants (so smoke / --no-augmentation collapse correctly). These define what "complete"
    # means on disk for the resume logic below: a block produces inner_reps * len(mask_rotation_angles)
    # files (simulator.run), and a sim is outer_reps * len(rotation_specs) such blocks.
    files_per_block = inner_reps * len(mask_rotation_angles)
    expected_files_per_sim = outer_reps * len(rotation_specs) * files_per_block
    if rank == 0:
        mode = "OVERWRITE (regen all)" if OVERWRITE else "resume (skip complete)"
        print(f"[rank 0] {mode}: {files_per_block} files/block, "
              f"{expected_files_per_sim} files/sim.")

    OUTPUT_DIR = args.output_dir
    CAMB_CACHE_DIR = args.camb_cache_dir
    COSMO_BASE_SEED_VAL = args.cosmo_base_seed
    # None => unseeded (production default). An int switches on paired/reproducible generation.
    RNG_SEED = args.rng_seed
    SAVE_CATALOGUES = args.save_catalogues
    if rank == 0 and SAVE_CATALOGUES:
        print(f"[rank 0] --save-catalogues: raw galaxy catalogues -> {OUTPUT_DIR}/catalogues/ "
              f"(~0.9 GB per mock at production n_eff — keep --num-sims small).", flush=True)
    if rank == 0 and RNG_SEED is not None:
        print(f"[rank 0] FIXED-RNG mode: --rng-seed {RNG_SEED}. Per-(sim,outer,rot) streams are "
              f"deterministic and split (matter fields independent of galaxy sampling), so runs "
              f"differing only in --galaxy-bias are PAIRED on the matter fields and nuisance "
              f"draws. The per-sim backend cache is bypassed to keep the stream position "
              f"resume-independent.", flush=True)
    # Create the output / CAMB-cache dirs on rank 0 ONLY, then barrier. Running
    # Path.mkdir(parents=True) concurrently from every rank races on the shared
    # /share filesystem: the losers get a transient FileNotFoundError (ENOENT)
    # which exist_ok=True does NOT suppress (it only catches FileExistsError),
    # crashing those ranks and failing the whole job (afterok validators then
    # never fire). Single-writer + Barrier makes the directory exist for all.
    if rank == 0:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if SIMULATOR_TYPE == "glass":
            CAMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    backend_states = {
        "glass": {
            "cache": {},
            "cosmo_sampler": cosmo_sampler,
            "cls_cache_dir": CAMB_CACHE_DIR,
            "cosmo_base_seed": COSMO_BASE_SEED_VAL,
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
    for num_sim_this_batch in range(len(recvbuf)):
        sim_num = sims[num_sim_this_batch]
        # Per-sim crash isolation: a failure on one sim (e.g. a CAMB error) is logged and the rank
        # moves on to its next sim_id rather than abandoning the rest of its chunk (see except below).
        try:
            # Resume fast-path: skip a sim whose full augmentation set is already on disk.
            if (not OVERWRITE) and sim_is_complete(OUTPUT_DIR, sim_num, expected_files_per_sim):
                print(f"[rank {rank}] sim {sim_num} complete ({expected_files_per_sim} files) — skipping.")
                continue
            for outer_idx in range(outer_reps):
                for rot_idx, rotation_spec in enumerate(rotation_specs):
                    # Block-level resume: skip a complete (outer,rot) block; otherwise clear any
                    # partial files left by an earlier kill so the block recomputes cleanly.
                    if not OVERWRITE:
                        if count_block_files(OUTPUT_DIR, sim_num, outer_idx, rot_idx) >= files_per_block:
                            print(f"[rank {rank}] sim {sim_num} block out{outer_idx}_rot{rot_idx} complete — skipping.")
                            continue
                        removed = remove_block_outputs(OUTPUT_DIR, sim_num, outer_idx, rot_idx)
                        if removed:
                            print(f"[rank {rank}] sim {sim_num} block out{outer_idx}_rot{rot_idx}: "
                                  f"removed {removed} partial files; recomputing.")
                    # Per-block random streams. In fixed-RNG mode (--rng-seed) these are
                    # deterministic in (seed, sim, outer, rot) AND mutually independent, so the
                    # matter fields + nuisance draws are shared across galaxy-bias variates while
                    # the galaxy sampling is free to diverge. See build_block_rngs.
                    rng, backend_rng, postproc_rng = build_block_rngs(
                        RNG_SEED, sim_num, outer_idx, rot_idx
                    )
                    if RNG_SEED is not None:
                        print(f"[rank {rank}] sim {sim_num} block out{outer_idx}_rot{rot_idx}: "
                              f"fixed-RNG seed={RNG_SEED} (streams split: sample/backend/postproc)",
                              flush=True)
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

                    # Fixed-RNG: hand the backend a FRESH cache so the in-memory sim-level cache
                    # cannot make the backend stream depend on which block happened to miss first.
                    # The cache only guards the IA draw + the (disk-cached) CAMB Cls load, so the
                    # extra cost is one cheap re-load per block — CAMB itself is never re-run.
                    glass_cache = ({} if RNG_SEED is not None
                                   else backend_states["glass"]["cache"])
                    if SMOKE:
                        backend = prepare_smoke_backend(backend_rng, SMOKE_CONFIG, ia_prior_spec=ia_prior_spec)
                    elif SIMULATOR_TYPE == "glass":
                        backend = prepare_glass_backend(
                            sim_num,
                            rng=backend_rng,
                            cosmo_sampler=backend_states["glass"]["cosmo_sampler"],
                            cosmo_base_seed=backend_states["glass"]["cosmo_base_seed"],
                            cls_cache_dir=backend_states["glass"]["cls_cache_dir"],
                            prior_ranges=ia_prior_spec,
                            sim_grid=SIM_GRID,
                            los_grid=LOS_GRID,
                            camb_limits=CAMB_LIMITS,
                            cache=glass_cache,
                        )
                    else:
                        backend = prepare_gower_backend(
                            sim_num,
                            rng=backend_rng,
                            loader=backend_states["gower_street"]["loader"],
                            prior_ranges=ia_prior_spec,
                            sim_grid=SIM_GRID,
                        )

                    param_dict = backend["param_dict"]
                    shells = backend["shells"]
                    matter = backend["matter"]
                    cosmo = backend["cosmo"]

                    # Tag the saved param group with the IA model so the read side can disambiguate
                    # which (per-model) IA parameters are present.
                    param_dict["ia_model"] = IA_MODEL

                    # IA params: the model tag + only this model's sampled parameters (a_ia plus
                    # b_ia / b_z / b_src as appropriate). f_red / log10_M_eff are used only by
                    # nla_m; the other models ignore them. nla_z's avg_a is added below once
                    # tomo_nz is available.
                    ia_params = {
                        "model": IA_MODEL,
                        "f_red": f_red,
                        "log10_M_eff": log10_M_eff,
                    }
                    for _ia_key in ia_prior_spec:
                        ia_params[_ia_key] = param_dict[_ia_key]

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

                    if IA_MODEL == "nla_z":
                        # Per-tomo-bin N(z)-weighted average scale factor <a>^(i) for the NLA-z
                        # redshift-dependent amplitude (Wright et al. 2025, eq. 7).
                        a_of_z = 1.0 / (1.0 + los_z_integration)
                        ia_params["avg_a"] = np.array([
                            np.average(a_of_z, weights=tomo_nz[i]) for i in range(nbins)
                        ])

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
                        # Theory-test mode applies NO forward shear bias (NoSystematics), so the
                        # 1/(1+m) de-bias inside make_alm_shear_convergence must ALSO be off — else
                        # clean mocks are spuriously rescaled by 1/(1+m)^2 per tomo bin, producing a
                        # z-tilt vs theory. See artifacts/compare/REPORT.md §2.
                        m_bias_for_shear = np.zeros_like(m_bias)
                        # Source clustering is a systematics-controlled effect: galaxies trace
                        # delta via positions_from_delta(..., galaxy_bias, ...). With systematics
                        # OFF this is a CLEAN theory test against a smooth-n(z) analytic theory, so
                        # disable clustering (galaxy_bias=0 -> uniform Poisson sampling). Otherwise
                        # source-lens clustering injects a low-z/high-l excess the theory does not
                        # model (precision-logbook H11/H12; the trusted kids_sbi samples uniformly).
                        # Production (systematics ON) keeps the realistic config bias.
                        galaxy_bias_sim = 0.0
                    else:
                        m_bias_for_shear = m_bias
                        # Default to the config `bias` constant; a CLI override (--galaxy-bias)
                        # lets us generate clustering variates (e.g. b_g=1.5 strong / 0.7 weak)
                        # without touching the protected physics. Only the systematics-ON path
                        # honours it; the clean theory-test path above keeps galaxy_bias=0.
                        galaxy_bias_sim = bias if args.galaxy_bias is None else args.galaxy_bias

                    kwargs = {
                        'cosmo': cosmo,
                        'los_z_integration': los_z_integration,
                        'tomo_nz': tomo_nz,
                        'galaxy_bias': galaxy_bias_sim,
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
                            catalogue, m_bias_for_shear, nbins, nside, lmax, nosh=False, mask=mask,
                            normalization=SHEAR_NORMALIZATION,
                            # Noise-only (random-rotation) alm: pass the block's dedicated stream
                            # instead of letting it fall back to the GLOBAL numpy RNG, which no
                            # seed can control. Unseeded mode is statistically unchanged.
                            rng=postproc_rng,
                        )
                        # mask_cls = unmixing_mask_cls(catalogue, nbins, nside, lmax, lmin, mask=mask)

                        mixed_cls = denoise_shear_cls(nbins, alm, alm_rand, lmax)
                        mixed_cut = mixed_cls[:, :, :, lower_lscale:upper_lscale+1]
                        cll_bands, mixed_bandpowers = compute_cl_bandpowers(
                            mixed_cut, nbins, lower_lscale, upper_lscale, nbands
                        )

                        # --- second (A3s8) map product ----------------------------------------
                        # Built AFTER the bandpowers so the 2-pt branch is provably untouched,
                        # and BEFORE `del catalogue` because it re-reads the galaxies. Its alms
                        # feed ONLY the map branch.
                        #
                        # RNG: a SPAWNED child stream, never `postproc_rng` itself. Drawing the
                        # random rotations from the parent would advance it and change every
                        # subsequent catalogue in this block, breaking byte-identity with the
                        # legacy output. Spawning leaves the parent's stream bit-identical
                        # (verified on numpy 2.2.6) at the cost of the two branches' noise
                        # references being independent realisations -- statistically fine, the
                        # random map is only a noise meter.
                        alm_sc8 = alm_rand_sc8 = None
                        if A3S8_VARIANT is not None:
                            sc8_rng = postproc_rng.spawn(1)[0]
                            alm_sc8, alm_rand_sc8 = make_alm_shear_convergence(
                                catalogue, m_bias_for_shear, nbins, nside, lmax, nosh=False,
                                mask=mask, normalization="smoothed_counts",
                                smoothed_counts_fwhm_arcmin=A3S8_FWHM_ARCMIN,
                                rng=sc8_rng,
                            )

                        # Optional raw-catalogue dump. MUST happen before the `del` below: the
                        # catalogue is the largest object in the pipeline and is freed here so the
                        # map-building peak stays bounded. Written alongside (not inside) the mock,
                        # so the output_*.h5 globs used by every dataset config are unaffected.
                        if SAVE_CATALOGUES:
                            cat_path = save_catalogue_h5(
                                OUTPUT_DIR / "catalogues"
                                / f"catalogue_{sim_num}_out{outer_idx}_rot{rot_idx}_{cat_idx}.h5",
                                catalogue, param_dict,
                                extra_attrs={
                                    "sim_id": int(sim_num), "outer_idx": int(outer_idx),
                                    "rot_idx": int(rot_idx), "cat_idx": int(cat_idx),
                                    "galaxy_bias": float(galaxy_bias_sim),
                                    "shear_normalization": SHEAR_NORMALIZATION,
                                    "systematics_model": SYSTEMATICS_MODEL,
                                    "nside": int(nside), "nside_out": int(nside_out),
                                    # make_alm_shear_convergence de-biases by 1/(1+m); replaying
                                    # the maps offline needs the m actually used.
                                    "m_bias_for_shear": np.asarray(m_bias_for_shear, dtype=float),
                                    "rng_seed": (-1 if RNG_SEED is None else int(RNG_SEED)),
                                },
                            )
                            print(f"[rank {rank}] wrote catalogue {cat_path.name} "
                                  f"({catalogue.shape[0]:,} galaxies)", flush=True)

                        del catalogue
                        gc.collect()

                        # Build E/B maps for each (fwhm, lmin, lcut) smoothing variant. Each
                        # triple adds its own pixelised_results keys
                        # (E_fwhm{f}[_lmin{lo}][_lcut{hi}] / B_...; both lmin & lcut None ->
                        # smoothing-only, keyed E_fwhm{f}).
                        map_types = {}
                        # tag -> {patch_name: (nbins,) f64, 'all': (nbins,)}; see patch_noise_std
                        noise_std = {}
                        want_noise_std = (list(EB_SMOOTHING_VARIANTS)
                                          if NOISE_STD_VARIANTS is None
                                          else [tuple(v) for v in NOISE_STD_VARIANTS])
                        patch_names = list(named_patches.keys())

                        # --- A3s8 branch: E only, one variant, float16 -------------------------
                        # Done FIRST so its alms (~0.8 GB) are released before the three counts
                        # variants allocate their nside_out maps -- this bounds the peak RSS.
                        sc8_tag = None
                        if alm_sc8 is not None:
                            fwhm_v, lmin_v, lcut_v = A3S8_VARIANT
                            sc8_tag = f"sc8_{eb_variant_tag(fwhm_v, lmin_v, lcut_v)}"
                            E_s, _ = filter_EB_alms_and_make_maps(
                                alm_list=alm_sc8, nside_out=nside_out, lmax_out=None,
                                fwhm_arcmin=fwhm_v, taper_start_frac=0.95,
                                lmin=lmin_v, lcut=lcut_v,
                            )
                            map_types[f"E_{sc8_tag}"] = E_s
                            Er_s, _ = filter_EB_alms_and_make_maps(
                                alm_list=alm_rand_sc8, nside_out=nside_out, lmax_out=None,
                                fwhm_arcmin=fwhm_v, taper_start_frac=0.95,
                                lmin=lmin_v, lcut=lcut_v,
                            )
                            noise_std[sc8_tag] = patch_noise_std(
                                Er_s, patches, nside_out, ang, patch_names)
                            del Er_s
                            alm_sc8 = alm_rand_sc8 = None
                            gc.collect()

                        for fwhm_v, lmin_v, lcut_v in EB_SMOOTHING_VARIANTS:
                            E_v, B_v = filter_EB_alms_and_make_maps(
                                alm_list=alm, nside_out=nside_out, lmax_out=None,
                                fwhm_arcmin=fwhm_v, taper_start_frac=0.95,
                                lmin=lmin_v, lcut=lcut_v,
                            )
                            tag = eb_variant_tag(fwhm_v, lmin_v, lcut_v)
                            map_types[f"E_{tag}"] = E_v
                            map_types[f"B_{tag}"] = B_v

                            # Noise meter for the A1 rescale: filter the random-rotation alms
                            # through the SAME variant filter, reduce to scalars, discard maps.
                            if (fwhm_v, lmin_v, lcut_v) in want_noise_std:
                                Er_v, _ = filter_EB_alms_and_make_maps(
                                    alm_list=alm_rand, nside_out=nside_out, lmax_out=None,
                                    fwhm_arcmin=fwhm_v, taper_start_frac=0.95,
                                    lmin=lmin_v, lcut=lcut_v,
                                )
                                noise_std[tag] = patch_noise_std(
                                    Er_v, patches, nside_out, ang, patch_names)
                                del Er_v

                        pixelised_results = {name:{} for name in map_types.keys()}
                        for name, cat_data in map_types.items():
                            pixelised_tomobin_patches = get_patch_values(cat_data, patches, nside_out, ang)
                            # The A3s8 branch is an ML-only derived product, so it is stored at
                            # A3S8_MAP_DTYPE (float16) -- half the delta on disk, and the prebake
                            # becomes a pass-through. The primary maps stay float32.
                            out_dtype = (A3S8_MAP_DTYPE if name.startswith("E_sc8_")
                                         else np.float32)
                            for patch_idx, patch_name in enumerate(patch_names):
                                # Store the pixelised E/B maps as float32: halves on-disk size, and
                                # the ML loader casts them to float32 on read anyway
                                # (src/ml/data/data_loading.py), so no analysis precision is lost.
                                # Only the maps are downcast; cls/bandpowers/cosmo stay float64.
                                arr = pixelised_tomobin_patches[patch_idx].astype(
                                    out_dtype, copy=False)
                                if out_dtype is not np.float32 and not np.isfinite(arr).all():
                                    print(f"[rank {rank}] WARNING non-finite values in {name}"
                                          f"/{patch_name} after the {np.dtype(out_dtype).name} "
                                          f"cast (max|x| before cast = "
                                          f"{np.abs(pixelised_tomobin_patches[patch_idx]).max():.3e})",
                                          flush=True)
                                pixelised_results[name][patch_name] = arr

                        # Noise-meter scalars (~288 B total) + provenance. Both are additive:
                        # every existing key is byte-identical to the legacy output.
                        for tag, per_patch in noise_std.items():
                            pixelised_results[f"noise_std_{tag}"] = per_patch
                        if noise_std:
                            pixelised_results["_provenance"] = {
                                "noise_std": (
                                    "std of the filtered random-rotation E map over ALL pixels of "
                                    "each stored patch grid, per tomographic bin; 'all' pools both "
                                    "patches. A1_wht_rand = E_<tag> / noise_std_<tag>. Differs from "
                                    "the offline harness (which restricts to the galaxy-occupied "
                                    "footprint) by a geometric factor that is fixed across mocks."
                                ),
                            }
                            if sc8_tag is not None:
                                pixelised_results["_provenance"]["a3s8"] = (
                                    f"E_sc8_* built by a SECOND make_alm_shear_convergence call with "
                                    f"normalization='smoothed_counts', "
                                    f"smoothed_counts_fwhm_arcmin={A3S8_FWHM_ARCMIN:g}. Its "
                                    "random-rotation noise reference is an INDEPENDENT realisation "
                                    "of the counts branch's (spawned RNG stream) -- the random map "
                                    "is a noise meter only. Bandpowers are NOT affected: they are "
                                    "computed from the primary counts-normalised alms."
                                )
                                pixelised_results["_provenance"]["a3s8_variant"] = sc8_tag

                        # cls_results['full'] = {"cls": mixed_cls, "mixed_bandpowers":mixed_bandpowers, "bandpower_ls":cll_bands}
                        cls_results['full'] = {"mixed_bandpowers":mixed_bandpowers, "bandpower_ls":cll_bands, "cls": mixed_cls[:, :, :2, :]}  # only save EE and BB

                        save_string = f"{sim_num}_out{outer_idx}_rot{rot_idx}"
                        total_idx = cat_idx
                        save_results_h5( OUTPUT_DIR / f"output_{save_string}.h5", total_idx, cls_results, pixelised_results, param_dict)

                        # free per-catalogue heavy products
                        del cls_results, pixelised_results, cll_bands, map_types, alm, alm_rand
                        del noise_std
                        gc.collect()

                        cat_idx += 1

                    del cat_queue, simulator
                    gc.collect()
                    print(f"Finished single rotation simulation in {time.time() - s_catalogue:.2f} seconds")

                print(f'Saved results for sim {sim_num}')
            
            print(f'Entire simulation took {time.time() - s:.2f} seconds')
        except Exception as e:
            # Per-sim failure: log (with traceback) and continue to the next sim_id so one bad sim
            # never kills the rank. Any partial block this sim wrote is reclaimed on a later visit
            # by the block-level "incomplete -> clean -> recompute" gate above; a permanently
            # failing sim simply never reaches its full file count, so it is never marked complete.
            print(f"[rank {rank}] sim {sim_num} FAILED: {e!r}; continuing to next sim.")
            traceback.print_exc()
            continue