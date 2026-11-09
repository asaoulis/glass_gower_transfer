"""MPI KiDS-Legacy mock generator: python master_kids_legacy_simulator.py --simulator-type <backend>.

One shared pipeline (shells -> shear -> galaxy catalogue -> bandpowers and pixelised E/B maps), with
the gower_street, glass, smoke and external backends differing in how the matter shells are produced.
Variable depth is on by default with --kids-systematics and the KiDS mask (quadcap table). Every mock
carries cosmo_dict/vd_code (0 off, 1 pooled, 2 patch, 3 quadcap, 4 v2pr).
"""
import numpy as np
import healpy as hp
import re
import h5py
from pathlib import Path
from collections import deque
import gc
import argparse
import sys
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
from src.KiDS.tomo import calculate_tomo_nz, n_arcmin2
from src.KiDS.rotations import KiDS_PATCH_GOWER_ROTATIONS, KiDS_PATCH_GLASS_ROTATIONS
from src.KiDS.simulation_config import (
    CAMB_CLS_CACHE_DIR,
    CAMB_LIMITS,
    COSMO_BASE_SEED,
    GALAXY_BIAS_CLIP,
    GALAXY_BIAS_PRIORS,
    GALAXY_BIAS_PRIOR_MEANS,
    GALAXY_BIAS_PRIOR_SIGMAS,
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
                        choices=["gower_street", "glass", "smoke", "external"],
                        help="Simulator backend ('smoke' = reduced-cost local pre-flight; "
                             "'external' = an ingested external shell cube, see --ext-variant)")

    parser.add_argument("--ext-variant", type=str, default="dmo",
                        choices=["dmo", "bary"],
                        help="--simulator-type external: which cube of the external box to "
                             "stream (dmo = gravity-only, bary = baryonified)")
    parser.add_argument("--ext-data-dir", type=str, default=None,
                        help="--simulator-type external: override the box directory "
                             "(default: src/external_mocks FLAMINGO_BIG['data_dir'])")
    parser.add_argument("--outer-indices", type=str, default=None,
                        help="Comma-separated global outer (shape-noise) indices to run in this "
                             "process. Shards one cosmology's blocks.")
    parser.add_argument("--rot-indices", type=str, default=None,
                        help="Comma-separated global rotation indices to run in this process "
                             "(for example '0,1').")

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
        "--variable-depth", action=argparse.BooleanOptionalAction, default=None,
        help="Variable-depth (VD) effect, implies full NLA and shear-bias systematics. "
             "Default: on with --kids-systematics and the KiDS mask.",
    )

    parser.add_argument(
        "--vd-table", type=str, default=None, choices=["pooled", "patch", "quadcap", "v2pr"],
        help="Variable-depth count-contrast table: pooled, patch, quadcap (default when VD is on) "
             "or v2pr.",
    )

    parser.add_argument(
        "--ia-model",
        type=str,
        default="nla_m",
        choices=["nla_m", "nla", "nla_z", "tatt"],
        help="Intrinsic-alignment model: nla_m (default), nla, nla_z or tatt.",
    )

    parser.add_argument("--no-rotations", action="store_true",
                        help="Disable rotations")

    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute every sim even if complete outputs exist. Default: resume.")

    parser.add_argument("--num-sims", type=int, default=None,
                        help="Cap the number of cosmologies/simulations to the first N "
                             "(default: all). Used for bounded validation/theory-test runs.")

    parser.add_argument("--no-augmentation", action="store_true",
                        help="One mock per cosmology: collapse rotations and shape-noise "
                             "realisations to one.")

    parser.add_argument("--shear-normalization", type=str, default="counts",
                        choices=["counts", "mean", "expected"],
                        help="Shear-map normalisation: counts (default), mean or expected.")

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

    # A shared CAMB-Cls cache and a base seed for the per-sim_id cosmology let variates reuse the Cls.
    parser.add_argument("--camb-cache-dir", type=Path, default=CAMB_CLS_CACHE_DIR,
                        help="Shared on-disk cache for CAMB-computed matter Cls, keyed by sim_id.")

    parser.add_argument("--cosmo-base-seed", type=int, default=COSMO_BASE_SEED,
                        help="Base seed for deterministic per-sim_id cosmology sampling.")

    parser.add_argument(
        "--galaxy-bias", type=float, default=None,
        help="Fixed source-galaxy clustering bias b_g, overriding the config constant "
             "(systematics-on path only).")

    parser.add_argument(
        "--galaxy-bias-prior", default=None, choices=sorted(GALAXY_BIAS_PRIORS),
        help="Per-block draw of the six per-tomo-bin galaxy biases from a named prior preset. "
             "Excludes --galaxy-bias.")

    parser.add_argument(
        "--eb-variants", default=EB_VARIANTS_DEFAULT, choices=sorted(EB_VARIANT_PRESETS),
        help="Which counts-normalised E/B map products the store carries: sc8only (default) "
             "or full.")

    parser.add_argument(
        "--rng-seed", type=int, default=None,
        help="Seed for paired-catalogue mode: derives every random stream from (seed, sim_id, "
             "outer, rot). Default: unseeded.")

    parser.add_argument(
        "--save-catalogues", action="store_true",
        help="Also dump each mock's raw galaxy catalogue next to its output file (about 0.9 GB "
             "per mock).")

    parser.add_argument(
        "--smoke-n-eff-scale", type=float, default=None,
        help="Smoke only: override the n_eff scale of the smoke config.")

    parser.add_argument(
        "--smoke-geometry", choices=["smoke", "production"], default="smoke",
        help="Smoke only: run the smoke backend at the production map and ell geometry.")
    parser.add_argument(
        "--catalogue-dtype", choices=["float32", "float64"], default="float32",
        help="Column dtype of the --save-catalogues dump (default float32).")

    parser.add_argument("--outer-reps", type=int, default=None,
                        help="Override the number of outer shape-noise realisations per sim. "
                             "Default: the simulator's config value.")

    parser.add_argument("--gower-sim-set", type=str, default="full",
                        help="Gower sim_id selection: 'full' or the basename of a lock-file under "
                             "config/fixed_test_sets/.")

    parser.add_argument("--ia-prior-set", type=str, default="default",
                        choices=sorted(IA_PRIOR_OVERRIDES),
                        help="Named IA-prior override set applied on top of IA_PRIOR_SPECS "
                             "(default, matched, ppc).")

    return parser.parse_args()


# Variable-depth provenance code stamped per mock as cosmo_dict/vd_code (a float, loader-safe).
VD_TABLE_CODES = {"off": 0, "pooled": 1, "patch": 2, "quadcap": 3, "v2pr": 4}   # 4 = VD v2 (PR #3, opt-in)
VD_TABLE_DEFAULT = "quadcap"   # production VD table


def _resolve_variable_depth(args):
    """Resolve the tri-state --variable-depth (None = auto) to (on, table, source).

    Must run before resolve_systematics_model(args). Raises SystemExit on a contradictory
    command line.
    """
    explicit = args.variable_depth
    if explicit is True and args.simulator_type == "external":
        raise SystemExit("--variable-depth with --simulator-type external is not supported: the VD "
                         "path was never validated on the external top-hat shell windows.")
    if explicit is None:
        # The systematics model WITHOUT variable depth (variable_depth is None -> falsy here).
        base_model = resolve_systematics_model(args)
        kids_geom = (args.simulator_type == "smoke"
                     or (args.simulator_type in ("glass", "gower_street") and args.use_kids_mask))
        if base_model != "nla":
            on, source = False, f"auto: systematics model '{base_model}'"
        elif not kids_geom:
            why = ("simulator 'external'" if args.simulator_type == "external"
                   else "no --use-kids-mask")
            on, source = False, f"auto: {why}"
        else:
            on, source = True, "auto"
    elif explicit:
        on, source = True, "explicit"
    else:
        on, source = False, "explicit --no-variable-depth"
    if not on and args.vd_table is not None:
        raise SystemExit(f"--vd-table {args.vd_table} given but variable depth resolves OFF "
                         f"({source}); drop --vd-table or pass --variable-depth.")
    table = (args.vd_table or VD_TABLE_DEFAULT) if on else None
    return on, table, source


def _vd_banner(on, table, source):
    if on:
        return f"[rank 0] variable-depth: ON vd_table={table} (source={source})"
    return f"[rank 0] variable-depth: OFF (source={source})"


GLASS_N_JOBS = 6500

# Cosmology parameters drawn (deterministically) per sim_id and fed to CAMB. These are exactly the
# keys the on-disk Cls cache guard compares (see src/cosmology/mpi_camb.COSMO_GUARD_KEYS).
COSMO_PARAM_NAMES = ["omega_m", "sigma_8", "ombh2", "h", "ns", "w0", "mnu"]

# (fwhm_arcmin, lmin, lcut) triples. Each adds pixelised_results keys E_fwhm{f}[_lmin{lo}][_lcut{hi}]
# and B_..., and None for lmin or lcut omits that suffix.
EB_SMOOTHING_VARIANTS = []

# --- sc8 / counts map products ---

# A3S8_VARIANTS: (fwhm_arcmin, lmin, lcut) triples built from the sc8 alms. Each must also appear in
# EB_SMOOTHING_VARIANTS while the counts branch is on.
A3S8_VARIANTS = [(4.0, 56, 1400), (8.0, 56, 1024)]
# Guard: while the counts branch is on, every sc8 variant needs a matching counts variant.
assert not EB_SMOOTHING_VARIANTS or all(
    tuple(v) in [tuple(x) for x in EB_SMOOTHING_VARIANTS] for v in A3S8_VARIANTS), (
    f"A3S8_VARIANTS {A3S8_VARIANTS} must each appear in EB_SMOOTHING_VARIANTS "
    f"{EB_SMOOTHING_VARIANTS}")
# A store with neither branch would carry no maps at all: always a mistake, never a config.
assert EB_SMOOTHING_VARIANTS or A3S8_VARIANTS, (
    "both EB_SMOOTHING_VARIANTS and A3S8_VARIANTS are empty: the store would contain no E/B maps")
# FWHM (arcmin) of the count map used as the denominator.
A3S8_FWHM_ARCMIN = 8.0
# sc8 maps are stored as float16 (ML-only product).
A3S8_MAP_DTYPE = np.float16

# noise_std_<tag>: per-(variant, bin, patch) std of the random-rotation noise map, stored as scalars
# so the A1 rescale happens at load time.
NOISE_STD_VARIANTS = ()

# EB_VARIANT_PRESETS: named store contents. EB_SMOOTHING_VARIANTS and NOISE_STD_VARIANTS must move together.
EB_VARIANT_PRESETS = {
    "sc8only": {"eb": [], "noise_std": ()},
    "full": {"eb": [(4.0, 56, 1400), (8.0, 56, 1400), (8.0, 56, 1024)], "noise_std": None},
}
# The preset whose contents match the module-level defaults above, so `--eb-variants` defaults to
# a genuine no-op and every existing caller keeps its exact behaviour.
EB_VARIANTS_DEFAULT = "sc8only"


def eb_variant_tag(fwhm_v, lmin_v, lcut_v):
    """HDF5 key suffix for one (fwhm, lmin, lcut) smoothing variant."""
    return (f"fwhm{fwhm_v:g}"
            + ("" if lmin_v is None else f"_lmin{int(lmin_v)}")
            + ("" if lcut_v is None else f"_lcut{int(lcut_v)}"))


def patch_noise_std(rand_E_maps, patches, nside_out, ang, patch_names):
    """Per-(patch, bin) std of the filtered random-rotation E map, plus the pooled 'all'.

    Returns {patch_name: (nbins,), 'all': (nbins,)}.
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

# Per-IA-model forward-sampling priors: parameter -> ("uniform", lo, hi) or ("normal", mu, sigma),
# drawn by sim_utils.sample_ia_params.
# nla_m: Fortuna et al. 2025 / Wright et al. 2025 (covers their posterior at 5 sigma).
# nla / nla_z / tatt: Wright et al. 2025 priors (A_IA ~ U[-6,6]; B_IA ~ N(-3.7,4.3); b_src ~ U[-0.5,1.5]).
IA_PRIOR_SPECS = {
    'nla_m': {'a_ia': ('uniform', 4.48, 7.0), 'b_ia': ('uniform', 0.28, 0.6)},
    'nla':   {'a_ia': ('uniform', -6.0, 6.0)},
    'nla_z': {'a_ia': ('uniform', -6.0, 6.0), 'b_z': ('normal', -3.7, 4.3)},
    'tatt':  {'a_ia': ('uniform', -6.0, 6.0), 'b_src': ('uniform', -0.5, 1.5)},
}

# --- IA prior override sets: applied per parameter on top of IA_PRIOR_SPECS (--ia-prior-set) ---
IA_PRIOR_OVERRIDES = {
    'default': {},
    'matched': {
        'nla_m': {'a_ia': ('uniform', 5.740000, 5.740000),
                  'b_ia': ('uniform', 0.440000, 0.440000)},
        'nla':   {'a_ia': ('uniform', 0.435641, 0.435641)},
        'nla_z': {'a_ia': ('uniform', -0.272397, -0.272397),
                  'b_z':  ('uniform', -3.700000, -3.700000)},
    },
    # 'ppc': a_ia drawn from a mildly inflated Gaussian fit to the pooled real-data field posterior.
    'ppc': {
        'nla':   {'a_ia': ('normal', -0.114, 0.29)},
        'nla_z': {'a_ia': ('normal', -0.077, 0.35)},
    },
}

# Nuisances pinned by a prior set. log10_M_eff is drawn per block and then overwritten (draw-then-pin).
NUISANCE_PINS = {
    'default': {},
    'matched': {'log10_M_eff': 'prior_mean'},   # multiplies the nla_m IA amplitude
    'ppc': {},
}


def resolve_ia_prior_spec(ia_model, prior_set='default'):
    """-> the forward-sampling prior spec for `ia_model` under the named override set."""
    if prior_set not in IA_PRIOR_OVERRIDES:
        raise ValueError(f"unknown --ia-prior-set {prior_set!r}; "
                         f"choose from {sorted(IA_PRIOR_OVERRIDES)}")
    base = IA_PRIOR_SPECS[ia_model]
    override = IA_PRIOR_OVERRIDES[prior_set].get(ia_model, {})
    unknown = set(override) - set(base)
    if unknown:
        raise ValueError(f"--ia-prior-set {prior_set!r} overrides parameters {sorted(unknown)} that "
                         f"IA model {ia_model!r} does not take ({sorted(base)})")
    return {**base, **override}


SIM_TYPE_CONFIGS = {
    "glass": {
        "rotation_specs": KiDS_PATCH_GLASS_ROTATIONS,
        "get_sim_samples": lambda: np.arange(GLASS_N_JOBS),
    },
    "gower_street": {
        "rotation_specs": KiDS_PATCH_GOWER_ROTATIONS,
        "get_sim_samples": lambda: np.arange(193, 781 + 1),
    },
    # One external cosmology (the ingested box), with Gower rotations to match the Gower suite.
    "external": {
        "rotation_specs": KiDS_PATCH_GOWER_ROTATIONS,
        "get_sim_samples": lambda: np.arange(1),
    },
    # Reduced-cost local pre-flight: one sim, rotations and augmentations collapsed below.
    "smoke": {
        "rotation_specs": KiDS_PATCH_GLASS_ROTATIONS,
        "get_sim_samples": lambda: np.arange(1),
    },
}

# Seed for the random top-up of --gower-sim-set beyond the lock size: the lock ids are kept,
# extras drawn without replacement.
GOWER_TOPUP_SEED = 20260710


# --- fixed-RNG (paired-catalogue) mode: stream tags spawned off the per-block SeedSequence ---
# The tag order is part of the on-disk contract.
RNG_STREAM_SAMPLE = 0   # nuisance draws + galaxy sampling / shape noise (Simulator rng)
RNG_STREAM_BACKEND = 1  # IA-prior draws + the GLASS log-normal matter fields
RNG_STREAM_POSTPROC = 2  # random ellipticity rotation for the noise-only alm (map_shears)
RNG_STREAM_LEGACY = 3   # seeds the GLOBAL numpy RNG (see build_block_rngs)


def build_block_rngs(seed, sim_num, outer_idx, rot_idx):
    """Return (sample_rng, backend_rng, postproc_rng) for one (sim, outer, rot) block.

    seed None: three independent unseeded generators. seed set: three streams spawned from
    (seed, sim, outer, rot), so runs sharing the seed share matter fields and nuisance draws.
    In seeded mode the global numpy RNG is also seeded, from a fourth stream, because the
    photo-z shift in src/KiDS/tomo.py draws from it.
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


def draw_galaxy_bias_prior(preset, parent_rng):
    """Draw the per-tomo-bin galaxy-bias 6-vector for one block from the named preset.

    b_i ~ N(mean_i, kappa * sigma_i), truncated at +-3 kappa sigma and clipped to GALAXY_BIAS_CLIP.
    """
    spec = GALAXY_BIAS_PRIORS[preset]
    kappa = float(spec["kappa"])
    bias_rng = parent_rng.spawn(1)[0]
    means = np.asarray(GALAXY_BIAS_PRIOR_MEANS, dtype=float)
    sigmas = kappa * np.asarray(GALAXY_BIAS_PRIOR_SIGMAS, dtype=float)
    draw = bias_rng.normal(means, sigmas)
    draw = np.clip(draw, means - 3.0 * sigmas, means + 3.0 * sigmas)
    return np.clip(draw, GALAXY_BIAS_CLIP[0], GALAXY_BIAS_CLIP[1])


# --- Raw galaxy-catalogue dumps --------------------------------------------------------------
# Column dtypes for --save-catalogues (the catalogue is the largest object in the pipeline).
CATALOGUE_DTYPES = {
    "RA": np.float32, "DEC": np.float32, "Z_TRUE": np.float32,
    "ZBIN": np.int8, "E1": np.float32, "E2": np.float32,
}


def save_catalogue_h5(path, catalogue, cosmo_dict, extra_attrs=None):
    """Write one mock's raw galaxy catalogue (RA, DEC, Z_TRUE, ZBIN, E1, E2, optional W) next to its output h5."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("catalogue")
        cols = dict(CATALOGUE_DTYPES)
        if catalogue.dtype.names and "W" in catalogue.dtype.names:
            cols["W"] = np.float32          # VD v2 (vd_table='v2pr') per-galaxy shear weight
        for col, dt in cols.items():
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
    """Ordered Gower sim_id array for --gower-sim-set.

    The 200 fixed-test ids come first, then an optional seeded random top-up from the complement.
    Returns a 1-D float64 array.
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

    s = time.time()

    seed_num = None

    if rank == 0:
        args = parse_args()
        # Resolve --variable-depth here so every rank sees a plain bool before resolve_systematics_model().
        try:
            _vd_on, _vd_table, _vd_source = _resolve_variable_depth(args)
        except SystemExit as exc:
            print(f"[rank 0] ERROR: {exc}", file=sys.stderr, flush=True)
            if size > 1:
                comm.Abort(2)   # the other ranks are already waiting in bcast
            raise SystemExit(2)
        args.variable_depth = _vd_on
        args.vd_table = _vd_table
        _banner = _vd_banner(_vd_on, _vd_table, _vd_source)
        print(_banner, flush=True)
        print(_banner, file=sys.stderr, flush=True)

        sim_samples = SIM_TYPE_CONFIGS[args.simulator_type]["get_sim_samples"]()

        if args.simulator_type == "gower_street" and args.gower_sim_set != "full":
            # --gower-sim-set carries the lock basename. A new selection criterion means a new
            # lock-file, not a new flag.
            from src.ml.data.fixed_test_set import load_fixed_test_ids_ordered
            lock_name = ("gower_test_ids" if args.gower_sim_set == "fixed_test"  # legacy alias
                         else args.gower_sim_set)
            if not re.fullmatch(r"gower_[A-Za-z0-9_]+", lock_name):
                raise ValueError(f"--gower-sim-set {args.gower_sim_set!r}: expected 'full', "
                                 f"'fixed_test', or a 'gower_*' lock basename")
            json_path = (Path(__file__).resolve().parent / "config" / "fixed_test_sets"
                         / f"{lock_name}.json")
            if not json_path.exists():
                raise FileNotFoundError(f"--gower-sim-set {args.gower_sim_set!r}: no lock file at "
                                        f"{json_path}")
            # Keep the lock-file order: it is farthest-point order, so a --num-sims prefix stays
            # spread over parameter space.
            fixed_ids = load_fixed_test_ids_ordered(str(json_path))
            full_gower_ids = SIM_TYPE_CONFIGS["gower_street"]["get_sim_samples"]()
            sim_samples = gower_fixed_test_with_topup(fixed_ids, args.num_sims, full_gower_ids)
            n_extra = len(sim_samples) - len(fixed_ids)
            print(f"[rank 0] --gower-sim-set {args.gower_sim_set}: {len(fixed_ids)} sim_ids from "
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
    VD_TABLE = args.vd_table
    VD_CODE = VD_TABLE_CODES[VD_TABLE] if SYSTEMATICS_MODEL == "nla_vd" else VD_TABLE_CODES["off"]
    IA_PRIOR_SET = args.ia_prior_set
    ia_prior_spec = resolve_ia_prior_spec(IA_MODEL, IA_PRIOR_SET)
    nuisance_pins = NUISANCE_PINS[IA_PRIOR_SET]
    if IA_PRIOR_SET != 'default':
        print(f"[rank {rank}] --ia-prior-set {IA_PRIOR_SET}: ia_prior_spec={ia_prior_spec} "
              f"nuisance_pins={nuisance_pins}", flush=True)
    NO_ROTATIONS = args.no_rotations
    NO_AUGMENTATION = args.no_augmentation
    OVERWRITE = args.overwrite  # OFF (default) = resume: skip complete sims/blocks (see loop below)
    SHEAR_NORMALIZATION = args.shear_normalization
    USE_KIDS_MASK = args.use_kids_mask
    csv_path = args.csv_path
    gower_data_dir = args.gower_data_dir
    data_dir = args.data_dir
    SMOKE = (SIMULATOR_TYPE == "smoke")
    if args.catalogue_dtype == "float64":
        for _col in ("RA", "DEC", "Z_TRUE", "E1", "E2"):
            CATALOGUE_DTYPES[_col] = np.float64
    if SMOKE:
        # Smoke runs fall back to the in-repo fixtures when the cluster --data-dir is absent.
        from src.smoke_sim import SMOKE_CONFIG, prepare_smoke_backend
        if not data_dir.exists():
            data_dir = Path(__file__).resolve().parent / "kids-legacy-sbi" / "data"
        gower_prior = None
    elif SIMULATOR_TYPE == "external":
        # The external box has one fixed cosmology from its own control files, so no Gower prior is needed.
        gower_prior = None
    else:
        gower_prior = GowerStPrior.from_csv(csv_path, drop_first=192)
    # Per-sim_id cosmology sampler (glass path): seeded by (cosmo_base_seed, sim_id), so a sim_id
    # has the same cosmology across runs.
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

    # --rot-indices / --outer-indices shard one cosmology's blocks over processes (MPI is over cosmologies).
    ROT_INDICES = None
    if args.rot_indices:
        ROT_INDICES = {int(v) for v in args.rot_indices.split(",") if v.strip() != ""}
        bad = {i for i in ROT_INDICES if i >= len(rotation_specs)}
        if bad:
            raise ValueError(f"--rot-indices {sorted(bad)} out of range "
                             f"(only {len(rotation_specs)} rotation specs)")
        if rank == 0:
            print(f"[rank 0] --rot-indices {sorted(ROT_INDICES)} of "
                  f"{len(rotation_specs)} rotation blocks", flush=True)

    if SMOKE:
        # Collapse the augmentation loops and rebind the cost-driving module globals (smoke only).
        rotation_specs = rotation_specs[:1]
        mask_rotation_angles = [0]
        USE_KIDS_MASK = True
        if args.smoke_n_eff_scale is not None:
            # Mutating the module dict reaches every SMOKE_CONFIG consumer. None keeps the config value.
            SMOKE_CONFIG["n_eff_scale"] = float(args.smoke_n_eff_scale)
        if args.smoke_geometry == "production":
            # Production map/ell geometry on smoke physics, for pipeline-identity checks only.
            SMOKE_CONFIG.update({"nside": 1024, "lmax": 2048, "nside_out": 512,
                                 "lower_lscale": 56, "upper_lscale": 1500, "nbands": 8})
            if rank == 0:
                print("[rank 0] --smoke-geometry production: nside 1024 / lmax 2048 / nside_out 512 "
                      "/ bands 56-1500 (smoke physics, validation only)", flush=True)
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
        # --outer-reps overrides the config default.
        outer_reps = (args.outer_reps if args.outer_reps is not None
                      else OUTER_NUM_SHAPE_NOISE_REALISATIONS[SIMULATOR_TYPE])
        inner_reps = INNER_NUM_SHAPE_NOISE_REALISATIONS[SIMULATOR_TYPE]

    if NO_AUGMENTATION and not SMOKE:
        # One mock per cosmology: one rotation and one shape-noise realisation.
        rotation_specs = rotation_specs[:1]
        outer_reps = 1
        inner_reps = 1
        if rank == 0:
            print("[rank 0] --no-augmentation: 1 mock/cosmology "
                  "(1 rotation, outer_reps=inner_reps=1).")

    # Full augmentation set per (outer, rot) block, derived from the finalised constants. This defines
    # 'complete' for resume.
    files_per_block = inner_reps * len(mask_rotation_angles)
    # Parsed HERE, not next to --rot-indices: `outer_reps` is assigned in the SMOKE / production
    # branches above, so validating the indices any earlier is a NameError.
    OUTER_INDICES = None
    if args.outer_indices:
        OUTER_INDICES = {int(v) for v in args.outer_indices.split(",") if v.strip() != ""}
        _bad_outer = {i for i in OUTER_INDICES if i >= outer_reps}
        if _bad_outer:
            raise ValueError(f"--outer-indices {sorted(_bad_outer)} out of range "
                             f"(outer_reps={outer_reps})")
        if rank == 0:
            print(f"[rank 0] --outer-indices {sorted(OUTER_INDICES)} of {outer_reps} outer reps",
                  flush=True)

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
    # Resolve --eb-variants before any mock is written: rebinds the two module globals the counts
    # branch reads.
    EB_VARIANTS_MODE = args.eb_variants
    _eb_preset = EB_VARIANT_PRESETS[EB_VARIANTS_MODE]
    EB_SMOOTHING_VARIANTS = list(_eb_preset["eb"])
    NOISE_STD_VARIANTS = _eb_preset["noise_std"]
    # Re-run the pairing guard against the resolved list: the import-time assert only saw the
    # default, so a preset whose counts branch is on must still cover every A3S8 variant.
    assert not EB_SMOOTHING_VARIANTS or all(
        tuple(v) in [tuple(x) for x in EB_SMOOTHING_VARIANTS] for v in A3S8_VARIANTS), (
        f"--eb-variants {EB_VARIANTS_MODE}: A3S8_VARIANTS {A3S8_VARIANTS} must each appear in "
        f"the preset's EB_SMOOTHING_VARIANTS {EB_SMOOTHING_VARIANTS}")
    assert EB_SMOOTHING_VARIANTS or A3S8_VARIANTS, (
        f"--eb-variants {EB_VARIANTS_MODE} leaves both EB_SMOOTHING_VARIANTS and A3S8_VARIANTS "
        f"empty: the store would contain no E/B maps")
    GALAXY_BIAS_PRIOR = args.galaxy_bias_prior
    if GALAXY_BIAS_PRIOR is not None and args.galaxy_bias is not None:
        raise SystemExit("--galaxy-bias-prior and --galaxy-bias are mutually exclusive "
                         "(the fixed override bypasses the prior; pick one).")
    if GALAXY_BIAS_PRIOR is not None:
        # Tripwire: a per-tomo bias vector needs the bias_tomo hook at positions_from_delta, else
        # GLASS broadcasts it as extra populations.
        import inspect as _inspect
        from src.cosmology.simulators import BaseSimulator as _BS
        if "bias_tomo" not in _inspect.getsource(_BS.run):
            raise SystemExit(
                "--galaxy-bias-prior requires the per-tomo 'bias_tomo' hook in "
                "src/cosmology/simulators.py (BaseSimulator.run); without it the (nbins,) "
                "bias vector is broadcast as extra galaxy populations and silently corrupts "
                "the mocks. Apply the protected hook first (spec_O3.md).")
    if rank == 0 and GALAXY_BIAS_PRIOR is not None:
        _spec = GALAXY_BIAS_PRIORS[GALAXY_BIAS_PRIOR]
        print(f"[rank 0] galaxy-bias prior '{GALAXY_BIAS_PRIOR}' (kappa={_spec['kappa']:g}): "
              f"per-(sim,outer,rot) per-tomo-bin draws around {GALAXY_BIAS_PRIOR_MEANS}; "
              f"realised values stored as cosmo_dict/b_g_bin1..6 + galaxy_bias_eff.", flush=True)
    if rank == 0:
        # Always print the store-content banner: it is the post-submit check of what the store contains.
        _ns = ("every EB variant" if NOISE_STD_VARIANTS is None
               else "none" if not NOISE_STD_VARIANTS
               else str([eb_variant_tag(*v) for v in NOISE_STD_VARIANTS]))
        print(f"[rank 0] --eb-variants {EB_VARIANTS_MODE}: counts-normalised E/B variants "
              f"{[eb_variant_tag(*v) for v in EB_SMOOTHING_VARIANTS]} "
              f"(noise_std for: {_ns}); sc8 arms "
              f"{[eb_variant_tag(*v) for v in A3S8_VARIANTS]} unaffected.", flush=True)
    if rank == 0 and SAVE_CATALOGUES:
        print(f"[rank 0] --save-catalogues: raw galaxy catalogues -> {OUTPUT_DIR}/catalogues/ "
              f"(~0.9 GB per mock at production n_eff — keep --num-sims small).", flush=True)
    if rank == 0 and RNG_SEED is not None:
        print(f"[rank 0] FIXED-RNG mode: --rng-seed {RNG_SEED}. Per-(sim,outer,rot) streams are "
              f"deterministic and split (matter fields independent of galaxy sampling), so runs "
              f"differing only in --galaxy-bias are PAIRED on the matter fields and nuisance "
              f"draws. The per-sim backend cache is bypassed to keep the stream position "
              f"resume-independent.", flush=True)
    # Create output and cache dirs on rank 0 only, then barrier: concurrent mkdir races on the
    # shared filesystem.
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
    if not SMOKE and SIMULATOR_TYPE != "external":
        # GowerStCosmologies reads cluster data on construction, so skip it for the smoke and the
        # external box.
        backend_states["gower_street"] = {
            "loader": GowerStCosmologies(gower_data_dir, csv_path),
        }
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
        displs_rows[i] = displs_rows[i-1] + counts_rows[i-1]

    counts = [r * cols for r in counts_rows]
    displs = [d * cols for d in displs_rows]

    recv_elems = counts_rows[rank] * cols
    recvbuf_flat = np.empty(recv_elems, dtype=np.float64)

    sendbuf_flat = sim_samples.flatten() if rank == 0 else None

    comm.Scatterv([sendbuf_flat, counts, displs, MPI.DOUBLE],
                recvbuf_flat,
                root=0)

    if cols > 0:
        recvbuf = recvbuf_flat.reshape(-1, cols)
    else:
        recvbuf = np.empty((0, cols), dtype=np.float64)

    print(f"[rank {rank}] received chunk (shape {recvbuf.shape}):\n{recvbuf}")

    sims =recvbuf[:, -1].astype(int) if recvbuf.size else np.array([], dtype=int)
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
                if OUTER_INDICES is not None and outer_idx not in OUTER_INDICES:
                    continue
                for rot_idx, rotation_spec in enumerate(rotation_specs):
                    if ROT_INDICES is not None and rot_idx not in ROT_INDICES:
                        continue
                    # Block-level resume: skip a complete (outer, rot) block, else clear its partial
                    # files and recompute it.
                    if not OVERWRITE:
                        if count_block_files(OUTPUT_DIR, sim_num, outer_idx, rot_idx) >= files_per_block:
                            print(f"[rank {rank}] sim {sim_num} block out{outer_idx}_rot{rot_idx} complete — skipping.")
                            continue
                        removed = remove_block_outputs(OUTPUT_DIR, sim_num, outer_idx, rot_idx)
                        if removed:
                            print(f"[rank {rank}] sim {sim_num} block out{outer_idx}_rot{rot_idx}: "
                                  f"removed {removed} partial files; recomputing.")
                    # Per-block streams. With --rng-seed they are deterministic in (seed, sim,
                    # outer, rot) and mutually independent.
                    rng, backend_rng, postproc_rng = build_block_rngs(
                        RNG_SEED, sim_num, outer_idx, rot_idx
                    )
                    if RNG_SEED is not None:
                        print(f"[rank {rank}] sim {sim_num} block out{outer_idx}_rot{rot_idx}: "
                              f"fixed-RNG seed={RNG_SEED} (streams split: sample/backend/postproc)",
                              flush=True)
                    log10_M_eff = rng.multivariate_normal(log10_M_eff_means, log10_M_eff_cov, size=1)[0]
                    if nuisance_pins.get('log10_M_eff') == 'prior_mean':
                        # Draw then overwrite: keeps the sample stream aligned with an un-pinned run,
                        # so --rng-seed pairing works across prior sets.
                        log10_M_eff = np.asarray(log10_M_eff_means, dtype=float).copy()
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

                    # Fixed-RNG: give the backend a fresh cache so its stream cannot depend on which
                    # block missed first.
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
                    elif SIMULATOR_TYPE == "external":
                        from src.external_mocks import FLAMINGO_BIG, prepare_external_backend
                        _ext_spec = dict(FLAMINGO_BIG)
                        if args.ext_data_dir:
                            _ext_spec["data_dir"] = args.ext_data_dir
                        backend = prepare_external_backend(
                            sim_num,
                            rng=backend_rng,
                            spec=_ext_spec,
                            prior_ranges=ia_prior_spec,
                            sim_grid=SIM_GRID,
                            variant=args.ext_variant,
                        )
                    else:
                        backend = prepare_gower_backend(
                            sim_num,
                            rng=backend_rng,
                            loader=backend_states["gower_street"]["loader"],
                            prior_ranges=ia_prior_spec,
                            sim_grid=SIM_GRID,
                        )

                    # Block-local copy: per-(outer, rot) keys must not leak into the per-sim cached param_dict.
                    param_dict = dict(backend["param_dict"])
                    shells = backend["shells"]
                    matter = backend["matter"]
                    cosmo = backend["cosmo"]

                    # Tag the saved param group with the IA model so the read side can disambiguate
                    # which (per-model) IA parameters are present.
                    param_dict["ia_model"] = IA_MODEL
                    # VD provenance code, stored as a float so the loader's float() read of cosmo_dict
                    # stays safe.
                    param_dict["vd_code"] = float(VD_CODE)

                    # IA params: the model tag plus only this model's sampled parameters.
                    ia_params = {
                        "model": IA_MODEL,
                        "f_red": f_red,
                        "log10_M_eff": log10_M_eff,
                    }
                    for _ia_key in ia_prior_spec:
                        ia_params[_ia_key] = param_dict[_ia_key]

                    # Compute the n(z) shift and tomo_nz before build_systematics: the VD objects are built from them.
                    shift_nz = model_shift_nz(SYSTEMATICS_MODEL)
                    zb = glass.shells.distance_grid(cosmo, zmin, zmax, dx=dx)
                    los_z_integration = np.linspace(zb[0], zb[-1], n_los_chi)
                    tomo_nz, dz_shift = calculate_tomo_nz(data_dir, n_los_chi, los_z_integration, shift_nz, return_shift=True)
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
                        # Shell z-edges for the LOS variable-depth fraction: GLASS windows overlap
                        # linearly, Gower windows are top-hats.
                        if SIMULATOR_TYPE == "gower_street":
                            zb_tuple = [(float(w.za[0]), float(w.za[-1])) for w in shells]
                        else:
                            zb_tuple = [(zb[k], zb[k + 1]) for k in range(len(zb) - 1)]
                        # The per-VD-bin n(z) must share the smoke's n_eff_scale on tomo_nz: the LOS
                        # fraction is dndz_vd / tomo_nz.
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
                            dz_shift=dz_shift,
                            vd_table=VD_TABLE,
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
                        # Theory-test mode applies no forward shear bias, so the 1/(1+m) de-bias in
                        # make_alm_shear_convergence must be off too.
                        m_bias_for_shear = np.zeros_like(m_bias)
                        # With systematics off, galaxy_bias is 0 (a clean theory test against a smooth field).
                        galaxy_bias_sim = 0.0
                    else:
                        m_bias_for_shear = m_bias
                        # --galaxy-bias overrides the config `bias` constant to make clustering variates.
                        galaxy_bias_sim = bias if args.galaxy_bias is None else args.galaxy_bias
                        if GALAXY_BIAS_PRIOR is not None:
                            # O3-diag prior: per-block 6-vector. The spawned child leaves the block's
                            # sample stream untouched.
                            galaxy_bias_sim = draw_galaxy_bias_prior(GALAXY_BIAS_PRIOR, rng)
                            print(f"[rank {rank}] sim {sim_num} block out{outer_idx}_rot{rot_idx}: "
                                  f"b_g draw {np.array2string(galaxy_bias_sim, precision=3)}",
                                  flush=True)

                    # Store scalar keys only: the ML reader does float(np.asarray(...)) per cosmo_dict key.
                    if np.ndim(galaxy_bias_sim) > 0:
                        for _bi, _bval in enumerate(np.asarray(galaxy_bias_sim, dtype=float)):
                            param_dict[f"b_g_bin{_bi + 1}"] = float(_bval)
                        param_dict["galaxy_bias_eff"] = float(
                            np.sum(n_arcmin2 * np.asarray(galaxy_bias_sim, dtype=float))
                            / np.sum(n_arcmin2))
                    else:
                        param_dict["galaxy_bias"] = float(galaxy_bias_sim)

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

                    if SIMULATOR_TYPE == "external":
                        # Shells stream off the external cube. The subclass builds its own windows
                        # from control.par, so no `matter` is passed.
                        from src.external_mocks import ShellCubeSimulator
                        simulator = ShellCubeSimulator(
                            backend["data_dir"], backend["cube_name"], **kwargs)
                    else:
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

                        # VD v2: per-galaxy weights W enter the estimator as in the real A2 build
                        # (weighted counts / smoothed-counts normalisation: sum W e / sum W per pixel).
                        _gal_w = ({"weights": catalogue["W"]} if catalogue.dtype.names and "W" in catalogue.dtype.names
                                  else {})
                        alm, alm_rand = make_alm_shear_convergence(
                            catalogue, m_bias_for_shear, nbins, nside, lmax, nosh=False, mask=mask,
                            normalization=SHEAR_NORMALIZATION, **_gal_w,
                            # Noise-only alm uses the block's postproc stream, not the global numpy RNG.
                            rng=postproc_rng,
                        )

                        mixed_cls = denoise_shear_cls(nbins, alm, alm_rand, lmax)
                        mixed_cut = mixed_cls[:, :, :, lower_lscale:upper_lscale+1]
                        cll_bands, mixed_bandpowers = compute_cl_bandpowers(
                            mixed_cut, nbins, lower_lscale, upper_lscale, nbands
                        )

                        # --- sc8 map product ---
                        # Built after the bandpowers so the 2-pt branch is untouched, and before `del catalogue`.
                        alm_sc8 = alm_rand_sc8 = None
                        if A3S8_VARIANTS:
                            sc8_rng = postproc_rng.spawn(1)[0]
                            alm_sc8, alm_rand_sc8 = make_alm_shear_convergence(
                                catalogue, m_bias_for_shear, nbins, nside, lmax, nosh=False,
                                mask=mask, normalization="smoothed_counts",
                                smoothed_counts_fwhm_arcmin=A3S8_FWHM_ARCMIN, **_gal_w,
                                rng=sc8_rng,
                            )

                        # Catalogue dump must precede `del catalogue` below.
                        if SAVE_CATALOGUES:
                            cat_path = save_catalogue_h5(
                                OUTPUT_DIR / "catalogues"
                                / f"catalogue_{sim_num}_out{outer_idx}_rot{rot_idx}_{cat_idx}.h5",
                                catalogue, param_dict,
                                extra_attrs={
                                    "sim_id": int(sim_num), "outer_idx": int(outer_idx),
                                    "rot_idx": int(rot_idx), "cat_idx": int(cat_idx),
                                    # Scalar for fixed b_g, (nbins,) under --galaxy-bias-prior.
                                    "galaxy_bias": np.asarray(galaxy_bias_sim, dtype=float),
                                    "shear_normalization": SHEAR_NORMALIZATION,
                                    "systematics_model": SYSTEMATICS_MODEL,
                                    "nside": int(nside), "nside_out": int(nside_out),
                                    # Offline map replay needs the m used for the 1/(1+m) de-bias.
                                    "m_bias_for_shear": np.asarray(m_bias_for_shear, dtype=float),
                                    "rng_seed": (-1 if RNG_SEED is None else int(RNG_SEED)),
                                },
                            )
                            print(f"[rank {rank}] wrote catalogue {cat_path.name} "
                                  f"({catalogue.shape[0]:,} galaxies)", flush=True)

                        del catalogue
                        gc.collect()

                        # E/B maps for each (fwhm, lmin, lcut) variant, keys as in EB_SMOOTHING_VARIANTS.
                        map_types = {}
                        # tag -> {patch_name: (nbins,) f64, 'all': (nbins,)}, as in patch_noise_std.
                        noise_std = {}
                        want_noise_std = (list(EB_SMOOTHING_VARIANTS)
                                          if NOISE_STD_VARIANTS is None
                                          else [tuple(v) for v in NOISE_STD_VARIANTS])
                        patch_names = list(named_patches.keys())

                        # sc8 branch first so its alms (~0.8 GB) are freed before the counts variants allocate.
                        sc8_tags = []
                        if alm_sc8 is not None:
                            # Every A3S8 variant re-filters the same sc8 alms.
                            for fwhm_v, lmin_v, lcut_v in A3S8_VARIANTS:
                                sc8_tag = f"sc8_{eb_variant_tag(fwhm_v, lmin_v, lcut_v)}"
                                sc8_tags.append(sc8_tag)
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
                            out_dtype = (A3S8_MAP_DTYPE if name.startswith("E_sc8_")
                                         else np.float32)
                            for patch_idx, patch_name in enumerate(patch_names):
                                # Primary maps stored as float32. The loader casts on read.
                                arr = pixelised_tomobin_patches[patch_idx].astype(
                                    out_dtype, copy=False)
                                if out_dtype is not np.float32 and not np.isfinite(arr).all():
                                    print(f"[rank {rank}] WARNING non-finite values in {name}"
                                          f"/{patch_name} after the {np.dtype(out_dtype).name} "
                                          f"cast (max|x| before cast = "
                                          f"{np.abs(pixelised_tomobin_patches[patch_idx]).max():.3e})",
                                          flush=True)
                                pixelised_results[name][patch_name] = arr

                        # Noise-meter scalars and provenance, additive to the legacy keys.
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
                        if GALAXY_BIAS_PRIOR is not None:
                            # String provenance lives outside cosmo_dict: a string dataset there breaks
                            # load_cosmo_params(cosmo_params=None).
                            _gbspec = GALAXY_BIAS_PRIORS[GALAXY_BIAS_PRIOR]
                            pixelised_results.setdefault("_provenance", {})["galaxy_bias_prior"] = (
                                f"preset={GALAXY_BIAS_PRIOR}; kappa={_gbspec['kappa']:g}; O3-diag "
                                "independent truncated Gaussians per tomo bin (Flamingo "
                                "calibration, data/galaxy_bias_priors.txt); realised values in "
                                "cosmo_dict/b_g_bin1..6 + galaxy_bias_eff (n_arcmin2-weighted)."
                            )
                        if noise_std:
                            if sc8_tags:
                                pixelised_results["_provenance"]["a3s8"] = (
                                    f"E_sc8_* built by a SECOND make_alm_shear_convergence call with "
                                    f"normalization='smoothed_counts', "
                                    f"smoothed_counts_fwhm_arcmin={A3S8_FWHM_ARCMIN:g}. Its "
                                    "random-rotation noise reference is an INDEPENDENT realisation "
                                    "of the counts branch's (spawned RNG stream) -- the random map "
                                    "is a noise meter only. Bandpowers are NOT affected: they are "
                                    "computed from the primary counts-normalised alms."
                                )
                                pixelised_results["_provenance"]["a3s8_variant"] = ",".join(sc8_tags)

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
            # A failed sim is logged and skipped. Its partial block is reclaimed on a later visit.
            print(f"[rank {rank}] sim {sim_num} FAILED: {e!r}; continuing to next sim.")
            traceback.print_exc()
            continue