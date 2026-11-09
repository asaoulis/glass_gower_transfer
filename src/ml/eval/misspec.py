"""Model-misspecification evaluation: one trained ensemble on the test split of many variate datasets.

The original training scalers are injected, never refit. VARIATE_SETS names the selectable sets
(gower_vdq, gower_vdq_hos, gower_bgp, gower_novd and others). run_misspecification_eval writes TARP
miscalibration, FoM and log-prob per variate under checkpoints/<exp>/misspec/<variate>/.
"""
import json
import os
import traceback
from copy import copy
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import torch

from ..data.data_loaders import H5CosmoDataset, build_nested_keys_from_quantities
from ..data.data_selection import (
    _filter_paths_by_shape_noise_idx,
    collect_paths,
    extract_cosmo_index,
)
from ..data.fixed_test_set import resolve_fixed_test_ids
from ..utils import (
    DataDictScalerTransform,
    TransformingDataset,
    build_ensemble_model_from_checkpoints,
    prepare_data_parameters,
)
from .evaluate_models import (
    DimNormalizedFoMDiagnostics,
    StandardFoMDiagnostics,
    TARPDiagnostics,
    _sample_from_prior,
    _split_samples_first_and_sims_first,
    rescale_parameters,
)
from .utils import (
    _config_preset_overrides,
    _parse_aug_id,
    _pop_credible_intervals,
    _save_posterior_samples,
    _to_json_compatible,
    build_gower_prior,
)

_GPU5 = "/share/gpu5/asaoulis/transfer_datasets"
_GPU4 = "/share/gpu4/asaoulis/transfer_datasets"

# exclude_params: parameters whose meaning differs between variate suites (a_ia for nla and nla_z)
# are dropped from calibration.
DEFAULT_VARIATES: List[Dict] = [
    {"name": "nla_m", "patterns": f"{_GPU5}/gower_mocks_nla_m_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5",
     "exclude_params": [], "in_distribution": True},
    {"name": "nla", "patterns": f"{_GPU4}/gower_mocks_nla_counts/output_*.h5", "exclude_params": ["a_ia"]},
    {"name": "nla_z", "patterns": f"{_GPU4}/gower_mocks_nla_z_counts/output_*.h5", "exclude_params": ["a_ia"]},
    {"name": "gb0p5", "patterns": f"{_GPU4}/gower_mocks_gb0p5_counts/output_*.h5", "exclude_params": []},
    {"name": "gb1p5", "patterns": f"{_GPU4}/gower_mocks_gb1p5_counts/output_*.h5", "exclude_params": []},
]

# GLASS pre-train variate suites (gpu5 f16 fwhm4_lmin56_lcut1400 prebakes, matching the GLASS z8
# foundation's maps). Their test ids come from the training split, since GLASS has no fixed lock.
GLASS_PRETRAIN_VARIATES: List[Dict] = [
    {"name": "glass_nla_m",
     "patterns": f"{_GPU5}/glass_mocks_nla_m_lmin50_f16_fwhm4_lmin56_lcut1400/output_*.h5",
     "exclude_params": [], "in_distribution": True},
    {"name": "glass_novd",
     "patterns": f"{_GPU5}/glass_mocks_nla_m_novd_f16_fwhm4_lmin56_lcut1400/output_*.h5",
     "exclude_params": []},
    {"name": "glass_nla",
     "patterns": f"{_GPU5}/glass_mocks_nla_f16_fwhm4_lmin56_lcut1400/output_*.h5",
     "exclude_params": ["a_ia"]},
    {"name": "glass_nla_z",
     "patterns": f"{_GPU5}/glass_mocks_nla_z_f16_fwhm4_lmin56_lcut1400/output_*.h5",
     "exclude_params": ["a_ia"]},
]

# --- no-VD suite ---
NOVD_GOWER_VARIATES: List[Dict] = [
    {"name": "nla_m", "patterns": f"{_GPU5}/gower_mocks_nla_m_novd_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5",
     "exclude_params": [], "in_distribution": True},
    {"name": "nla", "patterns": f"{_GPU4}/gower_mocks_nla_novd_counts/output_*.h5", "exclude_params": ["a_ia"]},
    {"name": "nla_z", "patterns": f"{_GPU4}/gower_mocks_nla_z_novd_counts/output_*.h5", "exclude_params": ["a_ia"]},
    {"name": "gb0p5", "patterns": f"{_GPU4}/gower_mocks_gb0p5_novd_counts/output_*.h5", "exclude_params": []},
    {"name": "gb1p5", "patterns": f"{_GPU4}/gower_mocks_gb1p5_novd_counts/output_*.h5", "exclude_params": []},
]

NOVD_GLASS_PRETRAIN_VARIATES: List[Dict] = [
    {"name": "glass_nla_m",
     "patterns": f"{_GPU5}/glass_mocks_nla_m_novd_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5",
     "exclude_params": [], "in_distribution": True},
    # Same in-distribution physics, read from the unbaked gpu4 source instead of the gpu5 f16 prebake.
    {"name": "glass_nla_m_raw",
     "patterns": f"{_GPU4}/glass_mocks_nla_m_novd_counts/output_*.h5",
     "exclude_params": []},
    {"name": "glass_gb0p5", "patterns": f"{_GPU4}/glass_mocks_gb0p5_novd_counts/output_*.h5",
     "exclude_params": []},
    {"name": "glass_gb1p5", "patterns": f"{_GPU4}/glass_mocks_gb1p5_novd_counts/output_*.h5",
     "exclude_params": []},
    {"name": "glass_nla", "patterns": f"{_GPU4}/glass_mocks_nla_novd_counts/output_*.h5",
     "exclude_params": ["a_ia"]},
    {"name": "glass_nla_z", "patterns": f"{_GPU4}/glass_mocks_nla_z_novd_counts/output_*.h5",
     "exclude_params": ["a_ia"]},
]

# --- dual-normalisation arm comparison ---
# One variate set per shear-processing arm, read from the same arm's bake as the model:
#   glass_dn_{nla_m|gb0p7|gb1p0|gb1p5}_f16_{a0|a1|sc8|sc8a1}_fwhm4_lmin56_lcut1400
_DN_EB = "fwhm4_lmin56_lcut1400"


def _dn_variate_set(arm_tag: str) -> List[Dict]:
    def store(which):
        return f"{_GPU5}/glass_dn_{which}_f16_{arm_tag}_{_DN_EB}/output_*.h5"
    return [
        {"name": f"glass_dn_{arm_tag}", "patterns": store("nla_m"),
         "exclude_params": [], "in_distribution": True},
        {"name": "glass_gb0p7", "patterns": store("gb0p7"), "exclude_params": []},
        {"name": "glass_gb1p0", "patterns": store("gb1p0"), "exclude_params": []},
        {"name": "glass_gb1p5", "patterns": store("gb1p5"), "exclude_params": []},
    ]


def _bgp_variate_set(arm_tag: str) -> List[Dict]:
    """BGP campaign (b_g marginalised at generation): the b_g ladder of _dn_variate_set.

    The probes 0.7, 1.0 and 1.5 share --rng-seed 4242, so they pair per event against gb1p0.
    """
    def store(which):
        return f"{_GPU5}/glass_bgp_{which}_f16_{arm_tag}_{_DN_EB}/output_*.h5"
    return [
        {"name": f"glass_bgp_{arm_tag}", "patterns": store("nla_m"),
         "exclude_params": [], "in_distribution": True},
        {"name": "glass_gb0p7", "patterns": store("gb0p7"), "exclude_params": []},
        {"name": "glass_gb1p0", "patterns": store("gb1p0"), "exclude_params": []},
        {"name": "glass_gb1p5", "patterns": store("gb1p5"), "exclude_params": []},
    ]


# --- BGP campaign, Gower side (the production 9-param sc8a1 chain's variates) ---
# All bakes carry bare `E` groups, matching the chain's eb_map_variant=None.
_BGP_EB_TAG = "sc8a1_fwhm4_lmin56_lcut1400"
GOWER_BGP_VARIATES: List[Dict] = [
    {"name": "gower_bgp_nla_m", "patterns": f"{_GPU5}/gower_bgp_nla_m_f16_{_BGP_EB_TAG}/output_*.h5",
     "exclude_params": [], "in_distribution": True},
    {"name": "gower_nla", "patterns": f"{_GPU5}/gower_bgp_nla_f16_{_BGP_EB_TAG}/output_*.h5", "exclude_params": ["a_ia"]},
    {"name": "gower_nla_z", "patterns": f"{_GPU5}/gower_bgp_nla_z_f16_{_BGP_EB_TAG}/output_*.h5", "exclude_params": ["a_ia"]},
    {"name": "gower_gb1p0", "patterns": f"{_GPU5}/gower_bgp_gb1p0_f16_{_BGP_EB_TAG}/output_*.h5", "exclude_params": []},
    {"name": "gower_gb1p3", "patterns": f"{_GPU5}/gower_bgp_gb1p3_f16_{_BGP_EB_TAG}/output_*.h5", "exclude_params": []},
    {"name": "gower_vd", "patterns": f"{_GPU5}/gower_bgp_nla_m_vd_f16_{_BGP_EB_TAG}/output_*.h5", "exclude_params": []},
    # kappa=2 galaxy-bias prior: the kappa=1 per-bin means with double the sigmas (truncated at +-3*kappa*sigma).
    {"name": "gower_gbk2", "patterns": f"{_GPU5}/gower_bgpk2_nla_m_f16_{_BGP_EB_TAG}/output_*.h5", "exclude_params": []},
]


# --- external boxes: GowerSt2 and Flamingo_big (DMO and FLAMINGO-baryonified) ---
GOWER_EXT_VARIATES: List[Dict] = [
    {"name": "gower_ext_dmo", "patterns": f"{_GPU5}/ext_dmo_f16_{_BGP_EB_TAG}/output_*.h5",
     "exclude_params": []},
    {"name": "gower_ext_bary", "patterns": f"{_GPU5}/ext_bary_f16_{_BGP_EB_TAG}/output_*.h5",
     "exclude_params": []},
]


# --- VDQ campaign: variable depth on (quadcap table), b_g kappa=1 ---
GOWER_VDQ_VARIATES: List[Dict] = [
    {"name": "gower_vdq_nla_m", "patterns": f"{_GPU5}/gower_vdq_nla_m_f16_{_BGP_EB_TAG}/output_*.h5",
     "exclude_params": [], "in_distribution": True},
    {"name": "gower_vdq_gb1p0", "patterns": f"{_GPU5}/gower_vdq_gb1p0_f16_{_BGP_EB_TAG}/output_*.h5",
     "exclude_params": []},
    {"name": "gower_vdq_nla", "patterns": f"{_GPU5}/gower_vdq_nla_f16_{_BGP_EB_TAG}/output_*.h5",
     "exclude_params": ["a_ia"]},
    {"name": "gower_vdq_nla_z", "patterns": f"{_GPU5}/gower_vdq_nla_z_f16_{_BGP_EB_TAG}/output_*.h5",
     "exclude_params": ["a_ia"]},
    # b_g fixed at 1.3, rng-seed 4242, paired with gb1p0.
    {"name": "gower_vdq_gb1p3", "patterns": f"{_GPU5}/gower_vdq_gb1p3_f16_{_BGP_EB_TAG}/output_*.h5",
     "exclude_params": []},
    {"name": "gower_vdq_gbk2", "patterns": f"{_GPU5}/gower_vdqk2_nla_m_f16_{_BGP_EB_TAG}/output_*.h5",
     "exclude_params": []},
    {"name": "gower_vdqk2_nla", "patterns": f"{_GPU5}/gower_vdqk2_nla_f16_{_BGP_EB_TAG}/output_*.h5",
     "exclude_params": ["a_ia"]},
    {"name": "gower_vd2pr_nla_m_p10k",
     "patterns": f"{_GPU5}/gower_vd2pr_nla_m_p10k_f16_{_BGP_EB_TAG}/output_*.h5", "exclude_params": []},
]


# The in-distribution VDQ variate read from its HOS sidecars, which share the map store's basenames.
def _hos_variates():
    from config.kids_legacy_hos import _HOS_GOWER
    return [{"name": "gower_vdq_nla_m", "patterns": _HOS_GOWER, "exclude_params": [], "in_distribution": True}]


def _hosx_variates():
    """The same in-distribution variate read from the hos_x sidecars (R1+Gmrg arm)."""
    from config.kids_legacy_hos import _HOSX_GOWER
    return [{"name": "gower_vdq_nla_m", "patterns": _HOSX_GOWER, "exclude_params": [], "in_distribution": True}]


VARIATE_SETS: Dict[str, List[Dict]] = {
    "gower_vdq": GOWER_VDQ_VARIATES,
    "gower_vdq_hos": _hos_variates(),
    "gower_vdq_hosx": _hosx_variates(),
    "gower_bgp": GOWER_BGP_VARIATES,
    "gower_bgp_ext": GOWER_BGP_VARIATES + GOWER_EXT_VARIATES,
    "gower_ext": GOWER_EXT_VARIATES,
    "gower": DEFAULT_VARIATES,
    "glass_pretrain": GLASS_PRETRAIN_VARIATES,
    "gower_novd": NOVD_GOWER_VARIATES,
    "glass_pretrain_novd": NOVD_GLASS_PRETRAIN_VARIATES,
    "glass_dn_a0": _dn_variate_set("a0"),
    "glass_dn_a1": _dn_variate_set("a1"),
    "glass_dn_b1": _dn_variate_set("a0"),      # A0 store + the loader knob
    "glass_dn_sc8": _dn_variate_set("sc8"),
    "glass_dn_sc8a1": _dn_variate_set("sc8a1"),
    "glass_bgp_sc8a1": _bgp_variate_set("sc8a1"),
    "glass_bgp_sc8": _bgp_variate_set("sc8"),
    # R1024: the conservative scale cut, 8 arcmin beam, lcut 1024.
    "glass_dn_r1024": [
        {"name": "glass_dn_r1024",
         "patterns": f"{_GPU5}/glass_dn_nla_m_f16_a0_fwhm8_lmin56_lcut1024/output_*.h5",
         "exclude_params": [], "in_distribution": True},
    ] + [
        {"name": f"glass_gb{b}",
         "patterns": f"{_GPU5}/glass_dn_gb{b}_f16_a0_fwhm8_lmin56_lcut1024/output_*.h5",
         "exclude_params": []}
        for b in ("0p7", "1p0", "1p5")
    ],
}


def _load_experiment_config(experiment_name: str):
    """Rebuild the experiment config exactly like eval.py's load_config + list-branch handling."""
    from config.default import get_default_config
    from config.experiments import experiments as base_experiments
    from config.paper_2026 import paper_2026_experiments
    from config.kids_legacy import kids_legacy_experiments
    from config.kids_legacy_novd import kids_legacy_novd_experiments
    from config.kids_legacy_bgp import kids_legacy_bgp_experiments
    from config.kids_legacy_vdq import kids_legacy_vdq_experiments
    from config.kids_legacy_hos import kids_legacy_hos_experiments

    exps = dict(base_experiments)
    exps.update(paper_2026_experiments)
    exps.update(kids_legacy_experiments)
    exps.update(kids_legacy_novd_experiments)
    exps.update(kids_legacy_bgp_experiments)
    exps.update(kids_legacy_vdq_experiments)
    exps.update(kids_legacy_hos_experiments)
    experiment_config = exps[experiment_name]

    config = get_default_config()
    config.experiment_name = experiment_name
    for key, val in experiment_config.items():
        if key == "max_trainval_cosmos":
            continue
        setattr(config, key, val)

    max_tv = experiment_config.get("max_trainval_cosmos", None)
    if isinstance(max_tv, (list, tuple)):
        if len(max_tv) != 1:
            print(f"[misspec] WARNING: max_trainval_cosmos sweep {max_tv}; using first entry.")
        config.max_trainval_cosmos = int(max_tv[0])
        config.match_num_cosmo = True  # match eval.py: match_string includes ncosmo
    elif max_tv is not None:
        config.max_trainval_cosmos = int(max_tv)
    return config


def _probe_variate_file(paths: Sequence[str], nested_keys: Dict, cosmo_param_names: Sequence[str]):
    """Open the first readable file and report (missing_data_keys, present_cosmo_params).

    Fails fast, on a wrong eb variant tag for one, instead of letting the loader skip every file.
    """
    last_err = None
    for p in paths[:16]:
        try:
            with h5py.File(p, "r") as f:
                missing = []
                for out_key, path in nested_keys.items():
                    node = f
                    for key in path:
                        if key not in node:
                            missing.append((out_key, "/".join(path)))
                            node = None
                            break
                        node = node[key]
                grp = f["cosmo_dict"]
                present_cosmo = [c for c in cosmo_param_names if c in grp]
            return missing, present_cosmo, p
        except OSError as e:  # truncated/corrupt file — try the next one
            last_err = e
            continue
    raise RuntimeError(f"No readable file among first {min(16, len(paths))} probe paths "
                       f"(last error: {last_err})")


def build_variate_test_loader(
    patterns,
    nested_keys: Dict,
    cosmo_param_names: Sequence[str],
    key_scalers: Dict,
    cosmo_scaler,
    test_id_pool,
    test_shape_noise_idx=(0, (0, 1)),
    batch_size: int = 64,
    num_workers: int = 4,
    max_test_files: Optional[int] = None,
    eb_noise_norm: Optional[str] = None,
):
    """Variate test loader with the original scalers injected (never refit). Returns (loader, meta).

    Test cosmologies are test_id_pool intersected with the on-disk ids, or every on-disk cosmology
    when they do not overlap. max_test_files caps the set by whole cosmologies in sim_id order.
    eb_noise_norm mirrors the training loader and runs before the key scalers.
    """
    all_paths = collect_paths(patterns)
    by_cosmo: Dict[int, List[str]] = {}
    for p in all_paths:
        by_cosmo.setdefault(extract_cosmo_index(p), []).append(p)
    for c in by_cosmo:
        by_cosmo[c].sort()

    pool = set(test_id_pool or [])
    test_ids = sorted(set(by_cosmo.keys()) & pool)
    used_fallback = False
    if not test_ids:
        test_ids = sorted(by_cosmo.keys())
        used_fallback = True
        print(f"[misspec] WARNING: no overlap with the {len(pool)} test-pool ids; "
              f"falling back to ALL {len(test_ids)} on-disk cosmologies (model may have seen "
              "these cosmologies at nla_m physics).", flush=True)

    test_paths = [p for c in test_ids for p in by_cosmo[c]]
    if test_shape_noise_idx is None:
        # Explicit opt-out: None keeps every file, for small purpose-built stores.
        print(f"[misspec] shape-noise filter DISABLED: using all {len(test_paths)} "
              "test-cosmology files.", flush=True)
        filtered = test_paths
    else:
        filtered = _filter_paths_by_shape_noise_idx(test_paths, list(test_shape_noise_idx))
        if not filtered:
            print(f"[misspec] WARNING: shape-noise filter {test_shape_noise_idx} matched no files; "
                  "using all test-cosmology files.", flush=True)
            filtered = test_paths

    if max_test_files is not None and len(filtered) > int(max_test_files):
        by_id: Dict[int, List[str]] = {}
        for p in filtered:
            by_id.setdefault(extract_cosmo_index(p), []).append(p)
        capped, kept_ids = [], []
        for c in sorted(by_id):
            if capped and len(capped) + len(by_id[c]) > int(max_test_files):
                break
            capped.extend(by_id[c])
            kept_ids.append(c)
        print(f"[misspec] max_test_files={max_test_files}: capped {len(filtered)} -> "
              f"{len(capped)} files ({len(kept_ids)}/{len(test_ids)} cosmologies, "
              "first by sorted sim_id).", flush=True)
        filtered = capped
        test_ids = kept_ids

    loader = _wrap_paths_as_loader(
        filtered, nested_keys, cosmo_param_names, key_scalers, cosmo_scaler,
        batch_size=batch_size, num_workers=num_workers, eb_noise_norm=eb_noise_norm,
    )
    meta = {
        "n_test_files": len(filtered),
        "n_test_cosmologies": len(test_ids),
        "test_ids_from_fixed_lock": not used_fallback,
        "test_paths": filtered,
    }
    return loader, meta


def _wrap_paths_as_loader(
    paths: Sequence[str],
    nested_keys: Dict,
    cosmo_param_names: Sequence[str],
    key_scalers: Dict,
    cosmo_scaler,
    *,
    batch_size: int = 64,
    num_workers: int = 4,
    eb_noise_norm: Optional[str] = None,
):
    """Plain loader (no augmentation, no shuffle) over explicit file paths with the original
    scalers injected. Shared by build_variate_test_loader and the summary extraction."""
    ds = H5CosmoDataset(
        list(paths),
        nested_keys,
        list(cosmo_param_names),
        transform=None,
        allow_missing_cosmo_params=True,
    )
    data_transform = DataDictScalerTransform(key_scalers)
    if eb_noise_norm:
        from ..data.data_augmentations import EBNoiseNormTransform, ChainedDataTransform
        data_transform = ChainedDataTransform([EBNoiseNormTransform(eb_noise_norm), data_transform])
    wrapped = TransformingDataset(ds, data_transform=data_transform, cosmo_scaler=cosmo_scaler)
    loader = torch.utils.data.DataLoader(
        wrapped, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False,
    )
    return loader


def resolve_shared_pool(variates: List[Dict]):
    """test_id_source='shared': the sim_ids common to every out-of-distribution variate on disk."""
    ood = [v for v in variates if not v.get("in_distribution")] or variates
    per_variate_ids = []
    for v in ood:
        ids = {extract_cosmo_index(p) for p in collect_paths(v["patterns"])}
        print(f"[misspec] shared-pool: {v['name']} has {len(ids)} cosmologies on disk", flush=True)
        if not ids:
            # An absent store must not empty the intersection. That variate fails soft later.
            print(f"[misspec] shared-pool: WARNING {v['name']} has NO files on disk — excluded from the pool", flush=True)
            continue
        per_variate_ids.append(ids)
    shared_pool = set.intersection(*per_variate_ids) if per_variate_ids else set()
    if not shared_pool:
        raise RuntimeError(
            "test_id_source='shared' but the out-of-distribution variates share no sim_ids: "
            f"{[len(s) for s in per_variate_ids]}"
        )
    print(f"[misspec] shared-pool: {len(shared_pool)} cosmologies common to {[v['name'] for v in ood]}", flush=True)
    return shared_pool


def derive_test_id_pool(cfg, in_dist_test_loader, repeat_index):
    """Variate test cosmologies: the lock file when the experiment pins one, else the experiment's
    own held-out test cosmologies, which are the same for every repeat."""
    from .utils import _resolve_test_paths
    lock_spec = getattr(cfg, "fixed_test_sim_ids", None)
    if lock_spec:
        return set(resolve_fixed_test_ids(lock_spec) or [])
    held_out = _resolve_test_paths(in_dist_test_loader) or []
    pool = {extract_cosmo_index(p) for p in held_out}
    print(f"[misspec] repeat {repeat_index}: derived test-id pool from the training split "
          f"({len(pool)} held-out cosmologies).", flush=True)
    return pool


def _compute_misspec_metrics(
    theta0s: torch.Tensor,
    samples: torch.Tensor,
    param_names: Sequence[str],
    available_idx: Sequence[int],
    param_scaler,
    prior_samples_scaled: Optional[torch.Tensor],
):
    """Calibration on the available dims, and FoM against the base prior on all sampled dims.

    theta0s [N, D] and samples [S, N, D] are in scaled (loader and flow) space, as in
    run_evaluation_on_samples.
    """
    param_names = list(param_names)
    available_idx = list(available_idx)
    available_names = [param_names[i] for i in available_idx]
    n_sims = theta0s.shape[0]

    _, samples_sims_first = _split_samples_first_and_sims_first(samples, n_sims, name="samples")
    scaled_theta0s = rescale_parameters(theta0s, param_scaler)          # physical units
    scaled_samples = rescale_parameters(samples, param_scaler)
    scaled_samples_first, scaled_samples_sims_first = _split_samples_first_and_sims_first(
        scaled_samples, n_sims, name="scaled_samples"
    )

    metrics: Dict = {"available_params": available_names}

    # TARP calibration on the available dims only.
    tarp = TARPDiagnostics(available_names, bootstrap=True, num_bootstrap=25, seed=None)
    metrics.update(tarp.compute_all(
        samples_sims_first[:, :, available_idx].contiguous(),
        theta0s[:, available_idx].contiguous(),
    ))

    # FoM against the base Gower prior over all sampled dims.
    prior_samples_unscaled = (
        rescale_parameters(prior_samples_scaled, param_scaler)
        if prior_samples_scaled is not None else None
    )
    metrics.update(DimNormalizedFoMDiagnostics(
        param_names, prior_samples_t=prior_samples_scaled,
    ).compute_all(samples_sims_first))
    metrics.update(StandardFoMDiagnostics(
        param_names, prior_samples_t=prior_samples_unscaled,
    ).compute_all(scaled_samples_sims_first))

    # Point statistics per available parameter, in physical units.
    sample_means = scaled_samples_first.mean(axis=0)
    bias = sample_means - scaled_theta0s
    std_devs = scaled_samples_first.std(axis=0)
    # Quantiles on CPU: torch.quantile's sort buffer runs a 16 GiB GPU out of memory.
    _ssf_cpu = scaled_samples_first.cpu()
    width_68 = (torch.quantile(_ssf_cpu, 0.84, dim=0)
                - torch.quantile(_ssf_cpu, 0.16, dim=0)).to(scaled_samples_first.device)
    for dim in available_idx:
        name = param_names[dim]
        metrics[name] = {
            "mse": ((bias[:, dim] ** 2).mean()).item(),
            "bias": bias[:, dim].mean().item(),
            "std_dev": std_devs[:, dim].mean().item(),
            "width_68": width_68[:, dim].mean().item(),
        }
    return metrics


def run_misspecification_eval(
    base_experiment: str = "gower_npe_finetune_nla_m_z8",
    repeat_index: Optional[int] = None,
    variates: Optional[List[Dict]] = None,
    num_samples: int = 10000,
    prior_num_samples: int = 20_000,
    test_shape_noise_idx=(0, (0, 1)),
    out_subdir: str = "misspec",
    repeat_indices: Sequence[int] = (0,),
    variate_set: Optional[str] = None,
    variate_names: Optional[Sequence[str]] = None,
    max_test_files: Optional[int] = None,
    test_id_source: str = "heldout",
):
    """Evaluate the base experiment's model(s) on every variate, per training repeat.

    Handles eval-time ensembles and single-model experiments. repeat_indices runs one full pass
    per repeat. With more than one repeat the per-event cross-repeat posterior disagreement is
    saved per variate. repeat_index is a single-repeat alias. Test cosmologies come from
    derive_test_id_pool, or the shared pool when test_id_source='shared'. max_test_files caps
    each variate's test set.
    """
    from ..models.utils import apply_repeat_config
    from .utils import _resolve_test_paths, load_best_model_and_build_posterior

    if variates is None:
        variates = VARIATE_SETS[variate_set] if variate_set else DEFAULT_VARIATES
    if variate_names:
        wanted = list(dict.fromkeys(variate_names))
        available = [v["name"] for v in variates]
        missing = [n for n in wanted if n not in available]
        if missing:
            raise KeyError(
                f"variate name(s) {missing} not in set '{variate_set or 'gower'}'; "
                f"available: {available}"
            )
        variates = [v for v in variates if v["name"] in set(wanted)]
        print(f"[misspec] variate filter: {available} -> {[v['name'] for v in variates]}",
              flush=True)
    if repeat_index is not None:
        repeat_indices = (int(repeat_index),)
    repeat_indices = [int(r) for r in repeat_indices]

    # A shared-cosmology run must never overwrite the strict held-out run's results.
    if test_id_source == "shared" and out_subdir == "misspec":
        out_subdir = "misspec_shared"

    # Flushed setup markers locate a crash that leaves no traceback in the SLURM log.
    print(f"[misspec] setup 1/4: loading config for '{base_experiment}'", flush=True)
    cfg0 = _load_experiment_config(base_experiment)
    param_names = list(cfg0.cosmo_param_names)
    eb_variant = getattr(cfg0, "eb_map_variant", None)
    nested_keys = build_nested_keys_from_quantities(list(cfg0.dataset_quantities), eb_variant)
    out_root = os.path.join(cfg0.base_path, "checkpoints", cfg0.experiment_name, out_subdir)
    print(f"[misspec] setup 2/4: config OK — params={param_names} eb_variant={eb_variant} "
          f"out_root={out_root}", flush=True)

    # One repeat-independent Gower prior and prior sample set, the FoM shrinkage reference.
    prior = build_gower_prior(param_names, preset_overrides=_config_preset_overrides(cfg0))
    print("[misspec] setup 3/4: Gower prior built", flush=True)
    prior_samples_scaled = _sample_from_prior(prior, prior_num_samples, target_dim=len(param_names))
    print(f"[misspec] setup 4/4: drew {prior_num_samples} prior samples "
          f"{tuple(prior_samples_scaled.shape)}", flush=True)

    # test_id_source='shared': evaluate every variate, the in-distribution reference included, on
    # the cosmologies the OOD variates have on disk. It complements the held-out run, never replaces it.
    shared_pool = None
    if test_id_source == "shared":
        shared_pool = resolve_shared_pool(variates)
    elif test_id_source != "heldout":
        raise ValueError(f"test_id_source must be 'heldout' or 'shared', got {test_id_source!r}")

    summary: Dict[str, Dict] = {}
    per_variate_repeat: Dict[str, Dict[str, Dict]] = {v["name"]: {} for v in variates}
    match_strings = []
    for r in repeat_indices:
        # Fresh config per repeat: apply_repeat_config mutates split_seed in place.
        cfg = _load_experiment_config(base_experiment)
        # Same test-point sub-selection for the in-distribution reference as for every
        # variate (rot0, inner noise {0,1}) so coverage curves are directly comparable.
        cfg.test_shape_noise_idx = (None if test_shape_noise_idx is None
                                    else list(test_shape_noise_idx))
        repeat_match, _ = apply_repeat_config(cfg, r)
        cfg.match_string = repeat_match
        match_strings.append(repeat_match)
        print(f"[misspec] base experiment '{base_experiment}' repeat={r} "
              f"match_string={cfg.match_string} params={param_names} eb_variant={eb_variant}",
              flush=True)

        # This repeat's original scalers, fit on its train+val split.
        scalers, _, _, in_dist_test_loader = prepare_data_parameters(cfg)
        orig_key_scalers = scalers["data"]
        orig_cosmo_scaler = scalers["cosmo"]
        print(f"[misspec] repeat {r}: original scalers rebuilt from {cfg.data_patterns} "
              f"(data keys: {sorted(orig_key_scalers)})", flush=True)

        test_id_pool = derive_test_id_pool(cfg, in_dist_test_loader, r)

        if test_id_source == "shared":
            test_id_pool = shared_pool
            print(f"[misspec] repeat {r}: test-id-source=shared -> every variate (including the "
                  f"in-distribution reference) uses the SAME {len(test_id_pool)} cosmologies.",
                  flush=True)

        # The repeat's model is built once and its loaders are swapped per variate.
        n_ens = int(getattr(cfg, "ensemble_repeats", 1) or 1)
        if n_ens > 1:
            model = build_ensemble_model_from_checkpoints(
                cfg,
                in_dist_test_loader,
                match_string=cfg.match_string,
                member_test_loaders=[in_dist_test_loader] * n_ens,
            )
        else:
            loaded = load_best_model_and_build_posterior(
                cfg, ds_string_match=cfg.match_string, data_parameters=in_dist_test_loader,
            )
            model = loaded[0] if loaded else None
        if model is None:
            print(f"[misspec] repeat {r}: no '{base_experiment}' {cfg.match_string} checkpoints "
                  "yet — skipping this repeat.", flush=True)
            summary[f"repeat{r}"] = {"error": f"no checkpoints for {cfg.match_string}"}
            continue

        for variate in variates:
            name = variate["name"]
            key = f"{name}@r{r}" if len(repeat_indices) > 1 else name
            try:
                result = _eval_one_variate(
                    variate, model, cfg, nested_keys, param_names,
                    orig_key_scalers, orig_cosmo_scaler, prior_samples_scaled,
                    num_samples=num_samples,
                    test_shape_noise_idx=test_shape_noise_idx,
                    out_dir=os.path.join(out_root, name),
                    repeat_index=r,
                    test_id_pool=test_id_pool,
                    max_test_files=max_test_files,
                )
                per_variate_repeat[name][repeat_match] = result.pop("_per_event")
                summary[key] = result
            except Exception as e:
                print(f"[misspec] {name} (repeat {r}): FAILED — {type(e).__name__}: {e}",
                      flush=True)
                traceback.print_exc()
                summary[key] = {"error": f"{type(e).__name__}: {e}"}
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Cross-repeat posterior disagreement per variate.
    if len(repeat_indices) > 1:
        for variate in variates:
            name = variate["name"]
            reps = per_variate_repeat[name]
            if len(reps) < 2:
                continue
            try:
                dis = _compute_repeat_disagreement(name, reps, os.path.join(out_root, name))
                for key in list(summary):
                    if key.startswith(f"{name}@"):
                        summary[key]["repeat_kl_mean"] = dis["kl_mean"]
            except Exception as e:
                print(f"[misspec] {name}: disagreement FAILED — {type(e).__name__}: {e}",
                      flush=True)
                traceback.print_exc()

    print("\n[misspec] ============ SUMMARY ============", flush=True)
    for key, res in summary.items():
        if "error" in res:
            print(f"[misspec] {key}: ERROR {res['error']}", flush=True)
        else:
            extra = (f" repeat_kl_mean={res['repeat_kl_mean']:.4f}"
                     if "repeat_kl_mean" in res else "")
            print(f"[misspec] {key}: n_test={res['n_test_files']} "
                  f"cal_full={res['cal_full']:.4f} cal_om_s8_w0={res['cal_om_s8_w0']:.4f}"
                  f"{extra} available={res['available_params']}", flush=True)
    return summary


_EXTRA_COSMO_PREFIXES = ("b_g_bin", "galaxy_bias")


def _read_nuisance_extras(test_paths) -> Dict[str, np.ndarray]:
    """Per-event scalar cosmo_dict entries that are not inference parameters (b_g_bin1..6 and
    galaxy_bias*), as extra_<key> float64 arrays [N], NaN where a file lacks the key. Returns {} on
    any failure."""
    try:
        vals: Dict[str, List[float]] = {}
        for i, p in enumerate(test_paths):
            try:
                f = h5py.File(p, "r")
            except OSError:  # unreadable file -> NaN row, keep the rest
                continue
            with f:
                grp = f["cosmo_dict"] if "cosmo_dict" in f else {}
                for k in grp:
                    if not k.startswith(_EXTRA_COSMO_PREFIXES):
                        continue
                    try:
                        v = float(np.asarray(grp[k][()]).reshape(-1)[0])
                    except (TypeError, ValueError, IndexError):
                        continue
                    vals.setdefault(k, [np.nan] * len(test_paths))[i] = v
        return {f"extra_{k}": np.asarray(v, dtype=np.float64) for k, v in vals.items()}
    except Exception as e:  # never let a diagnostic dump break the eval
        print(f"[save-moments] WARNING: nuisance extras not read: {e}", flush=True)
        return {}


def _save_posterior_moments(out_path, theta0s, samples, test_paths, param_names):
    """Compact per-event companion to the full posterior-sample dump (KB, not hundreds of MB).

      mean, std, theta0, z  [N, D] float32   with  z = (theta0 - mean) / std
      cov                   [N, D, D] float32  the full per-event sample covariance
      sim_ids, aug_ids, test_files [N], params [D], and extra_<key> nuisance columns

    Positive z means the truth sits above the posterior mean. Moments are in scaled space.
    A failure only warns.
    """
    try:
        samp = samples.detach().cpu().numpy() if hasattr(samples, "detach") else np.asarray(samples)
        theta = theta0s.detach().cpu().numpy() if hasattr(theta0s, "detach") else np.asarray(theta0s)
        mean = samp.mean(axis=0)
        std = samp.std(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (theta - mean) / std
        # Per-event covariance from samp [S, N, D]: [N, D, D] with np.cov's (S - 1) normalisation.
        centred = samp - mean[None, :, :]
        cov = np.einsum("sni,snj->nij", centred, centred) / max(samp.shape[0] - 1, 1)
        payload = {
            "mean": mean.astype(np.float32), "std": std.astype(np.float32),
            "theta0": theta.astype(np.float32), "z": z.astype(np.float32),
            "cov": cov.astype(np.float32),
            "params": np.array(list(param_names)),
        }
        if test_paths is not None and len(test_paths) == theta.shape[0]:
            files = [os.path.basename(p) for p in test_paths]
            payload["test_files"] = np.array(files)
            payload["sim_ids"] = np.array([extract_cosmo_index(p) for p in test_paths], dtype=np.int64)
            payload["aug_ids"] = np.array([_parse_aug_id(f) for f in files], dtype=np.int64)
            payload.update(_read_nuisance_extras(test_paths))
        np.savez_compressed(out_path, **payload)
        print(f"[save-moments] wrote {out_path}  (N={theta.shape[0]}, keys={sorted(payload)})",
              flush=True)
    except Exception as e:  # never let a diagnostic dump break the eval
        print(f"[save-moments] WARNING: failed to save posterior moments to {out_path}: {e}",
              flush=True)


def _compute_repeat_disagreement(name: str, per_repeat: Dict[str, Dict], out_dir: str):
    """Per-event posterior disagreement across training repeats for one variate.

    Events are aligned by test-file basename and the per-repeat moments stacked to [K, N, D].
    Each event is scored by the mean pairwise symmetric diag-Gaussian KL, in scaled space, and
    the result saved next to the per-repeat calibration JSONs."""
    from .ensemble_discrepancies import diag_gaussian_symmetric_kl

    matches = sorted(per_repeat)
    common = set(per_repeat[matches[0]]["test_files"])
    for m in matches[1:]:
        common &= set(per_repeat[m]["test_files"])
    common = sorted(common)
    if not common:
        raise RuntimeError("no common test files across repeats")

    mu, var = [], []
    for m in matches:
        d = per_repeat[m]
        idx = {f: i for i, f in enumerate(d["test_files"])}
        sel = [idx[f] for f in common]
        mu.append(d["mu"][sel])
        var.append(d["var"][sel])
    mu = np.stack(mu, axis=0)    # [K, N, D]
    var = np.stack(var, axis=0)  # [K, N, D]

    kl = diag_gaussian_symmetric_kl(mu, var)  # [N]
    payload = {
        "variate": name,
        "repeat_match_strings": matches,
        "n_events": int(len(common)),
        "kl_mean": float(np.mean(kl)),
        "kl_median": float(np.median(kl)),
        "kl_p90": float(np.quantile(kl, 0.90)),
    }
    os.makedirs(out_dir, exist_ok=True)
    tag = "_".join(matches)
    np.savez_compressed(
        os.path.join(out_dir, f"misspec_repeat_disagreement_{tag}.npz"),
        kl_score=kl, mu=mu, var=var,
        test_files=np.array(common), repeat_match_strings=np.array(matches),
    )
    with open(os.path.join(out_dir, f"misspec_repeat_disagreement_{tag}.json"), "w") as f:
        json.dump(_to_json_compatible(payload), f, indent=4)
    print(f"[misspec] {name}: repeat disagreement over {len(matches)} repeats, "
          f"kl_mean={payload['kl_mean']:.4f} kl_median={payload['kl_median']:.4f}", flush=True)
    return payload


def _summarise_variate_inputs(name: str, test_paths: Sequence[str], loader, n_raw_files: int = 32):
    """Data-level provenance check: is this variate on the same footing as the training set?

    Returns a stats dict: the raw mixed_bandpowers mean per band, per-file map shape statistics,
    per-file raw bandpower amplitudes, and the scaled input mean and std per key. A preprocessing
    mismatch shows up as a large offset after the training scalers.
    """
    stats: Dict = {"name": name}
    try:
        raws = []
        for p in list(test_paths)[:n_raw_files]:
            try:
                with h5py.File(p, "r") as f:
                    raws.append(np.asarray(f["cls_results"]["full"]["mixed_bandpowers"][()],
                                           dtype=np.float64))
            except (OSError, KeyError):
                continue
        if raws:
            arr = np.stack(raws)                       # [F, 21, nbands]
            per_band = np.nanmean(arr, axis=(0, 1))    # mean over files and spectra
            stats["raw_bandpowers_per_band"] = per_band.tolist()
            stats["raw_bandpowers_mean"] = float(np.nanmean(arr))
            stats["raw_bandpowers_n_files"] = len(raws)
            print(f"[misspec-inputs] {name}: RAW mixed_bandpowers over {len(raws)} files "
                  f"mean={np.nanmean(arr):.6e} per-band="
                  f"[{', '.join(f'{v:.4e}' for v in per_band)}]", flush=True)
    except Exception as e:  # diagnostics must never take the run down
        print(f"[misspec-inputs] {name}: raw bandpower summary failed: {e}", flush=True)

    # --- per-file map shape statistics ---
    # Peak fractions, skew and kurtosis use each mock's own std, so they are scale-free.
    try:
        per_file: Dict[str, list] = {}
        for batch in loader:
            data = batch[0] if isinstance(batch, (list, tuple)) else batch
            if not isinstance(data, dict):
                break
            for k, v in data.items():
                if v.dim() < 3:            # bandpowers etc. — amplitude handled above
                    continue
                x = v.float().flatten(1)   # [B, ...] -> per-sample vector
                mu = x.mean(dim=1)
                sd = x.std(dim=1)
                c = x - mu[:, None]
                sdc = sd.clamp_min(1e-12)[:, None]
                zc = c / sdc
                rows = per_file.setdefault(k, [])
                for j in range(x.shape[0]):
                    zj = zc[j]
                    rows.append({
                        "mean": float(mu[j]), "std": float(sd[j]),
                        "skew": float((zj ** 3).mean()), "kurtosis": float((zj ** 4).mean() - 3.0),
                        "zero_frac": float((x[j] == 0).float().mean()),
                        "peak_2sig": float((zj > 2).float().mean()),
                        "peak_3sig": float((zj > 3).float().mean()),
                        "peak_4sig": float((zj > 4).float().mean()),
                        "void_2sig": float((zj < -2).float().mean()),
                        "void_3sig": float((zj < -3).float().mean()),
                    })
        for k, rows in per_file.items():
            if not rows:
                continue
            agg = {f: float(np.mean([r[f] for r in rows])) for f in rows[0]}
            sem = {f: float(np.std([r[f] for r in rows]) / max(1, np.sqrt(len(rows))))
                   for f in rows[0]}
            stats.setdefault("per_file_map_stats", {})[k] = {
                "n_files": len(rows), "mean": agg, "sem": sem,
                "raw": {f: [r[f] for r in rows] for f in rows[0]},
            }
            print(f"[misspec-inputs] {name}: {k} SHAPE over {len(rows)} files — "
                  f"std={agg['std']:.4f}+-{sem['std']:.4f} skew={agg['skew']:+.4f} "
                  f"kurt={agg['kurtosis']:+.4f} zero_frac={agg['zero_frac']:.5f} "
                  f"peak2={agg['peak_2sig']:.5f} peak3={agg['peak_3sig']:.5f} "
                  f"peak4={agg['peak_4sig']:.6f} void2={agg['void_2sig']:.5f}", flush=True)
    except Exception as e:
        print(f"[misspec-inputs] {name}: per-file map stats failed: {e}", flush=True)

    # --- per-file raw bandpower amplitude ---
    try:
        amps = []
        for p in list(test_paths):
            try:
                with h5py.File(p, "r") as f:
                    amps.append(float(np.nanmean(
                        np.asarray(f["cls_results"]["full"]["mixed_bandpowers"][()],
                                   dtype=np.float64))))
            except (OSError, KeyError):
                amps.append(float("nan"))
        stats["per_file_bandpower_amp"] = amps
        good = np.asarray([a for a in amps if np.isfinite(a)])
        if good.size:
            print(f"[misspec-inputs] {name}: per-file bandpower amplitude over {good.size} files "
                  f"mean={good.mean():.6e} std={good.std():.6e}", flush=True)
    except Exception as e:
        print(f"[misspec-inputs] {name}: per-file bandpower amplitude failed: {e}", flush=True)

    try:
        acc: Dict[str, list] = {}
        for i, batch in enumerate(loader):
            data = batch[0] if isinstance(batch, (list, tuple)) else batch
            if not isinstance(data, dict):
                break
            for k, v in data.items():
                acc.setdefault(k, []).append(
                    (float(v.float().mean()), float(v.float().std()), v.numel())
                )
            if i >= 7:  # a handful of batches is plenty for a distribution check
                break
        for k, rows in acc.items():
            n = sum(r[2] for r in rows)
            mean = sum(r[0] * r[2] for r in rows) / n
            std = sum(r[1] * r[2] for r in rows) / n
            stats.setdefault("scaled", {})[k] = {"mean": mean, "std": std}
            print(f"[misspec-inputs] {name}: SCALED {k}: mean={mean:+.4f} std={std:.4f}",
                  flush=True)
    except Exception as e:
        print(f"[misspec-inputs] {name}: scaled input summary failed: {e}", flush=True)
    return stats



def _finish_no_truth_variate(name, cfg, variate, repeat_index, theta0s, samples, test_paths,
                             param_names, missing_params, exclude_params, n_bad_events,
                             num_samples, input_stats, meta, out_dir):
    """The observation branch of _eval_one_variate: with no truth vector there are no metrics.

    Writes the truth branch's JSON and npz files with metrics=None and has_truth=False, and
    returns the per-event moments for the cross-repeat KL. Prints counts only."""
    payload = {
        "variate": name,
        "experiment": cfg.experiment_name,
        "match_string": cfg.match_string,
        "repeat_index": int(repeat_index),
        "data_patterns": variate["patterns"],
        "n_test_files": len(test_paths),
        "n_test_cosmologies": meta["n_test_cosmologies"],
        "n_dropped_nonfinite": n_bad_events,
        "test_ids_from_fixed_lock": meta["test_ids_from_fixed_lock"],
        "missing_params": missing_params,
        "excluded_params": exclude_params,
        "dropped_from_calibration": list(param_names),
        "num_posterior_samples": int(num_samples),
        "input_stats": input_stats,
        "has_truth": False,
        "metrics": None,
        "metrics_skipped": "no finite truth vector (observation); nothing scored against theta0",
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"misspec_evaluation_results_{cfg.match_string}.json"), "w") as f:
        json.dump(_to_json_compatible(payload), f, indent=4)
    _save_posterior_samples(
        os.path.join(out_dir, f"misspec_posterior_samples_{cfg.match_string}.npz"),
        theta0s, samples, test_paths,
    )
    _save_posterior_moments(
        os.path.join(out_dir, f"misspec_posterior_moments_{cfg.match_string}.npz"),
        theta0s, samples, test_paths, param_names,
    )
    print(f"[misspec] {name}: no truth vector (observation): metrics skipped; "
          f"{len(test_paths)} event(s) sampled, samples/moments written to the blind dir.",
          flush=True)
    samp_np = samples.detach().cpu().numpy() if hasattr(samples, "detach") else np.asarray(samples)
    per_event = {
        "mu": samp_np.mean(axis=0).astype(np.float32),
        "var": samp_np.var(axis=0).astype(np.float32),
        "test_files": [os.path.basename(p) for p in test_paths],
    }
    return {
        "n_test_files": len(test_paths),
        "n_dropped_nonfinite": n_bad_events,
        "available_params": [],
        "cal_full": float("nan"),
        "cal_om_s8_w0": float("nan"),
        "has_truth": False,
        "out_dir": out_dir,
        "_per_event": per_event,
    }

def _eval_one_variate(
    variate: Dict,
    model,
    cfg,
    nested_keys: Dict,
    param_names: List[str],
    orig_key_scalers: Dict,
    orig_cosmo_scaler,
    prior_samples_scaled,
    *,
    num_samples: int,
    test_shape_noise_idx,
    out_dir: str,
    repeat_index: int = 0,
    test_id_pool=None,
    max_test_files: Optional[int] = None,
):
    name = variate["name"]
    exclude_params = list(variate.get("exclude_params", []))
    variate_nested_keys = nested_keys
    if variate.get("eb_variant") is not None:
        variate_nested_keys = build_nested_keys_from_quantities(
            list(cfg.dataset_quantities), variate["eb_variant"]
        )

    loader, meta = build_variate_test_loader(
        variate["patterns"],
        variate_nested_keys,
        param_names,
        orig_key_scalers,
        orig_cosmo_scaler,
        test_id_pool=test_id_pool,
        test_shape_noise_idx=test_shape_noise_idx,
        batch_size=min(64, int(getattr(cfg, "test_batch_size", None) or getattr(cfg, "batch_size", 64))),
        max_test_files=max_test_files,
        eb_noise_norm=getattr(cfg, "eb_noise_norm", None),
    )

    # Fail fast on a wrong eb-variant tag / absent params rather than skip-looping every file.
    missing_keys, present_cosmo, probe_path = _probe_variate_file(
        meta["test_paths"], variate_nested_keys, param_names
    )
    if missing_keys:
        with h5py.File(probe_path, "r") as f:
            available_groups = (sorted(f["pixelised_results"].keys())
                                if "pixelised_results" in f else [])
        raise RuntimeError(
            f"data keys missing from {probe_path}: {missing_keys}; "
            f"on-disk pixelised_results groups: {available_groups} — prebake or fix eb_variant."
        )
    missing_params = [p for p in param_names if p not in present_cosmo]
    print(f"[misspec] {name}: n_test={meta['n_test_files']} "
          f"({meta['n_test_cosmologies']} cosmologies, "
          f"fixed_lock={meta['test_ids_from_fixed_lock']}) "
          f"missing_params={missing_params} exclude_params={exclude_params}", flush=True)

    input_stats = _summarise_variate_inputs(name, meta["test_paths"], loader)

    # Swap the test set under the prebuilt model, and under each ensemble member.
    model.test_dataloader = loader
    for m in getattr(model, "members", []):
        m.test_dataloader = loader

    theta0s, samples = model.generate_samples(num_samples=num_samples)

    # Drop test points whose posterior samples came out non-finite (far-OOD conditioning can
    # degenerate the spline inverse), and report the count.
    test_paths = list(meta["test_paths"])
    event_ok = torch.isfinite(samples).all(dim=2).all(dim=0)  # samples [S, N, D] -> [N]
    n_bad_events = int((~event_ok).sum())
    if n_bad_events:
        print(f"[misspec] {name}: dropping {n_bad_events}/{len(event_ok)} test points with "
              "non-finite posterior samples (far-OOD sampling degeneracy).", flush=True)
        keep = event_ok.cpu().numpy().astype(bool)
        samples = samples[:, event_ok, :]
        theta0s = theta0s[event_ok, :]
        test_paths = [p for p, k in zip(test_paths, keep) if k]

    theta_np = theta0s.detach().cpu().numpy()
    finite = np.isfinite(theta_np).all(axis=0)
    available_idx = [i for i, p in enumerate(param_names)
                     if finite[i] and p not in exclude_params]
    dropped = [p for i, p in enumerate(param_names) if i not in available_idx]
    if not bool(finite.any()):
        return _finish_no_truth_variate(name, cfg, variate, repeat_index, theta0s, samples,
                                        test_paths, param_names, missing_params, exclude_params,
                                        n_bad_events, num_samples, input_stats, meta, out_dir)
    if not available_idx:
        raise RuntimeError("no finite/included cosmo params to calibrate on")

    metrics = _compute_misspec_metrics(
        theta0s, samples, param_names, available_idx, orig_cosmo_scaler, prior_samples_scaled,
    )
    # ΔMI needs log_prob at the true theta, undefined when any dim is NaN.
    if bool(finite.all()):
        metrics["test_log_prob"] = model.compute_avg_log_prob()
    else:
        metrics["test_log_prob"] = None

    payload = {
        "variate": name,
        "experiment": cfg.experiment_name,
        "match_string": cfg.match_string,
        "repeat_index": int(repeat_index),
        "data_patterns": variate["patterns"],
        "n_test_files": len(test_paths),
        "n_test_cosmologies": meta["n_test_cosmologies"],
        "n_dropped_nonfinite": n_bad_events,
        "test_ids_from_fixed_lock": meta["test_ids_from_fixed_lock"],
        "missing_params": missing_params,
        "excluded_params": exclude_params,
        "dropped_from_calibration": dropped,
        "num_posterior_samples": int(num_samples),
        "input_stats": input_stats,
        "metrics": metrics,
    }
    tarp_intervals = _pop_credible_intervals(payload["metrics"])

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"misspec_evaluation_results_{cfg.match_string}.json"), "w") as f:
        json.dump(_to_json_compatible(payload), f, indent=4)
    if tarp_intervals:
        # Plot inputs: x = credible_intervals (alpha edges), y = ecp_bootstrap mean/std.
        intervals_payload = {
            "variate": name,
            "match_string": cfg.match_string,
            "available_params": [param_names[i] for i in available_idx],
            **tarp_intervals,
        }
        with open(os.path.join(out_dir, f"misspec_tarp_credible_intervals_{cfg.match_string}.json"), "w") as f:
            json.dump(_to_json_compatible(intervals_payload), f, indent=4)
    _save_posterior_samples(
        os.path.join(out_dir, f"misspec_posterior_samples_{cfg.match_string}.npz"),
        theta0s, samples, test_paths,
    )
    _save_posterior_moments(
        os.path.join(out_dir, f"misspec_posterior_moments_{cfg.match_string}.npz"),
        theta0s, samples, test_paths, param_names,
    )

    cal_full = metrics["tarp"]["full"]["calibration_error"]
    subset_key = "sigma_8__omega_m__w0"
    cal_subset = metrics["tarp"]["subsets"].get(subset_key, {}).get("calibration_error", float("nan"))
    print(f"[misspec] {name}: DONE cal_full={cal_full:.4f} cal_om_s8_w0={cal_subset:.4f} "
          f"fom={metrics.get('fom')} dMI={metrics.get('test_log_prob')}", flush=True)
    # Per-event posterior moments (scaled space, over the sample axis) for the cross-repeat
    # disagreement statistic; keyed by test-file basename for alignment across repeats.
    samp_np = samples.detach().cpu().numpy() if hasattr(samples, "detach") else np.asarray(samples)
    per_event = {
        "mu": samp_np.mean(axis=0).astype(np.float32),
        "var": samp_np.var(axis=0).astype(np.float32),
        "test_files": [os.path.basename(p) for p in test_paths],
    }
    return {
        "n_test_files": len(test_paths),
        "n_dropped_nonfinite": n_bad_events,
        "available_params": [param_names[i] for i in available_idx],
        "cal_full": float(cal_full),
        "cal_om_s8_w0": float(cal_subset),
        "out_dir": out_dir,
        "_per_event": per_event,
    }
