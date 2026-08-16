"""Model-misspecification evaluation: ONE trained NPE ensemble evaluated on the TEST split of
MANY Gower variate datasets, reusing the ORIGINAL training scalers.

Motivation: the production NPE (`gower_npe_finetune_nla_m_z8`, repeat 0) was trained on the
nla_m Gower suite. Each physics variate (nla, nla_z, galaxy-bias, ...) is a controlled
misspecification of that forward model; running the SAME model on each variate's held-out test
cosmologies and measuring the TARP miscalibration quantifies how sensitive the inference is to
that misspecification.

Key differences from the standard `evaluate_best_checkpoint` path:
- Data scalers (bandpowers LogNormal + map Standard) are fit ONCE on the original nla_m
  train+val split via `prepare_data_parameters` and INJECTED into every variate test loader —
  never refit on a variate (a refit would absorb part of the covariate shift being measured).
- Variate test sets are built directly from the shared fixed-test lock file (lock ∩ on-disk),
  NOT via `split_by_cosmology`: variates were largely simulated ON the 200 held-out test ids,
  so forcing them all into test would trigger split_by_cosmology's no-train/val fallback and
  silently produce a train-heavy split.
- Cosmo params absent from a variate (e.g. b_ia in nla/nla_z) are NaN-filled by the loader
  (`allow_missing_cosmo_params`), keeping theta 9-dim to match the flow; calibration is then
  computed only over the finite dims. Per-variate `exclude_params` additionally drops params
  whose MEANING differs between suites (a_ia under NLA vs NLA-M parametrisations).
- FoM is still computed over all 9 sampled dims against the base Gower prior.

Invoked from eval.py (RUN_MISSPEC flag) so the cluster's `eval-submit` job needs no new verb.
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

# a_ia is EXCLUDED from calibration for variates whose IA parametrisation differs from the
# nla_m training suite (its meaning changes between suites — user directive 2026-07-08).
# nla_m / gb* share the nla_m IA model, so a_ia keeps its meaning there.
DEFAULT_VARIATES: List[Dict] = [
    {"name": "nla_m", "patterns": f"{_GPU5}/gower_mocks_nla_m_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5",
     "exclude_params": [], "in_distribution": True},
    {"name": "nla", "patterns": f"{_GPU4}/gower_mocks_nla_counts/output_*.h5", "exclude_params": ["a_ia"]},
    {"name": "nla_z", "patterns": f"{_GPU4}/gower_mocks_nla_z_counts/output_*.h5", "exclude_params": ["a_ia"]},
    {"name": "gb0p5", "patterns": f"{_GPU4}/gower_mocks_gb0p5_counts/output_*.h5", "exclude_params": []},
    {"name": "gb1p5", "patterns": f"{_GPU4}/gower_mocks_gb1p5_counts/output_*.h5", "exclude_params": []},
]

# GLASS pre-train variate suites (gpu5 f16 fwhm4_lmin56_lcut1400 prebakes, matching the GLASS
# z8 foundation's maps). In-dist = the foundation's own lmin50 training store; its test ids
# are derived from the training split at runtime (no fixed lock for GLASS).
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

# --- NO-VD suite (2026-07-29 variate switch; the MAIN analysis variate) -------------------------
# Added ALONGSIDE the VD-on sets above rather than repointing them: the counts-era misspec bases
# (e.g. gower_npe_finetune_nla_m_counts_z8) still resolve, and repointing in place would have made
# any such re-run silently evaluate a VD-on model against VD-off data — different forward physics,
# no error raised. Select these with `--variates gower_novd` / `glass_pretrain_novd`.
#
# Store provenance (datasets_checklist.md): S1 nla_m + G1 glass nla_m are consumed from the gpu5 f16
# fwhm4 PREBAKES; the misspec test sets (S2/S3 nla,nla_z and S4/S5 + G2/G3 gb0p5,gb1p5) are consumed
# RAW off gpu4 — bake optional, same as the VD-on sets do.
NOVD_GOWER_VARIATES: List[Dict] = [
    {"name": "nla_m", "patterns": f"{_GPU5}/gower_mocks_nla_m_novd_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5",
     "exclude_params": [], "in_distribution": True},
    {"name": "nla", "patterns": f"{_GPU4}/gower_mocks_nla_novd_counts/output_*.h5", "exclude_params": ["a_ia"]},
    {"name": "nla_z", "patterns": f"{_GPU4}/gower_mocks_nla_z_novd_counts/output_*.h5", "exclude_params": ["a_ia"]},
    {"name": "gb0p5", "patterns": f"{_GPU4}/gower_mocks_gb0p5_novd_counts/output_*.h5", "exclude_params": []},
    {"name": "gb1p5", "patterns": f"{_GPU4}/gower_mocks_gb1p5_novd_counts/output_*.h5", "exclude_params": []},
]

# Foundation-level (pre-Gower-finetune) misspec check — user 2026-07-29: run the gb0p5/gb1p5 GLASS
# sets against the 5 pre-trained foundations, i.e. BEFORE any Gower finetune. This is why G2/G3
# exist and launched early. NB the VD-on GLASS set above carries no gb variates; this one does.
NOVD_GLASS_PRETRAIN_VARIATES: List[Dict] = [
    {"name": "glass_nla_m",
     "patterns": f"{_GPU5}/glass_mocks_nla_m_novd_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5",
     "exclude_params": [], "in_distribution": True},
    # Same in-distribution physics, read from the UNBAKED gpu4 source instead of the gpu5 f16
    # prebake. A control: the gb variates are read raw off gpu4, so if raw-vs-prebake ever
    # mattered (f16 downcast, wrong extracted variant) this entry would diverge from
    # 'glass_nla_m' and the OOD comparison would be confounded. Also the correct reference for
    # bandpower-only models, which train off the raw store.
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

# --- DUAL-NORMALISATION arm comparison (task training-runs/improved-shear-tests) ----------------
# One variate set per shear-processing ARM. The arm is baked into the STORE, so the b_g test sets
# must be read from the SAME arm's bake as the model trained on — evaluating an A1 model on A0
# maps would measure the estimator mismatch, not the b_g robustness. Naming follows the bake DAG:
#   glass_dn_{nla_m|gb0p7|gb1p0|gb1p5}_f16_{a0|a1|sc8|sc8a1}_fwhm4_lmin56_lcut1400
# B1_selfstd reads the A0 store and applies its normalisation in the LOADER (eb_noise_norm='self'),
# so it shares a0's variate set; the transform is chained in build_variate_test_loader.
# All four b_g sets are nla_m (a_ia + b_ia present) => no exclude_params, all 9 params available.
# gb0p7/gb1p0/gb1p5 share --rng-seed 4242, so the headline statistic is the PAIRED per-event
# Δz = z(b_g) − z(b_g=1.0), matched on (sim_id, aug_id) offline from the sample npz.
_DN_EB = "fwhm4_lmin56_lcut1400"
_DN_EB_SC8 = "sc8_fwhm4_lmin56_lcut1400"


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
    """BGP campaign (`b_g` MARGINALISED at generation) — the same b_g ladder as `_dn_variate_set`.

    Identical machinery and identical probe values (0.7 / 1.0 / 1.5, all at `--rng-seed 4242`, so
    still bit-paired); the ONE thing that differs is the foundation the model was trained on: here
    `b_g` was drawn per (sim, outer, rot) from the Flamingo O3-diag prior instead of pinned at 1.
    That makes this set the direct test of the campaign's premise — does marginalising `b_g` shrink
    the paired Δz that the `_dn` arms showed at +3.9σ…+4.5σ?

    The gb1p0 store is NOT optional: the headline statistic is the PAIRED Δz = z(b_g) − z(b_g=1.0),
    so without the 1.0 reference only absolute z is available.
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


def _inject_variate_set(arm_tag: str) -> List[Dict]:
    """Synthetic noise-variance injection on the b_g=1.0 store — see src/ml/eval/inject.py.

    All arms share ONE file list (the gb1p0 store), so `paired_dz.py` pairs them on
    (sim_id, aug_id) against `glass_gb1p0` with zero cosmic variance. `glass_gb1p5` is carried
    as the real-effect reference the injections are measured against.
    """
    def store(which):
        return f"{_GPU5}/glass_dn_{which}_f16_{arm_tag}_{_DN_EB}/output_*.h5"
    base = store("gb1p0")
    return [
        {"name": "glass_gb1p0", "patterns": base, "exclude_params": [], "in_distribution": True},
        {"name": "glass_gb1p0_inj_null", "patterns": base, "exclude_params": [],
         "inject": {"source": "null", "target_b": 1.5}},
        {"name": "glass_gb1p0_inj_grf", "patterns": base, "exclude_params": [],
         "inject": {"source": "grf", "target_b": 1.5, "slope": 1.0}},
        {"name": "glass_gb1p0_inj_grfwhite", "patterns": base, "exclude_params": [],
         "inject": {"source": "grf", "target_b": 1.5, "slope": 0.0}},
        {"name": "glass_gb1p0_inj_kappa", "patterns": base, "exclude_params": [],
         "inject": {"source": "kappa", "target_b": 1.5}},
        {"name": "glass_gb1p5", "patterns": store("gb1p5"), "exclude_params": []},
    ]


VARIATE_SETS: Dict[str, List[Dict]] = {
    "gower": DEFAULT_VARIATES,
    "glass_pretrain": GLASS_PRETRAIN_VARIATES,
    "gower_novd": NOVD_GOWER_VARIATES,
    "glass_pretrain_novd": NOVD_GLASS_PRETRAIN_VARIATES,
    "glass_dn_a0": _dn_variate_set("a0"),
    "glass_dn_a1": _dn_variate_set("a1"),
    "glass_dn_b1": _dn_variate_set("a0"),      # A0 store + the loader knob
    "glass_dn_sc8": _dn_variate_set("sc8"),
    "glass_dn_sc8a1": _dn_variate_set("sc8a1"),
    # BGP campaign (b_g marginalised at generation) — the payoff test for this campaign.
    "glass_bgp_sc8a1": _bgp_variate_set("sc8a1"),
    # Co-primary unwhitened arm. Same ladder, same probe values, same rng-seed pairing — the only
    # difference is which shear product the maps were baked from, so sc8a1-vs-sc8 here isolates the
    # normalisation's effect on b_g robustness. Probe bakes: jobs 1344427/28/29 (2026-08-16).
    "glass_bgp_sc8": _bgp_variate_set("sc8"),
    # R1024 — the conservative-scale-cut arm (8' beam, hard ell <= 1024). Same `a0` arm tag (plain
    # counts, no noise-norm) but the fwhm8_lmin56_lcut1024 bake; those stores carry BARE `E` groups
    # (baked without --keep-variant-tag), which matches the config's eb_map_variant=None, so no
    # per-variate `eb_variant` override is needed here.
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
    "glass_inject_a0": _inject_variate_set("a0"),
    "glass_inject_sc8": _inject_variate_set("sc8"),
    # The "could a DES-sized channel pass at <0.3 sigma?" test. Same machinery, amplitudes from
    # Gatti 2024's 5%->1% per-bin bracket instead of our ebdiff-measured ones (src/ml/eval/inject.py).
    # Our network's reliance on the channel is frozen at TRAINING amplitude, so this UPPER-BOUNDS
    # what a DES-amplitude channel could do through a compressor that has fully learned it.
    # Spectral sweep: WHERE in scale is the network sensitive to a noise-variance modulation?
    # Every arm carries the SAME total modulation variance (the transform renormalises `g` to unit
    # variance), so the only thing varying is which scales that variance sits at. Patch pixels are
    # 6.87' at NSIDE 512, so k in cycles/pixel maps to 6.87/k arcmin.
    "glass_injectk_a0": [
        {"name": "glass_gb1p0", "patterns": f"{_GPU5}/glass_dn_gb1p0_f16_a0_{_DN_EB}/output_*.h5",
         "exclude_params": [], "in_distribution": True},
    ] + [
        {"name": f"glass_gb1p0_inj_{tag}",
         "patterns": f"{_GPU5}/glass_dn_gb1p0_f16_a0_{_DN_EB}/output_*.h5", "exclude_params": [],
         "inject": dict({"source": "grf", "target_b": 1.5}, **kw)}
        for tag, kw in [
            # band-limited: all the modulation variance in one octave
            ("k14_27",   {"kband": (0.25, 0.5)}),        # 14-27 arcmin
            ("k27_55",   {"kband": (0.125, 0.25)}),      # 27-55 arcmin
            ("k55_110",  {"kband": (0.0625, 0.125)}),    # 55-110 arcmin (~1-2 deg)
            ("k110_220", {"kband": (0.03125, 0.0625)}),  # 1.8-3.7 deg
            ("k220_550", {"kband": (0.0125, 0.03125)}),  # 3.7-9.2 deg
            # power laws. In 2-D the variance per log-k goes as k^2 P(k), so slope<2 is still
            # small-scale dominated and only slope>2 concentrates variance at large scales.
            ("slope2",   {"slope": 2.0}),
            ("slope3",   {"slope": 3.0}),
        ]
    ],
    "glass_injectdes_a0": [
        {"name": "glass_gb1p0", "patterns": f"{_GPU5}/glass_dn_gb1p0_f16_a0_{_DN_EB}/output_*.h5",
         "exclude_params": [], "in_distribution": True},
        {"name": "glass_gb1p0_inj_grfdes",
         "patterns": f"{_GPU5}/glass_dn_gb1p0_f16_a0_{_DN_EB}/output_*.h5", "exclude_params": [],
         "inject": {"source": "grf", "target_b": 1.5, "slope": 1.0, "profile": "des"}},
    ],
}


def _load_experiment_config(experiment_name: str):
    """Rebuild the experiment config exactly like eval.py's load_config + list-branch handling."""
    from config.default import get_default_config
    from config.experiments import experiments as base_experiments
    from config.ablations import ablation_experiments
    from config.kids_legacy import kids_legacy_experiments
    from config.kids_legacy_counts import kids_legacy_counts_experiments
    from config.kids_legacy_novd import kids_legacy_novd_experiments
    from config.kids_legacy_dn import kids_legacy_dn_experiments
    from config.kids_legacy_bgp import kids_legacy_bgp_experiments

    exps = dict(base_experiments)
    exps.update(ablation_experiments)
    exps.update(kids_legacy_experiments)
    exps.update(kids_legacy_counts_experiments)
    exps.update(kids_legacy_novd_experiments)
    exps.update(kids_legacy_dn_experiments)  # dual-normalisation arm-comparison suite
    exps.update(kids_legacy_bgp_experiments)  # BGP campaign (galaxy-bias prior marginalised)
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

    Fails fast with a useful message (e.g. wrong eb variant tag) instead of letting the
    loader skip every file one by one.
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
    inject: Optional[Dict] = None,
):
    """Variate TEST loader with the ORIGINAL scalers injected (never refit).

    Test cosmologies = (``test_id_pool`` ∩ on-disk ids); falls back to ALL on-disk
    cosmologies when there is no overlap (e.g. a small gb subset outside the pool).
    ``max_test_files`` caps the test set by accumulating whole cosmologies (sorted by
    sim_id) until the file budget is reached.

    ``eb_noise_norm`` mirrors the TRAIN-time loader (``src/ml/utils.py``): when the experiment
    sets it, the per-sample E/B normalisation must be chained BEFORE the key scalers here too.
    Such configs deliberately exclude the map keys from ``scaler_options['data']['keys']``
    (the transform standardises them), so omitting it would feed the model raw unstandardised
    maps and make the variate metrics meaningless.
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

    ds = H5CosmoDataset(
        filtered,
        nested_keys,
        list(cosmo_param_names),
        transform=None,
        allow_missing_cosmo_params=True,
    )
    data_transform = DataDictScalerTransform(key_scalers)
    inject_tf = None
    if eb_noise_norm or inject:
        from ..data.data_augmentations import EBNoiseNormTransform, ChainedDataTransform
        chain = []
        if inject:
            # FIRST in the chain: the injection synthesises a b_g shift on the RAW maps, so
            # everything downstream (eb_noise_norm, scalers) sees it exactly as it would see a
            # real variate read. See src/ml/eval/inject.py.
            from .inject import build_inject_transform
            inject_tf = build_inject_transform(inject)
            chain.append(inject_tf)
        if eb_noise_norm:
            chain.append(EBNoiseNormTransform(eb_noise_norm))
        chain.append(data_transform)
        data_transform = ChainedDataTransform(chain) if len(chain) > 1 else chain[0]
    wrapped = TransformingDataset(
        ds,
        data_transform=data_transform,
        cosmo_scaler=cosmo_scaler,
    )
    loader = torch.utils.data.DataLoader(
        wrapped, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    meta = {
        "n_test_files": len(filtered),
        "n_test_cosmologies": len(test_ids),
        "test_ids_from_fixed_lock": not used_fallback,
        "test_paths": filtered,
        "inject_transform": inject_tf,
    }
    return loader, meta


def _compute_misspec_metrics(
    theta0s: torch.Tensor,
    samples: torch.Tensor,
    param_names: Sequence[str],
    available_idx: Sequence[int],
    param_scaler,
    prior_samples_scaled: Optional[torch.Tensor],
):
    """Calibration on the available dims + FoM vs the base prior on all sampled dims.

    theta0s ~ [N, D] and samples ~ [S, N, D], both in SCALED space (loader/flow space) —
    mirrors run_evaluation_on_samples, which runs TARP in scaled space and FoM in both.
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

    # --- calibration (TARP) on the available dims only ------------------------------------
    tarp = TARPDiagnostics(available_names, bootstrap=True, num_bootstrap=25, seed=None)
    metrics.update(tarp.compute_all(
        samples_sims_first[:, :, available_idx].contiguous(),
        theta0s[:, available_idx].contiguous(),
    ))

    # --- FoM vs the base Gower prior over ALL sampled dims --------------------------------
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

    # --- per-available-param point stats in physical units --------------------------------
    sample_means = scaled_samples_first.mean(axis=0)
    bias = sample_means - scaled_theta0s
    std_devs = scaled_samples_first.std(axis=0)
    # torch.quantile allocates a large sort buffer over the (n_samples, n_sims, d) tensor and OOMs
    # a 16 GiB v100 here, which is why the misspec eval had to be pinned to a100/l40s. Same fix as
    # evaluate_models.py:330-336 — do the quantile on CPU. Identical numbers, and it is a sort over
    # an already-materialised tensor, so the transfer is the only cost.
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

    Works for eval-time ensembles (ensemble_repeats>1: the repeat's N members are loaded as
    one EnsembleNDELightningModule) AND single-model experiments (the repeat's best checkpoint).

    ``repeat_indices``: one full pass (scalers + model + all variates) per repeat. With >1
    repeat, per-event CROSS-REPEAT posterior disagreement (mean pairwise symmetric
    diag-Gaussian KL, as in ensemble_uncertainty.py) is computed per variate and saved next to
    the calibration results — the OOD statistic to correlate with miscalibration.
    ``repeat_index`` (int) is a legacy alias for a single-repeat run.

    Variate test cosmologies come from ``config.fixed_test_sim_ids`` when the experiment uses a
    lock file, else from the experiment's own held-out test split (derived at runtime), so no
    variate is ever evaluated on cosmologies the model trained on. ``max_test_files`` caps each
    variate's test set (whole cosmologies, sorted by sim_id).
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

    # Flushed step markers: this setup block runs before any per-repeat print, so without them a
    # hard crash here (a SIGILL/OOM leaves no traceback and loses block-buffered stdout) is
    # unlocalisable in the SLURM log.
    # A shared-cosmology run must never overwrite the strict held-out run's results.
    if test_id_source == "shared" and out_subdir == "misspec":
        out_subdir = "misspec_shared"

    print(f"[misspec] setup 1/4: loading config for '{base_experiment}'", flush=True)
    cfg0 = _load_experiment_config(base_experiment)
    param_names = list(cfg0.cosmo_param_names)
    eb_variant = getattr(cfg0, "eb_map_variant", None)
    nested_keys = build_nested_keys_from_quantities(list(cfg0.dataset_quantities), eb_variant)
    out_root = os.path.join(cfg0.base_path, "checkpoints", cfg0.experiment_name, out_subdir)
    print(f"[misspec] setup 2/4: config OK — params={param_names} eb_variant={eb_variant} "
          f"out_root={out_root}", flush=True)

    # Base Gower prior + one shared set of prior samples (FoM shrinkage reference) — the
    # prior is repeat-independent.
    prior = build_gower_prior(param_names, preset_overrides=_config_preset_overrides(cfg0))
    print("[misspec] setup 3/4: Gower prior built", flush=True)
    prior_samples_scaled = _sample_from_prior(prior, prior_num_samples, target_dim=len(param_names))
    print(f"[misspec] setup 4/4: drew {prior_num_samples} prior samples "
          f"{tuple(prior_samples_scaled.shape)}", flush=True)

    # ``test_id_source='shared'``: evaluate EVERY variate — the in-distribution reference
    # included — on the cosmologies the OOD variates actually have on disk.
    #
    # Why this mode exists: a variate suite is usually simulated over a small sim_id range
    # (e.g. 0..199), while the in-distribution experiment's held-out test split is a random
    # ~10% of thousands of cosmologies, so the strict intersection can collapse to a handful
    # (14 cosmologies / 28 events for the GLASS gb sets). 'shared' trades the held-out
    # guarantee for ~14x the statistics, and because the SAME cosmologies are used for the
    # reference curve, "the model trained on this cosmology (at the in-distribution physics)"
    # applies equally to every variate — so a difference between them is still attributable to
    # the physics shift, not to memorisation. Report it as a matched-cosmology companion to the
    # strict held-out run, never as a replacement.
    shared_pool = None
    if test_id_source == "shared":
        ood = [v for v in variates if not v.get("in_distribution")] or variates
        per_variate_ids = []
        for v in ood:
            ids = {extract_cosmo_index(p) for p in collect_paths(v["patterns"])}
            per_variate_ids.append(ids)
            print(f"[misspec] shared-pool: {v['name']} has {len(ids)} cosmologies on disk",
                  flush=True)
        shared_pool = set.intersection(*per_variate_ids) if per_variate_ids else set()
        if not shared_pool:
            raise RuntimeError(
                "test_id_source='shared' but the out-of-distribution variates share no sim_ids: "
                f"{[len(s) for s in per_variate_ids]}"
            )
        print(f"[misspec] shared-pool: {len(shared_pool)} cosmologies common to "
              f"{[v['name'] for v in ood]}", flush=True)
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
        cfg.test_shape_noise_idx = list(test_shape_noise_idx)
        repeat_match, _ = apply_repeat_config(cfg, r)
        cfg.match_string = repeat_match
        match_strings.append(repeat_match)
        print(f"[misspec] base experiment '{base_experiment}' repeat={r} "
              f"match_string={cfg.match_string} params={param_names} eb_variant={eb_variant}",
              flush=True)

        # ORIGINAL scalers for THIS repeat: fit on its nla_m train+val split (ensemble path).
        scalers, _, _, in_dist_test_loader = prepare_data_parameters(cfg)
        orig_key_scalers = scalers["data"]
        orig_cosmo_scaler = scalers["cosmo"]
        print(f"[misspec] repeat {r}: original scalers rebuilt from {cfg.data_patterns} "
              f"(data keys: {sorted(orig_key_scalers)})", flush=True)

        # Test-id pool for the variates: the lock file when the experiment pins one, else the
        # experiment's own held-out test cosmologies (identical across repeats — the test slice
        # comes from the fixed rng(42) shuffle before the per-repeat trainval reshuffle).
        lock_spec = getattr(cfg, "fixed_test_sim_ids", None)
        if lock_spec:
            test_id_pool = set(resolve_fixed_test_ids(lock_spec) or [])
        else:
            held_out = _resolve_test_paths(in_dist_test_loader) or []
            test_id_pool = {extract_cosmo_index(p) for p in held_out}
            print(f"[misspec] repeat {r}: derived test-id pool from the training split "
                  f"({len(test_id_pool)} held-out cosmologies).", flush=True)

        if test_id_source == "shared":
            test_id_pool = shared_pool
            print(f"[misspec] repeat {r}: test-id-source=shared -> every variate (including the "
                  f"in-distribution reference) uses the SAME {len(test_id_pool)} cosmologies.",
                  flush=True)

        # The repeat's model, built ONCE (loaders are swapped per variate): the N-member
        # eval-time ensemble when configured, else the repeat's single best checkpoint.
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
            # e.g. a repeat whose training jobs haven't finished — skip it, keep the others.
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

    # Cross-repeat posterior disagreement per variate (the ensemble-agreement OOD statistic).
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


def _save_posterior_moments(out_path, theta0s, samples, test_paths, param_names):
    """Compact per-event companion to the full posterior-sample dump (KB, not hundreds of MB).

    A z-score / bias analysis needs only the first two posterior moments per event, but the raw
    dump is [n_samples, N, D] — ~280 MB per variate at the default 10 000 samples, which makes a
    multi-arm comparison expensive to pull off the cluster. This writes the same information the
    analysis actually consumes:

      mean, std, theta0, z  [N, D] float32   with  z = (theta0 - mean) / std
      sim_ids, aug_ids, test_files [N]        (the pairing keys; cf. _save_posterior_samples)
      params [D]

    ``z`` follows the sign convention of the existing misspec z-score tooling
    (``first-npe-misspecification/artifacts/misspec_zscores.py``): positive z = the truth sits
    ABOVE the posterior mean. Moments are in SCALED space, but z is affine-invariant, so it is
    identical in physical space. Purely additive — the full sample dump is still written, and a
    failure here never breaks the eval.
    """
    try:
        samp = samples.detach().cpu().numpy() if hasattr(samples, "detach") else np.asarray(samples)
        theta = theta0s.detach().cpu().numpy() if hasattr(theta0s, "detach") else np.asarray(theta0s)
        mean = samp.mean(axis=0)
        std = samp.std(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (theta - mean) / std
        payload = {
            "mean": mean.astype(np.float32), "std": std.astype(np.float32),
            "theta0": theta.astype(np.float32), "z": z.astype(np.float32),
            "params": np.array(list(param_names)),
        }
        if test_paths is not None and len(test_paths) == theta.shape[0]:
            files = [os.path.basename(p) for p in test_paths]
            payload["test_files"] = np.array(files)
            payload["sim_ids"] = np.array([extract_cosmo_index(p) for p in test_paths], dtype=np.int64)
            payload["aug_ids"] = np.array([_parse_aug_id(f) for f in files], dtype=np.int64)
        np.savez_compressed(out_path, **payload)
        print(f"[save-moments] wrote {out_path}  (N={theta.shape[0]}, keys={sorted(payload)})",
              flush=True)
    except Exception as e:  # never let a diagnostic dump break the eval
        print(f"[save-moments] WARNING: failed to save posterior moments to {out_path}: {e}",
              flush=True)


def _compute_repeat_disagreement(name: str, per_repeat: Dict[str, Dict], out_dir: str):
    """Per-event posterior disagreement ACROSS training repeats for one variate.

    Aligns events by test-file basename (robust to per-repeat non-finite drops), stacks the
    per-repeat posterior moments to [K, N, D], and scores each event with the mean pairwise
    symmetric diag-Gaussian KL (ensemble_uncertainty.py formulation). Saved next to the
    per-repeat calibration JSONs so miscalibration and repeat-spread can be correlated
    directly. Moments are in SCALED space (same space as the TARP calibration)."""
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

    Answers two questions that a calibration number alone cannot separate — "did the physics
    shift?" vs "is this store preprocessed differently (e.g. mean- instead of counts-normalised
    shear, wrong smoothing variant)?":

    - RAW ``mixed_bandpowers`` mean per band, straight off disk, no scaler. A different shear
      normalisation moves these by a large factor; a physics variate at the same normalisation
      moves them by a few percent.
    - The SAME tensors after the ORIGINAL training scalers. In-distribution inputs land at
      mean~0 / std~1 (maps) and O(1) (log-scaled bandpowers); a normalisation mismatch shows up
      as a large offset, i.e. inputs the encoder has never seen.
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

    # --- PER-FILE map shape statistics ------------------------------------------------------
    # The point of these: mean/std alone cannot distinguish "the lensing amplitude moved" from
    # "the pixel occupancy / noise structure moved". Peak fractions and skew/kurtosis are
    # computed against EACH MOCK'S OWN std, so they are scale-free — cosmic variance in the
    # overall amplitude divides out and what is left is the SHAPE of the one-point distribution.
    # zero_frac tracks empty (unobserved) pixels, the channel a counts normalisation cannot
    # cancel: pixels with no galaxies are left at zero and their pattern depends on b_g.
    #
    # Paired with per-file raw bandpower amplitude (b_g-invariant to 0.3%, measured), the ratio
    # map_std / sqrt(bandpower) is a per-mock cosmic-variance-free amplitude probe — which is
    # what lets this run substitute for a fixed-seed paired simulation (the simulator's per-block
    # rng is UNSEEDED, master_kids_legacy_simulator.py:521, so realisations cannot be paired).
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

    # --- per-file RAW bandpower amplitude (the cosmic-variance reference) ---------------------
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
        # Cap at 64: eval-mode (no_grad) encodes fit fine on a v100 at 64-128; the OOMs seen on
        # jobs 1316362/1316364 were a ZOMBIE PROCESS squatting on that GPU (~15GB), not batch
        # size — mitigate by resubmitting / splitting repeats across GPUs, not by shrinking.
        batch_size=min(64, int(getattr(cfg, "test_batch_size", None) or getattr(cfg, "batch_size", 64))),
        max_test_files=max_test_files,
        # Mirror the TRAIN-time loader: a config with eb_noise_norm set (B1_selfstd) scales only
        # the bandpowers, so the map standardisation MUST come from this transform.
        eb_noise_norm=getattr(cfg, "eb_noise_norm", None),
        inject=variate.get("inject"),
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

    # Swap the test set under the prebuilt model. Ensembles: the ensemble-level loader feeds
    # theta0s and compute_avg_log_prob; each member's loader feeds its own encode+sample pass.
    # Single models have no .members — the one loader drives everything.
    model.test_dataloader = loader
    for m in getattr(model, "members", []):
        m.test_dataloader = loader

    theta0s, samples = model.generate_samples(num_samples=num_samples)

    # Injection calibration self-check: the achieved per-bin map power ratio must reproduce the
    # ebdiff-measured b_g=1.5 response, else the synthesised channel is the wrong size.
    inject_tf = meta.get("inject_transform")
    if inject_tf is not None:
        inject_report = inject_tf.summary()
        print(f"[misspec] {name}: injection {inject_tf!r}", flush=True)
        for side, rep in inject_report.items():
            print(f"[misspec]   {side}: achieved {rep['achieved_power_ratio']} "
                  f"vs target {rep['target_power_ratio']} (n={rep['n_events']})", flush=True)
        input_stats = dict(input_stats or {})
        input_stats["injection"] = {"spec": repr(inject_tf), "calibration": inject_report}

    # Drop test points whose posterior samples came out non-finite (far-OOD conditioning can
    # degenerate the spline inverse even with the clamped discriminant). Keep the analysis on
    # the finite events and report the count — silent NaNs would poison TARP/FoM wholesale.
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
    if not available_idx:
        raise RuntimeError("no finite/included cosmo params to calibrate on")

    metrics = _compute_misspec_metrics(
        theta0s, samples, param_names, available_idx, orig_cosmo_scaler, prior_samples_scaled,
    )
    # ΔMI needs log_prob at the TRUE theta — undefined when any dim is NaN.
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
