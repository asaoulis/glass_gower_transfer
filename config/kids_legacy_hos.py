"""Beyond-2-pt HOS hybrid (task eval-and-viz/hos-hybrid-training) — model configs.

The map branch of the VDQ field-level hybrid is replaced by a feed-forward MLP over the noise-texture-robust
higher-order statistics of the same baked maps: R1 = the 56 tomographic third moments at 8' and 16' (112 numbers;
src/ml/data/hos.py; selection in .claude/runs/eval-and-viz/robust-hos-selection/artifacts/RECOMMENDATIONS.md).

Data: per-mock HOS SIDECARS of the VDQ sc8a1 bakes (`eval.py --mode hos-bake`), written under
{MODELS_ROOT}/checkpoints/hos_sidecars/<store>/<basename> (FLAT: the glob order, hence the split, equals the bake's;
per-shard transfer archives live in <store>_tar/). They carry `hos/vector`, `hos_ext/moments`,
`mixed_bandpowers` and `cosmo_dict` verbatim, with the SAME basenames as the bake, so the frozen Stage-I band's split,
its scaler subsample and the Gower fixed-test lock are identical. Paths resolve from $GLASS_MODELS_ROOT, so the same
rows run locally (GLASS_MODELS_ROOT=/data/alex/hos_models) and on the cluster (default).

Each row starts from the production row it must match and overrides only the HOS-specific keys:
  kids_legacy_hos_nla_m_vdq                  GLASS pretraining, frozen per-repeat `kids_legacy_band_nla_m_vdq`
                                             (= `kids_legacy_band_nla_m_vdq` split/cosmo/scaler keys)
  gower_npe_finetune_hos_nla_m_vdq_ens9      Gower NPE ens9 fine-tune (= the band pack's split/eval/schedule keys)
  gower_npe_finetune_band_nla_m_vdq_ens9_side  the band-only baseline on the sidecar store (dict-diff vs the cluster
                                             pack = data_patterns, checkpoint_path, preload/worker keys only)
  *_r2                                       R2 control: only the 20 distinct-bin triples per scale (feature_index)
Loader notes: `dataset_quantities` lists mixed_bandpowers BEFORE hos (the scaler fit draws one subsample
permutation per key in order, so the band's scaler subsample is unchanged); `per_dim_keys=['hos']` gives the HOS
vector a per-dimension standard scaler; `preload_data` caches the ~32 kB sidecars in RAM (num_workers=0).
Hyperparameters marked SWEEP are set from artifacts/SWEEP_SPEC.md results.
"""
import copy
import os

from config.kids_legacy_bgp import _assert_final_summary_dim
from config.kids_legacy_vdq import _FINETUNE_VAL_CHECK_INTERVAL, kids_legacy_vdq_experiments as _VDQ

_MODELS = os.environ.get("GLASS_MODELS_ROOT", "/share/gpu5/asaoulis/transfer_models")
_SIDE = f"{_MODELS}/checkpoints/hos_sidecars"
_EB = "fwhm4_lmin56_lcut1400"
_HOS_GLASS = f"{_SIDE}/glass_vdq_nla_m_f16_sc8a1_{_EB}/output_*.h5"
_HOS_GOWER = f"{_SIDE}/gower_vdq_nla_m_f16_sc8a1_{_EB}/output_*.h5"
_BAND_CKPT = f"{_MODELS}/checkpoints/kids_legacy_band_nla_m_vdq/"
_HOS_CKPT = f"{_MODELS}/checkpoints/kids_legacy_hos_nla_m_vdq/"
_HOS_CKPT_R2 = f"{_MODELS}/checkpoints/kids_legacy_hos_r2_nla_m_vdq/"

# R2 = distinct-bin triples i < j < l of each scale block (56 per scale; R1 order = scale 8 then 16)
_TRIPLES = [(i, j, l) for i in range(6) for j in range(i, 6) for l in range(j, 6)]
R2_FEATURE_INDEX = [s * 56 + k for s in (0, 1) for k, (i, j, l) in enumerate(_TRIPLES) if i < j < l]

# Vector-store loader keys: everything in RAM, no worker processes (prefetch/persistent keys are meaningless).
_VECTOR_LOADER = {"preload_data": True, "num_workers": 0, "pin_memory": False, "persistent_workers": None,
                  "prefetch_factor": None}

HOS_MLP_DEFAULT = {"input_dim": 112, "hidden": 128, "depth": 2, "dropout": 0.1, "input_transform": None}  # sweep A1-A4


def hos_glass_row(hos_kwargs=None, data_patterns=_HOS_GLASS, band_ckpt=_BAND_CKPT, **top):
    """GLASS pretraining row: frozen Stage-I band (8-D) + HOS MLP (8-D), concat -> 8-D summary -> NPE flow."""
    band = _VDQ["kids_legacy_band_nla_m_vdq"]
    c = copy.deepcopy(band)                       # cosmo params, split + scaler keys == the band's
    c.update({
        "model_type": "kids_hybrid_bandpowers_hos",
        "dataset_quantities": ["mixed_bandpowers", "hos"],
        "data_patterns": data_patterns,
        "latent_dim": 16,                         # concat width: 8 band + 8 HOS
        "model_kwargs": {
            "bandpower_type": "mlp",
            "bandpower_latent_dim": 8,
            "bandpower_kwargs": copy.deepcopy(band["model_kwargs"]),   # == the checkpoint's architecture
            "map_kwargs": {**HOS_MLP_DEFAULT, **(hos_kwargs or {})},
            "hybrid_output_dim": 8,
            "fusion_type": "concat",
        },
        "pretrained_band_ckpt_path": band_ckpt,
        "freeze_band": True,
        "flow_kwargs": {"hidden_features": 32},
        "scaler_options": {"data": {"type": "standard", "keys": None, "per_dim_keys": ["hos"]},
                           "cosmo": {"type": "preset"}},
        "eb_map_variant": None,
        "augment_eb_patches": False,
        "batch_size": 256,                        # SWEEP
        "lr": 3e-4,                               # sweep A2/A4
        "optimizer_kwargs": {"weight_decay": 0.1, "betas": (0.9, 0.999)},   # sweep A2/A4 (best epochs 63-104)
        "epochs": 120,                            # sweep A4: ceil(1.25 x median best epoch 102), capped at 120
        "scheduler_type": "cyclic",
        "scheduler_kwargs": {"warmup": 2000, "min_factor": 0.1, "cyclic_period_steps": 6000},
        "use_KL_loss": False,
        "precision": "32",                        # tiny model: fp32 everywhere (CPU sweeps: bf16 autocast ~2x slower)
        "project": "hos-hybrid",
        "repeat_indices": [0, 1, 2, 3, 4],
        **_VECTOR_LOADER,
    })
    c.update(top)
    return c


def hos_gower_row(glass_ckpt=_HOS_CKPT, hos_kwargs=None, data_patterns=_HOS_GOWER, **top):
    """Gower NPE ens9 fine-tune of the GLASS HOS hybrid (whole-model warm start, per repeat)."""
    pack = _VDQ["gower_npe_finetune_band_nla_m_vdq_ens9"]
    c = hos_glass_row(hos_kwargs=hos_kwargs, data_patterns=data_patterns)
    for k in ("epochs", "lr", "batch_size", "scheduler_type", "scheduler_kwargs", "ensemble_repeats",
              "max_trainval_cosmos", "train_frac", "val_frac", "test_frac", "fixed_test_sim_ids",
              "match_num_cosmo", "repeat_indices", "val_check_interval"):
        c[k] = copy.deepcopy(pack[k])
    c.pop("pretrained_band_ckpt_path", None)      # the whole model comes from the GLASS checkpoint
    c.pop("optimizer_kwargs", None)               # Stage B keeps the default weight decay (as swept in B1)
    c["checkpoint_path"] = glass_ckpt
    # Stage B keeps the 2-pt pack keys (lr 1e-5, 10 epochs, all trainable): the B1 grid (lr x epochs x freeze, r0+r1)
    # is flat within 0.011 nats of its top cell vs ~0.03 seed noise; freezing band (-0.03) or band+HOS (-0.12) hurts.
    c["freeze_band"] = False
    c["project"] = "gower-finetuning-hos"
    c.update(top)
    return c


def band_side_row():
    """The cluster band-only Gower pack, re-pointed at the sidecar store (local apples-to-apples baseline)."""
    c = copy.deepcopy(_VDQ["gower_npe_finetune_band_nla_m_vdq_ens9"])
    c["data_patterns"] = _HOS_GOWER
    c["precision"] = "32"
    c["checkpoint_path"] = _BAND_CKPT
    c.update(_VECTOR_LOADER)
    return c


kids_legacy_hos_experiments = {}
kids_legacy_hos_experiments["kids_legacy_hos_nla_m_vdq"] = _assert_final_summary_dim(
    hos_glass_row(), 8, "kids_legacy_hos_nla_m_vdq")
kids_legacy_hos_experiments["gower_npe_finetune_hos_nla_m_vdq_ens9"] = _assert_final_summary_dim(
    hos_gower_row(), 8, "gower_npe_finetune_hos_nla_m_vdq_ens9")
kids_legacy_hos_experiments["gower_npe_finetune_band_nla_m_vdq_ens9_side"] = band_side_row()
# R2 control (user D6): distinct-bin triples only.
kids_legacy_hos_experiments["kids_legacy_hos_r2_nla_m_vdq"] = _assert_final_summary_dim(
    hos_glass_row(hos_kwargs={"feature_index": R2_FEATURE_INDEX}), 8, "kids_legacy_hos_r2_nla_m_vdq")
kids_legacy_hos_experiments["gower_npe_finetune_hos_r2_nla_m_vdq_ens9"] = _assert_final_summary_dim(
    hos_gower_row(_HOS_CKPT_R2, hos_kwargs={"feature_index": R2_FEATURE_INDEX}), 8,
    "gower_npe_finetune_hos_r2_nla_m_vdq_ens9")
# The local smoke fixture (665 Gower raw mocks -> bake -> sidecars) for the smoke gate.
kids_legacy_hos_experiments["kids_legacy_hos_nla_m_vdq_smoke"] = hos_glass_row(
    data_patterns="/data/alex/hos_models/fixture/sidecars/output_*.h5", band_ckpt=None, freeze_band=False,
    epochs=1, repeat_indices=[0])
