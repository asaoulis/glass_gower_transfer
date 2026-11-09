"""Experiment rows for the beyond-2-pt (HOS, third-moment) hybrid.

The map branch of the VDQ hybrid is replaced by an MLP over third-moment features read from the
HOS sidecar stores. The frozen Stage-I band is kept. Rows cover the GLASS NPE, the Gower NPE ens9
finetune and the NLE chain, plus the train waves run by `eval.py --mode hos-train`.
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

# R2: distinct-bin triples i < j < l of each scale block (56 per scale). R1 order is scale 8 then 16.
_TRIPLES = [(i, j, l) for i in range(6) for j in range(i, 6) for l in range(j, 6)]
R2_FEATURE_INDEX = [s * 56 + k for s in (0, 1) for k, (i, j, l) in enumerate(_TRIPLES) if i < j < l]

_VECTOR_LOADER = {"preload_data": True, "num_workers": 0, "pin_memory": False, "persistent_workers": None,
                  "prefetch_factor": None}

HOS_MLP_DEFAULT = {"input_dim": 112, "hidden": 128, "depth": 2, "dropout": 0.1, "input_transform": None}


def hos_glass_row(hos_kwargs=None, data_patterns=_HOS_GLASS, band_ckpt=_BAND_CKPT, **top):
    """GLASS pretraining row: frozen Stage-I band (8-D) + HOS MLP (8-D), concat -> 8-D summary -> NPE flow."""
    band = _VDQ["kids_legacy_band_nla_m_vdq"]
    c = copy.deepcopy(band)
    c.update({
        "model_type": "kids_hybrid_bandpowers_hos",
        "dataset_quantities": ["mixed_bandpowers", "hos"],
        "data_patterns": data_patterns,
        "latent_dim": 16,
        "model_kwargs": {
            "bandpower_type": "mlp",
            "bandpower_latent_dim": 8,
            "bandpower_kwargs": copy.deepcopy(band["model_kwargs"]),
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
        "batch_size": 256,
        "lr": 3e-4,
        "optimizer_kwargs": {"weight_decay": 0.1, "betas": (0.9, 0.999)},
        "epochs": 120,
        "scheduler_type": "cyclic",
        "scheduler_kwargs": {"warmup": 2000, "min_factor": 0.1, "cyclic_period_steps": 6000},
        "use_KL_loss": False,
        "precision": "32",
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
    c.pop("pretrained_band_ckpt_path", None)
    c.pop("optimizer_kwargs", None)
    c["checkpoint_path"] = glass_ckpt
    # Stage B keeps the 2-pt pack keys: lr 1e-5, 10 epochs, all parameters trainable
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


# Keys the launcher looks up are literal strings (it greps for them), not f-strings.
kids_legacy_hos_experiments = {}
kids_legacy_hos_experiments["kids_legacy_hos_nla_m_vdq"] = _assert_final_summary_dim(
    hos_glass_row(), 8, "kids_legacy_hos_nla_m_vdq")
kids_legacy_hos_experiments["gower_npe_finetune_hos_nla_m_vdq_ens9"] = _assert_final_summary_dim(
    hos_gower_row(), 8, "gower_npe_finetune_hos_nla_m_vdq_ens9")
kids_legacy_hos_experiments["gower_npe_finetune_band_nla_m_vdq_ens9_side"] = band_side_row()
# R2 control: distinct-bin triples only
kids_legacy_hos_experiments["kids_legacy_hos_r2_nla_m_vdq"] = _assert_final_summary_dim(
    hos_glass_row(hos_kwargs={"feature_index": R2_FEATURE_INDEX}), 8, "kids_legacy_hos_r2_nla_m_vdq")
kids_legacy_hos_experiments["gower_npe_finetune_hos_r2_nla_m_vdq_ens9"] = _assert_final_summary_dim(
    hos_gower_row(_HOS_CKPT_R2, hos_kwargs={"feature_index": R2_FEATURE_INDEX}), 8,
    "gower_npe_finetune_hos_r2_nla_m_vdq_ens9")
kids_legacy_hos_experiments["kids_legacy_hos_nla_m_vdq_smoke"] = hos_glass_row(
    data_patterns="/data/alex/hos_models/fixture/sidecars/output_*.h5", band_ckpt=None, freeze_band=False,
    epochs=1, repeat_indices=[0])


# --- HOS NLE chain: the band k=8 procedure on the HOS sidecar stores ---
from config.kids_legacy_vdq import _band_nle_a_vdq, _band_nle_b_vdq  # noqa: E402

HOS_NLE_K = 8
HOS_NLE_REPEATS = (0, 1, 2, 3, 4)
_HOS_SOURCE = "kids_legacy_hos_nla_m_vdq"


def _hos_nle_common(c, data_patterns):
    c["data_patterns"] = data_patterns
    c["scaler_options"] = copy.deepcopy(kids_legacy_hos_experiments[_HOS_SOURCE]["scaler_options"])
    c["precision"] = "32"
    c.update(_VECTOR_LOADER)
    return c


def hos_nle_a_row(r):
    return _hos_nle_common(_band_nle_a_vdq(r), _HOS_GLASS)


def hos_nle_b_row(r):
    c = _hos_nle_common(_band_nle_b_vdq(r), _HOS_GOWER)
    c["pretrained_band_ckpt_path"] = f"{_MODELS}/checkpoints/glass_nle_pretrain_hos_nla_m_vdq_k{HOS_NLE_K}_r{r}/"
    # Eval MCMC pool of 63 workers keeps the 124-item test set within one wall-time
    c["sampling_num_jobs"] = 63
    return c


HOS_NLE_SOURCES = {}
kids_legacy_hos_experiments["glass_nle_pretrain_hos_nla_m_vdq_k8_r0"] = hos_nle_a_row(0)
kids_legacy_hos_experiments["glass_nle_pretrain_hos_nla_m_vdq_k8_r1"] = hos_nle_a_row(1)
kids_legacy_hos_experiments["glass_nle_pretrain_hos_nla_m_vdq_k8_r2"] = hos_nle_a_row(2)
kids_legacy_hos_experiments["glass_nle_pretrain_hos_nla_m_vdq_k8_r3"] = hos_nle_a_row(3)
kids_legacy_hos_experiments["glass_nle_pretrain_hos_nla_m_vdq_k8_r4"] = hos_nle_a_row(4)
kids_legacy_hos_experiments["gower_nle_finetune_hos_nla_m_vdq_k8_r0_ens9_e150"] = hos_nle_b_row(0)
kids_legacy_hos_experiments["gower_nle_finetune_hos_nla_m_vdq_k8_r1_ens9_e150"] = hos_nle_b_row(1)
kids_legacy_hos_experiments["gower_nle_finetune_hos_nla_m_vdq_k8_r2_ens9_e150"] = hos_nle_b_row(2)
kids_legacy_hos_experiments["gower_nle_finetune_hos_nla_m_vdq_k8_r3_ens9_e150"] = hos_nle_b_row(3)
kids_legacy_hos_experiments["gower_nle_finetune_hos_nla_m_vdq_k8_r4_ens9_e150"] = hos_nle_b_row(4)
for _r in HOS_NLE_REPEATS:
    HOS_NLE_SOURCES[f"glass_nle_pretrain_hos_nla_m_vdq_k{HOS_NLE_K}_r{_r}"] = _HOS_SOURCE
    HOS_NLE_SOURCES[f"gower_nle_finetune_hos_nla_m_vdq_k{HOS_NLE_K}_r{_r}_ens9_e150"] = _HOS_SOURCE
assert set(HOS_NLE_SOURCES) <= set(kids_legacy_hos_experiments)


# --- single-node CPU training waves (eval.py --mode hos-train) ---
def _ptask(kind, exp, repeat, group, after=(), env=None):
    t = {"kind": kind, "exp": exp, "repeat": int(repeat), "group": group, "after": list(after), "env": dict(env or {})}
    if kind == "embed":
        t["sources"] = [HOS_NLE_SOURCES[exp]]
    return t


def _stage_a(r):
    return f"glass_nle_pretrain_hos_nla_m_vdq_k{HOS_NLE_K}_r{r}"


def _stage_b(r):
    return f"gower_nle_finetune_hos_nla_m_vdq_k{HOS_NLE_K}_r{r}_ens9_e150"


HOS_TRAIN_WAVES = {
    "hos_train": (
        [_ptask("embed", _stage_a(r), r, "glass_hos") for r in HOS_NLE_REPEATS]
        + [_ptask("train", e, r, g) for e, g in (("gower_npe_finetune_hos_nla_m_vdq_ens9", "gower_hos"),
                                                  ("gower_npe_finetune_hos_r2_nla_m_vdq_ens9", "gower_hos"),
                                                  ("gower_npe_finetune_band_nla_m_vdq_ens9_side", "gower_band"))
           for r in HOS_NLE_REPEATS]
        + [_ptask("embed", _stage_b(r), r, "gower_hos", after=[_stage_a(r)], env={"RUN_EVALUATION": "0"})
           for r in HOS_NLE_REPEATS]),
    "hos_rs": (
        [dict(_ptask("eval", _stage_b(r), r, "gower_hos"), label=f"rs_r{r}_m{m}",
              args=["--mode", "recover-scalers", "--experiments", _stage_b(r), "--repeat-indices", str(r),
                    "--members", str(m)]) for r in HOS_NLE_REPEATS for m in range(9)]
        + [dict(_ptask("eval", _stage_b(r), r, "gower_hos", after=[f"rs_r{r}_m{m}" for m in range(9)]),
                label=f"verify_r{r}", args=["--mode", "recover-scalers", "--verify-frames", "--experiments", _stage_b(r),
                                            "--repeat-indices", str(r)]) for r in HOS_NLE_REPEATS]),
    "hos_npe_eval": [
        dict(_ptask("eval", e, r, g), label=f"npe_eval_{e}_r{r}",
             args=["--mode", "misspec", "--misspec-base", e, "--variates", "gower_vdq_hos", "--repeat-indices", str(r)])
        for e, g in (("gower_npe_finetune_hos_nla_m_vdq_ens9", "gower_hos"),
                     ("gower_npe_finetune_hos_r2_nla_m_vdq_ens9", "gower_hos"),
                     ("gower_npe_finetune_band_nla_m_vdq_ens9_side", "gower_band"))
        for r in HOS_NLE_REPEATS],
    "hos_nle_eval": [_ptask("embed", _stage_b(r), r, "gower_hos", env={"RUN_TRAINING": "0"}) for r in HOS_NLE_REPEATS],
}


# --- hos_x superset stores ---
from src.ml.data.hos_x import HOSX_BAKE_NAMES, group_index  # noqa: E402

_HOSX_GLASS = f"{_SIDE}/glass_vdq_nla_m_f16_sc8a1_{_EB}_hosx/output_*.h5"
_HOSX_GOWER = f"{_SIDE}/gower_vdq_nla_m_f16_sc8a1_{_EB}_hosx/output_*.h5"


def _hosx_top():
    return {"dataset_quantities": ["mixed_bandpowers", "hos_x"],
            "scaler_options": {"data": {"type": "standard", "keys": None, "per_dim_keys": ["hos_x"]},
                               "cosmo": {"type": "preset"}}}


def hosx_kwargs(groups, hos_kwargs=None):
    return {"input_key": "hos_x", "input_dim": len(HOSX_BAKE_NAMES), "feature_index": group_index(groups),
            **(hos_kwargs or {})}


def hosx_glass_row(groups, hos_kwargs=None, data_patterns=_HOSX_GLASS, **top):
    """hos_glass_row on the hos_x store with the feature set = union of the hos_x `groups`."""
    return hos_glass_row(hos_kwargs=hosx_kwargs(groups, hos_kwargs), data_patterns=data_patterns,
                         **{**_hosx_top(), **top})


def hosx_gower_row(glass_ckpt, groups, hos_kwargs=None, data_patterns=_HOSX_GOWER, **top):
    """hos_gower_row (Gower NPE ens9 pack keys) on the hos_x store, warm-started from the GLASS hosx row."""
    return hos_gower_row(glass_ckpt=glass_ckpt, hos_kwargs=hosx_kwargs(groups, hos_kwargs), data_patterns=data_patterns,
                         **{**_hosx_top(), **top})


HOSX_FINALISTS = {"r1": ["R1"], "cmrgno4": ["R1", "Cmrg/no4"], "cmrg": ["R1", "Cmrg"], "gmrg": ["R1", "Gmrg"]}
for _tag, _groups in HOSX_FINALISTS.items():
    _g = f"kids_legacy_hosx_{_tag}_nla_m_vdq"
    kids_legacy_hos_experiments[_g] = _assert_final_summary_dim(hosx_glass_row(_groups), 8, _g)
    _w = f"gower_npe_finetune_hosx_{_tag}_nla_m_vdq_ens9"
    kids_legacy_hos_experiments[_w] = _assert_final_summary_dim(
        hosx_gower_row(f"{_MODELS}/checkpoints/{_g}/", _groups), 8, _w)


# Gower mocks with R1 more than 50 robust sigma out are excluded (config.exclude_files).
GOWER_HOS_OUTLIERS = "config/hos_assets/gower_hos_outliers.json"


def _xo(c):
    c = copy.deepcopy(c)
    c["exclude_files"] = GOWER_HOS_OUTLIERS
    return c


_XO = {e: f"{e}_xo" for e in ("gower_npe_finetune_hos_nla_m_vdq_ens9", "gower_npe_finetune_hos_r2_nla_m_vdq_ens9",
                              "gower_npe_finetune_band_nla_m_vdq_ens9_side")}
_XO.update({_stage_b(r): f"{_stage_b(r)}_xo" for r in HOS_NLE_REPEATS})
for _old, _new in _XO.items():
    kids_legacy_hos_experiments[_new] = _xo(kids_legacy_hos_experiments[_old])
    if _old in HOS_NLE_SOURCES:
        HOS_NLE_SOURCES[_new] = HOS_NLE_SOURCES[_old]


def _xo_task(t, relabelled=()):
    t = copy.deepcopy(t)
    t["exp"] = _XO.get(t["exp"], t["exp"])
    t["after"] = [f"{a}_xo" if a in relabelled else _XO.get(a, a) for a in t.get("after", [])]
    t["args"] = [_XO.get(a, a) for a in t.get("args", [])] or t.pop("args", None) or []
    if not t["args"]:
        t.pop("args")
    if t.get("label"):
        t["label"] += "_xo"
    if t["kind"] == "embed":
        t["sources"] = [HOS_NLE_SOURCES[t["exp"]]]
    return t


HOS_TRAIN_WAVES["hos_train_xo"] = [_xo_task(t) for t in HOS_TRAIN_WAVES["hos_train"] if t["exp"] in _XO]
for _w in ("hos_npe_eval", "hos_rs", "hos_nle_eval"):
    _rel = {t["label"] for t in HOS_TRAIN_WAVES[_w] if t.get("label")}
    HOS_TRAIN_WAVES[f"{_w}_xo"] = [_xo_task(t, _rel) for t in HOS_TRAIN_WAVES[_w]]


# --- R1 + Gmrg production arm ---
_HOSX_GMRG_SOURCE = "kids_legacy_hosx_gmrg_nla_m_vdq"
GOWER_HOSX_GMRG_OUTLIERS = "config/hos_assets/gower_hosx_gmrg_outliers.json"


def _hosx_gmrg_nle_common(c, data_patterns):
    c["data_patterns"] = data_patterns
    c["scaler_options"] = copy.deepcopy(kids_legacy_hos_experiments[_HOSX_GMRG_SOURCE]["scaler_options"])
    c["precision"] = "32"
    c.update(_VECTOR_LOADER)
    return c


def hosx_gmrg_nle_a_row(r):
    return _hosx_gmrg_nle_common(_band_nle_a_vdq(r), _HOSX_GLASS)


def hosx_gmrg_nle_b_row(r):
    c = _hosx_gmrg_nle_common(_band_nle_b_vdq(r), _HOSX_GOWER)
    c["pretrained_band_ckpt_path"] = f"{_MODELS}/checkpoints/glass_nle_pretrain_hosx_gmrg_nla_m_vdq_k{HOS_NLE_K}_r{r}/"
    c["sampling_num_jobs"] = 63
    c["exclude_files"] = GOWER_HOSX_GMRG_OUTLIERS
    return c


def _gmrg_a(r):
    return f"glass_nle_pretrain_hosx_gmrg_nla_m_vdq_k{HOS_NLE_K}_r{r}"


def _gmrg_b(r):
    return f"gower_nle_finetune_hosx_gmrg_nla_m_vdq_k{HOS_NLE_K}_r{r}_ens9_e150_xo"


kids_legacy_hos_experiments["glass_nle_pretrain_hosx_gmrg_nla_m_vdq_k8_r0"] = hosx_gmrg_nle_a_row(0)
kids_legacy_hos_experiments["glass_nle_pretrain_hosx_gmrg_nla_m_vdq_k8_r1"] = hosx_gmrg_nle_a_row(1)
kids_legacy_hos_experiments["glass_nle_pretrain_hosx_gmrg_nla_m_vdq_k8_r2"] = hosx_gmrg_nle_a_row(2)
kids_legacy_hos_experiments["glass_nle_pretrain_hosx_gmrg_nla_m_vdq_k8_r3"] = hosx_gmrg_nle_a_row(3)
kids_legacy_hos_experiments["glass_nle_pretrain_hosx_gmrg_nla_m_vdq_k8_r4"] = hosx_gmrg_nle_a_row(4)
kids_legacy_hos_experiments["gower_nle_finetune_hosx_gmrg_nla_m_vdq_k8_r0_ens9_e150_xo"] = hosx_gmrg_nle_b_row(0)
kids_legacy_hos_experiments["gower_nle_finetune_hosx_gmrg_nla_m_vdq_k8_r1_ens9_e150_xo"] = hosx_gmrg_nle_b_row(1)
kids_legacy_hos_experiments["gower_nle_finetune_hosx_gmrg_nla_m_vdq_k8_r2_ens9_e150_xo"] = hosx_gmrg_nle_b_row(2)
kids_legacy_hos_experiments["gower_nle_finetune_hosx_gmrg_nla_m_vdq_k8_r3_ens9_e150_xo"] = hosx_gmrg_nle_b_row(3)
kids_legacy_hos_experiments["gower_nle_finetune_hosx_gmrg_nla_m_vdq_k8_r4_ens9_e150_xo"] = hosx_gmrg_nle_b_row(4)
kids_legacy_hos_experiments["gower_npe_finetune_hosx_gmrg_nla_m_vdq_ens9_xo"] = dict(
    copy.deepcopy(kids_legacy_hos_experiments["gower_npe_finetune_hosx_gmrg_nla_m_vdq_ens9"]),
    exclude_files=GOWER_HOSX_GMRG_OUTLIERS)
for _r in HOS_NLE_REPEATS:
    HOS_NLE_SOURCES[_gmrg_a(_r)] = _HOSX_GMRG_SOURCE
    HOS_NLE_SOURCES[_gmrg_b(_r)] = _HOSX_GMRG_SOURCE
assert set(HOS_NLE_SOURCES) <= set(kids_legacy_hos_experiments)

_GMRG_NPE = "gower_npe_finetune_hosx_gmrg_nla_m_vdq_ens9_xo"
HOS_TRAIN_WAVES["hosx_gmrg_train"] = (
    [dict(_ptask("train", _HOSX_GMRG_SOURCE, r, "glass_hosx"), label=f"glass_gmrg_r{r}") for r in HOS_NLE_REPEATS]
    + [dict(_ptask("embed", _gmrg_a(r), r, "glass_hosx", after=[f"glass_gmrg_r{r}"]), label=f"nle_a_gmrg_r{r}")
       for r in HOS_NLE_REPEATS]
    + [dict(_ptask("embed", _gmrg_b(r), r, "gower_hosx", after=[f"nle_a_gmrg_r{r}"], env={"RUN_EVALUATION": "0"}),
            label=f"nle_b_gmrg_r{r}") for r in HOS_NLE_REPEATS]
    + [dict(_ptask("train", _GMRG_NPE, r, "gower_hosx", after=[f"glass_gmrg_r{r}"]), label=f"npe_gmrg_r{r}")
       for r in HOS_NLE_REPEATS])
HOS_TRAIN_WAVES["hosx_gmrg_rs"] = (
    [dict(_ptask("eval", _gmrg_b(r), r, "gower_hosx"), label=f"rs_gmrg_r{r}_m{m}",
          args=["--mode", "recover-scalers", "--experiments", _gmrg_b(r), "--repeat-indices", str(r), "--members", str(m)])
     for r in HOS_NLE_REPEATS for m in range(9)]
    + [dict(_ptask("eval", _gmrg_b(r), r, "gower_hosx", after=[f"rs_gmrg_r{r}_m{m}" for m in range(9)]),
            label=f"verify_gmrg_r{r}", args=["--mode", "recover-scalers", "--verify-frames", "--experiments", _gmrg_b(r),
                                             "--repeat-indices", str(r)]) for r in HOS_NLE_REPEATS])
HOS_TRAIN_WAVES["hosx_gmrg_npe_eval"] = [
    dict(_ptask("eval", _GMRG_NPE, r, "gower_hosx"), label=f"npe_eval_gmrg_r{r}",
         args=["--mode", "misspec", "--misspec-base", _GMRG_NPE, "--variates", "gower_vdq_hosx", "--repeat-indices", str(r)])
    for r in HOS_NLE_REPEATS]
HOS_TRAIN_WAVES["hosx_gmrg_nle_eval"] = [
    dict(_ptask("embed", _gmrg_b(r), r, "gower_hosx", env={"RUN_TRAINING": "0"}), label=f"nle_eval_gmrg_r{r}")
    for r in HOS_NLE_REPEATS]
