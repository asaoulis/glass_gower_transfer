"""Experiment rows for the galaxy-bias-marginalised (BGP) GLASS and Gower suite.

Stage I trains the bandpower MLP. Stage II freezes it inside the PreActResNet map hybrid.
"""
from config.kids_legacy import (
    _band_lmin50, _hybrid_lmin50_z8, _hybrid_lmin50_z8_smoke,
    _COSMO_8_NLA, _COSMO_9_NLAZ, _A_IA_NLA_BOX,
    _nle_pretrain, _nle_finetune, _nle_bake_repeat,
    _npe_finetune_z8,
)
from config.kids_legacy_novd import _RESNET_MAPKW

# --- stores ------------------------------------------------------------------------------------
_GPU5 = "/share/gpu5/asaoulis/transfer_datasets"
_CKPT = "/share/gpu5/asaoulis/transfer_models/checkpoints"

_EB = "fwhm4_lmin56_lcut1400"
_EB_SC8 = "sc8_fwhm4_lmin56_lcut1400"

# Prebaked f16 stores on gpu5. One frozen band serves every arm (both bakes glob the same store).
_BGP_A1 = f"{_GPU5}/glass_bgp_nla_m_f16_a1_{_EB}/output_*.h5"
_BGP_SC8A1 = f"{_GPU5}/glass_bgp_nla_m_f16_sc8a1_{_EB}/output_*.h5"
_BGP_SC8 = f"{_GPU5}/glass_bgp_nla_m_f16_sc8_{_EB}/output_*.h5"

_BAND_CKPT_BGP = f"{_CKPT}/kids_legacy_band_nla_m_bgp/"

_BAND_REPEATS = [0, 1, 2, 3, 4]
_HYBRID_REPEATS = (0, 1)

kids_legacy_bgp_experiments = {}


# --- Stage I: bandpower MLP, 5 repeats ---
def _band_bgp(data_patterns):
    """Stage-I bandpower MLP config reading mixed_bandpowers from the prebaked store `data_patterns`."""
    c = _band_lmin50()
    c["data_patterns"] = data_patterns
    c.pop("repeats", None)
    c["repeat_indices"] = list(_BAND_REPEATS)
    return c


kids_legacy_bgp_experiments["kids_legacy_band_nla_m_bgp"] = _band_bgp(_BGP_A1)


# --- Stage II: hybrid arms ---
def _hybrid_bgp(data_patterns, eb_variant, band_ckpt, top_extra=None,
                repeat_indices=_HYBRID_REPEATS):
    """PreActResNet z8 hybrid with the frozen per-repeat Stage-I band, on one arm's store."""
    c = _hybrid_lmin50_z8()
    c["data_patterns"] = data_patterns
    c["eb_map_variant"] = eb_variant
    c["pretrained_band_ckpt_path"] = band_ckpt
    c["model_kwargs"] = {**c["model_kwargs"], "map_kwargs": _RESNET_MAPKW}
    c.pop("repeats", None)
    c["repeat_indices"] = list(repeat_indices)
    for k, v in (top_extra or {}).items():
        c[k] = v
    return c


def _hybrid_bgp_smoke(top_extra=None):
    """Local smoke clone of _hybrid_bgp: same model kwargs, from-scratch band, fwhm8 fixture store."""
    c = _hybrid_lmin50_z8_smoke()
    c["model_kwargs"] = {**c["model_kwargs"], "map_kwargs": _RESNET_MAPKW}
    for k, v in (top_extra or {}).items():
        c[k] = v
    return c


# --- arms: sc8a1 (primary), a1, sc8, plus smoke clones ---
kids_legacy_bgp_experiments["kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1"] = \
    _hybrid_bgp(_BGP_SC8A1, None, _BAND_CKPT_BGP)

kids_legacy_bgp_experiments["kids_legacy_hybrid_nla_m_bgp_z8_resnet_a1"] = \
    _hybrid_bgp(_BGP_A1, None, _BAND_CKPT_BGP)

kids_legacy_bgp_experiments["kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8"] = \
    _hybrid_bgp(_BGP_SC8, None, _BAND_CKPT_BGP)

for _name in ("a1", "sc8", "sc8a1"):
    kids_legacy_bgp_experiments[f"kids_legacy_hybrid_nla_m_bgp_z8_resnet_{_name}_smoke"] = \
        _hybrid_bgp_smoke()


# --- b_g-sensitivity probe: 8-parameter finetune of each arm's foundation ---
_COSMO_8_BG = ["omega_m", "sigma_8"] + [f"b_g_bin{_i}" for _i in range(1, 7)]

# b_g scaler boxes: mean +- 3 sigma of the per-bin bias calibration.
_BG_PRIOR_MEANS = [1.0181, 1.0698, 1.1302, 1.2427, 1.3739, 1.4805]
_BG_PRIOR_SIGMAS = [0.1801, 0.1491, 0.1252, 0.0951, 0.0960, 0.0985]
_BG_BOXES = {
    f"b_g_bin{_i + 1}": (_m - 3.0 * _s, _m + 3.0 * _s)
    for _i, (_m, _s) in enumerate(zip(_BG_PRIOR_MEANS, _BG_PRIOR_SIGMAS))
}

_HYB_CKPT = {
    "sc8a1": f"{_CKPT}/kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1/",
    "sc8": f"{_CKPT}/kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8/",
    "a1": f"{_CKPT}/kids_legacy_hybrid_nla_m_bgp_z8_resnet_a1/",
}

# Repeat 3: the only a1 repeat whose training reached the final loss.
_BG8_REPEAT = 3


def _hybrid_bgp_bg8(data_patterns, arm_ckpt, repeat_index=_BG8_REPEAT):
    """One arm's foundation encoder, finetuned with a fresh flow head over omega_m, sigma_8, b_g_bin1..6."""
    c = _hybrid_bgp(data_patterns, None, _BAND_CKPT_BGP, repeat_indices=(repeat_index,))
    c.pop("pretrained_band_ckpt_path", None)
    c.pop("freeze_band", None)
    c["pretrained_embedding_ckpt_path"] = arm_ckpt
    c["freeze_embedding_net"] = False
    c["match_num_cosmo"] = False
    c["cosmo_param_names"] = list(_COSMO_8_BG)
    # The preset scaler raises on a parameter without a box, so the b_g boxes are listed here.
    c["scaler_options"] = {
        "data": {"type": "standard", "keys": None},
        "cosmo": {"type": "preset", "preset_overrides": dict(_BG_BOXES)},
    }
    c["epochs"] = 25
    c["scheduler_type"] = "exp"
    c["scheduler_kwargs"] = {"gamma": 0.938, "warmup_steps": 0}
    c["project"] = "glass-pretraining"
    return c


for _arm, _store in (("sc8a1", _BGP_SC8A1), ("sc8", _BGP_SC8), ("a1", _BGP_A1)):
    kids_legacy_bgp_experiments[f"kids_legacy_hybrid_nla_m_bgp_z8_resnet_{_arm}_bg8"] = \
        _hybrid_bgp_bg8(_store, _HYB_CKPT[_arm])


# --- 15-parameter foundation: sc8a1, 9 cosmology/IA + b_g_bin1..6 ---
_BG_PARAMS = [f"b_g_bin{_i}" for _i in range(1, 7)]

_P15_EPOCHS = 125
_P15_REPEATS = (0, 1, 2, 3, 4)


def _hybrid_bgp_p15(data_patterns, band_ckpt, repeat_indices=_P15_REPEATS):
    """The 9-parameter sc8a1 foundation recipe widened to the 15-parameter inference vector.

    The vector is the 9 cosmology/IA parameters plus b_g_bin1..6."""
    c = _hybrid_bgp(data_patterns, None, band_ckpt, repeat_indices=repeat_indices)
    base9 = list(c["cosmo_param_names"])
    assert len(base9) == 9, f"expected the 9-param foundation vector, got {base9}"
    c["cosmo_param_names"] = base9 + list(_BG_PARAMS)
    # The 9 boxes come from COSMO_PARAM_PRESET_MINMAX, only the 6 b_g boxes are overrides.
    c["scaler_options"] = {
        "data": {"type": "standard", "keys": None},
        "cosmo": {"type": "preset", "preset_overrides": dict(_BG_BOXES)},
    }
    c["epochs"] = _P15_EPOCHS
    return c


kids_legacy_bgp_experiments["kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1_p15"] = \
    _hybrid_bgp_p15(_BGP_SC8A1, _BAND_CKPT_BGP)


# --- p15 with a larger NDE head ---
_P15_BIGFLOW_REPEAT = (0,)


def _hybrid_bgp_p15_bigflow(data_patterns, band_ckpt):
    c = _hybrid_bgp_p15(data_patterns, band_ckpt, repeat_indices=_P15_BIGFLOW_REPEAT)
    base = dict(c.get("flow_kwargs") or {})
    c["flow_kwargs"] = {**base, "hidden_features": 2 * base.get("hidden_features", 32),
                        "num_transforms": 7}
    return c


kids_legacy_bgp_experiments["kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1_p15_bigflow"] = \
    _hybrid_bgp_p15_bigflow(_BGP_SC8A1, _BAND_CKPT_BGP)


# --- Cache-only embeddings of the p15 encoder, for a whitening-dimension sweep ---
_P15_EXP = "kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1_p15"


def _p15_embcache(repeat_index=0):
    """Cache-only embeddings run over the GLASS sc8a1 store on the frozen p15 encoder at one repeat."""
    src = kids_legacy_bgp_experiments[_P15_EXP]
    return {
        "data_patterns": src["data_patterns"],
        "eb_map_variant": src.get("eb_map_variant"),
        "dataset_quantities": [],
        "latent_dim": 8,
        "epochs": 1,
        "batch_size": 128,
        "project": "bgp-nle",
        "cosmo_param_names": list(src["cosmo_param_names"]),
        "scaler_options": {k: dict(v) for k, v in src["scaler_options"].items()},
        "inference_mode": "nle",
        "repeat_indices": [int(repeat_index)],
        "match_num_cosmo": False,
        "scale_embeddings": False,
        "whiten_embeddings": None,
        "run_training": False,
        "run_evaluation": False,
    }


kids_legacy_bgp_experiments["bgp_p15_embcache_r0"] = _p15_embcache(0)


# Raise if the final summary width differs from `expected`.
def _assert_final_summary_dim(c, expected, label):
    mk = c.get("model_kwargs") or {}
    got = mk.get("hybrid_output_dim")
    source = "hybrid_output_dim"
    if got is None:
        got, source = c.get("latent_dim"), "latent_dim (hybrid_output_dim unset)"
    assert got == expected, (
        f"{label}: FINAL SUMMARY WIDTH IS {got} (from {source}), expected {expected}. "
        f"The flow conditions on hybrid_output_dim when set, NOT on latent_dim "
        f"(latent_dim here = {c.get('latent_dim')}, the CONCAT width)."
    )
    return c


# --- p15 with a 16-D summary ---
_P15_Z16_REPEATS = (0, 1, 3)


def _hybrid_bgp_p15_z16(data_patterns, band_ckpt, repeat_indices=_P15_Z16_REPEATS):
    c = _hybrid_bgp_p15(data_patterns, band_ckpt, repeat_indices=repeat_indices)
    c["latent_dim"] = 24
    c["model_kwargs"] = {**c["model_kwargs"], "hybrid_output_dim": 16}
    return c


kids_legacy_bgp_experiments["kids_legacy_hybrid_nla_m_bgp_z16_resnet_sc8a1_p15"] = \
    _assert_final_summary_dim(_hybrid_bgp_p15_z16(_BGP_SC8A1, _BAND_CKPT_BGP), 16,
                              "kids_legacy_hybrid_nla_m_bgp_z16_resnet_sc8a1_p15")


# --- 9-parameter 16-D control, warm-started from the foundation ---
_P15_9P_Z16_REPEATS = (0, 3)
_SC8A1_9P_CKPT = f"{_CKPT}/kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1/"


def _hybrid_bgp_9p_z16(data_patterns, repeat_indices=_P15_9P_Z16_REPEATS):
    c = _hybrid_bgp(data_patterns, None, _BAND_CKPT_BGP, repeat_indices=repeat_indices)
    c.pop("pretrained_band_ckpt_path", None)
    c.pop("freeze_band", None)
    c["pretrained_embedding_ckpt_path"] = _SC8A1_9P_CKPT
    c["freeze_embedding_net"] = False
    c["match_num_cosmo"] = False
    c["latent_dim"] = 24
    c["model_kwargs"] = {**c["model_kwargs"], "hybrid_output_dim": 16}
    return c


kids_legacy_bgp_experiments["kids_legacy_hybrid_nla_m_bgp_z16_resnet_sc8a1_9p_warm"] = \
    _assert_final_summary_dim(_hybrid_bgp_9p_z16(_BGP_SC8A1), 16,
                              "kids_legacy_hybrid_nla_m_bgp_z16_resnet_sc8a1_9p_warm")


_P15_WARM_REPEATS = (0, 1, 2, 3, 4)


def _hybrid_bgp_p15_z16_warm(data_patterns, repeat_indices=_P15_WARM_REPEATS):
    """p15 16-D recipe inheriting the trained p9 foundation encoder."""
    c = _hybrid_bgp_p15_z16(data_patterns, _BAND_CKPT_BGP, repeat_indices=repeat_indices)
    c.pop("pretrained_band_ckpt_path", None)
    c.pop("freeze_band", None)
    c["pretrained_embedding_ckpt_path"] = _SC8A1_9P_CKPT
    c["freeze_embedding_net"] = False
    c["match_num_cosmo"] = False
    c["epochs"] = 50
    c["lr"] = 0.0002
    c["scheduler_type"] = "exp"
    c["scheduler_kwargs"] = {"gamma": 0.96832, "warmup_steps": 0}
    return c


kids_legacy_bgp_experiments["kids_legacy_hybrid_nla_m_bgp_z16_resnet_sc8a1_p15_warm"] = \
    _assert_final_summary_dim(_hybrid_bgp_p15_z16_warm(_BGP_SC8A1), 16,
                              "kids_legacy_hybrid_nla_m_bgp_z16_resnet_sc8a1_p15_warm")


# --- p15 warm-started from the p9 foundation on its cyclic LR, 75 epochs ---
def _hybrid_bgp_p15_z16_warm_cyc(data_patterns, repeat_indices=_P15_WARM_REPEATS):
    """p15 warm start on the p9 foundation's cyclic LR and a 75-epoch budget."""
    c = _hybrid_bgp_p15_z16_warm(data_patterns, repeat_indices=repeat_indices)
    c["epochs"] = 75
    c["lr"] = 0.0002
    c["scheduler_type"] = "cyclic"
    c["scheduler_kwargs"] = {"warmup": 2000, "min_factor": 0.1, "cyclic_period_steps": 6000}
    return c


kids_legacy_bgp_experiments["kids_legacy_hybrid_nla_m_bgp_z16_resnet_sc8a1_p15_warm_cyc"] = \
    _assert_final_summary_dim(_hybrid_bgp_p15_z16_warm_cyc(_BGP_SC8A1), 16,
                              "kids_legacy_hybrid_nla_m_bgp_z16_resnet_sc8a1_p15_warm_cyc")


# --- NLE Stage A: whitened (k=8) pretrain on GLASS embeddings ---
_NLE_REPEATS = (0, 1, 2, 3, 4)
_NLE_EPOCHS = 150
_BGP_NLE_PROJECT = "bgp-nle"


def _nle_pretrain_bgp(data_patterns, repeat, cosmo_param_names=None, preset_overrides=None):
    """Stage-A NLE pretrain row for one repeat on a bgp GLASS store (bare E groups, so eb_variant=None)."""
    kw = {}
    if cosmo_param_names is not None:
        kw["cosmo_param_names"] = cosmo_param_names
    if preset_overrides is not None:
        kw["preset_overrides"] = preset_overrides
    c = _nle_pretrain(data_patterns, None, whiten_k=8, epochs=_NLE_EPOCHS, **kw)
    c["project"] = _BGP_NLE_PROJECT
    return _nle_bake_repeat(c, repeat)


for _r in _NLE_REPEATS:
    kids_legacy_bgp_experiments[f"glass_nle_pretrain_nla_m_bgp_z8_r{_r}"] = \
        _nle_pretrain_bgp(_BGP_SC8A1, _r)


# --- nla variate encoder finetune (fixed LR) ---
_BGP_NLA = f"{_GPU5}/glass_bgp_nla_f16_sc8a1_{_EB}/output_*.h5"
_VARIATE_REPEATS = (0, 1, 2, 3, 4)


def _encoder_finetune_bgp(data_patterns, cosmo_param_names, preset_overrides=None,
                          repeat_indices=_VARIATE_REPEATS, scheduler="fixed"):
    """Whole-encoder finetune of the foundation on a bgp GLASS sub-variate store.

    8-D summary and a fresh flow head."""
    c = _hybrid_bgp(data_patterns, None, _BAND_CKPT_BGP, repeat_indices=repeat_indices)
    c.pop("pretrained_band_ckpt_path", None)
    c.pop("freeze_band", None)
    c["pretrained_embedding_ckpt_path"] = _SC8A1_9P_CKPT
    c["freeze_embedding_net"] = False
    c["match_num_cosmo"] = False
    c["cosmo_param_names"] = list(cosmo_param_names)
    if scheduler in ("exp", "exponential"):
        c["scheduler_type"] = "exp"
        c["scheduler_kwargs"] = {"gamma": 0.984, "warmup_steps": 0}
    else:
        c["scheduler_type"] = "fixed"
        c["scheduler_kwargs"] = {"warmup_steps": 0}
    c["lr"] = 0.0002
    c["epochs"] = 100
    if preset_overrides:
        c["scaler_options"] = {
            "data": {"type": "standard", "keys": None},
            "cosmo": {"type": "preset", "preset_overrides": dict(preset_overrides)},
        }
    return c


# scheduler 'exp' pinned: this row was trained with it.
kids_legacy_bgp_experiments["glass_encoder_finetune_nla_bgp_z8"] = \
    _assert_final_summary_dim(
        _encoder_finetune_bgp(_BGP_NLA, _COSMO_8_NLA, preset_overrides=_A_IA_NLA_BOX,
                              scheduler="exp"), 8,
        "glass_encoder_finetune_nla_bgp_z8")


# --- nla_z encoder finetune ---
_BGP_NLA_Z = f"{_GPU5}/glass_bgp_nla_z_f16_sc8a1_{_EB}/output_*.h5"

# scheduler 'exp' pinned: this row was trained with it.
kids_legacy_bgp_experiments["glass_encoder_finetune_nla_z_bgp_z8"] = \
    _assert_final_summary_dim(
        _encoder_finetune_bgp(_BGP_NLA_Z, _COSMO_9_NLAZ, preset_overrides=_A_IA_NLA_BOX,
                              scheduler="exp"), 8,
        "glass_encoder_finetune_nla_z_bgp_z8")


# --- nla_z encoder finetune at fixed LR ---
kids_legacy_bgp_experiments["glass_encoder_finetune_nla_z_bgp_z8_fixedlr"] = \
    _assert_final_summary_dim(
        _encoder_finetune_bgp(_BGP_NLA_Z, _COSMO_9_NLAZ, preset_overrides=_A_IA_NLA_BOX), 8,
        "glass_encoder_finetune_nla_z_bgp_z8_fixedlr")


# --- nla Stage A NLE pretrain ---
for _r in _NLE_REPEATS:
    kids_legacy_bgp_experiments[f"glass_nle_pretrain_nla_bgp_z8_r{_r}"] = \
        _nle_pretrain_bgp(_BGP_NLA, _r, cosmo_param_names=_COSMO_8_NLA,
                          preset_overrides=_A_IA_NLA_BOX)


# --- Gower transfer stage ---
_BGP_GOWER_NLA_M = f"{_GPU5}/gower_bgp_nla_m_f16_sc8a1_{_EB}/output_*.h5"
_GOWER_TEST_IDS = "config/fixed_test_sets/gower_test_ids.json"


# --- Gower NPE ensemble finetune ---
def _npe_finetune_bgp(data_patterns=_BGP_GOWER_NLA_M):
    c = _npe_finetune_z8(_SC8A1_9P_CKPT, data_patterns=data_patterns, eb_variant=None)
    c["model_kwargs"] = {**c["model_kwargs"], "map_kwargs": _RESNET_MAPKW}
    c["fixed_test_sim_ids"] = _GOWER_TEST_IDS
    c["project"] = "gower-finetuning"
    return c


kids_legacy_bgp_experiments["gower_npe_finetune_nla_m_bgp_z8"] = \
    _assert_final_summary_dim(_npe_finetune_bgp(), 8, "gower_npe_finetune_nla_m_bgp_z8")


# --- NLE Stage B: ens9 finetune per repeat ---
for _r in _NLE_REPEATS:
    _ft = _nle_finetune(f"glass_nle_pretrain_nla_m_bgp_z8_r{_r}", ensemble_repeats=9,
                        whiten_k=8, warmstart_max_gap_nats=22.0,
                        gower_data=_BGP_GOWER_NLA_M, gower_eb=None)
    _ft["max_trainval_cosmos"] = [300]
    _ft["train_frac"] = 0.8
    _ft["val_frac"] = 0.2
    _ft["test_frac"] = 0.0        # test set = fixed ids, so the fractions must sum to 1
    _ft["fixed_test_sim_ids"] = _GOWER_TEST_IDS
    _ft["project"] = _BGP_NLE_PROJECT
    kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_m_bgp_z8_r{_r}_ens9"] = \
        _nle_bake_repeat(_ft, _r)


# --- Ablation (repeat 4): adapt the compressor on Gower, then finetune the NLE on its embeddings ---
_ABLATION_REPEAT = 4


kids_legacy_bgp_experiments[f"gower_npe_finetune_nla_m_bgp_z8_r{_ABLATION_REPEAT}_ens1"] = \
    _assert_final_summary_dim(
        {**_npe_finetune_bgp(), "ensemble_repeats": 1, "repeat_indices": [_ABLATION_REPEAT]}, 8,
        f"gower_npe_finetune_nla_m_bgp_z8_r{_ABLATION_REPEAT}_ens1")


kids_legacy_bgp_experiments["gower_npe_finetune_nla_m_bgp_z8_ens1"] = \
    _assert_final_summary_dim(
        {**_npe_finetune_bgp(), "ensemble_repeats": 1, "repeat_indices": [0, 1, 2, 3, 4]}, 8,
        "gower_npe_finetune_nla_m_bgp_z8_ens1")


_ft_adapt = _nle_finetune(f"glass_nle_pretrain_nla_m_bgp_z8_r{_ABLATION_REPEAT}", ensemble_repeats=9,
                          whiten_k=8, warmstart_max_gap_nats=22.0,
                          gower_data=_BGP_GOWER_NLA_M, gower_eb=None)
_ft_adapt["max_trainval_cosmos"] = [300]
_ft_adapt["train_frac"] = 0.8
_ft_adapt["val_frac"] = 0.2
_ft_adapt["test_frac"] = 0.0
_ft_adapt["fixed_test_sim_ids"] = _GOWER_TEST_IDS
_ft_adapt["project"] = _BGP_NLE_PROJECT
_ft_adapt["match_num_cosmo"] = True
_ft_adapt["whiten_warmstart_max_gap_nats"] = 250.0
kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_m_bgp_z8_r{_ABLATION_REPEAT}_ens9_adapt"] = \
    _nle_bake_repeat(_ft_adapt, _ABLATION_REPEAT)


def _adapt_e150(load_flow: bool):
    c = _nle_finetune(f"glass_nle_pretrain_nla_m_bgp_z8_r{_ABLATION_REPEAT}", ensemble_repeats=9,
                      whiten_k=8, warmstart_max_gap_nats=250.0,
                      gower_data=_BGP_GOWER_NLA_M, gower_eb=None)
    c["max_trainval_cosmos"] = [300]
    c["train_frac"] = 0.8
    c["val_frac"] = 0.2
    c["test_frac"] = 0.0
    c["fixed_test_sim_ids"] = _GOWER_TEST_IDS
    c["project"] = _BGP_NLE_PROJECT
    c["match_num_cosmo"] = True
    c["epochs"] = _NLE_EPOCHS
    c["load_pretrained_flow"] = bool(load_flow)
    return _nle_bake_repeat(c, _ABLATION_REPEAT)


kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_m_bgp_z8_r{_ABLATION_REPEAT}_ens9_adapt_e150"] = \
    _adapt_e150(load_flow=True)

kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_m_bgp_z8_r{_ABLATION_REPEAT}_ens9_adapt_e150_scratch"] = \
    _adapt_e150(load_flow=False)


# --- Stage A NLE pretrain on the nla and nla_z encoders, k=5 ---
for _r in _VARIATE_REPEATS:
    _pre_k5 = _nle_pretrain_bgp(_BGP_NLA, _r, cosmo_param_names=_COSMO_8_NLA,
                                preset_overrides=_A_IA_NLA_BOX)
    _pre_k5["whiten_embeddings"] = {"k": 5}
    kids_legacy_bgp_experiments[f"glass_nle_pretrain_nla_bgp_z8_k5_r{_r}"] = _pre_k5


for _r in _VARIATE_REPEATS:
    _pre_nlaz_k5 = _nle_pretrain_bgp(_BGP_NLA_Z, _r, cosmo_param_names=_COSMO_9_NLAZ,
                                     preset_overrides=_A_IA_NLA_BOX)
    _pre_nlaz_k5["whiten_embeddings"] = {"k": 5}
    kids_legacy_bgp_experiments[f"glass_nle_pretrain_nla_z_bgp_z8_k5_r{_r}"] = _pre_nlaz_k5


# --- Stage B Gower NLE finetune for the nla and nla_z variates ---
_BGP_GOWER_NLA = f"{_GPU5}/gower_bgp_nla_f16_sc8a1_{_EB}/output_*.h5"
_GOWER_TEST_IDS_100 = "config/fixed_test_sets/gower_test_ids_100.json"

for _r in _VARIATE_REPEATS:
    _ft_nla = _nle_finetune(f"glass_nle_pretrain_nla_bgp_z8_k5_r{_r}", ensemble_repeats=9,
                            whiten_k=5, warmstart_max_gap_nats=22.0,
                            gower_data=_BGP_GOWER_NLA, gower_eb=None,
                            cosmo_param_names=_COSMO_8_NLA,
                            preset_overrides=_A_IA_NLA_BOX)
    # 199 of the 200 fixed test ids are on disk for the nla store
    _ft_nla["max_trainval_cosmos"] = None
    _ft_nla["train_frac"] = 0.8
    _ft_nla["val_frac"] = 0.2
    _ft_nla["test_frac"] = 0.0     # test set = fixed ids, so the fractions must sum to 1
    _ft_nla["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
    _ft_nla["project"] = _BGP_NLE_PROJECT
    kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_bgp_z8_r{_r}_ens9"] = \
        _nle_bake_repeat(_ft_nla, _r)

_E100_REPEAT = 4
_ft_nla_e100 = _nle_finetune(f"glass_nle_pretrain_nla_bgp_z8_k5_r{_E100_REPEAT}", ensemble_repeats=9,
                             whiten_k=5, warmstart_max_gap_nats=22.0,
                             gower_data=_BGP_GOWER_NLA, gower_eb=None,
                             cosmo_param_names=_COSMO_8_NLA,
                             preset_overrides=_A_IA_NLA_BOX)
_ft_nla_e100["max_trainval_cosmos"] = None
_ft_nla_e100["train_frac"] = 0.8
_ft_nla_e100["val_frac"] = 0.2
_ft_nla_e100["test_frac"] = 0.0
_ft_nla_e100["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
_ft_nla_e100["epochs"] = 100
_ft_nla_e100["project"] = _BGP_NLE_PROJECT
kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_bgp_z8_r{_E100_REPEAT}_ens9_e100"] = \
    _nle_bake_repeat(_ft_nla_e100, _E100_REPEAT)

for _r in _VARIATE_REPEATS:
    _ft_nla_e150 = _nle_finetune(f"glass_nle_pretrain_nla_bgp_z8_k5_r{_r}", ensemble_repeats=9,
                                 whiten_k=5, warmstart_max_gap_nats=22.0,
                                 gower_data=_BGP_GOWER_NLA, gower_eb=None,
                                 cosmo_param_names=_COSMO_8_NLA,
                                 preset_overrides=_A_IA_NLA_BOX)
    _ft_nla_e150["max_trainval_cosmos"] = None
    _ft_nla_e150["train_frac"] = 0.8
    _ft_nla_e150["val_frac"] = 0.2
    _ft_nla_e150["test_frac"] = 0.0
    _ft_nla_e150["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
    _ft_nla_e150["epochs"] = 150
    _ft_nla_e150["project"] = _BGP_NLE_PROJECT
    kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_bgp_z8_r{_r}_ens9_e150"] = \
        _nle_bake_repeat(_ft_nla_e150, _r)

_BGP_GOWER_NLA_Z = f"{_GPU5}/gower_bgp_nla_z_f16_sc8a1_{_EB}/output_*.h5"

for _r in _VARIATE_REPEATS:
    _ft_nlaz = _nle_finetune(f"glass_nle_pretrain_nla_z_bgp_z8_k5_r{_r}", ensemble_repeats=9,
                             whiten_k=5, warmstart_max_gap_nats=22.0,
                             gower_data=_BGP_GOWER_NLA_Z, gower_eb=None,
                             cosmo_param_names=_COSMO_9_NLAZ,
                             preset_overrides=_A_IA_NLA_BOX)
    _ft_nlaz["max_trainval_cosmos"] = None
    _ft_nlaz["train_frac"] = 0.8
    _ft_nlaz["val_frac"] = 0.2
    _ft_nlaz["test_frac"] = 0.0     # test set = fixed ids, so the fractions must sum to 1
    _ft_nlaz["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
    _ft_nlaz["epochs"] = 150
    _ft_nlaz["project"] = _BGP_NLE_PROJECT
    kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_z_bgp_z8_r{_r}_ens9_e150"] = \
        _nle_bake_repeat(_ft_nlaz, _r)


# --- Fixed-LR encoder re-run of both variate chains ---
kids_legacy_bgp_experiments["glass_encoder_finetune_nla_bgp_z8_fixedlr"] = \
    _assert_final_summary_dim(
        _encoder_finetune_bgp(_BGP_NLA, _COSMO_8_NLA, preset_overrides=_A_IA_NLA_BOX), 8,
        "glass_encoder_finetune_nla_bgp_z8_fixedlr")

for _r in _VARIATE_REPEATS:
    _pre_nla_fx = _nle_pretrain_bgp(_BGP_NLA, _r, cosmo_param_names=_COSMO_8_NLA,
                                    preset_overrides=_A_IA_NLA_BOX)
    _pre_nla_fx["whiten_embeddings"] = {"k": 5}
    kids_legacy_bgp_experiments[f"glass_nle_pretrain_nla_bgp_z8_k5_fx_r{_r}"] = _pre_nla_fx

    _pre_nlaz_fx = _nle_pretrain_bgp(_BGP_NLA_Z, _r, cosmo_param_names=_COSMO_9_NLAZ,
                                     preset_overrides=_A_IA_NLA_BOX)
    _pre_nlaz_fx["whiten_embeddings"] = {"k": 5}
    kids_legacy_bgp_experiments[f"glass_nle_pretrain_nla_z_bgp_z8_k5_fx_r{_r}"] = _pre_nlaz_fx

for _r in _VARIATE_REPEATS:
    _ft_nla_fx = _nle_finetune(f"glass_nle_pretrain_nla_bgp_z8_k5_fx_r{_r}", ensemble_repeats=9,
                               whiten_k=5, warmstart_max_gap_nats=22.0,
                               gower_data=_BGP_GOWER_NLA, gower_eb=None,
                               cosmo_param_names=_COSMO_8_NLA,
                               preset_overrides=_A_IA_NLA_BOX)
    _ft_nlaz_fx = _nle_finetune(f"glass_nle_pretrain_nla_z_bgp_z8_k5_fx_r{_r}", ensemble_repeats=9,
                                whiten_k=5, warmstart_max_gap_nats=22.0,
                                gower_data=_BGP_GOWER_NLA_Z, gower_eb=None,
                                cosmo_param_names=_COSMO_9_NLAZ,
                                preset_overrides=_A_IA_NLA_BOX)
    for _c in (_ft_nla_fx, _ft_nlaz_fx):
        _c["max_trainval_cosmos"] = None
        _c["train_frac"] = 0.8
        _c["val_frac"] = 0.2
        _c["test_frac"] = 0.0
        _c["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
        _c["epochs"] = 150
        _c["project"] = _BGP_NLE_PROJECT
    kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_bgp_z8_fx_r{_r}_ens9_e150"] = \
        _nle_bake_repeat(_ft_nla_fx, _r)
    kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_z_bgp_z8_fx_r{_r}_ens9_e150"] = \
        _nle_bake_repeat(_ft_nlaz_fx, _r)


# --- kappa=2 variate chain: encoder, Stage A, Stage B ---
_BG_CLIP = (0.3, 2.2)          # GALAXY_BIAS_CLIP, src/KiDS/simulation_config.py
_K2 = 2.0
_BG_BOXES_K2 = {
    f"b_g_bin{_i + 1}": (max(_m - 3.0 * _K2 * _s, _BG_CLIP[0]),
                         min(_m + 3.0 * _K2 * _s, _BG_CLIP[1]))
    for _i, (_m, _s) in enumerate(zip(_BG_PRIOR_MEANS, _BG_PRIOR_SIGMAS))
}

_BGPK2_GLASS = f"{_GPU5}/glass_bgpk2_nla_m_f16_sc8a1_{_EB}/output_*.h5"
_BGPK2_GOWER = f"{_GPU5}/gower_bgpk2_nla_m_f16_sc8a1_{_EB}/output_*.h5"
_P15_CYC_CKPT = f"{_CKPT}/kids_legacy_hybrid_nla_m_bgp_z16_resnet_sc8a1_p15_warm_cyc/"
_K2_REPEATS = (0, 1, 2, 3)


def _hybrid_bgpk2_z16(data_patterns, repeat_indices=_K2_REPEATS):
    """kappa=2 variate compressor: the p15 cyclic recipe, re-pointed at the kappa=2 store."""
    c = _hybrid_bgp_p15_z16_warm_cyc(data_patterns, repeat_indices=repeat_indices)
    c["pretrained_embedding_ckpt_path"] = _P15_CYC_CKPT
    c["scaler_options"] = {
        "data": {"type": "standard", "keys": None},
        "cosmo": {"type": "preset", "preset_overrides": dict(_BG_BOXES_K2)},
    }
    return c


kids_legacy_bgp_experiments["kids_legacy_hybrid_nla_m_bgpk2_z16_warm_cyc"] = \
    _assert_final_summary_dim(_hybrid_bgpk2_z16(_BGPK2_GLASS), 16,
                              "kids_legacy_hybrid_nla_m_bgpk2_z16_warm_cyc")

# --- kappa=2 Stage A and Stage B ---
_K2_THETA = list(
    kids_legacy_bgp_experiments["kids_legacy_hybrid_nla_m_bgpk2_z16_warm_cyc"]["cosmo_param_names"])
assert len(_K2_THETA) == 15, f"expected the 15-param kappa=2 vector, got {len(_K2_THETA)}"

for _r in _K2_REPEATS:
    _pre_k2 = _nle_pretrain_bgp(_BGPK2_GLASS, _r, cosmo_param_names=list(_K2_THETA),
                                preset_overrides=dict(_BG_BOXES_K2))
    _pre_k2["whiten_embeddings"] = {"k": 16}
    kids_legacy_bgp_experiments[f"glass_nle_pretrain_nla_m_bgpk2_z16_r{_r}"] = _pre_k2

    _ft_k2 = _nle_finetune(f"glass_nle_pretrain_nla_m_bgpk2_z16_r{_r}", ensemble_repeats=9,
                           whiten_k=16, warmstart_max_gap_nats=22.0,
                           gower_data=_BGPK2_GOWER, gower_eb=None,
                           cosmo_param_names=list(_K2_THETA),
                           preset_overrides=dict(_BG_BOXES_K2))
    _ft_k2["max_trainval_cosmos"] = None
    _ft_k2["train_frac"] = 0.8
    _ft_k2["val_frac"] = 0.2
    _ft_k2["test_frac"] = 0.0
    _ft_k2["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
    _ft_k2["epochs"] = 150
    _ft_k2["project"] = _BGP_NLE_PROJECT
    kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_m_bgpk2_z16_r{_r}_ens9_e150"] = \
        _nle_bake_repeat(_ft_k2, _r)

_K2_WHITEN_K5 = 5

for _r in _K2_REPEATS:
    _pre_k2b = _nle_pretrain_bgp(_BGPK2_GLASS, _r, cosmo_param_names=list(_K2_THETA),
                                 preset_overrides=dict(_BG_BOXES_K2))
    _pre_k2b["whiten_embeddings"] = {"k": _K2_WHITEN_K5}
    kids_legacy_bgp_experiments[f"glass_nle_pretrain_nla_m_bgpk2_z16_k5_r{_r}"] = _pre_k2b

    _ft_k2b = _nle_finetune(f"glass_nle_pretrain_nla_m_bgpk2_z16_k5_r{_r}", ensemble_repeats=9,
                            whiten_k=_K2_WHITEN_K5, warmstart_max_gap_nats=22.0,
                            gower_data=_BGPK2_GOWER, gower_eb=None,
                            cosmo_param_names=list(_K2_THETA),
                            preset_overrides=dict(_BG_BOXES_K2))
    _ft_k2b["max_trainval_cosmos"] = None
    _ft_k2b["train_frac"] = 0.8
    _ft_k2b["val_frac"] = 0.2
    _ft_k2b["test_frac"] = 0.0
    _ft_k2b["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
    _ft_k2b["epochs"] = 150
    _ft_k2b["project"] = _BGP_NLE_PROJECT
    _ft_k2b["run_training"] = False
    kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_m_bgpk2_z16_k5_r{_r}_ens9_e150"] = \
        _nle_bake_repeat(_ft_k2b, _r)


for _r in _K2_REPEATS:
    _pre_hf = _nle_pretrain_bgp(_BGPK2_GLASS, _r, cosmo_param_names=list(_K2_THETA),
                                preset_overrides=dict(_BG_BOXES_K2))
    _pre_hf["whiten_embeddings"] = {"k": _K2_WHITEN_K5}
    kids_legacy_bgp_experiments[f"glass_nle_pretrain_nla_m_bgpk2_z16_k5_hf_r{_r}"] = _pre_hf

    _ft_hf = _nle_finetune(f"glass_nle_pretrain_nla_m_bgpk2_z16_k5_hf_r{_r}", ensemble_repeats=9,
                           whiten_k=_K2_WHITEN_K5, warmstart_max_gap_nats=22.0,
                           gower_data=_BGPK2_GOWER, gower_eb=None,
                           cosmo_param_names=list(_K2_THETA),
                           preset_overrides=dict(_BG_BOXES_K2))
    _ft_hf["max_trainval_cosmos"] = None
    _ft_hf["train_frac"] = 0.8
    _ft_hf["val_frac"] = 0.2
    _ft_hf["test_frac"] = 0.0
    _ft_hf["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
    _ft_hf["epochs"] = 150
    _ft_hf["project"] = _BGP_NLE_PROJECT
    kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_m_bgpk2_z16_k5_hf_r{_r}_ens9_e150"] = \
        _nle_bake_repeat(_ft_hf, _r)


def _gower_npe_ft_bgpk2_z16(repeat_indices=(0,)):
    c = _hybrid_bgpk2_z16(_BGPK2_GOWER, repeat_indices=repeat_indices)
    c["eb_map_variant"] = None
    c["checkpoint_path"] = f"{_CKPT}/kids_legacy_hybrid_nla_m_bgpk2_z16_warm_cyc/"
    c.pop("pretrained_embedding_ckpt_path", None)
    c.pop("pretrained_band_ckpt_path", None)
    c["freeze_band"] = False
    c["epochs"] = 10
    c["lr"] = 1e-5
    c["batch_size"] = 128
    c["scheduler_type"] = "exp"
    c["scheduler_kwargs"] = {"warmup": 0}
    c["ensemble_repeats"] = 9
    c["max_trainval_cosmos"] = None
    c["train_frac"] = 0.8
    c["val_frac"] = 0.2
    c["test_frac"] = 0.0
    c["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
    c["match_num_cosmo"] = False
    c["project"] = "gower-finetuning"
    return c


kids_legacy_bgp_experiments["gower_npe_finetune_nla_m_bgpk2_z16_r0_ens9"] = \
    _assert_final_summary_dim(_gower_npe_ft_bgpk2_z16((0,)), 16,
                              "gower_npe_finetune_nla_m_bgpk2_z16_r0_ens9")


# --- vd variate chain: encoder finetune, Stage A, Stage B ---
_BGP_NLA_M_VD = f"{_GPU5}/glass_bgp_nla_m_vd_f16_sc8a1_{_EB}/output_*.h5"

_COSMO_9_NLAM = ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"]

kids_legacy_bgp_experiments["glass_encoder_finetune_nla_m_vd_bgp_z8"] = \
    _assert_final_summary_dim(
        _encoder_finetune_bgp(_BGP_NLA_M_VD, _COSMO_9_NLAM), 8,
        "glass_encoder_finetune_nla_m_vd_bgp_z8")

for _r in _VARIATE_REPEATS:
    _pre_vd = _nle_pretrain_bgp(_BGP_NLA_M_VD, _r, cosmo_param_names=_COSMO_9_NLAM)
    _pre_vd["whiten_embeddings"] = {"k": 5}
    kids_legacy_bgp_experiments[f"glass_nle_pretrain_nla_m_vd_bgp_z8_k5_r{_r}"] = _pre_vd

_BGP_GOWER_NLA_M_VD = f"{_GPU5}/gower_bgp_nla_m_vd_f16_sc8a1_{_EB}/output_*.h5"

for _r in _VARIATE_REPEATS:
    _ft_vd = _nle_finetune(f"glass_nle_pretrain_nla_m_vd_bgp_z8_k5_r{_r}", ensemble_repeats=9,
                           whiten_k=5, warmstart_max_gap_nats=22.0,
                           gower_data=_BGP_GOWER_NLA_M_VD, gower_eb=None,
                           cosmo_param_names=_COSMO_9_NLAM)
    _ft_vd["max_trainval_cosmos"] = None
    _ft_vd["train_frac"] = 0.8
    _ft_vd["val_frac"] = 0.2
    _ft_vd["test_frac"] = 0.0        # test set = fixed ids, so the fractions must sum to 1
    _ft_vd["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
    _ft_vd["epochs"] = 150
    _ft_vd["project"] = _BGP_NLE_PROJECT
    kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_m_vd_bgp_z8_r{_r}_ens9_e150"] = \
        _nle_bake_repeat(_ft_vd, _r)


# --- 2-parameter compression-ceiling suite (m15a-d) ---
_M15_REPEATS = [0, 1, 2]
_COSMO_2 = ["omega_m", "sigma_8"]
_M15_BAND_CKPT = f"{_CKPT}/m15a_band_nla_p2/"


def _m15_common(c):
    """Shared settings for the m15 arms: 2-parameter theta, 3 repeats, per-repeat parent resolution."""
    c["cosmo_param_names"] = list(_COSMO_2)
    c.pop("repeats", None)
    c["repeat_indices"] = list(_M15_REPEATS)
    c["match_num_cosmo"] = False
    c["project"] = "bgp-p2-ceiling"
    return c


_m15a = _m15_common(_band_bgp(_BGP_NLA))
_m15a["pretrained_embedding_ckpt_path"] = _BAND_CKPT_BGP
_m15a["freeze_embedding_net"] = False
kids_legacy_bgp_experiments["m15a_band_nla_p2"] = _m15a


def _m15_hybrid(band_ckpt, warm_map, key):
    """Hybrid arm with a frozen `band_ckpt`.

    warm_map warm-starts the map branch from the 9-parameter foundation."""
    c = _hybrid_bgp(_BGP_NLA, None, band_ckpt, repeat_indices=_M15_REPEATS)
    c = _m15_common(c)
    c["pretrained_band_ckpt_path"] = band_ckpt
    c["freeze_band"] = True
    if warm_map:
        c["pretrained_embedding_ckpt_path"] = _SC8A1_9P_CKPT
        c["freeze_embedding_net"] = False
        c["band_load_after_embedding"] = True
    else:
        c.pop("pretrained_embedding_ckpt_path", None)
        c["freeze_embedding_net"] = False
        c["band_load_after_embedding"] = True
    return _assert_final_summary_dim(c, 8, key)


kids_legacy_bgp_experiments["m15b_hybrid_nla_p2_newband_warmmap"] = \
    _m15_hybrid(_M15_BAND_CKPT, warm_map=True, key="m15b_hybrid_nla_p2_newband_warmmap")

kids_legacy_bgp_experiments["m15c_hybrid_nla_p2_oldband_warmmap"] = \
    _m15_hybrid(_BAND_CKPT_BGP, warm_map=True, key="m15c_hybrid_nla_p2_oldband_warmmap")

kids_legacy_bgp_experiments["m15d_hybrid_nla_p2_newband_scratchmap"] = \
    _m15_hybrid(_M15_BAND_CKPT, warm_map=False, key="m15d_hybrid_nla_p2_newband_scratchmap")


# --- _hf retrain rows: Stage A/B chains re-run from the corrected source encoders ---
_HF_CHAINS = (
    # (tag, glass_store, gower_store, theta, preset_overrides, source encoder)
    ("nla_bgp_z8",      _BGP_NLA,      _BGP_GOWER_NLA,      _COSMO_8_NLA,  _A_IA_NLA_BOX,
     "glass_encoder_finetune_nla_bgp_z8"),
    ("nla_z_bgp_z8",    _BGP_NLA_Z,    _BGP_GOWER_NLA_Z,    _COSMO_9_NLAZ, _A_IA_NLA_BOX,
     "glass_encoder_finetune_nla_z_bgp_z8"),
    ("nla_m_vd_bgp_z8", _BGP_NLA_M_VD, _BGP_GOWER_NLA_M_VD, _COSMO_9_NLAM, None,
     "glass_encoder_finetune_nla_m_vd_bgp_z8"),
)
HF_RETRAIN_SOURCES = {}   # {experiment_name: source encoder to pass as --sources}

for _tag, _glass, _gower, _theta, _box, _src in _HF_CHAINS:
    for _r in _VARIATE_REPEATS:
        _a_name = f"glass_nle_pretrain_{_tag}_k5_hf_r{_r}"
        _kw = {"cosmo_param_names": _theta}
        if _box is not None:
            _kw["preset_overrides"] = _box
        _a = _nle_pretrain_bgp(_glass, _r, **_kw)
        _a["whiten_embeddings"] = {"k": 5}
        kids_legacy_bgp_experiments[_a_name] = _a
        HF_RETRAIN_SOURCES[_a_name] = _src

        _b_name = f"gower_nle_finetune_{_tag}_hf_r{_r}_ens9_e150"
        _b = _nle_finetune(_a_name, ensemble_repeats=9, whiten_k=5, warmstart_max_gap_nats=50.0,
                           gower_data=_gower, gower_eb=None,
                           cosmo_param_names=_theta,
                           preset_overrides=_box)
        _b["max_trainval_cosmos"] = None
        _b["train_frac"] = 0.8
        _b["val_frac"] = 0.2
        _b["test_frac"] = 0.0
        _b["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
        _b["epochs"] = 150
        _b["project"] = _BGP_NLE_PROJECT
        kids_legacy_bgp_experiments[_b_name] = _nle_bake_repeat(_b, _r)
        HF_RETRAIN_SOURCES[_b_name] = _src

for _r in _VARIATE_REPEATS:
    _vd_b = kids_legacy_bgp_experiments.get(f"gower_nle_finetune_nla_m_vd_bgp_z8_hf_r{_r}_ens9_e150")
    if _vd_b is not None:
        _vd_b["sampling_num_jobs"] = 63

for _r in _K2_REPEATS:
    HF_RETRAIN_SOURCES[f"glass_nle_pretrain_nla_m_bgpk2_z16_k5_hf_r{_r}"] = \
        "kids_legacy_hybrid_nla_m_bgpk2_z16_warm_cyc"
    HF_RETRAIN_SOURCES[f"gower_nle_finetune_nla_m_bgpk2_z16_k5_hf_r{_r}_ens9_e150"] = \
        "kids_legacy_hybrid_nla_m_bgpk2_z16_warm_cyc"


# Source encoder of each Stage-B row (read by src/ml/eval/nle_external.py)
FLAGSHIP_NLE_SOURCES = {}
for _r in range(5):
    FLAGSHIP_NLE_SOURCES[f"gower_nle_finetune_nla_m_bgp_z8_r{_r}_ens9"] = \
        "kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1"
    FLAGSHIP_NLE_SOURCES[f"gower_nle_finetune_nla_m_z8_r{_r}_ens9"] = \
        "kids_legacy_hybrid_nla_m_lmin50_fwhm4_z8"


# --- 2-pt band NLE chain (Stage A/B), whitening k=8 ---
_BAND_NLE_REPEATS = (0, 1, 2, 3, 4)
_BAND_NLE_SOURCE = "kids_legacy_band_nla_m_bgp"


def _register_band_nle_chain(k):
    """Register the Stage A and Stage B rows of the band NLE chain at whitening k.

    k is part of the experiment name."""
    for _r in _BAND_NLE_REPEATS:
        pre_name = f"glass_nle_pretrain_band_nla_m_bgp_k{k}_r{_r}"
        ft_name = f"gower_nle_finetune_band_nla_m_bgp_k{k}_r{_r}_ens9_e150"

        pre = _nle_pretrain_bgp(_BGP_A1, _r)
        pre["whiten_embeddings"] = {"k": int(k)}
        kids_legacy_bgp_experiments[pre_name] = pre

        # guard threshold 50: member epoch-0 gaps exceed the default 22 on this chain
        ft = _nle_finetune(pre_name, ensemble_repeats=9, whiten_k=int(k),
                           warmstart_max_gap_nats=50.0,
                           gower_data=_BGP_GOWER_NLA_M, gower_eb=None)
        ft["max_trainval_cosmos"] = [300]
        ft["train_frac"] = 0.8
        ft["val_frac"] = 0.2
        ft["test_frac"] = 0.0         # test set = fixed ids, so the fractions must sum to 1
        ft["fixed_test_sim_ids"] = _GOWER_TEST_IDS
        ft["epochs"] = 150
        ft["project"] = _BGP_NLE_PROJECT
        kids_legacy_bgp_experiments[ft_name] = _nle_bake_repeat(ft, _r)

        FLAGSHIP_NLE_SOURCES[pre_name] = _BAND_NLE_SOURCE
        FLAGSHIP_NLE_SOURCES[ft_name] = _BAND_NLE_SOURCE


_BAND_NLE_WHITEN_K = 8
_register_band_nle_chain(4)
_register_band_nle_chain(_BAND_NLE_WHITEN_K)


def _npe_finetune_band_bgp(data_patterns=_BGP_GOWER_NLA_M):
    """Band-only (2-pt) Gower NPE ens9 finetune: `_npe_finetune_z8`'s recipe on `_band_bgp`'s arch."""
    c = _band_bgp(_BGP_A1)                     # architecture of the checkpoint being loaded
    c["data_patterns"] = data_patterns
    c["checkpoint_path"] = _BAND_CKPT_BGP      # whole-model load (band + NPE flow), per-repeat "_{i}"
    c["epochs"] = 10
    c["lr"] = 1e-5
    c["batch_size"] = 128
    c["scheduler_type"] = "exp"
    c["scheduler_kwargs"] = {"warmup": 0}
    c["ensemble_repeats"] = 9
    c["max_trainval_cosmos"] = [300]
    c["train_frac"] = 0.8
    c["val_frac"] = 0.2
    c["test_frac"] = 0.0                       # test set = fixed ids, so the fractions must sum to 1
    c["fixed_test_sim_ids"] = _GOWER_TEST_IDS
    c["match_num_cosmo"] = False
    c["repeat_indices"] = [0, 1, 2, 3, 4]
    c.pop("repeats", None)
    c["project"] = "gower-finetuning"
    return c


kids_legacy_bgp_experiments["gower_npe_finetune_band_nla_m_bgp_ens9"] = \
    _assert_final_summary_dim(_npe_finetune_band_bgp(), 8,
                              "gower_npe_finetune_band_nla_m_bgp_ens9")
