"""Experiment rows for the variable-depth (VD, quadcap table) bias-marginalised suite.

Stage I trains the bandpower MLP, and Stage II freezes it inside the ResNet map hybrid. Stage-A/B rows
register the NLE chains for the field, band, kappa=2 and variate arms.
"""
from config.kids_legacy import _nle_finetune
from config.kids_legacy import _npe_finetune_z8
from config.kids_legacy_bgp import (
    _EB,
    _GOWER_TEST_IDS,
    _GOWER_TEST_IDS_100,
    _NLE_REPEATS,
    _BGP_NLE_PROJECT,
    _RESNET_MAPKW,
    _assert_final_summary_dim,
    _band_bgp,
    _hybrid_bgp,
    _hybrid_bgp_p15,
    _hybrid_bgp_smoke,
    _nle_bake_repeat,
    _nle_pretrain_bgp,
    _npe_finetune_band_bgp,
)

# --- stores ---
_GPU5 = "/share/gpu5/asaoulis/transfer_datasets"
_CKPT = "/share/gpu5/asaoulis/transfer_models/checkpoints"

_VDQ_GLASS = f"{_GPU5}/glass_vdq_nla_m_f16_sc8a1_{_EB}/output_*.h5"
_VDQ_GLASS_CANARY = f"{_GPU5}/glass_vdq_nla_m_canary_f16_sc8a1_{_EB}/output_*.h5"
_VDQ_GOWER = f"{_GPU5}/gower_vdq_nla_m_f16_sc8a1_{_EB}/output_*.h5"
_VDQ_GOWER_NLA = f"{_GPU5}/gower_vdq_nla_f16_sc8a1_{_EB}/output_*.h5"
_VDQ_GOWER_NLA_Z = f"{_GPU5}/gower_vdq_nla_z_f16_sc8a1_{_EB}/output_*.h5"

# --- checkpoint dirs ---
_BAND_CKPT_VDQ = f"{_CKPT}/kids_legacy_band_nla_m_vdq/"
_VDQ_HYB_CKPT = f"{_CKPT}/kids_legacy_hybrid_nla_m_vdq_z8_resnet_sc8a1/"
_BAND_CKPT_VDQ_CANARY = f"{_CKPT}/kids_legacy_band_nla_m_vdq_canary/"
_VDQ_HYB_CKPT_CANARY = f"{_CKPT}/kids_legacy_hybrid_nla_m_vdq_z8_resnet_sc8a1_canary/"

_PACK_REPEATS = [0, 1, 2, 3, 4]
_HYBRID_REPEATS_VDQ = (0, 1, 2, 3, 4)

# Registry keys are literal strings (the launcher greps for them), never f-strings.
kids_legacy_vdq_experiments = {}


# --- Stage I band MLP and Stage II GLASS hybrid foundation ---
kids_legacy_vdq_experiments["kids_legacy_band_nla_m_vdq"] = _band_bgp(_VDQ_GLASS)

kids_legacy_vdq_experiments["kids_legacy_hybrid_nla_m_vdq_z8_resnet_sc8a1"] = \
    _hybrid_bgp(_VDQ_GLASS, None, _BAND_CKPT_VDQ, repeat_indices=_HYBRID_REPEATS_VDQ)

kids_legacy_vdq_experiments["kids_legacy_hybrid_nla_m_vdq_z8_resnet_sc8a1_smoke"] = _hybrid_bgp_smoke()


# --- Gower NPE finetunes (the encoder pack for misspec and Tier-2) ---
_FINETUNE_VAL_CHECK_INTERVAL = 0.25


def _npe_pack_vdq(hyb_ckpt, repeat_indices=_PACK_REPEATS, data_patterns=_VDQ_GOWER):
    c = _npe_finetune_z8(hyb_ckpt, data_patterns=data_patterns, eb_variant=None)
    c["model_kwargs"] = {**c["model_kwargs"], "map_kwargs": _RESNET_MAPKW}
    c["fixed_test_sim_ids"] = _GOWER_TEST_IDS
    c["project"] = "gower-finetuning"
    c["ensemble_repeats"] = 1
    c["repeat_indices"] = list(repeat_indices)
    # validate 4x per epoch: the 10-epoch finetunes otherwise pick from too few points
    c["val_check_interval"] = _FINETUNE_VAL_CHECK_INTERVAL
    return c


kids_legacy_vdq_experiments["gower_npe_finetune_nla_m_vdq_z8_ens1"] = _assert_final_summary_dim(
    _npe_pack_vdq(_VDQ_HYB_CKPT), 8, "gower_npe_finetune_nla_m_vdq_z8_ens1")


_band_canary = _band_bgp(_VDQ_GLASS_CANARY)
_band_canary["repeat_indices"] = [0]
kids_legacy_vdq_experiments["kids_legacy_band_nla_m_vdq_canary"] = _band_canary
kids_legacy_vdq_experiments["kids_legacy_hybrid_nla_m_vdq_z8_resnet_sc8a1_canary"] = \
    _hybrid_bgp(_VDQ_GLASS_CANARY, None, _BAND_CKPT_VDQ_CANARY, repeat_indices=(0,))
kids_legacy_vdq_experiments["gower_npe_finetune_nla_m_vdq_z8_canary_ens1"] = _assert_final_summary_dim(
    _npe_pack_vdq(_VDQ_HYB_CKPT_CANARY, repeat_indices=[0]), 8,
    "gower_npe_finetune_nla_m_vdq_z8_canary_ens1")


# --- 2-pt-only Gower NPE ens9 (band arch) ---
_band_pack = _npe_finetune_band_bgp(_VDQ_GOWER)
_band_pack["checkpoint_path"] = _BAND_CKPT_VDQ
_band_pack["val_check_interval"] = _FINETUNE_VAL_CHECK_INTERVAL
kids_legacy_vdq_experiments["gower_npe_finetune_band_nla_m_vdq_ens9"] = _assert_final_summary_dim(
    _band_pack, 8, "gower_npe_finetune_band_nla_m_vdq_ens9")


# --- Whitened NLE chain: Stage A on GLASS, Stage B ens9 on Gower, one row per repeat ---
def _nle_a_vdq(r):
    return _nle_pretrain_bgp(_VDQ_GLASS, r)


def _nle_b_vdq(r):
    ft = _nle_finetune(f"glass_nle_pretrain_nla_m_vdq_z8_r{r}", ensemble_repeats=9,
                       whiten_k=8, warmstart_max_gap_nats=22.0,
                       gower_data=_VDQ_GOWER, gower_eb=None)
    ft["max_trainval_cosmos"] = [300]
    ft["train_frac"] = 0.8
    ft["val_frac"] = 0.2
    ft["test_frac"] = 0.0        # test = the fixed 200 ids; fracs must sum to 1.0
    ft["fixed_test_sim_ids"] = _GOWER_TEST_IDS
    ft["project"] = _BGP_NLE_PROJECT
    return _nle_bake_repeat(ft, r)


assert tuple(_NLE_REPEATS) == (0, 1, 2, 3, 4), _NLE_REPEATS
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdq_z8_r0"] = _nle_a_vdq(0)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdq_z8_r1"] = _nle_a_vdq(1)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdq_z8_r2"] = _nle_a_vdq(2)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdq_z8_r3"] = _nle_a_vdq(3)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdq_z8_r4"] = _nle_a_vdq(4)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdq_z8_r0_ens9"] = _nle_b_vdq(0)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdq_z8_r1_ens9"] = _nle_b_vdq(1)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdq_z8_r2_ens9"] = _nle_b_vdq(2)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdq_z8_r3_ens9"] = _nle_b_vdq(3)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdq_z8_r4_ens9"] = _nle_b_vdq(4)



# --- 2-pt band NLE chain (k=8) on the VDQ stores ---
_BAND_NLE_K_VDQ = 8


def _band_nle_a_vdq(r):
    pre = _nle_pretrain_bgp(_VDQ_GLASS, r)
    pre["whiten_embeddings"] = {"k": _BAND_NLE_K_VDQ}
    return pre


def _band_nle_b_vdq(r):
    ft = _nle_finetune(f"glass_nle_pretrain_band_nla_m_vdq_k{_BAND_NLE_K_VDQ}_r{r}", ensemble_repeats=9,
                       whiten_k=_BAND_NLE_K_VDQ, warmstart_max_gap_nats=50.0,
                       gower_data=_VDQ_GOWER, gower_eb=None)
    ft["max_trainval_cosmos"] = [300]
    ft["train_frac"] = 0.8
    ft["val_frac"] = 0.2
    ft["test_frac"] = 0.0        # test = the fixed 200 ids; fracs must sum to 1.0
    ft["fixed_test_sim_ids"] = _GOWER_TEST_IDS
    ft["epochs"] = 150
    ft["project"] = _BGP_NLE_PROJECT
    return _nle_bake_repeat(ft, r)


kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_m_vdq_k8_r0"] = _band_nle_a_vdq(0)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_m_vdq_k8_r1"] = _band_nle_a_vdq(1)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_m_vdq_k8_r2"] = _band_nle_a_vdq(2)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_m_vdq_k8_r3"] = _band_nle_a_vdq(3)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_m_vdq_k8_r4"] = _band_nle_a_vdq(4)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_m_vdq_k8_r0_ens9_e150"] = _band_nle_b_vdq(0)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_m_vdq_k8_r1_ens9_e150"] = _band_nle_b_vdq(1)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_m_vdq_k8_r2_ens9_e150"] = _band_nle_b_vdq(2)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_m_vdq_k8_r3_ens9_e150"] = _band_nle_b_vdq(3)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_m_vdq_k8_r4_ens9_e150"] = _band_nle_b_vdq(4)

kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_m_vdq_k8_r3_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 100.0

# --- 15-parameter GLASS foundation, 125 epochs ---
kids_legacy_vdq_experiments["kids_legacy_hybrid_nla_m_vdq_z8_resnet_sc8a1_p15"] = \
    _hybrid_bgp_p15(_VDQ_GLASS, _BAND_CKPT_VDQ)


# source encoder of each Stage-B row (read by src/ml/eval/nle_external.py)
VDQ_NLE_SOURCES = {}
for _r in _NLE_REPEATS:
    VDQ_NLE_SOURCES[f"glass_nle_pretrain_nla_m_vdq_z8_r{_r}"] = "kids_legacy_hybrid_nla_m_vdq_z8_resnet_sc8a1"
    VDQ_NLE_SOURCES[f"gower_nle_finetune_nla_m_vdq_z8_r{_r}_ens9"] = "kids_legacy_hybrid_nla_m_vdq_z8_resnet_sc8a1"
    VDQ_NLE_SOURCES[f"glass_nle_pretrain_band_nla_m_vdq_k{_BAND_NLE_K_VDQ}_r{_r}"] = "kids_legacy_band_nla_m_vdq"
    VDQ_NLE_SOURCES[f"gower_nle_finetune_band_nla_m_vdq_k{_BAND_NLE_K_VDQ}_r{_r}_ens9_e150"] = "kids_legacy_band_nla_m_vdq"
assert set(VDQ_NLE_SOURCES) <= set(kids_legacy_vdq_experiments), \
    sorted(set(VDQ_NLE_SOURCES) - set(kids_legacy_vdq_experiments))


# --- GLASS VDQ analysis variates: encoder finetunes and Stage A ---
from config.kids_legacy import _A_IA_NLA_BOX, _COSMO_8_NLA, _COSMO_9_NLAZ  # noqa: E402
from config.kids_legacy_bgp import _BG_BOXES_K2, _COSMO_9_NLAM, _encoder_finetune_bgp  # noqa: E402

_VDQ_GLASS_NLA = f"{_GPU5}/glass_vdq_nla_f16_sc8a1_{_EB}/output_*.h5"
_VDQ_GLASS_NLA_Z = f"{_GPU5}/glass_vdq_nla_z_f16_sc8a1_{_EB}/output_*.h5"
_VDQ_GLASS_K2 = f"{_GPU5}/glass_vdqk2_nla_m_f16_sc8a1_{_EB}/output_*.h5"
_BG_THETA = [f"b_g_bin{_i}" for _i in range(1, 7)]
_VARIATE_NLE_K_VDQ = 5


def _encoder_finetune_vdq(data_patterns, cosmo_param_names, preset_overrides=None, load_flow=False):
    c = _encoder_finetune_bgp(data_patterns, cosmo_param_names, preset_overrides=preset_overrides)
    c["pretrained_embedding_ckpt_path"] = _VDQ_HYB_CKPT
    if load_flow:
        c["load_pretrained_flow"] = True
        c["pretrained_flow_ckpt_path"] = _VDQ_HYB_CKPT
    return c


kids_legacy_vdq_experiments["glass_encoder_finetune_nla_vdq_z8"] = _assert_final_summary_dim(
    _encoder_finetune_vdq(_VDQ_GLASS_NLA, _COSMO_8_NLA, preset_overrides=_A_IA_NLA_BOX), 8,
    "glass_encoder_finetune_nla_vdq_z8")
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_z_vdq_z8"] = _assert_final_summary_dim(
    _encoder_finetune_vdq(_VDQ_GLASS_NLA_Z, _COSMO_9_NLAZ, preset_overrides=_A_IA_NLA_BOX), 8,
    "glass_encoder_finetune_nla_z_vdq_z8")
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vdqk2_z8_cosmo"] = _assert_final_summary_dim(
    _encoder_finetune_vdq(_VDQ_GLASS_K2, _COSMO_9_NLAM, load_flow=True), 8,
    "glass_encoder_finetune_nla_m_vdqk2_z8_cosmo")
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vdqk2_z8_bg"] = _assert_final_summary_dim(
    _encoder_finetune_vdq(_VDQ_GLASS_K2, _BG_THETA, preset_overrides=_BG_BOXES_K2), 8,
    "glass_encoder_finetune_nla_m_vdqk2_z8_bg")


def _variate_nle_a_vdq(data_patterns, r, cosmo_param_names, preset_overrides=None):
    kw = {"cosmo_param_names": list(cosmo_param_names)}
    if preset_overrides is not None:
        kw["preset_overrides"] = preset_overrides
    pre = _nle_pretrain_bgp(data_patterns, r, **kw)
    pre["whiten_embeddings"] = {"k": _VARIATE_NLE_K_VDQ}
    return pre


kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_k5_r0"] = _variate_nle_a_vdq(_VDQ_GLASS_NLA, 0, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_k5_r1"] = _variate_nle_a_vdq(_VDQ_GLASS_NLA, 1, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_k5_r2"] = _variate_nle_a_vdq(_VDQ_GLASS_NLA, 2, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_k5_r3"] = _variate_nle_a_vdq(_VDQ_GLASS_NLA, 3, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_k5_r4"] = _variate_nle_a_vdq(_VDQ_GLASS_NLA, 4, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_z_vdq_z8_k5_r0"] = _variate_nle_a_vdq(_VDQ_GLASS_NLA_Z, 0, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_z_vdq_z8_k5_r1"] = _variate_nle_a_vdq(_VDQ_GLASS_NLA_Z, 1, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_z_vdq_z8_k5_r2"] = _variate_nle_a_vdq(_VDQ_GLASS_NLA_Z, 2, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_z_vdq_z8_k5_r3"] = _variate_nle_a_vdq(_VDQ_GLASS_NLA_Z, 3, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_z_vdq_z8_k5_r4"] = _variate_nle_a_vdq(_VDQ_GLASS_NLA_Z, 4, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k5_r0"] = _variate_nle_a_vdq(_VDQ_GLASS_K2, 0, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k5_r1"] = _variate_nle_a_vdq(_VDQ_GLASS_K2, 1, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k5_r2"] = _variate_nle_a_vdq(_VDQ_GLASS_K2, 2, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k5_r3"] = _variate_nle_a_vdq(_VDQ_GLASS_K2, 3, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k5_r4"] = _variate_nle_a_vdq(_VDQ_GLASS_K2, 4, _COSMO_9_NLAM)

VDQ_VARIATE_NLE_SOURCES = {}
for _r in _NLE_REPEATS:
    VDQ_VARIATE_NLE_SOURCES[f"glass_nle_pretrain_nla_vdq_z8_k5_r{_r}"] = "glass_encoder_finetune_nla_vdq_z8"
    VDQ_VARIATE_NLE_SOURCES[f"glass_nle_pretrain_nla_z_vdq_z8_k5_r{_r}"] = "glass_encoder_finetune_nla_z_vdq_z8"
    VDQ_VARIATE_NLE_SOURCES[f"glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k5_r{_r}"] = \
        "glass_encoder_finetune_nla_m_vdqk2_z8_cosmo"
assert set(VDQ_VARIATE_NLE_SOURCES) <= set(kids_legacy_vdq_experiments), \
    sorted(set(VDQ_VARIATE_NLE_SOURCES) - set(kids_legacy_vdq_experiments))


_K2_JOINT_SOURCES = "glass_encoder_finetune_nla_m_vdqk2_z8_cosmo,glass_encoder_finetune_nla_m_vdqk2_z8_bg"
_K2_JOINT_THETA = list(_COSMO_9_NLAM) + list(_BG_THETA)
assert len(_K2_JOINT_THETA) == 15, _K2_JOINT_THETA


def _k2_npe_head(r, k, theta, preset_overrides=None, cache_name=None):
    kw = {"cosmo_param_names": list(theta)}
    if preset_overrides is not None:
        kw["preset_overrides"] = preset_overrides
    c = _nle_pretrain_bgp(_VDQ_GLASS_K2, r, **kw)
    c["inference_mode"] = "npe"
    c["run_evaluation"] = True
    c["whiten_embeddings"] = {"k": int(k)}
    if cache_name is not None:
        c["embedding_cache_name"] = cache_name
        c["reuse_embedding_cache"] = True
    return c


kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k16_r0"] = _k2_npe_head(0, 16, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r0")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k12_r0"] = _k2_npe_head(0, 12, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r0")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k10_r0"] = _k2_npe_head(0, 10, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r0")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_cosmo_ctl_r0"] = _k2_npe_head(0, 8, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_bg_ctl_r0"] = _k2_npe_head(0, 8, _BG_THETA, _BG_BOXES_K2)
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k16_r1"] = _k2_npe_head(1, 16, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r1")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k12_r1"] = _k2_npe_head(1, 12, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r1")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k10_r1"] = _k2_npe_head(1, 10, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r1")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_cosmo_ctl_r1"] = _k2_npe_head(1, 8, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_bg_ctl_r1"] = _k2_npe_head(1, 8, _BG_THETA, _BG_BOXES_K2)
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k16_r2"] = _k2_npe_head(2, 16, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r2")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k12_r2"] = _k2_npe_head(2, 12, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r2")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k10_r2"] = _k2_npe_head(2, 10, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r2")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_cosmo_ctl_r2"] = _k2_npe_head(2, 8, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_bg_ctl_r2"] = _k2_npe_head(2, 8, _BG_THETA, _BG_BOXES_K2)
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k16_r3"] = _k2_npe_head(3, 16, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r3")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k12_r3"] = _k2_npe_head(3, 12, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r3")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k10_r3"] = _k2_npe_head(3, 10, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r3")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_cosmo_ctl_r3"] = _k2_npe_head(3, 8, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_bg_ctl_r3"] = _k2_npe_head(3, 8, _BG_THETA, _BG_BOXES_K2)
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k16_r4"] = _k2_npe_head(4, 16, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r4")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k12_r4"] = _k2_npe_head(4, 12, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r4")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k10_r4"] = _k2_npe_head(4, 10, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r4")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_cosmo_ctl_r4"] = _k2_npe_head(4, 8, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_bg_ctl_r4"] = _k2_npe_head(4, 8, _BG_THETA, _BG_BOXES_K2)


# --- Stage A NLE pretrain at k=6 for nla and nla_z ---
_VARIATE_NLE_K6_VDQ = 6


def _variate_nle_a_vdq_k6(data_patterns, r, cosmo_param_names, preset_overrides=None):
    pre = _variate_nle_a_vdq(data_patterns, r, cosmo_param_names, preset_overrides)
    pre["whiten_embeddings"] = {"k": _VARIATE_NLE_K6_VDQ}
    return pre


kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_k6_r0"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA, 0, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_k6_r1"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA, 1, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_k6_r2"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA, 2, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_k6_r3"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA, 3, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_k6_r4"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA, 4, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_z_vdq_z8_k6_r0"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA_Z, 0, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_z_vdq_z8_k6_r1"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA_Z, 1, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_z_vdq_z8_k6_r2"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA_Z, 2, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_z_vdq_z8_k6_r3"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA_Z, 3, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_z_vdq_z8_k6_r4"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA_Z, 4, _COSMO_9_NLAZ, _A_IA_NLA_BOX)

for _r in _NLE_REPEATS:
    VDQ_VARIATE_NLE_SOURCES[f"glass_nle_pretrain_nla_vdq_z8_k6_r{_r}"] = "glass_encoder_finetune_nla_vdq_z8"
    VDQ_VARIATE_NLE_SOURCES[f"glass_nle_pretrain_nla_z_vdq_z8_k6_r{_r}"] = "glass_encoder_finetune_nla_z_vdq_z8"
assert set(VDQ_VARIATE_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)


# --- kappa=2 joint NLE Stage A ---
_K2_JOINT_NLE_K = 10


def _k2_joint_nle_a(r):
    pre = _nle_pretrain_bgp(_VDQ_GLASS_K2, r, cosmo_param_names=list(_K2_JOINT_THETA),
                            preset_overrides=_BG_BOXES_K2)
    pre["whiten_embeddings"] = {"k": _K2_JOINT_NLE_K}
    pre["embedding_cache_name"] = f"vdqk2_joint_r{r}"
    pre["reuse_embedding_cache"] = True
    return pre


kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_joint_k10_r0"] = _k2_joint_nle_a(0)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_joint_k10_r1"] = _k2_joint_nle_a(1)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_joint_k10_r2"] = _k2_joint_nle_a(2)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_joint_k10_r3"] = _k2_joint_nle_a(3)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_joint_k10_r4"] = _k2_joint_nle_a(4)

for _r in _NLE_REPEATS:
    VDQ_VARIATE_NLE_SOURCES[f"glass_nle_pretrain_nla_m_vdqk2_joint_k10_r{_r}"] = _K2_JOINT_SOURCES
assert set(VDQ_VARIATE_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)


# --- nla_z r3 Stage A on the re-rolled encoder ---
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_z_vdq_z8_k6_r3b"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA_Z, 3, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
VDQ_VARIATE_NLE_SOURCES["glass_nle_pretrain_nla_z_vdq_z8_k6_r3b"] = "glass_encoder_finetune_nla_z_vdq_z8"


# --- Gower VDQ variate NLE Stage B ---
_VARIATE_NLE_B_SAMPLING_JOBS = 63


def _variate_nle_b_vdq(stage_a, gower_data, r, cosmo_param_names, preset_overrides):
    b = _nle_finetune(stage_a, ensemble_repeats=9, whiten_k=_VARIATE_NLE_K6_VDQ, warmstart_max_gap_nats=50.0,
                      gower_data=gower_data, gower_eb=None,
                      cosmo_param_names=list(cosmo_param_names),
                      preset_overrides=preset_overrides)
    b["max_trainval_cosmos"] = None
    b["train_frac"] = 0.8
    b["val_frac"] = 0.2
    b["test_frac"] = 0.0
    b["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
    b["epochs"] = 150
    b["project"] = _BGP_NLE_PROJECT
    b["sampling_num_jobs"] = _VARIATE_NLE_B_SAMPLING_JOBS
    return _nle_bake_repeat(b, r)


kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdq_z8_k6_r0_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdq_z8_k6_r0", _VDQ_GOWER_NLA, 0, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdq_z8_k6_r1_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdq_z8_k6_r1", _VDQ_GOWER_NLA, 1, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdq_z8_k6_r2_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdq_z8_k6_r2", _VDQ_GOWER_NLA, 2, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdq_z8_k6_r3_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdq_z8_k6_r3", _VDQ_GOWER_NLA, 3, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdq_z8_k6_r4_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdq_z8_k6_r4", _VDQ_GOWER_NLA, 4, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_z_vdq_z8_k6_r0_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_z_vdq_z8_k6_r0", _VDQ_GOWER_NLA_Z, 0, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_z_vdq_z8_k6_r1_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_z_vdq_z8_k6_r1", _VDQ_GOWER_NLA_Z, 1, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_z_vdq_z8_k6_r2_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_z_vdq_z8_k6_r2", _VDQ_GOWER_NLA_Z, 2, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_z_vdq_z8_k6_r3b_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_z_vdq_z8_k6_r3b", _VDQ_GOWER_NLA_Z, 3, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_z_vdq_z8_k6_r4_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_z_vdq_z8_k6_r4", _VDQ_GOWER_NLA_Z, 4, _COSMO_9_NLAZ, _A_IA_NLA_BOX)

kids_legacy_vdq_experiments["gower_nle_finetune_nla_z_vdq_z8_k6_r3b_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 100.0

kids_legacy_vdq_experiments["gower_nle_finetune_nla_z_vdq_z8_k6_r3b_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 300.0
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdq_z8_k6_r4_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 300.0
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdq_z8_k6_r3_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 3000.0

for _r in _NLE_REPEATS:
    _sfx = "3b" if _r == 3 else str(_r)
    VDQ_NLE_SOURCES[f"gower_nle_finetune_nla_vdq_z8_k6_r{_r}_ens9_e150"] = "glass_encoder_finetune_nla_vdq_z8"
    VDQ_NLE_SOURCES[f"gower_nle_finetune_nla_z_vdq_z8_k6_r{_sfx}_ens9_e150"] = "glass_encoder_finetune_nla_z_vdq_z8"
assert set(VDQ_NLE_SOURCES) <= set(kids_legacy_vdq_experiments), \
    sorted(set(VDQ_NLE_SOURCES) - set(kids_legacy_vdq_experiments))
assert set(VDQ_VARIATE_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)


# --- kappa=2 with NLA IA ---
_VDQ_GLASS_K2_NLA = f"{_GPU5}/glass_vdqk2_nla_f16_sc8a1_{_EB}/output_*.h5"
_K2_NLA_JOINT_SOURCES = "glass_encoder_finetune_nla_vdqk2_z8_cosmo,glass_encoder_finetune_nla_vdqk2_z8_bg"
_K2_NLA_JOINT_THETA = list(_COSMO_8_NLA) + list(_BG_THETA)
_K2_NLA_JOINT_BOXES = {**_A_IA_NLA_BOX, **_BG_BOXES_K2}
assert len(_K2_NLA_JOINT_THETA) == 14, _K2_NLA_JOINT_THETA

kids_legacy_vdq_experiments["glass_encoder_finetune_nla_vdqk2_z8_cosmo"] = _assert_final_summary_dim(
    _encoder_finetune_vdq(_VDQ_GLASS_K2_NLA, _COSMO_8_NLA, preset_overrides=_A_IA_NLA_BOX), 8,
    "glass_encoder_finetune_nla_vdqk2_z8_cosmo")
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_vdqk2_z8_bg"] = _assert_final_summary_dim(
    _encoder_finetune_vdq(_VDQ_GLASS_K2_NLA, _BG_THETA, preset_overrides=_BG_BOXES_K2), 8,
    "glass_encoder_finetune_nla_vdqk2_z8_bg")


def _k2_nla_joint_nle_a(r):
    pre = _nle_pretrain_bgp(_VDQ_GLASS_K2_NLA, r, cosmo_param_names=list(_K2_NLA_JOINT_THETA),
                            preset_overrides=_K2_NLA_JOINT_BOXES)
    pre["whiten_embeddings"] = {"k": _K2_JOINT_NLE_K}
    pre["embedding_cache_name"] = f"vdqk2_nla_joint_r{r}"
    pre["reuse_embedding_cache"] = True
    return pre


kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdqk2_joint_k10_r0"] = _k2_nla_joint_nle_a(0)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdqk2_joint_k10_r1"] = _k2_nla_joint_nle_a(1)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdqk2_joint_k10_r2"] = _k2_nla_joint_nle_a(2)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdqk2_joint_k10_r3"] = _k2_nla_joint_nle_a(3)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdqk2_joint_k10_r4"] = _k2_nla_joint_nle_a(4)

for _r in _NLE_REPEATS:
    VDQ_VARIATE_NLE_SOURCES[f"glass_nle_pretrain_nla_vdqk2_joint_k10_r{_r}"] = _K2_NLA_JOINT_SOURCES
assert set(VDQ_VARIATE_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)


# --- nla r3/r4/r5 encoder rerolls under new names (_rr) ---
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_vdq_z8_rr"] = _assert_final_summary_dim(
    _encoder_finetune_vdq(_VDQ_GLASS_NLA, _COSMO_8_NLA, preset_overrides=_A_IA_NLA_BOX), 8,
    "glass_encoder_finetune_nla_vdq_z8_rr")
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_rr_k6_r3"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA, 3, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_rr_k6_r4"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA, 4, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdq_z8_rr_k6_r3_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdq_z8_rr_k6_r3", _VDQ_GOWER_NLA, 3, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdq_z8_rr_k6_r4_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdq_z8_rr_k6_r4", _VDQ_GOWER_NLA, 4, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_rr_k6_r5"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA, 5, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdq_z8_rr_k6_r5_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdq_z8_rr_k6_r5", _VDQ_GOWER_NLA, 5, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdq_z8_rr_k6_r5_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 150.0
for _r in (3, 4, 5):
    VDQ_VARIATE_NLE_SOURCES[f"glass_nle_pretrain_nla_vdq_z8_rr_k6_r{_r}"] = "glass_encoder_finetune_nla_vdq_z8_rr"
    VDQ_NLE_SOURCES[f"gower_nle_finetune_nla_vdq_z8_rr_k6_r{_r}_ens9_e150"] = "glass_encoder_finetune_nla_vdq_z8_rr"
assert set(VDQ_VARIATE_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)
assert set(VDQ_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)


def _bg_side_input_row(side_input):
    c = _encoder_finetune_vdq(_VDQ_GLASS_K2, _BG_THETA, preset_overrides=_BG_BOXES_K2)
    c["side_input"] = side_input
    return c


kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vdqk2_z8_bg_oracle"] = _assert_final_summary_dim(
    _bg_side_input_row({"type": "true_theta", "params": list(_COSMO_9_NLAM)}), 8,
    "glass_encoder_finetune_nla_m_vdqk2_z8_bg_oracle")
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vdqk2_z8_bg_condcosmo"] = _assert_final_summary_dim(
    _bg_side_input_row({"type": "frozen_encoder",
                        "ckpt_path": f"{_CKPT}/glass_encoder_finetune_nla_m_vdqk2_z8_cosmo/"}), 8,
    "glass_encoder_finetune_nla_m_vdqk2_z8_bg_condcosmo")


from config.kids_legacy_bgp import _BG_BOXES  # noqa: E402  (kappa=1 boxes)
_BG_K1_EPOCHS = 30


def _bg_k1_row(side_input=None):
    c = _encoder_finetune_vdq(_VDQ_GLASS, _BG_THETA, preset_overrides=_BG_BOXES)
    c["epochs"] = _BG_K1_EPOCHS
    if side_input is not None:
        c["side_input"] = side_input
    return c


kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vdq_z8_bg_k1"] = _assert_final_summary_dim(
    _bg_k1_row(), 8, "glass_encoder_finetune_nla_m_vdq_z8_bg_k1")
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vdq_z8_bg_condcosmo_k1"] = _assert_final_summary_dim(
    _bg_k1_row({"type": "frozen_encoder", "ckpt_path": _VDQ_HYB_CKPT}), 8,
    "glass_encoder_finetune_nla_m_vdq_z8_bg_condcosmo_k1")
_c = _bg_side_input_row({"type": "frozen_encoder",
                         "ckpt_path": f"{_CKPT}/glass_encoder_finetune_nla_m_vdqk2_z8_cosmo/"})
_c["pretrained_embedding_ckpt_path"] = f"{_CKPT}/glass_encoder_finetune_nla_m_vdq_z8_bg_condcosmo_k1/"
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vdqk2_z8_bg_condcosmo_fromk1"] = _assert_final_summary_dim(
    _c, 8, "glass_encoder_finetune_nla_m_vdqk2_z8_bg_condcosmo_fromk1")


_VDQ_GLASS_K2_SC8 = f"{_GPU5}/glass_vdqk2_nla_m_f16_sc8_{_EB}/output_*.h5"
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vdqk2_z8_cosmo_sc8"] = _assert_final_summary_dim(
    _encoder_finetune_vdq(_VDQ_GLASS_K2_SC8, _COSMO_9_NLAM, load_flow=True), 8,
    "glass_encoder_finetune_nla_m_vdqk2_z8_cosmo_sc8")
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vdqk2_z8_bg_sc8"] = _assert_final_summary_dim(
    _encoder_finetune_vdq(_VDQ_GLASS_K2_SC8, _BG_THETA, preset_overrides=_BG_BOXES_K2), 8,
    "glass_encoder_finetune_nla_m_vdqk2_z8_bg_sc8")
_c = _encoder_finetune_vdq(_VDQ_GLASS_K2_SC8, _BG_THETA, preset_overrides=_BG_BOXES_K2)
_c["side_input"] = {"type": "frozen_encoder", "ckpt_path": f"{_CKPT}/glass_encoder_finetune_nla_m_vdqk2_z8_cosmo_sc8/"}
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vdqk2_z8_bg_condcosmo_sc8"] = _assert_final_summary_dim(
    _c, 8, "glass_encoder_finetune_nla_m_vdqk2_z8_bg_condcosmo_sc8")


_VDQ_GLASS_SC8 = f"{_GPU5}/glass_vdq_nla_m_f16_sc8_{_EB}/output_*.h5"
_c = _bg_k1_row()
_c["data_patterns"] = [_VDQ_GLASS_SC8] if isinstance(_c["data_patterns"], (list, tuple)) else _VDQ_GLASS_SC8
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vdq_z8_bg_k1_sc8"] = _assert_final_summary_dim(
    _c, 8, "glass_encoder_finetune_nla_m_vdq_z8_bg_k1_sc8")


# --- kappa=2 NLE chains, cosmology-only theta (b_g marginalised) ---
_VDQ_GOWER_K2 = f"{_GPU5}/gower_vdqk2_nla_m_f16_sc8a1_{_EB}/output_*.h5"
_VDQ_GOWER_K2_NLA = f"{_GPU5}/gower_vdqk2_nla_f16_sc8a1_{_EB}/output_*.h5"
_K2_COSMO_ENC = "glass_encoder_finetune_nla_m_vdqk2_z8_cosmo"
_K2_NLA_COSMO_ENC = "glass_encoder_finetune_nla_vdqk2_z8_cosmo"

kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k6_r0"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_K2, 0, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k6_r1"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_K2, 1, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k6_r2"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_K2, 2, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k6_r3"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_K2, 3, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k6_r4"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_K2, 4, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r0"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_K2_NLA, 0, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r1"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_K2_NLA, 1, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r2"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_K2_NLA, 2, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r3"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_K2_NLA, 3, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r4"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_K2_NLA, 4, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdqk2_z8_cosmo_k6_r0_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k6_r0", _VDQ_GOWER_K2, 0, _COSMO_9_NLAM, None)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdqk2_z8_cosmo_k6_r1_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k6_r1", _VDQ_GOWER_K2, 1, _COSMO_9_NLAM, None)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdqk2_z8_cosmo_k6_r2_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k6_r2", _VDQ_GOWER_K2, 2, _COSMO_9_NLAM, None)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdqk2_z8_cosmo_k6_r3_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k6_r3", _VDQ_GOWER_K2, 3, _COSMO_9_NLAM, None)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdqk2_z8_cosmo_k6_r4_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k6_r4", _VDQ_GOWER_K2, 4, _COSMO_9_NLAM, None)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdqk2_z8_cosmo_k6_r0_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r0", _VDQ_GOWER_K2_NLA, 0, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdqk2_z8_cosmo_k6_r1_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r1", _VDQ_GOWER_K2_NLA, 1, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdqk2_z8_cosmo_k6_r2_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r2", _VDQ_GOWER_K2_NLA, 2, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdqk2_z8_cosmo_k6_r3_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r3", _VDQ_GOWER_K2_NLA, 3, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdqk2_z8_cosmo_k6_r4_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r4", _VDQ_GOWER_K2_NLA, 4, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdqk2_z8_cosmo_k6_r3_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 800.0
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r5"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_K2_NLA, 5, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdqk2_z8_cosmo_k6_r5_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r5", _VDQ_GOWER_K2_NLA, 5, _COSMO_8_NLA, _A_IA_NLA_BOX)
VDQ_VARIATE_NLE_SOURCES["glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r5"] = _K2_NLA_COSMO_ENC
VDQ_NLE_SOURCES["gower_nle_finetune_nla_vdqk2_z8_cosmo_k6_r5_ens9_e150"] = _K2_NLA_COSMO_ENC
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdqk2_z8_cosmo_k6_r5_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 150.0

for _r in _NLE_REPEATS:
    VDQ_VARIATE_NLE_SOURCES[f"glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k6_r{_r}"] = _K2_COSMO_ENC
    VDQ_VARIATE_NLE_SOURCES[f"glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r{_r}"] = _K2_NLA_COSMO_ENC
    VDQ_NLE_SOURCES[f"gower_nle_finetune_nla_m_vdqk2_z8_cosmo_k6_r{_r}_ens9_e150"] = _K2_COSMO_ENC
    VDQ_NLE_SOURCES[f"gower_nle_finetune_nla_vdqk2_z8_cosmo_k6_r{_r}_ens9_e150"] = _K2_NLA_COSMO_ENC
assert set(VDQ_VARIATE_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)
assert set(VDQ_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)


# --- 2-pt band NLE chain, NLA variate ---
_BAND_NLA_REPEATS_VDQ = (0, 1, 2)
_BAND_NLA_SCALER = {
    "data": {"type": "standard", "keys": None},
    "cosmo": {"type": "preset", "preset_overrides": dict(_A_IA_NLA_BOX)},
}


def _band_nla_vdq():
    c = _band_bgp(_VDQ_GLASS_NLA)
    c["cosmo_param_names"] = list(_COSMO_8_NLA)
    c["scaler_options"] = {k: dict(v) for k, v in _BAND_NLA_SCALER.items()}
    c["repeat_indices"] = list(_BAND_NLA_REPEATS_VDQ)
    return c


def _band_nla_nle_a_vdq(r):
    pre = _nle_pretrain_bgp(_VDQ_GLASS_NLA, r, cosmo_param_names=list(_COSMO_8_NLA),
                            preset_overrides=_A_IA_NLA_BOX)
    pre["whiten_embeddings"] = {"k": _BAND_NLE_K_VDQ}
    return pre


def _band_nla_nle_b_vdq(r):
    ft = _nle_finetune(f"glass_nle_pretrain_band_nla_vdq_k{_BAND_NLE_K_VDQ}_r{r}", ensemble_repeats=9,
                       whiten_k=_BAND_NLE_K_VDQ, warmstart_max_gap_nats=50.0,
                       gower_data=_VDQ_GOWER_NLA, gower_eb=None,
                       cosmo_param_names=list(_COSMO_8_NLA), preset_overrides=_A_IA_NLA_BOX)
    ft["max_trainval_cosmos"] = None             # the Gower store holds only the 200 fixed ids
    ft["train_frac"] = 0.8
    ft["val_frac"] = 0.2
    ft["test_frac"] = 0.0
    ft["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
    ft["epochs"] = 150
    ft["project"] = _BGP_NLE_PROJECT
    return _nle_bake_repeat(ft, r)


assert _BAND_NLE_K_VDQ == 8, _BAND_NLE_K_VDQ
kids_legacy_vdq_experiments["kids_legacy_band_nla_vdq"] = _band_nla_vdq()
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_vdq_k8_r0"] = _band_nla_nle_a_vdq(0)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_vdq_k8_r1"] = _band_nla_nle_a_vdq(1)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_vdq_k8_r2"] = _band_nla_nle_a_vdq(2)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_vdq_k8_r0_ens9_e150"] = _band_nla_nle_b_vdq(0)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_vdq_k8_r1_ens9_e150"] = _band_nla_nle_b_vdq(1)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_vdq_k8_r2_ens9_e150"] = _band_nla_nle_b_vdq(2)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_vdq_k8_r0_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 150.0
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_vdq_k8_r2_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 150.0
for _r in _BAND_NLA_REPEATS_VDQ:
    VDQ_NLE_SOURCES[f"glass_nle_pretrain_band_nla_vdq_k{_BAND_NLE_K_VDQ}_r{_r}"] = "kids_legacy_band_nla_vdq"
    VDQ_NLE_SOURCES[f"gower_nle_finetune_band_nla_vdq_k{_BAND_NLE_K_VDQ}_r{_r}_ens9_e150"] = "kids_legacy_band_nla_vdq"
assert set(VDQ_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)


# --- 2-pt band NLE chain, NLA-z variate ---
def _band_nla_z_vdq():
    c = _band_bgp(_VDQ_GLASS_NLA_Z)
    c["cosmo_param_names"] = list(_COSMO_9_NLAZ)
    c["scaler_options"] = {k: dict(v) for k, v in _BAND_NLA_SCALER.items()}
    c["repeat_indices"] = list(_BAND_NLA_REPEATS_VDQ)
    return c


def _band_nla_z_nle_a_vdq(r):
    pre = _nle_pretrain_bgp(_VDQ_GLASS_NLA_Z, r, cosmo_param_names=list(_COSMO_9_NLAZ),
                            preset_overrides=_A_IA_NLA_BOX)
    pre["whiten_embeddings"] = {"k": _BAND_NLE_K_VDQ}
    return pre


def _band_nla_z_nle_b_vdq(r, pretrain_exp=None):
    pretrain_exp = pretrain_exp or f"glass_nle_pretrain_band_nla_z_vdq_k{_BAND_NLE_K_VDQ}_r{r}"
    ft = _nle_finetune(pretrain_exp, ensemble_repeats=9,
                       whiten_k=_BAND_NLE_K_VDQ, warmstart_max_gap_nats=50.0,
                       gower_data=_VDQ_GOWER_NLA_Z, gower_eb=None,
                       cosmo_param_names=list(_COSMO_9_NLAZ), preset_overrides=_A_IA_NLA_BOX)
    ft["max_trainval_cosmos"] = None             # the Gower store holds only the 200 fixed ids
    ft["train_frac"] = 0.8
    ft["val_frac"] = 0.2
    ft["test_frac"] = 0.0
    ft["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
    ft["epochs"] = 150
    ft["project"] = _BGP_NLE_PROJECT
    return _nle_bake_repeat(ft, r)


kids_legacy_vdq_experiments["kids_legacy_band_nla_z_vdq"] = _band_nla_z_vdq()
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_k8_r0"] = _band_nla_z_nle_a_vdq(0)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_k8_r1"] = _band_nla_z_nle_a_vdq(1)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_k8_r2"] = _band_nla_z_nle_a_vdq(2)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r0_ens9_e150"] = _band_nla_z_nle_b_vdq(0)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r1_ens9_e150"] = _band_nla_z_nle_b_vdq(1)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_ens9_e150"] = _band_nla_z_nle_b_vdq(2)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r0_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 150.0
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r1_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 150.0
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 150.0
for _r in _BAND_NLA_REPEATS_VDQ:
    VDQ_NLE_SOURCES[f"glass_nle_pretrain_band_nla_z_vdq_k{_BAND_NLE_K_VDQ}_r{_r}"] = "kids_legacy_band_nla_z_vdq"
    VDQ_NLE_SOURCES[f"gower_nle_finetune_band_nla_z_vdq_k{_BAND_NLE_K_VDQ}_r{_r}_ens9_e150"] = "kids_legacy_band_nla_z_vdq"
assert set(VDQ_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)


# --- band_nla_z 5-repeat retrain ---
def _band_nla_z_nle_b_e400(r, pretrain_exp=None):
    ft = _band_nla_z_nle_b_vdq(r, pretrain_exp)
    ft.update(lr=4e-4, epochs=400)
    return ft


_band_nla_z_rr = _band_nla_z_vdq()
_band_nla_z_rr["repeat_indices"] = [1]
kids_legacy_vdq_experiments["kids_legacy_band_nla_z_vdq_rr"] = _band_nla_z_rr
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_k8_r3"] = _band_nla_z_nle_a_vdq(3)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_k8_r4"] = _band_nla_z_nle_a_vdq(4)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_rr_k8_r1"] = _band_nla_z_nle_a_vdq(1)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_k8_r2b"] = _band_nla_z_nle_a_vdq(2)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r3_ens9_e400"] = _band_nla_z_nle_b_e400(3)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r4_ens9_e400"] = _band_nla_z_nle_b_e400(4)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_rr_k8_r1_ens9_e400"] = _band_nla_z_nle_b_e400(
    1, "glass_nle_pretrain_band_nla_z_vdq_rr_k8_r1")
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2b_ens9_e400"] = _band_nla_z_nle_b_e400(
    2, "glass_nle_pretrain_band_nla_z_vdq_k8_r2b")
for _n in ("glass_nle_pretrain_band_nla_z_vdq_k8_r3", "glass_nle_pretrain_band_nla_z_vdq_k8_r4",
           "glass_nle_pretrain_band_nla_z_vdq_k8_r2b", "gower_nle_finetune_band_nla_z_vdq_k8_r3_ens9_e400",
           "gower_nle_finetune_band_nla_z_vdq_k8_r4_ens9_e400", "gower_nle_finetune_band_nla_z_vdq_k8_r2b_ens9_e400"):
    VDQ_NLE_SOURCES[_n] = "kids_legacy_band_nla_z_vdq"
for _n in ("glass_nle_pretrain_band_nla_z_vdq_rr_k8_r1", "gower_nle_finetune_band_nla_z_vdq_rr_k8_r1_ens9_e400"):
    VDQ_NLE_SOURCES[_n] = "kids_legacy_band_nla_z_vdq_rr"
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2b_ens9_e400"]["whiten_warmstart_max_gap_nats"] = 150.0
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r4_ens9_e400"]["whiten_warmstart_max_gap_nats"] = 150.0
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_rr_k8_r1_ens9_e400"]["whiten_warmstart_max_gap_nats"] = 150.0
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r3_ens9_e400"]["whiten_warmstart_max_gap_nats"] = 150.0


def _band_nla_z_lr_sweep_b(lr, epochs):
    ft = _band_nla_z_nle_b_vdq(2)
    ft.update(ensemble_repeats=3, run_evaluation=False, lr=lr, epochs=epochs, whiten_warmstart_max_gap_nats=150.0)
    return ft


kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_lrs_8e5_e150"] = _band_nla_z_lr_sweep_b(8e-5, 150)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_lrs_8e5_e400"] = _band_nla_z_lr_sweep_b(8e-5, 400)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_lrs_3e5_e150"] = _band_nla_z_lr_sweep_b(3e-5, 150)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_lrs_3e5_e400"] = _band_nla_z_lr_sweep_b(3e-5, 400)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_lrs_2e4_e150"] = _band_nla_z_lr_sweep_b(2e-4, 150)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_lrs_2e4_e400"] = _band_nla_z_lr_sweep_b(2e-4, 400)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_lrs_4e4_e150"] = _band_nla_z_lr_sweep_b(4e-4, 150)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_lrs_4e4_e400"] = _band_nla_z_lr_sweep_b(4e-4, 400)
for _tag in ("8e5_e150", "8e5_e400", "3e5_e150", "3e5_e400", "2e4_e150", "2e4_e400", "4e4_e150", "4e4_e400"):
    VDQ_NLE_SOURCES[f"gower_nle_finetune_band_nla_z_vdq_k8_r2_lrs_{_tag}"] = "kids_legacy_band_nla_z_vdq"


def _band_nla_z_sweep_a(k, epochs=None):
    pre = _band_nla_z_nle_a_vdq(2)
    pre["whiten_embeddings"] = {"k": k}
    if epochs:
        pre["epochs"] = epochs
    return pre


def _band_nla_z_sweep_b(pretrain_exp, k):
    ft = _band_nla_z_nle_b_vdq(2, pretrain_exp)
    ft["whiten_embeddings"] = {"k": k}
    ft.update(ensemble_repeats=3, whiten_warmstart_max_gap_nats=150.0)
    return ft


kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_k6_r2_sw"] = _band_nla_z_sweep_a(6)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_k4_r2_sw"] = _band_nla_z_sweep_a(4)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_k8_r2_sw_a400"] = _band_nla_z_sweep_a(8, epochs=400)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_sw_base_ens3"] = _band_nla_z_sweep_b(
    "glass_nle_pretrain_band_nla_z_vdq_k8_r2", 8)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k6_r2_sw_ens3"] = _band_nla_z_sweep_b(
    "glass_nle_pretrain_band_nla_z_vdq_k6_r2_sw", 6)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k4_r2_sw_ens3"] = _band_nla_z_sweep_b(
    "glass_nle_pretrain_band_nla_z_vdq_k4_r2_sw", 4)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_sw_a400_ens3"] = _band_nla_z_sweep_b(
    "glass_nle_pretrain_band_nla_z_vdq_k8_r2_sw_a400", 8)
_ft =_band_nla_z_sweep_b("glass_nle_pretrain_band_nla_z_vdq_k8_r2", 8)
_ft["lr"] = 2e-4
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_sw_lr2e4_ens3"] = _ft
VDQ_NLE_SOURCES["gower_nle_finetune_band_nla_z_vdq_k8_r2_sw_lr2e4_ens3"] = "kids_legacy_band_nla_z_vdq"
_ft =_band_nla_z_sweep_b("glass_nle_pretrain_band_nla_z_vdq_k8_r2", 8)
_ft["lr"] = 4e-4
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_sw_lr4e4_ens3"] = _ft
VDQ_NLE_SOURCES["gower_nle_finetune_band_nla_z_vdq_k8_r2_sw_lr4e4_ens3"] = "kids_legacy_band_nla_z_vdq"
for _n in ("glass_nle_pretrain_band_nla_z_vdq_k6_r2_sw", "glass_nle_pretrain_band_nla_z_vdq_k4_r2_sw",
           "glass_nle_pretrain_band_nla_z_vdq_k8_r2_sw_a400", "gower_nle_finetune_band_nla_z_vdq_k8_r2_sw_base_ens3",
           "gower_nle_finetune_band_nla_z_vdq_k6_r2_sw_ens3", "gower_nle_finetune_band_nla_z_vdq_k4_r2_sw_ens3",
           "gower_nle_finetune_band_nla_z_vdq_k8_r2_sw_a400_ens3"):
    VDQ_NLE_SOURCES[_n] = "kids_legacy_band_nla_z_vdq"

# --- band_nla_z at whitening k=6 ---
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_k6_r2_sw2"] = _band_nla_z_sweep_a(6)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k6_r2_sw2_ens3"] = _band_nla_z_sweep_b(
    "glass_nle_pretrain_band_nla_z_vdq_k6_r2_sw2", 6)


def _band_nla_z_k6_a(r):
    pre = _band_nla_z_nle_a_vdq(r)
    pre["whiten_embeddings"] = {"k": 6}
    return pre


def _band_nla_z_k6_b(r, pretrain_exp):
    ft = _band_nla_z_nle_b_vdq(r, pretrain_exp)
    ft["whiten_embeddings"] = {"k": 6}
    return ft


kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_k6_r0"] = _band_nla_z_k6_a(0)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_k6_r3"] = _band_nla_z_k6_a(3)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_k6_r4"] = _band_nla_z_k6_a(4)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_rr_k6_r1"] = _band_nla_z_k6_a(1)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k6_r0_ens9_e150"] = _band_nla_z_k6_b(
    0, "glass_nle_pretrain_band_nla_z_vdq_k6_r0")
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k6_r3_ens9_e150"] = _band_nla_z_k6_b(
    3, "glass_nle_pretrain_band_nla_z_vdq_k6_r3")
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k6_r4_ens9_e150"] = _band_nla_z_k6_b(
    4, "glass_nle_pretrain_band_nla_z_vdq_k6_r4")
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_rr_k6_r1_ens9_e150"] = _band_nla_z_k6_b(
    1, "glass_nle_pretrain_band_nla_z_vdq_rr_k6_r1")
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k6_r2_ens9_e150"] = _band_nla_z_k6_b(
    2, "glass_nle_pretrain_band_nla_z_vdq_k6_r2_sw")
for _n in ("glass_nle_pretrain_band_nla_z_vdq_k6_r2_sw2", "gower_nle_finetune_band_nla_z_vdq_k6_r2_sw2_ens3",
           "glass_nle_pretrain_band_nla_z_vdq_k6_r0", "glass_nle_pretrain_band_nla_z_vdq_k6_r3",
           "glass_nle_pretrain_band_nla_z_vdq_k6_r4", "gower_nle_finetune_band_nla_z_vdq_k6_r0_ens9_e150",
           "gower_nle_finetune_band_nla_z_vdq_k6_r3_ens9_e150", "gower_nle_finetune_band_nla_z_vdq_k6_r4_ens9_e150",
           "gower_nle_finetune_band_nla_z_vdq_k6_r2_ens9_e150"):
    VDQ_NLE_SOURCES[_n] = "kids_legacy_band_nla_z_vdq"
for _n in ("glass_nle_pretrain_band_nla_z_vdq_rr_k6_r1", "gower_nle_finetune_band_nla_z_vdq_rr_k6_r1_ens9_e150"):
    VDQ_NLE_SOURCES[_n] = "kids_legacy_band_nla_z_vdq_rr"
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_rr_k6_r1_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 150.0
for _r in (2, 3):
    kids_legacy_vdq_experiments[f"glass_nle_pretrain_band_nla_z_vdq_rr_k6_r{_r}"] = _band_nla_z_k6_a(_r)
    kids_legacy_vdq_experiments[f"gower_nle_finetune_band_nla_z_vdq_rr_k6_r{_r}_ens9_e150"] = _band_nla_z_k6_b(
        _r, f"glass_nle_pretrain_band_nla_z_vdq_rr_k6_r{_r}")
    for _n in (f"glass_nle_pretrain_band_nla_z_vdq_rr_k6_r{_r}", f"gower_nle_finetune_band_nla_z_vdq_rr_k6_r{_r}_ens9_e150"):
        VDQ_NLE_SOURCES[_n] = "kids_legacy_band_nla_z_vdq_rr"
# Spare bands get their own ckpt dir: same-index runs must not share a dir.
_band_nla_z_rr2 = _band_nla_z_vdq()
_band_nla_z_rr2["repeat_indices"] = [1, 3, 4]
kids_legacy_vdq_experiments["kids_legacy_band_nla_z_vdq_rr2"] = _band_nla_z_rr2
for _tag, _idx in (("rr3", [1, 3, 4]), ("rr4", [3, 4])):
    _row = _band_nla_z_vdq()
    _row["repeat_indices"] = _idx
    kids_legacy_vdq_experiments[f"kids_legacy_band_nla_z_vdq_{_tag}"] = _row
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_rr4_k6_r3"] = _band_nla_z_k6_a(3)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_rr4_k6_r3_ens9_e150"] = _band_nla_z_k6_b(
    3, "glass_nle_pretrain_band_nla_z_vdq_rr4_k6_r3")
for _n in ("glass_nle_pretrain_band_nla_z_vdq_rr4_k6_r3", "gower_nle_finetune_band_nla_z_vdq_rr4_k6_r3_ens9_e150"):
    VDQ_NLE_SOURCES[_n] = "kids_legacy_band_nla_z_vdq_rr4"
assert set(VDQ_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)


# --- VD v2 (vd2pr) NLA-M kappa=1 chains ---
_VD2PR_GLASS = f"{_GPU5}/glass_vd2pr_nla_m_f16_sc8a1_{_EB}/output_*.h5"
_VD2PR_GOWER = f"{_GPU5}/gower_vd2pr_nla_m_f16_sc8a1_{_EB}/output_*.h5"
_VD2PR_ENC = "glass_encoder_finetune_nla_m_vd2pr_z8"
_VD2PR_BAND = "kids_legacy_band_nla_m_vdq"
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vd2pr_z8"] = _assert_final_summary_dim(
    _encoder_finetune_vdq(_VD2PR_GLASS, _COSMO_9_NLAM, load_flow=True), 8, "glass_encoder_finetune_nla_m_vd2pr_z8")
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vd2pr_z8_k6_r0"] = _variate_nle_a_vdq_k6(_VD2PR_GLASS, 0, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vd2pr_z8_k6_r1"] = _variate_nle_a_vdq_k6(_VD2PR_GLASS, 1, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vd2pr_z8_k6_r2"] = _variate_nle_a_vdq_k6(_VD2PR_GLASS, 2, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vd2pr_z8_k6_r3"] = _variate_nle_a_vdq_k6(_VD2PR_GLASS, 3, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vd2pr_z8_k6_r4"] = _variate_nle_a_vdq_k6(_VD2PR_GLASS, 4, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vd2pr_z8_k6_r0_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_m_vd2pr_z8_k6_r0", _VD2PR_GOWER, 0, _COSMO_9_NLAM, None)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vd2pr_z8_k6_r1_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_m_vd2pr_z8_k6_r1", _VD2PR_GOWER, 1, _COSMO_9_NLAM, None)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vd2pr_z8_k6_r2_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_m_vd2pr_z8_k6_r2", _VD2PR_GOWER, 2, _COSMO_9_NLAM, None)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vd2pr_z8_k6_r3_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_m_vd2pr_z8_k6_r3", _VD2PR_GOWER, 3, _COSMO_9_NLAM, None)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vd2pr_z8_k6_r4_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_m_vd2pr_z8_k6_r4", _VD2PR_GOWER, 4, _COSMO_9_NLAM, None)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_m_vd2pr_k6_r0"] = _variate_nle_a_vdq_k6(_VD2PR_GLASS, 0, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_m_vd2pr_k6_r1"] = _variate_nle_a_vdq_k6(_VD2PR_GLASS, 1, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_m_vd2pr_k6_r2"] = _variate_nle_a_vdq_k6(_VD2PR_GLASS, 2, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_m_vd2pr_k6_r3"] = _variate_nle_a_vdq_k6(_VD2PR_GLASS, 3, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_m_vd2pr_k6_r4"] = _variate_nle_a_vdq_k6(_VD2PR_GLASS, 4, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_m_vd2pr_k6_r0_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_band_nla_m_vd2pr_k6_r0", _VD2PR_GOWER, 0, _COSMO_9_NLAM, None)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_m_vd2pr_k6_r1_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_band_nla_m_vd2pr_k6_r1", _VD2PR_GOWER, 1, _COSMO_9_NLAM, None)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_m_vd2pr_k6_r2_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_band_nla_m_vd2pr_k6_r2", _VD2PR_GOWER, 2, _COSMO_9_NLAM, None)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_m_vd2pr_k6_r3_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_band_nla_m_vd2pr_k6_r3", _VD2PR_GOWER, 3, _COSMO_9_NLAM, None)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_m_vd2pr_k6_r4_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_band_nla_m_vd2pr_k6_r4", _VD2PR_GOWER, 4, _COSMO_9_NLAM, None)
for _r in _NLE_REPEATS:
    VDQ_VARIATE_NLE_SOURCES[f"glass_nle_pretrain_nla_m_vd2pr_z8_k6_r{_r}"] = _VD2PR_ENC
    VDQ_NLE_SOURCES[f"gower_nle_finetune_nla_m_vd2pr_z8_k6_r{_r}_ens9_e150"] = _VD2PR_ENC
    VDQ_NLE_SOURCES[f"glass_nle_pretrain_band_nla_m_vd2pr_k6_r{_r}"] = _VD2PR_BAND
    VDQ_NLE_SOURCES[f"gower_nle_finetune_band_nla_m_vd2pr_k6_r{_r}_ens9_e150"] = _VD2PR_BAND
assert set(VDQ_VARIATE_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)
assert set(VDQ_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)

# --- VD2PR preview Stage B r0 on the partial Gower store (never pool with the production rows) ---
_VD2PR_GOWER_P10K = f"{_GPU5}/gower_vd2pr_nla_m_p10k_f16_sc8a1_{_EB}/output_*.h5"
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vd2pr_p10k_z8_k6_r0_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_m_vd2pr_z8_k6_r0", _VD2PR_GOWER_P10K, 0, _COSMO_9_NLAM, None)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_m_vd2pr_p10k_k6_r0_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_band_nla_m_vd2pr_k6_r0", _VD2PR_GOWER_P10K, 0, _COSMO_9_NLAM, None)
VDQ_NLE_SOURCES["gower_nle_finetune_nla_m_vd2pr_p10k_z8_k6_r0_ens9_e150"] = _VD2PR_ENC
VDQ_NLE_SOURCES["gower_nle_finetune_band_nla_m_vd2pr_p10k_k6_r0_ens9_e150"] = _VD2PR_BAND
assert set(VDQ_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)


# --- vdqctl: matched quadcap control of the vd2pr rows ---
def _vdqctl_gower_regex():
    import json as _json
    import os as _os
    _lock = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "fixed_test_sets", "gower_test_ids.json")
    with open(_lock) as _f:
        _ids = sorted(int(_i) for _i in _json.load(_f)["sim_ids"])
    assert len(_ids) == 200, len(_ids)
    return r"^output_(" + "|".join(str(_i) for _i in _ids) + r")_"


# include_files_regex (config/default.py) keeps only matching basenames before the cosmology split.
_VDQCTL_GLASS_RX = r"^output_([0-9]|[1-9][0-9]|[1-9][0-9][0-9]|[1-3][0-9][0-9][0-9])_out0_"   # sim_id < 4000, outer 0
_VDQCTL_GOWER_RX = _vdqctl_gower_regex()                                                     # the 200 fixed ids


def _vdqctl(row, rx):
    row["include_files_regex"] = rx
    return row


_VDQCTL_ENC = "glass_encoder_finetune_nla_m_vdqctl_z8"
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vdqctl_z8"] = _assert_final_summary_dim(
    _vdqctl(_encoder_finetune_vdq(_VDQ_GLASS, _COSMO_9_NLAM, load_flow=True), _VDQCTL_GLASS_RX), 8, "glass_encoder_finetune_nla_m_vdqctl_z8")
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqctl_z8_k6_r0"] = _vdqctl(_variate_nle_a_vdq_k6(_VDQ_GLASS, 0, _COSMO_9_NLAM), _VDQCTL_GLASS_RX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqctl_z8_k6_r1"] = _vdqctl(_variate_nle_a_vdq_k6(_VDQ_GLASS, 1, _COSMO_9_NLAM), _VDQCTL_GLASS_RX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqctl_z8_k6_r2"] = _vdqctl(_variate_nle_a_vdq_k6(_VDQ_GLASS, 2, _COSMO_9_NLAM), _VDQCTL_GLASS_RX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqctl_z8_k6_r3"] = _vdqctl(_variate_nle_a_vdq_k6(_VDQ_GLASS, 3, _COSMO_9_NLAM), _VDQCTL_GLASS_RX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqctl_z8_k6_r4"] = _vdqctl(_variate_nle_a_vdq_k6(_VDQ_GLASS, 4, _COSMO_9_NLAM), _VDQCTL_GLASS_RX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdqctl_z8_k6_r0_ens9_e150"] = _vdqctl(_variate_nle_b_vdq("glass_nle_pretrain_nla_m_vdqctl_z8_k6_r0", _VDQ_GOWER, 0, _COSMO_9_NLAM, None), _VDQCTL_GOWER_RX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdqctl_z8_k6_r1_ens9_e150"] = _vdqctl(_variate_nle_b_vdq("glass_nle_pretrain_nla_m_vdqctl_z8_k6_r1", _VDQ_GOWER, 1, _COSMO_9_NLAM, None), _VDQCTL_GOWER_RX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdqctl_z8_k6_r2_ens9_e150"] = _vdqctl(_variate_nle_b_vdq("glass_nle_pretrain_nla_m_vdqctl_z8_k6_r2", _VDQ_GOWER, 2, _COSMO_9_NLAM, None), _VDQCTL_GOWER_RX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdqctl_z8_k6_r3_ens9_e150"] = _vdqctl(_variate_nle_b_vdq("glass_nle_pretrain_nla_m_vdqctl_z8_k6_r3", _VDQ_GOWER, 3, _COSMO_9_NLAM, None), _VDQCTL_GOWER_RX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdqctl_z8_k6_r4_ens9_e150"] = _vdqctl(_variate_nle_b_vdq("glass_nle_pretrain_nla_m_vdqctl_z8_k6_r4", _VDQ_GOWER, 4, _COSMO_9_NLAM, None), _VDQCTL_GOWER_RX)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_m_vdqctl_k6_r0"] = _vdqctl(_variate_nle_a_vdq_k6(_VDQ_GLASS, 0, _COSMO_9_NLAM), _VDQCTL_GLASS_RX)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_m_vdqctl_k6_r1"] = _vdqctl(_variate_nle_a_vdq_k6(_VDQ_GLASS, 1, _COSMO_9_NLAM), _VDQCTL_GLASS_RX)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_m_vdqctl_k6_r2"] = _vdqctl(_variate_nle_a_vdq_k6(_VDQ_GLASS, 2, _COSMO_9_NLAM), _VDQCTL_GLASS_RX)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_m_vdqctl_k6_r3"] = _vdqctl(_variate_nle_a_vdq_k6(_VDQ_GLASS, 3, _COSMO_9_NLAM), _VDQCTL_GLASS_RX)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_m_vdqctl_k6_r4"] = _vdqctl(_variate_nle_a_vdq_k6(_VDQ_GLASS, 4, _COSMO_9_NLAM), _VDQCTL_GLASS_RX)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_m_vdqctl_k6_r0_ens9_e150"] = _vdqctl(_variate_nle_b_vdq("glass_nle_pretrain_band_nla_m_vdqctl_k6_r0", _VDQ_GOWER, 0, _COSMO_9_NLAM, None), _VDQCTL_GOWER_RX)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_m_vdqctl_k6_r1_ens9_e150"] = _vdqctl(_variate_nle_b_vdq("glass_nle_pretrain_band_nla_m_vdqctl_k6_r1", _VDQ_GOWER, 1, _COSMO_9_NLAM, None), _VDQCTL_GOWER_RX)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_m_vdqctl_k6_r2_ens9_e150"] = _vdqctl(_variate_nle_b_vdq("glass_nle_pretrain_band_nla_m_vdqctl_k6_r2", _VDQ_GOWER, 2, _COSMO_9_NLAM, None), _VDQCTL_GOWER_RX)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_m_vdqctl_k6_r3_ens9_e150"] = _vdqctl(_variate_nle_b_vdq("glass_nle_pretrain_band_nla_m_vdqctl_k6_r3", _VDQ_GOWER, 3, _COSMO_9_NLAM, None), _VDQCTL_GOWER_RX)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_m_vdqctl_k6_r4_ens9_e150"] = _vdqctl(_variate_nle_b_vdq("glass_nle_pretrain_band_nla_m_vdqctl_k6_r4", _VDQ_GOWER, 4, _COSMO_9_NLAM, None), _VDQCTL_GOWER_RX)
for _r in _NLE_REPEATS:
    VDQ_VARIATE_NLE_SOURCES[f"glass_nle_pretrain_nla_m_vdqctl_z8_k6_r{_r}"] = _VDQCTL_ENC
    VDQ_NLE_SOURCES[f"gower_nle_finetune_nla_m_vdqctl_z8_k6_r{_r}_ens9_e150"] = _VDQCTL_ENC
    VDQ_NLE_SOURCES[f"glass_nle_pretrain_band_nla_m_vdqctl_k6_r{_r}"] = _VD2PR_BAND
    VDQ_NLE_SOURCES[f"gower_nle_finetune_band_nla_m_vdqctl_k6_r{_r}_ens9_e150"] = _VD2PR_BAND
assert set(VDQ_VARIATE_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)
assert set(VDQ_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)
