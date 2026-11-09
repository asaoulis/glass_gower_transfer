"""Experiment rows for the no-variable-depth (novd) GLASS and Gower suite.

Stage I trains the bandpower MLP. Stage II freezes it inside the ResNet map hybrid (the foundation).
"""
from config.kids_legacy import (
    _COSMO_9, _COSMO_8_NLA, _COSMO_9_NLAZ, _A_IA_NLA_BOX,
    _NLA_M_DATA as _SMOKE_DATA,
    _EB_VARIANT as _SMOKE_EB_VARIANT,
    _band_lmin50, _hybrid_lmin50_z8, _hybrid_lmin50_z8_smoke,
    _encoder_finetune_z8, _encoder_finetune_z8_smoke,
    _nle_pretrain, _nle_finetune, _nle_bake_repeat, _npe_finetune_z8,
)

# --- no-VD data stores ---
_GPU5 = "/share/gpu5/asaoulis/transfer_datasets"
_GPU4 = "/share/gpu4/asaoulis/transfer_datasets"
_CKPT = "/share/gpu5/asaoulis/transfer_models/checkpoints"
_EB_VARIANT = "fwhm4_lmin56_lcut1400"

_NLA_M_RAW = f"{_GPU4}/glass_mocks_nla_m_novd_counts/output_*.h5"
_NLA_M     = f"{_GPU5}/glass_mocks_nla_m_novd_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5"
_NLA       = f"{_GPU5}/glass_mocks_nla_novd_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5"
_NLA_Z     = f"{_GPU5}/glass_mocks_nla_z_novd_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5"

_GOWER_NLA_M_RAW = f"{_GPU4}/gower_mocks_nla_m_novd_counts/output_*.h5"
_GOWER_NLA_M = f"{_GPU5}/gower_mocks_nla_m_novd_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5"
_GOWER_NLA   = f"{_GPU5}/gower_mocks_nla_novd_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5"
_GOWER_NLA_Z = f"{_GPU5}/gower_mocks_nla_z_novd_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5"

_BAND_CKPT_DIR = f"{_CKPT}/kids_legacy_band_nla_m_novd/"
_FOUNDATION_CKPT = f"{_CKPT}/kids_legacy_hybrid_nla_m_novd_z8_resnet/"

# Production map-encoder kwargs (PreActResNet backbone).
# Every config that rebuilds the foundation architecture must use them.
_RESNET_MAPKW = {"encoder_type": "preact_resnet", "patch_conditioning": None,
                 "pool_types": ("avg", "gem"),
                 "stage_channels": (32, 64, 128, 256, 256), "blocks_per_stage": 3}

_GOWER_TEST_IDS = "config/fixed_test_sets/gower_test_ids.json"
# sub-variate split: first 100 of the 200 fixed-test ids = train/val pool, last 100 = test, in file order
_GOWER_TEST_IDS_LAST100 = "config/fixed_test_sets/gower_test_ids_last100.json"

kids_legacy_novd_experiments = {}


# --- Stage I band and Stage II foundation ---
def _band_novd():
    """Stage-I bandpower MLP config.

    Reads mixed_bandpowers from the raw store (bandpowers do not depend on map smoothing)."""
    c = _band_lmin50()
    c["data_patterns"] = _NLA_M_RAW
    c.pop("repeats", None)
    c["repeat_indices"] = [0, 1, 2, 3, 4]
    return c


kids_legacy_novd_experiments["kids_legacy_band_nla_m_novd"] = _band_novd()


def _hybrid_novd_z8():
    """z8 hybrid on the novd nla_m fwhm4 store, repeat i loads frozen band repeat i."""
    c = _hybrid_lmin50_z8()
    c["data_patterns"] = _NLA_M
    c["eb_map_variant"] = _EB_VARIANT
    c["pretrained_band_ckpt_path"] = _BAND_CKPT_DIR
    c.pop("repeats", None)
    c["repeat_indices"] = [0, 1, 2, 3, 4]
    return c


def _hybrid_novd_z8_variant(mk_extra=None, top_extra=None, repeat_indices=(0, 1, 2, 3, 4)):
    c = _hybrid_novd_z8()
    c["repeat_indices"] = list(repeat_indices)
    if mk_extra:
        c["model_kwargs"] = {**c["model_kwargs"], **mk_extra}
    for k, v in (top_extra or {}).items():
        c[k] = v
    return c


def _hybrid_novd_z8_variant_smoke(mk_extra=None, top_extra=None):
    """Local smoke clone of the z8 variant: same model kwargs, from-scratch band, fwhm8 fixture store."""
    c = _hybrid_lmin50_z8_smoke()
    if mk_extra:
        c["model_kwargs"] = {**c["model_kwargs"], **mk_extra}
    for k, v in (top_extra or {}).items():
        c[k] = v
    return c


kids_legacy_novd_experiments["kids_legacy_hybrid_nla_m_novd_z8"] = _hybrid_novd_z8()
kids_legacy_novd_experiments["kids_legacy_hybrid_nla_m_novd_z8_smoke"] = _hybrid_lmin50_z8_smoke()

# --- production foundation: plain PreActResNet map encoder ---
kids_legacy_novd_experiments["kids_legacy_hybrid_nla_m_novd_z8_resnet"] = _hybrid_novd_z8_variant(
    mk_extra={"map_kwargs": _RESNET_MAPKW})


kids_legacy_novd_experiments["kids_legacy_hybrid_nla_m_novd_z8_resnet_smoke"] = \
    _hybrid_novd_z8_variant_smoke(mk_extra={"map_kwargs": _RESNET_MAPKW})

# rescue-only variant: asymmetric spectral decoupling on the band-only readout
kids_legacy_novd_experiments["kids_legacy_hybrid_nla_m_novd_z8_resnet_sdband"] = \
    _hybrid_novd_z8_variant(mk_extra={"map_kwargs": _RESNET_MAPKW, "sd_band_coeff": 1e-2},
                            repeat_indices=(0,))
kids_legacy_novd_experiments["kids_legacy_hybrid_nla_m_novd_z8_resnet_sdband_smoke"] = \
    _hybrid_novd_z8_variant_smoke(mk_extra={"map_kwargs": _RESNET_MAPKW, "sd_band_coeff": 1e-2})


# --- Gower NPE ensemble finetune ---
_gower_npe_ft_novd = _npe_finetune_z8(_FOUNDATION_CKPT, data_patterns=_GOWER_NLA_M,
                                      eb_variant=_EB_VARIANT)
_gower_npe_ft_novd["model_kwargs"] = {**_gower_npe_ft_novd["model_kwargs"], "map_kwargs": _RESNET_MAPKW}
kids_legacy_novd_experiments["gower_npe_finetune_nla_m_novd_z8"] = _gower_npe_ft_novd


def _register_early_npe_ens5_preview_r4():
    c = _npe_finetune_z8(_FOUNDATION_CKPT, data_patterns=_GOWER_NLA_M, eb_variant=_EB_VARIANT)
    c["model_kwargs"] = {**c["model_kwargs"], "map_kwargs": _RESNET_MAPKW}
    c["ensemble_repeats"] = 5
    c.pop("max_trainval_cosmos", None)
    c["repeat_indices"] = [4]
    kids_legacy_novd_experiments["gower_npe_finetune_nla_m_novd_z8_r4_ens5_early"] = c


_register_early_npe_ens5_preview_r4()


def _register_early_npe_ens5_preview_r4_raw():
    c = _npe_finetune_z8(_FOUNDATION_CKPT, data_patterns=_GOWER_NLA_M_RAW, eb_variant=_EB_VARIANT)
    c["model_kwargs"] = {**c["model_kwargs"], "map_kwargs": _RESNET_MAPKW}
    c["ensemble_repeats"] = 5
    c["max_trainval_cosmos"] = [175]
    c["repeat_indices"] = [4]
    kids_legacy_novd_experiments["gower_npe_finetune_nla_m_novd_z8_r4_ens5_early_raw"] = c


_register_early_npe_ens5_preview_r4_raw()


def _npe_finetune_novd_z8_smoke():
    """Local smoke of gower_npe_finetune_nla_m_novd_z8: from-scratch encoder, fwhm8 fixture store."""
    c = _npe_finetune_z8(_FOUNDATION_CKPT, data_patterns=_SMOKE_DATA, eb_variant=_SMOKE_EB_VARIANT)
    c["model_kwargs"] = {**c["model_kwargs"], "map_kwargs": _RESNET_MAPKW}
    c["checkpoint_path"] = None
    c.pop("pretrained_band_ckpt_path", None)
    c["freeze_band"] = False
    c["ensemble_repeats"] = 1
    c.pop("max_trainval_cosmos", None)
    c.pop("fixed_test_sim_ids", None)
    c["train_frac"] = 0.8
    c["val_frac"] = 0.1
    c["test_frac"] = 0.1
    c["repeat_indices"] = [0]
    c["epochs"] = 3
    c["num_workers"] = 2
    c["prefetch_factor"] = 2
    c["persistent_workers"] = False
    return c


kids_legacy_novd_experiments["gower_npe_finetune_nla_m_novd_z8_smoke"] = _npe_finetune_novd_z8_smoke()


# --- NLE chain: 5 repeats, GLASS pretrain then Gower ens9 finetune (z8, pure-whiten k=8) ---
def _register_main_nle_novd_z8():
    for r in range(5):
        kids_legacy_novd_experiments[f"glass_nle_pretrain_nla_m_novd_z8_r{r}"] = _nle_bake_repeat(
            _nle_pretrain(_NLA_M, _EB_VARIANT, whiten_k=8, epochs=150), r)
        ft = _nle_finetune(f"glass_nle_pretrain_nla_m_novd_z8_r{r}", ensemble_repeats=9,
                           whiten_k=8, warmstart_max_gap_nats=22.0,
                           gower_data=_GOWER_NLA_M, gower_eb=_EB_VARIANT)
        ft["max_trainval_cosmos"] = [300]
        ft["train_frac"] = 0.8
        ft["val_frac"] = 0.2
        ft["test_frac"] = 0.0   # test set = fixed ids, so the fractions must sum to 1
        ft["fixed_test_sim_ids"] = _GOWER_TEST_IDS
        kids_legacy_novd_experiments[f"gower_nle_finetune_nla_m_novd_z8_r{r}_ens9"] = _nle_bake_repeat(ft, r)


_register_main_nle_novd_z8()


def _register_early_nle_ens5_preview_r4():
    ft = _nle_finetune("glass_nle_pretrain_nla_m_novd_z8_r4", ensemble_repeats=5,
                       whiten_k=8, warmstart_max_gap_nats=22.0,
                       gower_data=_GOWER_NLA_M, gower_eb=_EB_VARIANT)
    ft.pop("max_trainval_cosmos", None)
    ft["train_frac"] = 0.8
    ft["val_frac"] = 0.2
    ft["test_frac"] = 0.0   # test set = fixed ids, so the fractions must sum to 1
    ft["fixed_test_sim_ids"] = _GOWER_TEST_IDS
    kids_legacy_novd_experiments["gower_nle_finetune_nla_m_novd_z8_r4_ens5_early"] = \
        _nle_bake_repeat(ft, 4)


_register_early_nle_ens5_preview_r4()


# --- sub-variate chains {nla, nla_z}: encoder finetune, Stage A, Stage B ---
# _SUB_VARIATES[S] = (glass_store, gower_store, cosmo_param_names, preset_overrides, smoke_cosmo)
_SUB_VARIATES = {
    "nla":   (_NLA,   _GOWER_NLA,   _COSMO_8_NLA,  _A_IA_NLA_BOX, _COSMO_8_NLA),
    "nla_z": (_NLA_Z, _GOWER_NLA_Z, _COSMO_9_NLAZ, _A_IA_NLA_BOX, _COSMO_9),
}


def _encoder_finetune_novd(glass_data, cosmo, preset):
    c = _encoder_finetune_z8(glass_data, _EB_VARIANT, cosmo, repeat_indices=(0, 1, 2, 3, 4),
                             preset_overrides=preset)
    c["pretrained_embedding_ckpt_path"] = _FOUNDATION_CKPT
    c["model_kwargs"] = {**c["model_kwargs"], "map_kwargs": _RESNET_MAPKW}
    return c


def _register_sub_variate(S, glass_data, gower_data, cosmo, preset, smoke_cosmo):
    kids_legacy_novd_experiments[f"glass_encoder_finetune_{S}_novd_z8"] = _encoder_finetune_novd(
        glass_data, cosmo, preset)
    _enc_smoke = _encoder_finetune_z8_smoke(smoke_cosmo)
    _enc_smoke["model_kwargs"] = {**_enc_smoke["model_kwargs"], "map_kwargs": _RESNET_MAPKW}
    kids_legacy_novd_experiments[f"glass_encoder_finetune_{S}_novd_z8_smoke"] = _enc_smoke
    for r in range(5):
        kids_legacy_novd_experiments[f"glass_nle_pretrain_{S}_novd_z8_r{r}"] = _nle_bake_repeat(
            _nle_pretrain(glass_data, _EB_VARIANT, whiten_k=8, epochs=150,
                          cosmo_param_names=cosmo, preset_overrides=preset), r)
        ft = _nle_finetune(f"glass_nle_pretrain_{S}_novd_z8_r{r}", ensemble_repeats=9,
                           whiten_k=8, warmstart_max_gap_nats=22.0,
                           gower_data=gower_data, gower_eb=_EB_VARIANT,
                           cosmo_param_names=cosmo, preset_overrides=preset)
        ft["max_trainval_cosmos"] = [100]
        ft["train_frac"] = 0.7
        ft["val_frac"] = 0.3
        ft["test_frac"] = 0.0
        ft["fixed_test_sim_ids"] = _GOWER_TEST_IDS_LAST100
        kids_legacy_novd_experiments[f"gower_nle_finetune_{S}_novd_z8_r{r}_ens9"] = _nle_bake_repeat(ft, r)


for _S, (_g, _gow, _cos, _pre, _smk) in _SUB_VARIATES.items():
    _register_sub_variate(_S, _g, _gow, _cos, _pre, _smk)


# --- Gower-only single-fidelity probes ---
_GOWER_ONLY_BAND_CKPT_DIR = f"{_CKPT}/gower_only_band_nla_m_novd/"
_GOWER_ONLY_SPLIT = {"max_trainval_cosmos": [500], "train_frac": 0.891,
                     "val_frac": 0.099, "test_frac": 0.01}


def _gower_only_band():
    """Stage-I bandpower MLP trained on the Gower store (single fidelity)."""
    c = _band_lmin50()
    c["data_patterns"] = _GOWER_NLA_M
    c.update(_GOWER_ONLY_SPLIT)
    c.pop("repeats", None)
    c["repeat_indices"] = [0, 1, 2]
    return c


def _gower_only_hybrid_z8_resnet():
    """Stage-II hybrid with a frozen per-repeat Gower band and the PreActResNet map encoder.

    Trained on Gower only."""
    c = _hybrid_lmin50_z8()
    c["data_patterns"] = _GOWER_NLA_M
    c["eb_map_variant"] = _EB_VARIANT
    c["model_kwargs"] = {**c["model_kwargs"], "map_kwargs": _RESNET_MAPKW}
    c["pretrained_band_ckpt_path"] = _GOWER_ONLY_BAND_CKPT_DIR
    c["epochs"] = 225
    c.update(_GOWER_ONLY_SPLIT)
    c.pop("repeats", None)
    c["repeat_indices"] = [0, 1, 2]
    return c


def _gower_only_hybrid_z8_unet():
    """UNet-backbone variant of the Gower-only hybrid with extra regularisation.

    The extra regularisation is map and flow dropout 0.1 and weight decay 0.05."""
    c = _hybrid_lmin50_z8()
    c["data_patterns"] = _GOWER_NLA_M
    c["eb_map_variant"] = _EB_VARIANT
    c["model_kwargs"] = {**c["model_kwargs"],
                         "map_kwargs": {**c["model_kwargs"]["map_kwargs"], "dropout": 0.1}}
    c["optimizer_kwargs"] = {"weight_decay": 0.05, "betas": (0.9, 0.999)}
    c["flow_kwargs"] = {**c.get("flow_kwargs", {}), "dropout": 0.1}
    c["pretrained_band_ckpt_path"] = _GOWER_ONLY_BAND_CKPT_DIR
    c.update(_GOWER_ONLY_SPLIT)
    c.pop("repeats", None)
    c["repeat_indices"] = [0, 1, 2]
    c["epochs"] = 225
    return c


def _gower_only_hybrid_z8_resnet_banddrop():
    """Gower-only ResNet hybrid with band dropout and patch-head scale parity."""
    c = _gower_only_hybrid_z8_resnet()
    c["model_kwargs"] = {**c["model_kwargs"],
                         "band_dropout_p": 0.2, "patch_head_init_gain": 16.0}
    return c


kids_legacy_novd_experiments["gower_only_band_nla_m_novd"] = _gower_only_band()
kids_legacy_novd_experiments["gower_only_hybrid_nla_m_novd_z8_resnet"] = _gower_only_hybrid_z8_resnet()
kids_legacy_novd_experiments["gower_only_hybrid_nla_m_novd_z8_unet"] = _gower_only_hybrid_z8_unet()


def _gower_only_warmstart_glass_z8():
    """Gower-only hybrid warm-started from the GLASS foundation at the full training LR."""
    c = _gower_only_hybrid_z8_resnet()
    c["checkpoint_path"] = _FOUNDATION_CKPT
    c.pop("pretrained_band_ckpt_path", None)
    c["freeze_band"] = False
    c["match_num_cosmo"] = False
    c["repeat_indices"] = [0, 2]
    return c


kids_legacy_novd_experiments["gower_only_hybrid_nla_m_novd_z8_resnet_banddrop"] = \
    _gower_only_hybrid_z8_resnet_banddrop()
kids_legacy_novd_experiments["gower_only_warmstart_glass_z8"] = _gower_only_warmstart_glass_z8()
kids_legacy_novd_experiments["gower_only_hybrid_nla_m_novd_z8_resnet_banddrop_smoke"] = \
    _hybrid_novd_z8_variant_smoke(
        mk_extra={"map_kwargs": _RESNET_MAPKW,
                  "band_dropout_p": 0.2, "patch_head_init_gain": 16.0},
        top_extra={"train_frac": 0.891, "val_frac": 0.099, "test_frac": 0.01,
                   "repeat_indices": [0]})

kids_legacy_novd_experiments["gower_only_hybrid_nla_m_novd_z8_resnet_smoke"] = \
    _hybrid_novd_z8_variant_smoke(
        mk_extra={"map_kwargs": _RESNET_MAPKW},
        top_extra={"train_frac": 0.891, "val_frac": 0.099, "test_frac": 0.01,
                   "repeat_indices": [0]})

_gower_only_unet_smoke = _hybrid_lmin50_z8_smoke()
_gower_only_unet_smoke["model_kwargs"] = {
    **_gower_only_unet_smoke["model_kwargs"],
    "map_kwargs": {**_gower_only_unet_smoke["model_kwargs"]["map_kwargs"], "dropout": 0.1}}
_gower_only_unet_smoke["optimizer_kwargs"] = {"weight_decay": 0.05, "betas": (0.9, 0.999)}
_gower_only_unet_smoke["flow_kwargs"] = {**_gower_only_unet_smoke.get("flow_kwargs", {}),
                                        "dropout": 0.1}
_gower_only_unet_smoke.update({"train_frac": 0.891, "val_frac": 0.099, "test_frac": 0.01,
                               "repeat_indices": [0]})
kids_legacy_novd_experiments["gower_only_hybrid_nla_m_novd_z8_unet_smoke"] = _gower_only_unet_smoke
