"""Factories and experiment rows for the GLASS NLA-M Stage I band MLP and Stage II map hybrid.

Stage I trains the bandpower MLP. Stage II freezes it inside the hybrid map encoder.
"""

_NLA_M_DATA = "/share/gpu5/asaoulis/transfer_datasets/glass_mocks_nla_m_f16/output_*.h5"
_EB_VARIANT = "fwhm8_lmin50_lcut1400"

_NLA_M_DATA_FWHM4 = "/share/gpu5/asaoulis/transfer_datasets/glass_mocks_nla_m_f16_fwhm4_lmin50_lcut1400/output_*.h5"
_EB_VARIANT_FWHM4 = "fwhm4_lmin50_lcut1400"
_NLA_M_DATA_FWHM12 = "/share/gpu5/asaoulis/transfer_datasets/glass_mocks_nla_m_f16_fwhm12_lmin50_lcut1024/output_*.h5"
_EB_VARIANT_FWHM12 = "fwhm12_lmin50_lcut1024"
_COSMO_9 = ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"]
# Stage I writes here, Stage II loads the frozen band from it
_BAND_CKPT_DIR = "/share/gpu5/asaoulis/transfer_models/checkpoints/kids_legacy_band_nla_m/"

_NLA_M_DATA_LMIN50_RAW = "/share/gpu4/asaoulis/transfer_datasets/glass_mocks_nla_m_lmin50/output_*.h5"
# All band repeats live here as pretrain_ncosmoNone_{0..3}. Hybrid repeat i loads '_{i}'.
_BAND_CKPT_DIR_LMIN50 = "/share/gpu5/asaoulis/transfer_models/checkpoints/kids_legacy_band_nla_m_lmin50/"
# fwhm4 E-mode store, tag fwhm4_lmin56_lcut1400 (4 arcmin smoothing)
_NLA_M_DATA_LMIN50_FWHM4 = "/share/gpu5/asaoulis/transfer_datasets/glass_mocks_nla_m_lmin50_f16_fwhm4_lmin56_lcut1400/output_*.h5"
_EB_VARIANT_LMIN50_FWHM4 = "fwhm4_lmin56_lcut1400"

_ML_PERF = {"amp": True, "compile": "backbone", "tf32": False, "fused_adam": False}


def _hybrid(batch_size, data_patterns=_NLA_M_DATA, eb_variant=_EB_VARIANT):
    """Stage II kids_hybrid_bandpowers_maps config.

    Args: batch_size, data_patterns (prebaked store glob), eb_variant (E-map smoothing tag).
    The frozen band is _BAND_CKPT_DIR."""
    return {
        "data_patterns": data_patterns,
        "eb_map_variant": eb_variant,
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south"],
        "model_kwargs": {
            "bandpower_type": "mlp",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 8,
            "map_kwargs": {
                "encoder_type": "unet_o3",
                "pool_types": ("avg", "max", "gem"),
                "patch_conditioning": ("side_info"),
            },
            "bandpower_kwargs": {"hidden_multiple": 32, "dropout": 0},
        },
        "latent_dim": 8 + 8,
        "pretrained_band_ckpt_path": _BAND_CKPT_DIR,
        "freeze_band": True,
        "ml_perf": _ML_PERF,
        "num_workers": 16,
        "persistent_workers": True,
        "prefetch_factor": 4,
        "pin_memory": True,
        "epochs": 100,
        "batch_size": batch_size,
        "scheduler_type": "cyclic",
        "scheduler_kwargs": {"warmup": 2000, "min_factor": 0.1, "cyclic_period_steps": 6000},
        "lr": 0.0002,
        "use_KL_loss": False,
        "flow_kwargs": {"hidden_features": 32},
        "project": "glass-pretraining",
        "cosmo_param_names": _COSMO_9,
        "repeats": 1,
    }


kids_legacy_experiments = {
    # --- Stage I: bandpower MLP ---
    "kids_legacy_band_nla_m": {
        "data_patterns": _NLA_M_DATA,
        "model_type": "kids_bandpowers_mlp",
        "dataset_quantities": ["mixed_bandpowers"],
        "batch_size": 100,
        "latent_dim": 8,
        "flow_kwargs": {"hidden_features": 32, "dropout": 0.0},
        "model_kwargs": {"hidden_multiple": 32, "dropout": 0.0},
        "epochs": 40,
        "cosmo_param_names": _COSMO_9,
        "project": "glass-pretraining",
        "repeats": 1,
    },
    # --- Stage II: two hybrids, identical except batch_size ---
    "kids_legacy_hybrid_nla_m_b100": _hybrid(100),
    "kids_legacy_hybrid_nla_m_b224": _hybrid(224),
    # --- Stage II smoothing comparison: one hybrid per EB variant ---
    "kids_legacy_hybrid_nla_m_fwhm4":  _hybrid(100, _NLA_M_DATA_FWHM4,  _EB_VARIANT_FWHM4),
    "kids_legacy_hybrid_nla_m_fwhm8":  _hybrid(100, _NLA_M_DATA,        _EB_VARIANT),
    "kids_legacy_hybrid_nla_m_fwhm12": _hybrid(100, _NLA_M_DATA_FWHM12, _EB_VARIANT_FWHM12),
}


def _band_lmin50():
    """Stage I bandpower-MLP config (mixed_bandpowers only)."""
    return {
        "data_patterns": _NLA_M_DATA_LMIN50_RAW,
        "model_type": "kids_bandpowers_mlp",
        "dataset_quantities": ["mixed_bandpowers"],
        "batch_size": 100,
        "latent_dim": 8,
        "flow_kwargs": {"hidden_features": 32, "dropout": 0.0},
        "model_kwargs": {"hidden_multiple": 32, "dropout": 0.0},
        "epochs": 40,
        "cosmo_param_names": _COSMO_9,
        "project": "glass-pretraining",
        "repeats": 4,
    }


kids_legacy_experiments["kids_legacy_band_nla_m_lmin50"] = _band_lmin50()


def _hybrid_lmin50():
    """Stage II hybrid on the lmin50 fwhm4 store.

    Repeat i loads frozen band repeat i from _BAND_CKPT_DIR_LMIN50."""
    c = _hybrid(100, _NLA_M_DATA_LMIN50_FWHM4, _EB_VARIANT_LMIN50_FWHM4)
    c["pretrained_band_ckpt_path"] = _BAND_CKPT_DIR_LMIN50
    c["repeats"] = 4
    c["num_workers"] = 8
    c["prefetch_factor"] = 2
    return c


kids_legacy_experiments["kids_legacy_hybrid_nla_m_lmin50_fwhm4"] = _hybrid_lmin50()


def _hybrid_lmin50_z6():
    """_hybrid_lmin50() with a 6-D summary head (hybrid_output_dim=6)."""
    c = _hybrid_lmin50()
    c["model_kwargs"] = {**c["model_kwargs"], "hybrid_output_dim": 6}
    return c


kids_legacy_experiments["kids_legacy_hybrid_nla_m_lmin50_fwhm4_z6"] = _hybrid_lmin50_z6()


def _hybrid_lmin50_z6_smoke():
    """z6 hybrid with a from-scratch band on the local fwhm8 fixture store (no checkpoint needed)."""
    c = _hybrid(8, _NLA_M_DATA, _EB_VARIANT)
    c["model_kwargs"] = {**c["model_kwargs"], "hybrid_output_dim": 6}
    c["pretrained_band_ckpt_path"] = None
    c["freeze_band"] = False
    c["epochs"] = 3
    c["num_workers"] = 2
    c["prefetch_factor"] = 2
    c["persistent_workers"] = False
    return c


kids_legacy_experiments["kids_legacy_hybrid_nla_m_lmin50_fwhm4_z6_smoke"] = _hybrid_lmin50_z6_smoke()


def _hybrid_lmin50_z8():
    """_hybrid_lmin50() with an 8-D summary head (hybrid_output_dim=8), the foundation encoder."""
    c = _hybrid_lmin50()
    c["model_kwargs"] = {**c["model_kwargs"], "hybrid_output_dim": 8}
    return c


kids_legacy_experiments["kids_legacy_hybrid_nla_m_lmin50_fwhm4_z8"] = _hybrid_lmin50_z8()


def _hybrid_lmin50_z8_smoke():
    """z8 hybrid with a from-scratch band on the local fwhm8 fixture store (no checkpoint needed)."""
    c = _hybrid(8, _NLA_M_DATA, _EB_VARIANT)
    c["model_kwargs"] = {**c["model_kwargs"], "hybrid_output_dim": 8}
    c["pretrained_band_ckpt_path"] = None
    c["freeze_band"] = False
    c["epochs"] = 3
    c["num_workers"] = 2
    c["prefetch_factor"] = 2
    c["persistent_workers"] = False
    return c


kids_legacy_experiments["kids_legacy_hybrid_nla_m_lmin50_fwhm4_z8_smoke"] = _hybrid_lmin50_z8_smoke()


# --- Sub-variate encoder finetunes (warm-start the z8 foundation, fresh flow head) ---
_COSMO_9_NLAZ = ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_z"]
_COSMO_8_NLA = ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia"]
_NLA_Z_DATA_FWHM4 = "/share/gpu5/asaoulis/transfer_datasets/glass_mocks_nla_z_f16_fwhm4_lmin56_lcut1400/output_*.h5"
_FOUNDATION_Z8_CKPT = "/share/gpu5/asaoulis/transfer_models/checkpoints/kids_legacy_hybrid_nla_m_lmin50_fwhm4_z8/"


def _encoder_finetune_z8(data_patterns, eb_variant, cosmo_param_names, repeat_indices=(0,),
                         preset_overrides=None):
    """Whole-encoder finetune of the z8 foundation (band and map) with a fresh flow head.

    Trains on a GLASS sub-variate suite. preset_overrides: cosmo scaler boxes, for example
    _A_IA_NLA_BOX for the NLA family."""
    c = _hybrid_lmin50_z8()
    c["data_patterns"] = data_patterns
    c["eb_map_variant"] = eb_variant
    c["cosmo_param_names"] = list(cosmo_param_names)
    c.pop("pretrained_band_ckpt_path", None)
    c.pop("freeze_band", None)
    c["pretrained_embedding_ckpt_path"] = _FOUNDATION_Z8_CKPT
    c["freeze_embedding_net"] = False
    c["scheduler_type"] = "exp"
    c["scheduler_kwargs"] = {"gamma": 0.984, "warmup_steps": 0}   # decays to 1/5 of lr over the run
    c["lr"] = 0.0002
    c["epochs"] = 100
    c.pop("repeats", None)
    c["repeat_indices"] = list(repeat_indices)
    c["match_num_cosmo"] = False
    c["project"] = "glass-pretraining"
    if preset_overrides:
        c["scaler_options"] = {
            "data": {"type": "standard", "keys": None},
            "cosmo": {"type": "preset", "preset_overrides": dict(preset_overrides)},
        }
    return c


_A_IA_NLA_BOX = {"a_ia": (-6.0, 6.0)}   # a_ia box for NLA/NLA-z/TATT (the NLA-M default is 4.48-7.0)

# nla_z: a_ia ~ U[-6, 6], so the a_ia box is overridden
kids_legacy_experiments["glass_encoder_finetune_nla_z_z8"] = _encoder_finetune_z8(
    _NLA_Z_DATA_FWHM4, _EB_VARIANT_LMIN50_FWHM4, _COSMO_9_NLAZ, repeat_indices=(0, 1, 2, 3, 4),
    preset_overrides=_A_IA_NLA_BOX)


def _encoder_finetune_z8_smoke(cosmo_param_names):
    """_encoder_finetune_z8 on the local fixture with a from-scratch encoder and 3 epochs."""
    c = _encoder_finetune_z8(_NLA_M_DATA, _EB_VARIANT, cosmo_param_names, repeat_indices=(0,))
    c["pretrained_embedding_ckpt_path"] = None
    c["epochs"] = 3
    c["num_workers"] = 2
    c["prefetch_factor"] = 2
    c["persistent_workers"] = False
    return c


kids_legacy_experiments["glass_encoder_finetune_nla_z_z8_smoke"] = _encoder_finetune_z8_smoke(_COSMO_9)

_NLA_DATA_FWHM4 = "/share/gpu5/asaoulis/transfer_datasets/glass_mocks_nla_f16_fwhm4_lmin56_lcut1400/output_*.h5"
_NOVD_DATA_FWHM4 = "/share/gpu5/asaoulis/transfer_datasets/glass_mocks_nla_m_novd_f16_fwhm4_lmin56_lcut1400/output_*.h5"
kids_legacy_experiments["glass_encoder_finetune_nla_z8"] = _encoder_finetune_z8(   # nla: a_ia~U[-6,6]
    _NLA_DATA_FWHM4, _EB_VARIANT_LMIN50_FWHM4, _COSMO_8_NLA, repeat_indices=(0, 1, 2, 3, 4),
    preset_overrides=_A_IA_NLA_BOX)
kids_legacy_experiments["glass_encoder_finetune_no_vd_z8"] = _encoder_finetune_z8(   # no_vd=NLA-M box
    _NOVD_DATA_FWHM4, _EB_VARIANT_LMIN50_FWHM4, _COSMO_9, repeat_indices=(0, 1, 2, 3, 4))


def _hybrid_vicreg(batch_size, data_patterns=_NLA_M_DATA, eb_variant=_EB_VARIANT,
                   repeat_indices=None, band_ckpt_dir=_BAND_CKPT_DIR):
    """_hybrid() with the VICReg summary regulariser (VICRegRegularisedNDELightningModule).

    batch_size must be divisible by vicreg_m_per_cosmo."""
    c = _hybrid(batch_size, data_patterns, eb_variant)
    c["use_KL_loss"] = False
    c["use_vicreg_loss"] = True
    c["vicreg_sim_coeff"] = 1.0
    c["vicreg_var_coeff"] = 1.0
    c["vicreg_cov_coeff"] = 1.0
    c["vicreg_gamma"] = 1.0
    c["vicreg_eps"] = 1e-4
    c["vicreg_m_per_cosmo"] = 2
    c["pretrained_band_ckpt_path"] = band_ckpt_dir
    if repeat_indices is not None:
        c["repeat_indices"] = repeat_indices
        c["repeats"] = 4
    return c


def _hybrid_vicreg_smoke():
    """VICReg hybrid on the local fwhm8 fixture store with a from-scratch band."""
    c = _hybrid_vicreg(8)
    c["pretrained_band_ckpt_path"] = None
    c["freeze_band"] = False
    c["epochs"] = 3
    c["num_workers"] = 2
    c["prefetch_factor"] = 2
    c["persistent_workers"] = False
    return c


kids_legacy_experiments["kids_legacy_hybrid_vicreg_smoke"] = _hybrid_vicreg_smoke()


def _hybrid_vicreg_lmin50():
    """VICReg hybrid on the lmin50 fwhm4 store, reusing the frozen lmin50 bands."""
    c = _hybrid_vicreg(100, _NLA_M_DATA_LMIN50_FWHM4, _EB_VARIANT_LMIN50_FWHM4,
                       band_ckpt_dir=_BAND_CKPT_DIR_LMIN50)
    c["repeats"] = 4
    c["num_workers"] = 8
    c["prefetch_factor"] = 2
    return c


kids_legacy_experiments["kids_legacy_hybrid_nla_m_lmin50_fwhm4_vicreg"] = _hybrid_vicreg_lmin50()


def _gpu5_locality_test():
    """Throughput probe: the b100 hybrid reading the gpu5 store."""
    c = _hybrid(100)
    c["data_patterns"] = "/share/gpu5/asaoulis/transfer_datasets/glass_mocks_prior/output_*.h5"
    c.pop("eb_map_variant", None)
    c["pretrained_band_ckpt_path"] = None
    c["freeze_band"] = False
    c["epochs"] = 3
    return c


kids_legacy_experiments["kids_legacy_hybrid_gpu5_test"] = _gpu5_locality_test()


# --- NLE chain: Stage A flow pretrain on GLASS, Stage B finetune on Gower ---

# Gower fwhm4 prebake (tag as for the GLASS store)
_GOWER_NLA_M_DATA_FWHM4 = "/share/gpu5/asaoulis/transfer_datasets/gower_mocks_nla_m_f16_fwhm4_lmin56_lcut1400/output_*.h5"
_GOWER_EB_VARIANT_FWHM4 = "fwhm4_lmin56_lcut1400"
_NLE_PRETRAIN_CKPT = "/share/gpu5/asaoulis/transfer_models/checkpoints/{exp}/"


def _nle_pretrain(data_patterns=_NLA_M_DATA_LMIN50_FWHM4, eb_variant=_EB_VARIANT_LMIN50_FWHM4,
                  whiten_k=None, epochs=250, cosmo_param_names=_COSMO_9, preset_overrides=None):
    """NLE flow pre-trained on the full GLASS fwhm4 suite over a frozen hybrid-encoder source.

    whiten_k: PCA-whiten the embeddings to k dimensions (None = standardise only).
    epochs: training epochs. run_evaluation is False."""
    c = {
        "data_patterns": data_patterns,
        "eb_map_variant": eb_variant,
        "dataset_quantities": [],
        "latent_dim": 8,
        "epochs": epochs,
        "batch_size": 128,
        "lr": 0.001,
        "flow_kwargs": {"hidden_features": 64},
        "project": "gower-finetuning",
        "cosmo_param_names": list(cosmo_param_names),
        "inference_mode": "nle",
        "repeat_indices": [0],
        "match_num_cosmo": False,
        "scale_embeddings": False,
        "run_evaluation": False,
    }
    if whiten_k:
        c["whiten_embeddings"] = {"k": int(whiten_k)}
    if preset_overrides:
        c["scaler_options"] = {
            "data": {"type": "standard", "keys": None},
            "cosmo": {"type": "preset", "preset_overrides": dict(preset_overrides)},
        }
    return c


def _nle_finetune(pretrain_exp, ensemble_repeats=1, whiten_k=None, warmstart_max_gap_nats=None,
                  gower_data=_GOWER_NLA_M_DATA_FWHM4, gower_eb=_GOWER_EB_VARIANT_FWHM4,
                  cosmo_param_names=_COSMO_9, preset_overrides=None):
    """NLE flow finetuned on the prebaked Gower fwhm4 store.

    Loads the Stage-A flow from checkpoints/<pretrain_exp>/.
    Args: ensemble_repeats, whiten_k, warmstart_max_gap_nats."""
    c = _nle_pretrain(gower_data, gower_eb, whiten_k=whiten_k,
                      cosmo_param_names=cosmo_param_names, preset_overrides=preset_overrides)
    c["epochs"] = 50
    c["lr"] = 0.00008
    c["load_pretrained_flow"] = True
    c["pretrained_band_ckpt_path"] = _NLE_PRETRAIN_CKPT.format(exp=pretrain_exp)
    c["max_trainval_cosmos"] = [80]
    c["train_frac"] = 0.7
    c["val_frac"] = 0.2
    c["run_evaluation"] = True
    c["ensemble_repeats"] = ensemble_repeats
    if warmstart_max_gap_nats is not None:
        c["whiten_warmstart_max_gap_nats"] = warmstart_max_gap_nats
    return c


kids_legacy_experiments["glass_nle_pretrain_nla_m_base"] = _nle_pretrain()
kids_legacy_experiments["glass_nle_pretrain_nla_m_vicreg"] = _nle_pretrain()
kids_legacy_experiments["gower_nle_finetune_nla_m_base"] = _nle_finetune("glass_nle_pretrain_nla_m_base", ensemble_repeats=5)
kids_legacy_experiments["gower_nle_finetune_nla_m_vicreg"] = _nle_finetune("glass_nle_pretrain_nla_m_vicreg", ensemble_repeats=5)


# --- Whitened NLE chain: PCA-whiten the embedding, fit once on the GLASS train split ---
kids_legacy_experiments["glass_nle_pretrain_nla_m_base_white6"] = _nle_pretrain(whiten_k=6, epochs=150)
kids_legacy_experiments["gower_nle_finetune_nla_m_base_white6"] = _nle_finetune(
    "glass_nle_pretrain_nla_m_base_white6", ensemble_repeats=5, whiten_k=6)

kids_legacy_experiments["glass_nle_pretrain_nla_m_z6"] = _nle_pretrain(whiten_k=6, epochs=150)
kids_legacy_experiments["gower_nle_finetune_nla_m_z6"] = _nle_finetune(
    "glass_nle_pretrain_nla_m_z6", ensemble_repeats=5, whiten_k=6)


def _nle_bake_repeat(c, i):
    """Set repeat_indices=[i] on the NLE config c and return it."""
    c["repeat_indices"] = [int(i)]
    return c


kids_legacy_experiments["glass_nle_pretrain_nla_m_z6_r2"] = _nle_bake_repeat(_nle_pretrain(whiten_k=6, epochs=150), 2)
kids_legacy_experiments["gower_nle_finetune_nla_m_z6_r2"] = _nle_bake_repeat(
    _nle_finetune("glass_nle_pretrain_nla_m_z6_r2", ensemble_repeats=1, whiten_k=6), 2)
kids_legacy_experiments["gower_nle_finetune_nla_m_z6_r2_ens5"] = _nle_bake_repeat(
    _nle_finetune("glass_nle_pretrain_nla_m_z6_r2", ensemble_repeats=5, whiten_k=6), 2)

kids_legacy_experiments["glass_nle_pretrain_nla_m_z8_r0"] = _nle_bake_repeat(_nle_pretrain(whiten_k=8, epochs=150), 0)
# k=8 pure-whiten is rank-deficient, so the warm-start gap is large: guard threshold 22
kids_legacy_experiments["gower_nle_finetune_nla_m_z8_r0"] = _nle_bake_repeat(
    _nle_finetune("glass_nle_pretrain_nla_m_z8_r0", ensemble_repeats=1, whiten_k=8,
                  warmstart_max_gap_nats=22.0), 0)
kids_legacy_experiments["gower_nle_finetune_nla_m_z8_r0_ens5"] = _nle_bake_repeat(
    _nle_finetune("glass_nle_pretrain_nla_m_z8_r0", ensemble_repeats=5, whiten_k=8,
                  warmstart_max_gap_nats=22.0), 0)


# --- Main-variate NLE: 5 repeats, 9-member ensemble (z8, pure-whiten k=8) ---
def _register_main_nle_z8():
    for r in range(5):
        kids_legacy_experiments[f"glass_nle_pretrain_nla_m_z8_r{r}"] = _nle_bake_repeat(
            _nle_pretrain(whiten_k=8, epochs=150), r)
        ft = _nle_finetune(f"glass_nle_pretrain_nla_m_z8_r{r}", ensemble_repeats=9,
                           whiten_k=8, warmstart_max_gap_nats=22.0)
        ft["max_trainval_cosmos"] = [300]
        ft["train_frac"] = 0.8
        ft["val_frac"] = 0.2
        ft["test_frac"] = 0.0   # test set = fixed ids, so the fractions must sum to 1
        ft["fixed_test_sim_ids"] = "config/fixed_test_sets/gower_test_ids.json"
        kids_legacy_experiments[f"gower_nle_finetune_nla_m_z8_r{r}_ens9"] = _nle_bake_repeat(ft, r)


_register_main_nle_z8()


# --- NPE whole-model finetune on the Gower fwhm4 store (train.py) ---
_GLASS_HYBRID_CKPT = "/share/gpu5/asaoulis/transfer_models/checkpoints/{exp}/"


def _npe_finetune_lmin50(checkpoint_dir):
    c = _hybrid_lmin50()
    c["data_patterns"] = _GOWER_NLA_M_DATA_FWHM4
    c["eb_map_variant"] = _GOWER_EB_VARIANT_FWHM4
    c["checkpoint_path"] = checkpoint_dir
    c.pop("pretrained_band_ckpt_path", None)
    c["freeze_band"] = False
    c["epochs"] = 25
    c["lr"] = 1e-5
    c["batch_size"] = 128
    c["scheduler_type"] = "exp"
    c["scheduler_kwargs"] = {"warmup": 0}
    c["max_trainval_cosmos"] = [80]
    c["train_frac"] = 0.65
    c["val_frac"] = 0.25
    c["match_num_cosmo"] = False
    c["repeat_indices"] = [0]
    c.pop("repeats", None)
    c["project"] = "gower-finetuning"
    return c


kids_legacy_experiments["gower_npe_finetune_nla_m_base"] = _npe_finetune_lmin50(
    _GLASS_HYBRID_CKPT.format(exp="kids_legacy_hybrid_nla_m_lmin50_fwhm4"))
kids_legacy_experiments["gower_npe_finetune_nla_m_vicreg"] = _npe_finetune_lmin50(
    _GLASS_HYBRID_CKPT.format(exp="kids_legacy_hybrid_nla_m_lmin50_fwhm4_vicreg"))


def _npe_finetune_z8(checkpoint_dir, data_patterns=_GOWER_NLA_M_DATA_FWHM4,
                     eb_variant=_GOWER_EB_VARIANT_FWHM4):
    """Gower NPE 9-member ensemble finetune of the z8 hybrid.

    Whole-model load from checkpoint_dir, 300 train/val cosmologies, the fixed 200-id test set."""
    c = _hybrid_lmin50_z8()
    c["data_patterns"] = data_patterns
    c["eb_map_variant"] = eb_variant
    c["checkpoint_path"] = checkpoint_dir
    c.pop("pretrained_band_ckpt_path", None)
    c["freeze_band"] = False
    c["epochs"] = 10
    c["lr"] = 1e-5
    c["batch_size"] = 128
    c["scheduler_type"] = "exp"
    c["scheduler_kwargs"] = {"warmup": 0}
    c["ensemble_repeats"] = 9
    c["max_trainval_cosmos"] = [300]
    c["train_frac"] = 0.8
    c["val_frac"] = 0.2
    c["test_frac"] = 0.0   # test set = fixed ids, so the fractions must sum to 1
    c["fixed_test_sim_ids"] = "config/fixed_test_sets/gower_test_ids.json"
    c["match_num_cosmo"] = False
    c["repeat_indices"] = [0, 1, 2, 3, 4]
    c.pop("repeats", None)
    c["project"] = "gower-finetuning"
    return c


kids_legacy_experiments["gower_npe_finetune_nla_m_z8"] = _npe_finetune_z8(
    _GLASS_HYBRID_CKPT.format(exp="kids_legacy_hybrid_nla_m_lmin50_fwhm4_z8"))

kids_legacy_experiments["gower_npe_finetune_nla_m_vicreg_v2"] = _npe_finetune_lmin50(
    _GLASS_HYBRID_CKPT.format(exp="kids_legacy_hybrid_nla_m_lmin50_fwhm4_vicreg"))

kids_legacy_experiments["gower_nle_finetune_nla_m_vicreg_npesrc"] = _nle_finetune("glass_nle_pretrain_nla_m_vicreg")
kids_legacy_experiments["gower_nle_finetune_nla_m_vicreg_npesrc"]["match_num_cosmo"] = True
