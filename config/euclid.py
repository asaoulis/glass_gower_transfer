"""Experiment rows for the Euclid DR3 hybrid: the KiDS recipe retrofitted to 13 tomographic bins.

Stage I trains a (91, 8) bandpower MLP. Stage II freezes it inside the PreActResNet map hybrid.
The mocks are written straight to gpu5 as one f16 E variant, so no prebake is needed.
"""

_EUCLID_DATA = "/share/gpu5/asaoulis/transfer_datasets/euclid_mocks_v1/output_*.h5"
_CKPT = "/share/gpu5/asaoulis/transfer_models/checkpoints"
_BAND_CKPT_DIR = f"{_CKPT}/euclid_band/"

# Inference targets omega_m, sigma_8, w0. The sims draw all seven COSMO_PARAM_NAMES
# (src/Euclid/simulation_config.py).
_COSMO_3 = ["omega_m", "sigma_8", "w0"]

_N_TOMO = 13                 # 13 map channels
_BAND_SHAPE = (91, 8)        # 13*14/2 spectra x 8 Brown bands

# Map encoder kwargs identical to the KiDS foundation (_RESNET_MAPKW in config/kids_legacy_novd.py).
_RESNET_MAPKW = {"encoder_type": "preact_resnet", "patch_conditioning": None,
                 "pool_types": ("avg", "gem"),
                 "stage_channels": (32, 64, 128, 256, 256), "blocks_per_stage": 3}

_ML_PERF = {"amp": True, "compile": "backbone", "tf32": False, "fused_adam": False}

euclid_experiments = {}


# --- Stage I: bandpower MLP ---
def _euclid_band():
    """Stage-I bandpower MLP with the (91, 8) Euclid input shape, 200 epochs."""
    return {
        "data_patterns": _EUCLID_DATA,
        "model_type": "kids_bandpowers_mlp",
        "dataset_quantities": ["mixed_bandpowers"],
        "n_tomo_bins": _N_TOMO,
        "batch_size": 100,
        "latent_dim": 8,
        "flow_kwargs": {"hidden_features": 32, "dropout": 0.0},
        # input_shape (91, 8) is the only change: the MLP flattens it to the input width.
        "model_kwargs": {"hidden_multiple": 32, "dropout": 0.0, "input_shape": _BAND_SHAPE},
        "epochs": 200,
        "cosmo_param_names": _COSMO_3,
        "project": "euclid-pretraining",
        "repeat_indices": [0, 1, 2],
    }


euclid_experiments["euclid_band"] = _euclid_band()


# --- Stage II: frozen-band PreActResNet hybrid ---
def _euclid_hybrid_z8_resnet():
    """Stage-II hybrid on the Euclid store.

    13 tomographic channels, frozen per-repeat band, PreActResNet map encoder, batch size and
    prefetch set for 28 MB samples."""
    return {
        "data_patterns": _EUCLID_DATA,
        "eb_map_variant": None,          # bare `E` group, the store carries no variant tag
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south"],
        "n_tomo_bins": _N_TOMO,
        "model_kwargs": {
            "bandpower_type": "mlp",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 8,
            "map_kwargs": _RESNET_MAPKW,
            "bandpower_kwargs": {"hidden_multiple": 32, "dropout": 0,
                                 "input_shape": _BAND_SHAPE},
            "hybrid_output_dim": 8,
        },
        "latent_dim": 8 + 8,
        "pretrained_band_ckpt_path": _BAND_CKPT_DIR,
        "freeze_band": True,
        "ml_perf": _ML_PERF,
        # 28 MB per sample: the prefetch queue dominates host RAM
        "num_workers": 8,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "pin_memory": True,
        "epochs": 80,
        "batch_size": 16,
        "scheduler_type": "cyclic",
        "scheduler_kwargs": {"warmup": 2000, "min_factor": 0.1, "cyclic_period_steps": 6000},
        "lr": 0.0001,
        "use_KL_loss": False,
        "flow_kwargs": {"hidden_features": 32},
        "project": "euclid-pretraining",
        "cosmo_param_names": _COSMO_3,
        # scaler cap: 100 files for the 28 MB maps, 1000 for the bandpowers (per-key override)
        "scaler_fit_max_obs": 100,
        "scaler_fit_max_obs_by_key": {"mixed_bandpowers": 1000},
        # RA roll: exact symmetry of the full-RA footprint
        "augment_eb_ra_roll": True,
        "repeat_indices": [0, 1, 2],
    }


euclid_experiments["euclid_hybrid_z8_resnet"] = _euclid_hybrid_z8_resnet()


def _bench(batch, epochs=2):
    c = _euclid_hybrid_z8_resnet()
    c["batch_size"] = batch
    c["epochs"] = epochs
    c["repeat_indices"] = [0]
    c["pretrained_band_ckpt_path"] = None
    c["freeze_band"] = False
    c["project"] = "euclid-bench"
    c["log_throughput"] = True
    return c


euclid_experiments["euclid_hybrid_bench_b16"] = _bench(16)
euclid_experiments["euclid_hybrid_bench_b32"] = _bench(32)


# --- image-matched schedule: epochs scaled so images seen match the KiDS recipe ---
_IMG_CYCLE_STEPS = 37500
_IMG_EPOCHS = 155


def _euclid_hybrid_imgmatch(repeat_indices=(0, 1)):
    """euclid_hybrid_z8_resnet with the schedule matched on images seen rather than optimiser steps."""
    c = _euclid_hybrid_z8_resnet()
    c["epochs"] = _IMG_EPOCHS
    c["scheduler_kwargs"] = {**c["scheduler_kwargs"],
                             "cyclic_period_steps": _IMG_CYCLE_STEPS}
    c["repeat_indices"] = list(repeat_indices)
    return c


euclid_experiments["euclid_hybrid_z8_resnet_imgmatch"] = _euclid_hybrid_imgmatch()


# --- warm start: KiDS bgp map CNN into the Euclid hybrid (r0) ---
_KIDS_BGP_SC8A1_CKPT_DIR = f"{_CKPT}/kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1/"


def _euclid_hybrid_imgmatch_warm():
    """The image-matched recipe for r0 with the KiDS bgp map CNN as the initialisation.

    The backbone is not frozen."""
    c = _euclid_hybrid_imgmatch(repeat_indices=(0,))
    c["pretrained_backbone_ckpt_path"] = _KIDS_BGP_SC8A1_CKPT_DIR
    c["backbone_prefix"] = "model.embedding_net.patch_encoder.shared_cnn."
    c["freeze_backbone"] = False
    return c


euclid_experiments["euclid_hybrid_z8_resnet_imgmatch_warm"] = _euclid_hybrid_imgmatch_warm()


# --- resume: reload the finished image-matched runs ---
_IMGMATCH_CKPT_DIR = f"{_CKPT}/euclid_hybrid_z8_resnet_imgmatch/"


def _euclid_hybrid_imgmatch_resume():
    """euclid_hybrid_z8_resnet_imgmatch restarted from its own finished checkpoints."""
    c = _euclid_hybrid_imgmatch(repeat_indices=(0, 1))
    c["checkpoint_path"] = _IMGMATCH_CKPT_DIR
    return c


euclid_experiments["euclid_hybrid_z8_resnet_imgmatch_resume"] = _euclid_hybrid_imgmatch_resume()
