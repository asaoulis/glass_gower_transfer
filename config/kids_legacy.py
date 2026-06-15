"""KiDS-Legacy NLA-M hybrid training configs (architectures/cluster-nla-training task).

Two-stage hybrid on the production NLA-M mocks (the fwhm8_lmin50_lcut1400 smoothing variant)
stored on the gpu4 share. Stage I trains a bandpower MLP; Stage II trains the
kids_hybrid_bandpowers_maps model with that band encoder loaded FROZEN
(pretrained_band_ckpt_path), exactly like config/ablations.py.

The two hybrid runs are IDENTICAL except batch_size — `b100` is the original batch, `b224` is
the memory-optimal batch the AMP+torch.compile speedups free up (compile cuts peak mem 50->19 GB
@B100, so a bigger batch fits). Both carry the shipped speedups via `ml_perf`
(amp = bf16 autocast scoped to the map encoder, flow fp32; compile = torch.compile the CNN
backbone) — see config.ml_perf in config/default.py and the architectures/model-optimization task.

NLA-M uses the DEFAULT a_ia box [4.48, 7] + the full 9-param set (incl. b_ia), so no
scaler preset_overrides are needed (unlike the plain-NLA configs in experiments.py).

Workflow:
  1. train.py kids_legacy_band_nla_m          # Stage I (shared band MLP) -> band checkpoint
  2. wait for the band checkpoint, then:
  3. train.py kids_legacy_hybrid_nla_m_b100    # Stage II, batch 100 (loads frozen band)
  4. train.py kids_legacy_hybrid_nla_m_b224    # Stage II, batch 224 (loads frozen band)
"""

_NLA_M_DATA = "/share/gpu4/asaoulis/transfer_datasets/glass_mocks_nla_m/output_*.h5"
_EB_VARIANT = "fwhm8_lmin50_lcut1400"
_COSMO_9 = ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"]
# Stage I writes here; Stage II loads the frozen band from this folder.
_BAND_CKPT_DIR = "/share/gpu5/asaoulis/transfer_models/checkpoints/kids_legacy_band_nla_m/"

# Shipped training-speed options (default-OFF elsewhere; ON for the hybrid runs).
_ML_PERF = {"amp": True, "compile": "backbone", "tf32": False, "fused_adam": False}


def _hybrid(batch_size):
    """A kids_hybrid_bandpowers_maps run (mirrors ablation_glass_no_side) on NLA-M maps."""
    return {
        "data_patterns": _NLA_M_DATA,
        "eb_map_variant": _EB_VARIANT,
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south"],
        "model_kwargs": {
            "bandpower_type": "mlp",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 8,
            "map_kwargs": {
                "encoder_type": "unet_o3",
                "pool_types": ("avg", "max", "gem"),
                "patch_conditioning": None,
            },
            "bandpower_kwargs": {"hidden_multiple": 32, "dropout": 0},
        },
        "latent_dim": 8 + 8,
        "pretrained_band_ckpt_path": _BAND_CKPT_DIR,
        "freeze_band": True,
        "ml_perf": _ML_PERF,
        "persistent_workers": True,
        "prefetch_factor": 4,
        "pin_memory": True,
        "num_workers": 8,
        "epochs": 60,
        "batch_size": batch_size,
        "scheduler_type": "cyclic",
        "scheduler_kwargs": {"warmup": 2000, "min_factor": 0.1, "cyclic_period_steps": 6000},
        "lr": 0.0004,
        "use_KL_loss": False,
        "flow_kwargs": {"hidden_features": 32},
        "project": "kids-legacy-nla-m",
        "cosmo_param_names": _COSMO_9,
        "max_trainval_cosmos": 530,   # single int => ONE run (a list would sweep!)
        "repeats": 1,
    }


kids_legacy_experiments = {
    # --- Stage I: shared bandpower MLP on NLA-M (batch = the ablations default, 100) ---
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
        "project": "kids-legacy-nla-m",
        "max_trainval_cosmos": 530,   # single int => ONE band model (not a sweep)
        "repeats": 1,
    },
    # --- Stage II: two hybrids, identical except batch_size ---
    "kids_legacy_hybrid_nla_m_b100": _hybrid(100),
    "kids_legacy_hybrid_nla_m_b224": _hybrid(224),
}
