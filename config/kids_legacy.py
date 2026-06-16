"""KiDS-Legacy NLA-M hybrid training configs (architectures/cluster-nla-training task).

Two-stage hybrid on the NLA-M mocks (the fwhm8_lmin50_lcut1400 smoothing variant) on the gpu4
share. Mirrors the canonical glass config `glass_hybrid_patches_16_9param` (side_info patch
conditioning, lr 1e-4, cyclic schedule, NO max_trainval_cosmos — glass sims have no dataset cap),
project "glass-pretraining". Stage I trains a bandpower MLP; Stage II trains the
kids_hybrid_bandpowers_maps with that band loaded FROZEN (pretrained_band_ckpt_path), as in ablations.

The two hybrid runs are IDENTICAL except batch_size — `b100` (original) vs `b224` (the
memory-optimal batch the AMP+torch.compile speedups free up). Both carry the shipped speedups via
`ml_perf` (amp = bf16 autocast scoped to the map encoder, flow fp32; compile = torch.compile the CNN
backbone) — see config.ml_perf / the architectures/model-optimization task.

No max_trainval_cosmos + default match_num_cosmo=True ⇒ band + hybrid share the unique repeat-match
"ncosmoNone_0", so the hybrid's get_best_checkpoint finds the band ckpt under
checkpoints/kids_legacy_band_nla_m/pretrain_ncosmoNone_0/ (and does NOT collide with the old,
superseded ncosmo530_0 band from the first attempt).

Workflow:
  1. train.py kids_legacy_band_nla_m          # Stage I (shared band MLP) -> band checkpoint
  2. wait for the band checkpoint, then:
  3. train.py kids_legacy_hybrid_nla_m_b100    # Stage II, batch 100 (loads frozen band)
  4. train.py kids_legacy_hybrid_nla_m_b224    # Stage II, batch 224 (loads frozen band)

3-way smoothing comparison (b100, one hybrid per on-disk EB variant; share the one frozen band):
  - kids_legacy_hybrid_nla_m_fwhm4   # fwhm4_lmin50_lcut1400
  - kids_legacy_hybrid_nla_m_fwhm8   # fwhm8_lmin50_lcut1400 (existing store; == _b100)
  - kids_legacy_hybrid_nla_m_fwhm12  # fwhm12_lmin50_lcut1024
The on-disk EB tags are the b7203e0 set (lmin50), which the prebake reads directly (a missing tag
prints ok=0/all-bad), NOT the drifted HEAD set (lmin56). The two non-fwhm8 stores are prebaked onto
gpu5 from gpu4 glass_mocks_nla_m (jobs 1308302 / 1308303).
"""

# f16 extracted-E store PRE-BAKED on gpu5 (l40s-local, ~4.5x faster than the raw gpu4 NFS set).
# Built via: run_remote.py prebake --src-datasets-root gpu4 --src-dir glass_mocks_nla_m
#   --out-dir glass_mocks_nla_m_f16 --eb-variant fwhm8_lmin50_lcut1400 --keep-variant-tag
_NLA_M_DATA = "/share/gpu5/asaoulis/transfer_datasets/glass_mocks_nla_m_f16/output_*.h5"
_EB_VARIANT = "fwhm8_lmin50_lcut1400"

# The other TWO b7203e0 on-disk EB smoothing variants, prebaked onto gpu5 the same way
# (jobs 1308302 / 1308303, 2026-06-16). Used by the 3-way smoothing comparison below.
# NB: the on-disk tags are the b7203e0 set (lmin50), NOT the drifted HEAD set (lmin56).
_NLA_M_DATA_FWHM4 = "/share/gpu5/asaoulis/transfer_datasets/glass_mocks_nla_m_f16_fwhm4_lmin50_lcut1400/output_*.h5"
_EB_VARIANT_FWHM4 = "fwhm4_lmin50_lcut1400"
_NLA_M_DATA_FWHM12 = "/share/gpu5/asaoulis/transfer_datasets/glass_mocks_nla_m_f16_fwhm12_lmin50_lcut1024/output_*.h5"
_EB_VARIANT_FWHM12 = "fwhm12_lmin50_lcut1024"
_COSMO_9 = ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"]
# Stage I writes here; Stage II loads the frozen band from this folder.
_BAND_CKPT_DIR = "/share/gpu5/asaoulis/transfer_models/checkpoints/kids_legacy_band_nla_m/"

# Shipped training-speed options (default-OFF elsewhere; ON for the hybrid runs).
_ML_PERF = {"amp": True, "compile": "backbone", "tf32": False, "fused_adam": False}


def _hybrid(batch_size, data_patterns=_NLA_M_DATA, eb_variant=_EB_VARIANT):
    """A kids_hybrid_bandpowers_maps run mirroring glass_hybrid_patches_16_9param, on NLA-M maps.

    data_patterns / eb_variant select the prebaked smoothing-variant store (default fwhm8); the band
    is smoothing-independent so the SAME frozen _BAND_CKPT_DIR is loaded regardless of variant."""
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
        # NFS-out-of-core: more workers to hide gpu4 read latency (submit with --ncpu 16).
        "num_workers": 16,
        "persistent_workers": True,
        "prefetch_factor": 4,
        "pin_memory": True,
        "epochs": 150,
        "batch_size": batch_size,
        "scheduler_type": "cyclic",
        "scheduler_kwargs": {"warmup": 2000, "min_factor": 0.1, "cyclic_period_steps": 4000},
        "lr": 0.0002,
        "use_KL_loss": False,
        "flow_kwargs": {"hidden_features": 32},
        "project": "glass-pretraining",
        "cosmo_param_names": _COSMO_9,
        "repeats": 1,
    }


kids_legacy_experiments = {
    # --- Stage I: shared bandpower MLP on NLA-M (no dataset cap; project glass-pretraining) ---
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
    # --- Stage II 3-way smoothing comparison: one hybrid per b7203e0 on-disk EB variant ---
    # Identical except the prebaked map store + eb_map_variant; all b100, all load the SAME frozen
    # band (_BAND_CKPT_DIR, ncosmoNone_0). fwhm8 reuses the existing store (== _b100).
    "kids_legacy_hybrid_nla_m_fwhm4":  _hybrid(100, _NLA_M_DATA_FWHM4,  _EB_VARIANT_FWHM4),
    "kids_legacy_hybrid_nla_m_fwhm8":  _hybrid(100, _NLA_M_DATA,        _EB_VARIANT),
    "kids_legacy_hybrid_nla_m_fwhm12": _hybrid(100, _NLA_M_DATA_FWHM12, _EB_VARIANT_FWHM12),
}


def _gpu5_locality_test():
    """DATA-LOCALITY THROUGHPUT TEST: the SAME amp+compile B=100 hybrid, but reading gpu5
    glass_mocks_prior (LOCAL disk to the l40s node) instead of gpu4 glass_mocks_nla_m (NFS).
    Same simulator ⇒ same map geometry ⇒ same bytes/batch, so the it/s difference isolates the
    data-locality effect. Band trained from scratch (freeze_band=False, no ckpt) — the band is
    tiny so it doesn't affect map-read throughput. Short run; cancel after reading steady it/s."""
    c = _hybrid(100)
    c["data_patterns"] = "/share/gpu5/asaoulis/transfer_datasets/glass_mocks_prior/output_*.h5"
    c.pop("eb_map_variant", None)            # glass_mocks_prior uses bare E/B groups
    c["pretrained_band_ckpt_path"] = None    # no band-ckpt dependency for a throughput probe
    c["freeze_band"] = False
    c["epochs"] = 3
    return c


kids_legacy_experiments["kids_legacy_hybrid_gpu5_test"] = _gpu5_locality_test()
