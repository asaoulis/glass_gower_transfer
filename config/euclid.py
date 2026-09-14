"""Euclid DR3 hybrid-learning experiment suite.

Retrofit of the KiDS-Legacy production recipe (`config/kids_legacy_novd.py` M1 band -> M2
`kids_legacy_hybrid_nla_m_novd_z8_resnet`) onto the Euclid DR3 GLASS mocks in
`euclid_mocks_v1` (20 411 mocks, 542 GB, 1 mock per cosmology). Full rationale, the verified
KiDS<->Euclid data-shape diff and the step-matching arithmetic:
`.claude/runs/euclid/model-training/artifacts/SPEC.md`.

Depends on three additive code changes, all landed with unchanged defaults:
  C1 `config.n_tomo_bins`     -> a 13-channel map stem (without it the stem is silently built
                                 with KiDS's 6 channels against 13-channel data)
  C2 `config.scaler_fit_max_obs` -> caps the scaler fit at 100 files (the 1000 default would
                                 concatenate ~28 GB/key at Euclid's 28.46 MB/mock)
  C3 `config.augment_eb_ra_roll` -> RA cyclic roll, an EXACT symmetry of the full-RA Dec band

Numbers marked [MEASURE] are pending the B1 benchmark (`euclid_hybrid_bench_b16` / `_b32`).
"""

# The Euclid mocks are written straight to gpu5 as a single f16 E variant: no prebake, and the
# maps live on the l40s node's LOCAL disk already.
_EUCLID_DATA = "/share/gpu5/asaoulis/transfer_datasets/euclid_mocks_v1/output_*.h5"
_CKPT = "/share/gpu5/asaoulis/transfer_models/checkpoints"
_BAND_CKPT_DIR = f"{_CKPT}/euclid_band/"

# INFERENCE TARGETS (user decision 2026-09-14, Q1): just these three.
# The sims still DRAW all seven of src/Euclid/simulation_config.py:COSMO_PARAM_NAMES
# (omega_m, sigma_8, ombh2, h, ns, w0, mnu) from the Gower prior, so the four we do not infer
# (ombh2, h, ns, mnu) remain live scatter in the data and the network must marginalise them
# implicitly. That is the intended behaviour, not an oversight.
# All three already have boxes in src/ml/data/constants.py:COSMO_PARAM_PRESET_MINMAX
# -> no preset_overrides needed.
_COSMO_3 = ["omega_m", "sigma_8", "w0"]

_N_TOMO = 13                 # -> 13 map channels (code change C1)
_BAND_SHAPE = (91, 8)        # 13*14/2 spectra x 8 Brown bands

# The PRODUCTION map encoder, byte-identical to the KiDS foundation's `_RESNET_MAPKW`
# (config/kids_legacy_novd.py). Do not "improve" it: the plain PreActResNet is what broke the
# -4.4 UNet wall (memory `counts-hybrid-resnet-recipe`).
_RESNET_MAPKW = {"encoder_type": "preact_resnet", "patch_conditioning": None,
                 "pool_types": ("avg", "gem"),
                 "stage_channels": (32, 64, 128, 256, 256), "blocks_per_stage": 3}

_ML_PERF = {"amp": True, "compile": "backbone", "tf32": False, "fused_adam": False}

euclid_experiments = {}


# === Stage I — bandpower MLP (v100; bandpowers only, no maps) ===================================
def _euclid_band():
    """Clone of config/kids_legacy.py:_band_lmin50 with the (91, 8) Euclid bandpower shape.

    epochs 200 step-matches the KiDS band arm (40 ep x 810 steps = ~32k): Euclid has ~16.2k train
    files, so b100 gives ~162 steps/epoch and 200 epochs = ~32k steps.  [MEASURE in B1]
    3 repeats (user decision Q3) -> checkpoints/euclid_band/pretrain_ncosmoNone_{0,1,2}/;
    Stage-II repeat i then loads band i FROZEN."""
    return {
        "data_patterns": _EUCLID_DATA,
        "model_type": "kids_bandpowers_mlp",
        "dataset_quantities": ["mixed_bandpowers"],
        "n_tomo_bins": _N_TOMO,
        "batch_size": 100,
        "latent_dim": 8,
        "flow_kwargs": {"hidden_features": 32, "dropout": 0.0},
        # input_shape is the ONLY architectural change: KidsBandpowersMLP flattens
        # input_shape[0]*input_shape[1], and its (21, 8) default would build a 168-wide input
        # layer against Euclid's 728-D flat vector.
        "model_kwargs": {"hidden_multiple": 32, "dropout": 0.0, "input_shape": _BAND_SHAPE},
        "epochs": 200,
        "cosmo_param_names": _COSMO_3,
        "project": "euclid-pretraining",
        "repeat_indices": [0, 1, 2],
    }


euclid_experiments["euclid_band"] = _euclid_band()


# === Stage II — frozen-band PreActResNet hybrid (l40s/a100; NEVER v100) =========================
def _euclid_hybrid_z8_resnet():
    """Clone of kids_legacy_hybrid_nla_m_novd_z8_resnet on the Euclid store.

    Differences from the KiDS foundation, and only these:
      * data_patterns / eb_map_variant  -> the untagged bare-`E` Euclid store
      * n_tomo_bins 13                  -> a 13-channel stem (code change C1)
      * bandpower_kwargs.input_shape    -> (91, 8)
      * cosmo_param_names               -> the 3-vector (omega_m, sigma_8, w0)
      * batch_size / epochs             -> resized for a 6.4x larger image and 5x fewer files
      * scaler_fit_max_obs              -> 100 (code change C2; 1000 whole files = 28 GB/key)

    scheduler_kwargs are deliberately UNCHANGED: they are expressed in optimizer STEPS, so
    step-matching the epoch count preserves the LR schedule shape exactly.

    batch_size 16 / epochs 80 = ~1015 steps/epoch x 80 = ~81k steps, matching the KiDS
    foundation's 810 x 100.  BOTH NUMBERS ARE [MEASURE] — B1 sets them from measured peak GPU
    memory and smp/s.  [Q2, Q3, Q4]"""
    return {
        "data_patterns": _EUCLID_DATA,
        "eb_map_variant": None,          # bare `E` group; the store carries no variant tag
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
        # 28.46 MB/sample: the prefetch queue is the host-RAM driver.
        # 8 x 2 x 16 x 28.46 MB = 7.3 GB before pinning -> submit with --mem-gb 64.
        "num_workers": 8,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "pin_memory": True,
        "epochs": 80,                    # [MEASURE]
        "batch_size": 16,                # Q2: b16 start; B1 raises it if l40s/a100 memory allows
        "scheduler_type": "cyclic",
        "scheduler_kwargs": {"warmup": 2000, "min_factor": 0.1, "cyclic_period_steps": 6000},
        "lr": 0.0001,                    # Q4: halved from the KiDS b100 recipe for the small batch
        "use_KL_loss": False,
        "flow_kwargs": {"hidden_features": 32},
        "project": "euclid-pretraining",
        "cosmo_param_names": _COSMO_3,
        "scaler_fit_max_obs": 100,       # code change C2
        # Q5: RA cyclic roll — an EXACT symmetry of this footprint (full-RA band), so it is free
        # augmentation for a 20k-sample set that otherwise has none. Code change C3.
        "augment_eb_ra_roll": True,
        "repeat_indices": [0, 1, 2],         # Q3: 3 repeats
    }


euclid_experiments["euclid_hybrid_z8_resnet"] = _euclid_hybrid_z8_resnet()


# === B1 benchmark clones — 2 epochs, one per candidate batch size ===============================
def _bench(batch, epochs=2):
    c = _euclid_hybrid_z8_resnet()
    c["batch_size"] = batch
    c["epochs"] = epochs
    c["repeat_indices"] = [0]
    c["pretrained_band_ckpt_path"] = None   # band not trained yet; from-scratch band is fine here
    c["freeze_band"] = False
    c["project"] = "euclid-bench"
    return c


euclid_experiments["euclid_hybrid_bench_b16"] = _bench(16)
euclid_experiments["euclid_hybrid_bench_b32"] = _bench(32)
