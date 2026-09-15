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
        # C2, per-key. The cap is a MEMORY knob set by the biggest key: the E maps are
        # ~28 MB/file so they need 100, but `mixed_bandpowers` is ~3 kB/file and 1000 costs ~3 MB.
        # The bandpower cap MUST match Stage I's (`_euclid_band` leaves the 1000 default),
        # because Stage II FREEZES that band encoder: `mixed_bandpowers` is the first key in both
        # stages and the fit's RNG is seeded identically, so cap 1000 here reproduces Stage I's
        # scaler EXACTLY, while cap 100 would hand the frozen encoder inputs shifted ~0.095 sd.
        "scaler_fit_max_obs": 100,
        "scaler_fit_max_obs_by_key": {"mixed_bandpowers": 1000},
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
    c["log_throughput"] = True   # B1: the whole point of the bench is these numbers
    return c


euclid_experiments["euclid_hybrid_bench_b16"] = _bench(16)
euclid_experiments["euclid_hybrid_bench_b32"] = _bench(32)


# === Amendment 2026-09-15 — the IMAGE-matched schedule (user directive) =========================
# The first Stage-II submit (`euclid_hybrid_z8_resnet`, 3 repeats, jobs 1363863/4/5) was sized to
# STEP-match the KiDS b100 foundation: 1021 steps/ep x 80 ep = 81.7k steps vs KiDS's 810 x 100 =
# 81.0k, with `cyclic_period_steps` left at the foundation's 6000. Verified on W&B (the lr trace is
# triangular between 0.05*lr and lr) and correct as far as it goes — but it matches the wrong
# quantity. "Steps" changed MEANING when the batch dropped 100 -> 16; the user's intent is to match
# the number of IMAGES the network sees, per cycle and in total:
#
#     KiDS foundation:  81 000 img/epoch x 100 epochs   = 8.10 M images total
#                       6 000 steps x b100              = 600 000 images per cyclic period
#                                                       => 13.5 cycles
#     Euclid:           16 336 train images (b16 => 1021 steps/epoch)
#       -> cyclic_period_steps = 600 000 / 16 = 37 500   (ONE KiDS cycle's worth of images)
#       -> a FULL image match would be 8.10M / 16 336 = 496 epochs, and HALF of it 248 epochs.
#          NEITHER FITS. The gatekeeper caps a train job at 48 h
#          (`ssh_glass_gatekeeper.sh` train-submit: `need_num "$wall" 0.1 48`) and the MEASURED
#          steady state is 764-926 s/epoch (last-8 means of the three step-matched repeats under
#          3-way l40s contention, spikes to 1231 s); even the B1 bench's DEDICATED-card 733 s
#          gives 248 x 733 = 50.5 h. So no amount of staggering or running fewer jobs at once
#          rescues 248 - it does not fit uncontended either. There is no mid-run resume (gap C4).
#          User decision 2026-09-15: take the largest budget that fits ONE 48 h job, ~170 epochs.
#
# ⭐ Why 155 and not 170. The cyclic phase starts at the LR FLOOR (SequentialLR hands over to
# CyclicLR at `warmup_steps`, and CyclicLR's step 0 is base_lr = 0.05*lr), so a run ending on an
# INTEGER number of cycles ends annealed, while one ending mid-cycle stops on the way back up.
# That is a detail at the foundation's 13.5 cycles; at ~4 it is most of the schedule.
#     155 ep x 1021 = 158 255 steps; warmup = 5 % = 7 912; 150 343 cyclic steps / 37 500
#                   = 4.009 cycles -> ends 0.9 % of a cycle past the floor, i.e. annealed.
#     170 ep would give 4.40 cycles, ending at ~80 % of peak LR - and its 4th (last) LR minimum
#                   falls at epoch ~155 ANYWAY, so the extra 15 epochs only climb away from the
#                   point the best checkpoint would come from.
# 155 epochs = 2.53 M images = 31 % of the KiDS foundation, 4 full cycles, and at 900 s/epoch runs
# 39 h against the 48 h wall - ~9 h of margin, still fitting even at 1000 s/epoch.
#
# NOT changed, deliberately:
#   * `lr` stays 1e-4 — the user explicitly declined to rescale the PEAK LR (a linear batch rescale
#     would give 2e-4 * 16/100 = 3.2e-5; the halved 1e-4 already in place is the sqrt rescale).
#   * warmup needs no edit: `lightning/base.py` uses warmup_frac=0.05 of TOTAL steps (the config's
#     `warmup: 2000` key is DEAD — cf. the note at config/kids_legacy_novd.py:430), so it tracks the
#     epoch count automatically. 248 ep => 12 660 warmup steps = 5%, exactly as KiDS.
#
# ⭐ SEPARATE EXPERIMENT NAMES ARE LOAD-BEARING. These must NOT write into
# `euclid_hybrid_z8_resnet/pretrain_ncosmoNone_{0,1}/` alongside the step-matched run: the folder is
# shared per repeat index and `find_best_checkpoint` takes the GLOBAL minimum across it, which would
# silently merge two recipes into one "best" checkpoint (same trap documented for M11b in
# config/kids_legacy_bgp.py).
_IMG_CYCLE_STEPS = 37500
_IMG_EPOCHS = 155


def _euclid_hybrid_imgmatch(repeat_indices=(0, 1)):
    """`euclid_hybrid_z8_resnet` with the schedule matched on IMAGES instead of optimiser steps.

    Derived from the running recipe so the architecture, the frozen Euclid band, the data store and
    every tuning knob cannot drift — only the epoch count and the cyclic period move."""
    c = _euclid_hybrid_z8_resnet()
    c["epochs"] = _IMG_EPOCHS
    c["scheduler_kwargs"] = {**c["scheduler_kwargs"],
                             "cyclic_period_steps": _IMG_CYCLE_STEPS}
    c["repeat_indices"] = list(repeat_indices)
    return c


euclid_experiments["euclid_hybrid_z8_resnet_imgmatch"] = _euclid_hybrid_imgmatch()


# === The warm-start arm — KiDS bgp r0 map CNN into the Euclid hybrid ============================
# User directive 2026-09-15: one repeat (r0) warm-started from the KiDS "p9 foundation"
# `kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1` — the 9-param cosmology/IA encoder the BGP campaign
# itself uses as a warm-start parent — while the NEW Euclid bandpower encoder is still loaded
# separately and FROZEN (`pretrained_band_ckpt_path` + `freeze_band`, inherited untouched).
# So: KiDS supplies ONLY the map CNN; nothing KiDS-shaped touches the 2-pt branch.
#
# ⚠️ `backbone_prefix` MUST be overridden. The default `'shared_cnn.backbone.'` describes the UNet
# encoder's layout; the PreActResNet has NO `backbone.` level — its tensors are
# `...shared_cnn.{stem,stages.*,poolproj.*,fc.*}` (verified against the on-disk r0 checkpoint,
# epoch=56 val -5.3183). With the default prefix `k.startswith(...)` matches NOTHING and you get a
# SILENT zero-key "warm" start (the failure class of memory `e890aec-embeddings-source-encoder-cutover`).
#
# Shape mismatch is EXPECTED and is exactly one tensor: `stem.weight` is (32, 6, 3, 3) for KiDS's 6
# tomographic bins vs (32, 13, 3, 3) for Euclid's 13. `load_partial_weights` shape-skips it (random
# init) and loads everything downstream.
# ⭐ LAUNCH-VERIFY in the cluster log: `Loaded keys: 108`, exactly 1 skipped-shape entry naming
# `stem.weight`, and the resolved source path ending `pretrain_ncosmoNone_0/`. `Loaded keys: 0` means
# the prefix is wrong — cancel rather than train a config that only LOOKS warm.
# The local smoke CANNOT gate this (smoke_test_experiment.py nulls every `pretrained_*_ckpt_path`);
# it is gated instead by scripts' dry-run against the local checkpoint copy, see the task logbook.
_KIDS_BGP_SC8A1_CKPT_DIR = f"{_CKPT}/kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1/"


def _euclid_hybrid_imgmatch_warm():
    """The image-matched recipe, r0 only, with the KiDS bgp map CNN as a warm start.

    freeze_backbone stays False: the CNN is an initialisation to fine-tune, not a frozen feature
    extractor — the only frozen module is the Euclid bandpower encoder, as in every other Stage-II
    row here. lr stays 1e-4, the same as the arm it is compared against (user, 2026-09-15)."""
    c = _euclid_hybrid_imgmatch(repeat_indices=(0,))
    c["pretrained_backbone_ckpt_path"] = _KIDS_BGP_SC8A1_CKPT_DIR
    c["backbone_prefix"] = "model.embedding_net.patch_encoder.shared_cnn."
    c["freeze_backbone"] = False
    return c


euclid_experiments["euclid_hybrid_z8_resnet_imgmatch_warm"] = _euclid_hybrid_imgmatch_warm()
