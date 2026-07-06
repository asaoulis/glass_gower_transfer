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

# --- lmin50 PRODUCTION run (4 ensemble members; fwhm4 E-mode only) -------------------------------
# The completed full GLASS pre-training set glass_mocks_nla_m_lmin50 (raw f32 maps on gpu4).
# 4 members trained as TWO submissions per stage via repeat_indices: [0,1] (_r01) and [2,3] (_r23).
# The Stage-I band is bandpower-only (smoothing-independent) so it reads bandpowers DIRECTLY off the
# raw gpu4 store (no map prebake needed) and starts immediately, in parallel with the map prebake.
_NLA_M_DATA_LMIN50_RAW = "/share/gpu4/asaoulis/transfer_datasets/glass_mocks_nla_m_lmin50/output_*.h5"
# ONE unified band ckpt dir. All 4 band repeats live here as pretrain_ncosmoNone_{0,1,2,3}; the hybrid
# loads band i via pretrained_band_match_string "_{i}". Populate it EITHER by consolidating the earlier
# split _r01/_r23 band runs into this dir (no retrain — `mv .../kids_legacy_band_nla_m_lmin50_{r01,r23}/
# pretrain_ncosmoNone_* here`) OR by a fresh retrain of `kids_legacy_band_nla_m_lmin50` split across two
# jobs via `train ... --repeat-indices 0,1` / `--repeat-indices 2,3` (same exp name => one dir).
_BAND_CKPT_DIR_LMIN50 = "/share/gpu5/asaoulis/transfer_models/checkpoints/kids_legacy_band_nla_m_lmin50/"
# Prebaked fwhm4 E-mode store (prebake job 1309640 on COMPUTE). Tag CONFIRMED fwhm4_lmin56_lcut1400:
# the full 96985-file / 178G store proves the E_<tag> group exists on disk (a wrong tag => empty store).
# 4-arcmin smoothing scale, E mode only (N+S), f16. The hybrids train data-local on gpu5 (--gpu l40s).
_NLA_M_DATA_LMIN50_FWHM4 = "/share/gpu5/asaoulis/transfer_datasets/glass_mocks_nla_m_lmin50_f16_fwhm4_lmin56_lcut1400/output_*.h5"
_EB_VARIANT_LMIN50_FWHM4 = "fwhm4_lmin56_lcut1400"

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


def _band_lmin50():
    """Stage-I bandpower MLP on the lmin50 production mocks (mixed_bandpowers only).

    Smoothing-independent, so it reads bandpowers directly off the raw gpu4 store (no map prebake).
    ONE experiment, 4 repeats; all land under checkpoints/kids_legacy_band_nla_m_lmin50/
    pretrain_ncosmoNone_{0..3}/. Split across two parallel jobs at SUBMIT time:
      train --exp kids_legacy_band_nla_m_lmin50 --repeat-indices 0,1   (and 2,3).
    No repeat_indices in the config (left None) so the arg overrides cleanly."""
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
    """Stage-II hybrid on the lmin50 fwhm4 prebaked store; loads its FROZEN per-repeat band from the
    ONE unified band dir (_BAND_CKPT_DIR_LMIN50). ONE experiment, 4 repeats; each repeat i loads band i
    (pretrained_band_match_string '_{i}' -> pretrain_ncosmoNone_{i}). Split across two parallel jobs at
    SUBMIT time: train --exp kids_legacy_hybrid_nla_m_lmin50_fwhm4 --gpu l40s --mem-gb 28 --repeat-indices 0,1
    (and 2,3). NB: v100 OOMs on the maps -> hybrids run on l40s (or a100), NEVER v100."""
    c = _hybrid(100, _NLA_M_DATA_LMIN50_FWHM4, _EB_VARIANT_LMIN50_FWHM4)
    c["pretrained_band_ckpt_path"] = _BAND_CKPT_DIR_LMIN50
    c["repeats"] = 4
    # Fewer dataloader workers + smaller prefetch so the job fits a lowered --mem-gb (~28G) on a
    # RAM-contended l40s node. The prebaked store is gpu5-LOCAL (fast), so 8 workers still saturate.
    c["num_workers"] = 8
    c["prefetch_factor"] = 2
    return c


kids_legacy_experiments["kids_legacy_hybrid_nla_m_lmin50_fwhm4"] = _hybrid_lmin50()


def _hybrid_lmin50_z6():
    """6-D-summary variant of _hybrid_lmin50() — the ENCODER-LEVEL "whitening at the source" fix.

    IDENTICAL to kids_legacy_hybrid_nla_m_lmin50_fwhm4 (same lmin50 fwhm4 prebaked gpu5 store,
    same FROZEN per-repeat Stage-I band from _BAND_CKPT_DIR_LMIN50, same ml_perf amp+compile,
    l40s tuning, epochs 100 / lr 2e-4 / batch 100 / cyclic / flow hidden 32, 4 repeats) EXCEPT the
    hybrid head now projects the band(8)+patch(8)=16-D concat down to a 6-D FINAL summary z via
    model_kwargs['hybrid_output_dim']=6. The downstream NLE flow q(theta|z) then sees a 6-D z, so
    the ~4-D degenerate subspace that made the 16-D NLE joint-miscalibrated at N~80 never exists
    (research task nle-overconfidence-research P6: within-cosmology residuals live on a ~4-D
    manifold, cond number ~5e5; the top-6 PCs carry 100% of the sigma8 gradient). The frozen 8-D
    band is REUSED as-is (band NOT retrained); only the fresh 16->6 hybrid_head + map encoder
    train. build_model reads hybrid_output_dim from model_kwargs and mirrors it on the flow side
    (conditioning_dim <- 6). ONE experiment, 4 repeats; split across two parallel jobs at SUBMIT
    time: train --exp kids_legacy_hybrid_nla_m_lmin50_fwhm4_z6 --gpu l40s --mem-gb 28
    --repeat-indices 0,1 (and 2,3). v100 OOMs the maps -> l40s (or a100), NEVER v100."""
    c = _hybrid_lmin50()
    c["model_kwargs"] = {**c["model_kwargs"], "hybrid_output_dim": 6}
    return c


kids_legacy_experiments["kids_legacy_hybrid_nla_m_lmin50_fwhm4_z6"] = _hybrid_lmin50_z6()


def _hybrid_lmin50_z6_smoke():
    """LOCAL SMOKE-ONLY clone of kids_legacy_hybrid_nla_m_lmin50_fwhm4_z6: proves the 6-D-summary
    hybrid (hybrid_output_dim=6, 16->6 head) BUILDS and trains one epoch with a finite loss WITHOUT
    the cluster band ckpt. De-clustered exactly like _hybrid_vicreg_smoke: from-scratch band
    (pretrained_band_ckpt_path=None, freeze_band=False), epochs cut, low workers. Points at the
    fwhm8_lmin50_lcut1400 variant because the LOCAL smoke fixture (.claude/cluster/smoke_data_nla)
    only carries E_fwhm8 groups -> the production fwhm4_lmin56 config false-fails the smoke
    data-load (memory: "smoke fixture is fwhm8-only"). Run:
      OMP_NUM_THREADS=4 nice -n 10 /data/alex/glass/env/bin/python \\
        .claude/cluster/smoke_test_experiment.py \\
        --experiment kids_legacy_hybrid_nla_m_lmin50_fwhm4_z6_smoke
    NOT for cluster use (the real run is ..._z6)."""
    c = _hybrid(8, _NLA_M_DATA, _EB_VARIANT)          # fwhm8 variant the local fixture carries
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
    """8-D-summary variant of _hybrid_lmin50() — same ENCODER-LEVEL bottleneck as ..._z6 but at
    hybrid_output_dim=8 (an intermediate point between the raw 16-D concat and the 6-D whitened
    summary; a sweep companion to ..._z6). IDENTICAL to kids_legacy_hybrid_nla_m_lmin50_fwhm4
    (same lmin50 fwhm4 prebaked gpu5 store, same FROZEN per-repeat Stage-I band from
    _BAND_CKPT_DIR_LMIN50, same ml_perf amp+compile, l40s tuning, epochs 100 / lr 2e-4 / batch 100
    / cyclic / flow hidden 32, 4 repeats) EXCEPT the hybrid head projects the band(8)+patch(8)=16-D
    concat down to an 8-D FINAL summary z via model_kwargs['hybrid_output_dim']=8. The frozen 8-D
    band is REUSED as-is; only the fresh 16->8 hybrid_head + map encoder train. build_model mirrors
    hybrid_output_dim on the flow side (conditioning_dim <- 8). ONE experiment, 4 repeats; submit as
    train --exp kids_legacy_hybrid_nla_m_lmin50_fwhm4_z8 --gpu l40s --mem-gb 28 --repeat-indices 0,1
    (and 2,3). v100 OOMs the maps -> l40s (or a100), NEVER v100."""
    c = _hybrid_lmin50()
    c["model_kwargs"] = {**c["model_kwargs"], "hybrid_output_dim": 8}
    return c


kids_legacy_experiments["kids_legacy_hybrid_nla_m_lmin50_fwhm4_z8"] = _hybrid_lmin50_z8()


def _hybrid_lmin50_z8_smoke():
    """LOCAL SMOKE-ONLY clone of kids_legacy_hybrid_nla_m_lmin50_fwhm4_z8: proves the 8-D-summary
    hybrid (hybrid_output_dim=8, 16->8 head) BUILDS and trains one epoch with a finite loss WITHOUT
    the cluster band ckpt. Identical de-clustering to _hybrid_lmin50_z6_smoke: from-scratch band
    (pretrained_band_ckpt_path=None, freeze_band=False), epochs cut, low workers, fwhm8 variant the
    LOCAL smoke fixture (.claude/cluster/smoke_data_nla) carries. Run:
      OMP_NUM_THREADS=4 nice -n 10 /data/alex/glass/env/bin/python \\
        .claude/cluster/smoke_test_experiment.py \\
        --experiment kids_legacy_hybrid_nla_m_lmin50_fwhm4_z8_smoke
    NOT for cluster use (the real run is ..._z8)."""
    c = _hybrid(8, _NLA_M_DATA, _EB_VARIANT)          # fwhm8 variant the local fixture carries
    c["model_kwargs"] = {**c["model_kwargs"], "hybrid_output_dim": 8}
    c["pretrained_band_ckpt_path"] = None
    c["freeze_band"] = False
    c["epochs"] = 3
    c["num_workers"] = 2
    c["prefetch_factor"] = 2
    c["persistent_workers"] = False
    return c


kids_legacy_experiments["kids_legacy_hybrid_nla_m_lmin50_fwhm4_z8_smoke"] = _hybrid_lmin50_z8_smoke()


# --- SUB-VARIATE encoder warm-start finetune (production Phase 4) --------------------------------
# Warm-start the FOUNDATION z8 hybrid ENCODER (band + map, incl. the frozen Stage-I band — loaded as
# ONE embedding_net via pretrained_embedding_ckpt_path) onto a sub-variate GLASS suite, with a FRESH
# NDE flow head sized to that variate's parameter space. Train the WHOLE model (encoder unfrozen) with
# an exp-decay LR from the foundation pretrain LR (2e-4) down to 1/5 (~4e-5) over 100 epochs
# (exp gamma=0.984 => 0.984^100 ≈ 0.20). The map maths (fwhm4_lmin56_lcut1400) MATCH the foundation so
# the encoder features transfer; only the IA parameter set changes per variate (master_sim IA_PRIORS):
#   no_vd -> _COSMO_9 (a_ia, b_ia)   nla -> a_ia only (8-D)   nla_z -> a_ia + b_z (9-D)
# The compile-trained foundation ckpt's `_orig_mod.` backbone keys are aligned by the canonicalisation
# in load_partial_weights (so the warm start actually transfers the CNN backbone, not just the head).
# The sub-variate GLASS mocks live RAW on gpu4 (glass_mocks_{nla_m_novd,nla,nla_z}); prebake each to a
# gpu5 f16 fwhm4_lmin56_lcut1400 store first: run_remote.py prebake --src-datasets-root gpu4
#   --src-dir glass_mocks_<S> --out-dir glass_mocks_<S>_f16_fwhm4_lmin56_lcut1400
#   --eb-variant fwhm4_lmin56_lcut1400 --keep-variant-tag
_COSMO_9_NLAZ = ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_z"]
_COSMO_8_NLA = ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia"]
_NLA_Z_DATA_FWHM4 = "/share/gpu5/asaoulis/transfer_datasets/glass_mocks_nla_z_f16_fwhm4_lmin56_lcut1400/output_*.h5"
_FOUNDATION_Z8_CKPT = "/share/gpu5/asaoulis/transfer_models/checkpoints/kids_legacy_hybrid_nla_m_lmin50_fwhm4_z8/"


def _encoder_finetune_z8(data_patterns, eb_variant, cosmo_param_names, repeat_indices=(0,)):
    """Warm-start the foundation z8 hybrid ENCODER (band+map) + a FRESH flow head, then finetune the
    WHOLE model on a sub-variate GLASS suite. exp-decay LR 2e-4 -> ~4e-5 over 100 epochs."""
    c = _hybrid_lmin50_z8()                       # exact foundation architecture (hybrid_output_dim=8)
    c["data_patterns"] = data_patterns
    c["eb_map_variant"] = eb_variant
    c["cosmo_param_names"] = list(cosmo_param_names)
    # Warm-start the WHOLE encoder (band + map) from the foundation; leave the flow head FRESH.
    c.pop("pretrained_band_ckpt_path", None)      # band arrives inside the loaded embedding_net
    c.pop("freeze_band", None)
    c["pretrained_embedding_ckpt_path"] = _FOUNDATION_Z8_CKPT
    c["freeze_embedding_net"] = False             # finetune the encoder too (train all)
    c["scheduler_type"] = "exp"
    c["scheduler_kwargs"] = {"gamma": 0.984, "warmup_steps": 0}   # 0.984^100 ≈ 0.20: 2e-4 -> ~4e-5
    c["lr"] = 0.0002                              # foundation pretrain LR (decays to 1/5)
    c["epochs"] = 100
    c.pop("repeats", None)
    c["repeat_indices"] = list(repeat_indices)
    c["match_num_cosmo"] = False                  # resolve the embedding ckpt per-repeat as "_{i}"
    c["project"] = "glass-pretraining"
    return c


# nla_z, repeat 0 — HIGH-PRIORITY first test (check the warm-start training curve).
kids_legacy_experiments["glass_encoder_finetune_nla_z_z8"] = _encoder_finetune_z8(
    _NLA_Z_DATA_FWHM4, _EB_VARIANT_LMIN50_FWHM4, _COSMO_9_NLAZ, repeat_indices=(0,))


def _encoder_finetune_z8_smoke(cosmo_param_names):
    """LOCAL SMOKE clone: from-scratch encoder (no cluster ckpt) + FRESH flow on the fwhm8 nla_m local
    fixture, few epochs. Validates BUILD + train + fresh-flow + exp scheduler (the warm-start load path
    is exercised only on-cluster where the foundation ckpt exists). Uses _COSMO_9 (the local fixture
    carries a_ia+b_ia) — same 9-D flow head shape as nla_z's (a_ia, b_z)."""
    c = _encoder_finetune_z8(_NLA_M_DATA, _EB_VARIANT, cosmo_param_names, repeat_indices=(0,))
    c["pretrained_embedding_ckpt_path"] = None    # de-cluster: no foundation ckpt locally
    c["epochs"] = 3
    c["num_workers"] = 2
    c["prefetch_factor"] = 2
    c["persistent_workers"] = False
    return c


kids_legacy_experiments["glass_encoder_finetune_nla_z_z8_smoke"] = _encoder_finetune_z8_smoke(_COSMO_9)

# nla + no_vd sub-variate encoder-finetunes (Phase 4, all 5 repeats). Prebaked to gpu5 (jobs
# 1315704/1315705). nla -> 8-D params (a_ia only); no_vd -> full _COSMO_9 (a_ia,b_ia, same as main,
# just variable-depth OFF). Submit one repeat per l40s job via --repeat-indices <i> (like NPE).
_NLA_DATA_FWHM4 = "/share/gpu5/asaoulis/transfer_datasets/glass_mocks_nla_f16_fwhm4_lmin56_lcut1400/output_*.h5"
_NOVD_DATA_FWHM4 = "/share/gpu5/asaoulis/transfer_datasets/glass_mocks_nla_m_novd_f16_fwhm4_lmin56_lcut1400/output_*.h5"
kids_legacy_experiments["glass_encoder_finetune_nla_z8"] = _encoder_finetune_z8(
    _NLA_DATA_FWHM4, _EB_VARIANT_LMIN50_FWHM4, _COSMO_8_NLA, repeat_indices=(0, 1, 2, 3, 4))
kids_legacy_experiments["glass_encoder_finetune_no_vd_z8"] = _encoder_finetune_z8(
    _NOVD_DATA_FWHM4, _EB_VARIANT_LMIN50_FWHM4, _COSMO_9, repeat_indices=(0, 1, 2, 3, 4))


def _hybrid_vicreg(batch_size, data_patterns=_NLA_M_DATA, eb_variant=_EB_VARIANT,
                   repeat_indices=None, band_ckpt_dir=_BAND_CKPT_DIR):
    """Stage-II hybrid + VICReg summary regulariser (Williamson DES Y3 arXiv:2606.11309 §3.4).

    Identical to _hybrid() but selects VICRegRegularisedNDELightningModule via use_vicreg_loss and
    sets the paper's UNIT VIC weights (lambda=mu=nu=1; gamma=1, eps=1e-4). The frozen Stage-I band
    is REUSED (no VICReg in Stage I). The invariance term acts on the FINAL post-hybrid_head summary
    z and pulls together DIFFERENT REALISATIONS of the SAME cosmology (eq.13): use_vicreg_loss
    auto-enables the m-per-cosmology train batch sampler (k distinct cosmologies x vicreg_m_per_cosmo
    realisations per batch) + the per-sample cosmology-id 3rd batch element. NB: batch_size must be
    divisible by vicreg_m_per_cosmo, and k = batch_size//m must be <= the number of TRAIN
    cosmologies (the full lmin50 store has thousands, so k=50 is fine; only the smoke is constrained)."""
    c = _hybrid(batch_size, data_patterns, eb_variant)
    c["use_KL_loss"] = False
    c["use_vicreg_loss"] = True
    c["vicreg_sim_coeff"] = 1.0
    c["vicreg_var_coeff"] = 1.0
    c["vicreg_cov_coeff"] = 1.0
    c["vicreg_gamma"] = 1.0
    c["vicreg_eps"] = 1e-4
    c["vicreg_m_per_cosmo"] = 2   # m: realisations per cosmology per batch (m=2 == literal eq.13 pair)
    c["pretrained_band_ckpt_path"] = band_ckpt_dir
    if repeat_indices is not None:
        c["repeat_indices"] = repeat_indices
        c["repeats"] = 4
    return c


def _hybrid_vicreg_smoke():
    """Local SMOKE-only VICReg hybrid: from-scratch band (no ckpt dependency) on the fwhm8 variant
    the local smoke fixture carries (E_fwhm8) -> isolates the VICReg module (m-per-cosmology sampler
    + same-cosmology invariance + finite VIC loss) without needing the cluster band ckpt.

    batch_size=8, m=2 -> k=4 distinct cosmologies/batch. The smoke fixture has 8 cosmologies; the
    default ~80/10/10 split leaves ~6 train cosmologies, so k=4 <= 6 holds. num_workers low for a
    local CPU/GPU smoke on the tiny (~116-file) fixture."""
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
    """Stage-II VICReg hybrid on the lmin50 fwhm4 prebaked store; REUSES the SAME 4 frozen Stage-I
    bands as the standard lmin50 hybrid (_BAND_CKPT_DIR_LMIN50, no VICReg in Stage I). Identical
    data/band/perf to _hybrid_lmin50() but with the VICReg summary regulariser (same-cosmology
    invariance via the m-per-cosmology sampler, m=2). ONE experiment, 4 repeats; each repeat i loads
    band i. Split across two parallel jobs at SUBMIT time:
      train --exp kids_legacy_hybrid_nla_m_lmin50_fwhm4_vicreg --gpu l40s --mem-gb 28 --repeat-indices 0,1
      (and 2,3). v100 OOMs the maps -> l40s (or a100), NEVER v100. batch_size=100, m=2 -> k=50
      distinct cosmologies/batch (the lmin50 store has thousands of train cosmologies)."""
    c = _hybrid_vicreg(100, _NLA_M_DATA_LMIN50_FWHM4, _EB_VARIANT_LMIN50_FWHM4,
                       band_ckpt_dir=_BAND_CKPT_DIR_LMIN50)
    c["repeats"] = 4
    c["num_workers"] = 8
    c["prefetch_factor"] = 2
    return c


kids_legacy_experiments["kids_legacy_hybrid_nla_m_lmin50_fwhm4_vicreg"] = _hybrid_vicreg_lmin50()


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


# --- vicreg-nle-first-test: two-stage multifidelity NLE on FROZEN hybrid encoders ----------------
# Train an NLE flow p(z|theta) on the frozen 16-d hybrid-encoder embedding z, for a SINGLE model
# (repeat_indices=[0]), for BOTH the baseline and the VICReg encoder. Two stages per encoder, driven
# by `train_embeddings.py <target> <source-encoder-exp>`:
#   A. PRE-TRAIN on the full GLASS fwhm4 suite (no dataset cap; eval OFF -> no MCMC on GLASS; V100).
#   B. FINE-TUNE on the PREBAKED gower fwhm4 store (load the Stage-A flow; eval ON -> MCMC; CORES64).
# The SOURCE encoders are the kids_legacy_hybrid_nla_m_lmin50_fwhm4 / ..._vicreg experiments above,
# passed on the train_embeddings.py CLI (NOT in these dicts). match_num_cosmo=False so the repeat-0
# source/flow resolve via None_0 -> pretrain_ncosmoNone_0. dataset_quantities=[] is overwritten from
# the source encoder at runtime. eb_map_variant MUST match the encoder's fwhm4 smoothing so the
# frozen encoder sees identically-processed E maps.

# Prebaked gower fwhm4 store (Phase 5b: gower_mocks_nla_m is RAW -> prebaked to fwhm4 to MATCH the
# encoder maps). The tag is re-checked after the prebake auto-detect (try fwhm4_lmin56_lcut1400 first).
_GOWER_NLA_M_DATA_FWHM4 = "/share/gpu5/asaoulis/transfer_datasets/gower_mocks_nla_m_f16_fwhm4_lmin56_lcut1400/output_*.h5"
_GOWER_EB_VARIANT_FWHM4 = "fwhm4_lmin56_lcut1400"   # gower prebake tag; re-confirm after Phase 5b
_NLE_PRETRAIN_CKPT = "/share/gpu5/asaoulis/transfer_models/checkpoints/{exp}/"


def _nle_pretrain(data_patterns=_NLA_M_DATA_LMIN50_FWHM4, eb_variant=_EB_VARIANT_LMIN50_FWHM4,
                  whiten_k=None, epochs=250):
    """NLE flow pre-trained on the FULL GLASS fwhm4 suite on top of a frozen hybrid-encoder source.
    No max_trainval_cosmos (full suite); run_evaluation=False (skip the post-training MCMC on GLASS).

    whiten_k: if set, enable per-source embedding whitening + PCA truncation to k dims. The whitener
    is fit ONCE on THIS pretrain run's train split and persisted (datasets/whitener.pt); the finetune
    + eval RESOLVE and REUSE it (never refit) — the research Finding-C3 fix. k < emb_dim truncates
    (16-D base encoder), k == emb_dim is pure-whiten (the 6-D z6 encoder)."""
    c = {
        "data_patterns": data_patterns,
        "eb_map_variant": eb_variant,
        "dataset_quantities": [],            # overwritten from the source encoder at runtime
        "latent_dim": 8,
        "epochs": epochs,
        "batch_size": 128,
        "lr": 0.001,
        "flow_kwargs": {"hidden_features": 64},
        "project": "gower-finetuning",
        "cosmo_param_names": _COSMO_9,
        "inference_mode": "nle",
        "repeat_indices": [0],
        "match_num_cosmo": False,
        "scale_embeddings": False,
        "run_evaluation": False,
    }
    if whiten_k:
        c["whiten_embeddings"] = {"k": int(whiten_k)}
    return c


def _nle_finetune(pretrain_exp, ensemble_repeats=1, whiten_k=None, warmstart_max_gap_nats=None):
    """NLE flow fine-tuned on the prebaked gower fwhm4 store; loads the Stage-A flow from
    checkpoints/<pretrain_exp>/ via load_pretrained_flow. max_trainval_cosmos=[80] (single point);
    run_evaluation=True (MCMC eval on CORES64). ensemble_repeats>1 trains N flow members
    (ncosmo80_0_ens{j}) all warm-started from the SAME Stage-A flow, diverging only via ensemble_seed;
    the train-time deferred eval then writes ensemble_evaluation_results / ensemble_tarp json.

    whiten_k MUST match the pretrain_exp's whiten_k (the flow ckpt shapes differ per k). The finetune
    resolves the whitener from checkpoints/<pretrain_exp>/ (same run folder as the flow ckpt)."""
    c = _nle_pretrain(_GOWER_NLA_M_DATA_FWHM4, _GOWER_EB_VARIANT_FWHM4, whiten_k=whiten_k)
    c["epochs"] = 50
    c["lr"] = 0.00008   # 1/5 of the previous 4e-4 (user 2026-07-06: gentler NLE finetune, ALL variates)
    c["load_pretrained_flow"] = True
    c["pretrained_band_ckpt_path"] = _NLE_PRETRAIN_CKPT.format(exp=pretrain_exp)
    c["max_trainval_cosmos"] = [80]
    c["train_frac"] = 0.7
    c["val_frac"] = 0.2
    c["run_evaluation"] = True
    c["ensemble_repeats"] = ensemble_repeats
    # Override the warm-start regression guard threshold (guard-c, embeddings_utils.py). Default 12
    # nats is calibrated on ~6-D whitened prototypes with pretrain best ~0. A PURE-WHITEN of a
    # RANK-DEFICIENT summary inflates the genuine ep0 gap: near-null PCs are pretrain-fit sharply
    # (very negative best) AND amplify the GLASS->Gower shift (divide by a tiny sqrt-eigenvalue), so
    # ep0 Gower NLL is legitimately high without the warm start being broken. z8 (8-D, ~2.6 eff-rank,
    # ~4 near-null PCs) genuinely gaps ~16 nats vs z6's ~5 — see whitened-nle-z6-z8 log 2026-07-03.
    if warmstart_max_gap_nats is not None:
        c["whiten_warmstart_max_gap_nats"] = warmstart_max_gap_nats
    return c


# base and vicreg pretrain dicts are identical bodies; the source encoder differs at the CLI and the
# distinct names give separate checkpoint dirs (checkpoints/glass_nle_pretrain_nla_m_{base,vicreg}/).
kids_legacy_experiments["glass_nle_pretrain_nla_m_base"] = _nle_pretrain()
kids_legacy_experiments["glass_nle_pretrain_nla_m_vicreg"] = _nle_pretrain()
kids_legacy_experiments["gower_nle_finetune_nla_m_base"] = _nle_finetune("glass_nle_pretrain_nla_m_base", ensemble_repeats=5)
kids_legacy_experiments["gower_nle_finetune_nla_m_vicreg"] = _nle_finetune("glass_nle_pretrain_nla_m_vicreg", ensemble_repeats=5)


# --- WHITENED NLE chain (research nle-overconfidence-research: Finding-C3 fix promoted to prod) ---
# WhitenPCAScaler (standardise -> PCA rotate -> keep top-k -> unit-variance per PC), fit ONCE on the
# GLASS pretrain train split and persisted (datasets/whitener.pt), then RESOLVED + REUSED unchanged
# by the finetune + eval (never refit). k=6 is the prototype's SBC-filter winner (D_white6). The
# CRITICAL pairing constraint: each finetune's pretrain_exp MUST be the matching _white6/_z6 pretrain
# (the flow ckpt event-dim = k differs from the unwhitened k=16; pointing at the wrong pretrain would
# trip guard (a), a hard error). The source encoder is passed at the train_embeddings.py CLI.
#
# base_white6: the EXISTING 16-D base encoder (kids_legacy_hybrid_nla_m_lmin50_fwhm4) + TRUNCATION to
# the top-6 PCs (16 -> 6). Ready to launch now.
kids_legacy_experiments["glass_nle_pretrain_nla_m_base_white6"] = _nle_pretrain(whiten_k=6, epochs=150)
kids_legacy_experiments["gower_nle_finetune_nla_m_base_white6"] = _nle_finetune(
    "glass_nle_pretrain_nla_m_base_white6", ensemble_repeats=5, whiten_k=6)

# z6: the FUTURE 6-D-summary hybrid encoder (kids_legacy_hybrid_nla_m_lmin50_fwhm4_z6) as the source;
# whitening k=6 == PURE-WHITEN mode (no truncation) on the already-6-D summary z.
# ⚠️ WAITS ON the z6 ENCODER TRAINING JOB 1311931 finishing — the frozen z6 hybrid checkpoint is the
# required source encoder. Do NOT launch these until that job completes and the checkpoint exists.
kids_legacy_experiments["glass_nle_pretrain_nla_m_z6"] = _nle_pretrain(whiten_k=6, epochs=150)
kids_legacy_experiments["gower_nle_finetune_nla_m_z6"] = _nle_finetune(
    "glass_nle_pretrain_nla_m_z6", ensemble_repeats=5, whiten_k=6)


# --- z6-r2 / z8-r0 whitened NLE (task whitened-nle-z6-z8, 2026-07-03) ---------------------------
# Same PCA pure-whiten chain as the _z6 block above, but for the SPECIFIC landed encoder repeats the
# user requested (z6 repeat 2, z8 repeat 0), each with BOTH a single-model (ens=1) and a 5-member
# (ens=5) finetune. `embed` ignores --repeat-indices, so the repeat is BAKED into the config dict:
# match_num_cosmo=False => repeat_match "_{i}" resolves BOTH the source encoder's repeat-i ckpt
# (pretrain_ncosmoNone_i) AND trains the flow as repeat i. whiten_k = the encoder summary dim => PCA
# PURE-WHITEN, k=full, NO truncation (user decision 2026-07-03): z6->k=6, z8->k=8. The whitener is
# fit ONCE on the GLASS pretrain train split + persisted, reused by finetune/eval (C3-safe). Source
# encoders on the train_embeddings.py CLI: --sources kids_legacy_hybrid_nla_m_lmin50_fwhm4_{z6,z8}.
def _nle_bake_repeat(c, i):
    """Bake a single repeat index into an NLE config dict (each _nle_* factory returns a fresh dict)."""
    c["repeat_indices"] = [int(i)]
    return c


# z6, repeat 2 — 6-D summary, pure-whiten k=6
kids_legacy_experiments["glass_nle_pretrain_nla_m_z6_r2"] = _nle_bake_repeat(_nle_pretrain(whiten_k=6, epochs=150), 2)
kids_legacy_experiments["gower_nle_finetune_nla_m_z6_r2"] = _nle_bake_repeat(
    _nle_finetune("glass_nle_pretrain_nla_m_z6_r2", ensemble_repeats=1, whiten_k=6), 2)
kids_legacy_experiments["gower_nle_finetune_nla_m_z6_r2_ens5"] = _nle_bake_repeat(
    _nle_finetune("glass_nle_pretrain_nla_m_z6_r2", ensemble_repeats=5, whiten_k=6), 2)

# z8, repeat 0 — 8-D summary, pure-whiten k=8 (NEW z8 NLE chain; k=8 pure-whiten is new vs the k=6 chains)
kids_legacy_experiments["glass_nle_pretrain_nla_m_z8_r0"] = _nle_bake_repeat(_nle_pretrain(whiten_k=8, epochs=150), 0)
# z8 pure-whiten (k=8) is rank-deficient (~2.6 eff-rank, ~4 near-null PCs) => genuine ep0 warm-start
# gap ~16 nats trips the default guard-c (12). Raise the threshold to 22 (still well below a truly
# broken 8-D warm start); the gap is confirmed genuine (whitener reused OK, guard-a shape-check passed).
kids_legacy_experiments["gower_nle_finetune_nla_m_z8_r0"] = _nle_bake_repeat(
    _nle_finetune("glass_nle_pretrain_nla_m_z8_r0", ensemble_repeats=1, whiten_k=8,
                  warmstart_max_gap_nats=22.0), 0)
kids_legacy_experiments["gower_nle_finetune_nla_m_z8_r0_ens5"] = _nle_bake_repeat(
    _nle_finetune("glass_nle_pretrain_nla_m_z8_r0", ensemble_repeats=5, whiten_k=8,
                  warmstart_max_gap_nats=22.0), 0)


# --- PRODUCTION main-variate NLE: all 5 repeats × 9-member ensemble (z8, pure-whiten k=8) ---------
# Per repeat r: GLASS pretrain (glass_nle_pretrain_nla_m_z8_r{r}, v100) -> Gower ens9 finetune
# (gower_nle_finetune_nla_m_z8_r{r}_ens9, CORES64 MCMC eval). MAIN-variate split: 300 train/val cosmos
# (80/20 split), 200 fixed-test ids held out (config/fixed_test_sets/gower_test_ids.json). lr 8e-5 (1/5).
# Source encoder on the embed CLI: --sources kids_legacy_hybrid_nla_m_lmin50_fwhm4_z8.
def _register_main_nle_z8():
    for r in range(5):
        kids_legacy_experiments[f"glass_nle_pretrain_nla_m_z8_r{r}"] = _nle_bake_repeat(
            _nle_pretrain(whiten_k=8, epochs=150), r)
        ft = _nle_finetune(f"glass_nle_pretrain_nla_m_z8_r{r}", ensemble_repeats=9,
                           whiten_k=8, warmstart_max_gap_nats=22.0)
        ft["max_trainval_cosmos"] = [300]
        ft["train_frac"] = 0.8
        ft["val_frac"] = 0.2
        ft["test_frac"] = 0.0   # test = fixed 200 ids; train+val+test must sum to 1.0 (split_by_cosmology)
        ft["fixed_test_sim_ids"] = "config/fixed_test_sets/gower_test_ids.json"
        kids_legacy_experiments[f"gower_nle_finetune_nla_m_z8_r{r}_ens9"] = _nle_bake_repeat(ft, r)


_register_main_nle_z8()


# --- NPE whole-model fine-tune (Stage 3a) — the NPE arm of the NPE-vs-NLE comparison ------------
# Fine-tune the WHOLE GLASS-pretrained hybrid (encoder + NPE flow) on the prebaked gower fwhm4 store,
# via train.py (NOT train_embeddings.py). build_model loads the whole model from `checkpoint_path`,
# which train_model resolves PER-REPEAT: match_num_cosmo=False => repeat_match "_{i}" =>
# get_best_checkpoint(<glass hybrid ckpt dir>, "_0") -> pretrain_ncosmoNone_0/checkpoint-*.ckpt
# (the compile-trained ckpt's `_orig_mod.` keys are aligned by npe.py load_from_checkpoint). Cloned
# from _hybrid_lmin50() so the architecture EXACTLY matches the checkpoint. repeat_indices=[0]; L40s
# (maps OOM v100). max_trainval_cosmos=[80] MATCHES the NLE finetune for a fair NPE-vs-NLE compare.
# The vicreg variant just loads the vicreg-encoder hybrid ckpt as init and fine-tunes as plain NPE
# (no VICReg loss at finetune). Shares Phase-5b's prebaked gower store with the NLE finetune.
_GLASS_HYBRID_CKPT = "/share/gpu5/asaoulis/transfer_models/checkpoints/{exp}/"


def _npe_finetune_lmin50(checkpoint_dir):
    c = _hybrid_lmin50()                              # exact GLASS-hybrid architecture + l40s tuning + ml_perf
    c["data_patterns"] = _GOWER_NLA_M_DATA_FWHM4
    c["eb_map_variant"] = _GOWER_EB_VARIANT_FWHM4
    c["checkpoint_path"] = checkpoint_dir             # whole-model load, resolved per-repeat to ncosmoNone_0
    c.pop("pretrained_band_ckpt_path", None)          # ignored when checkpoint_path is set (band comes with it)
    c["freeze_band"] = False                          # whole-model fine-tune (band included)
    c["epochs"] = 25
    c["lr"] = 1e-5
    c["batch_size"] = 128
    c["scheduler_type"] = "exp"
    c["scheduler_kwargs"] = {"warmup": 0}
    c["max_trainval_cosmos"] = [80]                   # match the NLE finetune (fair NPE-vs-NLE)
    c["train_frac"] = 0.65
    c["val_frac"] = 0.25
    c["match_num_cosmo"] = False
    c["repeat_indices"] = [0]
    c.pop("repeats", None)                            # repeat_indices overrides repeats
    c["project"] = "gower-finetuning"
    return c


kids_legacy_experiments["gower_npe_finetune_nla_m_base"] = _npe_finetune_lmin50(
    _GLASS_HYBRID_CKPT.format(exp="kids_legacy_hybrid_nla_m_lmin50_fwhm4"))
kids_legacy_experiments["gower_npe_finetune_nla_m_vicreg"] = _npe_finetune_lmin50(
    _GLASS_HYBRID_CKPT.format(exp="kids_legacy_hybrid_nla_m_lmin50_fwhm4_vicreg"))


def _npe_finetune_z8(checkpoint_dir):
    """PRODUCTION NPE 9-member ensemble finetune, MAIN variate — z8 architecture (matches the z8
    foundation ckpt). Whole-model load (encoder + NPE flow) from the foundation, finetune all. Split:
    300 train/val cosmos (80/20), 200 fixed-test ids held out. ALL 5 repeats (one job per repeat)."""
    c = _hybrid_lmin50_z8()                            # z8 arch EXACTLY matches the z8 foundation ckpt
    c["data_patterns"] = _GOWER_NLA_M_DATA_FWHM4
    c["eb_map_variant"] = _GOWER_EB_VARIANT_FWHM4
    c["checkpoint_path"] = checkpoint_dir              # whole-model load, per-repeat "_{i}" (_orig_mod-safe)
    c.pop("pretrained_band_ckpt_path", None)
    c["freeze_band"] = False
    c["epochs"] = 25
    c["lr"] = 1e-5
    c["batch_size"] = 128
    c["scheduler_type"] = "exp"
    c["scheduler_kwargs"] = {"warmup": 0}
    c["ensemble_repeats"] = 9
    c["max_trainval_cosmos"] = [300]
    c["train_frac"] = 0.8
    c["val_frac"] = 0.2
    c["test_frac"] = 0.0   # test set comes from the fixed 200 ids; fracs must sum to 1.0 (split_by_cosmology)
    c["fixed_test_sim_ids"] = "config/fixed_test_sets/gower_test_ids.json"
    c["match_num_cosmo"] = False
    c["repeat_indices"] = [0, 1, 2, 3, 4]
    c.pop("repeats", None)
    c["project"] = "gower-finetuning"
    return c


kids_legacy_experiments["gower_npe_finetune_nla_m_z8"] = _npe_finetune_z8(
    _GLASS_HYBRID_CKPT.format(exp="kids_legacy_hybrid_nla_m_lmin50_fwhm4_z8"))

# NPE finetune RE-RUN vs the now-FROZEN vicreg hybrid encoder. The original
# gower_npe_finetune_nla_m_vicreg loaded an IN-FLIGHT encoder ckpt (epoch-58) because the encoder was
# still training when it ran; the frozen best is epoch-89. This _v2 produces a CLEAN Gower-finetuned
# vicreg embedding net, used as the SOURCE encoder for the 1-head NLE test below.
kids_legacy_experiments["gower_npe_finetune_nla_m_vicreg_v2"] = _npe_finetune_lmin50(
    _GLASS_HYBRID_CKPT.format(exp="kids_legacy_hybrid_nla_m_lmin50_fwhm4_vicreg"))

# 1-head NLE test (SINGLE flow, ensemble_repeats=1): take the pretrained vicreg NLE head
# (glass_nle_pretrain_nla_m_vicreg) and fine-tune it on embeddings produced by the NPE-finetuned vicreg
# embedding net (gower_npe_finetune_nla_m_vicreg_v2) instead of the frozen GLASS hybrid encoder. The
# source encoder is passed at the CLI: `embed --cpu --target ... --sources gower_npe_finetune_nla_m_vicreg_v2`.
# match_num_cosmo=True so the source-encoder lookup searches "ncosmo80_0" (the NPE finetune writes
# finetune_ncosmo80_0, which lacks the "None_0" tag the default resolver needs); the pretrained-FLOW
# lookup is independent (hardcoded None_0 -> glass_nle_pretrain_nla_m_vicreg/ncosmoNone_0), unaffected.
kids_legacy_experiments["gower_nle_finetune_nla_m_vicreg_npesrc"] = _nle_finetune("glass_nle_pretrain_nla_m_vicreg")
kids_legacy_experiments["gower_nle_finetune_nla_m_vicreg_npesrc"]["match_num_cosmo"] = True
