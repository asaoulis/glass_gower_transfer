"""NO-VD production suite — model configs (fwhm4 PRIMARY stack, M1-M6).

The 2026-07-29 variate switch made **no-variable-depth** the MAIN analysis variate: the KiDS-Legacy
VD forward model has an unresolved cosmology physics bug, so every VD-era model is invalid and
retrains here from scratch on the `_novd` stores (task training-runs/production-training-runs,
`models_checklist.md`). What is REUSED is the *recipe*, not the weights: the PreActResNet map
encoder, the whiten-k8 NLE chain, the 5-repeat / ens9 structure, the 300/200 Gower split.

**Why new experiment names (`_novd`, and `_counts` dropped):** checkpoints live at
`{base_path}/checkpoints/{experiment_name}/`, so reusing a counts-era name would mix new runs with
legacy checkpoints and `get_best_checkpoint` could silently resolve the WRONG one. Counts
normalisation is now universal, so it is no longer a name discriminator.

Scope: the fwhm4 PRIMARY stack (M1-M6). The conservative fwhm8 `_cons` stack (Section C of
`models_checklist.md`) is a LATER pass and is deliberately NOT built here (see the TODO at the end).

Merge: `kids_legacy_novd_experiments` is `.update()`-merged into the experiments dict by
train.py / eval.py / train_embeddings.py / .claude/cluster/smoke_test_experiment.py, and by
src/ml/eval/misspec.py:_load_experiment_config (so the M6 misspec base resolves).

Every MAP config carries a de-clustered `_smoke` clone on the fwhm8 single-cosmology LOCAL fixture
(.claude/cluster/smoke_data_nla, E_fwhm8_lmin50_lcut1400 only). The production fwhm4 config
false-fails that fwhm8-only local smoke, so the REAL runs submit with `--skip-smoke` (per the
models_checklist smoke-gate note).
"""
from config.kids_legacy import (
    # shared constants (parameter sets + the NLA/NLA-z a_ia box)
    _COSMO_9, _COSMO_8_NLA, _COSMO_9_NLAZ, _A_IA_NLA_BOX,
    # fwhm8 LOCAL smoke-fixture store + tag (smoke clones only; the harness overrides data_patterns
    # but keeps eb_map_variant, which must be the fixture's fwhm8 tag)
    _NLA_M_DATA as _SMOKE_DATA,
    _EB_VARIANT as _SMOKE_EB_VARIANT,
    # factories to clone
    _band_lmin50, _hybrid_lmin50_z8, _hybrid_lmin50_z8_smoke,
    _encoder_finetune_z8, _encoder_finetune_z8_smoke,
    _nle_pretrain, _nle_finetune, _nle_bake_repeat, _npe_finetune_z8,
)

# --- no-VD data stores (roots match config/kids_legacy.py) -------------------------------------
_GPU5 = "/share/gpu5/asaoulis/transfer_datasets"
_GPU4 = "/share/gpu4/asaoulis/transfer_datasets"
_CKPT = "/share/gpu5/asaoulis/transfer_models/checkpoints"
_EB_VARIANT = "fwhm4_lmin56_lcut1400"

# GLASS pre-training stores. The band (M1) reads bandpowers off the RAW gpu4 store (smoothing-
# independent, no prebake needed); the maps (M2+) read the prebaked f16 gpu5 stores (l40s-local).
_NLA_M_RAW = f"{_GPU4}/glass_mocks_nla_m_novd_counts/output_*.h5"
_NLA_M     = f"{_GPU5}/glass_mocks_nla_m_novd_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5"
_NLA       = f"{_GPU5}/glass_mocks_nla_novd_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5"
_NLA_Z     = f"{_GPU5}/glass_mocks_nla_z_novd_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5"

# Gower fine-tuning stores (prebaked f16 gpu5, fwhm4).
# _GOWER_NLA_M_RAW is the UNBAKED gpu4 source. Map training off it is ~4.5x slower (non-local NFS,
# f64, all smoothing variants on the wire) so it is NOT for production — it exists so a preview can
# run while the prebake is stuck behind our own sims in the CORES64 queue.
_GOWER_NLA_M_RAW = f"{_GPU4}/gower_mocks_nla_m_novd_counts/output_*.h5"
_GOWER_NLA_M = f"{_GPU5}/gower_mocks_nla_m_novd_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5"
_GOWER_NLA   = f"{_GPU5}/gower_mocks_nla_novd_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5"
_GOWER_NLA_Z = f"{_GPU5}/gower_mocks_nla_z_novd_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5"

# Checkpoint dirs (written by M1 band / M2 foundation; consumed by M2/M3/M4/M5).
_BAND_CKPT_DIR = f"{_CKPT}/kids_legacy_band_nla_m_novd/"
# PRODUCTION FOUNDATION = the PreActResNet hybrid (z8_resnet). All downstream (M3 NPE-finetune, M4
# NLE pretrain source, M5a encoder warm-start) sources this.
_FOUNDATION_CKPT = f"{_CKPT}/kids_legacy_hybrid_nla_m_novd_z8_resnet/"

# Production map-encoder kwargs — the PreActResNet backbone the foundation (z8_resnet) is trained
# with. Any config that REBUILDS the foundation arch to load its weights (M3 NPE whole-model load,
# M5a encoder warm-start) MUST set model_kwargs['map_kwargs']=_RESNET_MAPKW, or the state-dict load
# silently mismatches the map encoder (the default arch is the old UNet). Byte-identical to the
# counts-era production arch — do not "improve" it: the plain ResNet is what broke the -4.4 wall.
_RESNET_MAPKW = {"encoder_type": "preact_resnet", "patch_conditioning": None,
                 "pool_types": ("avg", "gem"),
                 "stage_channels": (32, 64, 128, 256, 256), "blocks_per_stage": 3}

_GOWER_TEST_IDS = "config/fixed_test_sets/gower_test_ids.json"
# Sub-variate (M5c) split of the SAME 200 fixed-test ids: first-100 = train/val pool,
# last-100 = held-out test. Halves are taken in FILE ORDER (the parent lock is stored in a
# maximally-separated order) and are verified balanced in omega_m/sigma_8/w. Only the last-100
# file is wired in: the sub-variate Gower stores hold exactly the 200 parent ids, so forcing the
# last-100 into test leaves precisely the first-100 for train/val.
_GOWER_TEST_IDS_LAST100 = "config/fixed_test_sets/gower_test_ids_last100.json"

kids_legacy_novd_experiments = {}


# === M1  Stage-I bandpower MLP (5 repeats; bandpowers off the RAW gpu4 no-VD store) =============
def _band_novd():
    """Stage-I bandpower encoder. Reads mixed_bandpowers straight off the RAW gpu4 store (bandpowers
    are smoothing-independent -> NO prebake dependency), so this is the FIRST model that can run
    once G1 lands. 5 repeats -> checkpoints/kids_legacy_band_nla_m_novd/pretrain_ncosmoNone_{0..4}/;
    each M2 foundation repeat i then loads band i FROZEN (pretrained_band_match_string '_{i}').
    Runs on v100 (bandpowers only, no maps). Submit split across two jobs:
      train --exp kids_legacy_band_nla_m_novd --gpu v100 --repeat-indices 0,1,2   (and 3,4)."""
    c = _band_lmin50()
    c["data_patterns"] = _NLA_M_RAW
    c.pop("repeats", None)
    c["repeat_indices"] = [0, 1, 2, 3, 4]
    return c


kids_legacy_novd_experiments["kids_legacy_band_nla_m_novd"] = _band_novd()


# === M2  foundation z8 hybrid (base + PRODUCTION resnet; 5 repeats, frozen per-repeat band) =====
def _hybrid_novd_z8():
    """z8-summary hybrid base on the no-VD nla_m fwhm4 store; loads the FROZEN per-repeat Stage-I
    band (repeat i -> band i). The 8-D whitened summary is the foundation encoder for ALL
    downstream (M3/M4/M5). NB this UNet-encoder base is the CONTROL, not production — it plateaus
    at the 2-pt band level (~-4.4). Production is `_z8_resnet` below."""
    c = _hybrid_lmin50_z8()                            # z8 arch + l40s tuning + ml_perf
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
    """De-clustered fwhm8-local smoke clone exercising the SAME model kwargs (from-scratch band)."""
    c = _hybrid_lmin50_z8_smoke()
    if mk_extra:
        c["model_kwargs"] = {**c["model_kwargs"], **mk_extra}
    for k, v in (top_extra or {}).items():
        c[k] = v
    return c


kids_legacy_novd_experiments["kids_legacy_hybrid_nla_m_novd_z8"] = _hybrid_novd_z8()
kids_legacy_novd_experiments["kids_legacy_hybrid_nla_m_novd_z8_smoke"] = _hybrid_lmin50_z8_smoke()

# --- ⭐ PRODUCTION FOUNDATION (M2) --------------------------------------------------------------
# Plain pre-activation ResNet map encoder. This is THE production recipe (user-confirmed
# 2026-07-20, memory `counts-hybrid-resnet-recipe`): it broke the hard -4.4 UNet wall to a durable
# -5.2..-5.3 GLASS val and is the Gower-downstream leader. One stage, no head redesign, no
# regulariser. Escape happens ~ep16-32 in roughly 3/5 seeds, so expect to re-run repeat indices
# that fail to break the barrier (pass mark: val NLL < -4.5).
kids_legacy_novd_experiments["kids_legacy_hybrid_nla_m_novd_z8_resnet"] = _hybrid_novd_z8_variant(
    mk_extra={"map_kwargs": _RESNET_MAPKW})


kids_legacy_novd_experiments["kids_legacy_hybrid_nla_m_novd_z8_resnet_smoke"] = \
    _hybrid_novd_z8_variant_smoke(mk_extra={"map_kwargs": _RESNET_MAPKW})

# --- RESCUE-ONLY variant (do NOT blanket-apply) -------------------------------------------------
# Asymmetric spectral decoupling: L2 on the band-only readout component. In the counts era this
# RESCUED a permanently-stuck seed (r1 -> -5.21) but it TAXES/KILLS healthy seeds. Use it only to
# re-run a specific repeat index that has repeatedly failed to escape, never as the default.
kids_legacy_novd_experiments["kids_legacy_hybrid_nla_m_novd_z8_resnet_sdband"] = \
    _hybrid_novd_z8_variant(mk_extra={"map_kwargs": _RESNET_MAPKW, "sd_band_coeff": 1e-2},
                            repeat_indices=(0,))
kids_legacy_novd_experiments["kids_legacy_hybrid_nla_m_novd_z8_resnet_sdband_smoke"] = \
    _hybrid_novd_z8_variant_smoke(mk_extra={"map_kwargs": _RESNET_MAPKW, "sd_band_coeff": 1e-2})


# === M3  main-variate Gower NPE 9-member ensemble finetune (5 repeats) ==========================
# WHOLE-MODEL load (encoder + NPE flow) from the resnet foundation, so the map encoder MUST be
# rebuilt as the resnet backbone (map_kwargs=_RESNET_MAPKW) or the state-dict load mismatches.
_gower_npe_ft_novd = _npe_finetune_z8(_FOUNDATION_CKPT, data_patterns=_GOWER_NLA_M,
                                      eb_variant=_EB_VARIANT)
_gower_npe_ft_novd["model_kwargs"] = {**_gower_npe_ft_novd["model_kwargs"], "map_kwargs": _RESNET_MAPKW}
kids_legacy_novd_experiments["gower_npe_finetune_nla_m_novd_z8"] = _gower_npe_ft_novd


# === M3-EARLY  premature 5-member NPE preview on the r4 foundation (user request 2026-08-04) =====
# GPU-side twin of the M4b-EARLY NLE preview: same store, same r4 foundation, same "use whatever has
# landed" split policy, but a `train.py` job on an l40s instead of a CORES64 embeddings job — so it
# does not queue behind our own sims. Only 10 epochs x 5 members, so it is cheap.
#
# ⚠️ PRIOR CAVEAT: NPE learns p(theta|x) UNDER THE SIMULATION PRIOR. NPELightningModule.generate_samples
# builds its posterior with no prior argument, so these samples are NOT reweightable to the KiDS
# analytic-S8 prior the way the NLE ensemble's MCMC is — NPE corner plots from this run are under the
# Gower training prior. Reweighting would need a p_target/p_train importance step that does not exist
# in the codebase today.
def _register_early_npe_ens5_preview_r4():
    c = _npe_finetune_z8(_FOUNDATION_CKPT, data_patterns=_GOWER_NLA_M, eb_variant=_EB_VARIANT)
    c["model_kwargs"] = {**c["model_kwargs"], "map_kwargs": _RESNET_MAPKW}
    c["ensemble_repeats"] = 5            # preview size (production is 9)
    c.pop("max_trainval_cosmos", None)   # default None => all on-disk non-test cosmologies
    c["repeat_indices"] = [4]            # the ONE finished foundation repeat, as for M4b-EARLY
    kids_legacy_novd_experiments["gower_npe_finetune_nla_m_novd_z8_r4_ens5_early"] = c


_register_early_npe_ens5_preview_r4()


# === M3-EARLY-RAW  bake-independent twin: identical preview, reading the UNBAKED gpu4 store ======
# Insurance against the prebake staying queued. Both prebake submits (CORES64 + CORES40) sat at
# PENDING(Priority) — our own two Gower sims hold 16 CORES64 nodes and depress our fairshare — so a
# preview that needs NO bake can start immediately on a GPU node, where we do get scheduled.
# Cost of skipping the bake: ~4.5x slower map reads (measured 30 vs 135 smp/s), so the cosmology
# budget is cut to 175 to keep 5 members x 10 epochs inside the deadline. Everything else — r4
# foundation, resnet arch, 200-id test lock — is identical to the baked twin, so the two runs are
# directly comparable and whichever finishes first is a usable answer.
def _register_early_npe_ens5_preview_r4_raw():
    c = _npe_finetune_z8(_FOUNDATION_CKPT, data_patterns=_GOWER_NLA_M_RAW, eb_variant=_EB_VARIANT)
    c["model_kwargs"] = {**c["model_kwargs"], "map_kwargs": _RESNET_MAPKW}
    c["ensemble_repeats"] = 5
    c["max_trainval_cosmos"] = [175]   # I/O-bound on raw gpu4 => smaller pool, not the full set
    c["repeat_indices"] = [4]
    kids_legacy_novd_experiments["gower_npe_finetune_nla_m_novd_z8_r4_ens5_early_raw"] = c


_register_early_npe_ens5_preview_r4_raw()


def _npe_finetune_novd_z8_smoke():
    """De-clustered LOCAL smoke of gower_npe_finetune_nla_m_novd_z8: from-scratch
    (checkpoint_path=None, from-scratch band), fwhm8 local fixture, single-cosmology-safe (no cosmo
    cap / fixed-test lock, ensemble_repeats=1), few epochs -> proves the config BUILDS + finite loss."""
    c = _npe_finetune_z8(_FOUNDATION_CKPT, data_patterns=_SMOKE_DATA, eb_variant=_SMOKE_EB_VARIANT)
    c["model_kwargs"] = {**c["model_kwargs"], "map_kwargs": _RESNET_MAPKW}  # exercise the real resnet arch
    c["checkpoint_path"] = None
    c.pop("pretrained_band_ckpt_path", None)
    c["freeze_band"] = False
    c["ensemble_repeats"] = 1
    c.pop("max_trainval_cosmos", None)
    c.pop("fixed_test_sim_ids", None)
    # The REAL run gets its test set from the fixed-id lock (test_frac=0.0); the smoke has no lock, so
    # it needs a NON-empty test split — on_validation_epoch_end's compute_avg_log_prob reduces over
    # the TEST loader and torch.cat's an empty list otherwise. Fracs must sum to 1.0.
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


# === M4  main-variate NLE chain: 5x (GLASS pretrain -> Gower ens9 finetune), z8 pure-whiten k=8 ==
# Per repeat r: GLASS pretrain (glass_nle_pretrain_nla_m_novd_z8_r{r}, v100) -> Gower ens9 finetune
# (gower_nle_finetune_nla_m_novd_z8_r{r}_ens9, CORES64 MCMC eval). MAIN split: 300 train/val
# (80/20), 200 fixed-test ids held out. Source encoder on the embed CLI (PRODUCTION = the resnet
# foundation): --sources kids_legacy_hybrid_nla_m_novd_z8_resnet.
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
        ft["test_frac"] = 0.0   # test = fixed 200 ids; fracs must sum to 1.0 (split_by_cosmology)
        ft["fixed_test_sim_ids"] = _GOWER_TEST_IDS
        kids_legacy_novd_experiments[f"gower_nle_finetune_nla_m_novd_z8_r{r}_ens9"] = _nle_bake_repeat(ft, r)


_register_main_nle_novd_z8()


# === M4b-EARLY  premature 5-member preview of the r4 NLE chain (user request 2026-08-04) ========
# PURPOSE: get end-to-end downstream numbers (ensemble eval + TARP + KiDS-prior posterior samples)
# out of the ONE M4a head that has finished (r4, best val -5.6765) BEFORE the Gower store and the
# other four heads are ready. It is a PREVIEW, not production: production stays
# `gower_nle_finetune_nla_m_novd_z8_r4_ens9` at max_trainval_cosmos=[300].
#
# Two deliberate differences from the production entry, both forced by the incomplete store:
#  1. ensemble_repeats=5 (not 9) — cheaper, and enough members for a meaningful ensemble spread.
#  2. max_trainval_cosmos is left at the default None = "every on-disk cosmology that is not in the
#     locked 200-id test set". A hard [300] would trip split_by_cosmology's
#     `Requested max_trainval_cosmos=300 but only N available` ValueError the moment the store has
#     fewer than 300 non-test cosmologies — which is exactly the situation while S1 is still
#     filling. None can never hard-fail and uses whatever has landed.
# The 200-id test lock is UNCHANGED from production, so the held-out set is the same one the
# production chain will use (intersected with what is on disk) and the preview's numbers stay
# comparable to the eventual production run.
def _register_early_nle_ens5_preview_r4():
    ft = _nle_finetune("glass_nle_pretrain_nla_m_novd_z8_r4", ensemble_repeats=5,
                       whiten_k=8, warmstart_max_gap_nats=22.0,
                       gower_data=_GOWER_NLA_M, gower_eb=_EB_VARIANT)
    ft.pop("max_trainval_cosmos", None)   # default None => all on-disk non-test cosmologies
    ft["train_frac"] = 0.8
    ft["val_frac"] = 0.2
    ft["test_frac"] = 0.0   # test = fixed 200 ids; fracs must sum to 1.0 (split_by_cosmology)
    ft["fixed_test_sim_ids"] = _GOWER_TEST_IDS
    kids_legacy_novd_experiments["gower_nle_finetune_nla_m_novd_z8_r4_ens5_early"] = \
        _nle_bake_repeat(ft, 4)


_register_early_nle_ens5_preview_r4()


# === M5  sub-variate chains {nla, nla_z}: encoder-finetune (M5a) + NLE chain (M5b/M5c) ==========
# Per S: warm-start the no-VD foundation ENCODER onto the sub-variate GLASS suite (M5a), then the
# NLE chain (M5b pretrain -> M5c Gower ens9 finetune). Both carry the a_ia~U[-6,6] box at EVERY
# stage; theta set per variate (nla 8-D a_ia; nla_z 9-D a_ia+b_z).
#
# NB the counts-era `no_vd` sub-variate slot is GONE — no-VD is now the MAIN variate (M2/M3/M4).
#
# _SUB_VARIATES[S] = (glass_store, gower_store, cosmo_param_names, preset_overrides, smoke_cosmo)
# smoke_cosmo differs from cosmo where the LOCAL fixture lacks a param: nla_z's real b_z is not in
# the fwhm8 NLA-M fixture, so its smoke uses _COSMO_9 (a_ia+b_ia) for the same 9-D flow shape.
_SUB_VARIATES = {
    "nla":   (_NLA,   _GOWER_NLA,   _COSMO_8_NLA,  _A_IA_NLA_BOX, _COSMO_8_NLA),
    "nla_z": (_NLA_Z, _GOWER_NLA_Z, _COSMO_9_NLAZ, _A_IA_NLA_BOX, _COSMO_9),
}


def _encoder_finetune_novd(glass_data, cosmo, preset):
    c = _encoder_finetune_z8(glass_data, _EB_VARIANT, cosmo, repeat_indices=(0, 1, 2, 3, 4),
                             preset_overrides=preset)
    c["pretrained_embedding_ckpt_path"] = _FOUNDATION_CKPT   # the no-VD resnet foundation
    # Warm-start LOADS the resnet foundation encoder -> rebuild the resnet map backbone or the
    # partial load silently skips it (leaving a random UNet map encoder). Same fix as M3.
    c["model_kwargs"] = {**c["model_kwargs"], "map_kwargs": _RESNET_MAPKW}
    return c


def _register_sub_variate(S, glass_data, gower_data, cosmo, preset, smoke_cosmo):
    # M5a: encoder finetune (+ de-clustered smoke via the kids_legacy encoder-smoke factory).
    kids_legacy_novd_experiments[f"glass_encoder_finetune_{S}_novd_z8"] = _encoder_finetune_novd(
        glass_data, cosmo, preset)
    _enc_smoke = _encoder_finetune_z8_smoke(smoke_cosmo)
    _enc_smoke["model_kwargs"] = {**_enc_smoke["model_kwargs"], "map_kwargs": _RESNET_MAPKW}
    kids_legacy_novd_experiments[f"glass_encoder_finetune_{S}_novd_z8_smoke"] = _enc_smoke
    # M5b/M5c: NLE pretrain -> Gower ens9 finetune per repeat. SUB split: 100 cosmos, 70/30.
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
        # Sub-variate split: the S2/S3 Gower stores are `--gower-sim-set fixed_test` = EXACTLY the
        # 200 parent gower_test_ids. Locking the PARENT file here would force all 200 into test,
        # leaving 0 train/val — and `_resolve_forced_test_cosmos` then hits its
        # `n_present - forced < 1` guard and SILENTLY FALLS BACK TO A NORMAL RANDOM SPLIT, i.e. the
        # "held-out" test would not be held out at all and would share cosmologies with train/val.
        # So M5c locks the LAST-100 as test, which leaves exactly the first-100 for train/val
        # (max_trainval_cosmos=[100], 70/30). Halves verified balanced in omega_m/sigma_8/w.
        ft["fixed_test_sim_ids"] = _GOWER_TEST_IDS_LAST100
        kids_legacy_novd_experiments[f"gower_nle_finetune_{S}_novd_z8_r{r}_ens9"] = _nle_bake_repeat(ft, r)


for _S, (_g, _gow, _cos, _pre, _smk) in _SUB_VARIATES.items():
    _register_sub_variate(_S, _g, _gow, _cos, _pre, _smk)


# TODO Section C conservative (fwhm8) stack — mirror M2-M6 with `_cons` names + fwhm8 stores
# (_EB_VARIANT_CONS="fwhm8_lmin56_lcut1024", *_fwhm8 map stores, _FOUNDATION_CONS_CKPT). Band M1 is
# REUSED (bandpowers are smoothing-independent). Deliberately deferred (LATER pass).


# === SCRATCH (task training-runs/scratch-gower-runs) — GOWER-ONLY single-fidelity probe =========
# WHY. The production stack shows a persistent model misspecification: the foundation is extremely
# sensitive to var(E) (the per-pixel std of the E-mode maps) as a function of cosmology, and a ~1 %
# map-variance offset under a galaxy-bias shift moves the posterior by several sigma. That may be
# real physics — or it may be an artefact of the LOG-NORMAL GLASS mocks, which plausibly carry
# little information beyond the 2-pt function, forcing the CNN onto such extreme statistics.
# This probe removes GLASS entirely: the SAME two-stage recipe (Stage-I band -> frozen band +
# PreActResNet map encoder) trained single-fidelity on the Gower Street N-body mocks. If the
# Gower-trained hybrid shows the same var(E) reliance, the effect is not a log-normal artefact.
#
# DIAGNOSTIC, NOT PRODUCTION. These deliberately read the SUPERSEDED `_novd_counts` Gower store
# (not the dual-norm regeneration, which has produced GLASS rows only): holding the normalisation
# fixed at what the GLASS-trained models saw is exactly what makes the comparison like-for-like.
# No production conclusion may rest on these checkpoints.
#
# SPLIT (user-specified 2026-08-06): 450 train / 50 val, "0 test". test_frac=0.0 is NOT usable
# without fixed_test_sim_ids (prepare_data_parameters always builds a test loader, and
# npe.py compute_avg_log_prob then does torch.cat([]) -> RuntimeError in the sanity check), so the
# 5 test cosmologies are taken from the 11 SURPLUS to the requested 500. Arithmetic on the store's
# 511 distinct cosmologies: n_test = round(511*0.01) = 5 -> 506 remain -> max_trainval_cosmos=500
# -> rel_train_frac = 0.891/0.99 = 0.9 exactly -> n_train = 450, n_val = 50. Nothing the user asked
# to train on is withheld. Verified against src/ml/data/data_selection.py:split_by_cosmology.
#
# NB max_trainval_cosmos=[500] (not None) makes match_num_cosmo emit `ncosmo500_{i}`, so the run
# dirs are pretrain_ncosmo500_{0,1,2} for BOTH stages and the hybrid's per-repeat band lookup
# resolves cleanly. Both stages MUST keep identical data_patterns + split keys or the frozen band
# would have been fitted on a different cosmology split than the hybrid it is frozen into.
_GOWER_ONLY_BAND_CKPT_DIR = f"{_CKPT}/gower_only_band_nla_m_novd/"
_GOWER_ONLY_SPLIT = {"max_trainval_cosmos": [500], "train_frac": 0.891,
                     "val_frac": 0.099, "test_frac": 0.01}


def _gower_only_band():
    """Stage-I bandpower MLP trained on GOWER (not GLASS). Reads the same prebaked gpu5 Gower store
    as Stage II — it carries cls_results/full/mixed_bandpowers — so both stages see an IDENTICAL
    cosmology split. 3 repeats -> checkpoints/gower_only_band_nla_m_novd/pretrain_ncosmo500_{0,1,2}/.
    Bandpowers only (no maps) => runs on v100.
      train --exp gower_only_band_nla_m_novd --gpu v100 --repeat-indices 0   (then 1, then 2)"""
    c = _band_lmin50()
    c["data_patterns"] = _GOWER_NLA_M
    c.update(_GOWER_ONLY_SPLIT)
    c.pop("repeats", None)
    c["repeat_indices"] = [0, 1, 2]
    return c


def _gower_only_hybrid_z8_resnet():
    """Stage-II hybrid: FROZEN per-repeat Gower Stage-I band + the PRODUCTION PreActResNet map
    encoder, trained single-fidelity on Gower. Identical architecture/optimiser to the production
    foundation `kids_legacy_hybrid_nla_m_novd_z8_resnet`; the ONLY differences are the data store
    (Gower, not GLASS), the 450/50/5 split, 3 repeats, and the band checkpoint dir.
    No checkpoint_path: this is a from-scratch pretrain, so the run dirs stay `pretrain_*` and
    cannot collide with a `finetune_*` sibling. v100 OOMs on the maps -> l40s (or a100), NEVER v100.
      train --exp gower_only_hybrid_nla_m_novd_z8_resnet --gpu l40s --ncpu 10 --mem-gb 28 \\
            --skip-smoke --repeat-indices 0   (then 1, then 2)"""
    c = _hybrid_lmin50_z8()
    c["data_patterns"] = _GOWER_NLA_M
    c["eb_map_variant"] = _EB_VARIANT
    c["model_kwargs"] = {**c["model_kwargs"], "map_kwargs": _RESNET_MAPKW}
    c["pretrained_band_ckpt_path"] = _GOWER_ONLY_BAND_CKPT_DIR
    # STEP-MATCHED to the GLASS foundation (user directive 2026-08-06). Gower has 450 cosmologies
    # x 80 mocks = 36 000 train samples/epoch = 360 steps at batch 100, vs GLASS's ~81 000/epoch
    # (810 steps). At the inherited 100 epochs this run would see 36 000 optimiser steps against
    # GLASS's 81 000 — a different budget AND a different LR trajectory, since the cyclic schedule
    # is ABSOLUTE-step based (cyclic_period_steps=6000) while warmup is a FRACTION of the total
    # (base.py:151 warmup_frac=0.05; note the config's "warmup"/"min_factor" keys are dead — the
    # code reads warmup_frac/warmup_steps and cyclic_min_factor. Left as-is deliberately: matching
    # the production recipe matters more than fixing an inherited no-op key).
    # 225 epochs x 360 steps = 81 000 steps => IDENTICAL total steps, identical 4 050-step warmup,
    # and 13.5 cyclic cycles, exactly as GLASS. Also necessary, not just tidy: GLASS escape occurs
    # at ep16-32 = 13k-26k steps, which lands at Gower epochs 36-72, so a 100-epoch run would only
    # just be entering the escape window and a null result would be unreadable.
    c["epochs"] = 225
    c.update(_GOWER_ONLY_SPLIT)
    c.pop("repeats", None)
    c["repeat_indices"] = [0, 1, 2]
    return c


kids_legacy_novd_experiments["gower_only_band_nla_m_novd"] = _gower_only_band()
kids_legacy_novd_experiments["gower_only_hybrid_nla_m_novd_z8_resnet"] = _gower_only_hybrid_z8_resnet()

# De-clustered LOCAL smoke clone (fwhm8 fixture, from-scratch band). Carries OUR fracs so the split
# arithmetic is gated too: the 8-cosmology fixture gives round(8*0.01)->max(1,..)=1 test, 7 trainval,
# int(7*0.9)=6 train / 1 val. max_trainval_cosmos is dropped (7 < 500 would raise).
kids_legacy_novd_experiments["gower_only_hybrid_nla_m_novd_z8_resnet_smoke"] = \
    _hybrid_novd_z8_variant_smoke(
        mk_extra={"map_kwargs": _RESNET_MAPKW},
        top_extra={"train_frac": 0.891, "val_frac": 0.099, "test_frac": 0.01,
                   "repeat_indices": [0]})
