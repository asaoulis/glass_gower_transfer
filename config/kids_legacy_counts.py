"""Counts-normalisation production rerun — model configs (fwhm4 PRIMARY stack, M1-M6).

The `--shear-normalization counts` rerun of the full production model checklist (task
training-runs/production-training-runs, `models_checklist.md`). These clone the validated
`config/kids_legacy.py` factories onto the counts-normalised datasets (glass_mocks_*_counts /
gower_mocks_*_counts, prebaked f16 at the fwhm4_lmin56_lcut1400 smoothing tag). The data-pattern
STORE NAMES below are the byte-for-byte contract with `datasets_checklist.md` (D1-D10).

Scope: the fwhm4 PRIMARY stack (M1-M6). The conservative fwhm8 `_cons` stack (Section C of
`models_checklist.md`) is a LATER pass and is deliberately NOT built here (see the TODO at the end).

Merge: `kids_legacy_counts_experiments` is `.update()`-merged into the experiments dict by
train.py / eval.py / train_embeddings.py / .claude/cluster/smoke_test_experiment.py, and by
src/ml/eval/misspec.py:_load_experiment_config (so the M6 misspec base resolves).

Every MAP config carries a de-clustered `_smoke` clone on the fwhm8 single-cosmology LOCAL fixture
(.claude/cluster/smoke_data_nla, E_fwhm8_lmin50_lcut1400 only, cosmo a_ia+b_ia — no b_z). The
production fwhm4 config false-fails that fwhm8-only local smoke, so the REAL runs submit with
--skip-smoke (per the models_checklist smoke-gate note).
"""
from config.kids_legacy import (
    # shared constants (parameter sets + the NLA/NLA-z a_ia box)
    _COSMO_9, _COSMO_8_NLA, _COSMO_9_NLAZ, _A_IA_NLA_BOX,
    # fwhm8 LOCAL smoke-fixture store + tag (smoke clones only; the harness overrides data_patterns
    # but keeps eb_map_variant, which must be the fixture's fwhm8 tag)
    _NLA_M_DATA as _SMOKE_DATA,
    _EB_VARIANT as _SMOKE_EB_VARIANT,
    # mean-norm lmin50 fwhm4 prebaked store (the H1 map-only mean-vs-counts comparison)
    _NLA_M_DATA_LMIN50_FWHM4 as _MEAN_NLA_M_FWHM4,
    _EB_VARIANT_LMIN50_FWHM4 as _MEAN_EB_VARIANT_FWHM4,
    # factories to clone
    _band_lmin50, _hybrid_lmin50_z8, _hybrid_lmin50_z8_smoke,
    _encoder_finetune_z8, _encoder_finetune_z8_smoke,
    _nle_pretrain, _nle_finetune, _nle_bake_repeat, _npe_finetune_z8,
)

# --- counts-normalised data stores (roots match config/kids_legacy.py) --------------------------
_GPU5 = "/share/gpu5/asaoulis/transfer_datasets"
_GPU4 = "/share/gpu4/asaoulis/transfer_datasets"
_CKPT = "/share/gpu5/asaoulis/transfer_models/checkpoints"
_EB_VARIANT = "fwhm4_lmin56_lcut1400"

# GLASS pre-training stores. The band (M1) reads bandpowers off the RAW gpu4 store (smoothing-
# independent, no prebake); the maps (M2+) read the prebaked f16 gpu5 stores.
_NLA_M_RAW = f"{_GPU4}/glass_mocks_nla_m_counts/output_*.h5"
_NLA_M     = f"{_GPU5}/glass_mocks_nla_m_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5"
_NLA       = f"{_GPU5}/glass_mocks_nla_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5"
_NLA_Z     = f"{_GPU5}/glass_mocks_nla_z_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5"
_NOVD      = f"{_GPU5}/glass_mocks_nla_m_novd_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5"

# Gower fine-tuning stores (prebaked f16 gpu5, fwhm4).
_GOWER_NLA_M = f"{_GPU5}/gower_mocks_nla_m_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5"
_GOWER_NLA   = f"{_GPU5}/gower_mocks_nla_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5"
_GOWER_NLA_Z = f"{_GPU5}/gower_mocks_nla_z_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5"
_GOWER_NOVD  = f"{_GPU5}/gower_mocks_nla_m_novd_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5"

# Checkpoint dirs (written by M1 band / M2 foundation; consumed by M2/M3/M4/M5).
_BAND_CKPT_DIR   = f"{_CKPT}/kids_legacy_band_nla_m_counts/"
_FOUNDATION_CKPT = f"{_CKPT}/kids_legacy_hybrid_nla_m_counts_z8/"

_GOWER_TEST_IDS = "config/fixed_test_sets/gower_test_ids.json"

kids_legacy_counts_experiments = {}


# === M1  Stage-I bandpower MLP (5 repeats; bandpowers off the RAW gpu4 counts store) ============
def _band_counts():
    c = _band_lmin50()
    c["data_patterns"] = _NLA_M_RAW
    c.pop("repeats", None)
    c["repeat_indices"] = [0, 1, 2, 3, 4]
    return c


kids_legacy_counts_experiments["kids_legacy_band_nla_m_counts"] = _band_counts()


# === M2  foundation z8 hybrid (5 repeats; loads the frozen per-repeat counts band) ==============
def _hybrid_counts_z8():
    """z8-summary foundation on the counts nla_m fwhm4 store; loads the FROZEN per-repeat Stage-I
    counts band (repeat i -> band i via pretrained_band_match_string '_{i}'), same as the lmin50
    foundation. 8-D whitened summary = the foundation encoder for ALL downstream (M3/M4/M5)."""
    c = _hybrid_lmin50_z8()                            # z8 arch + l40s tuning + ml_perf
    c["data_patterns"] = _NLA_M
    c["eb_map_variant"] = _EB_VARIANT
    c["pretrained_band_ckpt_path"] = _BAND_CKPT_DIR
    c.pop("repeats", None)
    c["repeat_indices"] = [0, 1, 2, 3, 4]
    return c


kids_legacy_counts_experiments["kids_legacy_hybrid_nla_m_counts_z8"] = _hybrid_counts_z8()
# Smoke = the kids_legacy z8 hybrid smoke verbatim (fwhm8 local fixture, from-scratch band).
kids_legacy_counts_experiments["kids_legacy_hybrid_nla_m_counts_z8_smoke"] = _hybrid_lmin50_z8_smoke()


# === M2-LR  LR-scheme DEBUG variants of the counts z8 foundation (1 repeat each; TOP PRIORITY) ====
# The baseline cyclic (lr 2e-4) hybrid plateaus at the 2-pt band level on the counts maps — the CNN
# is not breaking through. Probe the barrier with 3 LR schemes (user 2026-07-13). 1 repeat each
# (repeat 0 -> band _0); everything else identical to _hybrid_counts_z8 (counts fwhm4 data, frozen
# per-repeat band, 100 epochs). Cyclic max_lr = config.lr (base.py CyclicLR max_lr=base_lrs); exp
# gamma is PER-EPOCH (step_gamma = gamma**(1/steps_per_epoch)).
def _hybrid_counts_z8_lr(lr=None, scheduler_type=None, scheduler_kwargs=None):
    c = _hybrid_counts_z8()
    c["repeat_indices"] = [0]
    if lr is not None:
        c["lr"] = lr
    if scheduler_type is not None:
        c["scheduler_type"] = scheduler_type
    if scheduler_kwargs is not None:
        c["scheduler_kwargs"] = scheduler_kwargs
    return c


def _hybrid_counts_z8_smoke_lr(lr=None, scheduler_type=None, scheduler_kwargs=None):
    """De-clustered fwhm8-local smoke clone with the SAME LR override, for the pre-submit gate."""
    c = _hybrid_lmin50_z8_smoke()
    if lr is not None:
        c["lr"] = lr
    if scheduler_type is not None:
        c["scheduler_type"] = scheduler_type
    if scheduler_kwargs is not None:
        c["scheduler_kwargs"] = scheduler_kwargs
    return c


# gamma for 2e-4 -> 5e-5 over 100 epochs: 0.25**(1/100) ≈ 0.98623 (per-epoch).
_EXPDECAY_KW = {"gamma": 0.98623, "warmup_steps": 0}

kids_legacy_counts_experiments["kids_legacy_hybrid_nla_m_counts_z8_maxlr1e3"] = _hybrid_counts_z8_lr(lr=0.001)
kids_legacy_counts_experiments["kids_legacy_hybrid_nla_m_counts_z8_maxlr1e4"] = _hybrid_counts_z8_lr(lr=0.0001)
kids_legacy_counts_experiments["kids_legacy_hybrid_nla_m_counts_z8_expdecay"] = _hybrid_counts_z8_lr(
    lr=0.0002, scheduler_type="exp", scheduler_kwargs=_EXPDECAY_KW)

kids_legacy_counts_experiments["kids_legacy_hybrid_nla_m_counts_z8_maxlr1e3_smoke"] = _hybrid_counts_z8_smoke_lr(lr=0.001)
kids_legacy_counts_experiments["kids_legacy_hybrid_nla_m_counts_z8_maxlr1e4_smoke"] = _hybrid_counts_z8_smoke_lr(lr=0.0001)
kids_legacy_counts_experiments["kids_legacy_hybrid_nla_m_counts_z8_expdecay_smoke"] = _hybrid_counts_z8_smoke_lr(
    lr=0.0002, scheduler_type="exp", scheduler_kwargs=_EXPDECAY_KW)


# === M2-STAB  stability/ceiling DEBUG variants (counts-training-performance task, 2026-07-14) ====
# Forensics (task .claude/runs/training-runs/counts-training-performance) nailed the failure mode:
# at lr<=2e-4 on counts the map branch COLLAPSES to a constant output within ~10 epochs (patch_mu
# batch-std = 0 in every stuck ckpt) because at init patch_mu enters the concat ~18x smaller than
# the frozen band_mu and receives ~40x less step-0 gradient; at lr=1e-3 it escapes but then
# memorises noise (train -6.3 / val -3.3 by ep95). These variants test the counter-measures:
# LR escape+decay (cycexp), band modality-dropout, patch-head init gain, patch-variance hinge,
# wider summary, weight decay, band-unfreeze, and the map-only H1 upper bounds (counts vs mean).
def _hybrid_counts_z8_stab(mk_extra=None, repeat_indices=(0,), lr=None, scheduler_type=None,
                           scheduler_kwargs=None, optimizer_kwargs=None, freeze_band=None,
                           pretrained_band_lr=None):
    c = _hybrid_counts_z8()
    c["repeat_indices"] = list(repeat_indices)
    if mk_extra:
        c["model_kwargs"] = {**c["model_kwargs"], **mk_extra}
    if lr is not None:
        c["lr"] = lr
    if scheduler_type is not None:
        c["scheduler_type"] = scheduler_type
    if scheduler_kwargs is not None:
        c["scheduler_kwargs"] = scheduler_kwargs
    if optimizer_kwargs is not None:
        c["optimizer_kwargs"] = optimizer_kwargs
    if freeze_band is not None:
        c["freeze_band"] = freeze_band
    if pretrained_band_lr is not None:
        c["pretrained_band_lr"] = pretrained_band_lr
    return c


def _hybrid_counts_z8_stab_smoke(mk_extra=None, lr=None, scheduler_type=None,
                                 scheduler_kwargs=None, optimizer_kwargs=None):
    """De-clustered fwhm8-local smoke clone exercising the SAME new model kwargs / LR scheme."""
    c = _hybrid_lmin50_z8_smoke()
    if mk_extra:
        c["model_kwargs"] = {**c["model_kwargs"], **mk_extra}
    if lr is not None:
        c["lr"] = lr
    if scheduler_type is not None:
        c["scheduler_type"] = scheduler_type
    if scheduler_kwargs is not None:
        c["scheduler_kwargs"] = scheduler_kwargs
    if optimizer_kwargs is not None:
        c["optimizer_kwargs"] = optimizer_kwargs
    return c


# Decaying-peak cyclic (base.py 'cyclic_exp'): peaks 1e-3 * 0.98^epoch -> ~1.3e-4 by ep100; keeps
# the proven 1e-3 escape early while consolidating late (vs flat-peak cyclic that collapses).
_CYCEXP_1E3 = dict(lr=0.001, scheduler_type="cyclic_exp",
                   scheduler_kwargs={"gamma": 0.98, "cyclic_period_steps": 6000, "warmup_steps": 1000})

_STAB_VARIANTS = {
    # name suffix -> (real-config kwargs, smoke-config kwargs)
    "cycexp1e3": (dict(repeat_indices=(0, 1, 2), **_CYCEXP_1E3),
                  dict(lr=0.001, scheduler_type="cyclic_exp",
                       scheduler_kwargs={"gamma": 0.98, "cyclic_period_steps": 60, "warmup_steps": 5})),
    "banddrop02": (dict(mk_extra={"band_dropout_p": 0.2}),
                   dict(mk_extra={"band_dropout_p": 0.2})),
    "pgain16": (dict(mk_extra={"patch_head_init_gain": 16.0}),
                dict(mk_extra={"patch_head_init_gain": 16.0})),
    "banddrop02_pgain16": (dict(mk_extra={"band_dropout_p": 0.2, "patch_head_init_gain": 16.0}),
                           dict(mk_extra={"band_dropout_p": 0.2, "patch_head_init_gain": 16.0})),
    "pvar05": (dict(mk_extra={"patch_var_reg_coeff": 0.5}),
               dict(mk_extra={"patch_var_reg_coeff": 0.5})),
    "z16": (dict(mk_extra={"hybrid_output_dim": 16}),
            dict(mk_extra={"hybrid_output_dim": 16})),
    "wd05_cycexp1e3": (dict(optimizer_kwargs={"weight_decay": 0.05, "betas": (0.9, 0.999)}, **_CYCEXP_1E3),
                       dict(optimizer_kwargs={"weight_decay": 0.05, "betas": (0.9, 0.999)})),
    # Init-gain + LR-escape combo — the two top-ranked (init / gradient-flow) levers together.
    "pgain16_cycexp1e3": (dict(mk_extra={"patch_head_init_gain": 16.0}, **_CYCEXP_1E3),
                          dict(mk_extra={"patch_head_init_gain": 16.0}, lr=0.001)),
    "bandunfreeze": (dict(freeze_band=False, pretrained_band_lr=1e-5),
                     dict()),
}

for _suffix, (_real_kw, _smoke_kw) in _STAB_VARIANTS.items():
    kids_legacy_counts_experiments[f"kids_legacy_hybrid_nla_m_counts_z8_{_suffix}"] = \
        _hybrid_counts_z8_stab(**_real_kw)
    kids_legacy_counts_experiments[f"kids_legacy_hybrid_nla_m_counts_z8_{_suffix}_smoke"] = \
        _hybrid_counts_z8_stab_smoke(**_smoke_kw)


# --- Wave-2 combo (2026-07-15): z16 head + pgain16 init + sustained-escape-then-decay ------------
# Wave-1 remote verdicts: z16 passed the band barrier at 2e-4 (head de-bottleneck, -4.3977@ep33);
# pgain16 keeps the map branch alive but 2e-4 alone cannot escape; resume-based consolidation
# FAILED (the maxlr1e3 optimum is an unstable transient — val degrades even at decayed LR). So do
# escape+consolidate in ONE run: cyclic peaks HELD at the proven 1e-3 through the escape window
# (~ep35 > the observed ep27 breakthrough), then per-epoch 0.97 peak decay ('cyclic_hold_exp',
# base.py). ~970 steps/epoch at B=100 on the full store -> hold_steps 34000.
_ESC1E3_HOLD = dict(lr=0.001, scheduler_type="cyclic_hold_exp",
                    scheduler_kwargs={"gamma": 0.97, "cyclic_period_steps": 6000,
                                      "warmup_steps": 1000, "hold_steps": 34000})
kids_legacy_counts_experiments["kids_legacy_hybrid_nla_m_counts_z16_pgain16_esc1e3"] = \
    _hybrid_counts_z8_stab(mk_extra={"hybrid_output_dim": 16, "patch_head_init_gain": 16.0},
                           **_ESC1E3_HOLD)
# Wave-2c: exact maxlr1e3 replica (z8, NO init gain) + hold-then-decay. Both combo variants
# (p6000 AND p2000) failed to escape, so the remaining suspects are (a) z16/pgain16 interfering
# with the escape, (b) subtle cyclic_hold_exp-vs-CyclicLR difference, (c) the maxlr1e3 escape was
# seed-luck. This run tests (a)+(b) together; a plain maxlr1e3 repeat-1 resubmit tests (c).
kids_legacy_counts_experiments["kids_legacy_hybrid_nla_m_counts_z8_esc1e3_p2000"] = \
    _hybrid_counts_z8_lr(lr=0.001, scheduler_type="cyclic_hold_exp",
                         scheduler_kwargs={"gamma": 0.97, "cyclic_period_steps": 2000,
                                           "warmup_steps": 1000, "hold_steps": 34000})
kids_legacy_counts_experiments["kids_legacy_hybrid_nla_m_counts_z8_esc1e3_p2000_smoke"] = \
    _hybrid_counts_z8_smoke_lr(lr=0.001, scheduler_type="cyclic_hold_exp",
                               scheduler_kwargs={"gamma": 0.97, "cyclic_period_steps": 20,
                                                 "warmup_steps": 5, "hold_steps": 100})


# p2000 variant: identical combo but cyclic_period_steps=2000 — the DEFAULT period the one
# escaping run (maxlr1e3) actually used. Period 6000 gives peaks only every ~6 epochs (~3x less
# dwell near 1e-3); both stuck-slow cycexp runs and (if it fails) the esc1e3 combo share the 6000
# period, so this isolates the dwell-time confound.
kids_legacy_counts_experiments["kids_legacy_hybrid_nla_m_counts_z16_pgain16_esc1e3_p2000"] = \
    _hybrid_counts_z8_stab(mk_extra={"hybrid_output_dim": 16, "patch_head_init_gain": 16.0},
                           lr=0.001, scheduler_type="cyclic_hold_exp",
                           scheduler_kwargs={"gamma": 0.97, "cyclic_period_steps": 2000,
                                             "warmup_steps": 1000, "hold_steps": 34000})
kids_legacy_counts_experiments["kids_legacy_hybrid_nla_m_counts_z16_pgain16_esc1e3_p2000_smoke"] = \
    _hybrid_counts_z8_stab_smoke(mk_extra={"hybrid_output_dim": 16, "patch_head_init_gain": 16.0},
                                 lr=0.001, scheduler_type="cyclic_hold_exp",
                                 scheduler_kwargs={"gamma": 0.97, "cyclic_period_steps": 20,
                                                   "warmup_steps": 5, "hold_steps": 100})
kids_legacy_counts_experiments["kids_legacy_hybrid_nla_m_counts_z16_pgain16_esc1e3_smoke"] = \
    _hybrid_counts_z8_stab_smoke(mk_extra={"hybrid_output_dim": 16, "patch_head_init_gain": 16.0},
                                 lr=0.001, scheduler_type="cyclic_hold_exp",
                                 scheduler_kwargs={"gamma": 0.97, "cyclic_period_steps": 60,
                                                   "warmup_steps": 5, "hold_steps": 100})


# --- z64: WIDE latent (user 2026-07-14, corrected spec) ------------------------------------------
# The whole bottleneck widened to 64: frozen 8-D band (its ckpt must still load — do NOT change
# bandpower_latent_dim) + 56-D map branch -> 64-D concat fed STRAIGHT to the flow (latent_dim=64,
# no hybrid_output_dim => head is Linear(64->64), no compression, no expansion). Tests whether the
# map-side 8-D latent (not just the 16->8 summary head) was the starving constraint.
def _hybrid_counts_z64():
    c = _hybrid_counts_z8()
    c["latent_dim"] = 64                                  # band 8 + patch 56
    mk = {**c["model_kwargs"]}
    mk.pop("hybrid_output_dim", None)                     # output = latent_dim = 64
    c["model_kwargs"] = mk
    c["repeat_indices"] = [0]
    return c


def _hybrid_counts_z64_smoke():
    c = _hybrid_lmin50_z8_smoke()
    c["latent_dim"] = 64
    mk = {**c["model_kwargs"]}
    mk.pop("hybrid_output_dim", None)
    c["model_kwargs"] = mk
    return c


kids_legacy_counts_experiments["kids_legacy_hybrid_nla_m_counts_z64wide"] = _hybrid_counts_z64()
kids_legacy_counts_experiments["kids_legacy_hybrid_nla_m_counts_z64wide_smoke"] = _hybrid_counts_z64_smoke()


# --- Two-phase escape+consolidate: resume the maxlr1e3 transient optimum, decay from there ------
# Remote wave-1 finding (jobs 1321123/1321124, 2026-07-14): the decaying-peak cyclic (gamma
# 0.98/ep) NEVER escapes — its peak drops below the ~1e-3 escape threshold within ~15 epochs
# (bests frozen at the band level through ep 40+), while flat 1e-3 escaped by ep 13 but then
# memorised noise. Dose-response: escape needs SUSTAINED ~1e-3; consolidation needs decay.
# So do them in sequence: warm-start from the maxlr1e3 run's best ckpt (-4.8107 @ep27, the
# transient optimum) and fine-tune with a pure exp decay 3e-4 -> ~5e-6 over 60 epochs.
def _hybrid_counts_z8_resumedecay():
    c = _hybrid_counts_z8()
    c["repeat_indices"] = [0]
    c["checkpoint_path"] = f"{_CKPT}/kids_legacy_hybrid_nla_m_counts_z8_maxlr1e3/"
    c["lr"] = 0.0003
    c["scheduler_type"] = "exp"
    c["scheduler_kwargs"] = {"gamma": 0.97, "warmup_steps": 0}
    c["epochs"] = 60
    return c


kids_legacy_counts_experiments["kids_legacy_hybrid_nla_m_counts_z8_maxlr1e3_resumedecay"] = \
    _hybrid_counts_z8_resumedecay()
# Smoke: de-clustered (checkpoint_path is nulled by the smoke harness; proves lr/sched/epochs build).
kids_legacy_counts_experiments["kids_legacy_hybrid_nla_m_counts_z8_maxlr1e3_resumedecay_smoke"] = \
    _hybrid_counts_z8_stab_smoke(lr=0.0003, scheduler_type="exp",
                                 scheduler_kwargs={"gamma": 0.97, "warmup_steps": 0})


# --- H1 map-only upper bounds: does a band-free CNN beat the band level at all? -----------------
# Same map encoder/data/l40s tuning as the hybrid but NO band branch: model_type kids_o3_dual on
# E maps only. Run one on the counts store and one on the MEAN-norm store (same schedule) — the
# counts-vs-mean map-only gap measures how much standalone map info counts normalisation removed
# (H1 data ceiling vs fusion/optimisation failure).
def _maponly(data_patterns, eb_variant):
    c = _hybrid_counts_z8()               # inherit data/loader/l40s/ml_perf/epochs/project keys
    c["data_patterns"] = data_patterns
    c["eb_map_variant"] = eb_variant
    c["model_type"] = "kids_o3_dual"
    c["dataset_quantities"] = ["E_north", "E_south"]
    c["latent_dim"] = 8
    c["model_kwargs"] = {
        "encoder_type": "unet_o3",
        "pool_types": ("avg", "max", "gem"),
        "patch_conditioning": ("side_info"),
    }
    for k in ("pretrained_band_ckpt_path", "freeze_band"):
        c.pop(k, None)
    c["repeat_indices"] = [0]
    c["lr"] = _CYCEXP_1E3["lr"]
    c["scheduler_type"] = _CYCEXP_1E3["scheduler_type"]
    c["scheduler_kwargs"] = _CYCEXP_1E3["scheduler_kwargs"]
    return c


def _maponly_smoke():
    """De-clustered LOCAL smoke of the map-only config (fwhm8 fixture, few epochs)."""
    c = _maponly(_SMOKE_DATA, _SMOKE_EB_VARIANT)
    c.pop("repeat_indices", None)
    c["epochs"] = 3
    c["num_workers"] = 2
    c["prefetch_factor"] = 2
    c["persistent_workers"] = False
    c["batch_size"] = 8
    return c


kids_legacy_counts_experiments["kids_legacy_maponly_nla_m_counts"] = _maponly(_NLA_M, _EB_VARIANT)
kids_legacy_counts_experiments["kids_legacy_maponly_nla_m_mean"] = _maponly(
    _MEAN_NLA_M_FWHM4, _MEAN_EB_VARIANT_FWHM4)
kids_legacy_counts_experiments["kids_legacy_maponly_nla_m_counts_smoke"] = _maponly_smoke()


# === M3  main-variate Gower NPE 9-member ensemble finetune (5 repeats) ==========================
kids_legacy_counts_experiments["gower_npe_finetune_nla_m_counts_z8"] = _npe_finetune_z8(
    _FOUNDATION_CKPT, data_patterns=_GOWER_NLA_M, eb_variant=_EB_VARIANT)


def _npe_finetune_counts_z8_smoke():
    """De-clustered LOCAL smoke of gower_npe_finetune_nla_m_counts_z8 (there is NO precedent smoke
    for the z8 NPE finetune, so build one like the hybrid smoke): from-scratch (checkpoint_path=None,
    from-scratch band), fwhm8 local fixture, single-cosmology-safe (no cosmo cap / fixed-test lock,
    ensemble_repeats=1), few epochs -> proves the z8 NPE-finetune config BUILDS + finite loss."""
    c = _npe_finetune_z8(_FOUNDATION_CKPT, data_patterns=_SMOKE_DATA, eb_variant=_SMOKE_EB_VARIANT)
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


kids_legacy_counts_experiments["gower_npe_finetune_nla_m_counts_z8_smoke"] = _npe_finetune_counts_z8_smoke()


# === M4  main-variate NLE chain: 5x (GLASS pretrain -> Gower ens9 finetune), z8 pure-whiten k=8 =
# Per repeat r: GLASS pretrain (glass_nle_pretrain_nla_m_counts_z8_r{r}, v100) -> Gower ens9 finetune
# (gower_nle_finetune_nla_m_counts_z8_r{r}_ens9, CORES64 MCMC eval). MAIN split: 300 train/val (80/20),
# 200 fixed-test ids held out. Source encoder on the embed CLI: --sources kids_legacy_hybrid_nla_m_counts_z8.
def _register_main_nle_counts_z8():
    for r in range(5):
        kids_legacy_counts_experiments[f"glass_nle_pretrain_nla_m_counts_z8_r{r}"] = _nle_bake_repeat(
            _nle_pretrain(_NLA_M, _EB_VARIANT, whiten_k=8, epochs=150), r)
        ft = _nle_finetune(f"glass_nle_pretrain_nla_m_counts_z8_r{r}", ensemble_repeats=9,
                           whiten_k=8, warmstart_max_gap_nats=22.0,
                           gower_data=_GOWER_NLA_M, gower_eb=_EB_VARIANT)
        ft["max_trainval_cosmos"] = [300]
        ft["train_frac"] = 0.8
        ft["val_frac"] = 0.2
        ft["test_frac"] = 0.0   # test = fixed 200 ids; fracs must sum to 1.0 (split_by_cosmology)
        ft["fixed_test_sim_ids"] = _GOWER_TEST_IDS
        kids_legacy_counts_experiments[f"gower_nle_finetune_nla_m_counts_z8_r{r}_ens9"] = _nle_bake_repeat(ft, r)


_register_main_nle_counts_z8()


# === M5  sub-variate chains {nla, nla_z, no_vd}: encoder-finetune (M5a) + NLE chain (M5b/M5c) ====
# Per S: warm-start the counts foundation ENCODER onto the sub-variate GLASS suite (M5a), then the
# NLE chain (M5b pretrain -> M5c Gower ens9 finetune). nla/nla_z carry the a_ia~U[-6,6] box at EVERY
# stage; theta set per variate (nla 8-D a_ia; nla_z 9-D a_ia+b_z; no_vd 9-D a_ia+b_ia = NLA-M box).
# Experiment names use `no_vd` (the established kids_legacy convention, e.g. glass_encoder_finetune_no_vd_z8),
# even though the store is glass/gower_mocks_nla_m_novd_counts.
#
# _SUB_VARIATES[S] = (glass_store, gower_store, cosmo_param_names, preset_overrides, smoke_cosmo)
# smoke_cosmo differs from cosmo where the LOCAL fixture lacks a param: nla_z's real b_z is not in the
# fwhm8 NLA-M fixture, so its smoke uses _COSMO_9 (a_ia+b_ia) for the same 9-D flow shape (mirrors the
# existing glass_encoder_finetune_nla_z_z8_smoke).
_SUB_VARIATES = {
    "nla":   (_NLA,   _GOWER_NLA,   _COSMO_8_NLA,  _A_IA_NLA_BOX, _COSMO_8_NLA),
    "nla_z": (_NLA_Z, _GOWER_NLA_Z, _COSMO_9_NLAZ, _A_IA_NLA_BOX, _COSMO_9),
    "no_vd": (_NOVD,  _GOWER_NOVD,  _COSMO_9,      None,          _COSMO_9),
}


def _encoder_finetune_counts(glass_data, cosmo, preset):
    c = _encoder_finetune_z8(glass_data, _EB_VARIANT, cosmo, repeat_indices=(0, 1, 2, 3, 4),
                             preset_overrides=preset)
    c["pretrained_embedding_ckpt_path"] = _FOUNDATION_CKPT   # the COUNTS foundation (not the lmin50 one)
    return c


def _register_sub_variate(S, glass_data, gower_data, cosmo, preset, smoke_cosmo):
    # M5a: encoder finetune (+ de-clustered smoke via the kids_legacy encoder-smoke factory).
    kids_legacy_counts_experiments[f"glass_encoder_finetune_{S}_counts_z8"] = _encoder_finetune_counts(
        glass_data, cosmo, preset)
    kids_legacy_counts_experiments[f"glass_encoder_finetune_{S}_counts_z8_smoke"] = _encoder_finetune_z8_smoke(
        smoke_cosmo)
    # M5b/M5c: NLE pretrain -> Gower ens9 finetune per repeat. SUB split: 100 cosmos, 70/30.
    for r in range(5):
        kids_legacy_counts_experiments[f"glass_nle_pretrain_{S}_counts_z8_r{r}"] = _nle_bake_repeat(
            _nle_pretrain(glass_data, _EB_VARIANT, whiten_k=8, epochs=150,
                          cosmo_param_names=cosmo, preset_overrides=preset), r)
        ft = _nle_finetune(f"glass_nle_pretrain_{S}_counts_z8_r{r}", ensemble_repeats=9,
                           whiten_k=8, warmstart_max_gap_nats=22.0,
                           gower_data=gower_data, gower_eb=_EB_VARIANT,
                           cosmo_param_names=cosmo, preset_overrides=preset)
        ft["max_trainval_cosmos"] = [100]
        ft["train_frac"] = 0.7
        ft["val_frac"] = 0.3
        ft["test_frac"] = 0.0
        # TODO(BLOCKER before M5c runs) sub-variate first-100/last-100 lock files. The sub-variate
        # Gower stores (D6/D7/D8) are `--gower-sim-set fixed_test` = EXACTLY the 200 gower_test_ids, so
        # this placeholder would force ALL 200 into the test split and leave 0 train/val — it WILL
        # break M5c as-is (NOT a graceful fallback: the overlap is complete, not empty). Before M5c:
        # create gower_test_ids_first100.json (train/val pool) + gower_test_ids_last100.json (held-out
        # test), set fixed_test_sim_ids=last-100, and restrict train/val to the first-100. Placeholder
        # kept only so the module imports; M5c is far downstream (blocked on D6/D7/D8 + M5a/M5b).
        ft["fixed_test_sim_ids"] = _GOWER_TEST_IDS
        kids_legacy_counts_experiments[f"gower_nle_finetune_{S}_counts_z8_r{r}_ens9"] = _nle_bake_repeat(ft, r)


for _S, (_g, _gow, _cos, _pre, _smk) in _SUB_VARIATES.items():
    _register_sub_variate(_S, _g, _gow, _cos, _pre, _smk)


# TODO Section C conservative (fwhm8) stack — mirror M2-M6 with `_cons` names + fwhm8 stores
# (_EB_VARIANT_CONS="fwhm8_lmin56_lcut1024", *_fwhm8 map stores, _FOUNDATION_CONS_CKPT). Band M1 is
# REUSED (bandpowers are smoothing-independent). Deliberately deferred (LATER pass).


# =================================================================================================
# counts-training-performance-EXTENDED (2026-07-16): anti-starvation + head/architecture waves.
# Diagnosis (task artifacts/diagnosis.md): VMIM loss is minimised without the CNN (gradient
# starvation); the zero-init scale-shift UNet degenerates to a rank-1, DC-exploding representation.
# Levers here: patch_norm (LayerNorm barrier on patch_mu), patch_aux (map-only auxiliary VMIM
# head), mdn (mixture head), deep (backbone depth/width). All on the default z8 base (user).
# =================================================================================================

def _counts_ext(name, mk_extra=None, top_extra=None, smoke_mk=None, smoke_top=None,
                repeat_indices=(0,), **stab_kw):
    """Register a counts-extended variant + its _smoke clone (fwhm8 single-cosmo fixture)."""
    c = _hybrid_counts_z8_stab(mk_extra=mk_extra, repeat_indices=repeat_indices, **stab_kw)
    for k, v in (top_extra or {}).items():
        c[k] = v
    kids_legacy_counts_experiments[f"kids_legacy_hybrid_nla_m_counts_{name}"] = c
    s = _hybrid_counts_z8_stab_smoke(mk_extra=(smoke_mk if smoke_mk is not None else mk_extra))
    for k, v in ((smoke_top if smoke_top is not None else top_extra) or {}).items():
        s[k] = v
    kids_legacy_counts_experiments[f"kids_legacy_hybrid_nla_m_counts_{name}_smoke"] = s


def _deep_mapkw(base_mk, **kw):
    mk = dict(base_mk or {})
    mk["map_kwargs"] = {**mk.get("map_kwargs", {}), **kw}
    return mk


_BASE_MK = _hybrid_counts_z8()["model_kwargs"]

# LayerNorm barrier on patch_mu: kills the DC/amplitude pathology + scale imbalance structurally.
_counts_ext("z8_pnorm", mk_extra={"patch_norm": "layernorm"})
# Map-only auxiliary VMIM head: first-class gradient into the CNN that the band cannot satisfy.
_counts_ext("z8_mapaux05", top_extra={"patch_aux_weight": 0.5,
                                      "patch_aux_flow_kwargs": {"hidden_features": 32}})
# The mechanistically-complete combo: barrier + aux gradient (+ known-good banddrop reserve below).
_counts_ext("z8_pnorm_mapaux05", mk_extra={"patch_norm": "layernorm"},
            top_extra={"patch_aux_weight": 0.5,
                       "patch_aux_flow_kwargs": {"hidden_features": 32}})
# MDN/GMM head (build_made): smoother conditioning gradients than the spline flow.
_counts_ext("z8_mdn", top_extra={"flow_type": "mdn",
                                 "flow_kwargs": {"num_mixture_components": 12,
                                                 "hidden_features": 64}})
# Depth/width control: does capacity alone change anything? (prediction from diagnosis: no)
# 64ch = 2x the historical 32ch width; 96ch at full 1000x100 res risks OOM at batch 100.
_counts_ext("z8_deep", mk_extra=_deep_mapkw(_BASE_MK, model_channels=64, num_res_blocks=3))
# Wider NSF head on the z8 base (flow-capacity axis).
_counts_ext("z8_flowbig", top_extra={"flow_kwargs": {"hidden_features": 64}})
# Combo + banddrop02 (strongest known anti-collapse) as the belt-and-braces variant.
_counts_ext("z8_pnorm_mapaux05_banddrop02",
            mk_extra={"patch_norm": "layernorm", "band_dropout_p": 0.2},
            top_extra={"patch_aux_weight": 0.5,
                       "patch_aux_flow_kwargs": {"hidden_features": 32}})
# BatchNorm barrier variant (2026-07-16, post-probe): the LayerNorm barrier + aux head still
# collapses cross-sample (LN is per-sample; early training prefers constant conditioning for BOTH
# flows). Non-affine BatchNorm on patch_mu forbids that state structurally; the aux head then
# shapes WHICH forced variance is informative.
_counts_ext("z8_bnorm_mapaux05", mk_extra={"patch_norm": "batchnorm"},
            top_extra={"patch_aux_weight": 0.5,
                       "patch_aux_flow_kwargs": {"hidden_features": 32}})
# === PROMOTED (user 2026-07-16): clean pre-activation ResNet map encoder ========================
# Replace the degenerating diffusion-UNet with a boring, alive-at-init deep ResNet (15 blocks,
# GN, GeM head, symmetric downsampling). Highest-priority wave; aux-head/barrier variants are
# fallback companions, not the lead.
_RESNET_MAPKW = {"encoder_type": "preact_resnet", "patch_conditioning": None,
                 "pool_types": ("avg", "gem"),
                 "stage_channels": (32, 64, 128, 256, 256), "blocks_per_stage": 3}
_counts_ext("z8_resnet", mk_extra={"map_kwargs": _RESNET_MAPKW})
# Belt-and-braces companion: ResNet + the strongest known anti-collapse lever, in case starvation
# still bites a healthy encoder.
_counts_ext("z8_resnet_banddrop02",
            mk_extra={"map_kwargs": _RESNET_MAPKW, "band_dropout_p": 0.2})
# ResNet + wider NSF head (2026-07-17): with the encoder finally delivering map information into
# the 8-D summary, the hidden=32 flow may be the next bottleneck (lit-review flag). Low-hanging
# fruit push toward the -5.2..-5.5 floor target.
_counts_ext("z8_resnet_flowbig", mk_extra={"map_kwargs": _RESNET_MAPKW},
            top_extra={"flow_kwargs": {"hidden_features": 64}})
