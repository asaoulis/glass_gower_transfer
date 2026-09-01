"""BGP campaign (galaxy-bias-prior MARGINALISED) — model configs.

Campaign definition: `.claude/runs/training-runs/production-training-runs/STATUS.md`.
Rows: `models_checklist.md` (models) · `datasets_checklist.md` (data).

**What is different from `_dn`.** Nothing in the training recipe — the PreActResNet map encoder, the
z8 hybrid arch, the frozen-per-repeat Stage-I band, the l40s tuning are all imported, not copied, so
this suite cannot drift from the validated foundation. The ONE axis that changed is upstream, in the
data: each (sim, outer, rot) block of the foundation store drew an independent per-tomo-bin
`b_i ~ N(mean_i, sigma_i)` from the Flamingo KiDS-Legacy calibration (`flamingo_pt_diag`, kappa = 1)
instead of pinning `b_g = 1`. Consequently **no `_dn` result is a baseline here** — val NLLs and FoMs
must be re-measured within this campaign.

**Why `_bgp` names.** Checkpoints live at `{base_path}/checkpoints/{experiment_name}/`, so reusing a
`_dn` name would mix campaigns in one checkpoint dir and `get_best_checkpoint` could silently
resolve the wrong one. The token is used verbatim on raw stores, prebaked stores and experiment
names so greps line up across the three.

**`b_g` is stored, not inferred.** `b_g_bin1..6` (and `galaxy_bias_eff`) ride in `cosmo_dict` and are
copied wholesale by the bake, but they do NOT join `cosmo_param_names` — so no
`COSMO_PARAM_PRESET_MINMAX` entry and no `build_gower_prior` component are needed. Promoting one to a
10th inference parameter is a later, separate decision.

**One band, and why it is safe here.** `split_by_cosmology` shuffles with `random.Random(42)` over the
cosmology LIST, so a single extra cosmology re-randomises the whole split — which is why the `_dn`
suite had to budget a band per bake snapshot while its foundation store was still generating. That
hazard is absent here: the foundation sim was **stopped before the bakes were submitted**
(job 1343141 cancelled 2026-08-14 21:41Z at 100 600 files; bakes 1343804/1343805 submitted 21:44Z),
so both bakes glob a FROZEN raw store and see the same file list by construction.

⚠️ "By construction" is still not "measured". The bake drops truncated/corrupt files, and a run
stopped mid-block can leave a file whose `E_fwhm4…` group is complete while its `E_sc8_…` group is
not — which would drop from one bake but not the other. **Verify each bake's `ok=` count and, if they
disagree, give the arms their own bands** (as `_dn` had to for R1024) before trusting cross-arm
absolute val NLL. Until that check passes, only ΔNLL (a difference, so the val-draw offset cancels to
first order) is comparable across arms.

Merge: `kids_legacy_bgp_experiments` is `.update()`-merged into the experiments dict by train.py /
eval.py / train_embeddings.py / .claude/cluster/smoke_test_experiment.py, and by
src/ml/eval/misspec.py:_load_experiment_config.

Smoke gate: the local fixture carries only `E_fwhm8_lmin50_lcut1400`, so the real fwhm4 / sc8 map
configs **false-fail** it — production submits of those rows pass `--skip-smoke`. The band carries no
`eb_map_variant`, so it passes the fixture gate directly and needs no `_smoke` clone.
"""
from config.kids_legacy import (
    _band_lmin50, _hybrid_lmin50_z8, _hybrid_lmin50_z8_smoke,
    # sub-variate theta sets + the NLA-family a_ia box (nla/nla_z drop b_ia and widen a_ia)
    _COSMO_8_NLA, _COSMO_9_NLAZ, _A_IA_NLA_BOX,
    # the whitened-NLE chain factories (Stage A pretrain on GLASS; one repeat baked per row)
    _nle_pretrain, _nle_finetune, _nle_bake_repeat,
    # the production Gower NPE ensemble finetune
    _npe_finetune_z8,
)
# Import (never copy) the production map-encoder kwargs so this suite cannot drift from the
# validated PreActResNet foundation arch.
from config.kids_legacy_novd import _RESNET_MAPKW

# --- stores ------------------------------------------------------------------------------------
_GPU5 = "/share/gpu5/asaoulis/transfer_datasets"
_CKPT = "/share/gpu5/asaoulis/transfer_models/checkpoints"

_EB = "fwhm4_lmin56_lcut1400"                 # the primary scale cut (A0/A1 arms)
_EB_SC8 = "sc8_fwhm4_lmin56_lcut1400"         # the smoothed-counts map product (A3s8 arms)

# Prebaked f16 stores on gpu5 (l40s-local), baked from the frozen raw foundation store
# `glass_mocks_nla_m_novd_bgp` on gpu4. Both were baked WITHOUT `--keep-variant-tag`, so they carry
# BARE `E`/`B` groups, not `E_<tag>` — hence `eb_map_variant=None` on the hybrids below
# (cf. scripts/prebake_maps.py:109). This is the same convention the R1024 `_dn` arm used.
_BGP_A1 = f"{_GPU5}/glass_bgp_nla_m_f16_a1_{_EB}/output_*.h5"
_BGP_SC8A1 = f"{_GPU5}/glass_bgp_nla_m_f16_sc8a1_{_EB}/output_*.h5"
# Co-primary M2b (plain sc8, unwhitened). Needs its own bake:
#   prebake --src-datasets-root gpu4 --src-dir glass_mocks_nla_m_novd_bgp \
#     --out-dir glass_bgp_nla_m_f16_sc8_fwhm4_lmin56_lcut1400 \
#     --eb-variant sc8_fwhm4_lmin56_lcut1400 --dtype float16      (i.e. sc8a1's command minus
#                                                                  --noise-norm rand)
_BGP_SC8 = f"{_GPU5}/glass_bgp_nla_m_f16_sc8_{_EB}/output_*.h5"

# Stage-I band checkpoint dir. ONE band serves every arm — see the module docstring for the
# condition that makes that valid here, and the `ok=`-count check that must confirm it.
_BAND_CKPT_BGP = f"{_CKPT}/kids_legacy_band_nla_m_bgp/"

# Seeds. 5 band repeats (user, 2026-08-14): the hybrids run at r0/r1, so r2..r4 are plateau-rescue
# spares — a hybrid seed that stalls is re-paired against a fresh band rather than rescued in place.
_BAND_REPEATS = [0, 1, 2, 3, 4]
_HYBRID_REPEATS = (0, 1)

kids_legacy_bgp_experiments = {}


# === Stage I — bandpower MLP (5 repeats, shared by every arm; v100) ==============================
def _band_bgp(data_patterns):
    """Stage-I bandpower encoder on the PREBAKED store (which carries `mixed_bandpowers` verbatim).

    Reads the BAKE, not the raw gpu4 store, for the reason the `_dn` suite established: the frozen
    band is reused by every hybrid, so its train split must be the hybrids' train split. A band fed
    from a different file list would put its training cosmologies inside the hybrids' VALIDATION
    split and bias `val_nll_bandonly` by an unknown amount — of the same order as the effect being
    measured (cf. the `_dn` R1024 arm, which needed its own band for exactly this reason).
    Reading the bake also cuts ~13x the bytes per file, which was the Era-3 band's I/O bottleneck.
    """
    c = _band_lmin50()
    c["data_patterns"] = data_patterns
    c.pop("repeats", None)
    c["repeat_indices"] = list(_BAND_REPEATS)
    return c


# M1 — the campaign's CONTROL row. Bandpowers are `b_g`-immune (the counts normalisation cancels it
# in the 2-pt statistic to ~0.3 %), so this row should reproduce the `_dn` band result; if it does
# not, something other than the prior changed and the map rows cannot be interpreted.
kids_legacy_bgp_experiments["kids_legacy_band_nla_m_bgp"] = _band_bgp(_BGP_A1)


# === Stage II — the hybrid arms (2 repeats each; l40s) ===========================================
def _hybrid_bgp(data_patterns, eb_variant, band_ckpt, top_extra=None,
                repeat_indices=_HYBRID_REPEATS):
    """PreActResNet z8 hybrid ("hybrid v2"), frozen per-repeat Stage-I band, on one arm's store.

    Byte-identical to the validated `kids_legacy_hybrid_nla_m_novd_z8_resnet` production recipe
    apart from (a) which store it reads and (b) which band it freezes — so the only axis that varies
    across arms is the shear processing, and the only axis that varies against `_dn` is the prior.
    """
    c = _hybrid_lmin50_z8()                       # z8 arch + l40s tuning + ml_perf
    c["data_patterns"] = data_patterns
    c["eb_map_variant"] = eb_variant
    c["pretrained_band_ckpt_path"] = band_ckpt
    c["model_kwargs"] = {**c["model_kwargs"], "map_kwargs": _RESNET_MAPKW}
    c.pop("repeats", None)
    c["repeat_indices"] = list(repeat_indices)
    for k, v in (top_extra or {}).items():
        c[k] = v
    return c


def _hybrid_bgp_smoke(top_extra=None):
    """De-clustered fwhm8-local smoke clone exercising the SAME model kwargs (from-scratch band).

    The fixture carries only `E_fwhm8_lmin50_lcut1400`, so the smoke cannot gate the real fwhm4 /
    sc8 tag — that is gated by the bake's `ok=` count. What it DOES gate is the config building and
    the resnet map encoder wiring.
    """
    c = _hybrid_lmin50_z8_smoke()
    c["model_kwargs"] = {**c["model_kwargs"], "map_kwargs": _RESNET_MAPKW}
    for k, v in (top_extra or {}).items():
        c[k] = v
    return c


# --- the arms ------------------------------------------------------------------------------------
# M2 — A3s8_A1 (sc8a1), the PRIMARY: smoothed-counts denominator (8') before the spin-2 SHT, plus the
# A1 random-rotation rescale. Carried for robustness against SPATIALLY STRUCTURED misspecification,
# which a single per-mock scalar cannot absorb.
kids_legacy_bgp_experiments["kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1"] = \
    _hybrid_bgp(_BGP_SC8A1, None, _BAND_CKPT_BGP)

# M2c — A1_wht_rand (plain counts + whitening), the CONTROL arm.
kids_legacy_bgp_experiments["kids_legacy_hybrid_nla_m_bgp_z8_resnet_a1"] = \
    _hybrid_bgp(_BGP_A1, None, _BAND_CKPT_BGP)

# M2b — A3s8 alone (unwhitened sc8), the CO-PRIMARY: on the `_dn` stores plain sc8 reached -5.2675
# vs sc8a1's -5.2127 and both cleared the -4.5 plain-counts cluster, so the pair is run and this
# campaign's data picks the winner. ⚠️ Blocked until `_BGP_SC8` is baked (command above).
kids_legacy_bgp_experiments["kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8"] = \
    _hybrid_bgp(_BGP_SC8, None, _BAND_CKPT_BGP)

# --- smoke clones --------------------------------------------------------------------------------
for _name in ("a1", "sc8", "sc8a1"):
    kids_legacy_bgp_experiments[f"kids_legacy_hybrid_nla_m_bgp_z8_resnet_{_name}_smoke"] = \
        _hybrid_bgp_smoke()


# === M7 — the 8-parameter `b_g`-SENSITIVITY probe (one repeat per arm) ============================
# Question: with `b_g` marginalised at generation, how much per-tomo-bin galaxy-bias information do
# the maps actually carry, and does the SHEAR NORMALISATION change that? Each arm's finished
# foundation encoder is warm-started and finetuned with a FRESH 8-D flow head over
# {omega_m, sigma_8, b_g_bin1..6}. Same store, same architecture, same split as that arm's
# foundation run — the only thing that changes is what the flow is asked to infer, so any
# difference across arms is attributable to the shear processing.
#
# ⚠️ INTERPRETATION: the foundation encoders were VMIM-trained to compress for the NINE cosmo/IA
# params, with `b_g` deliberately NOT among them. A *positive* b_g constraint after 25 warm-start
# epochs is therefore meaningful; a NULL is weak evidence, because the encoder may simply need
# longer to re-learn a channel it was trained to discard. Read a null as "not recovered in 25
# epochs from this warm start", not "absent from the maps".
_COSMO_8_BG = ["omega_m", "sigma_8"] + [f"b_g_bin{_i}" for _i in range(1, 7)]

# Scaler boxes for the six per-bin biases = the generator's own truncated support: mean ± 3σ of the
# Flamingo O3-diag calibration at kappa=1. Derived here rather than transcribed so they cannot drift
# from the source of truth (src/KiDS/simulation_config.py: GALAXY_BIAS_PRIOR_MEANS/SIGMAS, ±3σ
# truncation). None of the six reaches GALAXY_BIAS_CLIP=(0.3, 2.2), so ±3σ IS the realised range.
_BG_PRIOR_MEANS = [1.0181, 1.0698, 1.1302, 1.2427, 1.3739, 1.4805]
_BG_PRIOR_SIGMAS = [0.1801, 0.1491, 0.1252, 0.0951, 0.0960, 0.0985]
_BG_BOXES = {
    f"b_g_bin{_i + 1}": (_m - 3.0 * _s, _m + 3.0 * _s)
    for _i, (_m, _s) in enumerate(zip(_BG_PRIOR_MEANS, _BG_PRIOR_SIGMAS))
}

# Foundation checkpoint dirs — one per arm, all 5-seed complete (2026-08-16).
_HYB_CKPT = {
    "sc8a1": f"{_CKPT}/kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1/",
    "sc8": f"{_CKPT}/kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8/",
    "a1": f"{_CKPT}/kids_legacy_hybrid_nla_m_bgp_z8_resnet_a1/",
}

# ⚠️ REPEAT 3, NOT 0 — deliberate and load-bearing. The a1 arm escaped the −4.4 wall on r3 ONLY
# (r0/r1/r2 are dead wall-stuck checkpoints at ≈ −4.46 that were cancelled); warm-starting a1 from
# r0 would measure a rank-degenerate encoder rather than the arm. r3 is also the BEST seed of both
# sc8a1 (−5.4014) and sc8 (−5.3345), and gives all three arms the same `split_seed = base + 3`, so
# the arms are compared on identical train/val/test splits (the three bakes share one file list).
_BG8_REPEAT = 3


def _hybrid_bgp_bg8(data_patterns, arm_ckpt, repeat_index=_BG8_REPEAT):
    """One arm's foundation encoder, warm-started + finetuned with a FRESH 8-D flow head.

    Built from that arm's OWN `_hybrid_bgp` recipe (not from `_encoder_finetune_z8`, which points at
    a different store and replaces `scaler_options` wholesale), then given the warm-start deltas:
    the band arrives INSIDE the loaded `embedding_net`, so the separate frozen-band path is dropped.
    """
    c = _hybrid_bgp(data_patterns, None, _BAND_CKPT_BGP, repeat_indices=(repeat_index,))
    c.pop("pretrained_band_ckpt_path", None)   # band comes in with the embedding_net
    c.pop("freeze_band", None)
    c["pretrained_embedding_ckpt_path"] = arm_ckpt
    c["freeze_embedding_net"] = False          # finetune the encoder — give each arm its best shot
    c["match_num_cosmo"] = False               # resolve the arm ckpt per-repeat as "_{i}"
    c["cosmo_param_names"] = list(_COSMO_8_BG)
    # Same shape as config/default.py's scaler_options (data untouched) + the six b_g boxes; the
    # preset scaler RAISES on an unknown parameter (src/ml/utils.py:170), so these are REQUIRED.
    c["scaler_options"] = {
        "data": {"type": "standard", "keys": None},
        "cosmo": {"type": "preset", "preset_overrides": dict(_BG_BOXES)},
    }
    c["epochs"] = 25
    c["scheduler_type"] = "exp"
    # 0.938^25 ≈ 0.20 — the same 5x LR decay the 100-epoch encoder-finetune recipe uses (gamma
    # 0.984), rescaled to 25 epochs. Keeping 0.984 here would only decay to ~0.67.
    c["scheduler_kwargs"] = {"gamma": 0.938, "warmup_steps": 0}
    c["project"] = "glass-pretraining"
    return c


for _arm, _store in (("sc8a1", _BGP_SC8A1), ("sc8", _BGP_SC8), ("a1", _BGP_A1)):
    kids_legacy_bgp_experiments[f"kids_legacy_hybrid_nla_m_bgp_z8_resnet_{_arm}_bg8"] = \
        _hybrid_bgp_bg8(_store, _HYB_CKPT[_arm])


# === M8 — ⭐ THE NEW PRODUCTION FOUNDATION: sc8a1, FULL 15-parameter set =========================
#
# User decision 2026-08-16: the foundation NPE now infers the **complete** parameter vector —
# the 9 cosmology/IA params AND the 6 per-tomo-bin galaxy biases — on the **sc8a1** arm, which is
# the single production shear estimator from this date. This supersedes the 9-param
# `kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1` row as the DEFAULT NPE training parameter set;
# that row is kept for continuity of the arm comparison, not as the production target.
#
# Why infer b_g rather than marginalise it implicitly (M6b, artifacts/M6b_REPORT.md): the counts
# normalisation cancels b_g in the bandpowers but NOT in the maps, and the measured sensitivity is
# **+5.2 sigma(Omega_m) per unit b_g — unchanged by drawing b_g from the prior at generation time**.
# Prior-marginalisation only moved the unbiased point to b_g* ~ 1.12; it did not make the network
# insensitive. Putting b_g in the inference vector is what lets the posterior absorb that direction
# instead of projecting it onto Omega_m.
#
# ⚠️ The 25-epoch M7 probe (artifacts/M7_REPORT.md) is NOT a counter-argument to this row. It failed
# on TRAINING BUDGET, not on physics: a fresh 8-D flow warm-started for 25 epochs came out
# miscalibrated (TARP 1.2-3.0 vs 0.009-0.013 for the 9-param runs) with one NaN-sample run. This row
# avoids that failure mode by training from the foundation recipe at FULL length rather than
# warm-starting a short fine-tune.
_BG_PARAMS = [f"b_g_bin{_i}" for _i in range(1, 7)]

# 25% longer than the 100-epoch foundation (user, 2026-08-16): 15 inference dims vs 9 is a harder
# density-estimation problem, and the M7 failure was under-training.
# ⚠️ The scheduler is deliberately NOT rescaled. `cyclic` here is STEP-based
# (cyclic_period_steps=6000, warmup=2000), so a longer run simply completes more cycles at the same
# LR envelope — unlike the `exp` schedules elsewhere in this file, whose gamma MUST be re-derived
# when the epoch count changes (cf. the bg8 rows, gamma 0.984 -> 0.938 for 100 -> 25 epochs).
_P15_EPOCHS = 125
_P15_REPEATS = (0, 1, 2, 3, 4)


def _hybrid_bgp_p15(data_patterns, band_ckpt, repeat_indices=_P15_REPEATS):
    """The 9-param sc8a1 foundation recipe, widened to the full 15-param inference vector.

    Everything except the inference vector, the b_g prior boxes and the epoch count is inherited
    from `_hybrid_bgp`, so this row cannot drift from the validated arch/tuning.
    """
    c = _hybrid_bgp(data_patterns, None, band_ckpt, repeat_indices=repeat_indices)
    # Derive the 9 from the inherited config rather than re-listing them: if the foundation's
    # parameter vector ever changes, this row follows it instead of silently disagreeing.
    base9 = list(c["cosmo_param_names"])
    assert len(base9) == 9, f"expected the 9-param foundation vector, got {base9}"
    c["cosmo_param_names"] = base9 + list(_BG_PARAMS)
    # The 9 cosmo/IA boxes come from the shared COSMO_PARAM_PRESET_MINMAX; only the 6 b_g boxes are
    # new, so they ride as preset_overrides rather than mutating the shared constant.
    # _build_cosmo_preset_scaler RAISES on a parameter with no box, so a missing entry here is a
    # loud failure at config build, not a silent mis-scaling.
    c["scaler_options"] = {
        "data": {"type": "standard", "keys": None},
        "cosmo": {"type": "preset", "preset_overrides": dict(_BG_BOXES)},
    }
    c["epochs"] = _P15_EPOCHS
    return c


kids_legacy_bgp_experiments["kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1_p15"] = \
    _hybrid_bgp_p15(_BGP_SC8A1, _BAND_CKPT_BGP)


# === M8b — the M8 foundation with a LARGER NDE head =============================================
# User request 2026-08-16: one extra run at higher density-estimator capacity, to test whether the
# 15-D posterior is head-limited rather than encoder-limited. Identical to M8 in every other
# respect (same store, same frozen Stage-I band, same 125 epochs, same repeat seed) so the flow
# capacity is the ONLY axis that varies.
#
# `flow_type='nsf'` dispatches to `build_nsf(hidden_features=..., num_transforms=...)`
# (src/ml/models/custom_sbi.py; both are popped from flow_kwargs in
# src/ml/models/lightning/npe.py:set_up_model). The M8 baseline sets hidden_features=32 and leaves
# num_transforms at its default of 5 — so "double the hidden dim, two more transforms" is:
#     hidden_features 32 -> 64      (2x)
#     num_transforms   5 -> 7       (+2)
# ⚠️ num_transforms is NOT in the baseline's flow_kwargs; it comes from the build_nsf default. Read
# that default before changing this, rather than assuming the dict shows the whole configuration.
#
# Run as ONE repeat at r0 so it is directly comparable to M8 r0 (same split_seed, same train/val
# split) — a capacity comparison against a different seed would confound the two axes.
_P15_BIGFLOW_REPEAT = (0,)


def _hybrid_bgp_p15_bigflow(data_patterns, band_ckpt):
    c = _hybrid_bgp_p15(data_patterns, band_ckpt, repeat_indices=_P15_BIGFLOW_REPEAT)
    base = dict(c.get("flow_kwargs") or {})
    c["flow_kwargs"] = {**base, "hidden_features": 2 * base.get("hidden_features", 32),
                        "num_transforms": 7}
    return c


kids_legacy_bgp_experiments["kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1_p15_bigflow"] = \
    _hybrid_bgp_p15_bigflow(_BGP_SC8A1, _BAND_CKPT_BGP)


# === M4 precursor — whitening-dimension study on the p15 r0 summary ==============================
# Before committing the M4 NLE chain to a whitening/truncation dimension, measure what truncation
# actually COSTS in constraining power on the summary M4 will use.
#
# ⚠️ The inherited "whiten to k=8" recipe is a NO-OP on this architecture. In the pre-BGP era the
# compressor emitted 16-D and k=8 was a genuine 2x truncation that fixed NLE over-confidence. The M8
# z8 hybrid sets model_kwargs['hybrid_output_dim']=8, which OVERRIDES latent_dim=16 for the encoder
# output (src/ml/utils.py:384-387) — so the summary fed to the flow is **8-D**, k=8 truncates
# nothing, and k>8 is undefined (PCA cannot exceed the input rank). The study therefore sweeps
# k = 2, 4, 6, 8 with k=8 as the pure-whiten INVARIANCE CONTROL (an invertible affine map: extracted
# MI must be unchanged vs the raw 8-D embedding, up to optimisation noise).
#
# This row is CACHE-ONLY (`run_training: False`): build_embedding_dataloaders computes and persists
# the frozen summary, and fit_nde_on_embeddings is skipped (src/ml/embeddings/train.py:407). The
# cache stores **unscaled** z + theta (_save_embedding_cache), so ONE cache serves every k in the
# sweep — whitening is applied locally, per k, on top of it. Deliberately NOT setting
# whiten_embeddings here: a whitener persisted at some arbitrary k would be the wrong artefact to
# hand the finetune, and picking k is the whole point of the study.
_P15_EXP = "kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1_p15"


def _p15_embcache(repeat_index=0):
    """Cache-only embeddings run over the GLASS sc8a1 store, on the FROZEN p15 encoder at one repeat.

    `cosmo_param_names` / the b_g prior boxes are READ FROM the p15 experiment rather than re-listed,
    so this can never disagree with the encoder it is caching (a mismatch would mis-scale theta and
    silently corrupt every downstream information estimate).

    match_num_cosmo=False => repeat_match "_{i}", which resolves the source encoder's
    `pretrain_ncosmoNone_{i}` checkpoint — the established idiom for the z6/z8 NLE chains.
    """
    src = kids_legacy_bgp_experiments[_P15_EXP]
    return {
        "data_patterns": src["data_patterns"],
        "eb_map_variant": src.get("eb_map_variant"),
        "dataset_quantities": [],          # overwritten from the source encoder at runtime
        "latent_dim": 8,                   # = hybrid_output_dim, the p15 summary width
        "epochs": 1,                       # unused (run_training False) — kept finite, not 0
        "batch_size": 128,
        "project": "bgp-nle",
        "cosmo_param_names": list(src["cosmo_param_names"]),   # the full 15
        "scaler_options": {k: dict(v) for k, v in src["scaler_options"].items()},
        "inference_mode": "nle",
        "repeat_indices": [int(repeat_index)],
        "match_num_cosmo": False,
        "scale_embeddings": False,
        "whiten_embeddings": None,         # raw cache; k applied locally in the sweep
        "run_training": False,             # CACHE ONLY
        "run_evaluation": False,
    }


kids_legacy_bgp_experiments["bgp_p15_embcache_r0"] = _p15_embcache(0)


# --- GUARD: the final summary width must be what we think it is -------------------------------
# This exists because the width silently changed under us. The flow conditions on
# `hybrid_output_dim` when set, and only falls back to `latent_dim` when it is None
# (src/ml/utils.py:384-387) — so `latent_dim: 16` in a config dict does NOT mean the summary is
# 16-D. `_hybrid_lmin50_z8` sets hybrid_output_dim=8, which is how the p15/M8 rows ended up on an
# 8-D summary while reading as latent_dim=16. Verified empirically: the embed job over the trained
# p15 r0 encoder reported "Computed embeddings ... with dimension 8".
#
# ⚠️ DOUBLE-CHECK THE FINAL SUMMARY SIZE (user, 2026-08-17). Call this on any row whose summary
# width matters, so a future edit that changes it fails LOUDLY at config-build time instead of
# silently retraining a different architecture.
def _assert_final_summary_dim(c, expected, label):
    mk = c.get("model_kwargs") or {}
    got = mk.get("hybrid_output_dim")
    source = "hybrid_output_dim"
    if got is None:
        got, source = c.get("latent_dim"), "latent_dim (hybrid_output_dim unset)"
    assert got == expected, (
        f"{label}: FINAL SUMMARY WIDTH IS {got} (from {source}), expected {expected}. "
        f"The flow conditions on hybrid_output_dim when set, NOT on latent_dim "
        f"(latent_dim here = {c.get('latent_dim')}, the CONCAT width)."
    )
    return c


# === M9 — WIDER SUMMARY: does the 8-D bottleneck starve the 15-param posterior? ==================
# Motivation (user, 2026-08-17), and it is NOT the bigflow/capacity hypothesis. Measured on the
# SAME sc8a1 store: FoM(omega_m,sigma_8) drops 24.7% / 24.8% (paired, r0 / r3) going 9-param ->
# 15-param, and that is NOT explained by physics (both arms marginalise the same b_g uncertainty,
# so exact inference gives the same 2-D marginal) NOR by overconfidence (the 15-param arm is BETTER
# calibrated). See ../../simulation-runs/galaxy-bias-priors/artifacts/FOM_15_VS_9.md.
#
# The hypothesis this row tests: with 6 of 15 outputs being b_g, the NLL is dominated by dimensions
# that carry no cosmology, while an 8-D summary has too little room to hold 15 parameters' worth of
# information — so omega_m/sigma_8 get squeezed out of the bottleneck.
#
# Two widths move together (user decision, both to 16):
#   latent_dim        16 -> 24   (= band 8 + patch 16, since dim_patch = latent_dim - dim_band)
#   hybrid_output_dim  8 -> 16   (the FINAL summary the flow conditions on)
# ⚠️ Two axes move at once, so a FoM change cannot be attributed to the map-branch width vs the
# bottleneck width without a further run holding one fixed. Recorded deliberately.
#
# The frozen Stage-I band is UNCHANGED at 8-D (bandpower_latent_dim stays 8), so the existing
# per-repeat band checkpoints still pair 1:1 and no new Stage I is needed.
_P15_Z16_REPEATS = (0, 1, 3)   # splits whose M8 counterparts all escaped; r0/r3 have evaluated FoMs
                               # to pair against directly, and r2's split (3 trapped inits) is avoided.


def _hybrid_bgp_p15_z16(data_patterns, band_ckpt, repeat_indices=_P15_Z16_REPEATS):
    c = _hybrid_bgp_p15(data_patterns, band_ckpt, repeat_indices=repeat_indices)
    c["latent_dim"] = 24                                   # concat = band 8 + patch 16
    c["model_kwargs"] = {**c["model_kwargs"], "hybrid_output_dim": 16}
    return c


kids_legacy_bgp_experiments["kids_legacy_hybrid_nla_m_bgp_z16_resnet_sc8a1_p15"] = \
    _assert_final_summary_dim(_hybrid_bgp_p15_z16(_BGP_SC8A1, _BAND_CKPT_BGP), 16,
                              "kids_legacy_hybrid_nla_m_bgp_z16_resnet_sc8a1_p15")


# === M10 — LOADING-PATCH VALIDATION + a 9-param wider-summary control ===========================
# Two jobs in one cheap row (user, 2026-08-18):
#
# (1) **Validate the cross-width warm start.** The variate plan (see
#     `.claude/runs/training-runs/production-training-runs/`) warm-starts 15-param z16 models from
#     the 9-param z8 breakthroughs, i.e. ACROSS a summary-width change. This row exercises exactly
#     that load path first, on a cheap 9-param run, before the expensive variate rows depend on it.
#     ⭐ **No code patch was needed.** `load_partial_weights` (src/ml/models/lightning/utils.py)
#     already skips shape-mismatched keys and loads the rest. Verified locally against
#     `…_sc8a1/pretrain_ncosmoNone_0` -> the z16 geometry: **125/129 keys loaded**, with exactly the
#     4 resized final-layer tensors skipped —
#       patch_encoder.head.2.{weight,bias}: (8,256)/(8,) vs (16,256)/(16,)   <- map summary head
#       hybrid_head.{weight,bias}:          (8,16)/(8,)  vs (16,24)/(16,)    <- final summary
#     The whole CNN backbone + band encoder transfer. This is precisely "load the compressor weights
#     UP TO the final layers with differing dimensions".
#     ⚠️ Use `pretrained_embedding_ckpt_path` (partial, tolerant), NOT `checkpoint_path` — the latter
#     goes through `NPELightningModule.load_from_checkpoint` -> `load_state_dict` STRICT, on purpose
#     ("so a genuine architecture mismatch still surfaces", npe.py:141). That strictness is what
#     would have caught the silent 8-vs-16 summary change, so it must NOT be loosened.
#     ⚠️ `load_partial_weights` prints "Some source keys not used — prefix may be wrong" on EVERY
#     encoder-only load from a full-model checkpoint (353 unused keys here = flow + duplicate
#     embedding refs). It is a FALSE alarm in this workflow — do not chase it.
#
# (2) **Does a wider summary help 9-param inference?** Same z16 geometry as M9 but on the ORIGINAL
#     9-param vector, warm-started from that seed's own 9-param foundation. Expected: little or no
#     gain (the user's prior) — 9 params fit comfortably in 8 dims, so this isolates whether the M9
#     15-param gain is really about *bottleneck capacity vs parameter count*, rather than the wider
#     summary just being better in general. A NULL here strengthens the 15-param interpretation.
_P15_9P_Z16_REPEATS = (0, 3)   # the two 9-param seeds with evaluated FoMs (r0 21.14, r3 21.79)
_SC8A1_9P_CKPT = f"{_CKPT}/kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1/"


def _hybrid_bgp_9p_z16(data_patterns, repeat_indices=_P15_9P_Z16_REPEATS):
    c = _hybrid_bgp(data_patterns, None, _BAND_CKPT_BGP, repeat_indices=repeat_indices)
    c.pop("pretrained_band_ckpt_path", None)   # band arrives inside the loaded embedding_net
    c.pop("freeze_band", None)
    c["pretrained_embedding_ckpt_path"] = _SC8A1_9P_CKPT
    c["freeze_embedding_net"] = False          # finetune the whole encoder
    c["match_num_cosmo"] = False               # resolve the source ckpt per-repeat as "_{i}"
    c["latent_dim"] = 24                       # concat = band 8 + patch 16   (M9 geometry)
    c["model_kwargs"] = {**c["model_kwargs"], "hybrid_output_dim": 16}
    return c


kids_legacy_bgp_experiments["kids_legacy_hybrid_nla_m_bgp_z16_resnet_sc8a1_9p_warm"] = \
    _assert_final_summary_dim(_hybrid_bgp_9p_z16(_BGP_SC8A1), 16,
                              "kids_legacy_hybrid_nla_m_bgp_z16_resnet_sc8a1_9p_warm")


# === M11 — p15 warm-started from the p9 foundation. SUPERSEDED by M11b below =====================
# ⛔ The exp-decay recipe here is ABANDONED (user, 2026-08-26): it plateaued at -7.74/-7.77/-7.84,
# below M9's own non-breakthrough seeds, and its 50-epoch budget made M9 r1's epoch-102 breakthrough
# unreachable by construction. The builder is kept ONLY because M11b derives from it; the row itself
# should not be launched. Use `..._p15_warm_cyc` (M11b).
#
# The warm-start mechanics below ARE still current and are what M11b inherits:
# ⭐ LAUNCH-VERIFY `Loaded keys: 125` (129 would mean the width did not change; 0 means the
#   arch/ckpt pairing is wrong). The 8-D p9 foundation loads into this 16-D geometry skipping
#   exactly the 4 resized final-layer tensors (map summary head + hybrid head).
# ⚠️ MUST use `pretrained_embedding_ckpt_path` (partial, tolerant), never `checkpoint_path` (STRICT
#   `load_state_dict`, npe.py:141) — that strictness is deliberate and must not be loosened.
# ⚠️ "Some source keys not used — prefix may be wrong" is a FALSE alarm on every encoder-only load
#   from a full-model checkpoint. Do not chase it.
# ⚠️ An `exp` gamma MUST be re-derived whenever the epoch count changes (cf. bg8, 0.984 -> 0.938 for
#   100 -> 25 epochs). 0.96832 below is the 50-epoch value for the usual ~0.20 total decay.
#
# Repeats are the full (0..4): the p9 foundation trained all five, so every repeat has a parent —
# unlike M9, which only ever ran (0, 1, 3).
_P15_WARM_REPEATS = (0, 1, 2, 3, 4)


def _hybrid_bgp_p15_z16_warm(data_patterns, repeat_indices=_P15_WARM_REPEATS):
    """M9's 15-param/16-D recipe, but inheriting the trained p9 foundation encoder (M10's load path).

    Built from `_hybrid_bgp_p15_z16` so the inference vector, the b_g prior boxes and the arch
    cannot drift from the validated 15-param row; the warm-start keys mirror `_hybrid_bgp_9p_z16`.
    """
    c = _hybrid_bgp_p15_z16(data_patterns, _BAND_CKPT_BGP, repeat_indices=repeat_indices)
    c.pop("pretrained_band_ckpt_path", None)   # the band arrives inside the loaded embedding_net
    c.pop("freeze_band", None)
    c["pretrained_embedding_ckpt_path"] = _SC8A1_9P_CKPT   # the KNOWN-GOOD p9 foundation
    c["freeze_embedding_net"] = False          # resume/finetune the whole encoder
    c["match_num_cosmo"] = False               # resolve the source ckpt per-repeat as "_{i}"
    c["epochs"] = 50                           # the warm start already carries the hard part
    c["lr"] = 0.0002                           # high-LR resume, as on every other finetune here
    c["scheduler_type"] = "exp"
    c["scheduler_kwargs"] = {"gamma": 0.96832, "warmup_steps": 0}  # 0.96832^50 ~ 0.20
    return c


kids_legacy_bgp_experiments["kids_legacy_hybrid_nla_m_bgp_z16_resnet_sc8a1_p15_warm"] = \
    _assert_final_summary_dim(_hybrid_bgp_p15_z16_warm(_BGP_SC8A1), 16,
                              "kids_legacy_hybrid_nla_m_bgp_z16_resnet_sc8a1_p15_warm")


# === M11b — p15 WARM START on the FOUNDATION's CYCLIC LR, 75 epochs (user-directed 2026-08-26) ===
# The exp-decay warm rows above are NOT working and the reason is now measured, not guessed.
# Best-checkpoint val_log_prob (lower is better; `find_best_checkpoint` takes the MINIMUM):
#     M9  p15 (COLD, cyclic, 125 ep):  r0 -7.9109 @ep72   r1 -8.3320 @ep102   r3 -7.9523 @ep73
#     M11 p15_warm (exp, 50 ep):       r0 -7.7386 @ep18   r2 -7.7673 @ep17    r3 -7.8443 @ep26
# Two things follow. First, the warm rows are tracking WORSE than M9's own non-breakthrough seeds,
# so the warm start is not yet buying what it should. Second — and this is the structural error —
# ⭐ **M9 r1's breakthrough landed at EPOCH 102, which is past M11's ENTIRE 50-EPOCH BUDGET.** The
# 125->50 shortening (c72a855) therefore made reproducing the one good run IMPOSSIBLE BY
# CONSTRUCTION, whatever the schedule did. Epoch budget was the binding constraint, not the decay.
#
# ⚠️ NOTE the breakthrough was a COLD run (user 2026-08-26): "That breakthrough happened without the
# warm start." So r1 is an existence proof that the 15-param optimum is REACHABLE, not evidence that
# warm-starting reproduces it. A warm start SHOULD help substantially — that is the hypothesis this
# row tests — but it has not been demonstrated yet, and nothing here should be read as assuming it.
#
# THE CHANGE: the FOUNDATION model's own cyclic schedule, applied to p15 on G1.
#     cyclic, warmup=2000, min_factor=0.1, cyclic_period_steps=6000, lr 2e-4
# taken verbatim from `kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1` (the p9 foundation these runs
# warm-start FROM) — the same schedule M9 ran, which is the only setting under which a breakthrough
# has ever been observed here. Cyclic matters mechanistically: a monotone decay anneals the model
# into whichever basin it found early, while the periodic LR restarts are what let a run climb OUT
# of the -7.9 plateau late (r1 at ep102). Decay cannot express that.
#
# At 805 iters/epoch (measured), 75 epochs = ~60 375 steps = ~10 full cycles, vs the foundation's
# ~13.4 at 100 epochs. ~6:17/epoch measured => ~7.9 h/run.
#
# ⭐ SEPARATE EXPERIMENT NAME IS LOAD-BEARING (same trap as the fixed-LR A/B): writing cyclic
# checkpoints into the exp rows' `pretrain_ncosmoNone_{i}/` folders would leave two recipes in one
# directory, and `find_best_checkpoint` picks the global minimum across the folder — silently mixing
# the comparison. The exp-decay checkpoints stay where they are as the record of that attempt.
def _hybrid_bgp_p15_z16_warm_cyc(data_patterns, repeat_indices=_P15_WARM_REPEATS):
    """p15 warm start, but on the p9 FOUNDATION's cyclic LR and a 75-epoch budget."""
    c = _hybrid_bgp_p15_z16_warm(data_patterns, repeat_indices=repeat_indices)
    c["epochs"] = 75
    c["lr"] = 0.0002
    c["scheduler_type"] = "cyclic"
    c["scheduler_kwargs"] = {"warmup": 2000, "min_factor": 0.1, "cyclic_period_steps": 6000}
    return c


kids_legacy_bgp_experiments["kids_legacy_hybrid_nla_m_bgp_z16_resnet_sc8a1_p15_warm_cyc"] = \
    _assert_final_summary_dim(_hybrid_bgp_p15_z16_warm_cyc(_BGP_SC8A1), 16,
                              "kids_legacy_hybrid_nla_m_bgp_z16_resnet_sc8a1_p15_warm_cyc")


# === M4a — the WHITENED (k=8) NLE chain, Stage A: GLASS pretrain ================================
# The NLE half of the multifidelity stack, resumed for this campaign (user 2026-08-18). Per repeat
# r: freeze the 9-param sc8a1 foundation encoder, cache its 8-D summary over the FULL GLASS bgp
# store, then train a flow q(z | theta) on those cached embeddings. Stage B (the Gower ens9
# finetune) is written when the Gower stores land — S1 is still generating.
#
# ⭐ **9 parameters, not 15.** The settled campaign direction (STATUS.md §CAMPAIGN DIRECTION
# SETTLED) makes b_g a VARIATE rather than part of the foundation's inference vector, and these rows
# hang off the 9-param breakthroughs — so the theta vector here is `_COSMO_9`, inherited from the
# source encoder rather than re-listed.
#
# ⭐ **whiten_k=8 is PURE-WHITEN, not truncation.** The z8 summary is 8-D
# (`hybrid_output_dim=8` overrides `latent_dim`), so k=8 keeps full rank: an invertible affine map
# that leaves the extracted mutual information exactly unchanged and buys only CONDITIONING.
# That is deliberate and measured — `../bgp-nle-whitening-dim/artifacts/KSWEEP_REPORT.md` swept
# k=2/4/6/8 on this very architecture and found NO free truncation (k=6 already costs 13 % of
# FoM(omega_m,sigma_8), k=4 costs 22 %), while the k=8 invariance control reproduced the raw
# summary to 0.003 nats. ⇒ Any k<8 here would throw away constraining power for nothing.
#
# ⚠️ The pretrain val NLL is NOT commensurable across repeats — it depends on each foundation seed's
# embedding scale. A wide spread is expected and is not a quality signal; compare at eval.
_NLE_REPEATS = (0, 1, 2, 3, 4)          # the 5 foundation seeds (r0..r4 all cleared the -4.5 wall)
_NLE_EPOCHS = 150
_BGP_NLE_PROJECT = "bgp-nle"            # keep this campaign's NLE runs in one W&B project


def _nle_pretrain_bgp(data_patterns, repeat, cosmo_param_names=None, preset_overrides=None):
    """Stage-A NLE pretrain row for one repeat, on a bgp GLASS store.

    `eb_variant=None` because the bgp bakes write BARE `E` groups (no `--keep-variant-tag`), exactly
    as the hybrid rows above read them. Passing a tag here would look for `E_<tag>` and find nothing.
    """
    kw = {}
    if cosmo_param_names is not None:
        kw["cosmo_param_names"] = cosmo_param_names
    if preset_overrides is not None:
        kw["preset_overrides"] = preset_overrides
    c = _nle_pretrain(data_patterns, None, whiten_k=8, epochs=_NLE_EPOCHS, **kw)
    c["project"] = _BGP_NLE_PROJECT
    return _nle_bake_repeat(c, repeat)


for _r in _NLE_REPEATS:
    # source encoder is passed at the train_embeddings.py CLI:
    #   embed --target glass_nle_pretrain_nla_m_bgp_z8_r<r> \
    #         --sources kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1 --gpu v100
    # match_num_cosmo=False (from the factory) => repeat_match "_{r}", which resolves that
    # encoder's `pretrain_ncosmoNone_{r}` checkpoint AND trains this flow as repeat r.
    kids_legacy_bgp_experiments[f"glass_nle_pretrain_nla_m_bgp_z8_r{_r}"] = \
        _nle_pretrain_bgp(_BGP_SC8A1, _r)


# === M5a — GLASS IA sub-variate encoder finetune (the `nla` variate) ============================
# Warm-start the whole 9-param foundation encoder (band + map) onto the `nla` IA store and train it
# with a FRESH flow head. This is the GLASS-side variate chain: it teaches the compressor an IA
# model it never saw, so the downstream Gower finetune inherits a representation that is not
# over-fitted to NLA-M.
#
# ⭐ Built from `_hybrid_bgp`, NOT from `config.kids_legacy._encoder_finetune_z8`. The latter starts
# from a bare `_hybrid_lmin50_z8()` with NO `map_kwargs`, so it would build a UNet map encoder and
# then be handed PreActResNet weights — the load would silently drop the whole map branch. The
# archived checklist flags this as CRITICAL for exactly this reason; going through `_hybrid_bgp`
# keeps `_RESNET_MAPKW` attached by construction.
#
# ⚠️ theta is `_COSMO_8_NLA` (8 params: the NLA-M vector minus `b_ia`, which `nla` does not have) and
# a_ia MUST be re-boxed to U[-6,6] — the global preset's a_ia box is the NLA-M range (4.48, 7.0), so
# without the override a_ia is mis-scaled in BOTH training and eval.
_BGP_NLA = f"{_GPU5}/glass_bgp_nla_f16_sc8a1_{_EB}/output_*.h5"     # G5 bake (job 1344916)
_VARIATE_REPEATS = (0, 1, 2, 3, 4)


def _encoder_finetune_bgp(data_patterns, cosmo_param_names, preset_overrides=None,
                          repeat_indices=_VARIATE_REPEATS, scheduler="fixed"):
    """Foundation-warm-started encoder finetune on a bgp GLASS sub-variate store, 8-D summary.

    Same geometry as the foundation (z8) per the settled SUMMARY-WIDTH RULE: 16-D is reserved for
    the FINAL 15-param variate rows that actually infer b_g. Here b_g is only marginalised, so the
    width does not matter and the 9-param warm start carries the load.
    """
    c = _hybrid_bgp(data_patterns, None, _BAND_CKPT_BGP, repeat_indices=repeat_indices)
    c.pop("pretrained_band_ckpt_path", None)   # the band arrives inside the loaded embedding_net
    c.pop("freeze_band", None)
    c["pretrained_embedding_ckpt_path"] = _SC8A1_9P_CKPT
    c["freeze_embedding_net"] = False          # finetune the whole encoder
    c["match_num_cosmo"] = False               # resolve the source ckpt per-repeat as "_{i}"
    c["cosmo_param_names"] = list(cosmo_param_names)
    # ⭐ FIXED LR IS THE DEFAULT (user, 2026-08-26). These encoder finetunes OVERFIT LATE under the
    # old `exp` decay: on `nla_z`, r1 peaked at epoch 56 and r2 at epoch 41 of 100, then val degraded
    # badly (best -7.5215/-7.6226 vs final -5.83/-6.14). Best-checkpoint selection protects model
    # QUALITY (`find_best_checkpoint` takes the min), so this is an EFFICIENCY fix — roughly half the
    # epoch budget was being spent getting worse. A flat LR keeps the optimiser's step size (and its
    # regularising gradient noise) up instead of annealing the model into whichever basin it found
    # early. `scheduler="exp"` remains available and is pinned on the rows ALREADY TRAINED with it,
    # so their configs still describe the checkpoints on disk.
    # If a flat 2e-4 proves too hot to settle, the knob is `lr` — do NOT reintroduce decay silently.
    if scheduler in ("exp", "exponential"):
        c["scheduler_type"] = "exp"
        c["scheduler_kwargs"] = {"gamma": 0.984, "warmup_steps": 0}  # 0.984^100 ~ 0.20: 2e-4 -> 4e-5
    else:
        c["scheduler_type"] = "fixed"
        c["scheduler_kwargs"] = {"warmup_steps": 0}
    c["lr"] = 0.0002
    c["epochs"] = 100
    if preset_overrides:
        c["scaler_options"] = {
            "data": {"type": "standard", "keys": None},
            "cosmo": {"type": "preset", "preset_overrides": dict(preset_overrides)},
        }
    return c


# scheduler="exp" PINNED: this row's 5 repeats were TRAINED with the exp decay, before fixed LR
# became the default. Keep it so the config still describes the checkpoints on disk.
kids_legacy_bgp_experiments["glass_encoder_finetune_nla_bgp_z8"] = \
    _assert_final_summary_dim(
        _encoder_finetune_bgp(_BGP_NLA, _COSMO_8_NLA, preset_overrides=_A_IA_NLA_BOX,
                              scheduler="exp"), 8,
        "glass_encoder_finetune_nla_bgp_z8")


# === M6a — the `nla_z` GLASS compressor finetune (IA sub-variate: a_ia + b_z) ===================
# Same factory, same geometry, same warm start as the `nla` row above; ONLY the store and theta
# change. Store = the G6 bake (11 880 files, job 1349193, zero drops).
#
# ⚠️ theta is `_COSMO_9_NLAZ` = the NLA-M vector with `b_ia` REPLACED by `b_z` (9 params, not 8 like
# `nla`). a_ia MUST still be re-boxed to U[-6,6] — the global preset's a_ia box is the NLA-M range
# (4.48, 7.0), so without the override a_ia is mis-scaled in BOTH training and eval. `b_z` needs no
# override: it already carries a preset box (-25.2, 17.8) = ~5 sigma around N(-3.7, 4.3)
# (src/ml/data/constants.py:24). `_build_cosmo_preset_scaler` raises on any parameter with no box,
# so a missing one fails loudly rather than silently mis-scaling.
#
# The summary stays **8-D** even though theta is 9-D: per the settled SUMMARY-WIDTH RULE, 16-D is
# reserved for the FINAL 15-param rows that actually INFER b_g. Here b_g is only marginalised.
# (The 9-param nla_m foundation this warm-starts from is likewise 9-D theta on an 8-D summary.)
_BGP_NLA_Z = f"{_GPU5}/glass_bgp_nla_z_f16_sc8a1_{_EB}/output_*.h5"   # G6 bake (job 1349193)

# scheduler="exp" PINNED: r0/r1/r2/r4 are already trained and r3 is IN FLIGHT under the exp decay.
# Switching this row now would give r3 (and any future re-roll) a different recipe from its own
# siblings, which is exactly the inconsistency the straggler check would then trip over.
# The fixed-LR default applies to the NEXT variate encoder row, not retroactively to this one.
kids_legacy_bgp_experiments["glass_encoder_finetune_nla_z_bgp_z8"] = \
    _assert_final_summary_dim(
        _encoder_finetune_bgp(_BGP_NLA_Z, _COSMO_9_NLAZ, preset_overrides=_A_IA_NLA_BOX,
                              scheduler="exp"), 8,
        "glass_encoder_finetune_nla_z_bgp_z8")


# === M6a-LR — CONTROLLED FIXED-LR TEST on `nla_z` (user, 2026-08-26) ============================
# A/B against the row above: **identical in every respect except the LR schedule** (`fixed` here vs
# `exp` there, same base lr 2e-4, same 100 epochs, same store, theta, warm start and split seeds).
#
# ⭐ REPEATS (1, 2) ARE CHOSEN, NOT ARBITRARY. Those are precisely the two that overfit WORST under
# the exp decay — r1 peaked at epoch 56 and r2 at epoch 41 of 100, then fell to final-epoch -5.83 /
# -6.14 (best -7.5215 / -7.6226). Re-running the SAME repeat indices keeps the split seed identical,
# so the schedule is the only變 variable and the comparison is clean.
#
# ⚠️ SEPARATE EXPERIMENT NAME IS LOAD-BEARING. Writing fixed-LR checkpoints into the exp row's
# `pretrain_ncosmoNone_{1,2}/` folders would leave two recipes' checkpoints in one directory, and
# `find_best_checkpoint` picks the global min across the folder — so the A/B would silently
# contaminate the very rows it is meant to be compared against.
#
# WHAT TO COMPARE (best-checkpoint val, from the filenames — never the final-epoch value):
#   exp baseline: r1 **-7.5215** @ep56, r2 **-7.6226** @ep41   (pack: -7.52..-7.96 over all 5)
# Success = the fixed-LR runs match or beat those AND keep improving late (best at a LATE epoch
# rather than ~40-56), which is the actual point: recovering the ~half of the budget the exp runs
# spend getting worse. A better best-val is a bonus; a later best-epoch is the signal.
#
# ⭐ PROMOTED FROM A/B TO PRODUCTION (user, 2026-08-26): the test PASSED 2/2 -- r1 -8.0265@ep65 vs
# exp -7.5215@ep56, r2 -7.8100@ep63 vs exp -7.6226@ep41, both beating the baseline AND peaking far
# later. Standing policy is that a measured-better setup gets re-run, not merely noted, so this row
# now carries the FULL repeat set and becomes the `nla_z` production compressor. r1/r2 are already
# trained; only r0/r3/r4 need launching.
kids_legacy_bgp_experiments["glass_encoder_finetune_nla_z_bgp_z8_fixedlr"] = \
    _assert_final_summary_dim(
        _encoder_finetune_bgp(_BGP_NLA_Z, _COSMO_9_NLAZ, preset_overrides=_A_IA_NLA_BOX), 8,
        "glass_encoder_finetune_nla_z_bgp_z8_fixedlr")   # scheduler defaults to "fixed"


# === M5b — Stage-A NLE pretrain on the `nla` variate encoder ====================================
# Blocked on M5a (its source encoder). Written now so the launch is a one-liner when it lands.
# theta + a_ia box MUST match M5a exactly, or theta is mis-shaped/mis-scaled in training and eval.
for _r in _NLE_REPEATS:
    kids_legacy_bgp_experiments[f"glass_nle_pretrain_nla_bgp_z8_r{_r}"] = \
        _nle_pretrain_bgp(_BGP_NLA, _r, cosmo_param_names=_COSMO_8_NLA,
                          preset_overrides=_A_IA_NLA_BOX)


# === M3 / M4b — the GOWER transfer stage ========================================================
# ⏸️ WRITTEN AHEAD OF THE DATA (2026-08-18). The Gower foundation store (S1,
# `gower_mocks_nla_m_novd_bgp`, job 1345036) is still generating, so NEITHER ROW CAN BE SUBMITTED
# YET. They are written now so that landing the store is a one-command launch instead of a
# config-writing session.
#
# ⚠️ **`_BGP_GOWER_NLA_M` IS AN ASSUMED NAME.** The bake does not exist yet; this follows the GLASS
# convention exactly (`glass_mocks_nla_m_novd_bgp` -> `glass_bgp_nla_m_f16_sc8a1_<tag>`), so the
# Gower analogue should be `gower_bgp_nla_m_f16_sc8a1_<tag>`. **Verify with `data-ls` before the
# first submit** — a wrong path yields an empty glob, which surfaces as a confusing split error
# rather than a missing-file error.
#
# Both rows are **9-param**, matching their parents: M3 finetunes the 9-param foundation, M4b
# finetunes M4a's flow. Per the settled campaign direction b_g is a variate, not part of the
# foundation's inference vector, so nothing here infers b_g.
_BGP_GOWER_NLA_M = f"{_GPU5}/gower_bgp_nla_m_f16_sc8a1_{_EB}/output_*.h5"
_GOWER_TEST_IDS = "config/fixed_test_sets/gower_test_ids.json"


# --- M3: NPE ensemble finetune on Gower --------------------------------------------------------
# Whole-model load (encoder + NPE flow) from each foundation repeat, then finetune everything.
# ⭐ `map_kwargs=_RESNET_MAPKW` is CRITICAL and easy to lose: `_npe_finetune_z8` starts from a bare
# `_hybrid_lmin50_z8()` with no map kwargs, so without this it builds a UNet and is then handed
# PreActResNet weights. Unlike the tolerant `pretrained_embedding_ckpt_path` path, this row uses the
# STRICT `checkpoint_path` loader, which is correct here (architectures match exactly) and is the
# thing that would catch such a mismatch loudly.
# Split: 300 train/val cosmologies (80/20) with the 200 fixed-test ids held out.
def _npe_finetune_bgp(data_patterns=_BGP_GOWER_NLA_M):
    c = _npe_finetune_z8(_SC8A1_9P_CKPT, data_patterns=data_patterns, eb_variant=None)
    c["model_kwargs"] = {**c["model_kwargs"], "map_kwargs": _RESNET_MAPKW}
    c["fixed_test_sim_ids"] = _GOWER_TEST_IDS
    c["project"] = "gower-finetuning"
    return c


kids_legacy_bgp_experiments["gower_npe_finetune_nla_m_bgp_z8"] = \
    _assert_final_summary_dim(_npe_finetune_bgp(), 8, "gower_npe_finetune_nla_m_bgp_z8")


# --- M4b: NLE Stage B — ens9 flow finetune + MCMC eval, one row per repeat ----------------------
# Loads the Stage-A flow from `checkpoints/glass_nle_pretrain_nla_m_bgp_z8_r{r}/` AND resolves that
# run's persisted whitener (fit once on the GLASS train split, reused unchanged — never refit).
# `whiten_k` MUST equal the pretrain's k or the flow checkpoint shapes disagree.
#
# ⚠️ `warmstart_max_gap_nats=22.0`: a PURE-WHITEN of a rank-deficient 8-D summary legitimately
# inflates the epoch-0 GLASS->Gower gap (near-null PCs are fit sharply on GLASS and then amplify the
# fidelity shift when divided by a tiny sqrt-eigenvalue), so the default 12-nat guard would fire on a
# perfectly healthy warm start. Established on the z6/z8 chains; carried over unchanged.
#
# Runs on CORES64 via `embed --cpu` (flow training + a many-core MCMC eval), so it queues behind the
# sims — sims win.
for _r in _NLE_REPEATS:
    _ft = _nle_finetune(f"glass_nle_pretrain_nla_m_bgp_z8_r{_r}", ensemble_repeats=9,
                        whiten_k=8, warmstart_max_gap_nats=22.0,
                        gower_data=_BGP_GOWER_NLA_M, gower_eb=None)
    _ft["max_trainval_cosmos"] = [300]
    _ft["train_frac"] = 0.8
    _ft["val_frac"] = 0.2
    _ft["test_frac"] = 0.0        # test = the fixed 200 ids; fracs must sum to 1.0
    _ft["fixed_test_sim_ids"] = _GOWER_TEST_IDS
    _ft["project"] = _BGP_NLE_PROJECT
    kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_m_bgp_z8_r{_r}_ens9"] = \
        _nle_bake_repeat(_ft, _r)


# === ABLATION (user 2026-08-22) — does ADAPTING THE COMPRESSOR to Gower buy anything? ===========
# The whole M4b chain keeps the compressor FROZEN at its GLASS-trained state: Stage B only moves the
# flow. This ablation asks the obvious counterfactual — let the encoder itself see Gower first, then
# fine-tune the same Stage-A flow on the embeddings *that adapted encoder* produces. Prior campaigns
# found no gain; this re-tests it on the bgp foundation.
#
# ⭐ ONE repeat only, and it MUST be the SAME index on both halves (r4) so the encoder shift is the
# only thing that differs from the M4b r4 baseline. There is still a shift — that is the point.
#
# Precedent for this exact shape: `kids_legacy.py` → `gower_npe_finetune_nla_m_vicreg_v2` +
# `gower_nle_finetune_nla_m_vicreg_npesrc` (the "1-head NLE test"). Two mechanics carried over:
#
# 1. ⚠️ **`match_num_cosmo = True` on the NLE row.** With the default False, the SOURCE-encoder
#    lookup searches `"None_" + repeat` (train.py:217) — right for the GLASS foundation, whose runs
#    are `pretrain_ncosmoNone_{r}`, but WRONG here: an NPE finetune writes `finetune_ncosmo300_{r}`,
#    which carries no `None_` tag, so the lookup would find nothing. True makes it search the full
#    `ncosmo300_{r}`. This does NOT disturb anything else: the target's own match_string comes from
#    `format_ncosmo_tag` regardless, and BOTH the pretrained-flow checkpoint and the whitener are
#    resolved with the hardcoded `whiten_repeat_match = f"None_{repeat_idx}"` (train.py:375), so the
#    Stage-A pairing stays exactly as in the M4b baseline.
# 2. ⭐ **`ensemble_repeats = 1` on the NPE row.** We need ONE unambiguous adapted encoder. At the
#    production ens9 the run dirs are `finetune_ncosmo300_4_ens{0..8}` and `get_best_checkpoint`
#    searching `ncosmo300_4` would match all nine. At ens1 the dir is plain `finetune_ncosmo300_4`
#    (`models/utils.py:182` only appends `_ens{j}` when ensemble_repeats > 1). It is also 9x cheaper,
#    and the ensemble was only ever there for the NPE posterior, which this ablation does not use.
#
# The ablation's own embedding cache cannot collide with the baseline's: the cache key is the run
# name, and the experiment name differs (`..._adapt`).
_ABLATION_REPEAT = 4


# --- A1: the adapted encoder — NPE finetune on Gower, ONE member, repeat 4 ----------------------
kids_legacy_bgp_experiments[f"gower_npe_finetune_nla_m_bgp_z8_r{_ABLATION_REPEAT}_ens1"] = \
    _assert_final_summary_dim(
        {**_npe_finetune_bgp(), "ensemble_repeats": 1, "repeat_indices": [_ABLATION_REPEAT]}, 8,
        f"gower_npe_finetune_nla_m_bgp_z8_r{_ABLATION_REPEAT}_ens1")


# --- A2: the same Stage-A flow, fine-tuned on the ADAPTED encoder's embeddings ------------------
# Identical to M4b r4 in every field except `match_num_cosmo`; the source encoder is swapped at the
# CLI:  embed --cpu --target gower_nle_finetune_nla_m_bgp_z8_r4_ens9_adapt \
#             --sources gower_npe_finetune_nla_m_bgp_z8_r4_ens1
_ft_adapt = _nle_finetune(f"glass_nle_pretrain_nla_m_bgp_z8_r{_ABLATION_REPEAT}", ensemble_repeats=9,
                          whiten_k=8, warmstart_max_gap_nats=22.0,
                          gower_data=_BGP_GOWER_NLA_M, gower_eb=None)
_ft_adapt["max_trainval_cosmos"] = [300]
_ft_adapt["train_frac"] = 0.8
_ft_adapt["val_frac"] = 0.2
_ft_adapt["test_frac"] = 0.0
_ft_adapt["fixed_test_sim_ids"] = _GOWER_TEST_IDS
_ft_adapt["project"] = _BGP_NLE_PROJECT
_ft_adapt["match_num_cosmo"] = True          # see note 1 above — source lookup, nothing else
# ⭐ 2026-08-22, MEASURED: the first attempt (job 1347063) aborted on guard-c at
# **gap = 161.843 nats** (finetune ep0 val NLL 160.970 vs the Stage-A best -0.873), against the
# 22.0 the M4b baseline inherited — where r4's frozen-encoder run sat at just **1.701 nats**.
# That two-orders-of-magnitude jump is NOT a broken load, and the checks that rule that out are:
#   * the embedding cache path proves the ADAPTED encoder was used
#     (`pretrain_ncosmo300_4_ens0_gower_npe_finetune_nla_m_bgp_z8_r4_ens1/`), so match_num_cosmo
#     did its job;
#   * the flow-load block is BYTE-FOR-BYTE what the baseline prints — `Loaded keys: 110`,
#     `Shape mismatches: 0`, `Missing target keys: 0` (the "Some source keys not used" line is the
#     documented false alarm);
#   * the whitener and the flow both resolved out of r4's own Stage-A run dir.
# The cause is a genuine COORDINATE MISMATCH, and it is intrinsic to the ablation: the whitener is
# an affine map (standardise → PCA rotate → divide by sqrt-eigenvalue) FIT ON `E_GLASS` outputs, and
# `E_adapt` emits embeddings in its own scale and rotation. The near-null PCs — the same ~4 that made
# the 22-nat override necessary in the first place — divide by a tiny sqrt-eigenvalue and amplify
# that mismatch enormously.
# ⇒ Raised to 250.0, which is exactly the override the guard's own message invites ("Raise
# whiten_warmstart_max_gap_nats to override if this gap is genuinely expected") while still catching
# a pathological run. ⚠️ INTERPRET ACCORDINGLY: at a 162-nat start the warm start contributes
# essentially nothing, so this arm is honestly "adapted encoder + flow retrained from a poor init",
# NOT a warm start. That asymmetry is itself part of the answer to "does adapting the compressor
# help" — the Stage-A flow AND its whitener are encoder-specific, so adapting the encoder throws the
# transfer away. The fairer-but-unbuilt variant would REFIT the whitener on the adapted encoder's
# own train split; the code cannot express it today (`whiten_is_pretrain_source` is hard-wired to
# `pretrained_band_ckpt_path is None`, so refit and flow-warm-start are mutually exclusive), and even
# then the PCA axes of `E_adapt` need not correspond to `E_GLASS`'s.
_ft_adapt["whiten_warmstart_max_gap_nats"] = 250.0
kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_m_bgp_z8_r{_ABLATION_REPEAT}_ens9_adapt"] = \
    _nle_bake_repeat(_ft_adapt, _ABLATION_REPEAT)


# --- A3/A4: the ablation RE-RUN at the Stage-A budget, warm-started vs RANDOM INIT -------------
# ⭐ User 2026-08-22 ~18:1xZ, after the 50-epoch A2 result: cancel it and run TWO variants at 150
# epochs, one with the pre-trained NLE flow and one without.
#
# WHY the 50-epoch run had to be redone: its budget was sized for a warm start it never received.
# Every A2 member peaked at epoch 45-49 of 50 — still improving when the budget ran out — whereas
# the frozen-encoder baseline peaked at epoch 16-27 and went flat. Comparing FoM from that run would
# have charged the adapted arm for a training-budget shortfall on top of the encoder change, which
# is not the question being asked. 150 matches `_NLE_EPOCHS`, the Stage-A budget.
#
# ⭐ WHY THIS PAIR IS THE RIGHT CONTROL. A3 and A4 differ in `load_pretrained_flow` and NOTHING else,
# so they isolate exactly one thing: whether the GLASS Stage-A flow is worth anything once the
# encoder has moved. Two mechanics make that clean:
#   * `pretrained_band_ckpt_path` stays SET in both. It is what drives
#     `whiten_is_pretrain_source = (pretrained_band_ckpt_path is None)`, so BOTH variants REUSE the
#     same persisted Stage-A whitener rather than one of them refitting. Same z-space in both arms.
#   * the ep0 warm-start guard is only invoked under `if getattr(base_cfg,'load_pretrained_flow')`
#     (`embeddings_utils.py:876`), so A4 skips it automatically — no threshold fiddling needed, and
#     A3 keeps the real 250-nat guard.
# ⇒ Because both share an encoder AND a whitener, their val NLLs ARE commensurable **with each
#   other** (unlike either vs the frozen-encoder baseline, where the change of variables on z shifts
#   the density by log|det J|). **If A3 ≈ A4 at convergence, the pre-training contributed nothing.**
#
# NEW NAMES rather than re-using `..._adapt`: that dir already holds the 50-epoch run's checkpoints
# AND its `datasets/` cache, and `get_best_checkpoint` would happily return a stale 50-epoch member.
# Fresh names keep the superseded run intact as a record and remove the hazard entirely.
def _adapt_e150(load_flow: bool):
    c = _nle_finetune(f"glass_nle_pretrain_nla_m_bgp_z8_r{_ABLATION_REPEAT}", ensemble_repeats=9,
                      whiten_k=8, warmstart_max_gap_nats=250.0,
                      gower_data=_BGP_GOWER_NLA_M, gower_eb=None)
    c["max_trainval_cosmos"] = [300]
    c["train_frac"] = 0.8
    c["val_frac"] = 0.2
    c["test_frac"] = 0.0
    c["fixed_test_sim_ids"] = _GOWER_TEST_IDS
    c["project"] = _BGP_NLE_PROJECT
    c["match_num_cosmo"] = True        # source lookup -> finetune_ncosmo300_4 (the adapted encoder)
    c["epochs"] = _NLE_EPOCHS          # 150 — the Stage-A budget, not the warm-start budget
    c["load_pretrained_flow"] = bool(load_flow)
    return _nle_bake_repeat(c, _ABLATION_REPEAT)


# A3 — adapted encoder + the pre-trained Stage-A flow (the warm-start arm)
kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_m_bgp_z8_r{_ABLATION_REPEAT}_ens9_adapt_e150"] = \
    _adapt_e150(load_flow=True)

# A4 — adapted encoder + a RANDOM-INIT flow (the no-pretraining control)
kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_m_bgp_z8_r{_ABLATION_REPEAT}_ens9_adapt_e150_scratch"] = \
    _adapt_e150(load_flow=False)


# --- ARCHIVED: the stacked-ensemble ablation (k=40 / k=16) moved out 2026-08-26 ------------------
# The question ("do 5 concatenated compressors beat one?") was closed NO on both fidelities on
# 2026-08-25. Its ~230 lines of rows and result write-up now live in
# `config/archive/legacy_bgp_stack5.py` (dict: `bgp_stack5_experiments`), still merged by train.py
# so the rows stay launchable. Do not extend them.

# === M5b-k5 — Stage-A NLE pretrain on the `nla` encoder at whiten_k=5 ==========================
# ⭐ WHY k=5 AND NOT THE k=8 USED EVERYWHERE ELSE (user decision 2026-08-25), with the measurement
# that drove it. The k=8 M5c rows aborted on guard-c at 22.6-30.8 nats against a 22 threshold. That
# threshold is a PER-CHAIN calibration, and the `nla` chain's summary is far more rank-deficient
# than the `nla_m` chain it was calibrated on. Both whiteners, compared directly:
#
#   nla_m (G1): EVR 6.42e-1 3.21e-1 1.90e-2 1.06e-2 4.45e-3 2.48e-3 4.32e-4 1.02e-4
#               scales 2.27 ... 2.85e-2   -> max/min spread  79.5
#   nla   (G5): EVR 8.08e-1 1.36e-1 4.03e-2 1.35e-2 1.72e-3 2.93e-4 7.06e-5 1.16e-5
#               scales 2.54 ... 9.63e-3   -> max/min spread 264
#
# The nla summary's smallest PC carries 8.8x less variance and is divided by a scale 3.0x smaller,
# so a GLASS->Gower shift along that near-null direction is amplified ~3.3x more in scale (~11x in
# variance). Entering the NLL quadratically, nla_m's healthy 1-5 nats maps to ~10-50 here - which is
# exactly where the observed 22.6-30.8 landed. So the gap was the DOCUMENTED pure-whiten mechanism,
# not a broken warm start.
#
# k=5 removes the pathology at source rather than moving the tripwire: PCs 5-8 carry only
# 1.72e-3 + 2.93e-4 + 7.06e-5 + 1.16e-5 = 0.21% of the variance, so k=5 keeps 99.96% of it while
# dropping the directions whose tiny eigenvalues do the amplifying.
#
# ⚠️ NEW EXPERIMENT NAMES, deliberately. `whiten_k` changes the flow's context width, so a k=5 run
# writing into the k=8 rows' checkpoint dirs would leave `get_best_checkpoint` free to resolve a
# SHAPE-MISMATCHED k=8 checkpoint. Separate names keep the two spectra's artefacts apart.
# ⚠️ The paired M5c rows below MUST use the same k - the finetune reuses this run's persisted
# whitener and loads its flow, and both would be shape-wrong at a different k.
for _r in _VARIATE_REPEATS:
    _pre_k5 = _nle_pretrain_bgp(_BGP_NLA, _r, cosmo_param_names=_COSMO_8_NLA,
                                preset_overrides=_A_IA_NLA_BOX)
    _pre_k5["whiten_embeddings"] = {"k": 5}
    kids_legacy_bgp_experiments[f"glass_nle_pretrain_nla_bgp_z8_k5_r{_r}"] = _pre_k5


# === M6a-B — Stage-A NLE pretrain on the `nla_z` encoder ========================================
# GATED ON M6a (`glass_encoder_finetune_nla_z_bgp_z8`), which is its source encoder. Written ahead
# so the launch is a one-liner when the encoders land:
#   embed --target glass_nle_pretrain_nla_z_bgp_z8_k5_r<r> \
#         --sources glass_encoder_finetune_nla_z_bgp_z8 --gpu v100 --skip-smoke
# ⭐ **v100 IS CORRECT HERE** and is the only stage that may use one. This trains a small flow over
# CACHED EMBEDDINGS, not maps, so it fits the 16 GB card that the encoder finetune does not
# (the encoder needs ~15.2 GiB and OOMs there — see the M6a v100 failures).
#
# ⚠️ theta is `_COSMO_9_NLAZ` and a_ia is re-boxed to U[-6,6] — these MUST match M6a EXACTLY. The
# NLE flow q(t|theta) conditions on theta, so a mismatch mis-shapes AND mis-scales theta in both
# training and eval, silently.
#
# ⚠️ k=5, mirroring the proven `nla` chain: pure-whitening a rank-deficient 8-D summary amplifies
# the near-null PCs (tiny eigenvalues divide into the GLASS->Gower shift), which is what inflated
# the k=8 warm-start gap. On `nla`, PCs 5-8 held only 0.21% of the variance, so k=5 kept 99.96% of
# it while dropping the amplifiers.
# ⚠️⚠️ **THAT 0.21% FIGURE IS THE `nla` ENCODER'S SPECTRUM, NOT THIS ONE'S.** k=5 is the sensible
# prior here, not a measured fact for `nla_z`. When M6a lands, CHECK nla_z's own embedding
# eigenvalues before trusting it; if PCs 5-8 carry materially more variance, raise k and rename the
# rows (see the naming warning below).
# ⚠️ NEW EXPERIMENT NAMES per k, deliberately: `whiten_k` sets the flow's context width, so a k=5
# run writing into a k=8 row's checkpoint dir would let `get_best_checkpoint` resolve a
# SHAPE-MISMATCHED checkpoint. The paired Gower NLE (Stage B) rows MUST use the same k — they reuse
# this run's persisted whitener and load its flow, and both are shape-wrong at another k.
for _r in _VARIATE_REPEATS:
    _pre_nlaz_k5 = _nle_pretrain_bgp(_BGP_NLA_Z, _r, cosmo_param_names=_COSMO_9_NLAZ,
                                     preset_overrides=_A_IA_NLA_BOX)
    _pre_nlaz_k5["whiten_embeddings"] = {"k": 5}
    kids_legacy_bgp_experiments[f"glass_nle_pretrain_nla_z_bgp_z8_k5_r{_r}"] = _pre_nlaz_k5


# === M5c — the `nla` variate GOWER NLE finetune (S2), one row per repeat ========================
# WRITTEN 2026-08-25 on user direction, replacing the "M5c is NOT written" note above: S2
# (`gower_mocks_nla_novd_bgp`, job 1348702) landed, so the store these rows need now exists.
#
# ⚠️ S2 HOLDS ONLY THE 200 FIXED-TEST COSMOLOGIES (`--gower-sim-set fixed_test --num-sims 200`;
# 16 000 files / 80 per sim = 200). It is NOT shaped like S1, which has 509 sims and therefore let
# M4b hold out all 200 and still train on 300. A straight M4b clone pointed here would leave ZERO
# trainval cosmologies. Per the user (2026-08-25) the split is instead **100 test / 100 finetune**:
#   * `fixed_test_sim_ids` locks 100 ids into test (see the lock file's own `derivation` note);
#   * the other 100 fall through to the trainval pool, which `split_by_cosmology` SHUFFLES before
#     cutting by train_frac/val_frac -- i.e. training + model selection on a randomly shuffled
#     100-cosmology subset, which is what the user asked for.
#
# ⭐ WHY THE 100 ARE THE **SORTED-FIRST** 100, not the parent file's stratified prefix: sorted order
# is exactly what `N_test_cosmologies` trims to (`data_selection.py`: "trim the TEST set to the
# first N cosmologies by sorted sim_id"). Locking the sorted-first 100 here therefore selects the
# SAME cosmologies that `N_test_cosmologies=100` would pick on any other row -- so an S1/M4b model
# scored with `N_test_cosmologies=100` is directly comparable to these rows. That is the point of
# the user's requirement that the 100 test sims be the same and fixed across all variates.
# Checked before adopting it: sim_id is essentially uncorrelated with cosmology across the 200
# (|corr| <= 0.11 for omega_m/sigma_8/w), and the resulting test vs trainval halves match in mean
# and spread on all three, so the contiguous-in-id cut is statistically a random one.
#
# ⚠️ theta is `_COSMO_8_NLA` + `_A_IA_NLA_BOX`, NOT the 9-param NLA-M vector: `_nle_finetune`'s
# docstring requires cosmo_param_names / preset_overrides to MATCH THE PAIRED PRETRAIN, and the
# parent here is M5b (`glass_nle_pretrain_nla_bgp_z8_r{r}`), which is 8-param with a_ia ~ U[-6,6].
# Getting this wrong mis-shapes and mis-scales theta in both training and eval.
#
# whiten_k=8 and warmstart_max_gap_nats=22.0 carry over from the proven M4b chain unchanged.
# `run_evaluation=True` (set inside `_nle_finetune`) bundles the MCMC eval, so these rows produce
# their own ensemble_evaluation_results json -- no separate eval submit is needed.
_BGP_GOWER_NLA = f"{_GPU5}/gower_bgp_nla_f16_sc8a1_{_EB}/output_*.h5"   # S2 bake
_GOWER_TEST_IDS_100 = "config/fixed_test_sets/gower_test_ids_100.json"

for _r in _VARIATE_REPEATS:
    _ft_nla = _nle_finetune(f"glass_nle_pretrain_nla_bgp_z8_k5_r{_r}", ensemble_repeats=9,
                            whiten_k=5, warmstart_max_gap_nats=22.0,
                            gower_data=_BGP_GOWER_NLA, gower_eb=None,
                            cosmo_param_names=_COSMO_8_NLA,
                            preset_overrides=_A_IA_NLA_BOX)
    # NOT a hard 100: S2 finished at 15 911/16 000 files, so ONE trainval cosmology is missing and
    # only 199 of the 200 are on disk (the locked 100 test ids are ALL present - the loss is entirely
    # from the finetune half). A hard [100] therefore raised
    #   "Requested max_trainval_cosmos=100 but only 99 ... available after reserving the test set"
    # and killed all five rows (jobs 1349075-79). `None` means "every cosmology not in the locked
    # test set", which is what the 100/100 design actually intends, is robust to each variate's own
    # shortfall (S3/K2 will land short too), and keeps the ONE invariant the user asked for - the
    # SAME fixed 100 test cosmologies everywhere. Trainval size may then differ by a sim or two
    # between variates; if strict parity is ever wanted, pin every variate to the min count once
    # S3/K2 land.
    _ft_nla["max_trainval_cosmos"] = None
    _ft_nla["train_frac"] = 0.8
    _ft_nla["val_frac"] = 0.2
    _ft_nla["test_frac"] = 0.0     # test = the locked 100; fracs must sum to 1.0
    _ft_nla["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
    # ⭐ 50 epochs is FINAL for this row; the under-training hypothesis was TESTED AND REJECTED.
    # M5e (same rows at 150 ep, giving optimiser-step parity with M4b's 7 500) moved the cosmology
    # FoM by only 1.034x and left calibration flat -- see the M5e block below for the n=5 numbers.
    # Do NOT raise epochs here again; the M4b gap is not a convergence problem.
    _ft_nla["project"] = _BGP_NLE_PROJECT
    kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_bgp_z8_r{_r}_ens9"] = \
        _nle_bake_repeat(_ft_nla, _r)

# --- M5d: the 100-epoch probe, single repeat. TEST COMPLETE, superseded by M5e -----------------
# One repeat at epochs=100 under its own name, probing the under-training hypothesis without
# disturbing the shipped 50-epoch rows (retraining those in place would re-roll all 45 members --
# there is no torch seeding on the train path). M5e then ran all 5 repeats at 150.
_E100_REPEAT = 4
_ft_nla_e100 = _nle_finetune(f"glass_nle_pretrain_nla_bgp_z8_k5_r{_E100_REPEAT}", ensemble_repeats=9,
                             whiten_k=5, warmstart_max_gap_nats=22.0,
                             gower_data=_BGP_GOWER_NLA, gower_eb=None,
                             cosmo_param_names=_COSMO_8_NLA,
                             preset_overrides=_A_IA_NLA_BOX)
_ft_nla_e100["max_trainval_cosmos"] = None
_ft_nla_e100["train_frac"] = 0.8
_ft_nla_e100["val_frac"] = 0.2
_ft_nla_e100["test_frac"] = 0.0
_ft_nla_e100["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
_ft_nla_e100["epochs"] = 100          # <-- THE ONLY DIFFERENCE vs the shipped r4 row
_ft_nla_e100["project"] = _BGP_NLE_PROJECT
kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_bgp_z8_r{_E100_REPEAT}_ens9_e100"] = \
    _nle_bake_repeat(_ft_nla_e100, _E100_REPEAT)

# --- M5e: 150-epoch retrain, all 5 repeats. ⭐ VERDICT: MORE EPOCHS DO NOT HELP -----------------
# 150 ep gives optimiser-step parity with M4b (7 500 each): M5c's 79 train cosmologies x 50
# iters/ep x 50 ep = 2 500, i.e. 3x fewer updates than M4b's 240 x 150 x 50.
#
# RESULT (n=5, vs the 50-epoch M5c rows): dim-norm FoM (om,s8) 2.8112 -> 2.9071 (**1.034x**);
# S8 width68 0.0797 -> 0.0776; TARP full 0.0242 -> 0.0248 (slightly WORSE); TARP w0 0.2006 ->
# 0.1924 (flat). With step parity achieved it still reaches only **0.695x** of M4b's cosmology FoM
# (M5c was 0.672x) -- tripling training closed 2.3 points of a 33-point gap.
# ⭐ **The under-training hypothesis is FALSIFIED. Do not spend more epochs on this arm.**
#
# ⚠️ The M4b comparison carries a confound found 2026-08-26: M4b's encoder is the p9 foundation on
# CYCLIC LR, while these variates add an EXP-DECAY encoder finetune on top of it, and the fixed-LR
# A/B measured that decay at 0.19-0.50 nats. Retraining the variate encoder on a good schedule is
# the outstanding test -- not more Stage-B epochs, and not k=8 alone.
#
# ⚠️ Do NOT judge on test_log_prob: k=5 here vs M4b's k=8 are different summary spaces whose NLE
# log-probs differ by a change-of-variables Jacobian.
# NEW ROW NAMES ON PURPOSE -- these do not overwrite the 50-epoch M5c rows, whose checkpoints are
# the baseline the whole comparison depends on.
for _r in _VARIATE_REPEATS:
    _ft_nla_e150 = _nle_finetune(f"glass_nle_pretrain_nla_bgp_z8_k5_r{_r}", ensemble_repeats=9,
                                 whiten_k=5, warmstart_max_gap_nats=22.0,
                                 gower_data=_BGP_GOWER_NLA, gower_eb=None,
                                 cosmo_param_names=_COSMO_8_NLA,
                                 preset_overrides=_A_IA_NLA_BOX)
    _ft_nla_e150["max_trainval_cosmos"] = None
    _ft_nla_e150["train_frac"] = 0.8
    _ft_nla_e150["val_frac"] = 0.2
    _ft_nla_e150["test_frac"] = 0.0
    _ft_nla_e150["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
    _ft_nla_e150["epochs"] = 150       # <-- THE ONLY DIFFERENCE vs the shipped M5c rows
    _ft_nla_e150["project"] = _BGP_NLE_PROJECT
    kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_bgp_z8_r{_r}_ens9_e150"] = \
        _nle_bake_repeat(_ft_nla_e150, _r)

# --- M6c: STAGE-B -- GOWER NLE FINETUNE for the `nla_z` variate, all 5 repeats, 150 epochs -------
# The `nla_z` column's second half. Chain per repeat r:
#   M6a  glass_encoder_finetune_nla_z_bgp_z8      (GLASS NPE encoder, 8-D summary)
#     -> M6b  glass_nle_pretrain_nla_z_bgp_z8_k5_r{r}   (Stage-A GLASS NLE on frozen embeddings, v100)
#     -> M6c  THIS ROW                                   (Stage-B Gower NLE finetune + MCMC eval)
#
# GATE CLEARED 2026-08-26 20:30Z -- all five Stage-A repeats finished (`max_epochs=150 reached`),
# each with top-3 checkpoints and a persisted `datasets/whitener.pt`. Best-checkpoint vals span
# -3.16..-3.55 with best epochs at 118-149, i.e. LATE (no early-overfit signature). Those numbers are
# NOT comparable to the `nla` chain's -- a different encoder means a different summary space and the
# NLE log-prob differs by a change-of-variables Jacobian -- they only establish that the five repeats
# are mutually consistent and none is a straggler.
#
# theta MUST match the paired Stage-A: `_COSMO_9_NLAZ` + `_A_IA_NLA_BOX` (9-param, a_ia ~ U[-6,6]).
# `_nle_finetune`'s docstring makes this a hard requirement -- a mismatch mis-shapes and mis-scales
# theta in BOTH training and eval, silently.
#
# whiten_k=5 MATCHES Stage-A and is VALIDATED ON nla_z's OWN SPECTRUM (not inherited from `nla`):
#   [whiten] explained-variance ratio (top-k): [0.8087, 0.1319, 0.0422, 0.015, 0.0017]  = 0.9995
# so k=5 keeps 99.95% and the dropped PCs 6-8 carry 0.05% -- a SMALLER discarded tail than `nla`'s
# 0.21%. k is load-bearing beyond information content: the Stage-A flow checkpoint shapes differ per
# k, so Stage-B must use the same value or the warm start cannot load.
#
# epochs=150 is the standing default for variate NLE Gower finetunes (user 2026-08-26), inherited
# from the M5e step-budget argument: at 50 epochs this split gets ~2 500 optimiser steps vs M4b's
# 7 500, and 150 restores parity. Unlike M5e there is no 50-epoch predecessor for `nla_z`, but the
# `_e150` SUFFIX IS KEPT DELIBERATELY -- the cross-variate comparison partner is
# `gower_nle_finetune_nla_bgp_z8_r{r}_ens9_e150`, and a bare `_ens9` here would collide in meaning
# with the `nla` bare rows, which are 50 epochs. Same suffix must mean same recipe across variates.
#
# Gower store VERIFIED PRESENT (2026-08-26): 15 920 `*.h5` / 30G, baked 16:48 today -- slightly MORE
# complete than the `nla` S2 bake that M5c/M5e run on (15 911). `max_trainval_cosmos=None` (every
# cosmology not in the locked test set) rather than a hard [100], for the same reason as M5c: a hard
# count dies on any variate that lands a sim or two short, while `None` keeps the ONE invariant that
# matters -- the SAME fixed 100 test cosmologies across every variate.
_BGP_GOWER_NLA_Z = f"{_GPU5}/gower_bgp_nla_z_f16_sc8a1_{_EB}/output_*.h5"   # verified 15 920 files

for _r in _VARIATE_REPEATS:
    _ft_nlaz = _nle_finetune(f"glass_nle_pretrain_nla_z_bgp_z8_k5_r{_r}", ensemble_repeats=9,
                             whiten_k=5, warmstart_max_gap_nats=22.0,
                             gower_data=_BGP_GOWER_NLA_Z, gower_eb=None,
                             cosmo_param_names=_COSMO_9_NLAZ,
                             preset_overrides=_A_IA_NLA_BOX)
    _ft_nlaz["max_trainval_cosmos"] = None
    _ft_nlaz["train_frac"] = 0.8
    _ft_nlaz["val_frac"] = 0.2
    _ft_nlaz["test_frac"] = 0.0     # test = the locked 100; fracs must sum to 1.0
    _ft_nlaz["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
    _ft_nlaz["epochs"] = 150
    _ft_nlaz["project"] = _BGP_NLE_PROJECT
    kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_z_bgp_z8_r{_r}_ens9_e150"] = \
        _nle_bake_repeat(_ft_nlaz, _r)


# ##################################################################################################
# === M12 — FIXED-LR ENCODER RE-RUN of BOTH variate chains (user policy, 2026-08-26) ==============
# ##################################################################################################
# ⭐ **STANDING POLICY (user): when a better setup is MEASURED, re-run the affected production work
# on it — do not leave a known-inferior run standing as the result.**
#
# What was measured: the controlled fixed-vs-exp A/B on the `nla_z` compressor, same repeat indices
# so identical split seeds, schedule the only variable. Fixed LR won 2/2 on BOTH pre-registered
# criteria — beat the baseline AND peaked much later:
#     r1  -8.0265 @ep65  vs exp  -7.5215 @ep56     (+0.50 nats)
#     r2  -7.8100 @ep63  vs exp  -7.6226 @ep41     (+0.19 nats, 22 epochs later)
# The exp runs had turned over by ep41-56 and spent the rest of the budget getting worse.
#
# Both shipped variate chains (`nla` M5c/M5e and `nla_z` M6) were built on EXP-decay compressors, so
# both are re-run here from the encoder down. New names throughout: mixing recipes into the existing
# run dirs would let `find_best_checkpoint` pick the global minimum across two recipes.
#
# ⚠️ WHY THIS ALSO MATTERS FOR THE M4b COMPARISON. `nla`'s deficit vs M4b (dim-norm FoM (om,s8,w0)
# 2.2162 vs 3.0149 = 0.735x) is confounded: M4b's compressor IS the p9 foundation trained on CYCLIC
# LR, while the variates add an EXP-decay finetune on top of it. This re-run removes the schedule
# from that list. It does NOT make the chains identical — M4b has no variate-encoder-finetune step at
# all, and ~99 vs 300 trainval cosmologies, k=5 vs k=8 and the store all still differ. Data volume
# remains the larger untested difference.
#
# ⚠️ FIXED, NOT CYCLIC. Fixed is the schedule that was actually measured on this row family; cyclic
# matches M4b but has never been tested head-to-head against fixed here. Cyclic stays the p15 hybrid
# schedule (different failure mode: a plateau to escape, not an early turnover).
#
# ⭐ THE EXP-BASED CHAINS ARE KEPT AND BECOME THE CONTROL ARM. The `nla_z` Stage-B rows running on
# the exp compressor are deliberately NOT cancelled: paired against these, they measure what the
# encoder schedule is worth END-TO-END (FoM/calibration), which no encoder val can tell us.

# --- M12a: the `nla` fixed-LR compressor (the exp row above is PINNED and stays as the control) ---
kids_legacy_bgp_experiments["glass_encoder_finetune_nla_bgp_z8_fixedlr"] = \
    _assert_final_summary_dim(
        _encoder_finetune_bgp(_BGP_NLA, _COSMO_8_NLA, preset_overrides=_A_IA_NLA_BOX), 8,
        "glass_encoder_finetune_nla_bgp_z8_fixedlr")   # scheduler defaults to "fixed"

# --- M12b: Stage-A NLE pretrain on the fixed-LR compressors -------------------------------------
# Separate experiment names (`_fx_`) rather than reusing the existing Stage-A rows with a different
# `--sources`: the run folder does embed the source name, but Stage-B resolves its warm start
# through `checkpoints/<stage_a_exp>/` and a match string, so two source folders under one Stage-A
# experiment would make that resolution ambiguous. k=5 is unchanged and MUST match Stage-B.
for _r in _VARIATE_REPEATS:
    _pre_nla_fx = _nle_pretrain_bgp(_BGP_NLA, _r, cosmo_param_names=_COSMO_8_NLA,
                                    preset_overrides=_A_IA_NLA_BOX)
    _pre_nla_fx["whiten_embeddings"] = {"k": 5}
    kids_legacy_bgp_experiments[f"glass_nle_pretrain_nla_bgp_z8_k5_fx_r{_r}"] = _pre_nla_fx

    _pre_nlaz_fx = _nle_pretrain_bgp(_BGP_NLA_Z, _r, cosmo_param_names=_COSMO_9_NLAZ,
                                     preset_overrides=_A_IA_NLA_BOX)
    _pre_nlaz_fx["whiten_embeddings"] = {"k": 5}
    kids_legacy_bgp_experiments[f"glass_nle_pretrain_nla_z_bgp_z8_k5_fx_r{_r}"] = _pre_nlaz_fx

# --- M12c: Stage-B Gower NLE finetune on the fixed-LR chains ------------------------------------
# Identical to the M5e / M6c rows in every respect except the parent Stage-A, so the pair isolates
# the compressor schedule. 150 ep, ens9, k=5, same locked 100 test cosmologies.
for _r in _VARIATE_REPEATS:
    _ft_nla_fx = _nle_finetune(f"glass_nle_pretrain_nla_bgp_z8_k5_fx_r{_r}", ensemble_repeats=9,
                               whiten_k=5, warmstart_max_gap_nats=22.0,
                               gower_data=_BGP_GOWER_NLA, gower_eb=None,
                               cosmo_param_names=_COSMO_8_NLA,
                               preset_overrides=_A_IA_NLA_BOX)
    _ft_nlaz_fx = _nle_finetune(f"glass_nle_pretrain_nla_z_bgp_z8_k5_fx_r{_r}", ensemble_repeats=9,
                                whiten_k=5, warmstart_max_gap_nats=22.0,
                                gower_data=_BGP_GOWER_NLA_Z, gower_eb=None,
                                cosmo_param_names=_COSMO_9_NLAZ,
                                preset_overrides=_A_IA_NLA_BOX)
    for _c in (_ft_nla_fx, _ft_nlaz_fx):
        _c["max_trainval_cosmos"] = None
        _c["train_frac"] = 0.8
        _c["val_frac"] = 0.2
        _c["test_frac"] = 0.0
        _c["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
        _c["epochs"] = 150
        _c["project"] = _BGP_NLE_PROJECT
    kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_bgp_z8_fx_r{_r}_ens9_e150"] = \
        _nle_bake_repeat(_ft_nla_fx, _r)
    kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_z_bgp_z8_fx_r{_r}_ens9_e150"] = \
        _nle_bake_repeat(_ft_nlaz_fx, _r)


# ##################################################################################################
# === M13 — the kappa=2 GALAXY-BIAS VARIATE chain (encoder -> Stage-A -> Stage-B) =================
# ##################################################################################################
# The kappa=2 arm widens the per-bin galaxy-bias prior: `b_i ~ N(mean_i, kappa*sigma_i)`, truncated
# at +-3*kappa*sigma_i and clipped to GALAXY_BIAS_CLIP (src/KiDS/simulation_config.py). Preset
# `flamingo_pt_diag_k2`.
#
# ⭐ PARENT = M11b r1. The kappa=2 compressor warm-starts from the p15 cyclic run, which is the only
# 15-param model that ever escaped the -7.9 wall: r1 reached **-8.0957 @ep40**, clearing BOTH of M9's
# cold non-breakthrough seeds (-7.9109 / -7.9523). Its siblings stalled at -7.81..-7.86, so the
# per-repeat warm start below inherits that spread honestly -- select downstream on Gower-val, do not
# average the weak seeds in. r4 has no parent (its M11b repeat never left the queue), hence 0..3.
#
# ⚠️⚠️ THE kappa=2 PRIOR BOXES ARE **CLIPPED**, AND kappa=1's ARE NOT. This is the subtle part.
# `_BG_BOXES` is a clean mean +- 3 sigma because at kappa=1 nothing reaches GALAXY_BIAS_CLIP (the
# config records 0.00 % of draws on the clip). At kappa=2 the sigmas double, +-3*2*sigma overruns the
# clip on the LOW side of bins 1-2 (bin1 would reach -0.06, below the 0.3 floor), and the config
# records **2.18 % of draws on the clip**. So the realised support is the CLIPPED interval, and the
# scaler box must match it -- a mean +- 6 sigma box would tell the scaler the data spans a range it
# never occupies and mis-scale theta. Derived here from the same constants as `_BG_BOXES` rather than
# transcribed, so both follow the source of truth together.
_BG_CLIP = (0.3, 2.2)          # GALAXY_BIAS_CLIP, src/KiDS/simulation_config.py
_K2 = 2.0
_BG_BOXES_K2 = {
    f"b_g_bin{_i + 1}": (max(_m - 3.0 * _K2 * _s, _BG_CLIP[0]),
                         min(_m + 3.0 * _K2 * _s, _BG_CLIP[1]))
    for _i, (_m, _s) in enumerate(zip(_BG_PRIOR_MEANS, _BG_PRIOR_SIGMAS))
}

_BGPK2_GLASS = f"{_GPU5}/glass_bgpk2_nla_m_f16_sc8a1_{_EB}/output_*.h5"
_BGPK2_GOWER = f"{_GPU5}/gower_bgpk2_nla_m_f16_sc8a1_{_EB}/output_*.h5"
_P15_CYC_CKPT = f"{_CKPT}/kids_legacy_hybrid_nla_m_bgp_z16_resnet_sc8a1_p15_warm_cyc/"
_K2_REPEATS = (0, 1, 2, 3)     # r4 has no M11b parent

# --- M13a: the kappa=2 compressor (p15 cyclic -> kappa=2 GLASS store) ---------------------------
# ⭐ LAUNCH-VERIFY **`Loaded keys: 129`** here, NOT 125. M11b's warm start was 8-D -> 16-D and so
# skipped the 4 resized final-layer tensors (125/129). This one is 16-D -> 16-D, same architecture
# and same widths, so EVERY key must load. A 125 here would mean the parent is the p9 foundation
# rather than the p15 cyclic run, i.e. the wrong lineage. Summary dim stays 16.
def _hybrid_bgpk2_z16(data_patterns, repeat_indices=_K2_REPEATS):
    """kappa=2 variate compressor: the p15 cyclic recipe, re-pointed at the kappa=2 store."""
    c = _hybrid_bgp_p15_z16_warm_cyc(data_patterns, repeat_indices=repeat_indices)
    c["pretrained_embedding_ckpt_path"] = _P15_CYC_CKPT      # parent = M11b, not the p9 foundation
    c["scaler_options"] = {
        "data": {"type": "standard", "keys": None},
        "cosmo": {"type": "preset", "preset_overrides": dict(_BG_BOXES_K2)},   # CLIPPED boxes
    }
    return c


kids_legacy_bgp_experiments["kids_legacy_hybrid_nla_m_bgpk2_z16_warm_cyc"] = \
    _assert_final_summary_dim(_hybrid_bgpk2_z16(_BGPK2_GLASS), 16,
                              "kids_legacy_hybrid_nla_m_bgpk2_z16_warm_cyc")

# --- M13b/M13c: Stage-A (GLASS NLE on frozen embeddings) + Stage-B (Gower NLE + MCMC eval) ------
# ⚠️ whiten_k = 16 = PURE-WHITEN on the 16-D summary, the exact analogue of k=8 on an 8-D one: a
# full-rank invertible affine map that improves conditioning and discards nothing. Deliberately NOT
# truncated -- the KSWEEP result is that there is no free truncation. (The nla/nla_z chains use k=5
# on 8-D, which IS a truncation; that choice was validated on their own eigenspectra and does not
# transfer here.) ⚠️ k=16 is UNVALIDATED for this arm: check the Stage-A log's
# `[whiten] explained-variance ratio` before trusting it, exactly as k=5 was checked for nla_z.
#
# ⚠️ `warmstart_max_gap_nats` is left at 22.0, the z8 value, DELIBERATELY. A 16-D pure-whiten may gap
# wider, but the guard exists to catch a broken warm start and raising it pre-emptively would defeat
# it. If guard-c fires here, DIAGNOSE -- do not simply widen the threshold.
# ⚠️⚠️ theta HERE IS THE 15-PARAM VECTOR, and it must be passed EXPLICITLY. Leaving
# `cosmo_param_names=None` falls through to the 9-param default, which silently builds a Stage-A/B
# that disagrees with its own 15-param compressor -- caught exactly that on the first write of this
# block. `_nle_finetune`'s docstring makes theta-matching a hard requirement: a mismatch mis-shapes
# and mis-scales theta in BOTH training and eval. Taken from the compressor row itself rather than
# re-listed, so the chain cannot drift from its own parent.
_K2_THETA = list(
    kids_legacy_bgp_experiments["kids_legacy_hybrid_nla_m_bgpk2_z16_warm_cyc"]["cosmo_param_names"])
assert len(_K2_THETA) == 15, f"expected the 15-param kappa=2 vector, got {len(_K2_THETA)}"

for _r in _K2_REPEATS:
    _pre_k2 = _nle_pretrain_bgp(_BGPK2_GLASS, _r, cosmo_param_names=list(_K2_THETA),
                                preset_overrides=dict(_BG_BOXES_K2))
    _pre_k2["whiten_embeddings"] = {"k": 16}
    kids_legacy_bgp_experiments[f"glass_nle_pretrain_nla_m_bgpk2_z16_r{_r}"] = _pre_k2

    _ft_k2 = _nle_finetune(f"glass_nle_pretrain_nla_m_bgpk2_z16_r{_r}", ensemble_repeats=9,
                           whiten_k=16, warmstart_max_gap_nats=22.0,
                           gower_data=_BGPK2_GOWER, gower_eb=None,
                           cosmo_param_names=list(_K2_THETA),
                           preset_overrides=dict(_BG_BOXES_K2))
    _ft_k2["max_trainval_cosmos"] = None
    _ft_k2["train_frac"] = 0.8
    _ft_k2["val_frac"] = 0.2
    _ft_k2["test_frac"] = 0.0
    _ft_k2["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
    _ft_k2["epochs"] = 150
    _ft_k2["project"] = _BGP_NLE_PROJECT
    kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_m_bgpk2_z16_r{_r}_ens9_e150"] = \
        _nle_bake_repeat(_ft_k2, _r)

# =============================================================================================
# kappa=2, k=5 whiten -- THE PRODUCTION kappa=2 CHAIN (supersedes the k=16 rows above).
# ---------------------------------------------------------------------------------------------
# The k=16 rows above are a full-rank "pure whiten" and were MEASURED RANK-DEFICIENT on this 16-D
# head (2026-08-27, Stage-A job 1349506):
#     EVR = [0.5361, 0.3476, 0.1002, 0.011, 0.0025, 0.0012, 0.0007, 0.0004,
#            0.0002, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# 3 components carry 98.4 %, 4 carry 99.49 %, 5 carry 99.74 % -- and SIX eigenvalues are EXACTLY
# 0.0. The 16-D head is effectively ~4-D, the same shape every z8 head shows.
# WHY THAT BREAKS: WhitenPCAScaler.fit does `scale = evals[:k].clamp(min=1e-12).sqrt()` and
# `transform` DIVIDES by scale, so each null direction gets scale=1e-6 and is AMPLIFIED ~1e6. The
# NLE would silently receive ~4 real dimensions plus 6 dimensions of blown-up float noise. Nothing
# raises -- only the EVR exposes it, which is why the Stage-A EVR check is mandatory for a new k.
# k=5 matches the recipe validated on every z8 chain (EVR ~0.9995, guard-c gap 17.140 vs 22.0).
# A DISTINCT experiment name is REQUIRED: `fit_and_persist_whitener` refuses to re-fit an existing
# whitener at a different k (fit-once, research Finding C3). NEVER delete whitener.pt to force one.
# guard-c stays at 22.0: if it fires it is reporting a real problem, not a threshold that is too tight.
# =============================================================================================
_K2_WHITEN_K5 = 5

for _r in _K2_REPEATS:
    _pre_k2b = _nle_pretrain_bgp(_BGPK2_GLASS, _r, cosmo_param_names=list(_K2_THETA),
                                 preset_overrides=dict(_BG_BOXES_K2))
    _pre_k2b["whiten_embeddings"] = {"k": _K2_WHITEN_K5}
    kids_legacy_bgp_experiments[f"glass_nle_pretrain_nla_m_bgpk2_z16_k5_r{_r}"] = _pre_k2b

    _ft_k2b = _nle_finetune(f"glass_nle_pretrain_nla_m_bgpk2_z16_k5_r{_r}", ensemble_repeats=9,
                            whiten_k=_K2_WHITEN_K5, warmstart_max_gap_nats=22.0,
                            gower_data=_BGPK2_GOWER, gower_eb=None,
                            cosmo_param_names=list(_K2_THETA),
                            preset_overrides=dict(_BG_BOXES_K2))
    _ft_k2b["max_trainval_cosmos"] = None
    _ft_k2b["train_frac"] = 0.8
    _ft_k2b["val_frac"] = 0.2
    _ft_k2b["test_frac"] = 0.0
    _ft_k2b["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
    _ft_k2b["epochs"] = 150
    _ft_k2b["project"] = _BGP_NLE_PROJECT
    kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_m_bgpk2_z16_k5_r{_r}_ens9_e150"] = \
        _nle_bake_repeat(_ft_k2b, _r)



# ##################################################################################################
# M14 — the `vd` (VARIABLE-DEPTH) variate chain: GLASS encoder -> Stage-A NLE -> Stage-B Gower NLE
# --------------------------------------------------------------------------------------------------
# The fourth production variate, alongside `nla` (M5) and `nla_z` (M6). Same three-stage shape and
# the same factories; only the store and the theta box differ.
#
# ⭐ **theta is the PLAIN 9-param NLA-M vector and needs NO `preset_overrides`.** This is the one
# real difference from `nla`/`nla_z`: those two re-box `a_ia` to U[-6,6] because they *replace* the
# IA model, so the global preset's NLA-M a_ia box (4.48, 7.0) would mis-scale them. `vd` does NOT
# touch the IA model at all — it varies the SURVEY DEPTH — so its a_ia really is drawn from the
# NLA-M box and the global preset is already correct. Adding an override here would be the bug.
# Confirmed against the store: `data-h5` reports exactly {omega_m, sigma_8, w0, mnu, h, ns, ombh2,
# a_ia, b_ia} + b_g_bin1..6 + galaxy_bias_eff — i.e. the NLA-M vector, with NO vd-specific parameter.
# Variable depth is a property of the FORWARD MODEL, not an inferred parameter.
#
# ⚠️ `eb_map_variant=None` (bare `E`/`B` groups), NOT the `fwhm4_lmin56_lcut1400` tag. The store name
# carries that suffix, but the suffix records WHICH variant was extracted, not whether it was written
# tagged. Every `glass_bgp_*_f16_sc8a1_*` store is baked WITHOUT `--keep-variant-tag` (cf. lines
# 66-68) and its sibling row `kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1` resolves to
# `eb_map_variant=None` — verified by loading the built config. **A wrong guess here does not raise:
# the loader SILENTLY SKIPS every file and trains on nothing.** Confirm on the first cluster run from
# the `ok=` / `Loaded keys` count in the log before trusting any number that comes out of it.
#
# Summary stays 8-D per the SUMMARY-WIDTH RULE (16-D is reserved for rows that actually INFER b_g;
# here b_g is only marginalised), and `_assert_final_summary_dim` pins that.
#
# Fixed LR: this is a NEW row, so it takes the post-2026-08-26 default (`scheduler="fixed"`). The
# `exp` decay pinned on the M5a/M6a rows is there only because their checkpoints were trained under it.
_BGP_NLA_M_VD = f"{_GPU5}/glass_bgp_nla_m_vd_f16_sc8a1_{_EB}/output_*.h5"
# Store VERIFIED PRESENT 2026-08-28: 11 880 `*.h5` / 22G on gpu5, baked 10:06-10:07, count stable
# across two listings 15 min apart (i.e. the bake had finished, not still running).

# theta: the plain 9-param NLA-M vector, read off the built foundation row rather than re-listed, so
# this cannot drift from `kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1`.
_COSMO_9_NLAM = ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"]

# === M14a — the `vd` GLASS compressor finetune ===================================================
kids_legacy_bgp_experiments["glass_encoder_finetune_nla_m_vd_bgp_z8"] = \
    _assert_final_summary_dim(
        _encoder_finetune_bgp(_BGP_NLA_M_VD, _COSMO_9_NLAM), 8,
        "glass_encoder_finetune_nla_m_vd_bgp_z8")

# === M14b — Stage-A: GLASS NLE on the frozen `vd` embeddings, k=5 whiten ========================
# GATED ON M14a (its source encoder). Submit per repeat with:
#   embed --target glass_nle_pretrain_nla_m_vd_bgp_z8_k5_r<r> \
#         --sources glass_encoder_finetune_nla_m_vd_bgp_z8 --gpu v100 --skip-smoke
# k=5 matches every other z8 chain (EVR ~0.9995 there). ⚠️ NEW NAME PER k, deliberately: `whiten_k`
# sets the flow's context width, so a k=5 run writing into a k=8 row's dir would let
# `get_best_checkpoint` resolve a SHAPE-MISMATCHED checkpoint. The paired Stage-B rows MUST use the
# same k — they reuse this run's persisted whitener and load its flow.
for _r in _VARIATE_REPEATS:
    _pre_vd = _nle_pretrain_bgp(_BGP_NLA_M_VD, _r, cosmo_param_names=_COSMO_9_NLAM)
    _pre_vd["whiten_embeddings"] = {"k": 5}
    kids_legacy_bgp_experiments[f"glass_nle_pretrain_nla_m_vd_bgp_z8_k5_r{_r}"] = _pre_vd

# === M14c — Stage-B: Gower NLE ens9 finetune + bundled MCMC eval ================================
# ⛔ **BLOCKED ON DATA, AND `_BGP_GOWER_NLA_M_VD` IS AN ASSUMED NAME.** As of 2026-08-28 the Gower VD
# raw sim (`sim_gower_mocks_nla_m_vd_bgp`, job 1349289) is STILL RUNNING and NO baked Gower VD store
# exists — `data-ls` shows only the unrelated `gower_mocks_nla_m_novd_counts_f16_...` (note `novd`).
# The name below follows the settled convention `gower_bgp_<variate>_f16_sc8a1_<tag>` (cf.
# `_BGP_GOWER_NLA_Z`). **Verify with `data-ls` before submitting.** Once the sim finishes, bake it:
#   prebake --src-datasets-root gpu4 --src-dir gower_mocks_nla_m_vd_bgp \
#           --out-dir gower_bgp_nla_m_vd_f16_sc8a1_fwhm4_lmin56_lcut1400 \
#           --eb-variant sc8a1_fwhm4_lmin56_lcut1400 --dtype float16
# (i.e. the sc8a1 recipe; omit `--keep-variant-tag` to keep bare `E` groups, matching gower_eb=None.)
# A sim leaving squeue is NOT evidence the store exists — only `data-ls` is.
#
# `max_trainval_cosmos=None` (every cosmology outside the locked test set) rather than a hard count,
# for the same reason as M5c/M6c: a hard count dies on a variate that lands a sim or two short, while
# `None` preserves the ONE invariant that matters — the SAME fixed 100 test cosmologies everywhere.
_BGP_GOWER_NLA_M_VD = f"{_GPU5}/gower_bgp_nla_m_vd_f16_sc8a1_{_EB}/output_*.h5"   # ASSUMED — verify

for _r in _VARIATE_REPEATS:
    _ft_vd = _nle_finetune(f"glass_nle_pretrain_nla_m_vd_bgp_z8_k5_r{_r}", ensemble_repeats=9,
                           whiten_k=5, warmstart_max_gap_nats=22.0,
                           gower_data=_BGP_GOWER_NLA_M_VD, gower_eb=None,
                           cosmo_param_names=_COSMO_9_NLAM)
    _ft_vd["max_trainval_cosmos"] = None
    _ft_vd["train_frac"] = 0.8
    _ft_vd["val_frac"] = 0.2
    _ft_vd["test_frac"] = 0.0        # test = the locked 100; fracs must sum to 1.0
    _ft_vd["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
    _ft_vd["epochs"] = 150
    _ft_vd["project"] = _BGP_NLE_PROJECT
    kids_legacy_bgp_experiments[f"gower_nle_finetune_nla_m_vd_bgp_z8_r{_r}_ens9_e150"] = \
        _nle_bake_repeat(_ft_vd, _r)


# ##################################################################################################
# === M15 — THE 2-PARAMETER COMPRESSION CEILING SUITE (user directive 2026-09-01) =================
# ##################################################################################################
# ⭐ PRIORITY: this suite takes precedence over ALL other training, because it decides whether the
# variate rows need re-running at all.
#
# THE QUESTION. On the `nla` variate, S8 at FIXED nuisances is already within ~11 % of the flagship
# (artifacts/variate_diagnostics/ASSESSMENT.md §7), but omega_m is still ~1.36x wider and that
# residual is NOT IA-degeneracy and NOT explained by any handicap kappa=2 also carries (same 11.8k
# GLASS store, same k=5, same ~99 Gower cosmologies, same epochs — kappa=2 lands at 1.02-1.05x).
# So: is the variate compressor actually AT the constraining power of the data, or is it
# under-trained / stuck in a poor compression minimum?
#
# THE DESIGN. Collapse to a 2-PARAMETER inference problem (omega_m, sigma_8; everything else
# varies in the mocks and is marginalised implicitly, exactly as b_g is on the flagship). A 2-D
# posterior is the cleanest possible read of compression quality: no nuisance volume, no IA prior
# asymmetry, nothing to condition out. Four arms, 3 repeats each, ALL on the same `nla` GLASS store
# with the same architecture/epochs/lr as the production variate chain — only theta and the
# pretrained loads differ.
#
#   M15a  band-only, warm-started from the NLA-M band  -> the 2-pt CEILING
#   M15b  NEW band (M15a, FROZEN) + warm-started map   -> does the map add anything on top?
#   M15c  OLD band (NLA-M, FROZEN) + warm-started map  -> does a variate-matched band matter?
#   M15d  NEW band (M15a, FROZEN) + SCRATCH map        -> is the map warm start load-bearing?
#
# Read: M15b ~ M15a  => the maps add nothing after the finetune (compression failure).
#       M15b >> M15a => the maps work and the variate width is closer to honest.
#       M15b vs M15c => whether the band must be re-fit per variate.
#       M15b vs M15d => whether the map warm start is what carries the transfer.
#
# ⚠️⚠️ **LOAD ORDER IS THE WHOLE EXPERIMENT.** `_load_pretrained_embedding_net` writes the ENTIRE
# hybrid embedding_net (band + map + fusion) and `build_model` runs it LAST. Setting both
# `pretrained_band_ckpt_path` and `pretrained_embedding_ckpt_path` naively therefore lets the
# NLA-M foundation SILENTLY OVERWRITE the band we just trained — M15b/M15d would secretly become
# M15c. `band_load_after_embedding=True` (added to src/ml/utils.py:build_model for this suite,
# default False everywhere else) defers the band load until after the embedding load so the
# intended band survives. ACCEPTANCE CHECK ON THE FIRST M15b JOB LOG: the embedding-load line must
# print BEFORE the band-load line, both with sane matched-key counts.
#
# ⚠️ ONE match string resolves BOTH parents: `get_best_checkpoint(..., pretrained_band_match_string)`
# is used for the band AND the embedding folder, so both parents must expose repeat r under the
# same `match_num_cosmo=False` -> "_{r}" convention. Verified for r in 0,1,2 before launch.
_M15_REPEATS = [0, 1, 2]
_COSMO_2 = ["omega_m", "sigma_8"]
_M15_BAND_CKPT = f"{_CKPT}/m15a_band_nla_p2/"          # written by M15a below


def _m15_common(c):
    """Shared: 2-param theta, 3 repeats, per-repeat source resolution, own W&B project."""
    c["cosmo_param_names"] = list(_COSMO_2)
    c.pop("repeats", None)
    c["repeat_indices"] = list(_M15_REPEATS)
    c["match_num_cosmo"] = False       # resolve every parent checkpoint per-repeat as "_{i}"
    c["project"] = "bgp-p2-ceiling"
    return c


# --- M15a: the 2-pt ceiling. Band-only MLP on the nla store, warm-started from the NLA-M band. ----
# For a band-only model the embedding_net IS the band, so the whole-embedding loader is the clean
# mechanism here (no `_find_band_module` subtlety) and there is no map branch to overwrite.
_m15a = _m15_common(_band_bgp(_BGP_NLA))
_m15a["pretrained_embedding_ckpt_path"] = _BAND_CKPT_BGP
_m15a["freeze_embedding_net"] = False               # warm start, then train it
kids_legacy_bgp_experiments["m15a_band_nla_p2"] = _m15a


def _m15_hybrid(band_ckpt, warm_map, key):
    """A hybrid arm: FROZEN `band_ckpt`, map branch warm-started from the 9-param foundation or not.

    `band_ckpt=None` is not offered — every hybrid arm here freezes a band, and which band is the
    variable under test.
    """
    c = _hybrid_bgp(_BGP_NLA, None, band_ckpt, repeat_indices=_M15_REPEATS)
    c = _m15_common(c)
    c["pretrained_band_ckpt_path"] = band_ckpt
    c["freeze_band"] = True
    if warm_map:
        # warm-start the map branch (and fusion head) from the 9-param NLA-M foundation, THEN lay
        # the frozen band on top -- see the LOAD ORDER warning above.
        c["pretrained_embedding_ckpt_path"] = _SC8A1_9P_CKPT
        c["freeze_embedding_net"] = False
        c["band_load_after_embedding"] = True
    else:
        c.pop("pretrained_embedding_ckpt_path", None)
        c["freeze_embedding_net"] = False
        # no embedding load => nothing can overwrite the band, but keep the flag on so the arms
        # differ ONLY in the map warm start and the printed load order is identical.
        c["band_load_after_embedding"] = True
    return _assert_final_summary_dim(c, 8, key)


# --- M15b: NEW band (frozen) + warm-started map -- the headline arm -------------------------------
kids_legacy_bgp_experiments["m15b_hybrid_nla_p2_newband_warmmap"] = \
    _m15_hybrid(_M15_BAND_CKPT, warm_map=True, key="m15b_hybrid_nla_p2_newband_warmmap")

# --- M15c: OLD (NLA-M) band (frozen) + warm-started map ------------------------------------------
kids_legacy_bgp_experiments["m15c_hybrid_nla_p2_oldband_warmmap"] = \
    _m15_hybrid(_BAND_CKPT_BGP, warm_map=True, key="m15c_hybrid_nla_p2_oldband_warmmap")

# --- M15d: NEW band (frozen) + SCRATCH map (no warm start) ---------------------------------------
# "Without the warm start" = the MAP warm start. The frozen new band stays, otherwise this is not
# the same-as-M15b control it is meant to be.
kids_legacy_bgp_experiments["m15d_hybrid_nla_p2_newband_scratchmap"] = \
    _m15_hybrid(_M15_BAND_CKPT, warm_map=False, key="m15d_hybrid_nla_p2_newband_scratchmap")
