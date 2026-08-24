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
    _COSMO_8_NLA, _A_IA_NLA_BOX,
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
                          repeat_indices=_VARIATE_REPEATS):
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
    c["scheduler_type"] = "exp"
    c["scheduler_kwargs"] = {"gamma": 0.984, "warmup_steps": 0}   # 0.984^100 ~ 0.20: 2e-4 -> ~4e-5
    c["lr"] = 0.0002
    c["epochs"] = 100
    if preset_overrides:
        c["scaler_options"] = {
            "data": {"type": "standard", "keys": None},
            "cosmo": {"type": "preset", "preset_overrides": dict(preset_overrides)},
        }
    return c


kids_legacy_bgp_experiments["glass_encoder_finetune_nla_bgp_z8"] = \
    _assert_final_summary_dim(
        _encoder_finetune_bgp(_BGP_NLA, _COSMO_8_NLA, preset_overrides=_A_IA_NLA_BOX), 8,
        "glass_encoder_finetune_nla_bgp_z8")


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


# === STACKED-ENSEMBLE ABLATION (user 2026-08-23) — is there information in the 5 compressors ======
# The 5 foundation repeats are 5 independently-seeded compressors of the SAME data. Each emits an
# 8-D summary; concatenating all five gives a **40-D** stacked summary. Two questions:
#   (1) how many PCA components does the 40-D stack actually need — i.e. are the 5 compressors
#       redundant (effective rank ~8) or do they see complementary things (rank >> 8)?
#   (2) does an NLE trained on the stack constrain better than one on a single 8-D summary?
# Intended as the final exploratory test before considering an ensemble-stacked production posterior.
#
# ⭐ `compute_embeddings` ALREADY concatenates feature-wise across sources
# (`torch.cat(zs_batch, dim=-1)`), so the 40-D vector needs no new machinery. The ONE obstacle was
# that `load_pretrained_models` applied a single match string to every source, so five REPEATS of one
# experiment could not be addressed separately — they share a checkpoint dir and differ only in the
# run subdir (`pretrain_ncosmoNone_0` … `_4`). Fixed additively by `per_source_match_strings`
# (defaults to None ⇒ previous behaviour byte-for-byte), driven from the config key below.
#
# ⚠️ The cached per-repeat `emb_*.pt` files CANNOT be concatenated instead: `split_seed = 42 + repeat`
# (`utils.py:763`), so each repeat's Stage-A cache is a DIFFERENT train/val/test partition and the
# rows are not aligned. The 40-D stack has to be computed fresh over one common split.
#
# ⚠️ SHORT ALIASES ARE DELIBERATE. `source_run_name = f"{run_name}_{'_'.join(sources)}"` becomes a
# single directory component; five copies of the 44-char foundation name would make it ~246 chars,
# a hair under Linux's 255-byte NAME_MAX. Aliasing to `bgpz8enc{r}` keeps it ~60. The alias carries
# `experiment_name` = the REAL foundation name, so `load_best_model_and_build_posterior` still finds
# `checkpoints/kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1/` — only the label is short.
_FOUNDATION_EXP = "kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1"
for _r in _NLE_REPEATS:
    kids_legacy_bgp_experiments[f"bgpz8enc{_r}"] = {
        **kids_legacy_bgp_experiments[_FOUNDATION_EXP],
        "experiment_name": _FOUNDATION_EXP,
    }

_STACK_SOURCES = ",".join(f"bgpz8enc{_r}" for _r in _NLE_REPEATS)   # pass to `embed --sources`
_STACK_MATCHES = [f"None_{_r}" for _r in _NLE_REPEATS]             # binds source i -> repeat i
_STACK_DIM = 8 * len(_NLE_REPEATS)                                  # 40


# --- S1: Stage-A NLE pretrain on the 40-D STACK (GLASS) -----------------------------------------
# whiten_k = 40 = PURE-WHITEN on a 40-D summary, the exact analogue of k=8 on the 8-D one: a
# full-rank invertible affine map that buys conditioning and throws nothing away. Deliberately NOT
# truncated — the KSWEEP result is that there is no free truncation, and truncating here would also
# pre-judge question (1), which the PCA analysis is meant to answer empirically.
_stack_pre = _nle_pretrain_bgp(_BGP_SC8A1, 0)
_stack_pre["whiten_embeddings"] = {"k": _STACK_DIM}
_stack_pre["source_match_strings"] = list(_STACK_MATCHES)
_stack_pre["embedding_cache_name"] = "bgp_stack5_glass"   # short, explicit cache dir
kids_legacy_bgp_experiments["glass_nle_pretrain_nla_m_bgp_stack5"] = _stack_pre


# --- S1b: the PCA PROBE — answer question (1) in ~40 min instead of ~12 h -----------------------
# The user wants the PCA to DECIDE the truncation dimension, so it has to land before the Stage-B
# run commits to a k. A v100 would embed the full store in ~30 min, but BOTH v100 nodes are
# IDLE+DRAIN (10 free GPUs, 376 G, not schedulable) and no other GPU can start, so the full-store
# Stage-A is on CPU and ~12 h from its cache.
#
# This probe gets the same answer far sooner, because a 40-D covariance does not need 100 600 rows:
#   * `max_trainval_cosmos=2000` (of ~25 150) ⇒ ~8 000 train/val mocks — 200 samples per dimension,
#     ample for a well-determined 40-D PCA;
#   * `N_test_cosmologies=100` keeps the test slice from dominating the pass (test_frac 0.1 of the
#     FULL suite would otherwise be ~2 515 cosmologies, i.e. bigger than the trainval subset);
#   * ⭐ `run_training=False` ⇒ `do_run_training` False, so `fit_nde_on_embeddings` is SKIPPED
#     entirely: the job computes the embeddings, writes the cache, and exits.
#
# ⭐ The cached `emb_*.pt` holds the **RAW** stack, not the whitened one — `_save_embedding_cache`
# runs BEFORE the whitening block, which is commented "both the cache-hit and fresh-compute paths
# converge here with raw train_z/val_z/test_z". So the PCA is genuine, not circular.
#
# Its own `embedding_cache_name` keeps it from colliding with the full-store run's cache.
_stack_probe = _nle_pretrain_bgp(_BGP_SC8A1, 0)
_stack_probe["whiten_embeddings"] = {"k": _STACK_DIM}
_stack_probe["source_match_strings"] = list(_STACK_MATCHES)
_stack_probe["embedding_cache_name"] = "bgp_stack5_pcaprobe"
_stack_probe["max_trainval_cosmos"] = [2000]
_stack_probe["N_test_cosmologies"] = 100
_stack_probe["run_training"] = False
kids_legacy_bgp_experiments["glass_nle_pretrain_nla_m_bgp_stack5_pcaprobe"] = _stack_probe


# --- S2: Stage-B fine-tune + MCMC eval on Gower, ens9 -------------------------------------------
_stack_ft = _nle_finetune("glass_nle_pretrain_nla_m_bgp_stack5", ensemble_repeats=9,
                          whiten_k=_STACK_DIM, warmstart_max_gap_nats=22.0,
                          gower_data=_BGP_GOWER_NLA_M, gower_eb=None)
_stack_ft["max_trainval_cosmos"] = [300]
_stack_ft["train_frac"] = 0.8
_stack_ft["val_frac"] = 0.2
_stack_ft["test_frac"] = 0.0
_stack_ft["fixed_test_sim_ids"] = _GOWER_TEST_IDS
_stack_ft["project"] = _BGP_NLE_PROJECT
_stack_ft["source_match_strings"] = list(_STACK_MATCHES)
kids_legacy_bgp_experiments["gower_nle_finetune_nla_m_bgp_stack5_ens9"] = _nle_bake_repeat(_stack_ft, 0)


# === STACK5 @ k=16 — NPE + NLE heads, GLASS then Gower (user 2026-08-23) ========================
# ⭐ WHY AN NPE HEAD IS THE RIGHT INSTRUMENT FOR "how much extra information".
# NLE val log-probs are densities over the WHITENED EMBEDDING, so they live in whatever coordinate
# system the encoder+whitener define — a 16-D stack and an 8-D single summary are simply not
# comparable (standing rule 2, and the reason the adapted-encoder arm's `test_log_prob` jump was
# meaningless). An **NPE** head models p(theta | z): a density over THETA, the same space for every
# encoder. So the NPE test log-prob IS directly comparable to the single-encoder foundation's
# (-5.2681 … -5.4014) and is a genuine measure of extracted information. That is what makes the
# user's "NPE head first, then NLE" ordering the informative one.
#
# k=16 (user's choice) sits between the PCA's 99 % mark (k=7) and 99.9 % (k=19), and is 2x the
# single-encoder width. The measured spectrum spans 1.95e1 -> 8.0e-5 (ratio 2.4e5), so k=40
# pure-whiten would amplify the worst direction ~500x; k=16 keeps ~99.8 % of the variance while
# cutting that conditioning problem by more than an order of magnitude.
#
# ⚡ The two GLASS rows REUSE the full-store embedding cache that `..._stack5` (job 1348636) is
# writing, so once it lands they train in MINUTES instead of repeating a ~10 h embedding pass.
# Reusing the raw cache across different k is exactly right: the cache stores RAW z, and whitening
# is applied per-run afterwards.
_STACK_K16 = 16


def _stack_head(inference_mode):
    """GLASS Stage-A head on the 40-D stack, whitened to k=16. `npe` => p(theta|z), `nle` => p(z|theta)."""
    c = _nle_pretrain_bgp(_BGP_SC8A1, 0)
    c["inference_mode"] = inference_mode
    c["whiten_embeddings"] = {"k": _STACK_K16}
    c["source_match_strings"] = list(_STACK_MATCHES)
    c["embedding_cache_name"] = "bgp_stack5_glass"     # share the full-store cache
    c["reuse_embedding_cache"] = True                  # ...and skip re-embedding
    return c


kids_legacy_bgp_experiments["glass_npe_pretrain_nla_m_bgp_stack5_k16"] = _stack_head("npe")
kids_legacy_bgp_experiments["glass_nle_pretrain_nla_m_bgp_stack5_k16"] = _stack_head("nle")


# --- The UNTRUNCATED NPE head: k=40 = pure-whiten -----------------------------------------------
# The k=40 arm was NLE-only, because it predates the NPE idea: k=40 was written to answer "does the
# stack help at all" and NPE only entered with the k=16 request. But NPE at k=40 is strictly the
# better instrument for "how much extra information does stacking buy", and it is nearly free (it
# reuses the same cache), so it is worth having:
#   * NPE models p(theta | z) -- a density over THETA, the same space for every encoder -- so it is
#     comparable to the single-encoder foundation's -5.2681 .. -5.4014. NLE is not (rule 5).
#   * k=40 is a FULL-RANK invertible affine map, so it discards NOTHING. NPE@k=40 therefore measures
#     the stack's TOTAL information, while NPE@k=16 (99.8 % of variance) is a LOWER BOUND on it.
#   * The pair also prices the truncation directly, which KSWEEP says must not be assumed free:
#     8 -> 6 cost 13 % FoM. (k40 - k16) is that cost measured on the stack.
# CAVEAT: the 40-D spectrum spans 1.94e1 -> 9.6e-5 (ratio ~2e5), so pure-whitening amplifies the
# worst direction ~450x. NPE only CONDITIONS on z rather than modelling its density, so it tolerates
# that far better than the NLE head would -- but if this row trains unstably, that ill-conditioning
# is the first suspect and k=16 is the answer, not a bug.
_stack_npe_k40 = _stack_head("npe")
_stack_npe_k40["whiten_embeddings"] = {"k": _STACK_DIM}
kids_legacy_bgp_experiments["glass_npe_pretrain_nla_m_bgp_stack5_k40"] = _stack_npe_k40


def _stack_gower(inference_mode, pretrain_exp):
    """Gower Stage-B finetune of a stacked head, ens9 + eval. Same split/store as the M4b baseline,
    so the resulting FoM is directly comparable to the 5-repeat production numbers."""
    c = _nle_finetune(pretrain_exp, ensemble_repeats=9, whiten_k=_STACK_K16,
                      warmstart_max_gap_nats=22.0,
                      gower_data=_BGP_GOWER_NLA_M, gower_eb=None)
    c["inference_mode"] = inference_mode
    c["max_trainval_cosmos"] = [300]
    c["train_frac"] = 0.8
    c["val_frac"] = 0.2
    c["test_frac"] = 0.0
    c["fixed_test_sim_ids"] = _GOWER_TEST_IDS
    c["project"] = _BGP_NLE_PROJECT
    c["source_match_strings"] = list(_STACK_MATCHES)
    return _nle_bake_repeat(c, 0)


kids_legacy_bgp_experiments["gower_npe_finetune_nla_m_bgp_stack5_k16_ens9"] = \
    _stack_gower("npe", "glass_npe_pretrain_nla_m_bgp_stack5_k16")
kids_legacy_bgp_experiments["gower_nle_finetune_nla_m_bgp_stack5_k16_ens9"] = \
    _stack_gower("nle", "glass_nle_pretrain_nla_m_bgp_stack5_k16")


# --- M5c (the `nla` variate Gower NLE finetune) is NOT written -----------------------------------
# It would need a Gower `nla` store (S2), and the dataset side's scope change of 2026-08-18 makes S1
# the only remaining sim. Writing a row against a store nobody plans to generate would be dead
# config that reads as ready. Add it if S2 is ever launched.
