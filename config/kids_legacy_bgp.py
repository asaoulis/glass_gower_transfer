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
