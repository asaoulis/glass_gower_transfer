"""DUAL-NORMALISATION arm comparison — model configs (task training-runs/improved-shear-tests).

**What this module is for.** The dual-normalisation superset store (`*_novd_dualnorm`, simulator
commit `66231fc`) carries, per mock, BOTH map products (`E_<tag>` and `E_sc8_<tag>`) plus the
per-mock random-rotation noise-meter scalars (`noise_std_<tag>`). That makes five shear estimators
trainable from ONE generation run, selected at PREBAKE time rather than in the model config:

| arm | store it trains on | how the store was made |
|---|---|---|
| A0_counts (baseline) | `_DN_A0`    | `prebake --eb-variant fwhm4_lmin56_lcut1400` |
| A1_wht_rand          | `_DN_A1`    | ... + `--noise-norm rand` (E / std of the matched rand map) |
| B1_selfstd           | `_DN_A0`    | same store as A0 + the loader knob `eb_noise_norm='self'` |
| A3s8                 | `_DN_SC8`   | `--eb-variant sc8_fwhm4_lmin56_lcut1400` |
| A3s8_A1              | `_DN_SC8A1` | ... + `--noise-norm rand` |

The question being answered: does ANY arm remove the source-galaxy-clustering (b_g) leakage that
moves the counts-normalised E amplitude by ~7σ, WITHOUT paying an unacceptable information price?
See `.claude/runs/training-runs/improved-shear-tests/plan.md` and the leaderboard in
`.claude/runs/kids-preparation/improved-shear-processing/artifacts/RESULTS.md`.

**Why `_dn` names.** Checkpoints live at `{base_path}/checkpoints/{experiment_name}/`, so reusing a
`_novd` (Era-3, counts-only store) name would mix eras in one checkpoint dir and
`get_best_checkpoint` could silently resolve the wrong one.

**One band, measured (not assumed).** `split_by_cosmology` shuffles with `random.Random(42)` over
the cosmology LIST, so ONE extra cosmology re-randomises the entire split — and the raw foundation
store is still being generated (~1 cosmology/min). The plan therefore budgeted a separate band per
bake snapshot. In the event the `a0` and `sc8` bakes globbed the growing raw store seconds apart
and captured **the same 83 001 files** — verified exhaustively by diffing the two stores' sorted
name lists, not inferred from the counts. The derived `a1`/`sc8a1` stores are then baked FROM those
finished parents, so all five arms share one file list by construction.

⇒ a SINGLE band (`kids_legacy_band_nla_m_dn`, trained on the `a0` store, which carries
`mixed_bandpowers` verbatim) is correct for every arm, and:
  * raw val NLL is directly comparable across ALL FIVE arms (identical files, split and frozen
    band; the only axis that varies is the shear processing);
  * the seed's band best is the exact per-seed baseline for "ΔNLL vs bandpowers".

If a future snapshot pair ever fails that identity check, the arms concerned need their own band
and only ΔNLL (a difference, so the val-draw offset cancels to first order) stays comparable.

Merge: `kids_legacy_dn_experiments` is `.update()`-merged into the experiments dict by train.py /
eval.py / train_embeddings.py / .claude/cluster/smoke_test_experiment.py, and by
src/ml/eval/misspec.py:_load_experiment_config.

Every MAP config carries a de-clustered `_smoke` clone on the fwhm8 single-cosmology LOCAL fixture
(`.claude/cluster/smoke_data_nla`, `E_fwhm8_lmin50_lcut1400` only). The real fwhm4 configs
false-fail that fixture, so production submits pass `--skip-smoke`.
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

# Prebaked f16 stores on gpu5 (l40s-local). Bake DAG: raw -> a0 -> a1 and raw -> sc8 -> sc8a1;
# the derived arm is baked FROM the finished parent so the file lists match exactly.
_DN_A0 = f"{_GPU5}/glass_dn_nla_m_f16_a0_{_EB}/output_*.h5"
_DN_A1 = f"{_GPU5}/glass_dn_nla_m_f16_a1_{_EB}/output_*.h5"
_DN_SC8 = f"{_GPU5}/glass_dn_nla_m_f16_sc8_{_EB}/output_*.h5"
_DN_SC8A1 = f"{_GPU5}/glass_dn_nla_m_f16_sc8a1_{_EB}/output_*.h5"

# Stage-I band checkpoint dir. ONE band serves ALL FIVE arms — see the module docstring
# ("one band, measured") for the proof.
_BAND_CKPT_DN = f"{_CKPT}/kids_legacy_band_nla_m_dn/"

# B1_selfstd: per-mock/per-bin footprint standardisation of the E maps at LOAD time
# (EBNoiseNormTransform, landed c707c94). The transform already standardises the maps, so the
# scaler must touch only the bandpowers — cf. config/kids_legacy_counts.py:869.
_EB_SELFSTD_TOP = {
    "eb_noise_norm": "self",
    "scaler_options": {"data": {"type": "standard", "keys": ["mixed_bandpowers"]},
                       "cosmo": {"type": "preset"}},
}

# Seeds. 3 band repeats (the 3rd is the spare that a plateau-rescue hybrid at r2 pairs with);
# 2 hybrid repeats per arm, extended to r2 only for a seed that plateaus.
_BAND_REPEATS = [0, 1, 2]
_HYBRID_REPEATS = (0, 1)

kids_legacy_dn_experiments = {}


# === Stage I — bandpower MLP (3 repeats, shared by every arm; v100) =============================
def _band_dn(data_patterns):
    """Stage-I bandpower encoder on the PREBAKED store (which carries `mixed_bandpowers` verbatim).

    Deliberate deviation from the Era-3 recipe, which read bandpowers off the RAW gpu4 store: that
    store is COMPLETE in Era 3, but the dual-norm foundation store is still generating, so a
    raw-fed band would see a different cosmology list — hence a different `random.Random(42)`
    split — from every hybrid that trains on the bake. Reading the bake also cuts ~13x the bytes
    per file, which was the Era-3 band's I/O bottleneck (~10 min/epoch wall on ~1.9 min of compute).
    """
    c = _band_lmin50()
    c["data_patterns"] = data_patterns
    c.pop("repeats", None)
    c["repeat_indices"] = list(_BAND_REPEATS)
    return c


kids_legacy_dn_experiments["kids_legacy_band_nla_m_dn"] = _band_dn(_DN_A0)
# NB no separate sc8 band: the a0 and sc8 snapshots are the SAME file list (see the
# module docstring), so `kids_legacy_band_nla_m_dn` is the band for every arm.


# === Stage II — the hybrid arms (2 repeats each; l40s/a100) =====================================
def _hybrid_dn(data_patterns, eb_variant, band_ckpt, top_extra=None,
               repeat_indices=_HYBRID_REPEATS):
    """PreActResNet z8 hybrid ("hybrid v2"), frozen per-repeat Stage-I band, on one arm's store.

    Byte-identical to the validated `kids_legacy_hybrid_nla_m_novd_z8_resnet` production recipe
    apart from (a) which store it reads, (b) which band it freezes, and (c) the B1 loader knob —
    which is the whole point: the ONLY axis that varies across arms is the shear processing.
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


def _hybrid_dn_smoke(top_extra=None):
    """De-clustered fwhm8-local smoke clone exercising the SAME model kwargs (from-scratch band).

    The fixture carries only `E_fwhm8_lmin50_lcut1400`, so the smoke cannot gate the real fwhm4 /
    sc8 tag — that is gated by the bake's `ok=` count. What it DOES gate is the config building,
    the resnet map encoder, and (for B1) the `eb_noise_norm` transform + scaler wiring.
    """
    c = _hybrid_lmin50_z8_smoke()
    c["model_kwargs"] = {**c["model_kwargs"], "map_kwargs": _RESNET_MAPKW}
    for k, v in (top_extra or {}).items():
        c[k] = v
    return c


# --- the four arms ------------------------------------------------------------------------------
# A0_counts — the BASELINE: the estimator whose b_g leakage (+6.9σ/+8.6σ paired) motivated the work.
kids_legacy_dn_experiments["kids_legacy_hybrid_nla_m_dn_z8_resnet_a0"] = \
    _hybrid_dn(_DN_A0, _EB, _BAND_CKPT_DN)

# A1_wht_rand — the DEPLOYED recommendation: E divided by the std of its matched random-rotation
# noise map (a per-mock noise meter). Paired residual −0.2σ/−0.35σ. Applied at prebake time, so the
# model config is identical to A0's apart from the store.
kids_legacy_dn_experiments["kids_legacy_hybrid_nla_m_dn_z8_resnet_a1"] = \
    _hybrid_dn(_DN_A1, _EB, _BAND_CKPT_DN)

# B1_selfstd — per-mock/bin self-standardisation at load time. Same store as A0; the amplitude
# information is deliberately discarded from the map branch (the bandpower branch keeps it).
kids_legacy_dn_experiments["kids_legacy_hybrid_nla_m_dn_z8_resnet_b1"] = \
    _hybrid_dn(_DN_A0, _EB, _BAND_CKPT_DN, top_extra=_EB_SELFSTD_TOP)

# A3s8_A1 — smoothed-counts denominator (8') BEFORE the spin-2 SHT, plus the A1 rescale. Carried
# for robustness against SPATIALLY STRUCTURED misspecification, which a single per-mock scalar
# cannot absorb; on b_g it ties A1.
kids_legacy_dn_experiments["kids_legacy_hybrid_nla_m_dn_z8_resnet_sc8a1"] = \
    _hybrid_dn(_DN_SC8A1, _EB_SC8, _BAND_CKPT_DN)

# A3s8 alone (OPTIONAL 5th arm — the sc8 intermediate is already on disk, so it costs 0 GB).
kids_legacy_dn_experiments["kids_legacy_hybrid_nla_m_dn_z8_resnet_sc8"] = \
    _hybrid_dn(_DN_SC8, _EB_SC8, _BAND_CKPT_DN)

# --- smoke clones --------------------------------------------------------------------------------
for _name in ("a0", "a1", "sc8", "sc8a1"):
    kids_legacy_dn_experiments[f"kids_legacy_hybrid_nla_m_dn_z8_resnet_{_name}_smoke"] = \
        _hybrid_dn_smoke()
kids_legacy_dn_experiments["kids_legacy_hybrid_nla_m_dn_z8_resnet_b1_smoke"] = \
    _hybrid_dn_smoke(top_extra=_EB_SELFSTD_TOP)
# NB no band `_smoke` clones: the band carries no `eb_map_variant`, so the real band config passes
# the local fixture gate directly (the harness overrides data_patterns and repeat_indices).
