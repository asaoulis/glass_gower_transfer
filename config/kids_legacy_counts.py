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
