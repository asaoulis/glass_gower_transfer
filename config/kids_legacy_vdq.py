"""VDQ campaign (variable depth, QUADCAP table; b_g prior marginalised) — model configs.

Campaign definition: `.claude/runs/training-runs/VD-final-train/plan.md` (§2 D1–D10) and
`STATUS.md` in the same dir. Rows: `models_checklist.md` (models) · `datasets_checklist.md` (data).

**What is different from `_bgp`.** Nothing in the training recipe: every row below is built by
CALLING the `config/kids_legacy_bgp.py` / `config/kids_legacy.py` factories and then overriding
only stores and checkpoint dirs. The one axis that changed is upstream, in the data: the mocks are
generated with the KiDS-Legacy variable-depth model, per-patch capped-quadratic count contrast
(`--vd-table quadcap`, commit 28e4f93, the simulator default since VD-final-train), with the same
`flamingo_pt_diag` (kappa = 1) b_g prior, `--shear-norm counts` and sc8only stores as `_bgp`.
The no-VD production CNN is biased by real depth (Omega_m -3.2 sigma / sigma_8 +5.1 sigma /
S8 +1.7 sigma, `simulation-runs/vd-model-aprime/artifacts/APRIME_SUMMARY.md` §8–10), so **no `_bgp`
result is a baseline here** — val NLLs, FoMs and the Tier-2 calibration are re-measured.

**Why `vdq` names.** Checkpoints live at `{base_path}/checkpoints/{experiment_name}/`; the token
`vdq` is used verbatim on raw stores, bakes and experiment names so greps line up across the three,
and no old checkpoint dir can be resolved by accident.

Stores (sc8a1 = sc8 E map + A1 random-rotation rescale, f16, BARE `E` groups ⇒ `eb_map_variant=None`):
  GLASS   gpu4 raw `glass_mocks_nla_m_vdq_bgp`  -> gpu5 `glass_vdq_nla_m_f16_sc8a1_fwhm4_lmin56_lcut1400`
  Gower   gpu4 raw `gower_mocks_nla_m_vdq_bgp`  -> gpu5 `gower_vdq_nla_m_f16_sc8a1_fwhm4_lmin56_lcut1400`
  canary  snapshot of the GLASS raw store at >= 30 000 files
          -> gpu5 `glass_vdq_nla_m_canary_f16_sc8a1_fwhm4_lmin56_lcut1400` (M0 only, never promoted)

FORCED, DOCUMENTED DEVIATIONS from the `_bgp` reference rows (enforced by
`.claude/runs/training-runs/VD-final-train/artifacts/check_vdq_configs.py`):
  1. The Stage-I band reads the **sc8a1** bake (the `_bgp` band read the a1 bake). An sc8only store
     has no a1 product, and band + hybrid sharing ONE bake is the safer file-list choice (the band's
     train split must be the hybrid's train split).
  2. The pack has its own factory: `_npe_finetune_bgp` hardcodes the `_bgp` foundation ckpt.
  3. The hybrid lists r0–r4 (the `_bgp` reference row lists (0, 1); its r2–r4 were trained via
     `--repeat-indices`). Spares r5, r6 are passed at the CLI.

Launch lines (full flags + verification strings: `models_checklist.md`):
  train --exp kids_legacy_band_nla_m_vdq --gpu v100 --repeat-indices 0,1,2,3        (and 4,5,6)
  train --exp kids_legacy_hybrid_nla_m_vdq_z8_resnet_sc8a1 --gpu l40s --repeat-indices <r> --skip-smoke --wall_h 48
  train --exp gower_npe_finetune_nla_m_vdq_z8_ens1 --gpu a100 --repeat-indices 0,1,2,3,4 --wall_h 12 --skip-smoke
  train --exp gower_npe_finetune_band_nla_m_vdq_ens9 --gpu v100 --repeat-indices <r> --skip-smoke
  embed --gpu v100 --target glass_nle_pretrain_nla_m_vdq_z8_r<r> --sources kids_legacy_hybrid_nla_m_vdq_z8_resnet_sc8a1
  embed --cpu --target gower_nle_finetune_nla_m_vdq_z8_r<r>_ens9 --sources kids_legacy_hybrid_nla_m_vdq_z8_resnet_sc8a1

Merge: `kids_legacy_vdq_experiments` is `.update()`-merged by train.py / eval.py / train_embeddings.py /
gen_samples.py / .claude/cluster/smoke_test_experiment.py / src/ml/eval/misspec.py:_load_experiment_config /
src/ml/eval/pooling.py, and scanned by run_remote.py:known_experiment (literal keys — keep them literal).

Smoke gate: the local fixture carries only `E_fwhm8_lmin50_lcut1400` and ONE cosmology, so the
sc8a1 / pack / NLE rows false-fail it (and resolve cluster ckpts) — they launch with `--skip-smoke`
and are gated by the dict-diff check instead. The band and the `_smoke` clone pass the fixture.
"""
from config.kids_legacy import _nle_finetune
from config.kids_legacy import _npe_finetune_z8
from config.kids_legacy_bgp import (
    _EB,
    _GOWER_TEST_IDS,
    _GOWER_TEST_IDS_100,
    _NLE_REPEATS,
    _BGP_NLE_PROJECT,
    _RESNET_MAPKW,
    _assert_final_summary_dim,
    _band_bgp,
    _hybrid_bgp,
    _hybrid_bgp_p15,
    _hybrid_bgp_smoke,
    _nle_bake_repeat,
    _nle_pretrain_bgp,
    _npe_finetune_band_bgp,
)

# --- stores --------------------------------------------------------------------------------------
_GPU5 = "/share/gpu5/asaoulis/transfer_datasets"
_CKPT = "/share/gpu5/asaoulis/transfer_models/checkpoints"

_VDQ_GLASS = f"{_GPU5}/glass_vdq_nla_m_f16_sc8a1_{_EB}/output_*.h5"
_VDQ_GLASS_CANARY = f"{_GPU5}/glass_vdq_nla_m_canary_f16_sc8a1_{_EB}/output_*.h5"
_VDQ_GOWER = f"{_GPU5}/gower_vdq_nla_m_f16_sc8a1_{_EB}/output_*.h5"
_VDQ_GOWER_NLA = f"{_GPU5}/gower_vdq_nla_f16_sc8a1_{_EB}/output_*.h5"
_VDQ_GOWER_NLA_Z = f"{_GPU5}/gower_vdq_nla_z_f16_sc8a1_{_EB}/output_*.h5"

# --- checkpoint dirs -----------------------------------------------------------------------------
_BAND_CKPT_VDQ = f"{_CKPT}/kids_legacy_band_nla_m_vdq/"
_VDQ_HYB_CKPT = f"{_CKPT}/kids_legacy_hybrid_nla_m_vdq_z8_resnet_sc8a1/"
_BAND_CKPT_VDQ_CANARY = f"{_CKPT}/kids_legacy_band_nla_m_vdq_canary/"
_VDQ_HYB_CKPT_CANARY = f"{_CKPT}/kids_legacy_hybrid_nla_m_vdq_z8_resnet_sc8a1_canary/"

_PACK_REPEATS = [0, 1, 2, 3, 4]
_HYBRID_REPEATS_VDQ = (0, 1, 2, 3, 4)

kids_legacy_vdq_experiments = {}


# === Stage I — bandpower MLP (r0–r4; spares r5, r6 via --repeat-indices; v100) ===================
# `_band_bgp` already sets repeat_indices [0..4]. Deviation 1: reads the sc8a1 bake.
kids_legacy_vdq_experiments["kids_legacy_band_nla_m_vdq"] = _band_bgp(_VDQ_GLASS)


# === Stage II — the GLASS hybrid foundation (sc8a1, PreActResNet z8; l40s) ========================
kids_legacy_vdq_experiments["kids_legacy_hybrid_nla_m_vdq_z8_resnet_sc8a1"] = \
    _hybrid_bgp(_VDQ_GLASS, None, _BAND_CKPT_VDQ, repeat_indices=_HYBRID_REPEATS_VDQ)

# fwhm8-fixture smoke clone: gates config building + the resnet map-encoder wiring only.
kids_legacy_vdq_experiments["kids_legacy_hybrid_nla_m_vdq_z8_resnet_sc8a1_smoke"] = _hybrid_bgp_smoke()


# === THE NPE PACK — 5 single-model Gower NPE finetunes (the misspec / Tier-2 encoder pack) ========
# Mirror of `gower_npe_finetune_nla_m_bgp_z8_ens1`. Deviation 2: own factory (the `_bgp` one
# hardcodes its foundation ckpt). ⭐ map_kwargs=_RESNET_MAPKW is CRITICAL: `_npe_finetune_z8` starts
# from a bare `_hybrid_lmin50_z8()` (UNet); the STRICT `checkpoint_path` loader would then refuse the
# PreActResNet weights — loud, but a wasted job. ensemble_repeats=1 ⇒ run dirs `finetune_ncosmo300_{r}`.
_FINETUNE_VAL_CHECK_INTERVAL = 0.25


def _npe_pack_vdq(hyb_ckpt, repeat_indices=_PACK_REPEATS, data_patterns=_VDQ_GOWER):
    c = _npe_finetune_z8(hyb_ckpt, data_patterns=data_patterns, eb_variant=None)
    c["model_kwargs"] = {**c["model_kwargs"], "map_kwargs": _RESNET_MAPKW}
    c["fixed_test_sim_ids"] = _GOWER_TEST_IDS
    c["project"] = "gower-finetuning"
    c["ensemble_repeats"] = 1
    c["repeat_indices"] = list(repeat_indices)
    # Deviation 3 (user, 2026-10-07): validate 4x per epoch. The BGP 10-epoch finetunes always picked
    # ep 1-2 from only 10 val points; 40 points give the best-ckpt selector a real minimum to find.
    c["val_check_interval"] = _FINETUNE_VAL_CHECK_INTERVAL
    return c


kids_legacy_vdq_experiments["gower_npe_finetune_nla_m_vdq_z8_ens1"] = _assert_final_summary_dim(
    _npe_pack_vdq(_VDQ_HYB_CKPT), 8, "gower_npe_finetune_nla_m_vdq_z8_ens1")


# === M0 CANARY rows (r0 only; canary snapshot store + canary ckpts; NEVER promoted/quoted) ==========
_band_canary = _band_bgp(_VDQ_GLASS_CANARY)
_band_canary["repeat_indices"] = [0]
kids_legacy_vdq_experiments["kids_legacy_band_nla_m_vdq_canary"] = _band_canary
kids_legacy_vdq_experiments["kids_legacy_hybrid_nla_m_vdq_z8_resnet_sc8a1_canary"] = \
    _hybrid_bgp(_VDQ_GLASS_CANARY, None, _BAND_CKPT_VDQ_CANARY, repeat_indices=(0,))
kids_legacy_vdq_experiments["gower_npe_finetune_nla_m_vdq_z8_canary_ens1"] = _assert_final_summary_dim(
    _npe_pack_vdq(_VDQ_HYB_CKPT_CANARY, repeat_indices=[0]), 8,
    "gower_npe_finetune_nla_m_vdq_z8_canary_ens1")


# === P2b — the 2-pt-only Gower NPE ens9 pack (band arch, `_npe_finetune_z8` recipe) ==============
# `_npe_finetune_band_bgp` builds the arch from `_band_bgp` (only data_patterns differs between
# bakes, and it is overwritten here) and hardcodes the `_bgp` band ckpt ⇒ override it.
_band_pack = _npe_finetune_band_bgp(_VDQ_GOWER)
_band_pack["checkpoint_path"] = _BAND_CKPT_VDQ
_band_pack["val_check_interval"] = _FINETUNE_VAL_CHECK_INTERVAL
kids_legacy_vdq_experiments["gower_npe_finetune_band_nla_m_vdq_ens9"] = _assert_final_summary_dim(
    _band_pack, 8, "gower_npe_finetune_band_nla_m_vdq_ens9")


# === P3 — whitened NLE chain (Stage A on GLASS, Stage B ens9 on Gower), one row per repeat ========
# Stage A: source encoder passed at the CLI (--sources kids_legacy_hybrid_nla_m_vdq_z8_resnet_sc8a1);
# VERIFY at launch that the embedding cache path names the **vdq** foundation (the e890aec bug class).
def _nle_a_vdq(r):
    return _nle_pretrain_bgp(_VDQ_GLASS, r)


# Stage B: the `_bgp` M4b loop body verbatim, with gower_data=_VDQ_GOWER and the vdq Stage-A name.
def _nle_b_vdq(r):
    ft = _nle_finetune(f"glass_nle_pretrain_nla_m_vdq_z8_r{r}", ensemble_repeats=9,
                       whiten_k=8, warmstart_max_gap_nats=22.0,
                       gower_data=_VDQ_GOWER, gower_eb=None)
    ft["max_trainval_cosmos"] = [300]
    ft["train_frac"] = 0.8
    ft["val_frac"] = 0.2
    ft["test_frac"] = 0.0        # test = the fixed 200 ids; fracs must sum to 1.0
    ft["fixed_test_sim_ids"] = _GOWER_TEST_IDS
    ft["project"] = _BGP_NLE_PROJECT
    return _nle_bake_repeat(ft, r)


assert tuple(_NLE_REPEATS) == (0, 1, 2, 3, 4), _NLE_REPEATS
# Literal keys (not an f-string loop) so run_remote.py:known_experiment's regex scan finds them.
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdq_z8_r0"] = _nle_a_vdq(0)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdq_z8_r1"] = _nle_a_vdq(1)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdq_z8_r2"] = _nle_a_vdq(2)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdq_z8_r3"] = _nle_a_vdq(3)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdq_z8_r4"] = _nle_a_vdq(4)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdq_z8_r0_ens9"] = _nle_b_vdq(0)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdq_z8_r1_ens9"] = _nle_b_vdq(1)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdq_z8_r2_ens9"] = _nle_b_vdq(2)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdq_z8_r3_ens9"] = _nle_b_vdq(3)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdq_z8_r4_ens9"] = _nle_b_vdq(4)



# === P3b — the 2-pt (bandpower) NLE chain: the `_bgp` M16 k=8 procedure on the VDQ stores ========
# Mirror of `glass_nle_pretrain_band_nla_m_bgp_k8_r{r}` -> `gower_nle_finetune_band_nla_m_bgp_k8_r{r}_ens9_e150`
# (config/kids_legacy_bgp.py `_register_band_nle_chain`): the frozen source is the per-repeat GLASS
# band MLP `kids_legacy_band_nla_m_vdq/pretrain_ncosmoNone_{r}` (ONE compression head per repeat — NOT
# the Gower ens9 NPE pack), whiten k=8 (pure-whiten, the production arm), Stage-B 150 epochs, guard-c 50.
# GLASS store = the band's OWN training store so the split reproduces the band's (Deviation 1: that is
# the sc8a1 bake here; bandpowers are byte-identical across bakes). Launch:
#   embed --gpu v100 --mem-gb 32 --target glass_nle_pretrain_band_nla_m_vdq_k8_r<r> --sources kids_legacy_band_nla_m_vdq
#   embed --cpu --partition CORES40 --target gower_nle_finetune_band_nla_m_vdq_k8_r<r>_ens9_e150 --sources kids_legacy_band_nla_m_vdq
# VERIFY Stage-A: `Loaded keys: 14` (KidsBandpowersMLP), summary dim 8, `[whiten] Fit whitener k=8`.
_BAND_NLE_K_VDQ = 8


def _band_nle_a_vdq(r):
    pre = _nle_pretrain_bgp(_VDQ_GLASS, r)
    pre["whiten_embeddings"] = {"k": _BAND_NLE_K_VDQ}
    return pre


def _band_nle_b_vdq(r):
    ft = _nle_finetune(f"glass_nle_pretrain_band_nla_m_vdq_k{_BAND_NLE_K_VDQ}_r{r}", ensemble_repeats=9,
                       whiten_k=_BAND_NLE_K_VDQ, warmstart_max_gap_nats=50.0,
                       gower_data=_VDQ_GOWER, gower_eb=None)
    ft["max_trainval_cosmos"] = [300]
    ft["train_frac"] = 0.8
    ft["val_frac"] = 0.2
    ft["test_frac"] = 0.0        # test = the fixed 200 ids; fracs must sum to 1.0
    ft["fixed_test_sim_ids"] = _GOWER_TEST_IDS
    ft["epochs"] = 150
    ft["project"] = _BGP_NLE_PROJECT
    return _nle_bake_repeat(ft, r)


# Literal keys (run_remote.py:known_experiment scans for them).
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_m_vdq_k8_r0"] = _band_nle_a_vdq(0)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_m_vdq_k8_r1"] = _band_nle_a_vdq(1)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_m_vdq_k8_r2"] = _band_nle_a_vdq(2)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_m_vdq_k8_r3"] = _band_nle_a_vdq(3)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_m_vdq_k8_r4"] = _band_nle_a_vdq(4)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_m_vdq_k8_r0_ens9_e150"] = _band_nle_b_vdq(0)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_m_vdq_k8_r1_ens9_e150"] = _band_nle_b_vdq(1)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_m_vdq_k8_r2_ens9_e150"] = _band_nle_b_vdq(2)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_m_vdq_k8_r3_ens9_e150"] = _band_nle_b_vdq(3)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_m_vdq_k8_r4_ens9_e150"] = _band_nle_b_vdq(4)

# ⭐ guard-c 50 -> 100 for **r3 ONLY** (user decision 2026-10-08 20:20Z). Stage-B r3 job 1382660 died on
# member 4: ep0 gaps r3 = {3.021, 2.179, 32.416, 1.748, 68.537}; every other repeat's 30 members sat at
# 1.458-6.125. The r3 whitener is well conditioned (EVR [0.4031 ... 0.0012]), so this is a member-specific
# val split holding outlier Gower rows (the bgp M16 r1 22.9-nat pattern, which raised 22 -> 50), not a
# broken warm start: a genuine scratch init reads ~4800 nats (e890aec). The other repeats keep 50.
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_m_vdq_k8_r3_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 100.0

# === P3 — the 15-param GLASS foundation (9 cosmo/IA + b_g_bin1..6), 125 epochs, r0–r4 ============
kids_legacy_vdq_experiments["kids_legacy_hybrid_nla_m_vdq_z8_resnet_sc8a1_p15"] = \
    _hybrid_bgp_p15(_VDQ_GLASS, _BAND_CKPT_VDQ)


# === Source encoders of the VDQ NLE Stage-B rows (read by src/ml/eval/nle_external.py) ============
# The frozen source encoder is part of a chain's identity (the e890aec bug class): register it here,
# exactly as passed to `embed --sources` at Stage A/B (models task VD-final-train-models), so the
# external (real-observation / matched-mock) sampler never guesses it.
VDQ_NLE_SOURCES = {}
for _r in _NLE_REPEATS:
    VDQ_NLE_SOURCES[f"glass_nle_pretrain_nla_m_vdq_z8_r{_r}"] = "kids_legacy_hybrid_nla_m_vdq_z8_resnet_sc8a1"
    VDQ_NLE_SOURCES[f"gower_nle_finetune_nla_m_vdq_z8_r{_r}_ens9"] = "kids_legacy_hybrid_nla_m_vdq_z8_resnet_sc8a1"
    VDQ_NLE_SOURCES[f"glass_nle_pretrain_band_nla_m_vdq_k{_BAND_NLE_K_VDQ}_r{_r}"] = "kids_legacy_band_nla_m_vdq"
    VDQ_NLE_SOURCES[f"gower_nle_finetune_band_nla_m_vdq_k{_BAND_NLE_K_VDQ}_r{_r}_ens9_e150"] = "kids_legacy_band_nla_m_vdq"
assert set(VDQ_NLE_SOURCES) <= set(kids_legacy_vdq_experiments), \
    sorted(set(VDQ_NLE_SOURCES) - set(kids_legacy_vdq_experiments))


# === PV-G — GLASS VDQ ANALYSIS VARIATES: encoder finetunes + NLE Stage A (user 2026-10-12) =========
# The `_bgp` M5a/M6a/M12 variate recipe (`_encoder_finetune_bgp`: whole foundation encoder = band + map
# warm-started, FRESH flow head, FIXED lr 2e-4, 100 epochs, z8) with ONE override: the warm start is the
# NLA-M **VDQ** foundation (`_VDQ_HYB_CKPT`, resolved per repeat as `_{r}` via match_num_cosmo=False)
# instead of the hardcoded `_bgp` one. Epochs are deliberately NOT raised for the larger stores (user).
# Stores: SIMS GLASS twins (4000 cosmologies x 4 mocks, outer-reps 1; datasets_checklist §G).
#
# kappa=2 (VD-final-train/plan.md 'FUTURE ARM — kappa=2', user 2026-10-09): NOT a 15-param p15 row but
# TWO NPEs on the kappa=2 store, both from the same VDQ foundation:
#   _cosmo : the 9 NLA-M params, b_g marginalised over the broad prior; theta + scaler are identical to
#            the foundation's, so its 9-param flow head is ALSO preloaded (as many weights as possible).
#   _bg    : the 6 b_g_bin params ONLY (kappa=2 clipped boxes `_BG_BOXES_K2`), fresh flow head.
#
# NLE Stage A (GLASS only — the Gower nla/nla_z stores were left unbaked, Gower k2 never ran): k=5, the
# documented variate value (`_hf` rows: pure-whitening a rank-deficient variate summary amplifies the
# near-null PCs). theta + a_ia box MUST equal the paired encoder's. Launch:
#   embed --gpu v100 --mem-gb 32 --target glass_nle_pretrain_<v>_vdq_z8_k5_r<r> --sources <encoder row>
from config.kids_legacy import _A_IA_NLA_BOX, _COSMO_8_NLA, _COSMO_9_NLAZ  # noqa: E402
from config.kids_legacy_bgp import _BG_BOXES_K2, _COSMO_9_NLAM, _encoder_finetune_bgp  # noqa: E402

_VDQ_GLASS_NLA = f"{_GPU5}/glass_vdq_nla_f16_sc8a1_{_EB}/output_*.h5"
_VDQ_GLASS_NLA_Z = f"{_GPU5}/glass_vdq_nla_z_f16_sc8a1_{_EB}/output_*.h5"
_VDQ_GLASS_K2 = f"{_GPU5}/glass_vdqk2_nla_m_f16_sc8a1_{_EB}/output_*.h5"
_BG_THETA = [f"b_g_bin{_i}" for _i in range(1, 7)]
_VARIATE_NLE_K_VDQ = 5


def _encoder_finetune_vdq(data_patterns, cosmo_param_names, preset_overrides=None, load_flow=False):
    c = _encoder_finetune_bgp(data_patterns, cosmo_param_names, preset_overrides=preset_overrides)
    c["pretrained_embedding_ckpt_path"] = _VDQ_HYB_CKPT
    if load_flow:
        c["load_pretrained_flow"] = True
        c["pretrained_flow_ckpt_path"] = _VDQ_HYB_CKPT
    return c


kids_legacy_vdq_experiments["glass_encoder_finetune_nla_vdq_z8"] = _assert_final_summary_dim(
    _encoder_finetune_vdq(_VDQ_GLASS_NLA, _COSMO_8_NLA, preset_overrides=_A_IA_NLA_BOX), 8,
    "glass_encoder_finetune_nla_vdq_z8")
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_z_vdq_z8"] = _assert_final_summary_dim(
    _encoder_finetune_vdq(_VDQ_GLASS_NLA_Z, _COSMO_9_NLAZ, preset_overrides=_A_IA_NLA_BOX), 8,
    "glass_encoder_finetune_nla_z_vdq_z8")
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vdqk2_z8_cosmo"] = _assert_final_summary_dim(
    _encoder_finetune_vdq(_VDQ_GLASS_K2, _COSMO_9_NLAM, load_flow=True), 8,
    "glass_encoder_finetune_nla_m_vdqk2_z8_cosmo")
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vdqk2_z8_bg"] = _assert_final_summary_dim(
    _encoder_finetune_vdq(_VDQ_GLASS_K2, _BG_THETA, preset_overrides=_BG_BOXES_K2), 8,
    "glass_encoder_finetune_nla_m_vdqk2_z8_bg")


def _variate_nle_a_vdq(data_patterns, r, cosmo_param_names, preset_overrides=None):
    kw = {"cosmo_param_names": list(cosmo_param_names)}
    if preset_overrides is not None:
        kw["preset_overrides"] = preset_overrides
    pre = _nle_pretrain_bgp(data_patterns, r, **kw)
    pre["whiten_embeddings"] = {"k": _VARIATE_NLE_K_VDQ}
    return pre


# Literal keys (run_remote.py:known_experiment scans for them).
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_k5_r0"] = _variate_nle_a_vdq(_VDQ_GLASS_NLA, 0, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_k5_r1"] = _variate_nle_a_vdq(_VDQ_GLASS_NLA, 1, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_k5_r2"] = _variate_nle_a_vdq(_VDQ_GLASS_NLA, 2, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_k5_r3"] = _variate_nle_a_vdq(_VDQ_GLASS_NLA, 3, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_k5_r4"] = _variate_nle_a_vdq(_VDQ_GLASS_NLA, 4, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_z_vdq_z8_k5_r0"] = _variate_nle_a_vdq(_VDQ_GLASS_NLA_Z, 0, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_z_vdq_z8_k5_r1"] = _variate_nle_a_vdq(_VDQ_GLASS_NLA_Z, 1, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_z_vdq_z8_k5_r2"] = _variate_nle_a_vdq(_VDQ_GLASS_NLA_Z, 2, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_z_vdq_z8_k5_r3"] = _variate_nle_a_vdq(_VDQ_GLASS_NLA_Z, 3, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_z_vdq_z8_k5_r4"] = _variate_nle_a_vdq(_VDQ_GLASS_NLA_Z, 4, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k5_r0"] = _variate_nle_a_vdq(_VDQ_GLASS_K2, 0, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k5_r1"] = _variate_nle_a_vdq(_VDQ_GLASS_K2, 1, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k5_r2"] = _variate_nle_a_vdq(_VDQ_GLASS_K2, 2, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k5_r3"] = _variate_nle_a_vdq(_VDQ_GLASS_K2, 3, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k5_r4"] = _variate_nle_a_vdq(_VDQ_GLASS_K2, 4, _COSMO_9_NLAM)

# Source encoder per Stage-A row (the e890aec bug class: a chain's encoder is part of its identity).
VDQ_VARIATE_NLE_SOURCES = {}
for _r in _NLE_REPEATS:
    VDQ_VARIATE_NLE_SOURCES[f"glass_nle_pretrain_nla_vdq_z8_k5_r{_r}"] = "glass_encoder_finetune_nla_vdq_z8"
    VDQ_VARIATE_NLE_SOURCES[f"glass_nle_pretrain_nla_z_vdq_z8_k5_r{_r}"] = "glass_encoder_finetune_nla_z_vdq_z8"
    VDQ_VARIATE_NLE_SOURCES[f"glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k5_r{_r}"] = \
        "glass_encoder_finetune_nla_m_vdqk2_z8_cosmo"
assert set(VDQ_VARIATE_NLE_SOURCES) <= set(kids_legacy_vdq_experiments), \
    sorted(set(VDQ_VARIATE_NLE_SOURCES) - set(kids_legacy_vdq_experiments))


# === PV-G kappa=2 JOINT-SUMMARY TEST — NPE heads on concat(cosmo, b_g) embeddings (user 2026-10-12) ==
# Question: the joint 15-param VMIM NPE lost FoM(Omega_m, sigma_8) vs the 9-param one (kappa=1 p15:
# ~21-25 %). Does compressing SEPARATELY (the `_cosmo` 9-param + `_bg` 6-param encoders above) and then
# fitting ONE 15-param NPE head on the CONCATENATED 16-D summary keep both encoders' constraining power?
# ONLY IF YES does the joint NLE (concat + whiten/truncate) Stage A follow; otherwise flag the user.
#
# Instrument = an NPE head on FROZEN embeddings (`train_embeddings.py`, `compute_embeddings` concatenates
# sources feature-wise: z = [z_cosmo(8) | z_bg(8)]). NPE models p(theta|z) — a density over THETA — so its
# FoM / test log-prob are comparable across summaries (an NLE's are not). Precedent + caveats: the
# archived stack5 ablation (config/archive/legacy_bgp_stack5.py).
#   * joint heads, 15-param theta (9 NLA-M + b_g_bin1..6, kappa=2 boxes): whiten k=16 (pure-whiten, the
#     TOTAL information), k=12, k=10 (truncations priced against k=16). The k=12/k=10 rows REUSE the k=16
#     row's raw-embedding cache (the cache stores RAW z; whitening is per run) — launch them after it.
#   * MATCHED CONTROLS (same pipeline, same frozen single-augmentation regime — a frozen-embedding head
#     overfits vs the end-to-end encoder, so the encoders' own val NLLs are not the like-for-like bar):
#     `_cosmo_ctl` = 9-param head on z_cosmo alone, `_bg_ctl` = 6-param head on z_bg alone, both k=8.
#   Pass = joint FoM(Om,s8) ~ cosmo_ctl (within seed spread) AND joint b_g widths ~ bg_ctl.
#   Secondary bar: the encoders' own end-to-end heads, scored by eval.py on the SAME split (same store,
#   same split_seed = 42 + r, same fracs).
# run_evaluation=True: NPE sampling is cheap (no MCMC), and it writes evaluation_results.json (FoM, TARP).
# Launch (sources comma-joined, cosmo FIRST so z[:8] is the cosmo summary):
#   embed --gpu v100 --mem-gb 32 --wall_h 6 --target glass_npe_pretrain_nla_m_vdqk2_joint_k16_r<r>
#         --sources glass_encoder_finetune_nla_m_vdqk2_z8_cosmo,glass_encoder_finetune_nla_m_vdqk2_z8_bg
_K2_JOINT_SOURCES = "glass_encoder_finetune_nla_m_vdqk2_z8_cosmo,glass_encoder_finetune_nla_m_vdqk2_z8_bg"
_K2_JOINT_THETA = list(_COSMO_9_NLAM) + list(_BG_THETA)
assert len(_K2_JOINT_THETA) == 15, _K2_JOINT_THETA


def _k2_npe_head(r, k, theta, preset_overrides=None, cache_name=None):
    kw = {"cosmo_param_names": list(theta)}
    if preset_overrides is not None:
        kw["preset_overrides"] = preset_overrides
    c = _nle_pretrain_bgp(_VDQ_GLASS_K2, r, **kw)
    c["inference_mode"] = "npe"
    c["run_evaluation"] = True
    c["whiten_embeddings"] = {"k": int(k)}
    if cache_name is not None:
        c["embedding_cache_name"] = cache_name
        c["reuse_embedding_cache"] = True
    return c


# Literal keys (run_remote.py:known_experiment scans for them).
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k16_r0"] = _k2_npe_head(0, 16, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r0")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k12_r0"] = _k2_npe_head(0, 12, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r0")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k10_r0"] = _k2_npe_head(0, 10, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r0")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_cosmo_ctl_r0"] = _k2_npe_head(0, 8, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_bg_ctl_r0"] = _k2_npe_head(0, 8, _BG_THETA, _BG_BOXES_K2)
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k16_r1"] = _k2_npe_head(1, 16, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r1")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k12_r1"] = _k2_npe_head(1, 12, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r1")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k10_r1"] = _k2_npe_head(1, 10, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r1")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_cosmo_ctl_r1"] = _k2_npe_head(1, 8, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_bg_ctl_r1"] = _k2_npe_head(1, 8, _BG_THETA, _BG_BOXES_K2)
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k16_r2"] = _k2_npe_head(2, 16, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r2")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k12_r2"] = _k2_npe_head(2, 12, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r2")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k10_r2"] = _k2_npe_head(2, 10, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r2")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_cosmo_ctl_r2"] = _k2_npe_head(2, 8, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_bg_ctl_r2"] = _k2_npe_head(2, 8, _BG_THETA, _BG_BOXES_K2)
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k16_r3"] = _k2_npe_head(3, 16, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r3")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k12_r3"] = _k2_npe_head(3, 12, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r3")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k10_r3"] = _k2_npe_head(3, 10, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r3")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_cosmo_ctl_r3"] = _k2_npe_head(3, 8, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_bg_ctl_r3"] = _k2_npe_head(3, 8, _BG_THETA, _BG_BOXES_K2)
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k16_r4"] = _k2_npe_head(4, 16, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r4")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k12_r4"] = _k2_npe_head(4, 12, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r4")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_joint_k10_r4"] = _k2_npe_head(4, 10, _K2_JOINT_THETA, _BG_BOXES_K2, "vdqk2_joint_r4")
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_cosmo_ctl_r4"] = _k2_npe_head(4, 8, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_npe_pretrain_nla_m_vdqk2_bg_ctl_r4"] = _k2_npe_head(4, 8, _BG_THETA, _BG_BOXES_K2)


# === PV-G k=6 NLE Stage A for nla / nla_z (user 2026-10-12: "Yes, use k=6") ======================
# The k=5 nla rows above kept only 99.84-99.95 % of the summary variance on the VDQ encoders (r0 0.9995,
# r1 0.9984, r2 0.9986, r3 0.9992, r4 0.9987; the `_bgp` nla chain kept 99.96 %), and KSWEEP priced a
# ~0.3 % truncation at ~13 % FoM. k=6 is the production value for the VDQ variate NLE chains; the k=5
# rows stay as the record of what ran. New names per k (the flow context width and the persisted
# whitener are k-specific; Stage B must use the same k).
_VARIATE_NLE_K6_VDQ = 6


def _variate_nle_a_vdq_k6(data_patterns, r, cosmo_param_names, preset_overrides=None):
    pre = _variate_nle_a_vdq(data_patterns, r, cosmo_param_names, preset_overrides)
    pre["whiten_embeddings"] = {"k": _VARIATE_NLE_K6_VDQ}
    return pre


kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_k6_r0"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA, 0, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_k6_r1"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA, 1, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_k6_r2"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA, 2, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_k6_r3"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA, 3, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_k6_r4"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA, 4, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_z_vdq_z8_k6_r0"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA_Z, 0, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_z_vdq_z8_k6_r1"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA_Z, 1, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_z_vdq_z8_k6_r2"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA_Z, 2, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_z_vdq_z8_k6_r3"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA_Z, 3, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_z_vdq_z8_k6_r4"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA_Z, 4, _COSMO_9_NLAZ, _A_IA_NLA_BOX)

for _r in _NLE_REPEATS:
    VDQ_VARIATE_NLE_SOURCES[f"glass_nle_pretrain_nla_vdq_z8_k6_r{_r}"] = "glass_encoder_finetune_nla_vdq_z8"
    VDQ_VARIATE_NLE_SOURCES[f"glass_nle_pretrain_nla_z_vdq_z8_k6_r{_r}"] = "glass_encoder_finetune_nla_z_vdq_z8"
assert set(VDQ_VARIATE_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)


# === PV-G kappa=2 JOINT NLE Stage A (user 2026-10-12: proceed if the joint-summary NPE test holds) ==
# Test result (artifacts/k2_joint_test/K2_JOINT_TEST.md, r0-r3): NPE heads on concat(z_cosmo, z_bg) keep
# the encoders' power at every k tried — paired FoM(Om,s8) joint/cosmo_ctl k16 1.04+-0.11, k12 1.05+-0.04,
# k10 1.07+-0.07; b_g widths 0.95-0.999x the b_g-only control. k10 is the smallest k tested (cumulative
# variance 99.78-99.90 %) and loses nothing on either side => the joint NLE uses k=10.
# 15-param theta (9 NLA-M + b_g_bin1..6, kappa=2 boxes), sources cosmo,bg (same order as the test), and
# the SAME raw-embedding cache as the test's joint heads (`vdqk2_joint_r<r>`; computed if absent).
# Launch: embed --gpu v100 --mem-gb 32 --target glass_nle_pretrain_nla_m_vdqk2_joint_k10_r<r>
#         --sources glass_encoder_finetune_nla_m_vdqk2_z8_cosmo,glass_encoder_finetune_nla_m_vdqk2_z8_bg
_K2_JOINT_NLE_K = 10


def _k2_joint_nle_a(r):
    pre = _nle_pretrain_bgp(_VDQ_GLASS_K2, r, cosmo_param_names=list(_K2_JOINT_THETA),
                            preset_overrides=_BG_BOXES_K2)
    pre["whiten_embeddings"] = {"k": _K2_JOINT_NLE_K}
    pre["embedding_cache_name"] = f"vdqk2_joint_r{r}"
    pre["reuse_embedding_cache"] = True
    return pre


kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_joint_k10_r0"] = _k2_joint_nle_a(0)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_joint_k10_r1"] = _k2_joint_nle_a(1)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_joint_k10_r2"] = _k2_joint_nle_a(2)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_joint_k10_r3"] = _k2_joint_nle_a(3)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_joint_k10_r4"] = _k2_joint_nle_a(4)

for _r in _NLE_REPEATS:
    VDQ_VARIATE_NLE_SOURCES[f"glass_nle_pretrain_nla_m_vdqk2_joint_k10_r{_r}"] = _K2_JOINT_SOURCES
assert set(VDQ_VARIATE_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)


# === PV-G nla_z r3 Stage A re-run on the RE-ROLLED encoder (user 2026-10-13) ======================
# The nla_z encoder r3 (best -7.166 vs -7.67..-7.86 for r0-r4) was re-rolled at the SAME index (jobs
# 1386148/1386150 write into the same pretrain_ncosmoNone_3/ folder; get_best_checkpoint takes the folder
# min, so the re-roll is picked automatically). The existing `..._k6_r3` Stage A was fit on the OLD
# encoder, and a same-name rerun is unsafe: fit_and_persist_whitener reuses an existing whitener.pt
# (fit-once) and old/new flow ckpts would mix under best-over-dir. Hence a fresh name, same r=3 split.
# `glass_nle_pretrain_nla_z_vdq_z8_k6_r3` is superseded — never source a Stage B from it.
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_z_vdq_z8_k6_r3b"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA_Z, 3, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
VDQ_VARIATE_NLE_SOURCES["glass_nle_pretrain_nla_z_vdq_z8_k6_r3b"] = "glass_encoder_finetune_nla_z_vdq_z8"


# === PV-G Gower VDQ variate NLE Stage B (user 2026-10-13: "pickup all the NLE variate finetuning") ==
# Recipe = the BGP `_hf` variate Stage B (config/kids_legacy_bgp.py `_HF_CHAINS` loop): ens9, guard-c
# 50, 150 epochs, all trainval cosmologies 80/20, fixed 100-id test lock; whiten k=6 to match the k6
# Stage A; one-wave sampling (63 workers, MUST launch with `embed --cpu --partition CORES64 --ncpu 64`).
# Stores: SIMS-baked gower_vdq_{nla,nla_z} (the 200-id fixed set). nla_z r3 is sourced from `_k6_r3b`.
# Launch: embed --cpu --partition CORES64 --ncpu 64 --mem-gb 128 --wall_h 48 --skip-smoke
#         --target <row> --sources glass_encoder_finetune_<v>_vdq_z8
_VARIATE_NLE_B_SAMPLING_JOBS = 63


def _variate_nle_b_vdq(stage_a, gower_data, r, cosmo_param_names, preset_overrides):
    b = _nle_finetune(stage_a, ensemble_repeats=9, whiten_k=_VARIATE_NLE_K6_VDQ, warmstart_max_gap_nats=50.0,
                      gower_data=gower_data, gower_eb=None,
                      cosmo_param_names=list(cosmo_param_names),
                      preset_overrides=preset_overrides)
    b["max_trainval_cosmos"] = None
    b["train_frac"] = 0.8
    b["val_frac"] = 0.2
    b["test_frac"] = 0.0
    b["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
    b["epochs"] = 150
    b["project"] = _BGP_NLE_PROJECT
    b["sampling_num_jobs"] = _VARIATE_NLE_B_SAMPLING_JOBS
    return _nle_bake_repeat(b, r)


kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdq_z8_k6_r0_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdq_z8_k6_r0", _VDQ_GOWER_NLA, 0, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdq_z8_k6_r1_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdq_z8_k6_r1", _VDQ_GOWER_NLA, 1, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdq_z8_k6_r2_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdq_z8_k6_r2", _VDQ_GOWER_NLA, 2, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdq_z8_k6_r3_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdq_z8_k6_r3", _VDQ_GOWER_NLA, 3, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdq_z8_k6_r4_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdq_z8_k6_r4", _VDQ_GOWER_NLA, 4, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_z_vdq_z8_k6_r0_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_z_vdq_z8_k6_r0", _VDQ_GOWER_NLA_Z, 0, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_z_vdq_z8_k6_r1_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_z_vdq_z8_k6_r1", _VDQ_GOWER_NLA_Z, 1, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_z_vdq_z8_k6_r2_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_z_vdq_z8_k6_r2", _VDQ_GOWER_NLA_Z, 2, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_z_vdq_z8_k6_r3b_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_z_vdq_z8_k6_r3b", _VDQ_GOWER_NLA_Z, 3, _COSMO_9_NLAZ, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_z_vdq_z8_k6_r4_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_z_vdq_z8_k6_r4", _VDQ_GOWER_NLA_Z, 4, _COSMO_9_NLAZ, _A_IA_NLA_BOX)

# ⭐ guard-c 50 -> 100 for **nla_z r3b ONLY** (user decision 2026-10-13 18:05Z, as the 2-pt r3 row above).
# Stage-B job 1386483 died on member 0: ep0 gap 94.354 nats (Gower val NLL 94.302 vs Stage-A best -0.052);
# r0/r1/r2/r4 first members sat at 24.5-31.8. Resolution verified correct (encoder pretrain_ncosmoNone_3
# epoch=32 -7.9193, Stage A `_k6_r3b` epoch=126, whitener from the `_k6_r3b` dir), so this is the
# re-rolled r3 encoder's larger GLASS->Gower embedding shift, not a broken warm start (a scratch init
# reads thousands of nats, e890aec). The other repeats keep 50.
kids_legacy_vdq_experiments["gower_nle_finetune_nla_z_vdq_z8_k6_r3b_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 100.0

# ⭐ guard-c raised again on the three tripping rows (user decision 2026-10-14 ~07:10Z: "raise the guard on
# repeats that failed"); every other variate row keeps 50. Observed ep0 gaps (Gower val NLL - Stage-A best):
#   nla_z r3b: 94.4 (m0, 1st attempt), 77.8 (m0) / 111.3 (m2, job 1386492 died at member 2 under the 100 guard)
#   nla   r4 : 75.5 (m0, job 1386497)          nla r3: 1293.9 (m0, job 1386496)
# siblings 24.5-42.5. Resolution verified correct for all three (own encoder repeat, own Stage-A ckpt + whitener).
# Diagnosis (local, Stage-B ens0 emb caches): a heavy tail of strongly negative-a_ia Gower mocks mapped off the GLASS
# manifold along the low-variance whitened components 4-6 (frac |u|^2>100: r0 ~1 %, r4 10-14 %, r3 23-28 %); the
# ep0-mean guard is dominated by them - not a broken warm start. New values = >=2.3x the worst observed gap.
kids_legacy_vdq_experiments["gower_nle_finetune_nla_z_vdq_z8_k6_r3b_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 300.0
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdq_z8_k6_r4_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 300.0
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdq_z8_k6_r3_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 3000.0

for _r in _NLE_REPEATS:
    _sfx = "3b" if _r == 3 else str(_r)
    VDQ_NLE_SOURCES[f"gower_nle_finetune_nla_vdq_z8_k6_r{_r}_ens9_e150"] = "glass_encoder_finetune_nla_vdq_z8"
    VDQ_NLE_SOURCES[f"gower_nle_finetune_nla_z_vdq_z8_k6_r{_sfx}_ens9_e150"] = "glass_encoder_finetune_nla_z_vdq_z8"
assert set(VDQ_NLE_SOURCES) <= set(kids_legacy_vdq_experiments), \
    sorted(set(VDQ_NLE_SOURCES) - set(kids_legacy_vdq_experiments))
assert set(VDQ_VARIATE_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)


# === PV-G kappa=2 + NLA ARM (user 2026-10-14: new variate row, k=2 with the NLA IA model, NOT NLA-M) ==
# Mirrors the NLA-M kappa=2 production chain above, with the NLA theta/box swapped in:
#   _cosmo : the 8 NLA params (`_COSMO_8_NLA`, a_ia box `_A_IA_NLA_BOX`), b_g marginalised over the broad
#            kappa=2 prior. FRESH flow head: the NLA-M foundation's 9-param head (a_ia + b_ia, NLA-M a_ia
#            box) does not match, exactly as for `glass_encoder_finetune_nla_vdq_z8`.
#   _bg    : the 6 b_g_bin params ONLY (kappa=2 boxes `_BG_BOXES_K2`), fresh flow head.
#   joint NLE Stage A: 14-param theta (8 NLA + b_g_bin1..6), sources cosmo,bg, whiten k=10 (the value the
#            NLA-M kappa=2 joint-summary test validated; not re-tested on this arm). OWN raw-embedding cache
#            `vdqk2_nla_joint_r<r>` — never the NLA-M `vdqk2_joint_r<r>` (different encoders/store).
# Store: SIMS GLASS twin glass_mocks_nla_vdq_bgpk2 (datasets_checklist §K2N, 4000 x 4) -> bake below.
# Gower Stage B: NOT written yet (as for the NLA-M kappa=2 arm) — needs the Gower bake
# gower_vdqk2_nla_f16_sc8a1_* (§K2N-W) and a joint (k=10, two-source) Stage-B recipe.
# Launch (encoders): train --gpu a100|l40s glass_encoder_finetune_nla_vdqk2_z8_{cosmo,bg}
# Launch (Stage A):  embed --gpu v100 --mem-gb 32 --target glass_nle_pretrain_nla_vdqk2_joint_k10_r<r>
#         --sources glass_encoder_finetune_nla_vdqk2_z8_cosmo,glass_encoder_finetune_nla_vdqk2_z8_bg
_VDQ_GLASS_K2_NLA = f"{_GPU5}/glass_vdqk2_nla_f16_sc8a1_{_EB}/output_*.h5"
_K2_NLA_JOINT_SOURCES = "glass_encoder_finetune_nla_vdqk2_z8_cosmo,glass_encoder_finetune_nla_vdqk2_z8_bg"
_K2_NLA_JOINT_THETA = list(_COSMO_8_NLA) + list(_BG_THETA)
_K2_NLA_JOINT_BOXES = {**_A_IA_NLA_BOX, **_BG_BOXES_K2}
assert len(_K2_NLA_JOINT_THETA) == 14, _K2_NLA_JOINT_THETA

kids_legacy_vdq_experiments["glass_encoder_finetune_nla_vdqk2_z8_cosmo"] = _assert_final_summary_dim(
    _encoder_finetune_vdq(_VDQ_GLASS_K2_NLA, _COSMO_8_NLA, preset_overrides=_A_IA_NLA_BOX), 8,
    "glass_encoder_finetune_nla_vdqk2_z8_cosmo")
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_vdqk2_z8_bg"] = _assert_final_summary_dim(
    _encoder_finetune_vdq(_VDQ_GLASS_K2_NLA, _BG_THETA, preset_overrides=_BG_BOXES_K2), 8,
    "glass_encoder_finetune_nla_vdqk2_z8_bg")


def _k2_nla_joint_nle_a(r):
    pre = _nle_pretrain_bgp(_VDQ_GLASS_K2_NLA, r, cosmo_param_names=list(_K2_NLA_JOINT_THETA),
                            preset_overrides=_K2_NLA_JOINT_BOXES)
    pre["whiten_embeddings"] = {"k": _K2_JOINT_NLE_K}
    pre["embedding_cache_name"] = f"vdqk2_nla_joint_r{r}"
    pre["reuse_embedding_cache"] = True
    return pre


# Literal keys (run_remote.py:known_experiment scans for them).
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdqk2_joint_k10_r0"] = _k2_nla_joint_nle_a(0)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdqk2_joint_k10_r1"] = _k2_nla_joint_nle_a(1)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdqk2_joint_k10_r2"] = _k2_nla_joint_nle_a(2)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdqk2_joint_k10_r3"] = _k2_nla_joint_nle_a(3)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdqk2_joint_k10_r4"] = _k2_nla_joint_nle_a(4)

for _r in _NLE_REPEATS:
    VDQ_VARIATE_NLE_SOURCES[f"glass_nle_pretrain_nla_vdqk2_joint_k10_r{_r}"] = _K2_NLA_JOINT_SOURCES
assert set(VDQ_VARIATE_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)


# === PV-G nla r3/r4 ENCODER REROLL under a NEW NAME (user 2026-10-14 ~14:30Z) =======================
# nla r3 / r4 Stage B came out calibrated but out of line: FoM(Om,s8) 5.13 / 5.58 vs r0-2 6.05-7.04, S8 w68
# 0.082 / 0.070 vs 0.058-0.068 (and they needed guard-c 3000 / 300). Both encoders were GLASS slow starters
# (best -6.739 / -6.757 vs -6.83..-6.99). User: retrain the compressors as a reroll with a NEW NAME so no
# frame / scaler / whitener / checkpoint dir is shared with the old chain. `_rr` = identical recipe to
# `glass_encoder_finetune_nla_vdq_z8` at the SAME repeat index (same split seed, same foundation repeat warm
# start; only the fresh flow-head init + data order differ). Guard-c stays at the sibling value 50 on the
# `_rr` Stage B: it is the in-family gate (a trip = not fixed -> ask the user, do not raise).
# Launch: train --exp glass_encoder_finetune_nla_vdq_z8_rr --repeat-indices <3|4>;
#         embed --target glass_nle_pretrain_nla_vdq_z8_rr_k6_r<r> --sources glass_encoder_finetune_nla_vdq_z8_rr;
#         embed --cpu --partition CORES64 --ncpu 64 --mem-gb 128 --wall_h 48 --skip-smoke
#               --target gower_nle_finetune_nla_vdq_z8_rr_k6_r<r>_ens9_e150 --sources glass_encoder_finetune_nla_vdq_z8_rr
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_vdq_z8_rr"] = _assert_final_summary_dim(
    _encoder_finetune_vdq(_VDQ_GLASS_NLA, _COSMO_8_NLA, preset_overrides=_A_IA_NLA_BOX), 8,
    "glass_encoder_finetune_nla_vdq_z8_rr")
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_rr_k6_r3"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA, 3, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_rr_k6_r4"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA, 4, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdq_z8_rr_k6_r3_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdq_z8_rr_k6_r3", _VDQ_GOWER_NLA, 3, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdq_z8_rr_k6_r4_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdq_z8_rr_k6_r4", _VDQ_GOWER_NLA, 4, _COSMO_8_NLA, _A_IA_NLA_BOX)
# r5 (user 2026-10-14 ~21:00Z): the r3 reroll did NOT move (three inits at split_seed 45: -6.739/-6.742/-6.710 vs r4's
# reroll -6.757 -> -6.867 at its own split) -> r3's ceiling is the SPLIT. Replace r3 by a NEW index 5 (new GLASS + Gower
# split, foundation repeat-5 warm start); the nla pool becomes r0, r1, r2, r4(rr), r5(rr).
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdq_z8_rr_k6_r5"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_NLA, 5, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdq_z8_rr_k6_r5_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdq_z8_rr_k6_r5", _VDQ_GOWER_NLA, 5, _COSMO_8_NLA, _A_IA_NLA_BOX)
# ⭐ guard-c 50 -> 150 for **rr r5 ONLY** (user decision 2026-10-15 ~15:00Z: "raise the guard"). Stage-B job 1387242 died on
# member 0: ep0 gap 52.66 nats (Gower val NLL 51.868 vs Stage-A best -0.791); siblings r0 29.9 / r1 41-42.5 / r2 30.8 /
# rr r4 passed 50. Resolution verified (encoder _rr r5 ep32 -6.7496, Stage A _rr_k6_r5 ep139 -0.7914, own whitener).
# 150 = >=2.3x the observed gap (the convention used for the other raised rows); rr r4 keeps 50.
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdq_z8_rr_k6_r5_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 150.0
for _r in (3, 4, 5):
    VDQ_VARIATE_NLE_SOURCES[f"glass_nle_pretrain_nla_vdq_z8_rr_k6_r{_r}"] = "glass_encoder_finetune_nla_vdq_z8_rr"
    VDQ_NLE_SOURCES[f"gower_nle_finetune_nla_vdq_z8_rr_k6_r{_r}_ens9_e150"] = "glass_encoder_finetune_nla_vdq_z8_rr"
assert set(VDQ_VARIATE_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)
assert set(VDQ_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)


# === PV-G kappa=2 b_g encoder CONDITIONED ON COSMOLOGY (user 2026-10-14) ===========================
# The b_g-only encoders (`..._vdqk2_z8_bg`) constrain ~ONE bin per repeat (a random one, posterior
# std ~0.4x the truth spread) and leave the rest at the prior; the learnt bin is 10-30 % overconfident.
# Hypothesis (user): b_g leaves a cosmology-dependent imprint, and a b_g-only head has to learn the
# cosmology implicitly with no supervision. Test: give the b_g density head the cosmology as an extra
# conditioning input (`side_input`, src/ml/data/side_inputs.py), so the CNN can learn features that
# only carry b_g information GIVEN cosmology. Same recipe as `_bg` otherwise (VDQ foundation warm
# start, fresh 6-param head, fixed LR, 100 ep, z8 summary; the flow conditions on [z_bg(8), side]).
#   _bg_oracle     : side = the TRUE 9 NLA-M cosmology params (preset-scaled) -> the CEILING. If b_g
#                    stays near the prior here, per-bin b_g is not in the maps at this noise level.
#   _bg_condcosmo  : side = the FROZEN kappa=2 cosmology encoder's z8 summary of the same (augmented)
#                    batch, same repeat (`glass_encoder_finetune_nla_m_vdqk2_z8_cosmo`) -> the
#                    practical version (a real observation has no true theta).
# ⚠️ Their embedding nets emit [z_bg, side]: never use these rows as NLE / embeddings SOURCES.
# Evaluate with eval.py --mode standard (b_g-only theta prior = analytic truncated Gaussians).
# Launch: train --exp <row> --gpu a100 --repeat-indices 0 --wall_h 6 --mem-gb 48
def _bg_side_input_row(side_input):
    c = _encoder_finetune_vdq(_VDQ_GLASS_K2, _BG_THETA, preset_overrides=_BG_BOXES_K2)
    c["side_input"] = side_input
    return c


kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vdqk2_z8_bg_oracle"] = _assert_final_summary_dim(
    _bg_side_input_row({"type": "true_theta", "params": list(_COSMO_9_NLAM)}), 8,
    "glass_encoder_finetune_nla_m_vdqk2_z8_bg_oracle")
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vdqk2_z8_bg_condcosmo"] = _assert_final_summary_dim(
    _bg_side_input_row({"type": "frozen_encoder",
                        "ckpt_path": f"{_CKPT}/glass_encoder_finetune_nla_m_vdqk2_z8_cosmo/"}), 8,
    "glass_encoder_finetune_nla_m_vdqk2_z8_bg_condcosmo")


# === kappa=1 b_g encoder conditioned on cosmology, FOUNDATION store (user 2026-10-14 ~21:35Z, "one last experiment") ===
# The kappa=2 conditioned encoders (above) still constrain ONE bin. Retry on the ~6x larger kappa=1 foundation store
# (glass_vdq_nla_m, 100k mocks, b_g ~ kappa=1 prior) with the FOUNDATION ITSELF as the frozen cosmology side encoder (it is
# the 9-param cosmology encoder of that store; warm start = foundation too), then transfer to kappa=2.
#   _bg_k1            : kappa=1 b_g-only baseline (no side input) -- the like-for-like reference on this store.
#   _bg_condcosmo_k1  : + side_input = frozen foundation z8 summary.
#   (k2) _bg_condcosmo_fromk1 : the kappa=2 condcosmo row warm-started from `_bg_condcosmo_k1` (its whole embedding_net)
#                       instead of the foundation; side encoder = the kappa=2 cosmo encoder as in `_bg_condcosmo`, loaded
#                       AFTER the warm start (overwrites the foundation copy that comes along in the k1 checkpoint).
# epochs 30 (not 100): ~800 steps/epoch on 100k vs 128 on the 16k kappa=2 store (the kappa=2 b_g encoders peaked at ep17-22,
# i.e. ~2.5k steps); 30 ep = ~24k steps, inside the 6 h GPU wall (~8-9 min/epoch on a100 incl. the frozen forward).
from config.kids_legacy_bgp import _BG_BOXES  # noqa: E402  (kappa=1 boxes)
_BG_K1_EPOCHS = 30


def _bg_k1_row(side_input=None):
    c = _encoder_finetune_vdq(_VDQ_GLASS, _BG_THETA, preset_overrides=_BG_BOXES)
    c["epochs"] = _BG_K1_EPOCHS
    if side_input is not None:
        c["side_input"] = side_input
    return c


kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vdq_z8_bg_k1"] = _assert_final_summary_dim(
    _bg_k1_row(), 8, "glass_encoder_finetune_nla_m_vdq_z8_bg_k1")
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vdq_z8_bg_condcosmo_k1"] = _assert_final_summary_dim(
    _bg_k1_row({"type": "frozen_encoder", "ckpt_path": _VDQ_HYB_CKPT}), 8,
    "glass_encoder_finetune_nla_m_vdq_z8_bg_condcosmo_k1")
_c = _bg_side_input_row({"type": "frozen_encoder",
                         "ckpt_path": f"{_CKPT}/glass_encoder_finetune_nla_m_vdqk2_z8_cosmo/"})
_c["pretrained_embedding_ckpt_path"] = f"{_CKPT}/glass_encoder_finetune_nla_m_vdq_z8_bg_condcosmo_k1/"
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vdqk2_z8_bg_condcosmo_fromk1"] = _assert_final_summary_dim(
    _c, 8, "glass_encoder_finetune_nla_m_vdqk2_z8_bg_condcosmo_fromk1")


# === kappa=2 b_g conditioning on the sc8 (NO a1 noise-norm) store (user 2026-10-14 ~22:10Z) ======================
# Hypothesis: the per-mock a1 rescale (E_sc8 / noise_std_sc8) divides out exactly the noise-AMPLITUDE channel that b_g
# modulates. Re-bake of the SAME kappa=2 raw store without it (prebake --noise-norm none, job 1387199). The frozen cosmology
# side encoder must be trained on the SAME product (the sc8a1 one would see out-of-distribution per-mock amplitudes), so:
#   _cosmo_sc8          : the kappa=2 cosmology encoder on sc8 (recipe of `_cosmo`: 9p, foundation flow preloaded)
#   _bg_sc8             : b_g-only baseline on sc8
#   _bg_condcosmo_sc8   : b_g + frozen `_cosmo_sc8` summary (train AFTER `_cosmo_sc8`)
# All warm-start from the (sc8a1-trained) VDQ foundation -- the only foundation there is; the whole encoder finetunes.
_VDQ_GLASS_K2_SC8 = f"{_GPU5}/glass_vdqk2_nla_m_f16_sc8_{_EB}/output_*.h5"
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vdqk2_z8_cosmo_sc8"] = _assert_final_summary_dim(
    _encoder_finetune_vdq(_VDQ_GLASS_K2_SC8, _COSMO_9_NLAM, load_flow=True), 8,
    "glass_encoder_finetune_nla_m_vdqk2_z8_cosmo_sc8")
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vdqk2_z8_bg_sc8"] = _assert_final_summary_dim(
    _encoder_finetune_vdq(_VDQ_GLASS_K2_SC8, _BG_THETA, preset_overrides=_BG_BOXES_K2), 8,
    "glass_encoder_finetune_nla_m_vdqk2_z8_bg_sc8")
_c = _encoder_finetune_vdq(_VDQ_GLASS_K2_SC8, _BG_THETA, preset_overrides=_BG_BOXES_K2)
_c["side_input"] = {"type": "frozen_encoder", "ckpt_path": f"{_CKPT}/glass_encoder_finetune_nla_m_vdqk2_z8_cosmo_sc8/"}
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vdqk2_z8_bg_condcosmo_sc8"] = _assert_final_summary_dim(
    _c, 8, "glass_encoder_finetune_nla_m_vdqk2_z8_bg_condcosmo_sc8")


# === kappa=1 b_g encoder on the sc8 (NO a1 noise-norm) FOUNDATION store (user 2026-10-15) =======================
# Key test: does dropping the per-mock a1 rescale let a b_g-only encoder constrain MORE THAN ONE bin (at least bins 1-3)
# on the ~6x larger kappa=1 store? Like-for-like with `_bg_k1` (30 ep, kappa=1 boxes, foundation warm start) except the
# store: no-a1 re-bake of the raw gpu4 glass_mocks_nla_m_vdq_bgp (prebake --eb-variant sc8_<EB> --noise-norm none, f16).
# No conditioned twin: its frozen cosmology side encoder would itself need a no-a1 kappa=1 retrain.
_VDQ_GLASS_SC8 = f"{_GPU5}/glass_vdq_nla_m_f16_sc8_{_EB}/output_*.h5"
_c = _bg_k1_row()
_c["data_patterns"] = [_VDQ_GLASS_SC8] if isinstance(_c["data_patterns"], (list, tuple)) else _VDQ_GLASS_SC8
kids_legacy_vdq_experiments["glass_encoder_finetune_nla_m_vdq_z8_bg_k1_sc8"] = _assert_final_summary_dim(
    _c, 8, "glass_encoder_finetune_nla_m_vdq_z8_bg_k1_sc8")


# === FINAL kappa=2 NLE chains: COSMOLOGY-ONLY theta, b_g MARGINALISED (user 2026-10-16) =====================
# The b_g study (VD-final-train-models AMENDMENTS 11-13, artifacts/k2_joint_test/BG_BINS_REPORT.md) showed no encoder
# constrains more than ONE b_g bin consistently, so the kappa=2 arms drop b_g from theta altogether: the NLE learns
# p(t | theta_cosmo) on the kappa=2 stores, so b_g is marginalised over its broad kappa=2 prior by the simulations.
# This SUPERSEDES the joint (cosmo+b_g, k=10) Stage A `glass_nle_pretrain_nla_m_vdqk2_joint_k10_r*` and the never-run
# kappa=2+NLA `glass_nle_pretrain_nla_vdqk2_joint_k10_r*` / `glass_encoder_finetune_nla_vdqk2_z8_bg` (record only).
# Recipe = the VDQ variate chain (`_variate_nle_a_vdq_k6` / `_variate_nle_b_vdq`): whiten k=6, Stage B ens9, 150 ep,
# guard-c 50, 80/20 over all trainval cosmologies, fixed 100-id test lock, 63-worker one-wave sampling.
#   NLA-M kappa=2 : 9 params `_COSMO_9_NLAM`, source `glass_encoder_finetune_nla_m_vdqk2_z8_cosmo` (trained r0-r4).
#   NLA   kappa=2 : 8 params `_COSMO_8_NLA` + `_A_IA_NLA_BOX`, source `glass_encoder_finetune_nla_vdqk2_z8_cosmo`.
# Gower stores (SIMS bakes, 200-id fixed set): gower_vdqk2_nla_m_* and gower_vdqk2_nla_*.
# Launch (Stage A): embed --cpu --target glass_nle_pretrain_<arm>_z8_cosmo_k6_r<r> --sources <encoder row>
# Launch (Stage B): embed --cpu --partition CORES64 --ncpu 64 --mem-gb 128 --wall_h 48 --skip-smoke
#                   --target gower_nle_finetune_<arm>_z8_cosmo_k6_r<r>_ens9_e150 --sources <encoder row>
_VDQ_GOWER_K2 = f"{_GPU5}/gower_vdqk2_nla_m_f16_sc8a1_{_EB}/output_*.h5"
_VDQ_GOWER_K2_NLA = f"{_GPU5}/gower_vdqk2_nla_f16_sc8a1_{_EB}/output_*.h5"
_K2_COSMO_ENC = "glass_encoder_finetune_nla_m_vdqk2_z8_cosmo"
_K2_NLA_COSMO_ENC = "glass_encoder_finetune_nla_vdqk2_z8_cosmo"

# Literal keys (run_remote.py:known_experiment scans for them).
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k6_r0"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_K2, 0, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k6_r1"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_K2, 1, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k6_r2"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_K2, 2, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k6_r3"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_K2, 3, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k6_r4"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_K2, 4, _COSMO_9_NLAM)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r0"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_K2_NLA, 0, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r1"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_K2_NLA, 1, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r2"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_K2_NLA, 2, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r3"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_K2_NLA, 3, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r4"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_K2_NLA, 4, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdqk2_z8_cosmo_k6_r0_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k6_r0", _VDQ_GOWER_K2, 0, _COSMO_9_NLAM, None)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdqk2_z8_cosmo_k6_r1_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k6_r1", _VDQ_GOWER_K2, 1, _COSMO_9_NLAM, None)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdqk2_z8_cosmo_k6_r2_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k6_r2", _VDQ_GOWER_K2, 2, _COSMO_9_NLAM, None)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdqk2_z8_cosmo_k6_r3_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k6_r3", _VDQ_GOWER_K2, 3, _COSMO_9_NLAM, None)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_m_vdqk2_z8_cosmo_k6_r4_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k6_r4", _VDQ_GOWER_K2, 4, _COSMO_9_NLAM, None)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdqk2_z8_cosmo_k6_r0_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r0", _VDQ_GOWER_K2_NLA, 0, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdqk2_z8_cosmo_k6_r1_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r1", _VDQ_GOWER_K2_NLA, 1, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdqk2_z8_cosmo_k6_r2_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r2", _VDQ_GOWER_K2_NLA, 2, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdqk2_z8_cosmo_k6_r3_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r3", _VDQ_GOWER_K2_NLA, 3, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdqk2_z8_cosmo_k6_r4_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r4", _VDQ_GOWER_K2_NLA, 4, _COSMO_8_NLA, _A_IA_NLA_BOX)
# ⭐ guard-c 50 -> 800 for **K2N r3 ONLY** (user 2026-10-17: "raise the guard if a repeat val loss looks sensible, if not,
# reroll"). Stage-B job 1388428 died on member 0: ep0 gap 337.785 nats (Gower val NLL 337.999 vs Stage-A best 0.214);
# siblings r0 31.8 / r1 27.6 / r2 33.8 / r4 36.7. Resolution verified (encoder pretrain_ncosmoNone_3 ep52 -6.6211, own
# Stage-A r3 whitener) = the split_seed r3 pattern (nla r3 1294, nla_z r3b 94). 800 = >=2.3x the gap; trained member vals
# are then checked against r0/1/2/4 (reroll as a new index if out of family).
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdqk2_z8_cosmo_k6_r3_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 800.0
# K2N r5 REPLACES r3 (2026-10-17, per the user's "if not, reroll"): under guard 800 r3 trained but is out of family on the
# SHARED 100-id Gower test lock: per-member test NLL 3.9-7.3 vs r1 1.85-2.26 / r2 0.15-1.80 / r4 0.18-0.77 (r3 encoder is the
# weakest, -6.621). As for nla r3 (same-index `_rr` did not move it: split ceiling), the reroll is a NEW index 5 (new GLASS +
# Gower split, foundation repeat-5 warm start). Encoder: `train --exp glass_encoder_finetune_nla_vdqk2_z8_cosmo
# --repeat-indices 5` (same experiment dir, pretrain_ncosmoNone_5). Stage B guard back at the in-family 50.
# The k2_nla pool becomes r0, r1, r2, r4, r5.
kids_legacy_vdq_experiments["glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r5"] = _variate_nle_a_vdq_k6(_VDQ_GLASS_K2_NLA, 5, _COSMO_8_NLA, _A_IA_NLA_BOX)
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdqk2_z8_cosmo_k6_r5_ens9_e150"] = _variate_nle_b_vdq("glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r5", _VDQ_GOWER_K2_NLA, 5, _COSMO_8_NLA, _A_IA_NLA_BOX)
VDQ_VARIATE_NLE_SOURCES["glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r5"] = _K2_NLA_COSMO_ENC
VDQ_NLE_SOURCES["gower_nle_finetune_nla_vdqk2_z8_cosmo_k6_r5_ens9_e150"] = _K2_NLA_COSMO_ENC
# guard-c 50 -> 150 for **K2N r5 ONLY** (user rule 2026-10-17 "raise the guard if a repeat val loss looks sensible"): Stage-B job
# 1389038 died on member 0: ep0 gap 60.250 nats (Gower val NLL 59.292 vs Stage-A best -0.958); siblings 27.6-36.7; resolution verified
# (encoder pretrain_ncosmoNone_5 ep46 -6.5297, own Stage-A r5 whitener). Same pattern as nla rr r5 (52.66 -> 150, came out in family).
# 150 = >=2.3x the gap; judged afterwards on per-member test NLL on the shared 100-id lock vs r1/r2/r4.
kids_legacy_vdq_experiments["gower_nle_finetune_nla_vdqk2_z8_cosmo_k6_r5_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 150.0

for _r in _NLE_REPEATS:
    VDQ_VARIATE_NLE_SOURCES[f"glass_nle_pretrain_nla_m_vdqk2_z8_cosmo_k6_r{_r}"] = _K2_COSMO_ENC
    VDQ_VARIATE_NLE_SOURCES[f"glass_nle_pretrain_nla_vdqk2_z8_cosmo_k6_r{_r}"] = _K2_NLA_COSMO_ENC
    VDQ_NLE_SOURCES[f"gower_nle_finetune_nla_m_vdqk2_z8_cosmo_k6_r{_r}_ens9_e150"] = _K2_COSMO_ENC
    VDQ_NLE_SOURCES[f"gower_nle_finetune_nla_vdqk2_z8_cosmo_k6_r{_r}_ens9_e150"] = _K2_NLA_COSMO_ENC
assert set(VDQ_VARIATE_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)
assert set(VDQ_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)


# === 2-pt (bandpower) NLE chain for the NLA variate (user 2026-10-17, unblinding-investigation-2 T3b) ===
# The real field NLA arm prefers A_IA ~ 0 (S8 0.786) while the NLA-M field arm is prior-locked at a_ia ~5.7
# (S8 0.844); no free-amplitude 2-pt arm existed. Mirror of the P3b band chain (`kids_legacy_band_nla_m_vdq`
# -> `glass_nle_pretrain_band_nla_m_vdq_k8_r{r}` -> `gower_nle_finetune_band_nla_m_vdq_k8_r{r}_ens9_e150`)
# moved onto the NLA stores with the NLA theta (`_COSMO_8_NLA`, a_ia box `_A_IA_NLA_BOX` = U[-6, 6]), exactly as
# the field NLA variate rows pass them. Kept: band arch/epochs, whiten k=8 (pure-whiten), ens9, 150 epochs,
# guard-c 50, the NLE project.
# ⚠️ FORCED DEVIATION (Stage B split): the Gower NLA store `gower_vdq_nla` was simulated on the 200-id
# `--gower-sim-set fixed_test` set ONLY, so the band chain's `max_trainval_cosmos=[300]` + 200-id lock would put
# every sim in test. Stage B therefore uses the field NLA Stage B split (`_variate_nle_b_vdq`): all trainval
# cosmologies 80/20 + the 100-id lock `_GOWER_TEST_IDS_100` => match string `ncosmoNone_{r}` (not ncosmo300).
# That also makes "2-pt vs field within the NLA variate" a single-variable comparison.
# Stage I trains a NEW band MLP on the NLA GLASS store (theta must carry the free a_ia); Stage A reads that
# band's OWN training store so its split reproduces the band's. Launch:
#   train --exp kids_legacy_band_nla_vdq --gpu v100 --mem-gb 32 --wall_h 6 --repeat-indices 0,1,2
#   embed --gpu v100 --mem-gb 32 --target glass_nle_pretrain_band_nla_vdq_k8_r<r> --sources kids_legacy_band_nla_vdq
#   embed --cpu --partition CORES40 --target gower_nle_finetune_band_nla_vdq_k8_r<r>_ens9_e150 --sources kids_legacy_band_nla_vdq
_BAND_NLA_REPEATS_VDQ = (0, 1, 2)
_BAND_NLA_SCALER = {
    "data": {"type": "standard", "keys": None},
    "cosmo": {"type": "preset", "preset_overrides": dict(_A_IA_NLA_BOX)},
}


def _band_nla_vdq():
    c = _band_bgp(_VDQ_GLASS_NLA)
    c["cosmo_param_names"] = list(_COSMO_8_NLA)
    c["scaler_options"] = {k: dict(v) for k, v in _BAND_NLA_SCALER.items()}
    c["repeat_indices"] = list(_BAND_NLA_REPEATS_VDQ)
    return c


def _band_nla_nle_a_vdq(r):
    pre = _nle_pretrain_bgp(_VDQ_GLASS_NLA, r, cosmo_param_names=list(_COSMO_8_NLA),
                            preset_overrides=_A_IA_NLA_BOX)
    pre["whiten_embeddings"] = {"k": _BAND_NLE_K_VDQ}
    return pre


def _band_nla_nle_b_vdq(r):
    ft = _nle_finetune(f"glass_nle_pretrain_band_nla_vdq_k{_BAND_NLE_K_VDQ}_r{r}", ensemble_repeats=9,
                       whiten_k=_BAND_NLE_K_VDQ, warmstart_max_gap_nats=50.0,
                       gower_data=_VDQ_GOWER_NLA, gower_eb=None,
                       cosmo_param_names=list(_COSMO_8_NLA), preset_overrides=_A_IA_NLA_BOX)
    ft["max_trainval_cosmos"] = None             # forced deviation: the 200-id store (see above)
    ft["train_frac"] = 0.8
    ft["val_frac"] = 0.2
    ft["test_frac"] = 0.0
    ft["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
    ft["epochs"] = 150
    ft["project"] = _BGP_NLE_PROJECT
    return _nle_bake_repeat(ft, r)


assert _BAND_NLE_K_VDQ == 8, _BAND_NLE_K_VDQ
kids_legacy_vdq_experiments["kids_legacy_band_nla_vdq"] = _band_nla_vdq()
# Literal keys (run_remote.py:known_experiment scans for them).
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_vdq_k8_r0"] = _band_nla_nle_a_vdq(0)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_vdq_k8_r1"] = _band_nla_nle_a_vdq(1)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_vdq_k8_r2"] = _band_nla_nle_a_vdq(2)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_vdq_k8_r0_ens9_e150"] = _band_nla_nle_b_vdq(0)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_vdq_k8_r1_ens9_e150"] = _band_nla_nle_b_vdq(1)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_vdq_k8_r2_ens9_e150"] = _band_nla_nle_b_vdq(2)
# guard-c 50 -> 150 for band_nla r0 and r2 ONLY (standing user rule 2026-10-17 "raise the guard if a repeat val loss looks
# sensible", max 150): Stage-B jobs 1389983 (r0) / 1389984 (r2) died at ep0 with gaps 70.437 nats (Gower val NLL 60.134 vs
# Stage-A best -10.303) / 52.804 (41.422 vs -11.382); r1 passed at 40.47. Resolution verified in both logs: own Stage-A
# whitener (glass_nle_pretrain_band_nla_vdq_k8_r{0,2}, fit on 12748 rows) + own Stage-A best flow (ep146 -10.3035 / ep143
# -11.3817). Judged afterwards on the shared 100-id-lock per-member test NLL vs r1 (-5.67..-8.02).
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_vdq_k8_r0_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 150.0
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_vdq_k8_r2_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 150.0
for _r in _BAND_NLA_REPEATS_VDQ:
    VDQ_NLE_SOURCES[f"glass_nle_pretrain_band_nla_vdq_k{_BAND_NLE_K_VDQ}_r{_r}"] = "kids_legacy_band_nla_vdq"
    VDQ_NLE_SOURCES[f"gower_nle_finetune_band_nla_vdq_k{_BAND_NLE_K_VDQ}_r{_r}_ens9_e150"] = "kids_legacy_band_nla_vdq"
assert set(VDQ_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)


# === 2-pt (bandpower) NLE chain for the NLA-z variate (user 2026-10-20, vdq-posterior-predictive PPC) ========
# Mirror of the `band_nla` chain above moved onto the NLA-z stores with the NLA-z theta (`_COSMO_9_NLAZ` = the 8
# NLA params + b_z; a_ia box `_A_IA_NLA_BOX`, b_z uses the global preset box) -- exactly as the field NLA-z rows
# (`glass_nle_pretrain_nla_z_vdq_z8_k6_r*` / `gower_nle_finetune_nla_z_vdq_z8_k6_r*_ens9_e150`) pass them. Kept:
# band arch/epochs, whiten k=8, ens9, 150 epochs, guard-c 50, the NLE project. Same FORCED DEVIATION as band_nla:
# the Gower NLA-z store `gower_vdq_nla_z` is the 200-id fixed set only => all trainval 80/20 + the 100-id lock,
# match string `ncosmoNone_{r}`. Launch:
#   train --exp kids_legacy_band_nla_z_vdq --gpu v100 --mem-gb 32 --wall_h 6 --repeat-indices 0,1,2
#   embed --gpu v100 --mem-gb 32 --target glass_nle_pretrain_band_nla_z_vdq_k8_r<r> --sources kids_legacy_band_nla_z_vdq
#   embed --cpu --partition CORES40 --target gower_nle_finetune_band_nla_z_vdq_k8_r<r>_ens9_e150 --sources kids_legacy_band_nla_z_vdq
def _band_nla_z_vdq():
    c = _band_bgp(_VDQ_GLASS_NLA_Z)
    c["cosmo_param_names"] = list(_COSMO_9_NLAZ)
    c["scaler_options"] = {k: dict(v) for k, v in _BAND_NLA_SCALER.items()}
    c["repeat_indices"] = list(_BAND_NLA_REPEATS_VDQ)
    return c


def _band_nla_z_nle_a_vdq(r):
    pre = _nle_pretrain_bgp(_VDQ_GLASS_NLA_Z, r, cosmo_param_names=list(_COSMO_9_NLAZ),
                            preset_overrides=_A_IA_NLA_BOX)
    pre["whiten_embeddings"] = {"k": _BAND_NLE_K_VDQ}
    return pre


def _band_nla_z_nle_b_vdq(r, pretrain_exp=None):
    pretrain_exp = pretrain_exp or f"glass_nle_pretrain_band_nla_z_vdq_k{_BAND_NLE_K_VDQ}_r{r}"
    ft = _nle_finetune(pretrain_exp, ensemble_repeats=9,
                       whiten_k=_BAND_NLE_K_VDQ, warmstart_max_gap_nats=50.0,
                       gower_data=_VDQ_GOWER_NLA_Z, gower_eb=None,
                       cosmo_param_names=list(_COSMO_9_NLAZ), preset_overrides=_A_IA_NLA_BOX)
    ft["max_trainval_cosmos"] = None             # forced deviation: the 200-id store (see above)
    ft["train_frac"] = 0.8
    ft["val_frac"] = 0.2
    ft["test_frac"] = 0.0
    ft["fixed_test_sim_ids"] = _GOWER_TEST_IDS_100
    ft["epochs"] = 150
    ft["project"] = _BGP_NLE_PROJECT
    return _nle_bake_repeat(ft, r)


kids_legacy_vdq_experiments["kids_legacy_band_nla_z_vdq"] = _band_nla_z_vdq()
# Literal keys (run_remote.py:known_experiment scans for them).
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_k8_r0"] = _band_nla_z_nle_a_vdq(0)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_k8_r1"] = _band_nla_z_nle_a_vdq(1)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_k8_r2"] = _band_nla_z_nle_a_vdq(2)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r0_ens9_e150"] = _band_nla_z_nle_b_vdq(0)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r1_ens9_e150"] = _band_nla_z_nle_b_vdq(1)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_ens9_e150"] = _band_nla_z_nle_b_vdq(2)
# guard-c 50 -> 150 for band_nla_z r0 ONLY (standing user rule "raise the guard if a repeat's val loss looks sensible", max 150):
# Stage-B job 1391289 died at ep0 with gap 50.552 nats (Gower val NLL 39.974 vs Stage-A best -10.579; band_nla r1 passed at 40.47).
# Resolution verified in the log: own Stage-A whitener (glass_nle_pretrain_band_nla_z_vdq_k8_r0, fit on 12739 rows) + own Stage-A
# best flow (ep144 -10.5787), 110 keys loaded. Judged afterwards on the shared 100-id-lock per-member test NLL vs band_nla (-5.67..-8.02).
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r0_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 150.0
# ... and for r1 ONLY: Stage-B job 1391294 died at ep0 with gap 95.586 nats (Gower val NLL 83.798 vs Stage-A best -11.788; higher than any
# band_nla repeat, max 70.4). Resolution verified: own whitener (..._band_nla_z_vdq_k8_r1, 12739 rows) + own best flow (ep149 -11.7883),
# 110 keys loaded. Retained only if its per-member test NLL on the 100-id lock is in family (judge afterwards; drop/replace if not).
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r1_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 150.0
# ... and for r2 ONLY: Stage-B job 1391381 died at ep0 with gap 54.075 nats (Gower val NLL 43.548 vs Stage-A best -10.527). Resolution
# verified: own whitener (..._band_nla_z_vdq_k8_r2, 12739 rows) + own best flow (ep144 -10.5271), 110 keys loaded.
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_ens9_e150"]["whiten_warmstart_max_gap_nats"] = 150.0
for _r in _BAND_NLA_REPEATS_VDQ:
    VDQ_NLE_SOURCES[f"glass_nle_pretrain_band_nla_z_vdq_k{_BAND_NLE_K_VDQ}_r{_r}"] = "kids_legacy_band_nla_z_vdq"
    VDQ_NLE_SOURCES[f"gower_nle_finetune_band_nla_z_vdq_k{_BAND_NLE_K_VDQ}_r{_r}_ens9_e150"] = "kids_legacy_band_nla_z_vdq"
assert set(VDQ_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)


# === band_nla_z 5-repeat retrain (user 2026-10-21, task eval-and-viz/band-nla-z-5rep-retrain) ===================
# Target: arm band_nla_z = repeats 0-4, EVERY repeat matching r0 (FoM(Om,s8) >= 3.4, S8 w68 <= 0.08, TARP <= ~0.06).
# r0 passes and is kept. Stage-I eval (job 1391458, GLASS test FoM(Om,s8)): band_nla_z 3.43 / 3.03 / 3.29 vs band_nla
# 3.74 / 4.14 / 3.72 -> the Stage-I spread is band_nla-like; r2 loses 30 % in the NLE chain (3.29 -> 2.30; median
# per-observation corr(Om,s8) -0.81 vs r0 -0.90 = a looser degeneracy, not lost Om/s8 information). Hence:
# - r1: BAND reroll (weakest Stage I) -> `kids_legacy_band_nla_z_vdq_rr` (identical recipe, own ckpt dir: a same-name
#   rerun would share pretrain_ncosmoNone_1 and folder-min would pick by GLASS val) + chain `_rr_k8_r1`.
# - r2: NLE-chain reroll on its EXISTING band -> `_k8_r2b` (the nla_z `_r3b` precedent).
# - r3, r4: fresh indices on the original rows (fresh dirs; Stage I via `--repeat-indices 3,4` at the CLI).
# Same index => same split_seed => match string stays `ncosmoNone_{r}`; the arm resolves the reroll rows through
# EXPERIMENT_OVERRIDES in src/ml/eval/arms.py. guard-c starts at 50 (raise <= 150 per repeat after verifying own whitener
# + own best Stage-A flow in the log; pre-authorised by the user 2026-10-21).
# Launch:
#   train --exp kids_legacy_band_nla_z_vdq --repeat-indices 3,4 ; train --exp kids_legacy_band_nla_z_vdq_rr --repeat-indices 1
#   embed --gpu v100 --ncpu 1 --mem-gb 24 --target glass_nle_pretrain_band_nla_z_vdq{,_rr}_k8_r<r>[b] --sources <its band row>
#   embed --cpu --partition CORES40 --wall_h 30 --target gower_nle_finetune_band_nla_z_vdq{,_rr}_k8_r<r>[b]_ens9_e400 --sources <its band row>
# ⭐ Stage B of these slots = lr 4e-4, 400 epochs (`_ens9_e400`), NOT the production 8e-5 / 150: the r2-split sweep below found
# the production Stage B under-trained (paired m0 best val NLL 8e-5/150 -9.67, 2e-4/400 -11.46, 4e-4/150 -11.10, 4e-4/400
# -12.20; held-out val still improving => under-fit, not over-fit). Calibration is checked on the `_sw_lr{2e4,4e4}_ens3`
# eval rows before any of these enters the arm. r0 keeps its original 8e-5 / 150 chain.
def _band_nla_z_nle_b_e400(r, pretrain_exp=None):
    ft = _band_nla_z_nle_b_vdq(r, pretrain_exp)
    ft.update(lr=4e-4, epochs=400)
    return ft


_band_nla_z_rr = _band_nla_z_vdq()
_band_nla_z_rr["repeat_indices"] = [1]
kids_legacy_vdq_experiments["kids_legacy_band_nla_z_vdq_rr"] = _band_nla_z_rr
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_k8_r3"] = _band_nla_z_nle_a_vdq(3)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_k8_r4"] = _band_nla_z_nle_a_vdq(4)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_rr_k8_r1"] = _band_nla_z_nle_a_vdq(1)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_k8_r2b"] = _band_nla_z_nle_a_vdq(2)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r3_ens9_e400"] = _band_nla_z_nle_b_e400(3)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r4_ens9_e400"] = _band_nla_z_nle_b_e400(4)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_rr_k8_r1_ens9_e400"] = _band_nla_z_nle_b_e400(
    1, "glass_nle_pretrain_band_nla_z_vdq_rr_k8_r1")
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2b_ens9_e400"] = _band_nla_z_nle_b_e400(
    2, "glass_nle_pretrain_band_nla_z_vdq_k8_r2b")
for _n in ("glass_nle_pretrain_band_nla_z_vdq_k8_r3", "glass_nle_pretrain_band_nla_z_vdq_k8_r4",
           "glass_nle_pretrain_band_nla_z_vdq_k8_r2b", "gower_nle_finetune_band_nla_z_vdq_k8_r3_ens9_e400",
           "gower_nle_finetune_band_nla_z_vdq_k8_r4_ens9_e400", "gower_nle_finetune_band_nla_z_vdq_k8_r2b_ens9_e400"):
    VDQ_NLE_SOURCES[_n] = "kids_legacy_band_nla_z_vdq"
for _n in ("glass_nle_pretrain_band_nla_z_vdq_rr_k8_r1", "gower_nle_finetune_band_nla_z_vdq_rr_k8_r1_ens9_e400"):
    VDQ_NLE_SOURCES[_n] = "kids_legacy_band_nla_z_vdq_rr"

# Stage-B LR/epoch side sweep on the r2 split (user 2026-10-21: "experiment with different (lower) LRs"). Production Stage B
# (lr 8e-5, exp decay, 150 ep) still descends at ep150 in every member (no LR-too-high signature), so the grid spans
# lower LR, higher LR and longer training. ens3 + run_evaluation=False: member j shares ensemble_seed j (= the same
# train/val reshuffle) with production r2 member j, so val NLL compares PAIRWISE per member. Warm-starts from the
# existing r2 Stage A (guard 150 as production r2; resolution already verified for that Stage A). A winner still needs
# one full bundled eval before it becomes the recipe.
#   embed --cpu --partition CORES40 --ncpu 8 --target gower_nle_finetune_band_nla_z_vdq_k8_r2_lrs_<tag> --sources kids_legacy_band_nla_z_vdq
def _band_nla_z_lr_sweep_b(lr, epochs):
    ft = _band_nla_z_nle_b_vdq(2)
    ft.update(ensemble_repeats=3, run_evaluation=False, lr=lr, epochs=epochs, whiten_warmstart_max_gap_nats=150.0)
    return ft


# Literal keys (run_remote.py:known_experiment scans for them).
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_lrs_8e5_e150"] = _band_nla_z_lr_sweep_b(8e-5, 150)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_lrs_8e5_e400"] = _band_nla_z_lr_sweep_b(8e-5, 400)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_lrs_3e5_e150"] = _band_nla_z_lr_sweep_b(3e-5, 150)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_lrs_3e5_e400"] = _band_nla_z_lr_sweep_b(3e-5, 400)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_lrs_2e4_e150"] = _band_nla_z_lr_sweep_b(2e-4, 150)
# Round 1 result (paired best val NLL per member vs production r2 m0-m2 -9.67/-9.55/-9.79; rerun 8e5_e150 reproduces to 0.1):
# 3e5_e150 +1.1 worse, 2e4_e150 -0.8 better, 8e5_e400 m0 -1.0 better => Stage B is under-trained, not over-stepped.
# Round 2 pushes LR/epochs up.
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_lrs_2e4_e400"] = _band_nla_z_lr_sweep_b(2e-4, 400)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_lrs_4e4_e150"] = _band_nla_z_lr_sweep_b(4e-4, 150)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_lrs_4e4_e400"] = _band_nla_z_lr_sweep_b(4e-4, 400)
for _tag in ("8e5_e150", "8e5_e400", "3e5_e150", "3e5_e400", "2e4_e150", "2e4_e400", "4e4_e150", "4e4_e400"):
    VDQ_NLE_SOURCES[f"gower_nle_finetune_band_nla_z_vdq_k8_r2_lrs_{_tag}"] = "kids_legacy_band_nla_z_vdq"

# NLE-dynamics sweep on the r2 band (user 2026-10-21: "focus on the NLE training dynamics ... the truncation dimension may
# matter"). Whitening truncation k in {6, 4} vs pure-whiten k=8, and a 400-epoch Stage A (production Stage A still descends
# at ep150). Val NLL is NOT commensurable across k (densities in different dimensions), so these Stage-B rows run the
# bundled eval: ens3 everywhere (incl. a k8 ens3 baseline on the production r2 Stage A) => FoM(Om,s8) compares at equal
# ensemble size. guard-c 150 up front (diagnostic rows; own-whitener resolution still checked in each log).
def _band_nla_z_sweep_a(k, epochs=None):
    pre = _band_nla_z_nle_a_vdq(2)
    pre["whiten_embeddings"] = {"k": k}
    if epochs:
        pre["epochs"] = epochs
    return pre


def _band_nla_z_sweep_b(pretrain_exp, k):
    ft = _band_nla_z_nle_b_vdq(2, pretrain_exp)
    ft["whiten_embeddings"] = {"k": k}
    ft.update(ensemble_repeats=3, whiten_warmstart_max_gap_nats=150.0)
    return ft


kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_k6_r2_sw"] = _band_nla_z_sweep_a(6)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_k4_r2_sw"] = _band_nla_z_sweep_a(4)
kids_legacy_vdq_experiments["glass_nle_pretrain_band_nla_z_vdq_k8_r2_sw_a400"] = _band_nla_z_sweep_a(8, epochs=400)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_sw_base_ens3"] = _band_nla_z_sweep_b(
    "glass_nle_pretrain_band_nla_z_vdq_k8_r2", 8)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k6_r2_sw_ens3"] = _band_nla_z_sweep_b(
    "glass_nle_pretrain_band_nla_z_vdq_k6_r2_sw", 6)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k4_r2_sw_ens3"] = _band_nla_z_sweep_b(
    "glass_nle_pretrain_band_nla_z_vdq_k4_r2_sw", 4)
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_sw_a400_ens3"] = _band_nla_z_sweep_b(
    "glass_nle_pretrain_band_nla_z_vdq_k8_r2_sw_a400", 8)
# Does the val-NLL gain of a higher Stage-B LR buy FoM? Same as the k8 ens3 baseline but lr 2e-4 (round-1 winner at 150 ep).
_ft = _band_nla_z_sweep_b("glass_nle_pretrain_band_nla_z_vdq_k8_r2", 8)
_ft["lr"] = 2e-4
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_sw_lr2e4_ens3"] = _ft
VDQ_NLE_SOURCES["gower_nle_finetune_band_nla_z_vdq_k8_r2_sw_lr2e4_ens3"] = "kids_legacy_band_nla_z_vdq"
# ... and lr 4e-4 = the pre-2026-07-06 NLE-finetune default (cut to 8e-5 then, "gentler, ALL variates"): round-2 m0 -11.10 at
# 150 ep (-1.4 vs production). Calibration (TARP) must be checked before it can enter a production row.
_ft = _band_nla_z_sweep_b("glass_nle_pretrain_band_nla_z_vdq_k8_r2", 8)
_ft["lr"] = 4e-4
kids_legacy_vdq_experiments["gower_nle_finetune_band_nla_z_vdq_k8_r2_sw_lr4e4_ens3"] = _ft
VDQ_NLE_SOURCES["gower_nle_finetune_band_nla_z_vdq_k8_r2_sw_lr4e4_ens3"] = "kids_legacy_band_nla_z_vdq"
for _n in ("glass_nle_pretrain_band_nla_z_vdq_k6_r2_sw", "glass_nle_pretrain_band_nla_z_vdq_k4_r2_sw",
           "glass_nle_pretrain_band_nla_z_vdq_k8_r2_sw_a400", "gower_nle_finetune_band_nla_z_vdq_k8_r2_sw_base_ens3",
           "gower_nle_finetune_band_nla_z_vdq_k6_r2_sw_ens3", "gower_nle_finetune_band_nla_z_vdq_k4_r2_sw_ens3",
           "gower_nle_finetune_band_nla_z_vdq_k8_r2_sw_a400_ens3"):
    VDQ_NLE_SOURCES[_n] = "kids_legacy_band_nla_z_vdq"
assert set(VDQ_NLE_SOURCES) <= set(kids_legacy_vdq_experiments)
