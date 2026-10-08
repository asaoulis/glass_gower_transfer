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
