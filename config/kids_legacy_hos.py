"""Beyond-2-pt HOS hybrid (task eval-and-viz/hos-hybrid-training) — model configs.

The map branch of the VDQ field-level hybrid is replaced by a feed-forward MLP over the noise-texture-robust
higher-order statistics of the same baked maps: R1 = the 56 tomographic third moments at 8' and 16' (112 numbers;
src/ml/data/hos.py; selection in .claude/runs/eval-and-viz/robust-hos-selection/artifacts/RECOMMENDATIONS.md).

Data: per-mock HOS SIDECARS of the VDQ sc8a1 bakes (`eval.py --mode hos-bake`), written under
{MODELS_ROOT}/checkpoints/hos_sidecars/<store>/<basename> (FLAT: the glob order, hence the split, equals the bake's;
per-shard transfer archives live in <store>_tar/). They carry `hos/vector`, `hos_ext/moments`,
`mixed_bandpowers` and `cosmo_dict` verbatim, with the SAME basenames as the bake, so the frozen Stage-I band's split,
its scaler subsample and the Gower fixed-test lock are identical. Paths resolve from $GLASS_MODELS_ROOT, so the same
rows run locally (GLASS_MODELS_ROOT=/data/alex/hos_models) and on the cluster (default).

Each row starts from the production row it must match and overrides only the HOS-specific keys:
  kids_legacy_hos_nla_m_vdq                  GLASS pretraining, frozen per-repeat `kids_legacy_band_nla_m_vdq`
                                             (= `kids_legacy_band_nla_m_vdq` split/cosmo/scaler keys)
  gower_npe_finetune_hos_nla_m_vdq_ens9      Gower NPE ens9 fine-tune (= the band pack's split/eval/schedule keys)
  gower_npe_finetune_band_nla_m_vdq_ens9_side  the band-only baseline on the sidecar store (dict-diff vs the cluster
                                             pack = data_patterns, checkpoint_path, preload/worker keys only)
  *_r2                                       R2 control: only the 20 distinct-bin triples per scale (feature_index)
Loader notes: `dataset_quantities` lists mixed_bandpowers BEFORE hos (the scaler fit draws one subsample
permutation per key in order, so the band's scaler subsample is unchanged); `per_dim_keys=['hos']` gives the HOS
vector a per-dimension standard scaler; `preload_data` caches the ~32 kB sidecars in RAM (num_workers=0).
Hyperparameters marked SWEEP are set from artifacts/SWEEP_SPEC.md results.
"""
import copy
import os

from config.kids_legacy_bgp import _assert_final_summary_dim
from config.kids_legacy_vdq import _FINETUNE_VAL_CHECK_INTERVAL, kids_legacy_vdq_experiments as _VDQ

_MODELS = os.environ.get("GLASS_MODELS_ROOT", "/share/gpu5/asaoulis/transfer_models")
_SIDE = f"{_MODELS}/checkpoints/hos_sidecars"
_EB = "fwhm4_lmin56_lcut1400"
_HOS_GLASS = f"{_SIDE}/glass_vdq_nla_m_f16_sc8a1_{_EB}/output_*.h5"
_HOS_GOWER = f"{_SIDE}/gower_vdq_nla_m_f16_sc8a1_{_EB}/output_*.h5"
_BAND_CKPT = f"{_MODELS}/checkpoints/kids_legacy_band_nla_m_vdq/"
_HOS_CKPT = f"{_MODELS}/checkpoints/kids_legacy_hos_nla_m_vdq/"
_HOS_CKPT_R2 = f"{_MODELS}/checkpoints/kids_legacy_hos_r2_nla_m_vdq/"

# R2 = distinct-bin triples i < j < l of each scale block (56 per scale; R1 order = scale 8 then 16)
_TRIPLES = [(i, j, l) for i in range(6) for j in range(i, 6) for l in range(j, 6)]
R2_FEATURE_INDEX = [s * 56 + k for s in (0, 1) for k, (i, j, l) in enumerate(_TRIPLES) if i < j < l]

# Vector-store loader keys: everything in RAM, no worker processes (prefetch/persistent keys are meaningless).
_VECTOR_LOADER = {"preload_data": True, "num_workers": 0, "pin_memory": False, "persistent_workers": None,
                  "prefetch_factor": None}

HOS_MLP_DEFAULT = {"input_dim": 112, "hidden": 128, "depth": 2, "dropout": 0.1, "input_transform": None}  # sweep A1-A4


def hos_glass_row(hos_kwargs=None, data_patterns=_HOS_GLASS, band_ckpt=_BAND_CKPT, **top):
    """GLASS pretraining row: frozen Stage-I band (8-D) + HOS MLP (8-D), concat -> 8-D summary -> NPE flow."""
    band = _VDQ["kids_legacy_band_nla_m_vdq"]
    c = copy.deepcopy(band)                       # cosmo params, split + scaler keys == the band's
    c.update({
        "model_type": "kids_hybrid_bandpowers_hos",
        "dataset_quantities": ["mixed_bandpowers", "hos"],
        "data_patterns": data_patterns,
        "latent_dim": 16,                         # concat width: 8 band + 8 HOS
        "model_kwargs": {
            "bandpower_type": "mlp",
            "bandpower_latent_dim": 8,
            "bandpower_kwargs": copy.deepcopy(band["model_kwargs"]),   # == the checkpoint's architecture
            "map_kwargs": {**HOS_MLP_DEFAULT, **(hos_kwargs or {})},
            "hybrid_output_dim": 8,
            "fusion_type": "concat",
        },
        "pretrained_band_ckpt_path": band_ckpt,
        "freeze_band": True,
        "flow_kwargs": {"hidden_features": 32},
        "scaler_options": {"data": {"type": "standard", "keys": None, "per_dim_keys": ["hos"]},
                           "cosmo": {"type": "preset"}},
        "eb_map_variant": None,
        "augment_eb_patches": False,
        "batch_size": 256,                        # SWEEP
        "lr": 3e-4,                               # sweep A2/A4
        "optimizer_kwargs": {"weight_decay": 0.1, "betas": (0.9, 0.999)},   # sweep A2/A4 (best epochs 63-104)
        "epochs": 120,                            # sweep A4: ceil(1.25 x median best epoch 102), capped at 120
        "scheduler_type": "cyclic",
        "scheduler_kwargs": {"warmup": 2000, "min_factor": 0.1, "cyclic_period_steps": 6000},
        "use_KL_loss": False,
        "precision": "32",                        # tiny model: fp32 everywhere (CPU sweeps: bf16 autocast ~2x slower)
        "project": "hos-hybrid",
        "repeat_indices": [0, 1, 2, 3, 4],
        **_VECTOR_LOADER,
    })
    c.update(top)
    return c


def hos_gower_row(glass_ckpt=_HOS_CKPT, hos_kwargs=None, data_patterns=_HOS_GOWER, **top):
    """Gower NPE ens9 fine-tune of the GLASS HOS hybrid (whole-model warm start, per repeat)."""
    pack = _VDQ["gower_npe_finetune_band_nla_m_vdq_ens9"]
    c = hos_glass_row(hos_kwargs=hos_kwargs, data_patterns=data_patterns)
    for k in ("epochs", "lr", "batch_size", "scheduler_type", "scheduler_kwargs", "ensemble_repeats",
              "max_trainval_cosmos", "train_frac", "val_frac", "test_frac", "fixed_test_sim_ids",
              "match_num_cosmo", "repeat_indices", "val_check_interval"):
        c[k] = copy.deepcopy(pack[k])
    c.pop("pretrained_band_ckpt_path", None)      # the whole model comes from the GLASS checkpoint
    c.pop("optimizer_kwargs", None)               # Stage B keeps the default weight decay (as swept in B1)
    c["checkpoint_path"] = glass_ckpt
    # Stage B keeps the 2-pt pack keys (lr 1e-5, 10 epochs, all trainable): the B1 grid (lr x epochs x freeze, r0+r1)
    # is flat within 0.011 nats of its top cell vs ~0.03 seed noise; freezing band (-0.03) or band+HOS (-0.12) hurts.
    c["freeze_band"] = False
    c["project"] = "gower-finetuning-hos"
    c.update(top)
    return c


def band_side_row():
    """The cluster band-only Gower pack, re-pointed at the sidecar store (local apples-to-apples baseline)."""
    c = copy.deepcopy(_VDQ["gower_npe_finetune_band_nla_m_vdq_ens9"])
    c["data_patterns"] = _HOS_GOWER
    c["precision"] = "32"
    c["checkpoint_path"] = _BAND_CKPT
    c.update(_VECTOR_LOADER)
    return c


kids_legacy_hos_experiments = {}
kids_legacy_hos_experiments["kids_legacy_hos_nla_m_vdq"] = _assert_final_summary_dim(
    hos_glass_row(), 8, "kids_legacy_hos_nla_m_vdq")
kids_legacy_hos_experiments["gower_npe_finetune_hos_nla_m_vdq_ens9"] = _assert_final_summary_dim(
    hos_gower_row(), 8, "gower_npe_finetune_hos_nla_m_vdq_ens9")
kids_legacy_hos_experiments["gower_npe_finetune_band_nla_m_vdq_ens9_side"] = band_side_row()
# R2 control (user D6): distinct-bin triples only.
kids_legacy_hos_experiments["kids_legacy_hos_r2_nla_m_vdq"] = _assert_final_summary_dim(
    hos_glass_row(hos_kwargs={"feature_index": R2_FEATURE_INDEX}), 8, "kids_legacy_hos_r2_nla_m_vdq")
kids_legacy_hos_experiments["gower_npe_finetune_hos_r2_nla_m_vdq_ens9"] = _assert_final_summary_dim(
    hos_gower_row(_HOS_CKPT_R2, hos_kwargs={"feature_index": R2_FEATURE_INDEX}), 8,
    "gower_npe_finetune_hos_r2_nla_m_vdq_ens9")
# The local smoke fixture (665 Gower raw mocks -> bake -> sidecars) for the smoke gate.
kids_legacy_hos_experiments["kids_legacy_hos_nla_m_vdq_smoke"] = hos_glass_row(
    data_patterns="/data/alex/hos_models/fixture/sidecars/output_*.h5", band_ckpt=None, freeze_band=False,
    epochs=1, repeat_indices=[0])


# === HOS NLE chain: the band P3b k=8 procedure (config/kids_legacy_vdq.py `_band_nle_a_vdq` / `_band_nle_b_vdq`) on the
# HOS sidecar stores. Frozen source = the per-repeat GLASS hybrid `kids_legacy_hos_nla_m_vdq/pretrain_ncosmoNone_{r}` (one
# compression head per repeat, NOT the Gower ens9 NPE pack); whiten k=8 (pure whiten of the 8-D summary); Stage B ens9,
# 150 epochs, guard-c 50, max_trainval_cosmos [300] + the 200-id fixed test lock. Dict-diff vs the band rows: the sidecar
# data_patterns, the HOS scaler (per-dim `hos`, identical to the source's so the GLASS split + input frame reproduce the
# encoder's), the vector-store loader keys, fp32 and (Stage B) the Stage-A checkpoint dir under $GLASS_MODELS_ROOT.
from config.kids_legacy_vdq import _band_nle_a_vdq, _band_nle_b_vdq  # noqa: E402

HOS_NLE_K = 8
HOS_NLE_REPEATS = (0, 1, 2, 3, 4)
_HOS_SOURCE = "kids_legacy_hos_nla_m_vdq"


def _hos_nle_common(c, data_patterns):
    c["data_patterns"] = data_patterns
    c["scaler_options"] = copy.deepcopy(kids_legacy_hos_experiments[_HOS_SOURCE]["scaler_options"])
    c["precision"] = "32"
    c.update(_VECTOR_LOADER)
    return c


def hos_nle_a_row(r):
    return _hos_nle_common(_band_nle_a_vdq(r), _HOS_GLASS)


def hos_nle_b_row(r):
    c = _hos_nle_common(_band_nle_b_vdq(r), _HOS_GOWER)
    c["pretrained_band_ckpt_path"] = f"{_MODELS}/checkpoints/glass_nle_pretrain_hos_nla_m_vdq_k{HOS_NLE_K}_r{r}/"
    # Eval MCMC pool: the default 36 workers put the 1592-mock test set (124 items) in ~3.5 waves of
    # >8 h each, past the 24 h wall (2026-10-22, hos_nle_eval_xo r0-4). 63 on a 64-core node, as the
    # vdq variate chains (_VARIATE_NLE_B_SAMPLING_JOBS); eval-only, run it with --wall_h 48.
    c["sampling_num_jobs"] = 63
    return c


# Source encoders of the HOS NLE rows (read by src/ml/eval/nle_external.py; the e890aec rule: never guessed at eval).
HOS_NLE_SOURCES = {}
# Literal keys (run_remote.py:known_experiment scans for them).
kids_legacy_hos_experiments["glass_nle_pretrain_hos_nla_m_vdq_k8_r0"] = hos_nle_a_row(0)
kids_legacy_hos_experiments["glass_nle_pretrain_hos_nla_m_vdq_k8_r1"] = hos_nle_a_row(1)
kids_legacy_hos_experiments["glass_nle_pretrain_hos_nla_m_vdq_k8_r2"] = hos_nle_a_row(2)
kids_legacy_hos_experiments["glass_nle_pretrain_hos_nla_m_vdq_k8_r3"] = hos_nle_a_row(3)
kids_legacy_hos_experiments["glass_nle_pretrain_hos_nla_m_vdq_k8_r4"] = hos_nle_a_row(4)
kids_legacy_hos_experiments["gower_nle_finetune_hos_nla_m_vdq_k8_r0_ens9_e150"] = hos_nle_b_row(0)
kids_legacy_hos_experiments["gower_nle_finetune_hos_nla_m_vdq_k8_r1_ens9_e150"] = hos_nle_b_row(1)
kids_legacy_hos_experiments["gower_nle_finetune_hos_nla_m_vdq_k8_r2_ens9_e150"] = hos_nle_b_row(2)
kids_legacy_hos_experiments["gower_nle_finetune_hos_nla_m_vdq_k8_r3_ens9_e150"] = hos_nle_b_row(3)
kids_legacy_hos_experiments["gower_nle_finetune_hos_nla_m_vdq_k8_r4_ens9_e150"] = hos_nle_b_row(4)
for _r in HOS_NLE_REPEATS:
    HOS_NLE_SOURCES[f"glass_nle_pretrain_hos_nla_m_vdq_k{HOS_NLE_K}_r{_r}"] = _HOS_SOURCE
    HOS_NLE_SOURCES[f"gower_nle_finetune_hos_nla_m_vdq_k{HOS_NLE_K}_r{_r}_ens9_e150"] = _HOS_SOURCE
assert set(HOS_NLE_SOURCES) <= set(kids_legacy_hos_experiments)


# === Single-node CPU training waves (`eval.py --mode hos-train --experiments <wave>`, src/ml/parallel_train.py) ========
# The HOS models are tiny and CPU-efficient (GPU kernel-launch bound), so one CORES64 node trains many of them at once.
# A task is a train.py run of one repeat (REPEAT_INDICES env) or a train_embeddings.py run against its registered source;
# `group` = one sidecar store + key set (the first task of a group builds the store's .preload_cache before the rest
# start); `after` = tasks that must finish with rc 0 first. Order = launch priority.
def _ptask(kind, exp, repeat, group, after=(), env=None):
    t = {"kind": kind, "exp": exp, "repeat": int(repeat), "group": group, "after": list(after), "env": dict(env or {})}
    if kind == "embed":
        t["sources"] = [HOS_NLE_SOURCES[exp]]
    return t


def _stage_a(r):
    return f"glass_nle_pretrain_hos_nla_m_vdq_k{HOS_NLE_K}_r{r}"


def _stage_b(r):
    return f"gower_nle_finetune_hos_nla_m_vdq_k{HOS_NLE_K}_r{r}_ens9_e150"


HOS_TRAIN_WAVES = {
    # NLE Stage A r0-4, the 15 Gower NPE ens9 fine-tunes (HOS, R2 control, band-only `_side` baseline), then NLE Stage B
    # once its own Stage A is done, with the deferred ensemble MCMC OFF (RUN_EVALUATION=0: 5 x the node-wide MCMC would
    # fight over one node; it runs per repeat in `hos_nle_eval`).
    "hos_train": (
        [_ptask("embed", _stage_a(r), r, "glass_hos") for r in HOS_NLE_REPEATS]
        + [_ptask("train", e, r, g) for e, g in (("gower_npe_finetune_hos_nla_m_vdq_ens9", "gower_hos"),
                                                  ("gower_npe_finetune_hos_r2_nla_m_vdq_ens9", "gower_hos"),
                                                  ("gower_npe_finetune_band_nla_m_vdq_ens9_side", "gower_band"))
           for r in HOS_NLE_REPEATS]
        + [_ptask("embed", _stage_b(r), r, "gower_hos", after=[_stage_a(r)], env={"RUN_EVALUATION": "0"})
           for r in HOS_NLE_REPEATS]),
    # Per-member input frames for external (observation) scoring: `recover-scalers` for every Stage-B member (scalers.pt,
    # the frame that member trained in), then `--verify-frames` per repeat once its 9 members are done.
    "hos_rs": (
        [dict(_ptask("eval", _stage_b(r), r, "gower_hos"), label=f"rs_r{r}_m{m}",
              args=["--mode", "recover-scalers", "--experiments", _stage_b(r), "--repeat-indices", str(r),
                    "--members", str(m)]) for r in HOS_NLE_REPEATS for m in range(9)]
        + [dict(_ptask("eval", _stage_b(r), r, "gower_hos", after=[f"rs_r{r}_m{m}" for m in range(9)]),
                label=f"verify_r{r}", args=["--mode", "recover-scalers", "--verify-frames", "--experiments", _stage_b(r),
                                            "--repeat-indices", str(r)]) for r in HOS_NLE_REPEATS]),
    # In-distribution misspec eval (FoM / TARP / test log-prob on the 200-id lock) of the three Gower NPE ens9 packs, one
    # task per (pack, repeat), on the HOS sidecar copy of the gower_vdq_nla_m store (misspec variate set gower_vdq_hos).
    "hos_npe_eval": [
        dict(_ptask("eval", e, r, g), label=f"npe_eval_{e}_r{r}",
             args=["--mode", "misspec", "--misspec-base", e, "--variates", "gower_vdq_hos", "--repeat-indices", str(r)])
        for e, g in (("gower_npe_finetune_hos_nla_m_vdq_ens9", "gower_hos"),
                     ("gower_npe_finetune_hos_r2_nla_m_vdq_ens9", "gower_hos"),
                     ("gower_npe_finetune_band_nla_m_vdq_ens9_side", "gower_band"))
        for r in HOS_NLE_REPEATS],
    # Eval-only re-run of the trained Stage B ensembles (same names, RUN_TRAINING=0 reuses the members + embedding
    # cache): the train-time deferred ensemble MCMC on the fixed test lock. One repeat per node (--repeat-indices r).
    "hos_nle_eval": [_ptask("embed", _stage_b(r), r, "gower_hos", env={"RUN_TRAINING": "0"}) for r in HOS_NLE_REPEATS],
}


# === hos_x superset (task eval-and-viz/hos-3pt-information; APPEND-ONLY, no existing row changes) ====================
# Sidecars of the hos_x bake (`eval.py --mode hos-bake --hos-variant x`) live in a NEW dir <store>_hosx/ (never '_hos':
# nle_external.store_glob resolves <store>_hos for the production arm). They carry hos/vector + hos_ext unchanged plus
# hos_x/vector (1344 = R1 | Geq | Gpool | Gmrg | Ceq | Cmrg, src/ml/data/hos_x.py), with the SAME basenames as the R1
# sidecars, so the frozen band's split, its scaler subsample and the Gower lock are identical. A row selects its feature
# set by hos_x group keys (hos_x.group_index); dict-diff vs the R1 rows = data_patterns, dataset_quantities ('hos_x'),
# the per-dim scaler key, map_kwargs input_key / input_dim / feature_index (and checkpoint_path for the Gower rows).
from src.ml.data.hos_x import HOSX_BAKE_NAMES, group_index  # noqa: E402

_HOSX_GLASS = f"{_SIDE}/glass_vdq_nla_m_f16_sc8a1_{_EB}_hosx/output_*.h5"
_HOSX_GOWER = f"{_SIDE}/gower_vdq_nla_m_f16_sc8a1_{_EB}_hosx/output_*.h5"


def _hosx_top():
    return {"dataset_quantities": ["mixed_bandpowers", "hos_x"],      # band FIRST (scaler subsample order)
            "scaler_options": {"data": {"type": "standard", "keys": None, "per_dim_keys": ["hos_x"]},
                               "cosmo": {"type": "preset"}}}


def hosx_kwargs(groups, hos_kwargs=None):
    return {"input_key": "hos_x", "input_dim": len(HOSX_BAKE_NAMES), "feature_index": group_index(groups),
            **(hos_kwargs or {})}


def hosx_glass_row(groups, hos_kwargs=None, data_patterns=_HOSX_GLASS, **top):
    """hos_glass_row on the hos_x store with the feature set = union of the hos_x `groups`."""
    return hos_glass_row(hos_kwargs=hosx_kwargs(groups, hos_kwargs), data_patterns=data_patterns,
                         **{**_hosx_top(), **top})


def hosx_gower_row(glass_ckpt, groups, hos_kwargs=None, data_patterns=_HOSX_GOWER, **top):
    """hos_gower_row (Gower NPE ens9 pack keys) on the hos_x store, warm-started from the GLASS hosx row."""
    return hos_gower_row(glass_ckpt=glass_ckpt, hos_kwargs=hosx_kwargs(groups, hos_kwargs), data_patterns=data_patterns,
                         **{**_hosx_top(), **top})


# Phase-C finalists (artifacts/PHASE_A.md): R1 control on the hosx store + Cmrg/no4 (production candidate) + Cmrg (incl.
# theta = 4', information reference) + Gmrg (Gaussian Linke-like).
HOSX_FINALISTS = {"r1": ["R1"], "cmrgno4": ["R1", "Cmrg/no4"], "cmrg": ["R1", "Cmrg"], "gmrg": ["R1", "Gmrg"]}
for _tag, _groups in HOSX_FINALISTS.items():
    _g = f"kids_legacy_hosx_{_tag}_nla_m_vdq"
    kids_legacy_hos_experiments[_g] = _assert_final_summary_dim(hosx_glass_row(_groups), 8, _g)
    _w = f"gower_npe_finetune_hosx_{_tag}_nla_m_vdq_ens9"
    kids_legacy_hos_experiments[_w] = _assert_final_summary_dim(
        hosx_gower_row(f"{_MODELS}/checkpoints/{_g}/", _groups), 8, _w)


# === Gower HOS outlier cut (user 2026-10-21: "just drop these extreme outliers", option F2) ============================
# 60 Gower mocks (17 sims) whose R1 vector sits > 50 robust sigma from the GLASS distribution (extended low-z bin-1
# structures; hos-hybrid-training artifacts/GOWER_HOS_OUTLIERS.md) broke NLE Stage B guard-c and inflated the Gower HOS
# scaler. `exclude_files` drops them AFTER the cosmology split (split unchanged; the eval lock's noise-index-0 test subset
# holds none of them). NEW names (`_xo`): the contaminated packs' checkpoints stay where they are and never mix in.
# Every Gower arm gets the same cut (HOS, R2 control, band-only `_side` baseline), so the comparison stays like-for-like.
GOWER_HOS_OUTLIERS = "config/hos_assets/gower_hos_outliers.json"   # repo-relative, like fixed_test_sim_ids


def _xo(c):
    c = copy.deepcopy(c)
    c["exclude_files"] = GOWER_HOS_OUTLIERS
    return c


_XO = {e: f"{e}_xo" for e in ("gower_npe_finetune_hos_nla_m_vdq_ens9", "gower_npe_finetune_hos_r2_nla_m_vdq_ens9",
                              "gower_npe_finetune_band_nla_m_vdq_ens9_side")}
_XO.update({_stage_b(r): f"{_stage_b(r)}_xo" for r in HOS_NLE_REPEATS})
for _old, _new in _XO.items():
    kids_legacy_hos_experiments[_new] = _xo(kids_legacy_hos_experiments[_old])
    if _old in HOS_NLE_SOURCES:
        HOS_NLE_SOURCES[_new] = HOS_NLE_SOURCES[_old]
# Literal keys (run_remote.py:known_experiment scans for them): gower_npe_finetune_hos_nla_m_vdq_ens9_xo,
# gower_npe_finetune_hos_r2_nla_m_vdq_ens9_xo, gower_npe_finetune_band_nla_m_vdq_ens9_side_xo,
# gower_nle_finetune_hos_nla_m_vdq_k8_r0_ens9_e150_xo .. gower_nle_finetune_hos_nla_m_vdq_k8_r4_ens9_e150_xo


def _xo_task(t, relabelled=()):
    t = copy.deepcopy(t)
    t["exp"] = _XO.get(t["exp"], t["exp"])
    t["after"] = [f"{a}_xo" if a in relabelled else _XO.get(a, a) for a in t.get("after", [])]
    t["args"] = [_XO.get(a, a) for a in t.get("args", [])] or t.pop("args", None) or []
    if not t["args"]:
        t.pop("args")
    if t.get("label"):
        t["label"] += "_xo"
    if t["kind"] == "embed":
        t["sources"] = [HOS_NLE_SOURCES[t["exp"]]]
    return t


# The `_xo` waves: same tasks on the cut rows. hos_train_xo drops NLE Stage A (unchanged: GLASS only; its checkpoints
# exist, and an `after` naming a task outside the wave counts as done).
HOS_TRAIN_WAVES["hos_train_xo"] = [_xo_task(t) for t in HOS_TRAIN_WAVES["hos_train"] if t["exp"] in _XO]
for _w in ("hos_npe_eval", "hos_rs", "hos_nle_eval"):
    _rel = {t["label"] for t in HOS_TRAIN_WAVES[_w] if t.get("label")}
    HOS_TRAIN_WAVES[f"{_w}_xo"] = [_xo_task(t, _rel) for t in HOS_TRAIN_WAVES[_w]]


# === R1 + Gmrg PRODUCTION arm (task eval-and-viz/hos-gmrg-observation = U7 of hos-3pt-information; APPEND-ONLY) =======
# The R1 production chain (GLASS NPE -> NLE Stage A k=8 -> Gower Stage B ens9 e150 `_xo` -> recover-scalers -> obs MCMC)
# on the hos_x store with the feature set R1 + Gmrg (476). GLASS NPE = the existing `kids_legacy_hosx_gmrg_nla_m_vdq` row,
# trained per repeat INSIDE the CPU wave (the hybrid is GPU kernel-launch bound; CPU ~3x faster per step). Dict-diff of the
# NLE rows vs the R1 `_xo` rows: data_patterns (hosx stores), the per-dim `hos_x` scaler (== the gmrg source's), the
# Stage-A checkpoint dir and exclude_files = F2 + the 18 Gmrg > 50 robust-sigma Gower mocks (same F2 rule on the Gmrg
# vector; config/hos_assets/gower_hosx_gmrg_outliers.json, gitignored dir -> git add -f).
_HOSX_GMRG_SOURCE = "kids_legacy_hosx_gmrg_nla_m_vdq"
GOWER_HOSX_GMRG_OUTLIERS = "config/hos_assets/gower_hosx_gmrg_outliers.json"


def _hosx_gmrg_nle_common(c, data_patterns):
    c["data_patterns"] = data_patterns
    c["scaler_options"] = copy.deepcopy(kids_legacy_hos_experiments[_HOSX_GMRG_SOURCE]["scaler_options"])
    c["precision"] = "32"
    c.update(_VECTOR_LOADER)
    return c


def hosx_gmrg_nle_a_row(r):
    return _hosx_gmrg_nle_common(_band_nle_a_vdq(r), _HOSX_GLASS)


def hosx_gmrg_nle_b_row(r):
    c = _hosx_gmrg_nle_common(_band_nle_b_vdq(r), _HOSX_GOWER)
    c["pretrained_band_ckpt_path"] = f"{_MODELS}/checkpoints/glass_nle_pretrain_hosx_gmrg_nla_m_vdq_k{HOS_NLE_K}_r{r}/"
    c["sampling_num_jobs"] = 63                    # as hos_nle_b_row (eval-only, --wall_h 48)
    c["exclude_files"] = GOWER_HOSX_GMRG_OUTLIERS
    return c


def _gmrg_a(r):
    return f"glass_nle_pretrain_hosx_gmrg_nla_m_vdq_k{HOS_NLE_K}_r{r}"


def _gmrg_b(r):
    return f"gower_nle_finetune_hosx_gmrg_nla_m_vdq_k{HOS_NLE_K}_r{r}_ens9_e150_xo"


# Literal keys (run_remote.py:known_experiment scans for them).
kids_legacy_hos_experiments["glass_nle_pretrain_hosx_gmrg_nla_m_vdq_k8_r0"] = hosx_gmrg_nle_a_row(0)
kids_legacy_hos_experiments["glass_nle_pretrain_hosx_gmrg_nla_m_vdq_k8_r1"] = hosx_gmrg_nle_a_row(1)
kids_legacy_hos_experiments["glass_nle_pretrain_hosx_gmrg_nla_m_vdq_k8_r2"] = hosx_gmrg_nle_a_row(2)
kids_legacy_hos_experiments["glass_nle_pretrain_hosx_gmrg_nla_m_vdq_k8_r3"] = hosx_gmrg_nle_a_row(3)
kids_legacy_hos_experiments["glass_nle_pretrain_hosx_gmrg_nla_m_vdq_k8_r4"] = hosx_gmrg_nle_a_row(4)
kids_legacy_hos_experiments["gower_nle_finetune_hosx_gmrg_nla_m_vdq_k8_r0_ens9_e150_xo"] = hosx_gmrg_nle_b_row(0)
kids_legacy_hos_experiments["gower_nle_finetune_hosx_gmrg_nla_m_vdq_k8_r1_ens9_e150_xo"] = hosx_gmrg_nle_b_row(1)
kids_legacy_hos_experiments["gower_nle_finetune_hosx_gmrg_nla_m_vdq_k8_r2_ens9_e150_xo"] = hosx_gmrg_nle_b_row(2)
kids_legacy_hos_experiments["gower_nle_finetune_hosx_gmrg_nla_m_vdq_k8_r3_ens9_e150_xo"] = hosx_gmrg_nle_b_row(3)
kids_legacy_hos_experiments["gower_nle_finetune_hosx_gmrg_nla_m_vdq_k8_r4_ens9_e150_xo"] = hosx_gmrg_nle_b_row(4)
# Gower NPE ens9 pack (in-distribution NPE eval vs gower_npe_finetune_hos_nla_m_vdq_ens9_xo; not on the obs path).
kids_legacy_hos_experiments["gower_npe_finetune_hosx_gmrg_nla_m_vdq_ens9_xo"] = dict(
    copy.deepcopy(kids_legacy_hos_experiments["gower_npe_finetune_hosx_gmrg_nla_m_vdq_ens9"]),
    exclude_files=GOWER_HOSX_GMRG_OUTLIERS)
for _r in HOS_NLE_REPEATS:
    HOS_NLE_SOURCES[_gmrg_a(_r)] = _HOSX_GMRG_SOURCE
    HOS_NLE_SOURCES[_gmrg_b(_r)] = _HOSX_GMRG_SOURCE
assert set(HOS_NLE_SOURCES) <= set(kids_legacy_hos_experiments)

_GMRG_NPE = "gower_npe_finetune_hosx_gmrg_nla_m_vdq_ens9_xo"
# Waves (literal names: hosx_gmrg_train, hosx_gmrg_rs, hosx_gmrg_npe_eval, hosx_gmrg_nle_eval). GLASS NPE r -> Stage A r
# -> Stage B r (RUN_EVALUATION=0) by LABEL (one exp name, five repeats); the Gower NPE packs fill the free slots last.
HOS_TRAIN_WAVES["hosx_gmrg_train"] = (
    [dict(_ptask("train", _HOSX_GMRG_SOURCE, r, "glass_hosx"), label=f"glass_gmrg_r{r}") for r in HOS_NLE_REPEATS]
    + [dict(_ptask("embed", _gmrg_a(r), r, "glass_hosx", after=[f"glass_gmrg_r{r}"]), label=f"nle_a_gmrg_r{r}")
       for r in HOS_NLE_REPEATS]
    + [dict(_ptask("embed", _gmrg_b(r), r, "gower_hosx", after=[f"nle_a_gmrg_r{r}"], env={"RUN_EVALUATION": "0"}),
            label=f"nle_b_gmrg_r{r}") for r in HOS_NLE_REPEATS]
    + [dict(_ptask("train", _GMRG_NPE, r, "gower_hosx", after=[f"glass_gmrg_r{r}"]), label=f"npe_gmrg_r{r}")
       for r in HOS_NLE_REPEATS])
HOS_TRAIN_WAVES["hosx_gmrg_rs"] = (
    [dict(_ptask("eval", _gmrg_b(r), r, "gower_hosx"), label=f"rs_gmrg_r{r}_m{m}",
          args=["--mode", "recover-scalers", "--experiments", _gmrg_b(r), "--repeat-indices", str(r), "--members", str(m)])
     for r in HOS_NLE_REPEATS for m in range(9)]
    + [dict(_ptask("eval", _gmrg_b(r), r, "gower_hosx", after=[f"rs_gmrg_r{r}_m{m}" for m in range(9)]),
            label=f"verify_gmrg_r{r}", args=["--mode", "recover-scalers", "--verify-frames", "--experiments", _gmrg_b(r),
                                             "--repeat-indices", str(r)]) for r in HOS_NLE_REPEATS])
HOS_TRAIN_WAVES["hosx_gmrg_npe_eval"] = [
    dict(_ptask("eval", _GMRG_NPE, r, "gower_hosx"), label=f"npe_eval_gmrg_r{r}",
         args=["--mode", "misspec", "--misspec-base", _GMRG_NPE, "--variates", "gower_vdq_hosx", "--repeat-indices", str(r)])
    for r in HOS_NLE_REPEATS]
HOS_TRAIN_WAVES["hosx_gmrg_nle_eval"] = [
    dict(_ptask("embed", _gmrg_b(r), r, "gower_hosx", env={"RUN_TRAINING": "0"}), label=f"nle_eval_gmrg_r{r}")
    for r in HOS_NLE_REPEATS]
