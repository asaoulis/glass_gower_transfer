"""ARCHIVED: the STACKED-ENSEMBLE ablation (k=40 and k=16), closed 2026-08-25.

The question -- does concatenating the 5 foundation compressors (40-D) carry information a single
8-D compressor misses? -- was answered NO on both fidelities, so these rows are kept only for
reproducibility and should not be launched or extended. The full result summary is retained in the
banner below.

Not part of the BGP production analysis. Live config: config/kids_legacy_bgp.py.
"""
from config.kids_legacy_bgp import (
    kids_legacy_bgp_experiments,
    _BGP_SC8A1, _BGP_GOWER_NLA_M, _BGP_NLE_PROJECT, _GOWER_TEST_IDS, _NLE_REPEATS,
    _nle_pretrain_bgp, _nle_finetune, _nle_bake_repeat,
)

bgp_stack5_experiments = {}

# ##################################################################################################
# 🗄️  ARCHIVED 2026-08-25 (user) — THE STACKED-ENSEMBLE QUESTION IS CLOSED. DO NOT LAUNCH THESE ROWS.
#
# Question asked: does stacking the 5 foundation compressors (40-D concat) carry information a
# single 8-D compressor misses? Answer, on BOTH fidelities: essentially no.
#
#   * GLASS  — `glass_npe_pretrain_nla_m_bgp_stack5_k16` best val -5.3953 vs the five single
#     encoders -5.4014 / -5.3484 / -5.3341 / -5.3183 / -5.2681 (monitored mode="min", so more
#     negative is better): the 5-encoder stack lands SECOND, a hair behind the best single encoder.
#   * GOWER  — `gower_npe_finetune_nla_m_bgp_stack5_k16_ens9` (job 1349015, ens9, 1 repeat):
#     FoM 29.94 vs the single-encoder M4b baseline's 33.91 +/- 2.65, i.e. 1.5 sigma BELOW the
#     baseline mean. Per-parameter 1-sigma widths are indistinguishable and marginally tighter
#     (S8 0.02331 vs 0.0235; omega_m 0.03109 vs 0.0316; sigma_8 0.03731 vs 0.0385; w0 0.14470 vs
#     0.1478), and calibration is slightly better (TARP-9D 0.00190 vs 0.0021, Mahalanobis 2.913).
#     So the stack matches the MARGINALS but loses ~12% of the JOINT volume — worse parameter
#     correlations, no information gain. Caveat on the record: ONE repeat vs a FIVE-repeat mean,
#     and the +/-2.65 IS the seed spread, so 29.94 sits inside seed noise.
#   * Both results are consistent with the PCA independence fraction of 2.9% measured on the full
#     80 481-row embedding cache (stacked participation 2.58, k=8 @ 99%, k=19 @ 99.9%).
#
# ⚠️ THE ROWS BELOW ARE DELIBERATELY KEPT IMPORTABLE, NOT DELETED. `eval.py`, `misspec.py` and
# `load_best_model_and_build_posterior` all rebuild a config BY EXPERIMENT NAME, so deleting these
# entries would orphan the checkpoints still on disk under the same names. Archived == "closed, do
# not launch", not "removed". If the on-disk artefacts are ever reclaimed (see the checklist for the
# directory list), delete the dirs first and these rows last.
# ##################################################################################################
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
    bgp_stack5_experiments[f"bgpz8enc{_r}"] = {
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
bgp_stack5_experiments["glass_nle_pretrain_nla_m_bgp_stack5"] = _stack_pre


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
bgp_stack5_experiments["glass_nle_pretrain_nla_m_bgp_stack5_pcaprobe"] = _stack_probe


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
bgp_stack5_experiments["gower_nle_finetune_nla_m_bgp_stack5_ens9"] = _nle_bake_repeat(_stack_ft, 0)


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


bgp_stack5_experiments["glass_npe_pretrain_nla_m_bgp_stack5_k16"] = _stack_head("npe")
bgp_stack5_experiments["glass_nle_pretrain_nla_m_bgp_stack5_k16"] = _stack_head("nle")


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
bgp_stack5_experiments["glass_npe_pretrain_nla_m_bgp_stack5_k40"] = _stack_npe_k40


# --- The MATCHED CONTROL for the stack: a SINGLE-encoder NPE head -------------------------------
# The stacked k=16 NPE head cannot be read against the foundation's -5.4014..-5.2681 directly: the
# foundation trains encoder+flow END-TO-END WITH RandomEBPatchAugment, whereas any head on frozen
# cached embeddings sees ONE augmentation draw and therefore overfits (measured on 1348942: val
# turns at ~epoch 20 and decays -5.40 -> -0.69 by epoch 80). That regime difference, not the
# information content, could explain a gap either way.
#
# This row removes the confound: SAME pipeline, SAME training regime, SAME epochs/flow/LR, one
# encoder instead of five. The comparison that answers "how much extra information does stacking
# buy" is then stack@k16 MINUS this, both being best-checkpoint values under identical conditions.
#
# It reuses repeat r0's already-computed Stage-A cache, so it is a cache-hit job (minutes, no GPU,
# no fresh embedding pass). The cache path was VERIFIED by fetching its emb_val.pt before this row
# was written. Its whitener lands in this row's OWN run folder (rev 2c20b6d), so it cannot collide
# with the k=8 whitener the r0 NLE row persisted.
_npe_ctl_r0 = _nle_pretrain_bgp(_BGP_SC8A1, 0)
_npe_ctl_r0["inference_mode"] = "npe"
_npe_ctl_r0["embedding_cache_name"] = (
    "glass_nle_pretrain_nla_m_bgp_z8_r0/"
    "pretrain_ncosmoNone_0_kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1"
)
_npe_ctl_r0["reuse_embedding_cache"] = True
bgp_stack5_experiments["glass_npe_pretrain_nla_m_bgp_z8_r0_ctl"] = _npe_ctl_r0


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


bgp_stack5_experiments["gower_npe_finetune_nla_m_bgp_stack5_k16_ens9"] = \
    _stack_gower("npe", "glass_npe_pretrain_nla_m_bgp_stack5_k16")
bgp_stack5_experiments["gower_nle_finetune_nla_m_bgp_stack5_k16_ens9"] = \
    _stack_gower("nle", "glass_nle_pretrain_nla_m_bgp_stack5_k16")


# --- M5c (the `nla` variate Gower NLE finetune) is NOT written -----------------------------------
# It would need a Gower `nla` store (S2), and the dataset side's scope change of 2026-08-18 makes S1
# the only remaining sim. Writing a row against a store nobody plans to generate would be dead
# config that reads as ready. Add it if S2 is ever launched.


