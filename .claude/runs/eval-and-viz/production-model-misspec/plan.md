# Plan — eval-and-viz / production-model-misspec

<!-- RESUME:START -->
**RESUME HERE:** RESUMED after gaia hard reboot (root NVMe /dev/nvme0n1p3 returned EIO mid-session, killing local tool access; CLUSTER WORK UNAFFECTED). plan.md/log.md were written at ~68% BEFORE Phases 3-4 finished and are STALE -- trust this checkpoint over them.

CLUSTER STATE (all verified complete, queue empty):
* Phase 0/1 DONE. gb1p3 prebake completed pre-outage: 800/800 files. All six gower_bgp variate stores present: nla_m 40725, nla 15911, nla_z 15920, vd 15863, gb1p0 787, gb1p3 800.
* errno-28 blocker FIXED + pushed as rev 4b8fdf6 (local-tmp checkpoint staging wired into BOTH fit_model and embeddings_utils). Root cause: fsspec staged the ~106MB ckpt in the compute node /tmp via mkstemp(dir=None). Verified by a full clean 10-epoch run. Cluster checkout synced to this rev.
* Phase 2 DONE: 5 encoders trained (gower_npe_finetune_nla_m_bgp_z8_ens1), 10/10 epochs. Best (argmin) val_log_prob per repeat: r0 -4.8414, r1 -4.9271, r2 -4.7993, r3 -4.8429, r4 -5.0423 (reference r4_ens1 = -5.0118). r0 dir held stale ckpts but the FRESH one won the argmin (confirmed in eval log) => no contamination.
* Phase 3 DONE (job 1355127): all 30 misspec cells, fixed_lock=TRUE. Repeat-0 calibration: in-dist 0.0036 / nla 0.7299 / nla_z 0.7005 / gb1p0 0.0677 / gb1p3 0.0399 / vd 0.0035. Cross-repeat KL means: in-dist 0.223, gb1p0 0.2080, gb1p3 0.2099, vd 0.2232 => the KL axis separates the IA variates ~30x but does NOT separate vd/gb1p0/gb1p3 from in-distribution.
* Phase 4 DONE (job 1355130): all 30 [ood] cells. Single-encoder AUROC: nla/nla_z ~0.94 knn and ~0.84-0.86 cond_knn; gb1p0 0.51-0.57; gb1p3 0.50-0.56; vd at chance.

REMAINING WORK, IN ORDER:
1. Post-reboot local disk sanity (dmesg/smartctl nvme0n1); verify the 7 UNCOMMITTED artifacts/ scripts are intact+non-empty (gower_paired_dz.py, gower_null_selfcheck.py, gower_s*.py, plot_combined_misspec.py, plot_aia_vs_misspec.py, plot_misspec_tarp.py, offline_repeat_disagreement.py); regenerate any lost/truncated from plan.md specs; COMMIT them this time.
2. summaries fetch ALREADY SUCCEEDED (6.5MB, 40 npz under ml-checkpoints/.../gower_npe_finetune_nla_m_bgp_z8_ens1/summaries) -- verify, do NOT re-pull.
3. misspec fetch was PARTIAL when the disk died: delete the partial and re-fetch with --out_dir on /data (healthy /dev/sda), NOT the NVMe.
4. RE-RUN THE PHASE-4 NULL GATE with a cluster-aware test. The pre-crash 'failure' (8/10 row-level KS rejections) is a TEST ARTIFACT: each half has 1590 rows but only ~199 cosmologies (~8 correlated rotation/noise augmentations each), so row-level KS assumes ~710 independent draws when n_eff~89. Observed D 0.047-0.080 vs the correct 0.05 critical value 1.36/sqrt(89)~0.144 => all clear. mean_p 0.458-0.486 is ~1 SE from 0.5 at cosmology n; the 5 repeats share the same _idtest cosmologies so '5/5 same sign' is ONE observation. FIX: draw one row per cosmology from half B, KS that, repeat over many random draws. This edit to gower_null_selfcheck.py did NOT land pre-crash. Confirm the gate EMPIRICALLY before releasing Phase 6.
5. Phase 5 figures -- the matched-N floor band is MANDATORY (gb1p0/gb1p3 have only ~80 events and sit at/below the floor).
6. Phase 6: relaunch a FRESH Fable agent with the plan.md Phase-6 spec. The prior agent finished stage-1 familiarisation but died with the session and CANNOT be resumed. Hand it the citations it already found so it need not re-derive: Morningstar et al. AISTATS 2021 'Density of States Estimation for OOD Detection' arXiv:2006.09273 (primary); Bergamin et al. AISTATS 2022 arXiv:2203.01097; the mean-p validity reference; Lin & Sullivan 2009 AJHG for GLS weights. It correctly REJECTED Lee et al. 2018 arXiv:1807.03888 as circular under the no-OOD-fitting constraint.  _(updated 2026-09-07T13:29:01+00:00)_
<!-- RESUME:END -->

---

## Context

The BGP Gower arm needs a **production model-misspecification analysis**: five independently-seeded
Gower-finetuned NPE compressors, run over the six `gower_bgp` physics variates, producing (a) the
posterior-space miscalibration matrix (TARP / FoM / z-scores), (b) the summary-space OOD scores, and
(c) paper-ready figures that put the two on the same axes. The prior task
(`eval-and-viz/npe-nle-model-misspec`, see its `artifacts/REPORT.md`) settled the *method* — direct
NPE sampling for the simulated matrix, per-model summary-space OOD with a calibrated null — and left
three things owed: the `gb1p3` sc8a1 bake, an untested multi-model aggregation path, and the fact
that only **one** Gower-finetuned NPE encoder exists (`..._r4_ens1`, job 1347009).

This task closes all three and delivers the figures.

**Intended outcome.** A committed config row + 5 trained single-model encoders; a complete
5 encoder × 6 variate misspec matrix and matching summary-space matrix on the cluster; three
reproduced "money plots" in two x-axis versions (cross-repeat KL, and the new OOD score); and a
multi-encoder OOD score designed by a Fable subagent with measured acceptance criteria.

---

## Established facts (verified 2026-09-04 — do not re-derive)

| Thing | Fact |
|---|---|
| Gower BGP variate stores (gpu5) | `gower_bgp_{nla_m, nla, nla_z, gb1p0, nla_m_vd}_f16_sc8a1_fwhm4_lmin56_lcut1400` **all present**. `gower_bgp_gb1p3_f16_sc8a1_...` **ABSENT**. |
| gb1p3 raw source | `/share/gpu4/asaoulis/transfer_datasets/gower_mocks_gb1p3_novd_bgp` — **800 files, 3.9 G**, `output_193_out0_rot0_*.h5` ⇒ sim ids inside the lock. |
| Event counts (from REPORT §2.3b, lock ∩ on-disk, filter `[0,[0,1]]`) | in-dist ~998, `gower_nla` 1000, `gower_nla_z` 1000, `gower_vd` 996, `gower_gb1p0` **80**; `gower_gb1p3` expected **~80**. |
| Existing NPE row | `gower_npe_finetune_nla_m_bgp_z8_r4_ens1` (config/kids_legacy_bgp.py ~line 880) = `{**_npe_finetune_bgp(), "ensemble_repeats": 1, "repeat_indices": [4]}` wrapped in `_assert_final_summary_dim(..., 8, ...)`. Trained OK (job 1347009). It is the **source encoder** for the `_adapt` / A3 / A4 NLE ablation rows ⇒ **must not be touched**. |
| Naming mechanics | `_npe_finetune_z8` sets `match_num_cosmo=False` ⇒ training's *source-checkpoint* match is `_{r}`; the run dir is always `finetune_ncosmo300_{r}` (`apply_repeat_config` run_string). `misspec._load_experiment_config` forces `match_num_cosmo=True` for a list `max_trainval_cosmos` ⇒ **eval match_string = `ncosmo300_{r}`**, which substring-matches exactly one run dir. |
| The r4 re-run trap | `find_best_checkpoint` takes the **global min val_log_prob over all `.ckpt` in a run dir**, and `get_best_checkpoint` the global min over all matching dirs. Re-running r4 under the SAME experiment name would drop a second init's checkpoints into `finetune_ncosmo300_4/` next to the old ones and silently resolve whichever scored better. (`restart-same-repeat-is-a-reroll` memory.) |
| Smoke fixture | carries only `E_fwhm8_lmin50_lcut1400`; this row is `eb_map_variant=None` (bare `E`) ⇒ the gate **false-fails**. `--skip-smoke` required. |
| Moments npz | `_save_posterior_moments` writes `mean/std/theta0/z` — **diagonal only**. `plot_combined_misspec.py` needs a per-event 3×3 covariance and therefore currently needs the FULL sample npz. |
| Gatekeeper `--args` charset | tokens must match `[A-Za-z0-9][A-Za-z0-9_.-]*` or `--[a-z-]*`, ≤32 tokens; no `/`, `*`, `=`. All commands below comply. |

---

## Approach

**Design decision 1 — ONE new experiment name, five repeats, `ensemble_repeats=1`.**
Add `gower_npe_finetune_nla_m_bgp_z8_ens1` with `repeat_indices=[0,1,2,3,4]`. Reasons:
* A fresh name means a virgin `checkpoints/<exp>/` tree — no stale-checkpoint hazard for r4 (see the
  trap above), and the proven `_r4_ens1` row stays intact for the NLE ablation that depends on it.
* `misspec.py` / `summaries.py` take **one** `--misspec-base` and loop `--repeat-indices`. With five
  separate `_r{i}_ens1` names the in-job cross-repeat disagreement (`misspec_repeat_disagreement_*`)
  — the x-axis of the like-for-like plots — could never fire.
* `ensemble_repeats=1` ⇒ run dirs are plain `finetune_ncosmo300_{r}` (no `_ens{j}`), so
  `get_best_checkpoint("ncosmo300_{r}")` resolves to exactly one folder.

**Design decision 2 — the row is defined by construction, not by copy.**
`{**_npe_finetune_bgp(), "ensemble_repeats": 1, "repeat_indices": [0,1,2,3,4]}` reuses the same
factory as the proven r4 row (so the `_RESNET_MAPKW` trap cannot re-open) and goes through
`_assert_final_summary_dim(..., 8, ...)`.

**Design decision 3 — `--test-id-source heldout` (the default), no `--max-test-files`.**
The experiment pins `fixed_test_sim_ids = gower_test_ids.json` (200 ids), so `derive_test_id_pool`
returns the lock and every variate is scored on lock ∩ on-disk — already a strict held-out set.
`shared` would intersect all six variates down to the gb probes' ~40 cosmologies for *everyone*, a
12× statistics loss for no gain. `--max-test-files` truncates by *sorted sim_id*, which biases to
low ids; omit it. (If ever used, use the identical value on the summaries run so the per-event join
is over the same events.)

**Design decision 4 — `--num-samples 4000` + add `cov` to the moments npz.**
TARP/FoM are insensitive above ~2k draws; 4000 halves the sample-npz fetch (≈4–5 GB total vs
≈11 GB at 10 000) while staying above the REPORT's 2000-draw working point. Separately, add a
`cov [N,D,D] float32` field to `_save_posterior_moments` (≈0.3 MB/variate) so the combined figure's
3-D Mahalanobis can be computed from the *small* npz — making every future analysis a KB-scale
fetch. Purely additive, inside the existing try/except.

**Design decision 5 — GPU tiers.** Training: **a100** (dedicated card, no dead-L40S lottery, and the
re-roll hazard makes the l40s twin-submit trick unsafe for training). Misspec/summaries: **l40s
preferred** (the stores are on `/share/gpu5`, whose local disk *is* the l40s node's — the run is
I/O-bound and this is the ~4.5× factor), using the twin-submit protocol with an immediate cancel of
the redundant survivor; **a100** as the no-fuss fallback. **NEVER v100** for either
(`encoder-finetune-never-v100`, `npe-eval-ooms-on-v100`).

**Files touched (production):**
* `config/kids_legacy_bgp.py` — one new row (additive; do not modify the `_r4_ens1` row).
* `src/ml/eval/misspec.py` — `_save_posterior_moments`: add `cov`. Additive.
* `EVAL_RECIPES.md` — add the two recipes used here.
* *(no `src/cosmology/`, `src/KiDS/systematics.py`, `src/KiDS/tomo.py` — none is involved.)*

**Analysis code (task artifacts, not production):** ports of
`.claude/runs/eval-and-viz/first-npe-misspecification/artifacts/{plot_aia_vs_misspec.py,
plot_combined_misspec.py,plot_misspec_tarp.py,offline_repeat_disagreement.py,misspec_zscores.py}`
and `.claude/runs/eval-and-viz/npe-nle-model-misspec/artifacts/join_ood_posterior.py`, copied into
`.claude/runs/eval-and-viz/production-model-misspec/artifacts/`.

---

## Phase 0 — config row + the `cov` field (local, no cluster)

- [x] Read `config/kids_legacy_bgp.py` lines ~798–890 to re-confirm `_npe_finetune_bgp()` and the
      `_r4_ens1` row are as described above.
- [x] Add, directly under the existing A1 row, a new block (keep the A1 row **unchanged**):
      `kids_legacy_bgp_experiments["gower_npe_finetune_nla_m_bgp_z8_ens1"] =
      _assert_final_summary_dim({**_npe_finetune_bgp(), "ensemble_repeats": 1,
      "repeat_indices": [0, 1, 2, 3, 4]}, 8, "gower_npe_finetune_nla_m_bgp_z8_ens1")`
      with a comment stating (i) why a fresh name (the `find_best_checkpoint` global-min trap),
      (ii) that `_r4_ens1` is deliberately left alone because the `_adapt`/A3/A4 NLE rows use it as
      `--sources`, and (iii) that all five repeats live in ONE checkpoints dir so `--misspec-base`
      can loop them.
- [x] **Equality check (this replaces the smoke gate).** Run locally:
      import both rows, assert
      `{k: v for k, v in new.items() if k != "repeat_indices"} == {k: v for k, v in r4.items() if k != "repeat_indices"}`
      and `new["ensemble_repeats"] == 1`, `new["max_trainval_cosmos"] == [300]`,
      `new["model_kwargs"]["map_kwargs"] is _RESNET_MAPKW`-equal,
      `new["eb_map_variant"] is None`, `new["fixed_test_sim_ids"].endswith("gower_test_ids.json")`.
      ✅ = the new row is byte-identical to a row that already trained successfully.
- [x] Also assert the row is visible to the eval path: `from src.ml.eval.misspec import
      _load_experiment_config; c = _load_experiment_config("gower_npe_finetune_nla_m_bgp_z8_ens1")`
      → `c.match_num_cosmo is True`, `c.max_trainval_cosmos == 300`, `len(c.cosmo_param_names) == 9`.
- [x] `src/ml/eval/misspec.py::_save_posterior_moments`: add
      `payload["cov"] = (per-event sample covariance, [N, D, D] float32)` computed from the already
      loaded `samp` (centre once, `np.einsum("sni,snj->nij", c, c) / (S - 1)`). Keep it inside the
      existing `try/except` so a failure still cannot break an eval. Update the docstring.
- [x] Local sanity of that edit: tiny synthetic `(S=200, N=5, D=3)` array → `cov` matches
      `np.cov(samp[:, i, :], rowvar=False)` for each event to 1e-6.
- [x] `git add config/kids_legacy_bgp.py src/ml/eval/misspec.py && git commit` on
      `kids-preparation`; `git push origin kids-preparation` (SSH push URL — see CLAUDE.md).
      **Verification:** `git log origin/kids-preparation -1` shows the new rev.

## Phase 1 — the owed `gb1p3` sc8a1 prebake (cluster; independent of Phase 0/2)

- [x] `python .claude/cluster/run_remote.py --dry-run prebake --src-datasets-root gpu4
      --src-dir gower_mocks_gb1p3_novd_bgp --eb-variant sc8_fwhm4_lmin56_lcut1400 --noise-norm rand
      --dtype float16 --out-dir gower_bgp_gb1p3_f16_sc8a1_fwhm4_lmin56_lcut1400`
      — inspect, then run for real (drop `--dry-run`).
      **Do NOT pass `--keep-variant-tag`** (the chain is `eb_map_variant=None` ⇒ bare `E` groups).
- [x] **Verification:** `run_remote.py logs --name prebake` (or the job name printed at submit) shows
      a final `ok=` count. Expect `ok=800` (or a small shortfall from dropped truncated files — note
      the exact number in `log.md`). Then
      `run_remote.py data-ls --rel gower_bgp_gb1p3_f16_sc8a1_fwhm4_lmin56_lcut1400 --n 5`
      → `*.h5 files: <ok count>`, and the file size is ≈1/32 of the raw (~0.15–0.2 MB each).
- [x] Confirm the other five stores once more with `data-ls` (already verified today; re-check only
      the file counts, they are the `n` the misspec run will report).

## Phase 2 — train the 5 single-model Gower NPE encoders (cluster, GPU)

**Gate:** Phase 0 committed AND pushed.

- [x] `python .claude/cluster/run_remote.py sync` (pre-checks the rev is on origin).
- [x] `python .claude/cluster/run_remote.py avail` — read the a100 free-RAM column
      (a100 nodes routinely show a free GPU with only 30–53 G free RAM).
- [x] `python .claude/cluster/run_remote.py --dry-run train --exp gower_npe_finetune_nla_m_bgp_z8_ens1
      --gpu a100 --repeat-indices 0,1,2,3,4 --wall_h 12 --skip-smoke --queue-anyway` then submit.
      ONE sequential job = zero same-index collision risk. **If it PENDs on RAM**, resubmit with
      `--mem-gb 48` — `defaults.train.mem_gb=64` exceeds what a100 nodes typically show free.
      *(Accelerator option: split into ≤5 jobs, each with a **distinct** `--repeat-indices <r>`.
      Distinct indices ⇒ distinct run dirs ⇒ safe. NEVER submit the same index twice, and never
      twin-submit a training job — two survivors write two inits into one run dir.)*
- [x] **Confirm the jobid appears in `squeue`** (`run_remote.py status`). A silently-dropped submit
      looks exactly like a queued one.
- [x] **Pre-step — capture the exact expected load-block strings.** The strict whole-model
      `checkpoint_path` loader prints a *different* block from the prefix/embeddings loaders, and
      the transcript of job 1347009 could not be read back that far (its log is tqdm-flooded and
      `logs --n` returns the TAIL). Before submitting, pull the head of that job's log
      (`run_remote.py logs --name gower_npe_finetune_nla_m_bgp_z8_r4_ens1 --n 4000`, or read the
      file via `view`) and **paste its actual checkpoint-load lines into this plan** as the expected
      strings. Do not assert on strings borrowed from another loader.
- [x] **Verification, per repeat, from `run_remote.py logs --name gower_npe_finetune_nla_m_bgp_z8_ens1`:**
  - `[Repeat r | Ensemble 0] split_seed=... ensemble_seed=None` header present for each r in 0..4;
  - `Will try to use checkpoint:` resolves into
    `checkpoints/kids_legacy_hybrid_nla_m_bgp_z8_resnet_sc8a1/pretrain_ncosmoNone_{r}/...`;
  - **NO** `[get_best_checkpoint] N run folders matched` warning (a warning = an ambiguous source);
  - the load block matches the strings captured in the pre-step (this is the `_RESNET_MAPKW`
    canary — a UNet built against ResNet weights fails loudly under the strict loader);
  - a `finetune_ncosmo300_{r}` run dir is created and a
    `checkpoint-epoch=*-val_log_prob=*.ckpt` lands in it.
- [x] **Cross-check against the MEASURED reference (job 1347009, read 2026-09-04):**
      run `gower_npe_finetune_nla_m_bgp_z8_r4_ens1/finetune_ncosmo300_4`, W&B project
      `gower-finetuning`, 10 epochs (epoch 0..9), **150 train batches + 38 val batches per epoch,
      ~2 min 09 s per epoch**, final `val_log_prob = -4.94211`, `train_log_prob = -5.00387`,
      `test_log_prob = -4.83809`; `val_log_prob` still improving at epoch 9. The five new repeats
      should land within a few tenths of a nat of **-4.94**. Read each repeat's value straight off
      its checkpoint filename (`-val_log_prob=<x>.ckpt`) or W&B. A repeat a nat+ worse is a bad
      init, not a bug — note it and decide whether to re-roll it **under a fresh name**, never
      in place (trap 1).
- [x] `run_remote.py fetch --exp gower_npe_finetune_nla_m_bgp_z8_ens1 --rel . ` is NOT needed here
      (checkpoints are tens of GB and stay on the cluster).

## Phase 3 — the misspec matrix, 5 encoders × 6 variates (cluster, GPU)

**Gate:** Phase 1 `ok=` verified AND all five Phase-2 run dirs hold a checkpoint.

- [ ] Submit ONE job so the in-job cross-repeat disagreement fires:
      `python .claude/cluster/run_remote.py eval --gpu l40s --wall_h 24 --args "--mode misspec
      --misspec-base gower_npe_finetune_nla_m_bgp_z8_ens1 --variates gower_bgp
      --repeat-indices 0 1 2 3 4 --num-samples 4000"`
      (14 tokens, charset-clean). **l40s twin protocol:** submit twice back-to-back; after ~10 min
      check `status` — if both are RUNNING, `cancel` one immediately (two survivors race-write the
      same output paths); if one died with an ECC/uncorrectable error, keep the survivor.
      Fallback if l40s is unavailable/flaky: `--gpu a100 --wall_h 36` (NFS reads ≈4.5× slower).
- [ ] **Verification from `run_remote.py logs --name run` (eval jobs log as `eval_run_<jobid>`; the
      matcher needs `run`, and it also matches `sample_run`/`plot_run` — filter by jobid):**
  - `[misspec] setup 1/4..4/4` all print (a crash before these is unlocalisable otherwise);
  - per repeat: `match_string=ncosmo300_{r}` and `Best checkpoints found:` lists **exactly one**
    path, under `gower_npe_finetune_nla_m_bgp_z8_ens1/finetune_ncosmo300_{r}/`;
  - per variate: `[misspec] <name>: n_test=... (N cosmologies, fixed_lock=True)`. **`fixed_lock=True`
    on all six** — a `False` means the fallback fired (no lock overlap) and that variate is not
    held out. Expect n≈1000 for nla_m / nla / nla_z / vd and n≈80 for gb1p0 / gb1p3;
  - `missing_params=['b_ia'] exclude_params=['a_ia']` on `gower_nla` / `gower_nla_z` only;
  - `[misspec] <name>: DONE cal_full=... cal_om_s8_w0=... fom=... dMI=...` for all 6 × 5 = 30 cells;
  - `n_dropped_nonfinite` is reported — a large drop count on `nla`/`nla_z` is expected
    (25–28 % in the first-NPE run) and must be carried into the figure captions;
  - `[misspec] <name>: repeat disagreement over 5 repeats, kl_mean=...` for each variate
    ⇒ `misspec_repeat_disagreement_ncosmo300_0_..._4.{npz,json}` written.
- [ ] If the job dies mid-way, re-submit; completed variates are re-computed but nothing is
      corrupted. If only one variate failed (e.g. gb1p3 late), recover with
      `--variate-names gower_gb1p3` — then the in-job disagreement will **not** rerun, so use the
      ported `offline_repeat_disagreement.py` for that variate.
- [ ] Fetch: `run_remote.py fetch --exp gower_npe_finetune_nla_m_bgp_z8_ens1 --rel misspec`
      (≈4–5 GB at 4000 samples; if that is too heavy, fetch per-variate with
      `--rel misspec/<variate>`). **Verification:** locally, 6 variate dirs × 5
      `misspec_evaluation_results_ncosmo300_{r}.json` + 5 tarp jsons + 5 samples npz + 5 moments npz
      (each moments npz now carrying `cov`), plus one `misspec_repeat_disagreement_*.npz` per variate.

## Phase 4 — the summary-space matrix on the SAME model (cluster, GPU)

**Gate:** Phase 2 done. Best run *after* Phase 1 so gb1p3 is included in one pass.

- [ ] `python .claude/cluster/run_remote.py eval --gpu l40s --wall_h 12 --args "--mode summaries
      --misspec-base gower_npe_finetune_nla_m_bgp_z8_ens1 --variates gower_bgp
      --repeat-indices 0 1 2 3 4 --max-train-files 20000"`
      (default `--test-id-source heldout` ⇒ the **same events** as Phase 3, so the per-event join is
      exact). Same l40s twin protocol.
- [ ] **Verification:**
  - `[summaries] repeat=r match_string=ncosmo300_{r}` and `1 encoder(s)` per repeat (this row is
    ens1 — if it ever prints >1 the ensemble path is in play and the `_m{j}` tags appear);
  - `encoding TRAIN cloud (20000 files)` and `encoding ID held-out test split (~1590 files)`;
  - one `[ood] <variate>@r{r}: knn_auroc=... cond_knn_auroc=... c2st_auroc_gbm=...` line per cell
    (30 lines);
  - `_train/`, `_idtest/` and 6 variate dirs each hold `summaries_ncosmo300_{r}.npz`.
- [ ] **Null self-check — the SPLIT-HALF protocol (do NOT do the naive version).**
      `summaries.py` writes `_idtest` with **no `ood_*` fields** (the "self-check" comment in the
      code has nothing behind it), and re-fitting `OODReference` with `z_id = _idtest` and then
      scoring `_idtest` gives p = rank/(n+1) — **uniform by construction, a tautology**. The real
      check, run locally after the fetch, per repeat:
      split `_idtest` in half by `sim_id` parity (never by row order — augmentations of one
      cosmology are correlated), fit `OODReference.fit(z_train, z_id=half_A, theta_train=..., theta_id=...)`,
      score **half B**, and apply `src.ml.eval.ood.null_uniformity` to the resulting p-values.
      **Pass = KS p > 0.05 on all five repeats, for `knn` and `cond_knn`.** This is the REPORT §2.4
      "unverified aggregation path" gate in its ens1 form, and it is the calibration the Phase-6
      combiner is built on top of — if it fails, stop and diagnose before Phase 6.
- [ ] `run_remote.py fetch --exp gower_npe_finetune_nla_m_bgp_z8_ens1 --rel summaries`
      (small — z is 8-D; tens of MB).

## Phase 5 — reproduce the three money plots, x = cross-repeat KL (local)

Copy the four scripts into this task's `artifacts/` and port them. Ports needed in **all** of them:
the `COLORS` / `LABELS` / `IN_DIST` / `ORDER` dicts gain the six `gower_bgp_nla_m`, `gower_nla`,
`gower_nla_z`, `gower_gb1p0`, `gower_gb1p3`, `gower_vd` names; match strings become
`ncosmo300_0..4` (not `_0..4`); `PARAMS` is already the 9-param `nla_m` vector
(`omega_m, sigma_8, w0, mnu, h, ns, ombh2, a_ia, b_ia`) — unchanged.

- [ ] Port `offline_repeat_disagreement.py` → `gower_offline_disagreement.py`. Only needed as a
      fallback (Phase 3 computes the KL in-job) and to emit the
      `gower_misspec_cal_vs_disagreement.json` table that `plot_combined_misspec.py` consumes.
      Run: `--root ml-checkpoints/gower_npe_finetune_nla_m_bgp_z8_ens1/misspec
      --matches ncosmo300_0 ncosmo300_1 ncosmo300_2 ncosmo300_3 ncosmo300_4
      --out-prefix gower_misspec`.
- [ ] Port `plot_misspec_tarp.py` → `gower_tarp_coverage.png`.
      `--root .../misspec --match ncosmo300_0` for the like-for-like single-repeat figure; then a
      5-repeat variant that plots the mean ± sd of the ECP curves across repeats (small addition:
      loop the matches and average `ecp_bootstrap`). Title names the experiment + "fixed 200-id test
      lock". **Check:** the in-dist curve sits on y=x; the `nla`/`nla_z` curves depart strongly.
- [ ] Port `plot_combined_misspec.py` → `gower_misspec_combined.png`. Change
      `per_event_mahalanobis` to read `cov` from `misspec_posterior_moments_<m>.npz` when present
      (falling back to the full samples npz), which makes the figure reproducible from KB-scale
      files. Keep the two-anchor log-axis alignment. Add the six variate colours/labels.
      **Check:** the in-dist diamond sits at the χ₃ reference line; the OOD diamonds land on the
      tail of their own event clouds.
- [ ] Port `plot_aia_vs_misspec.py` → `gower_aia_vs_kl.png`, run with `--kl-mode`
      `--variates gower_nla gower_nla_z --indist gower_bgp_nla_m --match ncosmo300_0`.
      `AIA_BOX = (4.48, 7.0)` is unchanged (still the NLA-M training box); **keep the original's
      caveat** that `a_ia` under `nla`/`nla_z` is a different parametrisation *and* the caveat that
      the ~25 % intractable events are absent, so the trend is a lower bound.
- [ ] Port `misspec_zscores.py` → `gower_misspec_zscores.py` and run it for the z-score report +
      the **matched-N calibration floor**: `gower_gb1p0` and `gower_gb1p3` have only ~80 events, and
      the REPORT's floor at N≈100 is cal_full≈0.05 / cal3≈0.09 — i.e. comparable to a real signal.
      Build the floor band by subsampling the in-dist events **matched on CLUSTER STRUCTURE, not
      row count**: `gower_gb1p0`/`gower_gb1p3` are **40 cosmologies × 2 rows**, so draw 40 in-dist
      cosmologies and 2 rows from each (≥200 draws). Drawing 80 random in-dist ROWS would span ~80
      cosmologies and so carry ~8× the effective sample, which risks a floor band that is too
      narrow. **Measured outcome (2026-09-07): the effect was small** — row-matched
      0.0535 ± 0.0089 vs cluster-matched 0.0560 ± 0.0105 — so the correction is right in
      principle but changed no verdict here; what actually mattered was the DRAW COUNT (200 vs a
      2-draw probe, which understated the floor and wrongly put `gb1p0` above its 95th
      percentile). Put the resulting band on the two gb diamonds in the combined figure.
      **Without this the gb points are not interpretable.**
- [ ] Check whether `gower_gb1p0` and `gower_gb1p3` are **paired** (same `--rng-seed`): compare
      their `sim_ids`/`aug_ids` arrays in the samples npz. If they match, add the paired
      Δz = z(b_g=1.3) − z(b_g=1.0) statistic (as on the GLASS ladder) — it removes cosmic variance
      and is far more sensitive than either absolute z at N=80. **Also report the shift in PHYSICAL
      units** (`gower_paired_physical.py`): Δz is in units of each event's own posterior width, so
      it is not comparable to a published bound such as DES Y3's |Δθ(Ω_m)| < 0.007, and posterior
      width and bias move together (`vare-fragility-tracks-sharpness`).
- [ ] Save all figures + the JSON tables into `artifacts/`; record the numbers in `log.md`.

---

## AMENDMENT 2026-09-07 (post-reboot resume) — cluster-aware statistics

`_idtest` is 1590 rows over **199 cosmologies** (~8 correlated augmentations each), and the two
`gower_gb*` variates are **40 cosmologies × 2 rows**, not 80 independent events. Every uniformity
test, confidence interval and matched-N floor in Phases 4–6 must therefore work at the
**cosmology** level. The Phase-4 gate was rewritten accordingly and **PASSES** 10/10 cells with a
positive control firing 10/10 (`artifacts/phase4_null_gate_PASSED.txt`, commit 4d63d86); the
row-level KS it originally specified is ~2.8× too strict and rejects a calibrated null. Phase 6's
constraint 4 and acceptance criteria 1–2, and Phase 5's matched-N floor, are amended above.


## Phase 6 — multi-encoder OOD metric (Fable 5.1 subagent)

**HARD GATE: only launch after all five Phase-2 encoders exist and Phase 4's summaries are fetched
and null-verified.** The agent must have real per-encoder data to design against.

- [ ] Launch ONE `Agent` with `model: "fable"` and this narrow spec:

  > **Objective.** Design a metric that combines the **five** encoders' summary-space measurements
  > into a single, more discriminative per-event OOD score for the Gower BGP arm.
  >
  > **Inputs (read-only, already on disk).**
  > `.claude/runs/eval-and-viz/production-model-misspec/artifacts/summaries/` — per repeat
  > `r ∈ {0..4}`: `_train/summaries_ncosmo300_{r}.npz` (20 000-mock reference cloud: `z[N,8]`,
  > `theta[N,9]`), `_idtest/summaries_ncosmo300_{r}.npz` (the null), and
  > `<variate>/summaries_ncosmo300_{r}.npz` for the six `gower_*` variates (with the existing
  > per-event `ood_*` score/p-value fields). Posterior side for the association test:
  > `.../misspec/<variate>/misspec_posterior_moments_ncosmo300_{r}.npz` (`z`, `mean`, `std`, `cov`,
  > `test_files`). Existing implementation to adapt: `src/ml/eval/ood.py` (`OODReference`,
  > `TrainWhitener`, `ConditionalResidualModel`, `empirical_pvalues`, `auroc`,
  > `combine_pvalues_across_models`) and `src/ml/eval/summaries.py`; the design rationale and the
  > measured single-encoder numbers are in
  > `.claude/runs/eval-and-viz/npe-nle-model-misspec/artifacts/REPORT.md` **Part 2**.
  >
  > **Non-negotiable constraints.**
  > 1. **No circularity.** Anything fit (whitener, regressor, combiner weights, calibration) may use
  >    ONLY in-distribution data — each encoder's `_train` cloud and its `_idtest` null. The six
  >    variate npz are **evaluation-only**; a combiner tuned on variate labels is disqualified.
  > 2. **Never pool raw `z` across encoders** — the five summary spaces are not commensurable.
  >    Inputs to the combiner must be per-encoder *calibrated* quantities (p-values, or scores in
  >    that encoder's own train-whitened frame).
  > 3. Truth-free at prediction time is preferred; a θ-conditional variant is allowed if it states
  >    the real-data form (θ from posterior draws, posterior-predictive).
  > 4. **Second-stage recalibration is MANDATORY.** Any combined statistic — yours or a baseline —
  >    must be turned into an empirical p-value against *its own* null distribution, and that null
  >    must come from a **held-out half of `_idtest`**, split at the **COSMOLOGY level** (never by
  >    row order — the rows are ~8 correlated augmentations of each cosmology). Use a **fixed,
  >    seeded RANDOM cosmology split**, and average results over many such splits; do **not** use
  >    the even/odd `sim_id` parity split — it is one structured draw that lands at the ~3rd–28th
  >    percentile of the random-split reference. Reference implementation:
  >    `artifacts/gower_null_selfcheck.py` (read it before designing the protocol).
  >    Uniformity is then evaluated on the *other* half. Reason: five independent U[0,1]
  >    p-values averaged give **Bates(5)**, concentrated near 0.5, not uniform — so a raw mean-p is
  >    *guaranteed* to fail a uniformity test, and Fisher's χ² assumes an independence the five
  >    members do not have (they share the data). The held-out null is exactly what corrects both.
  >    Whatever fitting the combiner needs (weights, bandwidths) uses the FIRST half only.
  > 5. Read-only outside your own artifacts dir. **Do not edit `src/`, `config/`, or any file under
  >    `src/cosmology/`, `src/KiDS/systematics.py`, `src/KiDS/tomo.py`.**
  >
  > **Deliverable — TWO options.**
  > (a) **Literature-grounded**, with a real citation (e.g. ensemble/Mahalanobis-style deep OOD
  >     detection). You may spawn **at most ONE** short-lived Sonnet subagent to find and verify the
  >     citation. Do not pre-commit to a particular paper — report what you find.
  > (b) **Your own design**, intended to beat (a); state the mechanism you expect to exploit
  >     (e.g. that different encoders fail on different directions, so the *disagreement structure*
  >     across encoders carries information a per-encoder p-value does not).
  >
  > **Baselines to beat**, each put through the SAME second-stage recalibration (constraint 4) so
  > the comparison is like-for-like: `combine_pvalues_across_models(..., "mean")` and `"fisher"` on
  > the per-encoder `cond_knn` p-values, and the best single-encoder `cond_knn` score.
  >
  > **Acceptance criteria (report all three, per option, as a table):**
  > 1. **Calibrated null — and it MUST be tested cluster-aware.** `_idtest` is 1590 rows over
  >    only 199 cosmologies, so a held-out half has ~710 rows but only ~89 INDEPENDENT draws. A
  >    row-level `kstest` on all ~710 rows uses a critical value of 1.36/√710 ≈ 0.051 where the
  >    honest one is 1.36/√89 ≈ 0.144 — **~2.8× too strict, and it WILL reject a perfectly
  >    calibrated null.** This is measured, not hypothetical: it failed 8/10 cells on the
  >    single-encoder gate before the fix. **Do not use it.** Instead report, for your metric and
  >    for both baselines:
  >      (a) `mean_p` averaged over ≥50 random cosmology-level splits, which must sit within one
  >          **per-split sd** of 0.5 (tolerance is the per-split sd, NOT sd/√n_splits — the splits
  >          re-use the same rows, so more splits buy no independence);
  >      (b) the median KS statistic over draws of **one row per cosmology**, which must be below
  >          1.36/√n_cos (the H0 median is 0.83/√n_cos — report it alongside).
  >    All three should pass, since recalibration guarantees it up to finite-sample noise; a
  >    FAILURE means the two halves are not exchangeable and is a bug to fix, not a verdict on the
  >    metric. For calibration, the single-encoder gate measured (a) 0.4954–0.5063 and
  >    (b) D 0.084–0.110 vs crit 0.144 — see `artifacts/phase4_null_gate_PASSED.txt`.
  >    Include a **positive control** (e.g. `gower_nla`, D ≈ 0.77–0.82) so a pass cannot come from
  >    a powerless test.
  > 2. **AUROC on the known variate ladder** (null vs variate, all six): ≥ both baselines on every
  >    variate, with the gain measured where it matters — `gower_gb1p0`, `gower_gb1p3` and
  >    `gower_vd` sit at 0.52–0.58 for a single encoder (`nla`/`nla_z` are already ~0.94, so a gain
  >    there is uninformative). **Confidence intervals MUST bootstrap by COSMOLOGY, never by row**
  >    — resample `sim_id`s and take all of each one's rows. `gower_gb1p0`/`gower_gb1p3` are only
  >    **40 cosmologies × 2 rows** (n=80 rows is NOT n=80 independent events), which puts the
  >    single-encoder AUROC sd at roughly 0.06. **Do not chase differences smaller than that**; an
  >    apparent gain inside the CI is not a result, and saying so is the correct outcome.
  > 3. **Per-event Spearman ρ** between −log p_combined and the posterior |z| of the truth for
  >    `omega_m` and `sigma_8` (joined on `test_files`): ≥ the best single-encoder `cond_knn` ρ.
  >
  > **Output.** (i) A short design note (mechanism, citation for (a), the acceptance table, and an
  > honest statement of which option wins and where it does not). (ii) ONE script in this task's
  > `artifacts/` that writes a per-event npz with `combined_score`, `combined_p` and `test_files`
  > for each variate + `_idtest`, so the plotting scripts can consume it uniformly. (iii) Any
  > discrimination figure the design motivates (ROC curves / AUROC bars).

- [ ] Review the Fable output against the three acceptance criteria yourself before using it.
      Criterion 1 is a **plumbing** check (recalibration makes it near-automatic); the real verdict
      is criteria 2 and 3. If neither option beats the recalibrated mean-p baseline on the HARD
      variates (`gower_gb1p0`, `gower_gb1p3`, `gower_vd`) and on the per-event Spearman, the honest
      outcome is "the recalibrated mean-p combiner remains the recommendation" — report that rather
      than tuning until something wins.
- [ ] Pick the winner (or ship both, labelled) and record the decision + numbers in `log.md`.

## Phase 7 — the final figure set (local)

- [ ] Port `join_ood_posterior.py` → `gower_join_ood_posterior.py` (`--exp
      gower_npe_finetune_nla_m_bgp_z8_ens1 --summaries ml-checkpoints/<exp>/summaries
      --moments ml-checkpoints/<exp>/misspec --repeats 0 1 2 3 4 --match-fmt ncosmo300_{r}`) and
      produce the per-variate OOD tables + `ood_overview.png` for this arm.
- [ ] **Set B — the same three money plots with the single-model OOD score on x.** Add an
      `--score-source {kl,ood}` switch to the ported `plot_combined_misspec.py` and
      `plot_aia_vs_misspec.py`. The statistic replaced is the per-event cross-repeat KL: in
      `plot_combined_misspec.py` that is the **x** axis; in `plot_aia_vs_misspec.py --kl-mode` it is
      the **y** axis (x stays true `A_IA`). New statistic = per-event `−log p` of the recalibrated
      repeat-averaged `cond_knn` p-value; dataset marker at the variate mean. Everything
      else — colours, KDE regions, axis alignment, the χ₃ reference — is unchanged so the two sets
      are visually comparable.
- [ ] **Set C — the same three with the Fable multi-encoder score on x**, consuming the npz from
      Phase 6, plus the discrimination figures the design recommends.
- [ ] Paper style pass on all nine figures: minimal text, one clear title, no jargon in axis labels
      that the paper does not define, consistent colours across all three sets, dpi ≥ 160.
- [ ] Write a short results section in `artifacts/REPORT.md` (tables + which figures say what) —
      **not** a process log; the log lives in `log.md`.
- [ ] `EVAL_RECIPES.md`: add the two exact `run_remote.py eval --args` recipes from Phases 3–4 and
      note the `--num-samples 4000` fetch-size rationale. Commit + push.

---

## Cost / wall-clock estimate

| Step | Where | Est. wall-clock | Notes |
|---|---|---|---|
| Phase 0 config + `cov` | local | 30–45 min | no GPU |
| gb1p3 prebake (800 files, 3.9 G) | CORES64 | 10–30 min | ~1/32 output size |
| 5 × NPE finetune (10 ep, ncosmo300) | a100, 1 job | **~2.5–4 h** | MEASURED on job 1347009: 150 train + 38 val batches/epoch at **~2 min 09 s/epoch** ⇒ ~22 min training + startup per repeat. Budget `--wall_h 12`. |
| misspec 5 × 6 (≈4 200 events/repeat, 4 000 draws) | l40s (data-local) | 3–6 h | a100 fallback 12–24 h (NFS ≈4.5× slower); `--wall_h 24` |
| summaries 5 × 6 (encoder forward only) | l40s | 1–2 h | no posterior sampling |
| fetch misspec | local | 15–40 min | ≈4–5 GB at 4 000 draws (≈11 GB at 10 000) |
| fetch summaries | local | < 5 min | tens of MB |
| plot ports + matched-N floor | local | 3–4 h | |
| Fable metric + review | agent + local | 2–4 h | |
| Final figure set + report | local | 3–4 h | |

Cluster jobs submitted: **1 prebake + 1 train + 1 misspec (+1 twin, cancelled) + 1 summaries
(+1 twin, cancelled)** = 3 productive jobs plus up to 2 cancelled twins.

---

## Traps (carried forward from memory + the code)

1. **Same-index restart is a re-roll and stale checkpoints stay.** There is no torch seeding on the
   train path, so re-running r4 under the old name puts a *different* init's checkpoints beside the
   old ones in `finetune_ncosmo300_4/`, and `find_best_checkpoint` returns the global min. **Fresh
   experiment name — this is the whole reason for `..._ens1`.** (`restart-same-repeat-is-a-reroll`)
2. **`_RESNET_MAPKW` must ride on the row.** `_npe_finetune_z8` builds a bare `_hybrid_lmin50_z8()`
   with no map kwargs ⇒ a UNet, then hands it PreActResNet weights. Reusing `_npe_finetune_bgp()`
   (which re-injects it) is the guard; the strict `checkpoint_path` loader is the canary.
3. **Never v100** — encoder finetunes need >16 GiB (`encoder-finetune-never-v100`), and the eval
   path has its own v100 history (`npe-eval-ooms-on-v100`; fixed in 72ec561 but the user directive
   stands). a100 / l40s only.
4. **The l40s dead card + twin submit.** One L40S on `compute-gpu-0-5` throws uncorrectable ECC and
   SLURM hands it to the *next* job every time, so a lone resubmit dies again. Twin-submit for
   *eval* jobs and cancel the redundant survivor within ~10 min. **Never twin-submit training** —
   two survivors would write two inits into one run dir (trap 1). (`l40s-dead-gpu-compute-gpu-0-5`,
   `l40s-npe-gpu-collision-cap3`)
5. **The smoke gate false-fails this row** (`eb_map_variant=None` vs the fixture's
   `E_fwhm8_lmin50_lcut1400` only) ⇒ `--skip-smoke`, with the Phase-0 dict-equality check standing
   in for it. (`smoke-fixture-eb-variant-fwhm8-only`)
6. **Gatekeeper `--args` charset**: `[A-Za-z0-9][A-Za-z0-9_.-]*` or `--[a-z-]*`, ≤32 tokens, no `/`,
   `*`, `=`. `--data-patterns`, `--variate-glob`, `--base-path` are **local-only**. The commands in
   Phases 3–4 are already compliant. (`eval-submit-arg-charset`)
7. **Commit AND push (over SSH) before `sync`** — an unpushed rev fails with `reference is not a
   tree`. Push may be blocked if the command line also contains a dot-ssh path; put it in a script.
8. **`logs --name`**: eval jobs are `eval_run_<jobid>`, and the gatekeeper matcher wants
   `--name run` (which also matches `sample_run`/`plot_run` — filter by jobid). Train jobs match on
   the experiment name. A bare dataset name (no `sim_` prefix) is what `sim` logs want.
   (`run-remote-logs-name-prefix`)
9. **`status` can omit a LIVE row** — two reads seconds apart are one sample. Never conclude a job
   died from a single empty `status`. (`status-no-rows-false-death`)
10. **Run the orchestrator with an ABSOLUTE path** — CWD drift makes it fail with `No such file`,
    and a grep on the empty output reads exactly like every job dying.
11. **Background bash tasks are killed at turn end** in this harness — run long local steps in the
    foreground; the fetch and the plot ports are restart-safe.
12. **`guard_unsafe` trips on the substring `rm`** inside a bash heredoc (which includes the word
    "warm") — use the Write/Edit tools for any file content containing it.
13. **Scaler-fit is not bit-reproducible**: `_fit_data_key_scalers_from_paths` calls
    `np.random.shuffle` on the global RNG and keeps ≤1000 files, so the misspec-time scalers are a
    *re-fit on the same train split*, not the literal training scalers. With N=1000 the moments are
    well converged (effect ≪ the shifts being measured) — a caveat for the report, not an action.
14. **`fixed_lock=False` in a misspec variate line means the fallback fired** (no lock overlap) and
    that variate was scored on ALL its on-disk cosmologies, some of which the model trained on.
    Treat any such cell as invalid until explained.

---

## Risks / unknowns / decisions that may need the user

* **Repeat quality spread.** If one of the five new encoders lands well off the `_r4_ens1`
  reference val_log_prob, the cross-repeat KL x-axis is partly measuring a bad init rather than
  OOD-ness. Flag it and ask before either re-rolling (under a fresh name) or dropping the repeat.
* **`--num-samples 4000` vs 10 000.** Chosen for fetch size; TARP/FoM are insensitive above ~2k, but
  it is a change from the first-NPE run's 10 000. If exact comparability with those figures is
  required, say so and it becomes 10 000 (≈11 GB fetch).
* **gb1p0 / gb1p3 at N≈80.** Their calibration numbers sit near the matched-N floor. The plan puts a
  floor band on them; if the paper needs a *resolved* b_g statement at this fidelity, more mocks (a
  new sim) — not more analysis — is the answer.
* **`gower_vd` is barely detectable** in summary space (C2ST ≈0.55, REPORT §2.3b). That is a result,
  not a failure; the multi-encoder metric is precisely aimed at it, but it may not move.
* **The Fable metric may fail the uniform-null criterion.** If so it must not be shipped; the honest
  outcome is "the mean-p baseline remains the recommended combiner". Flag rather than tune.
* **No physics code is touched.** `src/cosmology/`, `src/KiDS/systematics.py`, `src/KiDS/tomo.py`
  are read-only for this whole task and no step requires an edit there. If any step ever seems to,
  **STOP and ask the user** — the guard hook blocks it by default and for good reason.

---

## Phase B — the errno-28 blocker: root cause + the FIX-IN-WAITING (NOT applied)

**Status 2026-09-04T16:00Z: no code change has been made. User directive: "Wait for you
entirely" — apply nothing, submit nothing.**

### Root cause (established, not speculative)

Lightning's `_atomic_save` does:
```python
with fs.transaction, fs.open(urlpath, "wb") as f:
    f.write(bytesbuffer.getvalue())
```
Under `fs.transaction`, fsspec opens with `autocommit=False`, which at
`fsspec/implementations/local.py:401` calls `tempfile.mkstemp()` **with no `dir=` argument**. So the
~35 MB checkpoint is written to `$TMPDIR`/`/tmp` **on the compute node** first, then `shutil.move`d
onto `/share/gpu5`. The failing filesystem is therefore **the compute node's /tmp, not gpu5**.

This explains every observation that previously made no sense:

| Observation | Explained |
|---|---|
| Login-node `dd` into `checkpoints/` works | direct write, no temp file |
| Compute-node `dd` to gpu5 works (2.6 GB/s) | direct write, no temp file |
| Compute-node checkpoint fails errno 28 | 35 MB into a full `/tmp` first |
| SLURM `.out`/`.err` write fine on the same node | small direct appends, no temp |
| BOTH a100 (1352585) and l40s (1352642) fail | both have a small/full `/tmp` |
| `stat -f` shows 3.7 T + 1.53 B inodes free on gpu5 | gpu5 was never the problem |

Corroboration in Lightning's own source: the same function special-cases `errno.EXDEV` with
"Upgrade fsspec to enable cross-device local checkpoints" — that error only arises **because** the
temp file lands on a different device from the target.

### The fix in waiting (≈6 lines, additive, NO gatekeeper redeploy)

Top of `src/ml/models/utils.py::fit_model` (it already receives `base_path`; covers `train.py`,
`train_multi.sh`, and `train_embeddings.py` provided they route through `fit_model` — VERIFY that
before relying on it):
```python
d = os.path.join(base_path, ".tmp")
os.makedirs(d, exist_ok=True)
tempfile.tempdir = d          # the one that actually matters in-process
os.environ["TMPDIR"] = d      # for anything spawned afterwards
```

- ⚠️ **`tempfile.tempdir` is mandatory, not belt-and-braces.** `tempfile.gettempdir()` caches its
  answer in `tempfile.tempdir` on first call, and `mkstemp(dir=None)` reads that cache. By the time
  `fit_model` runs, wandb/torch/matplotlib imports have almost certainly already called it, so
  setting **only** `os.environ["TMPDIR"]` is a **silent no-op** and the probe would fail identically.
- Put the dir at `{base_path}/.tmp`, **not** under `checkpoints/` — same filesystem (so the
  `shutil.move` becomes a cheap rename) but outside the tree that `get_best_checkpoint` and the
  `status` ckpt-counter walk.
- `makedirs(exist_ok=True)` from every DDP rank is safe; concurrent jobs sharing the dir is safe
  (`mkstemp` names are unique). Expect torch-inductor's cache to relocate there too — acceptable.
- The alternative home is the job preamble (`.claude/cluster/remote/_common.sh`), which is arguably
  more correct but **requires the user to re-run `bootstrap_install.sh`**. The `fit_model` version
  deliberately avoids that manual gate and also fixes local runs.

### Verification required BEFORE any resubmit (this replaces the smoke gate for this change)

1. **Unit, with a poisoned cache** — call `tempfile.gettempdir()` FIRST, then the helper, then assert
   `tempfile.mkstemp()[1]` starts with `d`. Testing in a fresh interpreter misses the exact failure
   mode above.
2. **End-to-end** — `smoke_test_experiment.py` on a **fixture-compatible** row (one with
   `eb_map_variant='fwhm8_lmin50_lcut1400'`; NOT the `ens1` row, which false-fails per the
   `smoke-fixture-eb-variant-fwhm8-only` memory). Green = a `.ckpt` lands AND `{base_path}/.tmp` is
   empty afterwards (the move consumed the temp file).

### Cheap discriminator to run FIRST, once logs are reachable

The innermost frame of the errno-28 traceback decides whether this fix even applies:
- `f.write` / fsspec `LocalFileOpener.write` ⇒ the **temp** filesystem ⇒ the fix works.
- `shutil.copyfile` / `copyfileobj` inside `commit()` ⇒ the **target** on gpu5 ⇒ the redirect will
  NOT help and the diagnosis needs reopening.
Read that frame before committing five jobs to the theory.

### Resume order once hypatia's sshd is back

- [ ] `run_remote.py status` — confirm the queue is empty (no zombies from the cancelled cycle).
- [ ] Confirm `gower_npe_finetune_nla_m_bgp_z8_ens1` still shows **ckpts=0** — a partial checkpoint
      would arm the `find_best_checkpoint` global-min trap and must be cleared first.
- [ ] Pull the old jobs' tracebacks; run the innermost-frame discriminator above.
- [ ] Decide fix vs no-fix on that evidence; if applying, run BOTH verifications, then commit + push
      to `kids-preparation` over SSH, then `sync`.
- [ ] **`data-ls` for a partial `gower_bgp_gb1p3_f16_sc8a1_fwhm4_lmin56_lcut1400`.** The prebake
      writes via h5py (direct, not fsspec) so the cancelled job 1352460 may have left a partial or
      truncated store. Check `scripts/prebake_maps.py`'s skip-existing/overwrite behaviour: if it
      skips existing files and the loader then silently drops truncated ones, gb1p3's ~80 events
      shrink with **no warning**. Clear or force-overwrite; do not assume.
- [ ] **Fire r0 ALONE as the probe.** Discriminating signal: `finetune_ncosmo300_0` shows ckpts≥1
      after epoch 0 (~4 min). Only on that success fire r1–r4 + the gb1p3 prebake.
