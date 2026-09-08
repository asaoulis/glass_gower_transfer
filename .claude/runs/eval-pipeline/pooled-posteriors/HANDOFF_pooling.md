# HANDOFF — pooled (repeat-concatenated) posteriors on the cluster

**For:** the unblinding agent.
**What this gives you:** one posterior per (arm, label, prior) instead of one per (arm, repeat,
label, prior) — the equal-weight seed mixture, as a normal posterior-sample npz that every
existing consumer can read with a path swap.

---

## 0. Verification ledger — read this before you run anything

| # | check | where | status |
|---|---|---|---|
| G1a | pooling one member against itself is byte-identical to that member's draws | local | **PASS** |
| G1b | rescoring a member reproduces the production `evaluation_results.json` — 80 deterministic metrics (`mse/bias/std_dev/width_68/width_95`) | local | **PASS**, worst abs diff **2.5e-7**. FoM agrees to 0.7 % / 0.03 % (Monte-Carlo, not asserted) |
| E2E | full mock pool: 3 x 8000 draws over 2000 rows x 15-D -> build + score | local | **PASS** — 2.6 GB npz built in ~2 min, scored in ~14 min (~16 min total, peak RSS well under 32 GB). `test_log_prob` null with reason; `fom_dim_normalized` 1.224 pooled vs 1.236-1.285 per seed |
| G2 | pooled-npz TARP vs the independent `f`-identity curve | local | **PASS** — max\|dECP\| **0.0040** against a measured reference-draw floor of 0.0071 +- 0.0022 (p90 0.0101) |
| OBS | the OBSERVATION branch on a synthetic fixture (NaN truth, `external/` layout) | local | **PASS** — writes under `pooled/<arm>/external/<tag>/`, draws byte-equal to `concat(members)`, `member_of_draw` counts correct, **no metrics json**, no sample statistic logged |
| CLU | one real cluster `--pool-check` | hypatia | **PASS** — job `1357612`, CORES64. G1a identity PASS; G1b 80 deterministic metrics vs production, **worst \|diff\| = 0** (exact — same environment wrote both, unlike the 2.5e-7 local mirror comparison). Report at `checkpoints/pooled/k2/gates/pool_check_report.json` |

### 0.1 Status line

**Commits:** `89a2c8b` (the mode + primitive) then `e668a36`, `c12768a` (two gate fixes, below).
Cluster checkout synced to `e668a36`; **re-sync to `c12768a` before running the gate.**

**Cluster: job `1357612` (2026-09-08, CORES64, 16 cpu, 128 G, `--pool-arm k2 --repeat-indices
0 1 2 --pool-check`) ran at rev `e668a36` and returned **PASS**. `--pool-check` exercises G1 only,
which `c12768a` does not touch (that commit changes G2), so the verdict stands for what it tested.
The cluster checkout has since been synced to `a9038fb`.**

**Every row of the ledger is now green.** The path is exercised end to end on the cluster.

⚠️ **Two bugs were found by running the gates, both in the GATE, neither in the pooling or scoring
path** (G1 and E2E predate and are unaffected):
* `e668a36` — G2 transposed the dump's sample axis. Here it raised `IndexError`; on a dump with
  `S <= N` it would have scored garbage and reported a passing-looking number.
* `c12768a` — G2 compared a row-bootstrap-mean ECP against a plain one. Those differ by 0.0048 on
  identical `f` values, which was most of the 0.0072 the gate first reported; it would have read
  as a defect in the pooling. It now compares like with like and judges against a floor it
  measures from the cache.

The lesson worth carrying: a gate that has never been run is not a gate.

**Independent metric cross-check** (not a formal gate): `tarp_ensembled.py`'s pooled k2 full-vector
`calibration_error` = 0.00678 vs the real pooled draws' 0.00604 — ratio 0.89, inside the 0.89-1.21
band the per-repeat reproductions spanned.

### 0.2 What is NOT verified

* Nothing has been run on a **real observation** dump. The OBS row used mock draws with the truth
  blanked — it proves the code path, not the science.
* The consumers (§6) have **not** been repointed at pooled files.
* `scripts/sample_observation.py` still carries **its own copy** of the arm table (§7).

---

## 1. What pooling is, and the one thing to keep in mind

    q_pool(theta | x) = (1/R) sum_r q_r(theta | x)

whose sample set is literally the concatenation of the per-repeat draws along the draw axis.

⚠️ **A mixture is broader than its members whenever they disagree.** Pooling can only add spread
across seeds, never remove it — so if the members are individually overconfident the pooled
posterior looks better calibrated *by construction*. Always quote pooled **next to** per-seed.

Measured on the Gower mocks (2026-09-08, `figures_ensembled/`): pooling removes **all** the 1-D
marginal overconfidence (nla_m std(z) 1.07/1.09/1.09 per seed → **1.00** pooled; central-68 %
coverage 0.64 → 0.68) but the **joint** TARP under-coverage survives (full-vector
calibration-error/null only 3.0 → 2.3, still one-signed negative at every alpha). The residual is
in the *correlation structure*, which all seeds share. Do not read a calibrated pooled 1-D error
bar as evidence the joint posterior is calibrated.

`test_log_prob` is written as **`null`** with a stated reason: the mixture density is
`logsumexp_r log q_r(theta|x) - log R`, which needs the member flows, not their draws. It cannot be
recovered from a pooled sample file.

---

## 2. Preconditions — check these before submitting

1. **Every requested repeat has an external dump for that `(label, prior)`.** The GATE-3b pilots
   ran `--repeats 0` only, so two or more members may not exist yet. Pooling needs **>= 2**.
   Check with `run_remote.py data-ls` / `status`, or just let §5's "no posterior dump" error name
   the exact paths it looked for.
2. **`k2` has repeats 0–3, not 0–4.** Asking for repeat 4 is a hard error, deliberately: a pooled
   posterior quietly built from 3 of 4 requested seeds is a different object from the one asked for.
3. **`--prior-mode` is MANDATORY on the observation path** and must equal the prior the samples
   were *drawn* under (`kids_s8_analytic` or `LCDM_fixed_w0`). `eval.py`'s default is `gower`,
   which will look for `external_posterior_samples_gower_*` and exit with the candidate list.
4. **Equal draw counts per member.** Unequal `S` is refused rather than silently down-weighting the
   short member (a job that died early would reweight the science). If a member genuinely is short,
   pass `--pool-draws <N>` to subsample every member to a common `N` — a deliberate choice, not a
   default.
5. **Arm table lockstep** (§7): run
   `python -c "import sys;sys.path.insert(0,'.');from src.ml.eval.arms import assert_matches_sample_observation as c;print(c() or 'IN SYNC')"`.

---

## 3. Commands

### 3.1 Gatekeeper constraints (these bite)

Every `--args` token is re-validated against `^[A-Za-z0-9][A-Za-z0-9_.-]*$` or `^--[a-z][a-z-]*$`,
**max 32 tokens**. So:

* Experiment names, arm names, integers, label and prior names: **fine**.
* **No paths, no globs, no `=`.** `--base-path` therefore works **locally only** — never in
  `--args`. Names map to paths in code.
* Always use `--pool-arm`, not a bare `--experiments` list: with an explicit list there is no arm
  table to consult and the match string is *inferred* from the experiment name (a heuristic —
  the `nla_m`-era chains carry `ncosmo300_<r>` and later chains `ncosmoNone_<r>`).

### 3.2 The observation pool — the one you actually need

One job per `(arm, label, prior)`:

```bash
python .claude/cluster/run_remote.py eval --cpu --partition CORES64 \
    --ncpu 8 --mem-gb 64 --wall_h 1 \
    --args "--mode pool --pool-arm nla_m --repeat-indices 0 1 2 3 4 \
            --data-tag <LABEL> --prior-mode kids_s8_analytic"
```

`--dry-run` first, always. An observation is a handful of rows (N ~ 1–4), so this job is
**seconds of work** — the wall/mem above are slack, not requirements. No GPU: pooling is a memory
copy and the scoring is skipped when there is no truth.

Repeat per arm (`nla_m`, `nla_m_nobgp`, `nla`, `nla_z`, `vd`, `k2`) and per prior.

### 3.3 The mock validation pool — the expensive one

Only if you want the pooled *mock* calibration numbers alongside. This scores 3 × 8000 draws over
~2000–4000 rows with the full TARP bootstrap:

```bash
python .claude/cluster/run_remote.py eval --cpu --partition CORES64 \
    --ncpu 32 --mem-gb 200 --wall_h 6 \
    --args "--mode pool --pool-arm k2 --repeat-indices 0 1 2"
```

⚠️ `--partition CORES64`, not CORES40 — CORES40 jobs vanish silently. Memory: the pooled array is
~2.6 GB at 3 × 8000 × 2000 × 15, and the production TARP bootstrap materialises
`samples[:, idx, :]` per iteration on top of it.

### 3.4 The gate, on the cluster

```bash
python .claude/cluster/run_remote.py eval --cpu --partition CORES64 \
    --ncpu 16 --mem-gb 128 --wall_h 2 \
    --args "--mode pool --pool-arm k2 --repeat-indices 0 1 2 --pool-check"
```

Runs G1 (identity + reproduction of the production metrics) and **exits non-zero** if it fails.
Re-run this after any change to `src/ml/eval/pooling.py`.

### 3.5 Fetching

```bash
python .claude/cluster/run_remote.py fetch --exp pooled --rel <ARM>/external/<LABEL>
```

`fetch` confines under MODELS_ROOT and takes any subpath, so `pooled` works even though it is not
an experiment name. Pull `--rel` narrowly — the mock pooled npz is gigabytes.

---

## 4. Outputs

```
checkpoints/pooled/<arm>/                                  # mock
    pooled_posterior_samples.npz
    pooled_evaluation_results.json
    pooled_tarp_credible_intervals.json
    pooled_provenance.json
checkpoints/pooled/<arm>/external/<label>/                 # observation
    pooled_posterior_samples_<prior>.npz
    pooled_provenance.json                                 # NO metrics json — there is no truth
```

The npz is **schema-identical** to `_save_posterior_samples`' output (`samples [S,N,D]`,
`theta0s [N,D]`, `test_files`, `sim_ids`, `aug_ids`) plus:

* **`member_of_draw`** `int8 [S_total]` — which repeat each draw came from. Persisted so seed
  disagreement stays measurable after the fact without redoing 14 h/repeat of MCMC. Split on it to
  recover any member, or to compare members without reloading them.
* **`pooled_provenance`** — a JSON string: member experiments, match strings, draws per member,
  arm, label, prior, row-alignment key, `has_truth`, git rev, timestamp.

---

## 5. Failure messages and what they mean

| message | meaning | do |
|---|---|---|
| `members are NOT row-aligned (test_files: ...)` | the members were scored on different observations or in a different order | do not override — find out why the two sampling jobs saw different inputs |
| `members have DIFFERENT draw counts [...]` | a member's sampling job ended short | check that job's log; only then `--pool-draws <min>` |
| `members do not share a parameter frame` | `cosmo_param_names` or `preset_overrides` differ across members | you are mixing arms or eras; pooling them would give a posterior in no member's frame |
| `no posterior dump for <exp> (match ...)` | wrong `--prior-mode`, wrong `--data-tag`, or that repeat was never sampled | the message lists every path it tried |
| `arm 'k2' has no repeat(s) [4]` | k2 has 0–3 | drop the repeat, or use a different arm |

---

## 6. Consumers — NOT yet repointed

These discover `external_posterior_samples_*` and would need a path swap to prefer pooled:

* `scripts/plot_blind_posteriors.py`
* `scripts/build_unblinding_notebook.py`
* `scripts/tier1_report.py`
* `scripts/tier2_tables.py`

The pooled npz is schema-identical, so this is a path change, not a rewrite. It has **not** been
done — decide per consumer whether the unblinding wants pooled or per-seed (Tier-1 nulls arguably
want per-seed so seed disagreement stays visible).

---

## 7. Known gaps

1. **The arm table is duplicated.** `src/ml/eval/arms.py` is the intended single source, but
   `scripts/sample_observation.py` still has its own copy — it was being edited by another session
   when this was written. The two **already drifted once** (the `nla_m_nobgp` bake suffix went
   `a1` → `a0_tagged` in the script while `arms.py` still said `a1`; now synced). Run
   `arms.assert_matches_sample_observation()` before trusting either, and repoint the script when
   it is quiet.
2. **`--experiments` infers the match string** from the experiment name. Use `--pool-arm`.
3. **Blind protocol.** The pooled observation npz is fetched blind and opened only inside the
   unblinding protocol. The pooling code prints shapes and provenance only, never a sample
   statistic, when `has_truth` is False. The local self-tests use mock dumps only.

---

## 8. Where the code is

| file | role |
|---|---|
| `src/ml/eval/pooling.py` | the primitive (`pool_sample_dumps`), scoring (`score_pooled_dump`), gates |
| `src/ml/eval/pool_mode.py` | `eval.py --mode pool` — resolution, output layout, gate driver |
| `src/ml/eval/arms.py` | arm → (experiment template, bake, repeats) + match strings + the drift guard |
| `.claude/runs/training-runs/production-training-runs/artifacts/variate_diagnostics/tarp_ensembled.py` | INDEPENDENT cross-check: pooled TARP via `f_i(pooled) = mean_r f_i(r)`, no pooled array. Not a substitute — that identity is specific to TARP |
