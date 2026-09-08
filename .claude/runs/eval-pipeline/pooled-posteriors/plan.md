# Pooled (repeat-concatenated) posteriors as the standard path

## Goal

Make the **pooled posterior** — the equal-weight mixture over repeats, i.e. the literal
concatenation of the per-repeat posterior draws — a first-class, cluster-produced, **tested**
artifact, for BOTH:

* the **production/mock** eval dumps `checkpoints/<exp>/ensemble_posterior_samples_<match>.npz`, and
* the **observation/external** dumps
  `checkpoints/<exp>/external/<label>/external_posterior_samples_<prior>_<match>.npz`.

User requirement (2026-09-08): *"when we come to run it on the observations, we will work directly
with pooled samples posteriors, so this NEEDS to be the standard, tested path."*

### What already exists and what it is NOT

`artifacts/variate_diagnostics/tarp_ensembled.py` (built earlier today) computes **TARP** on the
pooled posterior WITHOUT pooling any samples — it exploits `f_i(pooled) = mean_r f_i(repeat r)`,
which is exact but **specific to TARP** (f is linear in the draw set). It produces no pooled
samples and nothing else in the pipeline sees them. After this task its role changes from
*product* to *independent cross-check* of the real pooled path.

FoM / widths / bias / corners all need the actual pooled draws. `test_log_prob` cannot be pooled
from samples at all (the mixture density is `logsumexp_r log q_r(theta|x) - log R`, which needs the
flows) — it must be written as `null` with a reason, never silently omitted.

## Design

### 1. `src/ml/eval/pooling.py` — the primitive

`pool_sample_dumps(member_npzs, out_path, *, draws_per_member=None, provenance=None)`

* **Row alignment**, asserted not assumed. Primary key `test_files`; fall back to `theta0s` with
  `equal_nan=True` (⚠️ **observation dumps have NaN truths** — a plain `array_equal` fails there,
  and older production dumps e.g. `gower_nle_finetune_nla_m_z8_*` carry no `test_files`).
* **Equal draws per member** required (`S` identical). Otherwise "equal-weight mixture" is false
  and a member whose job died short silently down-weights itself. `draws_per_member=N` is the
  tolerated escape (subsample each member to a common N), never the default.
* Concatenate along the draw axis. Same npz schema out, plus:
  * `member_of_draw` (int8, `[S_total]`) — which member each draw came from. This is what lets
    anyone later split, reweight, or diagnose seed disagreement without redoing 14 h of MCMC.
  * `pooled_provenance` (json string) — member experiments, match strings, S each, prior tag,
    git rev, timestamp.

### 2. Scoring — reuse, do not reimplement

`run_evaluation_on_samples` verbatim (`src/ml/eval/evaluate_models.py`), scaler from
`_build_cosmo_preset_scaler`, prior from `build_gower_prior`. **Assert `cosmo_param_names` and
`preset_overrides` identical across members before pooling** — pooling two different parameter
frames is silent nonsense. Mirror `nle_external`'s `has_truth` gate: non-finite truth ⇒ no metrics.

### 3. `eval.py --mode pool` — the cluster entry

Each repeat is its own experiment directory (`r{r}` is in the name), so the input is a **list of
experiment names**, not `--repeat-indices` on one experiment. Experiment names are valid gatekeeper
tokens (`[A-Za-z0-9][A-Za-z0-9_.-]*`), so `--experiments A B C` passes through unchanged.

* mock:      `--mode pool --experiments <exp_r0> <exp_r1> <exp_r2>`
* external:  `... --data-tag <label> --prior-mode <prior>`
* output:    `checkpoints/pooled/<arm>/[external/<label>/]pooled_posterior_samples_<...>.npz`
  (+ `pooled_evaluation_results_*.json`, `pooled_tarp_credible_intervals_*.json`) — NOT under a
  member, and fetchable with the existing `fetch --exp pooled --rel ...` pattern.
* Submit CPU: `run_remote.py eval --cpu --partition CORES64` (CORES40 jobs vanish silently) with
  enough `--mem-gb`: the production TARP bootstrap copies `samples[:, idx, :]` 25x and the distance
  temporary at S=24000 is several GB.
* `--pool-arm <name>` resolves an arm to its per-repeat experiment names. The `ARMS` table moves
  from `scripts/sample_observation.py` into `src/ml/eval/arms.py` and both import it — a second
  copy will drift.

### 4. Gates ("tested" is the requirement, not a bonus)

- [ ] **G1 pool-of-one**: pooling a single member is byte-equal to its npz, and rescoring
      reproduces the production json EXACTLY on the deterministic metrics
      (`mse/bias/std_dev/width_68/width_95`). FoM-vs-prior (`prior.sample`) and TARP (`seed=None`)
      only within noise.
- [ ] **G2 identity cross-check**: pooled-npz TARP vs `tarp_ensembled.py`'s f-cache concat curve
      for the same variate — two independent computations of one quantity.
- [ ] **G3 local end-to-end before any sync**: `/data/alex/variate_samples/<exp>/` is already
      checkpoints-shaped; run `--mode pool` against it with a `--base-path` override.
- [ ] **G4 cluster**: `--pool-check` runs G1+G2 on the cluster, the way `nle_external` has its
      reproduction gate.

### 5. Consumers (surface, do not rewrite unasked)

`scripts/plot_blind_posteriors.py`, `build_unblinding_notebook.py`, `tier1_report.py`,
`tier2_tables.py` discover `external_posterior_samples_*`. The pooled npz is schema-identical so a
path swap suffices; note which ones should prefer pooled once this lands.

## ⚠️ Blind protocol

Observation posteriors are pooled ON THE CLUSTER and fetched blind. Pooling code logs **no sample
statistic** when `has_truth` is False; the local self-test uses mock dumps only; no fetched
observation npz is ever opened locally.

## Checklist

- [ ] `src/ml/eval/arms.py` — the arm table, imported by `sample_observation.py`
- [ ] `src/ml/eval/pooling.py` — primitive + scoring + gates
- [ ] `eval.py --mode pool` wiring (+ `--pool-arm`, `--pool-draws`, `--pool-check`, `--base-path`)
- [ ] G3 local end-to-end on `/data/alex/variate_samples`
- [ ] G1 + G2 locally
- [ ] commit + push over SSH, `sync`, `--dry-run`, then a real cluster pool run (G4)
- [ ] report: pooled-vs-per-repeat numbers, and note the consumer swap as the next step
