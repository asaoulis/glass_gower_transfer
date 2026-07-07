# Recipe: same-cosmology / same-mock posterior comparisons across models

The recurring pattern: **generate posterior samples for a fixed, specified set of test
cosmologies → save them tagged with their mock identity → later overlay/compare posteriors
across models (or repeats) on the *same* mock**. Validated 2026-07-07 on the production
`gower_nle_finetune_nla_m_z8_r{0..4}_ens9` NLE ensembles (task
`.claude/runs/eval-and-viz/plots-and-prior-analysis/`).

## 1. Fixing WHICH test points are sampled

Three config knobs, all honoured by `prepare_data_parameters` → `split_by_cosmology`:

| Knob | What it does | Production value |
|---|---|---|
| `fixed_test_sim_ids` | Lock-file of sim_ids forced into the test split (`config/fixed_test_sets/gower_test_ids.json`, 200 ids). **Keep the experiment's own value** — never shrink it: removed ids would re-enter the trainval pool and silently change the fitted scalers → wrong embeddings → garbage inference. | the exp's own lock-file |
| `N_test_cosmologies` | Eval-time trim of the *resolved* test set to the first N cosmologies **by sorted sim_id**, applied AFTER train/val selection — trainval and scalers stay byte-identical to training. Deterministic, so every model/repeat sees the same N cosmologies. | `40` |
| `test_shape_noise_idx` | Filename filter on the test files, `[rot, shape]`; each slot an int or a **list** of ints. The gower store layout is `out{0,1} × rot{0..4} × _{0..3}`, so `[0, [0, 1]]` keeps `out{0,1}_rot0_{0,1}` = 4 noise variants per cosmology (2 outer × 2 inner, same footprint rotation). | `[0, [0, 1]]` |

40 cosmologies × 4 shape-noise variants = **160 inference points** per model.

## 2. Generating samples: `gen_samples.py`

- Experiment list entries are `(experiment_name, match_string, [source_experiments])`
  3-tuples for NLE/NPE-on-embeddings models (the ensemble is built automatically when
  `ensemble_repeats > 1`), e.g.
  `("gower_nle_finetune_nla_m_z8_r0_ens9", "ncosmo300_0", ["kids_legacy_hybrid_nla_m_lmin50_fwhm4_z8"])`.
- `PRIOR_MODE` picks the sampling prior: `"gower"` (empirical flow prior) or
  `"kids_s8_analytic"` (the paper's flat-S8-box analytic prior; NLA-M `(a_ia, b_ia)`
  Gaussian block ⇒ it is **restricted to the scaled box by default** — MCMC must stay in
  the `[0,1]^D` region the flow was trained on). The prior name is encoded in the output
  filename, so different-prior runs coexist.
- `CONFIG_OVERRIDES` carries the three knobs above plus `emb_test_batch_size`. The MCMC
  joblib parallelism unit is **one test-loader batch** (`generate_samples`: one job per
  batch, slice sampling vectorised within the batch), so choose
  `batch ≈ N_points / NUM_JOBS` for a single wave: 160 points / batch 8 = 20 jobs =
  `NUM_JOBS = 20` < the CPU allocation.
- Output (eval npz schema, one file per model repeat):
  `{base_path}/checkpoints/<exp>/samples_<PRIOR_MODE>_<match_string>.npz`
  — under `checkpoints/<exp>/` so `run_remote.py fetch --exp <exp>` pulls it.
  An existing file is **skipped** (delete/rename it to regenerate).

## 3. The saved-sample schema (shared with the eval dumps)

Written by `src/ml/eval/utils.py:_save_posterior_samples` (`np.savez_compressed`):

| key | shape | meaning |
|---|---|---|
| `samples` | `[S, N, D]` | posterior samples, **scaled [0,1] space** |
| `theta0s` | `[N, D]` | true parameters, **scaled [0,1] space** |
| `test_files` | `[N]` str | mock basenames `output_<sim>_out<o>_rot<r>_<n>.h5` |
| `sim_ids` | `[N]` int64 | cosmology id parsed from the filename |
| `aug_ids` | `[N]` int64 | **trailing** `_<n>.h5` index only |

Two caveats that matter for matching:

- **`(sim_id, aug_id)` is NOT a unique key**: `aug_id` is only the trailing noise index,
  so `out0_rot0_0` and `out1_rot0_0` of one cosmology both carry `aug_id = 0`. **Match
  test points across models by the full `test_files` basename** (what
  `scripts/plot_posteriors.py` does).
- Positional row↔file alignment assumes no corrupt-file skips in the loader (true for the
  clean prebaked stores; `H5CosmoDataset` silently substitutes a neighbour on a corrupt
  read). `_save_posterior_samples` drops the id columns if the path count mismatches —
  if a npz lacks `test_files`, treat its rows as unmatchable across models.
- For **embeddings models** the test loader is an `EmbeddingDataset`; it carries the
  source H5 file list (`.paths`, attached when embeddings are computed fresh) so the ids
  get tagged. Cached-embedding paths are deliberately NOT trusted.

Samples/truths are stored **scaled**; convert to physical units with the preset box
(`src/ml/data/constants.py:COSMO_PARAM_PRESET_MINMAX`): `x_phys = x*(max−min)+min`.
Mind per-experiment `preset_overrides` (e.g. `a_ia ∈ [−6,6]` for nla/nla_z variates) —
`plot_posteriors.py --override a_ia=-6,6`.

## 4. Cluster workflow (the whole loop)

```bash
# 0. local sanity: build the prior, import the edited modules, run the plot script on
#    mock fixtures (scripts/make_mock_posterior_samples.py) BEFORE touching the cluster.

# 1. commit + push (SSH push URL) — the cluster runs the pushed rev
git push origin kids-preparation
python .claude/cluster/run_remote.py sync            # --dry-run first

# 2. sampling job (CPU MCMC) — CORES64, NOT COMPUTE (old nodes SIGILL on the AVX stack)
python .claude/cluster/run_remote.py sample --where CORES64 --ncpu 30 --wall_h 24
python .claude/cluster/run_remote.py logs --name run     # job name is sample_run

# 3. verify the dumps: N=160 and all 5 schema keys
ssh hypatia-glass view ls checkpoints/<exp>

# 4. overlay plots ON the cluster (plot-submit; needs the control plane redeployed once
#    via bootstrap_install.sh after any .claude/cluster/remote edit)
python .claude/cluster/run_remote.py plot \
    --exps <exp_r0>,<exp_r1>,<exp_r2> \
    --pattern 'samples_kids_s8_analytic_*.npz' \
    --out plots/<name> --max-points 4          # PNGs -> MODELS_ROOT/plots/<name>/

# 5. pull results locally
python .claude/cluster/run_remote.py fetch --exp <exp>      # npz (+ eval json)
ssh hypatia-glass send plots/<name> | tar xzf - -C <local-dir>   # PNGs
```

`scripts/plot_posteriors.py` also runs locally on fetched npz (same CLI, `--samples`
mode). Chainconsumer ≥1.x: one `Chain(samples=df, ...)` per model, `shade=False,
bar_shade=False` for line-only contours; the DataFrame columns ARE the parameter labels
(there is no `parameters=` field); derived `S_8` inserted after Ωm; `ombh2` displayed ×100.

## 5. Reproducibility contract

The loader rebuild inside `gen_samples.py`/`eval.py` re-runs the exact training split
(`split_seed`, `ensemble_seed`, `max_trainval_cosmos`, `fixed_test_sim_ids`) so the
scalers refit to the training state; the knobs in §1 only ever act on the **test** side.
If two sample files disagree on `test_files` for the same nominal setup, the split
configs differ — do not compare them silently (the plot script intersects basenames and
errors when nothing overlaps).
