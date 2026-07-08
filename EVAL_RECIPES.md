# Evaluation recipes — standard, misspecification, ensemble-agreement

`eval.py` is the single evaluation entrypoint. It has two modes plus a repeat-ensemble
extension of the misspecification mode:

| Mode | What it does | Primary outputs |
|---|---|---|
| `list` | Standard in-distribution eval: loop experiment names, rebuild config/loaders exactly as training (same match_string/repeat logic), evaluate best checkpoints | `evaluation_results.json` + `tarp_credible_intervals.json` per run folder; `ensemble_evaluation_results_<match>.json` + `ensemble_posterior_samples_<match>.npz` at the experiment level for `ensemble_repeats>1` configs |
| `misspec` | One trained model × the TEST split of every Gower variate dataset, with the ORIGINAL training scalers injected (never refit per variate) | per variate under `checkpoints/<exp>/misspec/<variate>/`: `misspec_evaluation_results_<match>.json`, `misspec_tarp_credible_intervals_<match>.json`, `misspec_posterior_samples_<match>.npz` |
| `misspec --repeat-indices 0 1 …` | Full misspec pass per training repeat + per-event CROSS-REPEAT posterior disagreement (mean pairwise symmetric diag-Gaussian KL — the `ensemble_uncertainty.py` formulation) | additionally `misspec_repeat_disagreement_<matches>.{json,npz}` per variate (per-event KL scores + per-repeat posterior moments); summary lines carry `cal_full`, `cal_om_s8_w0`, `repeat_kl_mean` side by side |

## CLI

```bash
python eval.py --mode list  [--experiments EXP ...] [--repeat-indices I ...]
python eval.py --mode misspec [--misspec-base gower_npe_finetune_nla_m_z8] \
               [--repeat-indices 0 1 2 3 4] [--num-samples 10000]
python eval.py            # bare: runs DEFAULT_MODE (see below)
```

- `--repeat-indices` — `list`: restrict which repeats are evaluated. `misspec`: one full pass
  (repeat-bound scalers + that repeat's 9-member ensemble + all variates) per index; ≥2 indices
  additionally computes the cross-repeat disagreement statistic.
- Variate list/globs/`exclude_params` live in `src/ml/eval/misspec.py:DEFAULT_VARIATES`.
  `exclude_params` drops params whose MEANING changes between suites (a_ia for nla/nla_z);
  params absent on disk (b_ia there) are NaN-filled by the loader and auto-dropped from
  calibration. FoM is always against the base Gower prior over all sampled dims.

## Cluster submission

```bash
# standard in-distribution eval (also what a bare eval.py runs: DEFAULT_MODE=list):
python .claude/cluster/run_remote.py eval --gpu v100 --args "--mode list --experiments gower_npe_finetune_nla_m_z8"

# misspecification eval, single repeat:
python .claude/cluster/run_remote.py eval --gpu v100 --args "--mode misspec --repeat-indices 0"

# misspec + cross-repeat ensemble-disagreement (the combined OOD statistic run):
python .claude/cluster/run_remote.py eval --gpu v100 --args "--mode misspec --repeat-indices 0 1 2 3 4"
```

(The arg pass-through went live with the gatekeeper redeploy on 2026-07-08; repeats whose
training hasn't finished are skipped with a warning, so the 5-repeat form is safe to submit
while later repeats are still in the train queue.)

Logs: the job file is `eval_run_<jobid>.{out,err}`; the gatekeeper's matcher needs
`logs --name run` (NOT `eval_run`), and that also matches `sample_run`/`plot_run` — filter by
jobid section headers. Progress lines are `[misspec] <variate>: ...` on stdout.

Fetch results locally, then plot:

```bash
python .claude/cluster/run_remote.py fetch --exp gower_npe_finetune_nla_m_z8
python .claude/runs/eval-and-viz/first-npe-misspecification/artifacts/plot_misspec_tarp.py \
    --root ml-checkpoints/gower_npe_finetune_nla_m_z8/misspec --match ncosmo300_0 \
    --out misspec_tarp_coverage.png
```

## CLI pass-through plumbing (for reference)

The `eval-submit` arg pass-through (charset-validated tokens after `<mods>`) lives in
`.claude/cluster/remote/{ssh_glass_gatekeeper.sh,submit_eval.sh}` (local-only, NOT in git) and
was deployed via `bootstrap_install.sh` on 2026-07-08. Any future edit to those files needs
another bootstrap redeploy (the gatekeeper is the trust anchor). `DEFAULT_MODE` in `eval.py`
is `"list"`, so a bare `python eval.py` means the standard eval.

## Design invariants (do not break these)

- **Scalers**: misspec mode fits data scalers (bandpowers→LogNormal, maps→Standard) ONCE per
  repeat on the ORIGINAL `nla_m` train+val split (`prepare_data_parameters`, ensemble path) and
  injects them into every variate test loader via `TransformingDataset` — refitting on a variate
  would absorb the covariate shift being measured.
- **Test sets**: variate test cosmologies = fixed lock (`config/fixed_test_sets/gower_test_ids.json`)
  ∩ on-disk, built DIRECTLY (not via `split_by_cosmology`, whose no-train/val fallback silently
  produces a train-heavy split on variate stores that only contain test ids). Test-point filter
  `[0, [0, 1]]` = rot0, inner noise {0,1}, any outer index — identical for the in-dist reference.
- **Sampling robustness**: far-OOD conditioning can degenerate the RQS spline inverse; the
  local spline (`patched_rqs.py`) clamps the discriminant, and the driver drops test points with
  non-finite samples, reporting `n_dropped_nonfinite` in the JSON. A large drop count means the
  variate is being sampled at the edge of tractability — treat its metrics with care.
- **Cross-repeat alignment**: disagreement aligns events by test-file basename; per-event
  posterior moments are in the same scaled space as TARP.
