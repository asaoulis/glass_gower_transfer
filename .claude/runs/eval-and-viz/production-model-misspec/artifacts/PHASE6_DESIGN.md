# Phase 6 — multi-encoder OOD metric for the Gower BGP arm: design note

Script: `artifacts/gower_multiencoder_ood.py` (run: `/data/alex/glass/env/bin/python … --n-splits 60
--n-draws 200 --n-boot 1000`, seed 0). Outputs in `artifacts/phase6/`: one
`multiencoder_ood_<set>.npz` per variate + `_idtest` (`combined_score`, `combined_p`, `test_files`,
plus `p_<method>` for every method and the per-encoder stage-1 `p1_cond_knn_r<r>`),
`phase6_acceptance.{json,md}`, `phase6_auroc.png`, `phase6_spearman.png`, `run_full.log`.
Every number below is copied from that run.

## 1. Verdict up front

**The recalibrated mean-p combiner remains the recommendation** (`combined_p` = `meanp`). Nothing
tested — the literature option (a, DoSE) or my own option (b, GLS-Stouffer) — beats it on the hard
variates (`gower_gb1p0`, `gower_gb1p3`, `gower_vd`) by more than one by-cosmology bootstrap sd, nor on
the per-event Spearman. On the easy IA variates all right-tailed combiners tie (0.87 ± 0.01), ~+0.02
over the single-encoder average — real but uninformative. DoSE is worse on every variate that carries
signal and at chance on `vd` like everything else (§4). The reasons are mechanism-level and measured, not a tuning
failure, so I stopped rather than searching for a combiner that wins by noise.

## 2. The mechanism, and the bound it puts on any combiner

Per encoder r, the production score is `cond_knn`: the residual r = z − E[z|θ] (θ-kNN regression on
the encoder's own `_train` cloud), whitened by the train residual covariance and kNN-scored against
the train residual cloud; then a right-tail empirical p against the encoder's ID null. Five encoders
give five p-values per event, which are the only cross-encoder currency (raw `z` is never pooled;
constraint 2).

Measured on the shipped per-encoder p-values (probit scale, u = Φ⁻¹(1−p)), the cross-encoder
correlation of the same event's five scores is **ρ ≈ 0.70** on the null (range 0.68–0.74) and on every
variate (`vd` 0.70, `gb1p0` 0.69, `gb1p3` 0.77, `nla` 0.90). That correlation is the *physical*
fluctuation of the specific mock (shape noise, cosmic variance, the map's actual departure from the
mean at θ), which every encoder sees; only the encoder-specific approximation noise (~30 % of the
variance) is averaged away. So any scalar combination of the five gets an SNR gain of at most
√(5/(1+4ρ)) ≈ **1.15×**. The per-encoder mean probit shift is 0.20 on `gb1p0`, 0.10 on `gb1p3` and
**0.00 on `vd`**; a 0.20σ shift at 1.15× becomes 0.23σ, i.e. AUROC 0.556 → 0.564 — invisible against the
by-cosmology sd of the 40-cosmology gb sets. Any combiner *must* land inside the CI on the hard
variates, and it does.

Why the hard variates carry (almost) no per-event signal to amplify — primary-sourced, not from
memory: the training store `gower_mocks_nla_m_novd_bgp` (= the `gower_bgp_nla_m` sc8a1 bake) was
simulated with `--galaxy-bias-prior flamingo_pt_diag`, κ = 1
(`.claude/runs/training-runs/production-training-runs/log.md:809,863`): per-tomo-bin galaxy biases
b_i ~ N(mean_i, σ_i) truncated at ±3σ, means 1.018 → 1.481 and σ 0.18 → 0.098 from bin 1 to 6
(`src/KiDS/simulation_config.py:GALAXY_BIAS_PRIOR_MEANS/SIGMAS`;
`master_kids_legacy_simulator.py:draw_galaxy_bias_prior`). The two probes pin a *scalar* b_g in all
six bins (`--galaxy-bias 1.0 / 1.3`, `misspec.py:GOWER_BGP_VARIATES`):

| bin | prior support (κ=1) | b_g = 1.3 offset | b_g = 1.0 offset |
|---|---|---|---|
| 1 | [0.48, 1.56] | +1.6σ | −0.1σ |
| 2 | [0.62, 1.52] | +1.5σ | −0.5σ |
| 3 | [0.75, 1.51] | +1.4σ | −1.0σ |
| 4 | [0.96, 1.53] | +0.6σ | −2.6σ |
| 5 | [1.09, 1.66] | −0.8σ | **−3.9σ (outside)** |
| 6 | [1.19, 1.78] | −1.8σ | **−4.9σ (outside)** |

So **`gb1p3` is a slice of the training prior** — inside the support of every bin, an ordinary
training event — and a label-free statistic fit on ID data cannot flag an event for sitting at a
typical b_g; only a labelled two-sample test (C2ST 0.67–0.75 in Part 2) sees the *distributional*
narrowing. **`gb1p0` is genuinely out of support in the two highest-z bins**, so it is a real
misspecification — and the fact that five encoders still only reach 0.57–0.58 (per-encoder probit
shift 0.20) says the 8-D summary is only weakly sensitive to high-z-bin b_g at the per-event level;
that sensitivity is an encoder property, and no combination of five ρ = 0.70-correlated views of it
can manufacture more than the 1.15× above. `gower_vd` (variable depth added to the same mocks,
1586/1588 shared file names) leaves no trace in summary space at all (mean shift 0.00, single-encoder
AUROC 0.49). These are the honest "hard cases", and the figure of merit there is the width of the CI,
not the point estimate. (The memory note quoting a "b_g prior U[0.5,1.5]" refers to an earlier
campaign, not to this store.)

## 3. The two options

**(a) Literature — DoSE, Morningstar et al., AISTATS 2021 (PMLR 130:3232–3240, arXiv:2006.09273).**
DoSE computes a *vector* of per-input statistics under the model, fits a density to that vector on
in-distribution data only (KDE / one-class SVM in the paper), and flags low-density vectors — i.e. it
uses the joint distribution of the statistics, not each marginal tail. Instantiated here as the
Mahalanobis² of the 5-vector u (one probit per encoder; `dose5`) or the 10-vector
[u(cond_knn); u(knn)] (`dose10`) under the half-A Gaussian (mean + covariance fit on the FIT half of
`_idtest` only). A Gaussian rather than a KDE because n_eff ≈ 100 cosmologies cannot support a 5–10-D
KDE. This is the design that exploits the *disagreement structure* across encoders: an event whose
five scores are jointly atypical scores high even if no marginal is. (Bergamin et al., AISTATS 2022,
arXiv:2203.01097, is the p-value-combination cousin — Fisher over typicality + score tests — and is
already covered by the Fisher baseline.)

**(b) Own — GLS-Stouffer with the ID correlation, Lin & Sullivan 2009, AJHG 85:862–872.** T =
1ᵀΣ⁻¹u / √(1ᵀΣ⁻¹1), Σ the half-A correlation of u (shrunk 10 % to the diagonal). This is the optimal
linear combination of correlated Z-statistics for a *common* shift; it down-weights encoders that are
redundant with the others and collapses to plain Stouffer (mean u) when Σ is exchangeable. The
mechanism I hoped to exploit: if some encoders were near-duplicates and others more independent, GLS
would re-weight toward the independent ones. Measured Σ is near-exchangeable (0.68–0.74), so GLS,
Stouffer and mean-p agree within 0.002 AUROC on every variate — the honest statement of where (b) does not win.
Also run, as cheap extras: plain Stouffer and Tippett (min p, sensitive to an encoder-specific failure).

Lee et al. 2018 (arXiv:1807.03888) was rejected earlier as circular (it fits on OOD labels) and is not
reintroduced.

## 4. What DoSE's *loss* says

`dose5`/`dose10` sit **at or below 0.5** on both gb variates (0.45 / 0.42 on `gb1p0`, 0.47 / 0.43 on
`gb1p3`; 1–2.5 bootstrap sd below chance) and at 0.69–0.76 on the IA variates where the right-tailed
combiners give 0.87. Two mechanisms: (i) on a common right-tail shift the Mahalanobis² spends 4 (or 9)
of its degrees of freedom on disagreement directions that carry only noise — the usual dilution of a
radial statistic; (ii) on the gb sets the five-vector sits, if anything, *closer* to the ID mean than a
null event's, so a density-of-states score ranks it as *less* OOD. For `gb1p3` that is consistent with
the "slice of the prior" reading in §2 (a narrowing, not a shift); for `gb1p0` it says the
out-of-support high-z bias does not push the summaries into a low-density region of the encoders'
joint statistic space either. Flipping DoSE to a left tail would give ≈ 0.53–0.58 — no better than
mean-p — and choosing the tail from variate labels would be circular, so this is a diagnostic, not a
method. The disagreement structure across encoders, in short, carries no misspecification information
on this arm: the encoders disagree by their own approximation noise, which is the same on null and
variate events.

## 5. Protocol (constraints 1–5, all honoured)

* **Fit only on ID data.** Whitener, θ-regressor and residual whitener: `_train` only (split-
  independent, as in `gower_null_selfcheck.py`). Stage-1 nulls, DoSE moments, GLS Σ and every stage-2
  null: the FIT half A of a cosmology-level split of `_idtest`. The six variate files are read only to
  be scored. No combiner has any variate-dependent parameter; the *choice* of recommended method is
  by a rule fixed in advance in the script (`beats()`: must exceed mean-p on every hard variate by > 1
  bootstrap sd and not lose on their Spearman), and that rule selected mean-p.
* **Per-variate θ-dims.** `gower_nla`/`gower_nla_z` exclude `a_ia` and lack `b_ia`, so their
  conditional model is 7-dim; their null `cond_knn` scores are the 7-dim null on the same `_idtest`
  rows (`OODReference._conditional(dims)`), and the stage-2 null for those sets is built from the
  7-dim stage-1 p's. (The control's cluster-KS D is ≈ 0.58 here vs ≈ 0.8 in the Phase-4 gate — consistent with the
  gate's `ref.score(z, theta)` call conditioning on all finite dims, a_ia included, so its θ-regression
  extrapolates on the IA variate; not verified by recomputation. 0.58 is still 4× the critical value.)
* **Stage 2 is mandatory and applied to every method, baselines included.** Half-A rows enter the
  stage-2 null through leave-one-out stage-1 p's (n_ge/n with self counted = the out-of-sample formula
  on n−1 points), so the null of T carries the same cross-encoder dependence as an unseen event; this
  is what turns Bates(5)-shaped mean-p and dependence-violating Fisher into uniform p's.
* **Criterion 1 the reference way.** 60 random cosmology splits (99 fit / 100 evaluate); mean_p on the
  evaluate half within one PER-SPLIT sd of 0.5; median one-row-per-cosmology KS D (200 draws) below
  1.36/√100 = 0.136 (H0 median 0.083); positive control `gower_nla` restricted to the evaluate half's
  cosmologies must exceed 0.136. No row-level KS anywhere.
* **Shipped p-values and criteria 2–3: 2-fold cross-fit at the seed-0 split.** Every row — null *or*
  variate — of cosmology c is scored against the fold that does not contain c, so every `_idtest`
  row has an out-of-sample p (the plotting scripts need the whole null) and null-vs-variate stays
  paired (all six variates re-simulate the `_idtest` cosmologies; `gower_bgp_nla_m` is literally the
  same 1590 files, hence its AUROC is exactly 0.500 — a plumbing null, not a sixth test).
* **Baselines as specified.** `combine_pvalues_across_models("mean"/"fisher")` are computed inline as
  −mean p and −2Σlog p; after stage 2 they are rank-identical to the library calls (the recalibration
  erases the χ² transform), so the tables are the spec's baselines exactly.
* **Inner join for criterion 3.** Posterior moments are missing for ≤ 30/1592 rows on `nla`/`nla_z`
  (repeats 2–3; 1–7 rows elsewhere); those rows are dropped from ρ for that repeat only.
* **Uncertainty by cosmology.** AUROC and Spearman CIs resample `sim_id`s (each drawn id brings all
  its null rows and all its variate rows — a paired cluster bootstrap, 1000 draws); the split-to-split
  sd over the 60 random splits is reported alongside (≤ 0.007 everywhere — the split only moves
  the stage-2 ranks marginally, so cosmology sampling dominates the uncertainty). The paired design is why the gb sd (~0.04) is
  below the ~0.06 quoted for an unpaired 40-cluster comparison: cosmology-level variance cancels.
* **Truth-free at prediction time?** `cond_knn` conditions on θ (sims: truth). The real-data form is
  unchanged from Part 2: evaluate through posterior draws of θ (posterior-predictive); the combiner is
  a function of the five p's and does not care where θ came from. `knn` (in `dose10`) is truth-free.

## 6. Acceptance tables

(pasted verbatim from `artifacts/phase6/phase6_acceptance.md`)

### Criterion 1 - calibrated null on held-out half B (100 cosmologies; KS crit 0.136, H0 median 0.083); +control = gower_nla

| method | mean_p (per-split sd) | (a) | median cluster-KS D | (b) | control D | power | verdict |
|---|---|---|---|---|---|---|---|
| single encoder, avg (baseline) | 0.4977 (0.0305) | ok | 0.086-0.092 | ok | 0.51-0.57 | ok | PASS (5 encoders) |
| single encoder, best [oracle] | 0.4977 (0.0305) | ok | 0.086-0.092 | ok | 0.51-0.57 | ok | PASS (5 encoders) |
| mean-p (baseline) | 0.4973 (0.0321) | ok | 0.090 | ok | 0.58 | ok | PASS |
| Fisher (baseline) | 0.4970 (0.0320) | ok | 0.089 | ok | 0.58 | ok | PASS |
| (a) DoSE-5 Mahalanobis | 0.4961 (0.0194) | ok | 0.084 | ok | 0.36 | ok | PASS |
| (a) DoSE-10 (knn+cond) | 0.4898 (0.0175) | ok | 0.085 | ok | 0.50 | ok | PASS |
| (b) GLS-Stouffer (Lin-Sullivan) | 0.4971 (0.0321) | ok | 0.091 | ok | 0.58 | ok | PASS |
| Stouffer | 0.4973 (0.0319) | ok | 0.090 | ok | 0.59 | ok | PASS |
| Tippett (min p) | 0.4958 (0.0321) | ok | 0.089 | ok | 0.58 | ok | PASS |

### Criterion 2 - AUROC(_idtest vs variate) +- by-cosmology bootstrap sd  [split-to-split sd in brackets]

| method | gower_bgp_nla_m (n_cos=199) | gower_nla (n_cos=199) | gower_nla_z (n_cos=199) | gower_vd (n_cos=199) | gower_gb1p0 (n_cos=40) | gower_gb1p3 (n_cos=40) |
|---|---|---|---|---|---|---|
| single encoder, avg (baseline) | 0.500 +- 0.000 | 0.847 +- 0.011 | 0.851 +- 0.011 | 0.492 +- 0.009 | 0.568 +- 0.037 | 0.532 +- 0.041 |
| single encoder, best [oracle] | 0.500 +- 0.000 (r1) | 0.859 +- 0.010 (r3) | 0.861 +- 0.011 (r2) | 0.497 +- 0.009 (r1) | 0.582 +- 0.037 (r1) | 0.562 +- 0.040 (r4) |
| mean-p (baseline) | 0.500 +- 0.000 [0.000] | 0.866 +- 0.011 [0.001] | 0.872 +- 0.011 [0.001] | 0.490 +- 0.008 [0.000] | 0.580 +- 0.037 [0.002] | 0.537 +- 0.043 [0.002] |
| Fisher (baseline) | 0.500 +- 0.000 [0.000] | 0.865 +- 0.011 [0.001] | 0.872 +- 0.010 [0.001] | 0.491 +- 0.008 [0.000] | 0.579 +- 0.037 [0.002] | 0.536 +- 0.041 [0.002] |
| (a) DoSE-5 Mahalanobis | 0.500 +- 0.000 [0.000] | 0.691 +- 0.015 [0.004] | 0.700 +- 0.015 [0.004] | 0.512 +- 0.010 [0.001] | 0.449 +- 0.032 [0.006] | 0.473 +- 0.040 [0.006] |
| (a) DoSE-10 (knn+cond) | 0.500 +- 0.000 [0.000] | 0.749 +- 0.012 [0.004] | 0.758 +- 0.012 [0.004] | 0.509 +- 0.009 [0.001] | 0.416 +- 0.034 [0.006] | 0.434 +- 0.037 [0.007] |
| (b) GLS-Stouffer (Lin-Sullivan) | 0.500 +- 0.000 [0.000] | 0.868 +- 0.010 [0.001] | 0.873 +- 0.010 [0.001] | 0.490 +- 0.008 [0.000] | 0.579 +- 0.036 [0.002] | 0.539 +- 0.041 [0.002] |
| Stouffer | 0.500 +- 0.000 [0.000] | 0.867 +- 0.010 [0.001] | 0.874 +- 0.011 [0.001] | 0.489 +- 0.008 [0.000] | 0.579 +- 0.036 [0.002] | 0.537 +- 0.039 [0.002] |
| Tippett (min p) | 0.500 +- 0.000 [0.000] | 0.856 +- 0.011 [0.002] | 0.867 +- 0.010 [0.002] | 0.496 +- 0.008 [0.001] | 0.579 +- 0.036 [0.003] | 0.532 +- 0.042 [0.002] |

### Criterion 3 - Spearman rho(-log p, |z_post(truth)|) for (omega_m, sigma_8), mean over the 5 posteriors, +- by-cosmology bootstrap sd

| method | gower_bgp_nla_m | gower_nla | gower_nla_z | gower_vd | gower_gb1p0 | gower_gb1p3 |
|---|---|---|---|---|---|---|
| single encoder, avg (baseline) | +0.106+-0.036, +0.087+-0.027 | +0.385+-0.027, +0.535+-0.021 | +0.357+-0.027, +0.534+-0.022 | +0.071+-0.034, +0.095+-0.032 | +0.045+-0.111, +0.207+-0.118 | +0.080+-0.091, -0.071+-0.082 |
| single encoder, best [oracle] | +0.100+-0.035, +0.087+-0.029 (r2) | +0.394+-0.026, +0.546+-0.021 (r4) | +0.370+-0.026, +0.551+-0.022 (r4) | +0.083+-0.035, +0.100+-0.031 (r3) | +0.097+-0.116, +0.219+-0.113 (r2) | +0.123+-0.098, +0.016+-0.091 (r3) |
| mean-p (baseline) | +0.101+-0.040, +0.081+-0.031 | +0.406+-0.027, +0.535+-0.022 | +0.373+-0.026, +0.543+-0.021 | +0.074+-0.038, +0.093+-0.032 | +0.035+-0.128, +0.195+-0.128 | +0.064+-0.102, -0.111+-0.085 |
| Fisher (baseline) | +0.105+-0.036, +0.084+-0.031 | +0.406+-0.028, +0.539+-0.022 | +0.377+-0.028, +0.552+-0.024 | +0.077+-0.035, +0.093+-0.033 | +0.032+-0.127, +0.182+-0.130 | +0.063+-0.105, -0.100+-0.091 |
| (a) DoSE-5 Mahalanobis | +0.015+-0.031, +0.027+-0.029 | +0.284+-0.028, +0.425+-0.025 | +0.275+-0.027, +0.449+-0.024 | +0.001+-0.032, -0.012+-0.025 | -0.013+-0.108, -0.029+-0.114 | -0.004+-0.115, -0.027+-0.113 |
| (a) DoSE-10 (knn+cond) | +0.010+-0.030, +0.053+-0.027 | +0.275+-0.030, +0.416+-0.026 | +0.225+-0.027, +0.401+-0.026 | +0.019+-0.029, +0.007+-0.025 | -0.011+-0.113, -0.003+-0.119 | +0.030+-0.106, +0.112+-0.100 |
| (b) GLS-Stouffer (Lin-Sullivan) | +0.101+-0.036, +0.081+-0.031 | +0.404+-0.027, +0.543+-0.022 | +0.379+-0.026, +0.555+-0.023 | +0.074+-0.039, +0.092+-0.034 | +0.037+-0.126, +0.198+-0.132 | +0.075+-0.110, -0.106+-0.084 |
| Stouffer | +0.103+-0.038, +0.082+-0.033 | +0.406+-0.029, +0.539+-0.021 | +0.376+-0.026, +0.550+-0.022 | +0.075+-0.038, +0.091+-0.035 | +0.042+-0.120, +0.200+-0.131 | +0.076+-0.110, -0.106+-0.088 |
| Tippett (min p) | +0.111+-0.037, +0.087+-0.030 | +0.372+-0.030, +0.533+-0.023 | +0.356+-0.028, +0.535+-0.025 | +0.077+-0.036, +0.089+-0.034 | +0.031+-0.135, +0.147+-0.142 | +0.097+-0.107, -0.082+-0.091 |


## 7. Reading the tables honestly

* **Criterion 1** passes for every combiner and both baselines on (a) and (b) with the control firing —
  recalibration makes this near-automatic, as the spec anticipated: mean_p 0.490–0.498 (per-split sd
  0.02–0.03), median cluster-KS D 0.084–0.092 vs crit 0.136 (H0 median 0.083), control D 0.36–0.59. Each
  of the five single encoders passes too (D 0.086–0.092), reproducing the Phase-4 gate at this split
  size. (In a 4-split smoke pass two single encoders tripped (a) by a hair — the per-split sd at 4
  splits is not a tolerance; at 60 splits nothing trips.)
* **Criterion 2.** `nla`/`nla_z`: every right-tailed combiner 0.87 ± 0.01 vs 0.85 single-encoder average
  — the +0.02 is the 1.15× SNR gain and it is uninformative (already easy). `gb1p0` 0.58, `gb1p3` 0.54,
  `vd` 0.49 for mean-p, GLS, Stouffer, Fisher and Tippett alike, all within one sd of the single-encoder
  average and of each other. DoSE loses everywhere (§4). No option beats the recalibrated mean-p on a
  hard variate by more than one sd — and, per the spec, an apparent gain inside the CI is not a result.
  **Where mean-p does not win:** the *oracle* best single encoder is nominally ahead of it on `gb1p3`
  (0.562 vs 0.537 AUROC; ρ_σ8 +0.02 vs −0.11) — inside one sd on both — but the winning encoder
  differs per variate (r1, r3, r2, r1, r1, r4 in the table), so there is no label-free way to pick it;
  selecting it would be selection on variate labels, which constraint 1 forbids for a shipped method.
* **Criterion 3.** On the IA variates ρ ≈ +0.40 / +0.54 for every right-tailed combiner (single-encoder
  +0.38 / +0.53). On the hard variates ρ ranges −0.11…+0.20 with sd ≈ 0.04–0.10 and is identical across
  right-tailed combiners (single-encoder average included); DoSE is ≈ 0. The combined score is a per-event predictor of posterior bias exactly where
  the single-encoder score already is, and nowhere else.

## 8. What would actually move the hard variates

Not a combiner. `gb1p3` is inside the training prior in every bin: "b_g = 1.3" is not misspecification
for this chain, and the posterior-calibration axis (Phase 3) agrees (`gb1p3` 0.040 vs in-dist 0.004);
its only summary-space signature is a narrowing that a labelled two-sample test sees and a per-event
score cannot. `gb1p0` *is* out of support (bins 5–6 at −3.9σ/−4.9σ), and its weak per-event signature
(0.57–0.58) is an encoder-sensitivity limit: the ID-only route to more power is conditioning on b_g
as well — the sc8a1 15-parameter chain infers `b_g_bin1..6`, and its `cond_knn` residual would treat a
pinned high-z b_g as a parameter shift (visible in the posterior) rather than as unexplained scatter —
or a probe that pins b_g far enough out (0.7 / 1.5, the GLASS ladder's 0.90 / 0.78) that the
summaries actually move. `vd` has no summary-space signature in any of the five encoders and no
combination of them can invent one.
