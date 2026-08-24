# BGP campaign — Gower NLE production results

Gower-fidelity NLE (M4b) transferred from the GLASS foundation, on the `b_g`-marginalised (`bgp`)
campaign. All numbers below are from **5 independently-seeded ens9 stacks** (`r0`–`r4`), each
evaluated on the **same 3968 test points** (the 199 on-disk cosmologies of the 200-id fixed test
set × 20 mocks) under the **matched Gower prior**.

Data: `gower_mocks_nla_m_novd_bgp` → prebake `gower_bgp_nla_m_f16_sc8a1_fwhm4_lmin56_lcut1400`
(40 725 files, `ok=40725 skip=0 bad=0`). Split: 511 cosmologies → 240 train / 60 val / 199 test.

---

## 1. Production headline

| quantity | value |
|---|---|
| **FoM** | **33.91 ± 2.65** |
| **TARP 9-D calibration error** | **0.00212 ± 0.00029** |
| **Mahalanobis mean** | **2.982 ± 0.013** (≈ 2.92 expected for correct 9-D coverage) |
| **± S8** | **0.02349 ± 0.00019** (0.8 % seed spread) |
| **± Ω_m** | **0.03159 ± 0.00123** |
| **± σ₈** | **0.03851 ± 0.00194** |
| **± w₀** | **0.14775 ± 0.00097** (0.7 %) |

Per repeat:

| rep | FoM | TARP-9D | Mahal | ±S8 | ±Ω_m | ±σ₈ | ±w₀ |
|---|---|---|---|---|---|---|---|
| r0 | 33.63 | 0.00182 | 2.975 | 0.02365 | 0.03168 | 0.03884 | 0.14819 |
| r1 | 36.75 | 0.00249 | 2.989 | 0.02327 | 0.03010 | 0.03609 | 0.14788 |
| r2 | 35.82 | 0.00178 | 3.003 | 0.02326 | 0.03092 | 0.03747 | 0.14885 |
| r3 | 34.24 | 0.00240 | 2.971 | 0.02368 | 0.03146 | 0.03824 | 0.14786 |
| r4 | 29.09 | 0.00213 | 2.971 | 0.02361 | 0.03381 | 0.04192 | 0.14595 |

**Calibration is confirmed twice, independently.** TARP's full-9-D error is ~0.002 on every repeat
(0 = perfect), and the mean Mahalanobis distance sits on the value expected for correct 9-D
coverage. Per-parameter TARP is ≤ 0.025 everywhere except **w₀ at 0.105** — the one weakly
constrained direction, where miscalibration is least surprising and least costly.

## 2. Forecast under the KiDS S8 analytic **wCDM** prior

100 test cosmologies (sim_ids 193–470, one mock each) × 20 000 MCMC samples, 5 repeats:

| parameter | mean 68 % credible interval | seed spread |
|---|---|---|
| **S8** | **± 0.0248** | 1.20 % |
| **Ω_m** | **± 0.0354** | 3.04 % |
| **w₀** | **± 0.1731** | 0.71 % |
| **σ₈** | **± 0.0428** | 4.65 % |

These are uniformly **wider** than the matched-prior numbers in §1, as a less-informative prior
should be — an internal consistency check across two independent pipelines.

> ⚠️ **Bias/z-scores from these dumps are not a model diagnostic.** Inference here uses a *different*
> prior from the one that generated the test set, so shrinkage toward it necessarily biases against
> Gower-drawn truths, and TARP is defined w.r.t. the generating prior. Under the **matched** prior
> the biases collapse to ≈ 0 (S8 −0.0017, Ω_m −0.00000) versus −0.006 / −0.0067 here.

## 3. Constraining power vs the pre-`bgp` analysis

| | old | new (matched prior) | change |
|---|---|---|---|
| ± Ω_m | 0.030 | **0.0316** | **+5 %** |
| ± w₀ | 0.14 | **0.1478** | **+5 %** |
| ± S8 | 0.018 | **0.0235** | **+30 %** |

**Ω_m and w₀ are essentially unchanged; the entire degradation sits in S8.** That is the signature
expected from marginalising `b_g`: galaxy bias is an *amplitude* nuisance degenerate with the
lensing amplitude, so it should cost S8 and spare shape and geometry.

**The "old numbers were a subset near w = −1" hypothesis was tested and does not explain it.**
`w0`'s preset box is (−1.0097, −0.3391), so w = −1 is the *lower edge* — a natural worry that edge
cosmologies get artificially tight posteriors. Splitting the 100 test cosmologies by true w₀
(averaged over the repeats):

| subset | N | ±S8 | ±Ω_m | ±w₀ |
|---|---|---|---|---|
| all | 100 | 0.02474 | 0.03489 | 0.17353 |
| w < −0.9 | 25 | 0.02495 | 0.03507 | **0.16173** |
| w > −0.7 | 38 | 0.02519 | 0.03421 | 0.18315 |

Restricting to w near −1 tightens **w₀ by ~7 %** (prior-edge truncation, as expected) but leaves
**S8 and Ω_m flat to < 1 %**. So the subset explains a little of the w₀ difference and none of S8's.

> ⚠️ Before quoting the +30 %: confirm the old run marginalised `b_g` and used the same estimator
> arm and 9-param vector. If it did not marginalise `b_g`, +30 % *is* the price of that
> marginalisation and this is exactly the intended comparison.

## 4. Ablation — does adapting the compressor to Gower help? **No.**

Fine-tune the foundation encoder on Gower (NPE, 1 member), then fine-tune the same Stage-A NLE flow
on *that* encoder's embeddings. Both arms GPU-trained, both from repeat r4, same 3968 test points:

| metric | r4 (frozen encoder) | adapted encoder | change |
|---|---|---|---|
| **FoM** | **29.09** | **29.07** | **−0.1 %** |
| FoM(Ω_m, σ₈) | 15.35 | 15.33 | −0.1 % |
| Mahalanobis | 2.971 | 2.975 | +0.1 % |
| ±S8 | 0.02361 | 0.02379 | +0.8 % |
| ±Ω_m | 0.03381 | 0.03356 | −0.7 % |
| ±σ₈ | 0.04192 | 0.04108 | −2.0 % |
| ±w₀ | 0.14595 | 0.14841 | +1.7 % |

FoM is identical to 0.1 %; every width moves ≤ 2 % and **in both directions**. For scale, the
baseline's own seed-to-seed spread is FoM 29.09–36.75 (26 %) and ±Ω_m 12 %. **No gain — the current
approach needs no change.**

> ⛔ Do **not** read `test_log_prob` −0.9939 → +0.4526 as an improvement. That is the NLE density of
> the *whitened embedding*; the adapted arm has a different encoder, so the two are in different
> coordinate systems separated by a log|det J| offset.

### 4b. Warm-starting the NLE flow onto the adapted encoder is **worth a lot** (corrected)

> ⚠️ **This section previously said the opposite.** It read the val numbers with the sign inverted.
> `val_log_prob` is what `ModelCheckpoint` monitors with `mode="min"` — it is a **loss**, so
> **lower is better**. The warm-started arm's 1.76 beats the random-init arm's 6.72; the earlier
> text took 6.72 as the winner. The completed evals settle it independently and in the same
> direction, so the corrected reading is the one below.

Same encoder (r4's, adapted on Gower), same whitener, same 150-epoch budget, same 3968 test points;
the two arms differ **only** in `load_pretrained_flow`:

| | warm-started (A3) | random init (A4) | ratio |
|---|---|---|---|
| **best val (lower better)** | **1.761** | 6.723 | — |
| **FoM** | **29.07** | **18.96** | **1.53×** |
| ± S8 | **0.02379** | 0.05540 | 2.33× wider |
| ± Ω_m | **0.03356** | 0.08001 | 2.38× wider |
| ± σ₈ | **0.04108** | 0.10464 | 2.55× wider |
| ± w₀ | **0.14841** | 0.30224 | 2.04× wider |
| Mahalanobis | 2.975 | 2.871 | both ≈ calibrated |

**The pre-trained GLASS flow is carrying most of the constraining power.** Random init stays
*calibrated* (Mahalanobis 2.87, near the 2.92 expected for 9-D) but is **~2.3× less informative on
every parameter** — it is honest about knowing less, which is exactly the failure mode a
calibration test alone would not flag.

The warm start does open at a large gap — 145–162 nats, because the Stage-A flow lives in the old
encoder's whitened coordinates — but it recovers from it and still ends far ahead. "Opens badly"
and "ends badly" are not the same thing, and the earlier text conflated them.

> Caveat, unchanged: A3 was GPU-trained and A4 CPU-trained, so the pair is hardware-confounded.
> It does not threaten this conclusion — bf16-vs-fp32 rounding cannot produce a 2.3× posterior
> width — and val and FoM agree in direction. It *would* matter for a small effect; it does not
> for this one.

## 5. Gotchas worth keeping

* **Evals are all-or-nothing.** `evaluate_best_checkpoint` writes its JSON only after all 124 MCMC
  tasks finish — a mid-run counter yields nothing.
* **MCMC completions arrive in bursts,** one chain-length apart (~3–4 h). A flat task counter for
  <4 h is wave structure, not a stall. Take the elapsed time from the tqdm bar's own field, not from
  when you first noticed the job.
* **`show_progress_bars=True` puts sbi's own MCMC bar on stderr** — a far better liveness probe than
  the joblib task counter. Its "N chains" is the *batch*, not `num_chains`.
* **Augmentation randomises the train split only** (`transform=transform` for train, `None` for
  val/test), so repeated runs re-draw train embeddings. On a chain whose whitener was fit on a
  *different* encoder, the near-null PCs amplify this into a ~0.6-nat run-to-run noise floor on val.
* **Stage-A NLE vals are not comparable across repeats or encoder versions.** Select on FoM and
  calibration.

---

*Analysis tool: `.claude/runs/training-runs/production-training-runs/artifacts/forecast_from_samples.py`
(self-tested against a synthetic npz with an analytically known answer).*
