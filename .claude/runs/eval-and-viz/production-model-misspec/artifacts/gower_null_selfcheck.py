"""SPLIT-HALF null self-check for the summary-space OOD calibration.

This is the gate on Phase 6: the multi-encoder combiner is built on top of this
calibration, so if the null is not uniform the combiner's p-values mean nothing.

Why split-half, and NOT the naive version
-----------------------------------------
`summaries.py` writes `_idtest` with no `ood_*` fields, and the tempting "self-check" --
fit `OODReference` with `z_id = _idtest` and then score that same `_idtest` -- is a
TAUTOLOGY. `empirical_pvalues` ranks each query score against the null scores, so scoring
the very set that defined the null returns p = rank/(n+1), which is uniform BY
CONSTRUCTION regardless of whether anything is calibrated. It cannot fail, so it tests
nothing.

The real check fits the null on one half of `_idtest` and scores the OTHER half, splitting
at the COSMOLOGY level and never by row order: the rows are augmentations (footprint
rotations x shape-noise realisations) of a much smaller set of cosmologies, so a row-order
split would put augmentations of the SAME cosmology on both sides and leak. (An earlier
version used the even/odd `sim_id` parity split; see correction (2) for why the verdict is
now taken over many RANDOM cosmology splits, with parity kept only as a diagnostic.)

Two corrections that took three tries to get right  (2026-09-07)
----------------------------------------------------------------
(1) THE TEST MUST BE CLUSTER-AWARE. `_idtest` is 1590 rows over only 199 cosmologies
    (~8 correlated augmentations each), so half B is ~710 rows but only ~89 INDEPENDENT
    draws. A row-level `kstest` on all 710 rows uses a critical value of 1.36/sqrt(710)
    ~ 0.051; the honest one at n_eff ~ 89 is ~0.144. The first version of this script ran
    the row-level test, measured D = 0.047-0.080 (all comfortably inside 0.144) and
    "failed" 8/10 cells. That was a TEST ARTIFACT. Fix: draw ONE row per half-B cosmology
    and KS that ~89-point sample, repeated over many draws.

(2) THE CLUSTER-AWARE TEST STILL CANNOT BE READ AGAINST ITS NOMINAL LEVEL. Measured on
    this data (and recorded in log.md):
      * RANDOM cosmology-level splits give mean_p = 0.4954-0.5063 across all ten
        (repeat x score) cells -- centred on 0.500, so the null IS calibrated -- while
        the reject-fraction statistic's OWN null distribution is centred at ~0.10
        (sd ~0.11), NOT at the nominal 0.05. Judging a cell against 0.05, or even
        against 0.10, fails ~half of correctly-calibrated cells by construction. The
        second version of this script did exactly that and "failed" 5/10 cells.
      * The single even/odd PARITY split lands low in that reference (~10-25th
        percentile, mean_p ~ 0.46-0.49). That is one draw, not evidence: the five
        encoders see the SAME 199 cosmologies through the SAME split, so the ten cells
        are ~1-2 independent observations. Parity is therefore reported as a diagnostic
        and is no longer the basis of the verdict.
    Fix: judge the calibration against a THEORETICAL reference, over RANDOM cosmology-level
    splits rather than the single parity split. Two criteria per cell:
      (a) averaged over `--n-splits` random cosmology splits, mean_p must sit within one
          per-split sd of 0.5. The tolerance is the PER-SPLIT sd, not sd/sqrt(n_splits):
          the splits re-use the same 1590 rows, so more splits buy no extra independence
          and shrinking the tolerance with them would spuriously reject any finite sample.
      (b) the cluster-KS D at one row per cosmology must be below the 0.05 critical value
          1.36/sqrt(n_cos); the H0 median 0.83/sqrt(n_cos) is printed beside it.
    Both are absolute, so neither is circular. The parity split and the naive row-level
    number are still printed, as labelled DIAGNOSTICS with no pass/fail attached: the five
    encoders see the same 199 cosmologies through the same even/odd split, so their ten
    cells are ~1-2 independent observations of one draw, not ten.

This is cheap because `OODReference`'s whitener and conditional-residual model are fit on
`_train` ALONE: `ref.score()` is split-INDEPENDENT. We score all 1590 `_idtest` rows once
per repeat and then re-partition, so hundreds of random splits cost almost nothing.

POSITIVE CONTROL. A gate that cannot fail is worthless, so the same protocol is applied to
`gower_nla` (single-encoder AUROC ~0.94): its cluster-KS D must EXCEED the same critical
value. Measured D ~ 0.77-0.82 against a crit of ~0.144, so the ~89-point test has ample
power and a PASS on the in-distribution half is meaningful.
"""

import argparse
import os
import sys

import numpy as np
from scipy.stats import kstest

sys.path.insert(0, "/home/alex/work/glass_gower_transfer")

from src.ml.eval.ood import OODReference, empirical_pvalues  # noqa: E402

SCORES = ["knn", "cond_knn"]

# KS critical value / H0 median coefficients at level 0.05
KS_CRIT, KS_MED = 1.36, 0.83


def load(path):
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def icc_by_group(x, groups):
    """One-way random-effects ICC(1) + Kish effective sample size.

    ICC ~ 0 => rows are effectively independent and a row-level KS would be valid.
    ICC ~ 1 => all the information is at the cosmology level; n_eff ~ n_cosmologies.
    """
    x = np.asarray(x, dtype=np.float64)
    uniq = np.unique(groups)
    k = len(uniq)
    if k < 2:
        return float("nan"), float("nan")
    means = np.array([x[groups == g].mean() for g in uniq])
    ns = np.array([float((groups == g).sum()) for g in uniq])
    ssw = sum(((x[groups == g] - m) ** 2).sum() for g, m in zip(uniq, means))
    n_bar, grand = ns.mean(), x.mean()
    msw = ssw / max(len(x) - k, 1)
    msb = (ns * (means - grand) ** 2).sum() / max(k - 1, 1)
    denom = msb + (n_bar - 1.0) * msw
    icc = float(np.clip((msb - msw) / denom, 0.0, 1.0)) if denom > 0 else 0.0
    n_eff = len(x) / (1.0 + (n_bar - 1.0) * icc) if icc > 0 else float(len(x))
    return icc, float(n_eff)


def cluster_stats(pvals, groups, n_draws, rng):
    """KS-vs-uniform on ONE row per cosmology, repeated n_draws times.

    Returns (median KS stat, reject fraction at 0.05, n independent points per draw).
    """
    uniq = np.unique(groups)
    idx_by_g = [np.flatnonzero(groups == g) for g in uniq]
    stats, ps = np.empty(n_draws), np.empty(n_draws)
    for m in range(n_draws):
        pick = np.fromiter((ix[rng.integers(len(ix))] for ix in idx_by_g), int, len(idx_by_g))
        ks = kstest(pvals[pick], "uniform")
        stats[m], ps[m] = ks.statistic, ks.pvalue
    return float(np.median(stats)), float((ps < 0.05).mean()), len(uniq)


def split_stats(s_all, sim, maskA, n_draws, rng):
    """mean_p, median cluster-KS D and cluster reject-fraction for one A/B split."""
    B = ~maskA
    p = empirical_pvalues(s_all[maskA], s_all[B])
    D, rej, npts = cluster_stats(p, sim[B], n_draws, rng)
    return {"mean_p": float(p.mean()), "D_med": D, "reject_frac": rej,
            "n_cos": npts, "n_rows": int(B.sum()),
            "row_D": float(kstest(p, "uniform").statistic)}


def pct_of(value, ref):
    return 100.0 * float((np.asarray(ref) < value).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="checkpoints/<exp>/summaries")
    ap.add_argument("--repeats", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--match-fmt", default="ncosmo300_{r}")
    ap.add_argument("--max-train", type=int, default=20000)
    ap.add_argument("--n-draws", type=int, default=200,
                    help="cluster draws (one row per cosmology) per KS test")
    ap.add_argument("--n-splits", type=int, default=60,
                    help="random cosmology splits forming the reference distribution")
    ap.add_argument("--control-variate", default="gower_nla",
                    help="known-OOD variate used as the positive control; '' disables")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=== SPLIT-HALF NULL SELF-CHECK ===")
    print(f"    null fit on even sim_ids, scored on odd; {args.n_draws} cluster draws per KS.")
    print(f"    VERDICT: (a) mean_p over {args.n_splits} random cosmology splits within one")
    print( "             per-split sd of 0.5; (b) cluster-KS D below 1.36/sqrt(n_cos).")
    print( "    Parity split + naive row-level D are printed as DIAGNOSTICS only.\n")
    verdicts, controls = {}, {}

    for r in args.repeats:
        m = args.match_fmt.format(r=r)
        ftr = os.path.join(args.root, "_train", f"summaries_{m}.npz")
        fid = os.path.join(args.root, "_idtest", f"summaries_{m}.npz")
        if not (os.path.exists(ftr) and os.path.exists(fid)):
            print(f"[r{r}] SKIP - missing {ftr if not os.path.exists(ftr) else fid}")
            continue

        tr, idt = load(ftr), load(fid)
        if "sim_ids" not in idt:
            print(f"[r{r}] SKIP - _idtest npz has no sim_ids, cannot split by cosmology.")
            continue

        sim = idt["sim_ids"].astype(np.int64)
        uniq = np.unique(sim)
        mA = (sim % 2 == 0)
        nA_cos = len(np.unique(sim[mA]))
        overlap = len(np.intersect1d(np.unique(tr["sim_ids"]), uniq))
        print(f"[r{r}] _idtest n={len(sim)} / {len(uniq)} cosmologies | "
              f"parity halfA: {mA.sum()} rows / {nA_cos} cos | "
              f"train-idtest cosmology overlap={overlap} {'<-- LEAK!' if overlap else '(clean)'}")

        # fit ONCE; score ALL idtest rows once. Both are split-independent because the
        # whitener and the conditional model are fit on _train alone.
        ref = OODReference.fit(
            tr["z"], z_id=idt["z"], theta_train=tr.get("theta"), theta_id=idt["theta"],
            max_train=args.max_train,
        )
        s_id = ref.score(idt["z"], idt["theta"])

        ctl = None
        if args.control_variate:
            fct = os.path.join(args.root, args.control_variate, f"summaries_{m}.npz")
            if os.path.exists(fct):
                c = load(fct)
                ctl = {"sim": c["sim_ids"].astype(np.int64), "s": ref.score(c["z"], c["theta"])}

        row, crow = {}, {}
        for name in SCORES:
            if name not in s_id:
                print(f"[r{r}]   {name}: unavailable")
                continue
            s_all = np.asarray(s_id[name], dtype=np.float64)
            icc, n_eff = icc_by_group(s_all[~mA], sim[~mA])

            obs = split_stats(s_all, sim, mA, args.n_draws, np.random.default_rng(args.seed))

            # reference distribution: random cosmology splits of the same size
            rng = np.random.default_rng(args.seed + 1000 + r)
            refd = [split_stats(s_all, sim,
                                np.isin(sim, rng.choice(uniq, nA_cos, replace=False)),
                                args.n_draws, rng)
                    for _ in range(args.n_splits)]
            mp_ref = np.array([d["mean_p"] for d in refd])
            rj_ref = np.array([d["reject_frac"] for d in refd])

            ncos = obs["n_cos"]
            crit, hmed = KS_CRIT / np.sqrt(ncos), KS_MED / np.sqrt(ncos)
            # (a) random-split mean_p centred on 0.5, tolerance = the PER-SPLIT sd
            bias, tol = abs(mp_ref.mean() - 0.5), mp_ref.std()
            ok_a = bias <= tol
            # (b) cluster-KS D below the 0.05 critical value at n_cos independent points
            ok_b = obs["D_med"] < crit
            ok = ok_a and ok_b
            row[name] = ok
            print(f"[r{r}]   {name:>9}: ICC={icc:.3f} n_eff={n_eff:6.1f} | "
                  f"(a) random-split mean_p={mp_ref.mean():.4f}+-{mp_ref.std():.4f} "
                  f"|bias|={bias:.4f} vs tol={tol:.4f} {'ok' if ok_a else 'BAD'} | "
                  f"(b) D_med={obs['D_med']:.4f} vs crit={crit:.3f} (H0 median {hmed:.3f}) "
                  f"{'ok' if ok_b else 'BAD'}  {'PASS' if ok else '*** FAIL ***'}")
            print(f"[r{r}]   {'':>9}  [diagnostic, no pass/fail] parity split: "
                  f"mean_p={obs['mean_p']:.4f} (pct {pct_of(obs['mean_p'], mp_ref):.0f} of the "
                  f"random-split reference) reject_frac={obs['reject_frac']:.3f} "
                  f"(ref {rj_ref.mean():.3f}+-{rj_ref.std():.3f})")
            print(f"[r{r}]   {'':>9}  [diagnostic] naive row-level D={obs['row_D']:.4f} on "
                  f"n={obs['n_rows']} rows -- ANTICONSERVATIVE: crit~"
                  f"{KS_CRIT / np.sqrt(obs['n_rows']):.3f} vs honest ~{crit:.3f} at "
                  f"n_eff={ncos} cosmologies ({KS_CRIT / np.sqrt(obs['n_rows']) / crit:.1f}x too strict)")

            if ctl is not None and name in ctl["s"]:
                cs = np.asarray(ctl["s"][name], dtype=np.float64)
                cB = (ctl["sim"] % 2 != 0)
                pc = empirical_pvalues(s_all[mA], cs[cB])
                Dc, rejc, nc = cluster_stats(pc, ctl["sim"][cB], args.n_draws,
                                             np.random.default_rng(args.seed))
                fires = Dc > KS_CRIT / np.sqrt(nc)
                crow[name] = fires
                print(f"[r{r}]   {'':>9}  [+control {args.control_variate}] "
                      f"mean_p={pc.mean():.3f} D_med={Dc:.4f} vs crit="
                      f"{KS_CRIT / np.sqrt(nc):.3f} reject_frac={rejc:.3f} "
                      f"(n={nc} cos)  {'POWER OK' if fires else '*** NO POWER ***'}")
        verdicts[r] = row
        if crow:
            controls[r] = crow
        print()

    print("=== VERDICT ===")
    if not verdicts:
        print("no repeats evaluated.")
        return 1
    allpass = True
    for name in SCORES:
        res = {r: v.get(name) for r, v in verdicts.items() if name in v}
        bad = [r for r, ok in res.items() if not ok]
        allpass &= not bad and bool(res)
        print(f"  {name:>9}: {len(res) - len(bad)}/{len(res)} repeats calibrated"
              + (f"  FAILING: {bad}" if bad else ""))
    if controls:
        print("  --- positive control (D must EXCEED the critical value; a pass here means "
              "the test has no power) ---")
        for name in SCORES:
            res = {r: v.get(name) for r, v in controls.items() if name in v}
            if not res:
                continue
            weak = [r for r, f in res.items() if not f]
            allpass &= not weak
            print(f"  {name:>9}: {len(res) - len(weak)}/{len(res)} repeats rejected the control"
                  + (f"  NO POWER: {weak}" if weak else ""))
    print("\n" + ("GATE PASSED - the null is calibrated and the test has power; "
                  "Phase 6 may proceed."
                  if allpass else
                  "GATE FAILED - diagnose before building the multi-encoder combiner."))
    return 0 if allpass else 1


if __name__ == "__main__":
    raise SystemExit(main())
