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

The real check fits the null on one half of `_idtest` and scores the OTHER half:

  * split by `sim_id` PARITY, never by row order. The rows are augmentations (footprint
    rotations x shape-noise realisations) of a much smaller set of cosmologies, so
    consecutive rows are strongly correlated; a row-order split would put augmentations of
    the SAME cosmology on both sides and leak, making the null look better than it is.
  * pass = the held-out half's p-values are KS-uniform (p > 0.05) for `knn` and
    `cond_knn`, on every repeat.

A FAILURE here means the two halves are not exchangeable (a correlated split, or too few
cosmologies per half), and is a bug to fix before Phase 6 -- not a verdict on the metric.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "/home/alex/work/glass_gower_transfer")

from src.ml.eval.ood import OODReference, empirical_pvalues, null_uniformity  # noqa: E402

SCORES = ["knn", "cond_knn"]


def load(path):
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="checkpoints/<exp>/summaries")
    ap.add_argument("--repeats", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--match-fmt", default="ncosmo300_{r}")
    ap.add_argument("--max-train", type=int, default=20000)
    args = ap.parse_args()

    print("=== SPLIT-HALF NULL SELF-CHECK (fit on even sim_ids, score odd) ===\n")
    verdicts = {}

    for r in args.repeats:
        m = args.match_fmt.format(r=r)
        ftr = os.path.join(args.root, "_train", f"summaries_{m}.npz")
        fid = os.path.join(args.root, "_idtest", f"summaries_{m}.npz")
        if not (os.path.exists(ftr) and os.path.exists(fid)):
            print(f"[r{r}] SKIP — missing {ftr if not os.path.exists(ftr) else fid}")
            continue

        tr, idt = load(ftr), load(fid)
        if "sim_ids" not in idt:
            print(f"[r{r}] SKIP — _idtest npz has no sim_ids, cannot split by cosmology.")
            continue

        sim = idt["sim_ids"].astype(np.int64)
        A = (sim % 2 == 0)          # fit the null on these cosmologies
        B = ~A                      # score these -- disjoint COSMOLOGIES, not just rows
        nA_cos, nB_cos = len(np.unique(sim[A])), len(np.unique(sim[B]))
        print(f"[r{r}] _idtest n={len(sim)}  halfA: {A.sum()} rows / {nA_cos} cosmologies | "
              f"halfB: {B.sum()} rows / {nB_cos} cosmologies")
        if A.sum() < 30 or B.sum() < 30:
            print(f"[r{r}] SKIP — a half is too small to test uniformity.")
            continue

        ref = OODReference.fit(
            tr["z"], z_id=idt["z"][A],
            theta_train=tr.get("theta"), theta_id=idt["theta"][A],
            max_train=args.max_train,
        )
        s_B = ref.score(idt["z"][B], idt["theta"][B])

        row = {}
        for name in SCORES:
            # the conditional null lives in the lazily-built per-dims cache, so pull the
            # null the same way `evaluate` does rather than assuming null_scores has it
            if name in ref.null_scores:
                null = ref.null_scores[name]
            else:
                dims = ref._query_dims(idt["theta"][B], None)
                _, cnull = ref._conditional(dims) if dims is not None else (None, None)
                if not cnull or name not in cnull:
                    print(f"[r{r}]   {name}: unavailable")
                    continue
                null = cnull[name]
            if name not in s_B:
                print(f"[r{r}]   {name}: unavailable")
                continue
            p = empirical_pvalues(null, s_B[name])
            u = null_uniformity(p)
            ok = u["ks_pvalue"] > 0.05
            row[name] = ok
            print(f"[r{r}]   {name:>9}: KS={u['ks_stat']:.4f}  p={u['ks_pvalue']:.4f}  "
                  f"mean_p={p.mean():.3f}  {'PASS' if ok else '*** FAIL ***'}")
        verdicts[r] = row
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
        print(f"  {name:>9}: {len(res) - len(bad)}/{len(res)} repeats uniform"
              + (f"  FAILING: {bad}" if bad else ""))
    print("\n" + ("GATE PASSED — the null is calibrated; Phase 6 may proceed."
                  if allpass else
                  "GATE FAILED — diagnose before building the multi-encoder combiner."))
    return 0 if allpass else 1


if __name__ == "__main__":
    raise SystemExit(main())
