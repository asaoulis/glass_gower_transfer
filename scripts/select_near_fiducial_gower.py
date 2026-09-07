#!/usr/bin/env python3
"""Pick the ~10 most near-fiducial Gower-Street cosmologies and write the lock file.

WHY
---
The Gower suite is a FIXED set of N-body runs, so "choose a realistic cosmology" means *select*
from what exists, never sample. For the matched-nuisance comparison we want the arms compared at
cosmologies a reader recognises as realistic: w close to -1 and (Omega_m, sigma_8) close to Planck.

WHICH POOL
----------
The candidates are restricted to `config/fixed_test_sets/gower_test_ids_100.json`. Those 100 ids are
a verified SUBSET of the 200-id lock, which the `nla_m` flagship held out entirely and which every
`_hf` variate arm locks into its own test split. Selecting from the 100 therefore guarantees that
**every arm has held out every cosmology we plot** -- no arm is being scored on something it trained
on. Selecting from the full 193..781 range would NOT be safe.

METRIC
------
    filter  |w + 1| <= --w-tol
    rank    d = sqrt( ((Omega_m - 0.315)/0.05)^2 + ((sigma_8 - 0.811)/0.06)^2 )

Planck-2018 TT,TE,EE+lowE+lensing central values; the denominators are deliberately loose "how many
interesting-scale steps away" units, not error bars -- they only set the relative weight of the two
axes in the ranking.

⚠️ The other parameters (h, n_s, m_nu, Omega_b h^2) are NOT constrained: the suite is fixed, so
whatever they are at the selected ids is what we get. Each mock carries its own `theta0`, so the
posteriors stay correct -- but the figure must be described as "ten near-fiducial cosmologies", never
"the same cosmology". The residual spread is printed so it can be quoted.

USAGE
-----
    python scripts/select_near_fiducial_gower.py            # print only
    python scripts/select_near_fiducial_gower.py --write    # + write the lock file
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

CSV = os.path.join(REPO, "kids-legacy-sbi", "data", "gower_st",
                   "PKDGRAV3_on_DiRAC_DES_330.csv")
POOL = os.path.join(REPO, "config", "fixed_test_sets", "gower_test_ids_100.json")
OUT = os.path.join(REPO, "config", "fixed_test_sets", "gower_near_fiducial_10.json")

PLANCK_OMEGA_M, PLANCK_SIGMA_8 = 0.315, 0.811
SCALE_OMEGA_M, SCALE_SIGMA_8 = 0.05, 0.06
# The Gower fiducial run. It must survive the cut -- if it does not, the metric is wrong.
FIDUCIAL_SID = 395


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--w-tol", type=float, default=0.1)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(CSV, skiprows=1)          # same read as GowerStCosmologies
    df.columns = df.columns.str.strip()
    pool = json.load(open(POOL))["sim_ids"]

    df = df[df["Serial Number"].isin(pool)].copy()
    print("[pool] %d of the %d lock ids found in the CSV" % (len(df), len(pool)))
    if len(df) != len(pool):
        missing = sorted(set(pool) - set(df["Serial Number"]))
        raise SystemExit("lock ids missing from the CSV: %s" % missing[:20])

    keep = df[np.abs(df["w"] + 1.0) <= args.w_tol].copy()
    print("[filter] |w+1| <= %.2f keeps %d of %d" % (args.w_tol, len(keep), len(df)))

    keep["d"] = np.sqrt(((keep["Omega_m"] - PLANCK_OMEGA_M) / SCALE_OMEGA_M) ** 2
                        + ((keep["sigma_8"] - PLANCK_SIGMA_8) / SCALE_SIGMA_8) ** 2)
    keep = keep.sort_values("d")
    chosen = keep.head(args.n)

    print("\n rank  sim_id   Omega_m   sigma_8       S_8        w       h      n_s     m_nu    d")
    for r, (_, row) in enumerate(chosen.iterrows(), 1):
        s8 = row["sigma_8"] * np.sqrt(row["Omega_m"] / 0.3)
        print("  %2d   %6d   %.4f    %.4f    %.4f   %+.4f  %.4f  %.4f  %.4f  %.3f"
              % (r, int(row["Serial Number"]), row["Omega_m"], row["sigma_8"], s8,
                 row["w"], row["little_h"], row["n_s"], row["m_nu"], row["d"]))

    ids = [int(x) for x in chosen["Serial Number"]]
    if FIDUCIAL_SID not in ids:
        rank = int(np.where(keep["Serial Number"].values == FIDUCIAL_SID)[0][0]) + 1 \
            if FIDUCIAL_SID in keep["Serial Number"].values else None
        raise SystemExit("the Gower fiducial run %d is NOT in the top %d (rank %s) -- check the "
                         "metric before proceeding" % (FIDUCIAL_SID, args.n, rank))
    print("\n  [check] the Gower fiducial run %d is in the set  OK" % FIDUCIAL_SID)

    print("\nResidual spread in the UNCONSTRAINED parameters across the %d (min .. max):" % len(ids))
    for col, nice in (("little_h", "h"), ("n_s", "n_s"), ("m_nu", "m_nu"),
                      ("Omega_b little_h^2", "ombh2")):
        v = chosen[col].astype(float)
        print("  %-6s  %.4f .. %.4f   (spread %.4f)" % (nice, v.min(), v.max(), v.max() - v.min()))

    if not args.write:
        print("\n(--write not given; lock file NOT written)")
        return 0

    payload = {
        "min_id": int(min(ids)),
        "max_id": int(max(ids)),
        "selection": "near_fiducial",
        "selection_params": {
            "pool": "config/fixed_test_sets/gower_test_ids_100.json",
            "pool_rationale": "subset of the 200-id lock; held out by the nla_m flagship AND locked "
                              "into test by every _hf variate arm, so no arm trained on these",
            "w_filter": "abs(w + 1) <= %.2f" % args.w_tol,
            "rank_metric": "sqrt(((Omega_m-%.3f)/%.2f)^2 + ((sigma_8-%.3f)/%.2f)^2)"
                           % (PLANCK_OMEGA_M, SCALE_OMEGA_M, PLANCK_SIGMA_8, SCALE_SIGMA_8),
            "reference": "Planck 2018 TT,TE,EE+lowE+lensing central values",
            "unconstrained": "h, n_s, m_nu, Omega_b h^2 are whatever the fixed N-body suite has at "
                             "these ids; each mock carries its own theta0",
        },
        "order": "rank_by_metric_ascending",
        "source_csv": "kids-legacy-sbi/data/gower_st/PKDGRAV3_on_DiRAC_DES_330.csv",
        "n_target": args.n,
        "n_selected": len(ids),
        "purpose": "The near-fiducial cosmologies for the matched-nuisance variate comparison "
                   "(eval-and-viz/matched-nuisance-variate-corners). 16 mocks per cosmology per arm.",
        "generated_by": "scripts/select_near_fiducial_gower.py",
        "sim_ids": ids,
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\n  wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
