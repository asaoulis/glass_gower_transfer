"""A galaxy-bias SEVERITY axis, and the detectability curve along it.

The IA axis has a natural severity knob (A_IA^total) and `gower_aia_vs_kl.py` shows the
cross-encoder KL falling to a floor exactly inside the training band and rising away from it.
The galaxy-bias axis had no such knob: gb1p0/gb1p3 are two fixed points, one of which is
entirely inside the training prior. The kappa=2 suite supplies the knob, because its per-mock
6-vector b_g^(i) is drawn from the SAME per-bin means as the training prior with DOUBLE the
widths, so individual mocks land anywhere from ordinary to far outside support.

SEVERITY VARIABLE. For a mock with per-bin galaxy biases b^(i), define the per-bin z against
the kappa=1 NLA-M training prior (`src/KiDS/simulation_config.py`),

    z_i = (b^(i) - mu_i) / sigma_i,
    mu    = [1.0181, 1.0698, 1.1302, 1.2427, 1.3739, 1.4805]
    sigma = [0.1801, 0.1491, 0.1252, 0.0951, 0.0960, 0.0985]

and summarise the mock by  d = sqrt(sum_i z_i^2)  (Mahalanobis under the diagonal training
prior; null distribution chi_6, median 2.35, 95th percentile 3.55) and by max_i |z_i|, which
says whether ANY bin is outside the +-3 sigma truncation that bounds the training support.

MEASURED (200-mock samples of each store):

  suite               median d   90th d   frac with any |z_i| > 3   b_g range
  in-dist (kappa=1)     2.33      3.20            0.5 %            [0.527, 1.703]
  kappa=2               4.63      6.53           61.5 %            [0.300, 2.099]

So 61.5 % of kappa=2 mocks sit outside the training support in at least one tomographic bin,
with max|z| reaching 6 -- and neither detector registers the suite (KL AUROC 0.537
[0.517, 0.558], kNN 0.499 [0.482, 0.515]). That is the finding this axis exists to quantify.

DATA NOTE. b_g^(i) lives in each mock's HDF5 `cosmo_dict` on the cluster, NOT in the fetched
npz, and the gatekeeper's read verb (`data-h5`) is bounded at 500 files with at most 200
records emitted. Because b_g is drawn per (sim, outer, rot) block and the analysis test set is
rot0 only (~10 % of the store), an evenly-spread 500-file sample yields ~21 usable test mocks
-- enough for the DISTRIBUTIONS above, not for a per-event detectability curve. `--records`
consumes a full per-event table once one is available; until then `--dist-only` reports the
severity distributions, which need no join.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MU = np.array([1.0181, 1.0698, 1.1302, 1.2427, 1.3739, 1.4805])
SIGMA = np.array([0.1801, 0.1491, 0.1252, 0.0951, 0.0960, 0.0985])
TRUNC = 3.0          # the training prior is truncated at +-3*kappa*sigma; kappa=1 here


def bg_matrix(records):
    return np.array([[r[f"b_g_bin{i}"] for i in range(1, 7)] for r in records], float)


def severity(B):
    """Per-bin z against the training prior, the chi_6 distance d, and max|z|."""
    Z = (B - MU) / SIGMA
    return Z, np.sqrt((Z ** 2).sum(1)), np.abs(Z).max(1)


def is_test_mock(r):
    """The analysis test filter: rotation 0, inner shape-noise realisation 0 or 1."""
    return r.get("rot") == 0 and r.get("cat") in (0, 1)


def describe(tag, records):
    B = bg_matrix(records)
    Z, d, mx = severity(B)
    out = {
        "n": len(records),
        "median_d": float(np.median(d)), "p90_d": float(np.percentile(d, 90)),
        "max_d": float(d.max()),
        "median_max_abs_z": float(np.median(mx)),
        "frac_any_bin_outside_support": float(np.mean(mx > TRUNC)),
        "frac_any_bin_abs_z_gt2": float(np.mean(mx > 2.0)),
        "b_g_min": float(B.min()), "b_g_max": float(B.max()),
    }
    print(f"{tag:22s} n={out['n']:<5d} median d={out['median_d']:.2f}  "
          f"90th d={out['p90_d']:.2f}  max d={out['max_d']:.2f}")
    print(f"{'':22s} any bin outside +-{TRUNC:.0f}sigma support: "
          f"{out['frac_any_bin_outside_support']:.1%}   "
          f"b_g in [{out['b_g_min']:.3f}, {out['b_g_max']:.3f}]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist-json", nargs="+", required=True,
                    metavar="TAG=PATH",
                    help="data-h5 dumps, e.g. 'kappa2=k2.json' 'in-dist=ind.json'")
    ap.add_argument("--test-only", action="store_true",
                    help="restrict to the analysis test filter (rot0, cat 0/1)")
    ap.add_argument("--out", default="gower_bg_distance.json")
    args = ap.parse_args()

    print("b_g severity vs the kappa=1 NLA-M training prior.")
    print(f"d = sqrt(sum_i z_i^2) over 6 bins; null is chi_6 (median 2.35, 95th 3.55).")
    print(f"'outside support' = ANY bin with |z_i| > {TRUNC:.0f}.\n")

    res = {}
    for spec in args.dist_json:
        if "=" not in spec:
            raise SystemExit(f"expected TAG=PATH, got {spec!r}")
        tag, path = spec.split("=", 1)
        with open(path) as f:
            recs = json.load(f)["records"]
        if args.test_only:
            recs = [r for r in recs if is_test_mock(r)]
        if not recs:
            print(f"{tag}: no records after filtering")
            continue
        res[tag] = describe(tag, recs)
        print()

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {args.out}")
    if "kappa2" in res and "in-dist" in res:
        a, b = res["in-dist"], res["kappa2"]
        print(f"\nkappa=2 puts {b['frac_any_bin_outside_support']:.1%} of mocks outside the "
              f"training support in at least one bin,")
        print(f"against {a['frac_any_bin_outside_support']:.1%} in-distribution -- and no "
              f"detector separates the suite (KL 0.537, kNN 0.499).")


if __name__ == "__main__":
    main()
