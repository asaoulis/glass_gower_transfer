"""Paired b_g shift in PHYSICAL parameter units, not posterior-width (z) units.

`gower_paired_dz.py` reports Delta-z = (theta0 - mu_post)/sigma_post, which is a shift
measured in units of each event's OWN posterior width. That is the right scale for asking
"is this model biased for its claimed precision", but it is NOT comparable to a published
bias bound such as DES Y3's b_g test (Jeffrey 2025 sec 7.3.1), which is quoted as a shift in
Omega_m itself. Since posterior width and bias move together (see the
`vare-fragility-tracks-sharpness` note), a z-unit number can look large simply because the
posterior is sharp.

This script differences the paired posterior MEANS and rescales by the width of the
COSMO_PARAM_PRESET_MINMAX box, which is the min/max scaling the loaders apply, giving
Delta_theta in physical units. It also reports the typical in-distribution posterior sd in
the same units, so the two framings sit side by side.
"""

import argparse

import numpy as np

from src.ml.data.constants import COSMO_PARAM_PRESET_MINMAX as MINMAX

PARAMS = ["omega_m", "sigma_8", "w0"]


def moments(root, variate, match):
    with np.load(f"{root}/{variate}/misspec_posterior_moments_{match}.npz",
                 allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="checkpoints/<exp>/misspec")
    ap.add_argument("--a", default="gower_gb1p0")
    ap.add_argument("--b", default="gower_gb1p3")
    ap.add_argument("--indist", default="gower_bgp_nla_m")
    ap.add_argument("--matches", nargs="+",
                    default=[f"ncosmo300_{i}" for i in range(5)])
    args = ap.parse_args()

    print(f"=== PAIRED Delta-theta ({args.b} minus {args.a}), PHYSICAL units ===")
    print("    paired posterior means differenced per event, rescaled by the")
    print("    COSMO_PARAM_PRESET_MINMAX box width; sigma_post is the in-distribution mean.\n")

    shifts = {p: [] for p in PARAMS}
    sigmas = {p: [] for p in PARAMS}
    for m in args.matches:
        a, b = moments(args.root, args.a, m), moments(args.root, args.b, m)
        ind = moments(args.root, args.indist, m)
        if not np.array_equal(a["test_files"], b["test_files"]):
            raise SystemExit(f"[{m}] {args.a} and {args.b} are not row-aligned")
        params = list(a["params"])
        for p in PARAMS:
            j = params.index(p)
            lo, hi = MINMAX[p]
            width = hi - lo
            shifts[p].append(float(((b["mean"][:, j] - a["mean"][:, j]) * width).mean()))
            sigmas[p].append(float((ind["std"][:, j] * width).mean()))

    hdr = "    param | Delta_theta (phys) | sd over encoders | in-dist sigma_post | ratio"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for p in PARAMS:
        d = np.array(shifts[p])
        s = float(np.mean(sigmas[p]))
        print(f"    {p:8s} | {d.mean():+18.4f} | {d.std():16.4f} | "
              f"{s:18.4f} | {d.mean() / s:+.2f} sigma")

    print("\n    For context, DES Y3 (Jeffrey 2025 sec 7.3.1) quote their b_g bound as a shift")
    print("    in Omega_m itself: |Delta_theta(omega_m)| < 0.007.")


if __name__ == "__main__":
    main()
