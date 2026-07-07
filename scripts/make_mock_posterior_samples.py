#!/usr/bin/env python
"""Generate MOCK posterior-sample npz files mirroring the eval save schema, for local
validation of scripts/plot_posteriors.py without touching the cluster.

Schema (= src/ml/eval/utils.py:_save_posterior_samples, commit 8748b2c):
  samples    float32 [S, N, D]   posterior samples, scaled [0,1] space
  theta0s    float32 [N, D]      true params, scaled [0,1] space
  test_files str     [N]         basenames output_<sim>_out<o>_rot<r>_<n>.h5
  sim_ids    int64   [N]
  aug_ids    int64   [N]         trailing _<n>.h5 index (NOT unique across out<o> variants)

The generated model files share identical theta0s/test_files (same test split, as repeats
of one experiment do) but have slightly different posterior clouds so overlays are
visually distinct.

Usage:
  python scripts/make_mock_posterior_samples.py --out-dir /tmp/mock_samples
"""
import argparse
import os

import numpy as np

PARAM_NAMES = ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-models", type=int, default=3)
    ap.add_argument("--sim-ids", type=int, nargs="+", default=[193, 205])
    ap.add_argument("--n-outer", type=int, default=4, help="shape-noise variants per cosmology (out0..3_rot0_0)")
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    D = len(PARAM_NAMES)

    test_files, sim_ids = [], []
    for sid in args.sim_ids:
        for o in range(args.n_outer):
            test_files.append(f"output_{sid}_out{o}_rot0_0.h5")
            sim_ids.append(sid)
    N = len(test_files)
    aug_ids = np.zeros(N, dtype=np.int64)  # trailing index is 0 for all out*_rot0_0 files
    sim_ids = np.asarray(sim_ids, dtype=np.int64)

    # One truth per cosmology (shared across its augmentations), comfortably inside [0,1].
    theta_by_sim = {sid: rng.uniform(0.25, 0.75, size=D) for sid in args.sim_ids}
    theta0s = np.stack([theta_by_sim[s] for s in sim_ids]).astype(np.float32)

    os.makedirs(args.out_dir, exist_ok=True)
    for m in range(args.n_models):
        # Per-model bias + width so the three repeats' contours are distinguishable.
        bias = rng.normal(0.0, 0.015, size=(N, D))
        width = 0.05 * (1.0 + 0.25 * m)
        samples = (
            theta0s[None, :, :]
            + bias[None, :, :]
            + rng.normal(0.0, width, size=(args.n_samples, N, D))
        ).astype(np.float32)
        samples = np.clip(samples, 0.0, 1.0)

        out = os.path.join(args.out_dir, f"samples_kids_s8_analytic_ncosmo300_{m}.npz")
        np.savez_compressed(
            out,
            samples=samples,
            theta0s=theta0s,
            test_files=np.array(test_files),
            sim_ids=sim_ids,
            aug_ids=aug_ids,
        )
        print(f"wrote {out}  samples={samples.shape} theta0s={theta0s.shape} N={N}")


if __name__ == "__main__":
    main()
