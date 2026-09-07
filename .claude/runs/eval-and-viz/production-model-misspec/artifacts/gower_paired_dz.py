"""Paired Delta-z test for the two fixed-b_g Gower variates.

Why paired
----------
`gower_gb1p0` and `gower_gb1p3` survive the fixed test lock with only ~80 events each.
At that N the calibration statistics sit at or below the matched-N floor (cal_full ~0.05,
cal3 ~0.09), so neither ABSOLUTE number is interpretable on its own.

But if the two catalogues were generated with the same `--rng-seed`, they are *paired*:
the same cosmology, the same footprint rotation and the same shape-noise realisation,
differing ONLY in the galaxy bias. Then

    dz = z(b_g = 1.3) - z(b_g = 1.0)

cancels cosmic variance and the shared noise realisation event-by-event, and its standard
error is far smaller than that of either absolute z. This is the same trick the GLASS
b_g ladder used.

The script first VERIFIES the pairing (identical (sim_id, aug_id) keys) rather than
assuming it -- an unpaired join would silently produce a meaningless difference.

Reads the compact moments npz (KB-scale), not the full sample dump.
"""

import argparse
import glob
import os

import numpy as np

PARAMS = ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"]
KEY = ["omega_m", "sigma_8"]


def load_moments(root, variate, match):
    p = os.path.join(root, variate, f"misspec_posterior_moments_{match}.npz")
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=True) as f:
        d = {k: f[k] for k in f.files}
    if "sim_ids" not in d or "aug_ids" not in d:
        raise KeyError(f"{p} has no sim_ids/aug_ids -- cannot pair (old-format npz).")
    return d


def pair(a, b):
    """Join two variates on (sim_id, aug_id). Returns aligned index arrays."""
    ka = {(int(s), int(u)): i for i, (s, u) in enumerate(zip(a["sim_ids"], a["aug_ids"]))}
    kb = {(int(s), int(u)): i for i, (s, u) in enumerate(zip(b["sim_ids"], b["aug_ids"]))}
    common = sorted(set(ka) & set(kb))
    return np.array([ka[k] for k in common]), np.array([kb[k] for k in common]), common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="checkpoints/<exp>/misspec")
    ap.add_argument("--a", default="gower_gb1p0", help="baseline variate")
    ap.add_argument("--b", default="gower_gb1p3", help="shifted variate")
    ap.add_argument("--matches", nargs="+",
                    default=[f"ncosmo300_{i}" for i in range(5)])
    args = ap.parse_args()

    print(f"=== PAIRED Delta-z : {args.b} minus {args.a} ===\n")
    per_repeat = {}

    for m in args.matches:
        A, B = load_moments(args.root, args.a, m), load_moments(args.root, args.b, m)
        if A is None or B is None:
            print(f"[{m}] SKIP (missing moments npz)")
            continue
        ia, ib, common = pair(A, B)
        na, nb = len(A["sim_ids"]), len(B["sim_ids"])
        frac = len(common) / max(min(na, nb), 1)
        print(f"[{m}] N({args.a})={na}  N({args.b})={nb}  paired={len(common)} "
              f"({frac:.0%} of the smaller set)")
        if len(common) < 10:
            print(f"[{m}] NOT PAIRED -- too few common (sim_id, aug_id) keys. "
                  f"The two catalogues were probably generated with different --rng-seed, "
                  f"so the paired statistic does not apply.\n")
            continue

        dz = B["z"][ib] - A["z"][ia]          # [Npair, D]
        per_repeat[m] = dz

        print(f"{'param':>9} | {'paired mean dz':>15} | {'unpaired diff':>14} | "
              f"{'SE gain':>7} | same-sign")
        print("-" * 72)
        for j, p in enumerate(PARAMS):
            d = dz[:, j][np.isfinite(dz[:, j])]
            if d.size == 0:
                continue
            se_pair = d.std(ddof=1) / np.sqrt(d.size)
            za, zb = A["z"][ia][:, j], B["z"][ib][:, j]
            za, zb = za[np.isfinite(za)], zb[np.isfinite(zb)]
            unp = zb.mean() - za.mean()
            se_unp = np.sqrt(za.var(ddof=1) / za.size + zb.var(ddof=1) / zb.size)
            gain = se_unp / max(se_pair, 1e-12)
            same = max((d > 0).mean(), (d < 0).mean())
            # >3 SE is a significance flag only -- at N~400 a physically negligible shift
            # (e.g. mnu ~ 0.002 sigma) still clears it, so read the magnitude, not the star.
            star = "  <-- >3 SE" if abs(d.mean()) > 3 * se_pair else ""
            print(f"{p:>9} | {d.mean():+8.3f} +- {se_pair:4.3f} | {unp:+8.3f} +- {se_unp:4.3f} | "
                  f"{gain:6.2f}x | {same:4.0%}{star}")
        print()

    # ---- stack repeats: the encoders are independent, the EVENTS are the same ----
    if len(per_repeat) > 1:
        print("=== ACROSS REPEATS (mean over encoders of the paired mean dz) ===")
        print(f"{'param':>9} | {'mean over repeats':>19} | spread (sd over repeats)")
        print("-" * 60)
        for j, p in enumerate(PARAMS):
            vals = [np.nanmean(dz[:, j]) for dz in per_repeat.values()]
            vals = [v for v in vals if np.isfinite(v)]
            if not vals:
                continue
            flag = "  *" if abs(np.mean(vals)) > 2 * (np.std(vals) or 1e-12) else ""
            print(f"{p:>9} | {np.mean(vals):+18.3f} | {np.std(vals):.3f}{flag}")
        print("\n(* = the shift is large compared with the encoder-to-encoder spread, i.e. it is a "
              "property of the physics rather than of one encoder's init.)")


if __name__ == "__main__":
    main()
