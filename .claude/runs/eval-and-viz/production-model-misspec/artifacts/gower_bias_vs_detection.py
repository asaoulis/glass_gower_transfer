"""How much BIAS do the undetected suites carry, and do the detectors fire on the biased events?

Detectability alone is only half the question. A suite that no detector flags is only benign
if it is also unbiased; a suite that is biased AND unflagged is the dangerous case. This script
puts the three quantities on one footing, per event:

  BIAS       per-event Mahalanobis distance of the truth from the posterior, using the stored
             posterior covariance: d^2 = (theta0 - mu)^T Sigma^-1 (theta0 - mu). Reported both
             on the 3 cosmological parameters of interest (Omega_m, sigma_8, w0; null = chi^2_3)
             and signed per parameter in units of the posterior width.
  MISSPEC    the cross-encoder KL between the five posteriors (no reference data needed).
  OOD        the single-encoder summary-space kNN score (needs a reference cloud).

The headline comparison is the RANK CORRELATION between bias and each detector WITHIN a suite:
a detector that is useful does not merely separate suites on average, it flags the individual
events that are actually biased. A suite where the bias is real but uncorrelated with every
detector is the failure mode worth worrying about.

Mahalanobis is computed from `misspec_posterior_moments_*.npz` (mean, cov, theta0), which
reproduces the full-sample path to 3e-5 relative while being ~940x smaller.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/alex/work/glass_gower_transfer")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = "/home/alex/work/glass_gower_transfer"
EXP = f"{REPO}/ml-checkpoints/gower_npe_finetune_nla_m_bgp_z8_ens1"
IND = "gower_bgp_nla_m"
SUITES = [IND, "gower_gbk2", "gower_gb1p0", "gower_gb1p3", "gower_vd", "gower_nla", "gower_nla_z"]
MATCHES = [f"ncosmo300_{i}" for i in range(5)]
PARAMS = ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"]
KEY3 = ["omega_m", "sigma_8", "w0"]
I3 = [PARAMS.index(p) for p in KEY3]


def load(p):
    with np.load(p, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def moments(v, m):
    return load(f"{EXP}/misspec/{v}/misspec_posterior_moments_{m}.npz")


def mahalanobis3(d):
    """chi^2_3 statistic per event on (omega_m, sigma_8, w0)."""
    mu, th, cov = d["mean"][:, I3], d["theta0"][:, I3], d["cov"][:, I3][:, :, I3]
    r = (th - mu).astype(np.float64)
    out = np.empty(len(r))
    for i in range(len(r)):
        try:
            out[i] = float(r[i] @ np.linalg.solve(cov[i].astype(np.float64), r[i]))
        except np.linalg.LinAlgError:
            out[i] = np.nan
    return out


def kl_of(v):
    d = load(f"{EXP}/misspec/{v}/misspec_repeat_disagreement_{'_'.join(MATCHES)}.npz")
    return dict(zip(d["test_files"].tolist(), np.asarray(d["kl_score"], float)))


def knn_of(v):
    """kNN OOD score per event, averaged over the five encoders (joined on test_files)."""
    acc = {}
    for m in MATCHES:
        s = load(f"{EXP}/summaries/{v}/summaries_{m}.npz")
        if "ood_knn_score" not in s:
            return {}
        for f, x in zip(s["test_files"].tolist(), np.asarray(s["ood_knn_score"], float)):
            acc.setdefault(f, []).append(x)
    return {k: float(np.mean(v_)) for k, v_ in acc.items()}


def spearman(a, b):
    from scipy.stats import spearmanr
    k = np.isfinite(a) & np.isfinite(b)
    if k.sum() < 20:
        return np.nan
    return float(spearmanr(a[k], b[k]).statistic)


def cluster_ci(x, sim, n_boot, rng, fn=np.nanmedian):
    uniq = np.unique(sim)
    idx = {g: np.flatnonzero(sim == g) for g in uniq}
    vals = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(uniq, len(uniq), replace=True)
        vals[b] = fn(x[np.concatenate([idx[g] for g in draw])])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gower_bias_vs_detection.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("BIAS vs DETECTION, per event.  chi2_3 = Mahalanobis of the truth from the posterior")
    print("on (omega_m, sigma_8, w0); under a correctly calibrated posterior its median is 2.37.")
    print("rho columns are Spearman WITHIN the suite: does the detector flag the biased events?\n")
    hdr = (f"{'suite':<16} {'N':>5} | {'median chi2_3':>22} | {'mean z(om)':>11} "
           f"{'mean z(s8)':>11} | {'rho(chi2,KL)':>12} {'rho(chi2,kNN)':>13}")
    print(hdr)
    print("-" * len(hdr))

    out = {}
    for v in SUITES:
        # Repeats can differ in length: the driver drops events whose posterior sampling went
        # non-finite, and it drops different ones per encoder. So join on test_files, never on
        # position, and average over whichever encoders retained each event.
        accum, simof = {}, {}
        for m in MATCHES:
            d = moments(v, m)
            ch = mahalanobis3(d)
            zo = d["z"][:, PARAMS.index("omega_m")]
            zs = d["z"][:, PARAMS.index("sigma_8")]
            for i, f in enumerate(d["test_files"].tolist()):
                accum.setdefault(f, []).append((ch[i], zo[i], zs[i]))
                simof[f] = int(d["sim_ids"][i])
        fl = sorted(accum)
        arr = np.array([np.nanmean(np.asarray(accum[f], float), axis=0) for f in fl])
        chi_m, zom_m, zs8_m = arr[:, 0], arr[:, 1], arr[:, 2]
        sim = np.array([simof[f] for f in fl])

        klm, knm = kl_of(v), knn_of(v)
        kl = np.array([klm.get(f, np.nan) for f in fl])
        kn = np.array([knm.get(f, np.nan) for f in fl]) if knm else np.full(len(fl), np.nan)

        lo, hi = cluster_ci(chi_m, sim, args.n_boot, rng)
        r_kl, r_kn = spearman(chi_m, kl), spearman(chi_m, kn)
        out[v] = {
            "n": int(len(chi_m)),
            "median_chi2_3": float(np.nanmedian(chi_m)), "chi2_3_ci": [lo, hi],
            "frac_chi2_gt_7_81": float(np.nanmean(chi_m > 7.815)),   # chi2_3 95th pct
            "mean_z_omega_m": float(np.nanmean(zom_m)),
            "mean_z_sigma_8": float(np.nanmean(zs8_m)),
            "spearman_chi2_kl": r_kl, "spearman_chi2_knn": r_kn,
        }
        print(f"{v:<16} {len(chi_m):>5} | {np.nanmedian(chi_m):>8.2f} [{lo:.2f}, {hi:.2f}] | "
              f"{np.nanmean(zom_m):>+11.3f} {np.nanmean(zs8_m):>+11.3f} | "
              f"{r_kl:>12.3f} {r_kn:>13.3f}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    print("\nReference: a calibrated posterior gives median chi2_3 = 2.37 and 5% of events above 7.81.")


if __name__ == "__main__":
    main()
