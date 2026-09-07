"""The full detector matrix over all SEVEN Gower BGP suites, including kappa=2.

One table, both detectors on the same footing:
  * cross-encoder KL (5 encoders)      -- posterior disagreement, no reference data needed
  * single-encoder kNN summary-space OOD, and the conditional (theta-residual) variant

Every interval is a PAIRED CLUSTER BOOTSTRAP over cosmologies (resample sim_ids, take all
their rows on both sides), never over rows: the mocks are ~8 correlated augmentations of each
cosmology, so a row-level interval is roughly 3x too narrow.

The kappa=2 suite is the one that makes the galaxy-bias axis readable. gb1p0/gb1p3 are single
b_g values over only 40 cosmologies (80 mocks); kappa=2 keeps the Flamingo per-bin MEANS and
doubles the sigmas, spanning a continuous range of b_g over the full 199 test cosmologies
(1592 mocks) -- 20x the statistics, and a genuine population rather than a slice.

Read its numbers with care: because kappa=2 has the same means as the kappa=1 training prior,
its population CONTAINS the training distribution and only its tails are out of support. A
population AUROC therefore averages real tail signal against a majority of ordinary events and
is expected near 0.5. That is NOT the same statement as "undetectable".
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/alex/work/glass_gower_transfer")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.ml.eval.ood import auroc  # noqa: E402

REPO = "/home/alex/work/glass_gower_transfer"
EXP = f"{REPO}/ml-checkpoints/gower_npe_finetune_nla_m_bgp_z8_ens1"
IND = "gower_bgp_nla_m"
SUITES = ["gower_gb1p3", "gower_gb1p0", "gower_gbk2", "gower_vd", "gower_nla", "gower_nla_z"]
MATCHES = [f"ncosmo300_{i}" for i in range(5)]


def load_npz(p):
    with np.load(p, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def kl_of(variate):
    tag = "_".join(MATCHES)
    d = load_npz(f"{EXP}/misspec/{variate}/misspec_repeat_disagreement_{tag}.npz")
    return np.asarray(d["kl_score"], float), np.asarray(d["test_files"])


def sim_of_files(files):
    """sim_id from an `output_<sim>_out<o>_rot<r>_<n>.h5` basename."""
    return np.array([int(str(f).split("_")[1]) for f in files])


def summ(variate, match):
    return load_npz(f"{EXP}/summaries/{variate}/summaries_{match}.npz")


def cond_is_comparable(q, ind):
    """Is the CONDITIONAL score meaningful for this suite?

    The conditional detector removes a theta-dependence learned on the training suite. That
    is only valid if the suite's theta means the SAME thing. It does not for nla / nla_z:
    their `a_ia` is the KiDS-Legacy A_IA^total amplitude (centred on 0) pushed through the
    NLA-M preset scaler, so in scaled units it lands around -1.8 against the training range
    [0, 1]. The conditional model is then extrapolating on a quantity it never saw, and it
    reports a large residual for a reason that has nothing to do with detecting the physics.
    Reporting that as an AUROC would overstate the detector by ~0.12 (0.97 vs a like-for-like
    0.85). We flag it instead of printing it.
    """
    pr = list(q["params"])
    j = pr.index("a_ia")
    a_q, a_i = q["theta"][:, j], ind["theta"][:, j]
    lo, hi = np.nanmin(a_i), np.nanmax(a_i)
    span = hi - lo
    frac_out = float(np.mean((a_q < lo - 0.1 * span) | (a_q > hi + 0.1 * span)))
    return frac_out < 0.10, frac_out


def boot_auroc(s_null, sim_null, s_q, sim_q, n_boot, rng):
    """Paired cluster bootstrap: resample cosmologies, take all their rows on both sides."""
    uniq = np.unique(sim_null)
    idx_n = {g: np.flatnonzero(sim_null == g) for g in uniq}
    idx_q = {g: np.flatnonzero(sim_q == g) for g in uniq}
    out = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(uniq, len(uniq), replace=True)
        rn = np.concatenate([idx_n[g] for g in draw])
        rq = np.concatenate([idx_q[g] for g in draw if idx_q[g].size])
        out[b] = auroc(s_null[rn], s_q[rq]) if rq.size else np.nan
    return out


def ci(vals):
    v = vals[np.isfinite(vals)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gower_detector_matrix.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = {}

    # ---- KL: one score per event, computed across the 5 encoders -------------------
    kl_ind, f_ind = kl_of(IND)
    sim_ind_kl = sim_of_files(f_ind)

    # ---- kNN / cond-kNN: per encoder, then averaged over encoders ------------------
    ind_s = {m: summ(IND, m) for m in MATCHES}

    print(f"{'suite':<16} {'N':>5} {'Ncos':>5} | {'KL AUROC (95% CI)':>26} | "
          f"{'kNN AUROC (95% CI)':>26} | {'cond-kNN':>20}")
    print("-" * 108)

    for v in SUITES:
        kl_q, f_q = kl_of(v)
        sim_q_kl = sim_of_files(f_q)
        klv = boot_auroc(kl_ind, sim_ind_kl, kl_q, sim_q_kl, args.n_boot, rng)
        kl_pt = auroc(kl_ind, kl_q)
        kl_lo, kl_hi = ci(klv)

        knn_pts, cknn_pts, knn_lo, knn_hi, cknn_lo, cknn_hi = [], [], [], [], [], []
        for m in MATCHES:
            qi, ii = summ(v, m), ind_s[m]
            sq, si = qi["sim_ids"].astype(int), ii["sim_ids"].astype(int)
            ok_cond, frac_out = cond_is_comparable(qi, ii)
            for key, pts, los, his in (("ood_knn_score", knn_pts, knn_lo, knn_hi),
                                       ("ood_cond_knn_score", cknn_pts, cknn_lo, cknn_hi)):
                if key not in qi:
                    continue
                if key == "ood_cond_knn_score" and not ok_cond:
                    continue
                pts.append(auroc(ii[key], qi[key]))
                bv = boot_auroc(ii[key], si, qi[key], sq, max(args.n_boot // 5, 100), rng)
                lo, hi = ci(bv)
                los.append(lo)
                his.append(hi)

        rows[v] = {
            "cond_comparable": bool(ok_cond), "frac_a_ia_out_of_training_range": frac_out,
            "n_mocks": int(len(kl_q)), "n_cosmologies": int(len(np.unique(sim_q_kl))),
            "kl_auroc": kl_pt, "kl_ci": [kl_lo, kl_hi],
            "knn_auroc": float(np.mean(knn_pts)), "knn_ci": [float(np.mean(knn_lo)), float(np.mean(knn_hi))],
            "cond_knn_auroc": float(np.mean(cknn_pts)) if cknn_pts else None,
            "cond_knn_ci": ([float(np.mean(cknn_lo)), float(np.mean(cknn_hi))] if cknn_pts else None),
        }
        r = rows[v]
        print(f"{v:<16} {r['n_mocks']:>5} {r['n_cosmologies']:>5} | "
              f"{kl_pt:>10.3f} [{kl_lo:.3f}, {kl_hi:.3f}] | "
              f"{r['knn_auroc']:>10.3f} [{r['knn_ci'][0]:.3f}, {r['knn_ci'][1]:.3f}] | "
              + (f"{r['cond_knn_auroc']:.3f} [{r['cond_knn_ci'][0]:.3f}, {r['cond_knn_ci'][1]:.3f}]"
                 if r["cond_knn_auroc"] is not None
                 else f"n/a  (a_ia is a different parametrisation; {frac_out:.0%} of mocks "
                      f"outside the training range)"))

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {args.out}")
    print("\nAUROC 0.5 = no separation. An interval containing 0.5 is not a detection.")
    print("kappa=2 shares the training prior's MEANS, so its population contains the training")
    print("distribution; a population AUROC near 0.5 is expected and is NOT 'undetectable'.")
    print("\ncond-kNN is suppressed for nla/nla_z: their a_ia is A_IA^total, not the NLA-M")
    print("prefactor, so conditioning on it extrapolates far outside training and inflates the")
    print("score (0.97 vs a like-for-like 0.85). Use the report's recomputed value there.")


if __name__ == "__main__":
    main()
