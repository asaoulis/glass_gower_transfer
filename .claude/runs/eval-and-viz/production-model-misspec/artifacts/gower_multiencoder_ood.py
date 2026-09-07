"""Phase 6 -- multi-encoder OOD combiner for the Gower BGP arm (five independently-seeded encoders).

What this script does
---------------------
Stage 0  Per encoder r: ``OODReference.fit`` on that encoder's own ``_train`` cloud (whitener,
         theta-kNN regressor, residual whitener -- ALL fit on in-distribution train data only);
         score ``_idtest`` (the null) and every variate ONCE.  Variates that lack/exclude
         parameters (``gower_nla``/``gower_nla_z``: a_ia excluded, b_ia NaN) use the 7-dim
         conditional model, and their null ``cond_knn`` scores are the 7-dim null on the SAME
         ``_idtest`` rows (``OODReference._conditional(dims)``), so null and query always share
         the model.  Scores are cached (``stage0_scores.npz``).
Stage 1  Per encoder: right-tail empirical p-value of every score against the encoder's null
         restricted to the FIT half A of a cosmology-level split of ``_idtest``.  These p-values
         (and their probits) are the only cross-encoder currency -- raw ``z`` is never pooled.
Stage 2  Combine the five per-encoder p-values into ONE statistic T (see METHODS), then
         RECALIBRATE T into an empirical p-value against T's own null on half A (constraint 4).
         Anything the combiner fits (DoSE mean/covariance, GLS correlation) uses half A only.
         Half-A rows enter the stage-2 null through leave-one-out stage-1 p-values
         (``n_ge / n`` with self counted == the out-of-sample ``(1+n_ge)/(1+n)`` on n-1 points),
         so the null of T has the same joint dependence across encoders as an unseen event.

Split conventions (from ``gower_null_selfcheck.py`` -- read its docstring for the two traps)
  * Criterion 1 (calibration): ``--n-splits`` RANDOM cosmology-level splits; fit on A, evaluate
    the recalibrated p on B; (a) mean_p over splits within one PER-SPLIT sd of 0.5,
    (b) median KS D on ONE ROW PER COSMOLOGY draws below 1.36/sqrt(n_cos) (H0 median
    0.83/sqrt(n_cos)).  Positive control: ``gower_nla`` restricted to B cosmologies must EXCEED
    the same critical value.  Never a row-level KS (2.8x too strict, n_eff ~ n_cos).
  * Shipped p-values + criteria 2-3: ONE fixed seeded random cosmology split, 2-FOLD CROSS-FIT:
    every row (null or variate) of cosmology c is scored against the fold that does NOT contain
    c, so every ``_idtest`` row has an out-of-sample p and null-vs-variate stays paired (all six
    variates are re-simulations of the SAME ``_idtest`` cosmologies).  AUROC / Spearman CIs
    bootstrap by COSMOLOGY (resample sim_ids, take every row of both sides), never by row.
    The AUROC/Spearman verdicts are invariant to stage 2 (a monotone map); stage 2 matters for
    criterion 1 and for the interpretability of the shipped p.

METHODS (T, higher = more OOD; every one is put through the SAME stage-2 recalibration)
  baselines   meanp      -mean_r p_r                       (combine_pvalues_across_models "mean")
              fisher     -2 sum_r log p_r                  (combine_pvalues_across_models "fisher")
              single_r   -p_r  for each encoder (reported as the encoder AVERAGE and, labelled
                         as an ORACLE, the best encoder per variate -- that is selection on labels)
  (a) DoSE    dose5      Mahalanobis^2 of the 5-vector u_r = Phi^-1(1-p_r) (cond_knn) under the
                         half-A Gaussian (Morningstar et al. 2021, density-of-states on the
                         vector of per-model statistics, ID-fit); dose10 = same on the 10-vector
                         [u(cond_knn); u(knn)] per encoder.
  (b) own     gls        Lin & Sullivan (2009) correlated-Stouffer: T = 1'S^-1 u / sqrt(1'S^-1 1),
                         S = half-A correlation of u (shrunk 10 % to the diagonal); collapses to
                         plain Stouffer (mean u) for exchangeable encoders.
              stouffer   mean_r u_r ;  tippett  -min_r p_r (encoder-specific failures).

Outputs (``--out``)
  multiencoder_ood_<set>.npz   combined_score, combined_p (the RECOMMENDED method), test_files,
                               sim_ids, fold, plus p_<method> for every method and the stage-1
                               per-encoder p ``p1_cond_knn_r<r>``.
  phase6_acceptance.json/.md   the three acceptance criteria per method.
  phase6_auroc.png, phase6_spearman.png
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.stats import kstest, norm, rankdata, spearmanr

REPO = "/home/alex/work/glass_gower_transfer"
sys.path.insert(0, REPO)
from src.ml.eval.ood import OODReference, empirical_pvalues, auroc  # noqa: E402

VARIATES = ["gower_bgp_nla_m", "gower_nla", "gower_nla_z", "gower_vd", "gower_gb1p0", "gower_gb1p3"]
EXCLUDE = {"gower_nla": ["a_ia"], "gower_nla_z": ["a_ia"]}   # mirrors misspec.GOWER_BGP_VARIATES
HARD = ["gower_gb1p0", "gower_gb1p3", "gower_vd"]
METHODS = ["meanp", "fisher", "single", "dose5", "dose10", "gls", "stouffer", "tippett"]
LABELS = {"meanp": "mean-p (baseline)", "fisher": "Fisher (baseline)", "single": "single encoder, avg (baseline)",
          "single_best": "single encoder, best [oracle]", "dose5": "(a) DoSE-5 Mahalanobis", "dose10": "(a) DoSE-10 (knn+cond)",
          "gls": "(b) GLS-Stouffer (Lin-Sullivan)", "stouffer": "Stouffer", "tippett": "Tippett (min p)"}
KS_CRIT, KS_MED = 1.36, 0.83
EPS = 5e-4


def load(path):
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


# ------------------------------------------------------------------------------------------
# Stage 0: per-encoder scores (fit on _train only; split-independent)
# ------------------------------------------------------------------------------------------
def stage0(root, repeats, match_fmt, max_train, cache):
    if cache and os.path.exists(cache):
        d = load(cache)
        meta = json.loads(str(d["meta"]))
        sc = {int(r): {S: {"knn": d[f"{r}/{S}/knn"], "cond_knn": d[f"{r}/{S}/cond_knn"],
                           "null_knn": d[f"{r}/{S}/null_knn"], "null_cond_knn": d[f"{r}/{S}/null_cond_knn"]}
                       for S in meta["sets"]} for r in meta["repeats"]}
        ids = {S: {"sim_ids": d[f"ids/{S}/sim_ids"], "test_files": d[f"ids/{S}/test_files"]} for S in meta["sets"]}
        print(f"[stage0] loaded cache {cache}")
        return sc, ids, meta
    sets = ["_idtest"] + VARIATES
    sc, ids, meta = {}, {}, {"sets": sets, "repeats": list(repeats), "dims": {}}
    for r in repeats:
        m = match_fmt.format(r=r)
        tr = load(os.path.join(root, "_train", f"summaries_{m}.npz"))
        idt = load(os.path.join(root, "_idtest", f"summaries_{m}.npz"))
        params = [str(p) for p in idt["params"]]
        assert len(np.intersect1d(np.unique(tr["sim_ids"]), np.unique(idt["sim_ids"]))) == 0, "train/idtest leak"
        t0 = time.time()
        ref = OODReference.fit(tr["z"], z_id=idt["z"], theta_train=tr["theta"], theta_id=idt["theta"], max_train=max_train)
        sc[r] = {}
        for S in sets:
            v = idt if S == "_idtest" else load(os.path.join(root, S, f"summaries_{m}.npz"))
            th = np.asarray(v["theta"], dtype=np.float64)
            dims = [i for i, p in enumerate(params) if p not in EXCLUDE.get(S, []) and np.isfinite(th[:, i]).all()]
            s = ref.score(v["z"], th, theta_dims=dims)
            _, cnull = ref._conditional(dims)
            sc[r][S] = {"knn": s["knn"], "cond_knn": s["cond_knn"],
                        "null_knn": ref.null_scores["knn"], "null_cond_knn": cnull["cond_knn"]}
            meta["dims"][S] = dims
            if r == repeats[0]:
                ids[S] = {"sim_ids": v["sim_ids"].astype(np.int64), "test_files": v["test_files"].astype(str)}
            else:
                assert (v["test_files"].astype(str) == ids[S]["test_files"]).all(), f"{S} rows not aligned across repeats"
        assert np.allclose(sc[r]["_idtest"]["cond_knn"], sc[r]["_idtest"]["null_cond_knn"])
        print(f"[stage0] r{r}: fit+scored {len(sets)} sets in {time.time() - t0:.1f}s; dims={ {S: len(dd) for S, dd in meta['dims'].items()} }")
    if cache:
        payload = {"meta": np.array(json.dumps(meta))}
        for r in sc:
            for S in sc[r]:
                for k, a in sc[r][S].items():
                    payload[f"{r}/{S}/{k}"] = a
        for S in ids:
            payload[f"ids/{S}/sim_ids"] = ids[S]["sim_ids"]
            payload[f"ids/{S}/test_files"] = ids[S]["test_files"]
        np.savez_compressed(cache, **payload)
    return sc, ids, meta


# ------------------------------------------------------------------------------------------
# Stage 1 + 2
# ------------------------------------------------------------------------------------------
def p_insample(null_scores):
    """Leave-one-out empirical p of each null score against the null it belongs to (self counted)."""
    null = np.sort(np.asarray(null_scores, dtype=np.float64))
    n_ge = null.size - np.searchsorted(null, null_scores, side="left")
    return n_ge / null.size


def probit(P):
    return norm.isf(np.clip(P, EPS, 1.0 - EPS))


class Combiner:
    """Everything fit on half A: stage-1 p on A (in-sample LOO), DoSE moments, GLS weights, stage-2 nulls."""

    def __init__(self, P_A, Pk_A, shrink=0.1):
        self.R = P_A.shape[0]
        U, Uk = probit(P_A), probit(Pk_A)
        self.mu5, self.S5 = U.mean(1), np.cov(U)
        X10 = np.vstack([U, Uk])
        self.mu10, self.S10 = X10.mean(1), np.cov(X10)
        C = np.corrcoef(U)
        C = (1 - shrink) * C + shrink * np.eye(self.R)
        w = np.linalg.solve(C, np.ones(self.R))
        self.w_gls = w / np.sqrt(w.sum())
        self.T_A = {m: self.T(m, P_A, Pk_A) for m in METHODS if m != "single"}
        for r in range(self.R):
            self.T_A[f"single{r}"] = -P_A[r]

    def T(self, method, P, Pk=None):
        if method == "meanp":
            return -P.mean(0)
        if method == "fisher":
            return -2.0 * np.log(np.clip(P, 1e-300, 1.0)).sum(0)
        if method == "tippett":
            return -P.min(0)
        if method.startswith("single"):
            return -P[int(method[6:])]
        U = probit(P)
        if method == "stouffer":
            return U.mean(0)
        if method == "gls":
            return self.w_gls @ U
        if method == "dose5":
            d = U - self.mu5[:, None]
            return np.einsum("in,ij,jn->n", d, np.linalg.pinv(self.S5), d)
        if method == "dose10":
            d = np.vstack([U, probit(Pk)]) - self.mu10[:, None]
            return np.einsum("in,ij,jn->n", d, np.linalg.pinv(self.S10), d)
        raise ValueError(method)

    def p2(self, method, P, Pk=None):
        """Stage-2 recalibrated p of query events (their stage-1 p's are vs half A)."""
        return empirical_pvalues(self.T_A[method], self.T(method, P, Pk))


def all_methods(R):
    return [m for m in METHODS if m != "single"] + [f"single{r}" for r in range(R)]


def fit_fold(sc, ids, maskA_rows, S_key):
    """Build the half-A machinery for the dims-key of set S_key (null scores on _idtest rows)."""
    reps = sorted(sc)
    P_A = np.stack([p_insample(sc[r][S_key]["null_cond_knn"][maskA_rows]) for r in reps])
    Pk_A = np.stack([p_insample(sc[r][S_key]["null_knn"][maskA_rows]) for r in reps])
    return Combiner(P_A, Pk_A)


def stage1_query(sc, maskA_rows, S, rows):
    reps = sorted(sc)
    P = np.stack([empirical_pvalues(sc[r][S]["null_cond_knn"][maskA_rows], sc[r][S]["cond_knn"][rows]) for r in reps])
    Pk = np.stack([empirical_pvalues(sc[r][S]["null_knn"][maskA_rows], sc[r][S]["knn"][rows]) for r in reps])
    return P, Pk


def crossfit(sc, ids, cosA, sets):
    """2-fold cross-fit: p2[method][S] for EVERY row, scored against the fold not containing its cosmology."""
    sim_id = ids["_idtest"]["sim_ids"]
    out = {S: {} for S in sets}
    fold = {S: np.full(len(ids[S]["sim_ids"]), -1, int) for S in sets}
    p1 = {S: {} for S in sets}
    for f, (fitA) in enumerate([cosA, np.setdiff1d(np.unique(sim_id), cosA)]):
        maskA = np.isin(sim_id, fitA)
        combs = {}
        for S in sets:
            rows = ~np.isin(ids[S]["sim_ids"], fitA)
            if rows.sum() == 0:
                continue
            key = S  # each set's own dims-key null
            if key not in combs:
                combs[key] = fit_fold(sc, ids, maskA, key)
            P, Pk = stage1_query(sc, maskA, S, rows)
            fold[S][rows] = f
            for m in all_methods(len(sc)):
                out[S].setdefault(m, np.full(len(rows), np.nan))[rows] = combs[key].p2(m, P, Pk)
            for j, r in enumerate(sorted(sc)):
                p1[S].setdefault(r, np.full(len(rows), np.nan))[rows] = P[j]
    return out, fold, p1


# ------------------------------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------------------------------
def cluster_ks(pvals, groups, n_draws, rng):
    uniq = np.unique(groups)
    idx = [np.flatnonzero(groups == g) for g in uniq]
    D = np.empty(n_draws)
    for m in range(n_draws):
        pick = np.fromiter((ix[rng.integers(len(ix))] for ix in idx), int, len(idx))
        D[m] = kstest(pvals[pick], "uniform").statistic
    return float(np.median(D)), len(uniq)


def auroc_fast(s_null, s_q):
    ranks = rankdata(np.concatenate([s_null, s_q]))
    rb = ranks[s_null.size:].sum()
    return float((rb - s_q.size * (s_q.size + 1) / 2.0) / (s_null.size * s_q.size))


def boot_cosmo(fn, sim_null, sim_q, n_boot, rng):
    """Cluster bootstrap: resample the NULL cosmologies with replacement; each drawn id brings
    ALL its null rows AND all its query rows (paired design -- variates re-simulate the same ids)."""
    uniq = np.unique(sim_null)
    idx_n = {g: np.flatnonzero(sim_null == g) for g in uniq}
    idx_q = {g: np.flatnonzero(sim_q == g) for g in uniq}
    vals = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(uniq, len(uniq), replace=True)
        rn = np.concatenate([idx_n[g] for g in draw])
        rq = np.concatenate([idx_q[g] for g in draw])
        vals[b] = fn(rn, rq) if rq.size else np.nan
    return vals


def neglog(p):
    return -np.log(np.clip(p, 1e-12, 1.0))


# ------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{REPO}/ml-checkpoints/gower_npe_finetune_nla_m_bgp_z8_ens1/summaries")
    ap.add_argument("--moments", default=f"{REPO}/ml-checkpoints/gower_npe_finetune_nla_m_bgp_z8_ens1/misspec")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase6"))
    ap.add_argument("--repeats", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--match-fmt", default="ncosmo300_{r}")
    ap.add_argument("--max-train", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-splits", type=int, default=60)
    ap.add_argument("--n-draws", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--recommend", default="auto", help="method written to combined_p, or 'auto' (see rule in main)")
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R = len(args.repeats)
    methods = all_methods(R)

    sc, ids, meta = stage0(args.root, args.repeats, args.match_fmt, args.max_train,
                           cache=os.path.join(args.out, "stage0_scores.npz"))
    sets = ["_idtest"] + VARIATES
    sim_id = ids["_idtest"]["sim_ids"]
    uniq = np.unique(sim_id)
    nA_cos = len(uniq) // 2
    rng = np.random.default_rng(args.seed)
    splits = [rng.choice(uniq, nA_cos, replace=False) for _ in range(args.n_splits)]
    print(f"_idtest: {len(sim_id)} rows / {len(uniq)} cosmologies; {args.n_splits} random splits of {nA_cos} fit-cosmologies")

    # ---------------- Criterion 1: calibration over random cosmology splits ----------------
    ctlS = "gower_nla"
    c1 = {m: {"mean_p": [], "D": [], "ctl_D": [], "ctl_mean_p": []} for m in methods}
    ncosB = None
    for s, cosA in enumerate(splits):
        maskA = np.isin(sim_id, cosA)
        rowsB = ~maskA
        comb_id = fit_fold(sc, ids, maskA, "_idtest")
        comb_ctl = fit_fold(sc, ids, maskA, ctlS)
        P_B, Pk_B = stage1_query(sc, maskA, "_idtest", rowsB)
        rows_ctl = ~np.isin(ids[ctlS]["sim_ids"], cosA)
        P_c, Pk_c = stage1_query(sc, maskA, ctlS, rows_ctl)
        for m in methods:
            pB = comb_id.p2(m, P_B, Pk_B)
            D, ncosB = cluster_ks(pB, sim_id[rowsB], args.n_draws, np.random.default_rng(args.seed + s))
            c1[m]["mean_p"].append(pB.mean()); c1[m]["D"].append(D)
            pc = comb_ctl.p2(m, P_c, Pk_c)
            Dc, _ = cluster_ks(pc, ids[ctlS]["sim_ids"][rows_ctl], args.n_draws, np.random.default_rng(args.seed + s))
            c1[m]["ctl_D"].append(Dc); c1[m]["ctl_mean_p"].append(pc.mean())
        if s % 10 == 0:
            print(f"[crit1] split {s}/{args.n_splits} done")
    crit, hmed = KS_CRIT / np.sqrt(ncosB), KS_MED / np.sqrt(ncosB)
    crit1 = {}
    print(f"\n=== CRITERION 1: calibration on held-out half B ({ncosB} cosmologies; crit={crit:.3f}, H0 median {hmed:.3f}) ===")
    for m in methods:
        mp = np.array(c1[m]["mean_p"]); D = np.array(c1[m]["D"]); Dc = np.array(c1[m]["ctl_D"])
        ok_a = abs(mp.mean() - 0.5) <= mp.std()
        ok_b = np.median(D) < crit
        power = np.median(Dc) > crit
        crit1[m] = {"mean_p": float(mp.mean()), "mean_p_sd": float(mp.std()), "D_med": float(np.median(D)),
                    "frac_splits_D_below_crit": float((D < crit).mean()), "crit": float(crit), "h0_median": float(hmed),
                    "ctl_D_med": float(np.median(Dc)), "ctl_mean_p": float(np.mean(c1[m]["ctl_mean_p"])),
                    "pass_a": bool(ok_a), "pass_b": bool(ok_b), "control_fires": bool(power), "n_cos_B": int(ncosB)}
        print(f"  {m:>9}: (a) mean_p={mp.mean():.4f}+-{mp.std():.4f} {'ok' if ok_a else 'BAD'} | "
              f"(b) D_med={np.median(D):.4f} ({(D < crit).mean() * 100:.0f}% splits < crit) {'ok' if ok_b else 'BAD'} | "
              f"+control {ctlS}: D_med={np.median(Dc):.3f} mean_p={np.mean(c1[m]['ctl_mean_p']):.3f} "
              f"{'POWER OK' if power else 'NO POWER'}  {'PASS' if ok_a and ok_b and power else '*** FAIL ***'}")

    # ---------------- Cross-fit at the fixed split; ship p-values ----------------
    cosA0 = splits[0]
    p2, fold, p1 = crossfit(sc, ids, cosA0, sets)
    assert all(np.isfinite(p2[S][m]).all() for S in sets for m in methods)

    # ---------------- Criterion 2: AUROC null vs variate, cluster bootstrap ----------------
    rngb = np.random.default_rng(args.seed + 7)
    crit2 = {S: {} for S in VARIATES}
    print("\n=== CRITERION 2: AUROC(_idtest vs variate), by-cosmology bootstrap sd / 95% CI ===")
    for S in VARIATES:
        sim_q = ids[S]["sim_ids"]
        for m in methods:
            sn, sq = neglog(p2["_idtest"][m]), neglog(p2[S][m])
            a = auroc_fast(sn, sq)
            bs = boot_cosmo(lambda rn, rq: auroc_fast(sn[rn], sq[rq]), sim_id, sim_q, args.n_boot, rngb)
            crit2[S][m] = {"auroc": a, "sd": float(np.nanstd(bs)), "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]}
        singles = [crit2[S][f"single{r}"]["auroc"] for r in range(R)]
        crit2[S]["single"] = {"auroc": float(np.mean(singles)), "sd": float(np.mean([crit2[S][f"single{r}"]["sd"] for r in range(R)])),
                              "ci95": [float("nan")] * 2, "per_encoder": singles}
        rb = int(np.argmax(singles))
        crit2[S]["single_best"] = dict(crit2[S][f"single{rb}"], which=rb)
        line = " ".join(f"{m}={crit2[S][m]['auroc']:.3f}+-{crit2[S][m]['sd']:.3f}" for m in ["single", "single_best", "meanp", "fisher", "dose5", "dose10", "gls", "stouffer", "tippett"])
        print(f"  {S:16s} n_cos={len(np.unique(sim_q))} n={len(sim_q)}: {line}")

    # split-to-split sd of AUROC (re-cross-fit on every random split)
    split_auroc = {S: {m: [] for m in methods} for S in VARIATES}
    for cosA in splits:
        q, _, _ = crossfit(sc, ids, cosA, sets)
        for S in VARIATES:
            for m in methods:
                split_auroc[S][m].append(auroc_fast(neglog(q["_idtest"][m]), neglog(q[S][m])))
    for S in VARIATES:
        for m in methods:
            crit2[S][m]["split_mean"] = float(np.mean(split_auroc[S][m])); crit2[S][m]["split_sd"] = float(np.std(split_auroc[S][m]))

    # ---------------- Criterion 3: Spearman(-log p, posterior |z| of truth) ----------------
    print("\n=== CRITERION 3: Spearman rho(-log p_combined, |z_post(truth)|), mean over the 5 posteriors; by-cosmology bootstrap sd ===")
    crit3 = {S: {} for S in VARIATES}
    rngs = np.random.default_rng(args.seed + 11)
    zabs = {}
    for S in VARIATES:
        tf = ids[S]["test_files"]
        zs = []
        for r in args.repeats:
            mo = load(os.path.join(args.moments, S, f"misspec_posterior_moments_{args.match_fmt.format(r=r)}.npz"))
            pos = {f: i for i, f in enumerate(mo["test_files"].astype(str))}
            have = np.array([f in pos for f in tf])                      # inner join on test_files
            sel = np.array([pos[f] for f in tf[have]])
            assert np.allclose(mo["z"][sel][:, :2], ((mo["theta0"][sel] - mo["mean"][sel]) / mo["std"][sel])[:, :2]), "moments z != (theta0-mean)/std"
            za = np.full((len(tf), 2), np.nan)
            za[have] = np.abs(mo["z"][sel][:, :2])
            if (~have).sum():
                print(f"  [join] {S} r{r}: {(~have).sum()} summaries rows have no posterior moments -> dropped from rho")
            zs.append(za)
        zabs[S] = np.stack(zs)  # [R, n, 2], NaN where the join has no moments row
    zabs["_idtest"] = zabs["gower_bgp_nla_m"][:, [ {f: i for i, f in enumerate(ids["gower_bgp_nla_m"]["test_files"])}[f] for f in ids["_idtest"]["test_files"]]]
    for S in VARIATES:
        sim_q = ids[S]["sim_ids"]
        for m in methods + ["single"]:
            def rho_fn(rows, m=m):
                out = np.zeros(2)
                for j, r in enumerate(args.repeats):
                    mm = f"single{r}" if m == "single" else m       # single: encoder r's p vs encoder r's posterior
                    x = neglog(p2[S][mm][rows])
                    ok = np.isfinite(zabs[S][j, rows, 0])
                    out += np.array([spearmanr(x[ok], zabs[S][j, rows, k][ok]).statistic for k in range(2)])
                return out / R
            rho = rho_fn(np.arange(len(sim_q)))
            bs = np.stack([rho_fn(np.concatenate([np.flatnonzero(sim_q == g) for g in rngs.choice(np.unique(sim_q), len(np.unique(sim_q)), replace=True)]))
                           for _ in range(min(args.n_boot, 300))])
            crit3[S][m] = {"rho_om": float(rho[0]), "rho_s8": float(rho[1]), "sd_om": float(bs[:, 0].std()), "sd_s8": float(bs[:, 1].std())}
        singles = [(crit3[S][f"single{r}"]["rho_om"] + crit3[S][f"single{r}"]["rho_s8"]) / 2 for r in range(R)]
        crit3[S]["single_best"] = dict(crit3[S][f"single{int(np.argmax(singles))}"], which=int(np.argmax(singles)))
        print(f"  {S:16s}: " + " ".join(f"{m}=({crit3[S][m]['rho_om']:+.3f},{crit3[S][m]['rho_s8']:+.3f})" for m in ["single", "meanp", "fisher", "dose5", "dose10", "gls", "stouffer", "tippett"]))

    # ---------------- Recommendation rule (fixed in advance) ----------------
    # A candidate replaces mean-p only if it beats it on EVERY hard variate by more than one
    # bootstrap sd in AUROC AND is not worse on the hard-variate Spearman (either parameter).
    def beats(m):
        for S in HARD:
            if crit2[S][m]["auroc"] - crit2[S]["meanp"]["auroc"] <= crit2[S]["meanp"]["sd"]:
                return False
            if crit3[S][m]["rho_om"] < crit3[S]["meanp"]["rho_om"] or crit3[S][m]["rho_s8"] < crit3[S]["meanp"]["rho_s8"]:
                return False
        return crit1[m]["pass_a"] and crit1[m]["pass_b"] and crit1[m]["control_fires"]
    if args.recommend == "auto":
        winners = [m for m in ["dose5", "dose10", "gls", "stouffer", "tippett", "fisher"] if beats(m)]
        rec = winners[0] if winners else "meanp"
        print(f"\n[recommend] candidates beating mean-p on all hard variates by > 1 sd AND on Spearman: {winners or 'none'} -> combined_p = {rec}")
    else:
        rec = args.recommend

    # ---------------- Write per-set npz ----------------
    for S in sets:
        payload = {"combined_score": neglog(p2[S][rec]), "combined_p": p2[S][rec], "test_files": ids[S]["test_files"],
                   "sim_ids": ids[S]["sim_ids"], "fold": fold[S], "method": np.array(rec), "split_seed": np.array(args.seed),
                   "fit_cosmologies_fold0": cosA0}
        for m in methods:
            payload[f"p_{m}"] = p2[S][m]
        for r in args.repeats:
            payload[f"p1_cond_knn_r{r}"] = p1[S][r]
        np.savez_compressed(os.path.join(args.out, f"multiencoder_ood_{S}.npz"), **payload)
    print(f"[write] per-set npz -> {args.out}/multiencoder_ood_<set>.npz (combined_p = {rec})")

    # ---------------- Acceptance tables ----------------
    show = ["single", "single_best", "meanp", "fisher", "dose5", "dose10", "gls", "stouffer", "tippett"]
    res = {"recommended": rec, "criterion1": crit1, "criterion2": crit2, "criterion3": crit3, "n_splits": args.n_splits,
           "n_draws": args.n_draws, "n_boot": args.n_boot, "seed": args.seed, "dims": meta["dims"], "hard": HARD}
    with open(os.path.join(args.out, "phase6_acceptance.json"), "w") as f:
        json.dump(res, f, indent=1)
    L = [f"# Phase 6 acceptance tables (seed {args.seed}, {args.n_splits} splits, {args.n_boot} cosmology bootstraps)\n",
         f"Recommended combiner written to `combined_p`: **{rec}**\n",
         f"\n## Criterion 1 - calibrated null on held-out half B ({ncosB} cosmologies; KS crit {crit:.3f}, H0 median {hmed:.3f}); +control = gower_nla\n",
         "| method | mean_p (per-split sd) | (a) | median cluster-KS D | (b) | control D | power | verdict |", "|---|---|---|---|---|---|---|---|"]
    for m in show:
        if m in ("single", "single_best"):
            rows = [crit1[f"single{r}"] for r in range(R)]
            L.append(f"| {LABELS[m]} | {np.mean([x['mean_p'] for x in rows]):.4f} ({np.mean([x['mean_p_sd'] for x in rows]):.4f}) | {'ok' if all(x['pass_a'] for x in rows) else 'BAD'} | "
                     f"{min(x['D_med'] for x in rows):.3f}-{max(x['D_med'] for x in rows):.3f} | {'ok' if all(x['pass_b'] for x in rows) else 'BAD'} | "
                     f"{min(x['ctl_D_med'] for x in rows):.2f}-{max(x['ctl_D_med'] for x in rows):.2f} | {'ok' if all(x['control_fires'] for x in rows) else 'NO'} | "
                     f"{'PASS' if all(x['pass_a'] and x['pass_b'] and x['control_fires'] for x in rows) else 'FAIL'} ({R} encoders) |")
            continue
        x = crit1[m]
        L.append(f"| {LABELS[m]} | {x['mean_p']:.4f} ({x['mean_p_sd']:.4f}) | {'ok' if x['pass_a'] else 'BAD'} | {x['D_med']:.3f} | {'ok' if x['pass_b'] else 'BAD'} | "
                 f"{x['ctl_D_med']:.2f} | {'ok' if x['control_fires'] else 'NO'} | {'PASS' if x['pass_a'] and x['pass_b'] and x['control_fires'] else 'FAIL'} |")
    L += ["\n## Criterion 2 - AUROC(_idtest vs variate) +- by-cosmology bootstrap sd  [split-to-split sd in brackets]\n",
          "| method | " + " | ".join(f"{S} (n_cos={len(np.unique(ids[S]['sim_ids']))})" for S in VARIATES) + " |", "|---|" + "---|" * len(VARIATES)]
    for m in show:
        cells = []
        for S in VARIATES:
            x = crit2[S][m]
            cells.append(f"{x['auroc']:.3f} +- {x['sd']:.3f}" + (f" [{x['split_sd']:.3f}]" if "split_sd" in x else "") + (f" (r{x['which']})" if m == "single_best" else ""))
        L.append(f"| {LABELS[m]} | " + " | ".join(cells) + " |")
    L += ["\n## Criterion 3 - Spearman rho(-log p, |z_post(truth)|) for (omega_m, sigma_8), mean over the 5 posteriors, +- by-cosmology bootstrap sd\n",
          "| method | " + " | ".join(VARIATES) + " |", "|---|" + "---|" * len(VARIATES)]
    for m in show:
        cells = []
        for S in VARIATES:
            x = crit3[S][m]
            cells.append(f"{x['rho_om']:+.3f}+-{x['sd_om']:.3f}, {x['rho_s8']:+.3f}+-{x['sd_s8']:.3f}" + (f" (r{x['which']})" if m == "single_best" else ""))
        L.append(f"| {LABELS[m]} | " + " | ".join(cells) + " |")
    with open(os.path.join(args.out, "phase6_acceptance.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))

    if not args.no_figs:
        make_figures(args.out, crit2, crit3, ids, p2, show, R)
    return 0


# ------------------------------------------------------------------------------------------
def make_figures(out, crit2, crit3, ids, p2, show, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PAL = {"single": "#52514e", "single_best": "#a3a29e", "meanp": "#2a78d6", "fisher": "#eb6834", "dose5": "#1baf7a",
           "dose10": "#008300", "gls": "#eda100", "stouffer": "#e87ba4", "tippett": "#4a3aa7"}
    names = {"gower_bgp_nla_m": "in-dist (plumbing null)", "gower_nla": "NLA", "gower_nla_z": "NLA-z", "gower_vd": "variable depth",
             "gower_gb1p0": "b_g = 1.0", "gower_gb1p3": "b_g = 1.3"}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharex=False)
    for ax, S in zip(axes.ravel(), VARIATES):
        y = np.arange(len(show))[::-1]
        for yi, m in zip(y, show):
            x = crit2[S][m]
            lo, hi = x["ci95"] if np.isfinite(x["ci95"][0]) else (x["auroc"] - 1.96 * x["sd"], x["auroc"] + 1.96 * x["sd"])
            ax.barh(yi, x["auroc"] - 0.5, left=0.5, height=0.62, color=PAL[m], alpha=0.9)
            ax.plot([lo, hi], [yi, yi], color="#0b0b0b", lw=1.2)
            ax.text(max(hi, 0.5) + 0.01, yi, f"{x['auroc']:.2f}", va="center", fontsize=7.5, color="#52514e")
        ax.axvline(0.5, color="#8a8984", lw=1, ls="--")
        ax.set_yticks(y); ax.set_yticklabels([LABELS[m] for m in show] if ax in axes[:, 0] else [""] * len(show), fontsize=7.5)
        ax.set_title(f"{names[S]}  (n_cos={len(np.unique(ids[S]['sim_ids']))})", fontsize=9.5, loc="left")
        hardx = S in HARD
        ax.set_xlim(0.35 if hardx else 0.35, 0.8 if hardx else 1.05)
        ax.set_xlabel("AUROC (in-dist null vs variate)")
        ax.grid(axis="x", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    fig.suptitle("Multi-encoder OOD combiners: AUROC with 95% by-cosmology bootstrap intervals (5 encoders, Gower BGP arm)", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out, "phase6_auroc.png"), dpi=170); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for ax, S in zip(axes.ravel(), VARIATES):
        y = np.arange(len(show))[::-1]
        for yi, m in zip(y, show):
            x = crit3[S][m]
            for k, (key, sdk, off, mk) in enumerate([("rho_om", "sd_om", 0.17, "o"), ("rho_s8", "sd_s8", -0.17, "s")]):
                ax.errorbar(x[key], yi + off, xerr=x[sdk], fmt=mk, ms=4.5, color=PAL[m], ecolor="#0b0b0b", elinewidth=0.9, capsize=0)
        ax.axvline(0, color="#8a8984", lw=1, ls="--")
        ax.set_yticks(y); ax.set_yticklabels([LABELS[m] for m in show] if ax in axes[:, 0] else [""] * len(show), fontsize=7.5)
        ax.set_title(names[S], fontsize=9.5, loc="left"); ax.set_xlim(-0.15, 0.6)
        ax.set_xlabel("Spearman rho(-log p, |z| of truth)")
        ax.grid(axis="x", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    fig.suptitle("Per-event association with posterior bias: circle = omega_m, square = sigma_8 (mean over the 5 posteriors; bars = by-cosmology bootstrap sd)", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out, "phase6_spearman.png"), dpi=170); plt.close(fig)
    print(f"[figs] {out}/phase6_auroc.png, phase6_spearman.png")


if __name__ == "__main__":
    raise SystemExit(main())
