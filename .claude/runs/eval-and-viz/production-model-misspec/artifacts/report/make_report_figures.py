"""Compute every number and figure in misspec_report.tex from the on-disk analysis products.

Outputs (all in this directory):
  fig_headtohead.pdf   AUROC head-to-head (KL vs OOD) + cost curve (AUROC vs number of encoders)
  fig_detectability.pdf  per-event KL vs A_IA^total and the detection-power curve / threshold
  numbers.tex          \newcommand macros consumed by misspec_report.tex
  numbers.json         the same numbers, machine-readable

Statistical rules: every CI is a paired cluster bootstrap over COSMOLOGIES (sim_id); the null
cosmologies are resampled with replacement and each drawn id brings all its in-distribution rows
and all its variate rows (identical to `gower_multiencoder_ood.boot_cosmo`).
"""
import json
import os
import sys
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
REPO = "/home/alex/work/glass_gower_transfer"
sys.path.insert(0, ART)
sys.path.insert(0, REPO)
from gower_style import COLORS, LABELS  # noqa: E402
from gower_aia_vs_kl import nla_m_effective, unscale, AIA_BOX, BIA_BOX, J_AIA, J_BIA  # noqa: E402
from src.ml.eval.ensemble_discrepancies import diag_gaussian_symmetric_kl  # noqa: E402
from src.KiDS.simulation_config import GALAXY_BIAS_PRIOR_MEANS, GALAXY_BIAS_PRIOR_SIGMAS  # noqa: E402

CK = f"{REPO}/ml-checkpoints/gower_npe_finetune_nla_m_bgp_z8_ens1"
MIS = f"{CK}/misspec"
P6 = f"{ART}/phase6"
MATCHES = [f"ncosmo300_{i}" for i in range(5)]
TAG = "_".join(MATCHES)
INDIST = "gower_bgp_nla_m"
VARIATES = ["gower_gb1p0", "gower_gb1p3", "gower_vd", "gower_nla", "gower_nla_z"]
N_BOOT = 1000
SEED = 0
FPR = 0.05

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"], "mathtext.fontset": "dejavuserif",
})
DET_COL = {"kl": "#000000", "knn": "#56B4E9", "cond_knn": "#0072B2", "meanp": "#E69F00"}
DET_LAB = {"kl": "cross-encoder KL (5 encoders)", "knn": "OOD $k$NN (1 encoder)",
           "cond_knn": "OOD conditional $k$NN (1 encoder)", "meanp": "OOD combined mean-$p$ (5 encoders)"}


# ----------------------------------------------------------------------------- utilities
def auroc(s_null, s_q):
    ranks = rankdata(np.concatenate([s_null, s_q]))
    rb = ranks[s_null.size:].sum()
    return float((rb - s_q.size * (s_q.size + 1) / 2.0) / (s_null.size * s_q.size))


def cosmo_draws(sim_null, sim_q, n_boot, rng):
    """Paired cluster bootstrap index draws (null cosmologies resampled; query rows follow)."""
    uniq = np.unique(sim_null)
    idx_n = {g: np.flatnonzero(sim_null == g) for g in uniq}
    idx_q = {g: np.flatnonzero(sim_q == g) for g in uniq}
    out = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, len(uniq), replace=True)
        rn = np.concatenate([idx_n[g] for g in draw])
        rq = np.concatenate([idx_q[g] for g in draw]) if any(idx_q[g].size for g in draw) else np.array([], int)
        out.append((rn, rq))
    return out


def boot_stat(fn, draws):
    v = np.array([fn(rn, rq) if rq.size else np.nan for rn, rq in draws], float)
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)), float(np.nanstd(v))


def empirical_p(null, q):
    """Right-tail empirical p against the null: (1 + #{null >= s}) / (1 + n)."""
    ns = np.sort(null)
    n_ge = ns.size - np.searchsorted(ns, q, side="left")
    return (1.0 + n_ge) / (1.0 + ns.size)


# ----------------------------------------------------------------------------- loading
def load_stage0():
    d = np.load(f"{P6}/stage0_scores.npz", allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    sc = {r: {S: {k: d[f"{r}/{S}/{k}"] for k in ["knn", "cond_knn", "null_knn", "null_cond_knn"]}
              for S in meta["sets"]} for r in meta["repeats"]}
    ids = {S: {"sim_ids": d[f"ids/{S}/sim_ids"], "test_files": d[f"ids/{S}/test_files"].astype(str)}
           for S in meta["sets"]}
    return sc, ids, meta


def load_kl(variate):
    with np.load(f"{MIS}/{variate}/misspec_repeat_disagreement_{TAG}.npz", allow_pickle=True) as f:
        return {"kl": f["kl_score"].astype(float), "mu": f["mu"].astype(np.float64),
                "var": f["var"].astype(np.float64), "test_files": f["test_files"].astype(str)}


def load_p6(setname):
    with np.load(f"{P6}/multiencoder_ood_{setname}.npz", allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def load_theta(variate, match="ncosmo300_0"):
    with np.load(f"{MIS}/{variate}/misspec_posterior_samples_{match}.npz", allow_pickle=True) as f:
        return {"theta0s": f["theta0s"].astype(float), "test_files": f["test_files"].astype(str),
                "sim_ids": f["sim_ids"], "aug_ids": f["aug_ids"]}


def main():
    rng = np.random.default_rng(SEED)
    sc, ids, meta = load_stage0()
    R = meta["repeats"]
    numbers = {}

    # ---- the null: KL of the in-distribution set and the OOD scores of _idtest (same files) ----
    kl_null = load_kl(INDIST)
    # same 1590 files as the OOD null, stored in a different order: re-order the KL null by filename
    assert set(kl_null["test_files"]) == set(ids["_idtest"]["test_files"]), "KL null / OOD null are not the same files"
    ik = {f: i for i, f in enumerate(kl_null["test_files"])}
    sel = np.array([ik[f] for f in ids["_idtest"]["test_files"]])
    kl_null = {"kl": kl_null["kl"][sel], "mu": kl_null["mu"][:, sel], "var": kl_null["var"][:, sel],
               "test_files": kl_null["test_files"][sel]}
    p6_null = load_p6("_idtest")
    assert (p6_null["test_files"].astype(str) == ids["_idtest"]["test_files"]).all()
    sim_null = ids["_idtest"]["sim_ids"]
    null = {
        "kl": kl_null["kl"],
        "meanp": -p6_null["p_meanp"].astype(float),
        **{f"knn{r}": sc[r]["_idtest"]["knn"] for r in R},
        **{f"cond_knn{r}": sc[r]["_idtest"]["cond_knn"] for r in R},
    }
    numbers["n_null_rows"] = int(sim_null.size)
    numbers["n_null_cos"] = int(np.unique(sim_null).size)

    # ---- per-variate joined tables ----
    tables = {}
    for S in VARIATES:
        klv = load_kl(S)
        p6v = load_p6(S)
        files0 = ids[S]["test_files"]
        assert (p6v["test_files"].astype(str) == files0).all()
        common = sorted(set(klv["test_files"]) & set(files0))
        i0 = {f: i for i, f in enumerate(files0)}
        ik = {f: i for i, f in enumerate(klv["test_files"])}
        s0 = np.array([i0[f] for f in common]); sk = np.array([ik[f] for f in common])
        for r in R:  # the unconditional kNN null is the _idtest cloud itself; the conditional null depends on the theta-dims
            assert np.allclose(sc[r][S]["null_knn"], sc[r]["_idtest"]["knn"])
        t = {
            "files": np.array(common), "sim_ids": ids[S]["sim_ids"][s0],
            **{f"null_cond_knn{r}": sc[r][S]["null_cond_knn"] for r in R},
            **{f"null_knn{r}": sc[r][S]["null_knn"] for r in R},
            "kl": klv["kl"][sk], "mu": klv["mu"][:, sk], "var": klv["var"][:, sk],
            "meanp": -p6v["p_meanp"].astype(float)[s0],
            **{f"knn{r}": sc[r][S]["knn"][s0] for r in R},
            **{f"cond_knn{r}": sc[r][S]["cond_knn"][s0] for r in R},
            "n_rows_ood": int(files0.size), "n_rows_kl": int(klv["test_files"].size), "n_common": len(common),
        }
        tables[S] = t
        numbers[f"{S}_n_rows"] = int(files0.size)
        numbers[f"{S}_n_common"] = len(common)
        numbers[f"{S}_n_cos"] = int(np.unique(t["sim_ids"]).size)
        numbers[f"{S}_kl_mean"] = float(klv["kl"].mean())
        numbers[f"{S}_kl_median"] = float(np.median(klv["kl"]))
    numbers[f"{INDIST}_kl_mean"] = float(kl_null["kl"].mean())
    numbers[f"{INDIST}_kl_median"] = float(np.median(kl_null["kl"]))

    # ---- head-to-head AUROC with paired cluster bootstrap ----
    detectors = ["kl", "knn", "cond_knn", "meanp"]
    H = {}
    print("\n=== HEAD-TO-HEAD AUROC (variate vs in-distribution), 95% by-cosmology CI ===")
    for S in VARIATES:
        t = tables[S]
        draws = cosmo_draws(sim_null, t["sim_ids"], N_BOOT, rng)
        H[S] = {}
        for det in detectors:
            if det in ("knn", "cond_knn"):
                def fn(rn, rq, det=det):
                    return float(np.mean([auroc(t[f"null_{det}{r}"][rn], t[f"{det}{r}"][rq]) for r in R]))
                point = fn(np.arange(sim_null.size), np.arange(t["sim_ids"].size))
                per_enc = [auroc(t[f"null_{det}{r}"], t[f"{det}{r}"]) for r in R]
            else:
                def fn(rn, rq, det=det):
                    return auroc(null[det][rn], t[det][rq])
                point = fn(np.arange(sim_null.size), np.arange(t["sim_ids"].size))
                per_enc = None
            lo, hi, sd = boot_stat(fn, draws)
            H[S][det] = {"auroc": point, "lo": lo, "hi": hi, "sd": sd, "per_encoder": per_enc}
            numbers[f"{S}_auroc_{det}"] = point; numbers[f"{S}_auroc_{det}_lo"] = lo; numbers[f"{S}_auroc_{det}_hi"] = hi
            numbers[f"{S}_auroc_{det}_sd"] = sd
            print(f"  {S:14s} {det:9s} AUROC={point:.3f} [{lo:.3f}, {hi:.3f}] sd={sd:.3f}"
                  + (f"  per-encoder {np.round(per_enc, 3).tolist()}" if per_enc else ""))
        # power at FPR=5% (row-level null quantile), CI by cosmology
        for det in ["kl", "meanp", "knn", "cond_knn"]:
            if det in ("knn", "cond_knn"):
                taus = [np.quantile(t[f"null_{det}{r}"], 1 - FPR) for r in R]
                def pw(rn, rq, det=det, taus=taus):
                    return float(np.mean([np.mean(t[f"{det}{r}"][rq] > taus[r]) for r in R]))
            else:
                tau = np.quantile(null[det], 1 - FPR)
                def pw(rn, rq, det=det, tau=tau):
                    return float(np.mean(t[det][rq] > tau))
            point = pw(None, np.arange(t["sim_ids"].size))
            lo, hi, sd = boot_stat(pw, draws)
            numbers[f"{S}_power_{det}"] = point; numbers[f"{S}_power_{det}_lo"] = lo; numbers[f"{S}_power_{det}_hi"] = hi
            print(f"  {S:14s} {det:9s} power@5%FPR={point:.3f} [{lo:.3f}, {hi:.3f}]")
        # per-event agreement between the two detectors
        rho = spearmanr(t["kl"], t["meanp"]).correlation
        numbers[f"{S}_spearman_kl_meanp"] = float(rho)
        print(f"  {S:14s} Spearman(KL, -p_meanp) = {rho:+.3f}")
    numbers["fpr"] = FPR
    numbers["kl_null_p95"] = float(np.quantile(null["kl"], 0.95))
    numbers["kl_null_median"] = float(np.median(null["kl"]))
    # the null's own false-alarm rate for meanp at p<0.05 (calibration check)
    numbers["meanp_null_frac_lt005"] = float(np.mean(-null["meanp"] < 0.05))

    # ---- cost curve: AUROC vs number of encoders K ----
    print("\n=== COST CURVE: AUROC vs number of encoders ===")
    cost = {}
    for S in ["gower_nla", "gower_gb1p0", "gower_nla_z", "gower_gb1p3", "gower_vd"]:
        t = tables[S]
        draws = cosmo_draws(sim_null, t["sim_ids"], N_BOOT, rng)
        cost[S] = {"kl": {}, "cond_knn": {}, "knn": {}}
        # KL over every K-subset of encoders
        for K in range(2, 6):
            subs = list(combinations(range(5), K))
            kl_null_K = [diag_gaussian_symmetric_kl(kl_null["mu"][list(s)], kl_null["var"][list(s)]) for s in subs]
            kl_q_K = [diag_gaussian_symmetric_kl(t["mu"][list(s)], t["var"][list(s)]) for s in subs]
            def fn(rn, rq):
                return float(np.mean([auroc(a[rn], b[rq]) for a, b in zip(kl_null_K, kl_q_K)]))
            point = fn(np.arange(sim_null.size), np.arange(t["sim_ids"].size))
            lo, hi, sd = boot_stat(fn, draws)
            cost[S]["kl"][K] = (point, lo, hi)
            print(f"  {S:12s} KL   K={K}: {point:.3f} [{lo:.3f},{hi:.3f}]  ({len(subs)} subsets)")
        # OOD: mean of the per-encoder empirical p-values over every K-subset (K=1 is the single encoder)
        for det in ["cond_knn", "knn"]:
            p_null = np.stack([empirical_p(t[f"null_{det}{r}"], t[f"null_{det}{r}"]) for r in range(5)])
            p_q = np.stack([empirical_p(t[f"null_{det}{r}"], t[f"{det}{r}"]) for r in range(5)])
            for K in range(1, 6):
                subs = list(combinations(range(5), K))
                def fn(rn, rq):
                    return float(np.mean([auroc(-p_null[list(s)].mean(0)[rn], -p_q[list(s)].mean(0)[rq]) for s in subs]))
                point = fn(np.arange(sim_null.size), np.arange(t["sim_ids"].size))
                lo, hi, sd = boot_stat(fn, draws)
                cost[S][det][K] = (point, lo, hi)
                print(f"  {S:12s} {det:8s} K={K}: {point:.3f} [{lo:.3f},{hi:.3f}]")
    for S in cost:
        for det in cost[S]:
            for K, (p, lo, hi) in cost[S][det].items():
                numbers[f"cost_{S}_{det}_K{K}"] = p; numbers[f"cost_{S}_{det}_K{K}_lo"] = lo; numbers[f"cost_{S}_{det}_K{K}_hi"] = hi

    # ---- detectability vs A_IA^total ----
    print("\n=== DETECTABILITY vs A_IA^total ===")
    th_id = load_theta(INDIST)
    a_eff = nla_m_effective(unscale(th_id["theta0s"][:, J_AIA], AIA_BOX), unscale(th_id["theta0s"][:, J_BIA], BIA_BOX))
    band_lo, band_hi = np.percentile(a_eff, [5, 95]); band_med = float(np.median(a_eff))
    numbers.update({"aia_band_lo": float(band_lo), "aia_band_hi": float(band_hi), "aia_band_median": band_med,
                    "aia_band_min": float(a_eff.min()), "aia_band_max": float(a_eff.max())})
    print(f"  NLA-M training support A_IA^total: median {band_med:.3f}, 5-95% [{band_lo:.3f},{band_hi:.3f}]")

    grid = np.linspace(-6, 6, 241)
    HALF = 0.3  # sliding-window half-width in A_IA^total
    numbers["aia_window_halfwidth"] = HALF
    det_curves = {}
    thr = {}
    for S in ["gower_nla", "gower_nla_z"]:
        t = tables[S]
        th = load_theta(S)
        # a_ia for nla / nla_z is already A_IA^total; align by file
        amap = dict(zip(th["test_files"], unscale(th["theta0s"][:, J_AIA], AIA_BOX)))
        A = np.array([amap[f] for f in t["files"]])
        numbers[f"{S}_aia_min"] = float(A.min()); numbers[f"{S}_aia_max"] = float(A.max()); numbers[f"{S}_aia_median"] = float(np.median(A))
        draws = cosmo_draws(sim_null, t["sim_ids"], N_BOOT, rng)
        det_curves[S] = {"A": A, "kl": t["kl"], "meanp": t["meanp"]}
        thr[S] = {}
        for det in ["kl", "meanp", "knn"]:
            if det == "knn":  # single-encoder detector: per-row detection fraction averaged over the 5 encoders
                flag = np.mean([t[f"knn{r}"] > np.quantile(t[f"null_knn{r}"], 1 - FPR) for r in R], axis=0)
            else:
                tau = np.quantile(null[det], 1 - FPR)
                flag = (t[det] > tau).astype(float)
            def curve(rq):
                Aq, fq = A[rq], flag[rq]
                out = np.full(grid.size, np.nan)
                for i, g in enumerate(grid):
                    k = np.abs(Aq - g) <= HALF
                    if k.sum() >= 20:
                        out[i] = fq[k].mean()
                return out
            c0 = curve(np.arange(A.size))
            cb = np.array([curve(rq) for _, rq in draws])
            lo_c, hi_c = np.nanpercentile(cb, 2.5, axis=0), np.nanpercentile(cb, 97.5, axis=0)
            det_curves[S][f"curve_{det}"] = (c0, lo_c, hi_c)
            # power inside the training band
            inband = (A >= band_lo) & (A <= band_hi)
            pb = float(flag[inband].mean())
            pb_b = [float(flag[rq][(A[rq] >= band_lo) & (A[rq] <= band_hi)].mean()) for _, rq in draws]
            numbers[f"{S}_{det}_power_inband"] = pb
            numbers[f"{S}_{det}_power_inband_lo"] = float(np.nanpercentile(pb_b, 2.5))
            numbers[f"{S}_{det}_power_inband_hi"] = float(np.nanpercentile(pb_b, 97.5))
            numbers[f"{S}_n_inband"] = int(inband.sum())
            # crossings: walk outward from the band median on each side
            def crossings(c, level):
                res = {}
                i0 = int(np.argmin(np.abs(grid - band_med)))
                up = [g for g, v in zip(grid[i0:], c[i0:]) if np.isfinite(v) and v >= level]
                dn = [g for g, v in zip(grid[:i0 + 1][::-1], c[:i0 + 1][::-1]) if np.isfinite(v) and v >= level]
                res["up"] = up[0] if up else np.nan
                res["dn"] = dn[0] if dn else np.nan
                return res
            for level, tagl in [(0.5, "50"), (0.8, "80")]:
                c = crossings(c0, level)
                cbs = [crossings(cc, level) for cc in cb]
                for side in ["up", "dn"]:
                    vals = np.array([x[side] for x in cbs], float)
                    thr[S][f"{det}_{tagl}_{side}"] = (c[side], np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5))
                    numbers[f"{S}_thr_{det}_{tagl}_{side}"] = float(c[side])
                    numbers[f"{S}_thr_{det}_{tagl}_{side}_lo"] = float(np.nanpercentile(vals, 2.5))
                    numbers[f"{S}_thr_{det}_{tagl}_{side}_hi"] = float(np.nanpercentile(vals, 97.5))
                    numbers[f"{S}_dthr_{det}_{tagl}_{side}"] = float(abs(c[side] - band_med))
                    print(f"  {S:12s} {det:6s} power>={level:.1f}: {side} crossing at A={c[side]:+.2f} "
                          f"[{np.nanpercentile(vals, 2.5):+.2f},{np.nanpercentile(vals, 97.5):+.2f}]  "
                          f"|dA|={abs(c[side] - band_med):.2f}")
            # power in the windows centred at band_med +- 0.5 (the "invisible" zone quoted in the text)
            for side, sgn in [("dn", -1), ("up", +1)]:
                ig = int(np.argmin(np.abs(grid - (band_med + sgn * 0.5))))
                numbers[f"{S}_powerhalf_{det}_{side}"] = float(c0[ig])
                numbers[f"{S}_powerhalf_{det}_{side}_lo"] = float(lo_c[ig])
                numbers[f"{S}_powerhalf_{det}_{side}_hi"] = float(hi_c[ig])
                print(f"  {S:12s} {det:6s} power at A=median{sgn:+d}*0.5 ({grid[ig]:+.2f}): {c0[ig]:.3f} [{lo_c[ig]:.3f},{hi_c[ig]:.3f}]")
            # onset: first grid point outward where the LOWER CI exceeds the false-alarm rate
            i0 = int(np.argmin(np.abs(grid - band_med)))
            up = [g for g, v in zip(grid[i0:], lo_c[i0:]) if np.isfinite(v) and v > FPR]
            dn = [g for g, v in zip(grid[:i0 + 1][::-1], lo_c[:i0 + 1][::-1]) if np.isfinite(v) and v > FPR]
            numbers[f"{S}_onset_{det}_up"] = float(up[0]) if up else np.nan
            numbers[f"{S}_onset_{det}_dn"] = float(dn[0]) if dn else np.nan
            print(f"  {S:12s} {det:6s} onset (lower CI > {FPR}): up {numbers[f'{S}_onset_{det}_up']:+.2f}, dn {numbers[f'{S}_onset_{det}_dn']:+.2f}; "
                  f"in-band power {pb:.3f} [{numbers[f'{S}_{det}_power_inband_lo']:.3f},{numbers[f'{S}_{det}_power_inband_hi']:.3f}] (n={inband.sum()})")
        # where does the KL bottom out? (binned median of log KL)
        cen = np.array([np.median(t["kl"][np.abs(A - g) <= HALF]) if (np.abs(A - g) <= HALF).sum() >= 20 else np.nan for g in grid])
        numbers[f"{S}_kl_argmin_A"] = float(grid[np.nanargmin(cen)])
        numbers[f"{S}_kl_min_median"] = float(np.nanmin(cen))
        print(f"  {S:12s} binned-median KL minimum {np.nanmin(cen):.3f} at A={grid[np.nanargmin(cen)]:+.2f}")
        det_curves[S]["kl_median_curve"] = cen

    # ---- galaxy-bias prior z-scores of the two fixed-b_g variates ----
    m = np.array(GALAXY_BIAS_PRIOR_MEANS); s = np.array(GALAXY_BIAS_PRIOR_SIGMAS)
    for name, bg in [("gb1p0", 1.0), ("gb1p3", 1.3)]:
        z = (bg - m) / s
        for i, zi in enumerate(z):
            numbers[f"{name}_bgz_bin{i + 1}"] = float(zi)
        numbers[f"{name}_n_bins_outside3sig"] = int((np.abs(z) > 3).sum())
        print(f"  {name}: b_g z-scores {np.round(z, 2).tolist()}")
    for i in range(6):
        numbers[f"bg_prior_mean_bin{i + 1}"] = float(m[i]); numbers[f"bg_prior_sigma_bin{i + 1}"] = float(s[i])

    # ---- paired physical shift (from the on-disk result file) ----
    import re
    with open(f"{ART}/gower_paired_physical_RESULT.txt") as f:
        txt = f.read()
    for par in ["omega_m", "sigma_8", "w0"]:
        mm = re.search(rf"{par}\s*\|\s*([+-]?\d+\.\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([+-]?\d+\.\d+) sigma", txt)
        numbers[f"dtheta_{par}"] = float(mm.group(1)); numbers[f"dtheta_{par}_sd"] = float(mm.group(2))
        numbers[f"sigpost_{par}"] = float(mm.group(3)); numbers[f"dtheta_{par}_ratio"] = float(mm.group(4))
    numbers["des_bg_bound_omega_m"] = 0.007
    numbers["dtheta_omega_m_over_des"] = abs(numbers["dtheta_omega_m"]) / 0.007

    # ---- calibration numbers (from the on-disk json) ----
    with open(f"{ART}/gower_misspec_cal_vs_disagreement.json") as f:
        cal = json.load(f)
    with open(f"{ART}/gower_misspec_matched_floor.json") as f:
        floor = json.load(f)
    for S, d in cal.items():
        numbers[f"{S}_cal_mean"] = d["cal_full_mean"]; numbers[f"{S}_cal_sd"] = d["cal_full_std"]
    numbers["floor40_mean"] = floor["40x2"]["cal_full_mean"]; numbers["floor40_sd"] = floor["40x2"]["cal_full_sd"]
    numbers["floor40_p95"] = floor["40x2"]["cal_full_p95"]
    numbers["floor199_mean"] = floor["199x8"]["cal_full_mean"]; numbers["floor199_sd"] = floor["199x8"]["cal_full_sd"]
    numbers["floor199_p95"] = floor["199x8"]["cal_full_p95"]
    for S in ["gower_gb1p0", "gower_gb1p3"]:
        numbers[f"{S}_cal_zfloor"] = (cal[S]["cal_full_mean"] - floor["40x2"]["cal_full_mean"]) / floor["40x2"]["cal_full_sd"]
    numbers["gower_vd_cal_zfloor"] = (cal["gower_vd"]["cal_full_mean"] - floor["199x8"]["cal_full_mean"]) / floor["199x8"]["cal_full_sd"]

    # ---- cross-encoder score correlation on the null (probit scale) from the per-encoder p's ----
    from scipy.stats import norm
    U = np.stack([norm.isf(np.clip(p6_null[f"p1_cond_knn_r{r}"], 1e-6, 1 - 1e-6)) for r in range(5)])
    C = np.corrcoef(U)
    numbers["rho_cross_encoder_null"] = float(np.mean(C[np.triu_indices(5, 1)]))
    numbers["snr_gain_5"] = float(np.sqrt(5 / (1 + 4 * numbers["rho_cross_encoder_null"])))
    numbers["n_train_summaries"] = 19167

    # ============================================================== FIGURE 1: head-to-head + cost
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.25), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    order = ["gower_gb1p3", "gower_gb1p0", "gower_vd", "gower_nla", "gower_nla_z"]
    W = 0.19
    for j, det in enumerate(detectors):
        xs = np.arange(len(order)) + (j - 1.5) * W
        ys = [H[S][det]["auroc"] for S in order]
        err = np.array([[H[S][det]["auroc"] - H[S][det]["lo"] for S in order],
                        [H[S][det]["hi"] - H[S][det]["auroc"] for S in order]])
        ax.bar(xs, ys, W * 0.92, color=DET_COL[det], edgecolor="black", linewidth=0.4, label=DET_LAB[det],
               hatch=("//" if det == "kl" else None), zorder=3)
        ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor="0.25", elinewidth=0.7, capsize=1.5, zorder=4)
    ax.axhline(0.5, color="0.4", lw=0.8, ls=":", zorder=2)
    ax.set_xticks(np.arange(len(order)))
    short = {"gower_gb1p3": "$b_g=1.3$\nfixed", "gower_gb1p0": "$b_g=1.0$\nfixed", "gower_vd": "variable\ndepth",
             "gower_nla": "NLA", "gower_nla_z": "NLA-$z$"}
    ax.set_xticklabels([short[S] for S in order])
    for lab, S in zip(ax.get_xticklabels(), order):
        lab.set_color(COLORS[S])
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("AUROC (variate vs in-distribution)")
    ax.set_title("(a) Detector head-to-head", loc="left")
    ax.legend(loc="upper left", frameon=False, ncol=1, handlelength=1.4)
    ax.grid(axis="y", alpha=0.3, lw=0.5)

    ax = axes[1]
    for S, ls in [("gower_nla", "-"), ("gower_gb1p0", "--")]:
        for det, col, mk in [("kl", DET_COL["kl"], "s"), ("cond_knn", DET_COL["cond_knn"], "o"), ("knn", DET_COL["knn"], "^")]:
            Ks = sorted(cost[S][det]); y = [cost[S][det][K][0] for K in Ks]
            lo = [cost[S][det][K][1] for K in Ks]; hi = [cost[S][det][K][2] for K in Ks]
            ax.plot(Ks, y, ls=ls, color=col, marker=mk, ms=3.5, lw=1.1,
                    label=(f"{ {'kl': 'KL', 'cond_knn': 'cond. $k$NN', 'knn': '$k$NN'}[det]}" if S == "gower_nla" else None))
            ax.fill_between(Ks, lo, hi, color=col, alpha=0.12, lw=0)
    ax.axhline(0.5, color="0.4", lw=0.8, ls=":")
    ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xlabel("number of trained encoders $K$")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.3, 1.0)
    ax.set_title("(b) Cost: AUROC vs. number of encoders", loc="left")
    from matplotlib.lines import Line2D
    h1, l1 = ax.get_legend_handles_labels()
    h2 = [Line2D([0], [0], color="0.3", ls="-"), Line2D([0], [0], color="0.3", ls="--")]
    ax.legend(h1 + h2, l1 + ["NLA", r"$b_g=1.0$ fixed"], loc="center", bbox_to_anchor=(0.6, 0.6), frameon=False,
              ncol=2, handlelength=1.8, fontsize=6.5, columnspacing=0.8)
    ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(f"{HERE}/fig_headtohead.pdf")
    plt.close(fig)

    # ============================================================== FIGURE 2: detectability
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
    ax = axes[0]
    S = "gower_nla"; d = det_curves[S]
    ax.axvspan(band_lo, band_hi, color="0.85", zorder=0)
    ax.axvline(band_med, color="0.45", lw=0.9, zorder=1)
    ax.scatter(d["A"], d["kl"], s=4, alpha=0.25, color=COLORS[S], lw=0, zorder=2, rasterized=True)
    ax.plot(grid, d["kl_median_curve"], color=COLORS[S], lw=1.8, zorder=4, label="NLA: running median")
    dz = det_curves["gower_nla_z"]
    ax.plot(grid, dz["kl_median_curve"], color=COLORS["gower_nla_z"], lw=1.4, ls="--", zorder=4, label="NLA-$z$: running median")
    ax.axhline(numbers["kl_null_p95"], color="black", lw=0.9, ls=":", zorder=3,
               label=f"in-distribution 95th pct. ({numbers['kl_null_p95']:.2f})")
    ax.axhline(numbers["kl_null_median"], color="black", lw=0.9, ls="-.", zorder=3,
               label=f"in-distribution median ({numbers['kl_null_median']:.2f})")
    ax.set_yscale("log"); ax.set_xlim(-6, 6); ax.set_ylim(0.02, 2000)
    ax.set_xlabel(r"true $A_{\rm IA}^{\rm total}$ of the test mock")
    ax.set_ylabel("cross-encoder KL (5 encoders, nats)")
    ax.set_title("(a) Disagreement vs. IA amplitude", loc="left")
    ax.legend(loc="upper left", frameon=False, handlelength=1.8, ncol=2, columnspacing=1.0)
    ax.annotate("NLA-M training\nsupport (5$-$95%)", xy=(band_hi, 30), xytext=(1.6, 25), fontsize=6.5, color="0.3",
                ha="left", va="center", arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6))
    ax.grid(alpha=0.25, lw=0.5)

    ax = axes[1]
    ax.axvspan(band_lo, band_hi, color="0.85", zorder=0)
    ax.axvline(band_med, color="0.45", lw=0.9, zorder=1)
    for S, ls in [("gower_nla", "-"), ("gower_nla_z", "--")]:
        for det, col in [("kl", DET_COL["kl"]), ("knn", DET_COL["knn"]), ("meanp", DET_COL["meanp"])]:
            c0, lo_c, hi_c = det_curves[S][f"curve_{det}"]
            ax.plot(grid, c0, ls=ls, color=col, lw=1.4, zorder=3,
                    label=(f"{ {'kl': 'KL (5 enc.)', 'knn': 'OOD $k$NN (1 enc.)', 'meanp': 'OOD mean-$p$ (5 enc.)'}[det]}" if S == "gower_nla" else None))
            ax.fill_between(grid, lo_c, hi_c, color=col, alpha=0.12, lw=0, zorder=2)
    ax.axhline(FPR, color="0.4", lw=0.8, ls=":", zorder=2)
    ax.axhline(0.5, color="0.4", lw=0.6, ls=":", zorder=2)
    for side in ["dn", "up"]:
        x = numbers[f"gower_nla_thr_kl_50_{side}"]
        ax.plot([x], [0.5], marker="v", color=DET_COL["kl"], ms=5, zorder=5)
    ax.set_xlim(-6, 6); ax.set_ylim(0, 1)
    ax.set_xlabel(r"true $A_{\rm IA}^{\rm total}$ of the test mock")
    ax.set_ylabel(f"power at {int(FPR * 100)}% false-alarm rate")
    ax.set_title("(b) Detection power vs. IA amplitude", loc="left")
    h1, l1 = ax.get_legend_handles_labels()
    h2 = [Line2D([0], [0], color="0.3", ls="-"), Line2D([0], [0], color="0.3", ls="--")]
    ax.legend(h1 + h2, l1 + ["NLA", "NLA-$z$"], loc="lower left", frameon=True, framealpha=0.92, edgecolor="none",
              ncol=1, handlelength=1.8, fontsize=6.5)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(f"{HERE}/fig_detectability.pdf", dpi=300)
    plt.close(fig)

    # ============================================================== numbers out
    def clean(v):
        if isinstance(v, (np.floating, float)):
            return None if not np.isfinite(v) else float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        return v
    numbers = {k: clean(v) for k, v in numbers.items()}
    with open(f"{HERE}/numbers.json", "w") as f:
        json.dump(numbers, f, indent=1)

    def macro(k):
        # LaTeX macro names: letters only
        s = k.replace("gower_", "").replace("_", " ").title().replace(" ", "")
        digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five",
                  "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
        s = "".join(digits.get(ch, ch) for ch in s if ch.isalnum())
        return "N" + s
    lines = []
    seen = {}
    for k, v in numbers.items():
        name = macro(k)
        if name in seen:
            raise RuntimeError(f"macro clash {name}: {k} vs {seen[name]}")
        seen[name] = k
        if v is None:
            txt = "n/a"
        elif isinstance(v, int):
            txt = f"{v}"
        elif k.endswith("over_des") or k.endswith("_ratio") or "zfloor" in k:
            txt = f"{v:.1f}" if (k.endswith("over_des") or "zfloor" in k) else f"{v:.2f}"
        elif "bgz" in k:
            txt = f"{v:+.1f}"
        elif abs(v) >= 100:
            txt = f"{v:.0f}"
        elif abs(v) >= 10:
            txt = f"{v:.1f}"
        elif "power" in k or "auroc" in k or "thr" in k or "onset" in k or "aia" in k or "spearman" in k or "rho" in k or "snr" in k or "kl_" in k or "dthr" in k:
            txt = f"{v:.2f}"
        elif "cal" in k or "floor" in k or "dtheta" in k or "sigpost" in k:
            txt = f"{v:.4f}" if abs(v) < 0.1 else f"{v:.3f}"
        else:
            txt = f"{v:.3f}"
        lines.append(f"\\newcommand{{\\{name}}}{{{txt}}}  % {k}")
    with open(f"{HERE}/numbers.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {len(lines)} macros to numbers.tex")


if __name__ == "__main__":
    main()
