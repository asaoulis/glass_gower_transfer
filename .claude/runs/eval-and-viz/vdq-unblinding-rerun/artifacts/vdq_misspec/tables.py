#!/usr/bin/env python
"""P5.4 (kNN-axis detector matrix), P5.5 (KL axis + calibration + z bias) and P5.7 (b_g / IA
framing numbers) for the VDQ packs. Ports of production-model-misspec's
gower_detector_matrix.py, offline_repeat_disagreement.py + gower_misspec_zscores.py,
gower_paired_dz.py, gower_paired_physical.py, gower_aia_vs_kl.py (conversion) -- parametrised
through vdq_config, cosmology-cluster statistics throughout.

    python tables.py --pack field --sections detector kl bgia [--n-boot 1000]

Writes $OUT_MISSPEC/tables/<pack>_{detector,klaxis,bgia}.json. Variates = every VARIATES dir
present locally (so the same command runs on the 2 local variates now and on all 7 later).

Detector conventions
  per-encoder detectors (knn, cond_knn, mahalanobis): in-job p-values of each repeat's member-0
    summaries (each against ITS OWN null -- for the NLA family the conditional model drops a_ia,
    so raw cond scores are not comparable across variates but p-values are); AUROC on -log10 p
    vs the in-dist variate (== the _idtest files), averaged over encoders.
  meanp_knn: the obs-identical event-level statistic (meanp_knn.py; in-dist rows LOCO).
  meanp_cond: phase6 combined_p (multiencoder_ood --recommend meanp), if the phase6 dir exists.
  kl: cross-repeat disagreement, pack-adopted statistic (+ 'full' on the field pack).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")      # get_tarp_coverage's per-bootstrap progress bars
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K  # noqa: E402
import vdq_config as C  # noqa: E402

PER_ENC = {"knn": "ood_knn_p", "cond_knn": "ood_cond_knn_p", "mahalanobis": "ood_mahalanobis_p"}


def _nlp(p):
    return -np.log10(np.clip(np.asarray(p, float), 1e-12, 1))


# =============================================================================================
# P5.4 detector matrix
# =============================================================================================
def _per_encoder(pk, v):
    """{det: [N, R] p}, files, sims, theta_dims per encoder (aligned on the j=0 file order)."""
    R = len(pk["repeats"])
    s0 = K.summ(pk, v, 0)
    files = np.array([K.b(x) for x in s0["test_files"]])
    out = {d: np.full((len(files), R), np.nan) for d in PER_ENC}
    dims = []
    for j in range(R):
        s = s0 if j == 0 else K.summ(pk, v, j)
        pos = {K.b(x): i for i, x in enumerate(s["test_files"])}
        idx = np.array([pos[f] for f in files])
        for d, key in PER_ENC.items():
            if key in s:
                out[d][:, j] = np.asarray(s[key], float)[idx]
        tag = pk["summ_matches"][j]
        jp = os.path.join(pk["summaries"], v, f"ood_{tag}.json")
        if os.path.exists(jp):
            dims.append(K.load_json(jp).get("theta_dims"))
    return out, files, np.asarray(s0["sim_ids"], int), dims


def _meanp_knn(pk, v):
    d = K.load_npz(pk["meanp_npz"])
    m = d["variate"] == v
    return np.array([K.b(f) for f in d["test_files"][m]]), d["sim_ids"][m].astype(int), d["meanp_knn_recal"][m]


def _meanp_cond(pk, v):
    S = "_idtest" if v == C.IN_DIST else v
    p = os.path.join(pk["phase6_dir"], f"multiencoder_ood_{S}.npz")
    if not os.path.exists(p):
        return None
    d = K.load_npz(p)
    return np.array([K.b(f) for f in d["test_files"]]), d["sim_ids"].astype(int), np.asarray(d["combined_p"], float)


def detector_matrix(pk, variates, n_boot, seed):
    rng = np.random.default_rng(seed)
    R = len(pk["repeats"])
    P_ind, f_ind, sim_ind, dims_ind = _per_encoder(pk, C.IN_DIST)
    ev_ind = {"meanp_knn": _meanp_knn(pk, C.IN_DIST), "meanp_cond": _meanp_cond(pk, C.IN_DIST)}
    res = {"pack": pk["exp"], "repeats": pk["repeats"], "n_boot": n_boot, "in_dist": C.IN_DIST,
           "in_dist_n_rows": int(len(f_ind)), "in_dist_n_cos": int(len(np.unique(sim_ind))),
           "phase6_present": ev_ind["meanp_cond"] is not None, "variates": {}}
    # in-dist rates (the reference column)
    ref = {}
    for d in PER_ENC:
        ref[d] = {a: float(np.nanmean(P_ind[d] < a)) for a in (0.05, 0.01)}
    for d, ev in ev_ind.items():
        if ev is not None:
            ref[d] = {a: float(np.mean(ev[2] < a)) for a in (0.05, 0.01)}
    res["in_dist_flag_rates"] = ref
    for v in variates:
        if v == C.IN_DIST:
            continue
        t0 = time.time()
        P_v, f_v, sim_v, dims_v = _per_encoder(pk, v)
        n_cos, rpc = K.shape_of(sim_v)
        draws = K.cosmo_draws(sim_ind, sim_v, n_boot, rng)
        row = {"n_rows": int(len(f_v)), "n_cos": int(n_cos), "rows_per_cos": rpc,
               "theta_dims_cond": dims_v[0] if dims_v else None, "detectors": {}}
        dets = {}
        for d in PER_ENC:
            if not np.isfinite(P_v[d]).any():
                continue
            dets[d] = ("enc", P_ind[d], P_v[d])
        for d in ("meanp_knn", "meanp_cond"):
            q = _meanp_knn(pk, v) if d == "meanp_knn" else _meanp_cond(pk, v)
            if q is None or ev_ind[d] is None:
                continue
            dets[d] = ("event", ev_ind[d], q)
        for d, (kind, A, B) in dets.items():
            if kind == "enc":
                sA, sB = _nlp(A), _nlp(B)
                fn = (lambda rn, rq: float(np.nanmean([K.auroc(sA[rn, j], sB[rq, j]) for j in range(R)])))
                au = fn(np.arange(len(sA)), np.arange(len(sB)))
                bs = np.array([fn(rn, rq) for rn, rq in draws])
                flag = {a: float(np.nanmean(B < a)) for a in (0.05, 0.01)}
                flag_ci = {a: K.cluster_ci((B < a).mean(1), sim_v, n_boot=n_boot, seed=seed) for a in (0.05, 0.01)}
                med = float(np.mean([np.nanmedian(B[:, j]) for j in range(R)]))
                null_med = []
                for _ in range(2000):
                    sel = K.matched_draw(sim_ind, n_cos, rpc, rng)
                    null_med.append(np.mean([np.nanmedian(A[sel, j]) for j in range(R)]))
                per_enc_auroc = [K.auroc(sA[:, j], sB[:, j]) for j in range(R)]
            else:
                (fa, sa, pa), (fb, sb, pb) = A, B
                # event-level sides keep their own row order; pair by cosmology for the bootstrap
                dr = K.cosmo_draws(sa, sb, n_boot, rng)
                au = K.auroc(_nlp(pa), _nlp(pb))
                bs = np.array([K.auroc(_nlp(pa[rn]), _nlp(pb[rq])) for rn, rq in dr])
                flag = {a: float(np.mean(pb < a)) for a in (0.05, 0.01)}
                flag_ci = {a: K.cluster_ci((pb < a).astype(float), sb, n_boot=n_boot, seed=seed) for a in (0.05, 0.01)}
                med = float(np.median(pb))
                null_med = [np.median(pa[K.matched_draw(sa, n_cos, rpc, rng)]) for _ in range(2000)]
                per_enc_auroc = None
            null_med = np.asarray(null_med)
            row["detectors"][d] = {
                "auroc": au, "auroc_ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                "auroc_per_encoder": per_enc_auroc,
                "frac_p_lt_0.05": flag[0.05], "frac_p_lt_0.05_ci95": flag_ci[0.05],
                "frac_p_lt_0.01": flag[0.01], "frac_p_lt_0.01_ci95": flag_ci[0.01],
                "in_dist_frac_p_lt_0.05": ref[d][0.05], "in_dist_frac_p_lt_0.01": ref[d][0.01],
                "median_p": med, "median_p_null_mean": float(null_med.mean()),
                "median_p_pte": float((np.sum(null_med <= med) + 1) / (null_med.size + 1)),
            }
        res["variates"][v] = row
        print(f"[detector] {pk['key']} {v}: " + "; ".join(
            f"{d} AUROC {x['auroc']:.3f} [{x['auroc_ci95'][0]:.3f},{x['auroc_ci95'][1]:.3f}] "
            f"flag05 {x['frac_p_lt_0.05']:.3f} PTE {x['median_p_pte']:.3g}" for d, x in row["detectors"].items())
            + f"  ({time.time() - t0:.0f}s)", flush=True)
    return res


# =============================================================================================
# P5.5 KL axis + calibration + z bias
# =============================================================================================
def _cal_from_samples(S, th, seed):
    from src.ml.eval.evaluate_models import TARPDiagnostics
    from src.ml.eval.tarp import get_tarp_coverage
    # seed=None exactly as production (misspec.py TARPDiagnostics(seed=None)): get_tarp_coverage
    # re-seeds np.random INSIDE every bootstrap iteration when given a seed, which makes all 25
    # bootstrap resamples identical (no averaging -> a ~1/(N/100) rank-histogram MSE, e.g. 1.15 at
    # N=80 instead of ~0.05). Reproducibility comes from seeding the global stream once here.
    np.random.seed(seed)
    cov = get_tarp_coverage(S.copy(), th.copy(), bootstrap=True, num_bootstrap=25, seed=None, num_alpha_bins=100)
    return TARPDiagnostics._calibration_error_from_coverage(cov)


def _cal_json(pk, v):
    full, sub, avail = [], [], None
    for j in range(len(pk["repeats"])):
        m = K.results(pk, v, j)["metrics"]
        full.append(m["tarp"]["full"]["calibration_error"])
        sub.append(m["tarp"]["subsets"].get(K.SUBSET_KEY, {}).get("calibration_error", np.nan))
        avail = m["available_params"]
    return np.array(full), np.array(sub), avail


def _floor(pk, shape, avail, n_draws, seed, ind_cache):
    """Cluster-matched in-dist calibration floor at the target's (n_cos x rows/cos) and params.
    Draws are spread over the pack's repeats (each repeat's own in-dist posterior)."""
    rng = np.random.default_rng(seed)
    n_cos, rpc = shape
    R = len(pk["repeats"])
    full, sub = [], []
    for i in range(n_draws):
        j = i % R
        S, th, sims, names = ind_cache(j)
        ia = [names.index(p) for p in avail]
        i3 = [names.index(p) for p in K.KEY3]
        sel = K.matched_draw(sims, n_cos, rpc, rng)
        full.append(_cal_from_samples(S[:, sel][:, :, ia], th[sel][:, ia], seed + i))
        sub.append(_cal_from_samples(S[:, sel][:, :, i3], th[sel][:, i3], seed + 10_000 + i))
    full, sub = np.array(full), np.array(sub)
    return {"n_draws": n_draws, "shape": [n_cos, rpc], "params": avail,
            "cal_full_mean": float(full.mean()), "cal_full_sd": float(full.std()), "cal_full_p95": float(np.percentile(full, 95)),
            "cal3_mean": float(sub.mean()), "cal3_sd": float(sub.std()), "cal3_p95": float(np.percentile(sub, 95))}


def kl_axis(pk, variates, n_boot, seed, floor_draws):
    rng = np.random.default_rng(seed + 1)
    rows = K.rows(pk, variates)
    kls = [pk["kl"]] + (["full"] if pk["kl"] != "full" else [])
    ev = {(v, w): K.kl_event(pk, v, w) for v in variates for w in kls}
    f_i, s_i, _ = ev[(C.IN_DIST, kls[0])]

    _ind = {}

    def ind_cache(j):
        if j not in _ind:
            d = K.load_npz(K.samples_path(pk, C.IN_DIST, j))
            names = [str(x) for x in d["params"]] if "params" in d else K.PARAMS9
            _ind.clear()                       # keep ONE repeat's samples in memory (~230 MB)
            _ind[j] = (np.asarray(d["samples"], np.float64), np.asarray(d["theta0s"], np.float64),
                       np.asarray(d["sim_ids"], int), names)
        return _ind[j]

    cal_i_full, cal_i_sub, avail_i = _cal_json(pk, C.IN_DIST)
    res = {"pack": pk["exp"], "kl_adopted": pk["kl"], "kl_reported": kls, "variates": {}, "floors": {}}
    zr = {}
    id_rows = rows["variate"] == C.IN_DIST
    for p in ("omega_m", "sigma_8", "S8"):
        z = rows[f"z_{p}"][id_rows]
        zr[p] = {t: float(np.nanmean(np.abs(z) > t)) for t in K.THRESH}
    for v in variates:
        t0 = time.time()
        row = {}
        # ---- KL ----
        for w in kls:
            f_v, s_v, k_v = ev[(v, w)]
            _, _, k_i = ev[(C.IN_DIST, w)]
            thr = float(np.quantile(k_i, 0.95))
            e = {"n_events": int(len(k_v)), "n_cos": int(len(np.unique(s_v))),
                 "mean": float(np.mean(k_v)), "median": float(np.median(k_v)), "p90": float(np.percentile(k_v, 90)),
                 "median_ci95": K.cluster_ci(k_v, s_v, np.median, n_boot=n_boot, seed=seed),
                 "in_dist_p95_threshold": thr}
            if v != C.IN_DIST:
                dr = K.cosmo_draws(s_i, s_v, n_boot, rng)
                e["auroc"] = K.auroc(k_i, k_v)
                bs = [K.auroc(k_i[rn], k_v[rq]) for rn, rq in dr]
                e["auroc_ci95"] = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
                e["power_at_5pct"] = float(np.mean(k_v > thr))
                e["power_at_5pct_ci95"] = K.cluster_ci((k_v > thr).astype(float), s_v, n_boot=n_boot, seed=seed)
            row[f"kl_{w}"] = e
        # ---- calibration ----
        cf, cs, avail = _cal_json(pk, v)
        row["cal"] = {"available_params": avail, "cal_full_per_repeat": cf.tolist(), "cal3_per_repeat": cs.tolist(),
                      "cal_full_mean": float(cf.mean()), "cal_full_sd": float(cf.std()),
                      "cal3_mean": float(cs.mean()), "cal3_sd": float(cs.std())}
        sv = rows["sim_id"][(rows["variate"] == v) & (rows["repeat"] == pk["repeats"][0])]
        shape = K.shape_of(sv)
        ish = K.shape_of(rows["sim_id"][id_rows & (rows["repeat"] == pk["repeats"][0])])
        if v == C.IN_DIST:
            pass
        elif shape[0] * shape[1] >= 0.8 * ish[0] * ish[1]:
            # big variate: the floor IS the in-dist repeat spread on the same params
            key = "indist_" + "_".join(avail)
            if key not in res["floors"]:
                if list(avail) == list(avail_i):
                    fl = {"kind": "in-dist repeat spread (json)", "params": avail,
                          "cal_full_mean": float(cal_i_full.mean()), "cal_full_sd": float(cal_i_full.std()),
                          "cal3_mean": float(cal_i_sub.mean()), "cal3_sd": float(cal_i_sub.std())}
                else:
                    vals = []
                    for j in range(len(pk["repeats"])):
                        S, th, sims, names = ind_cache(j)
                        ia = [names.index(p) for p in avail]
                        vals.append(_cal_from_samples(S[:, :, ia], th[:, ia], seed + j))
                    vals = np.array(vals)
                    fl = {"kind": "in-dist repeat spread (recomputed on the variate's params)", "params": avail,
                          "cal_full_per_repeat": vals.tolist(),
                          "cal_full_mean": float(vals.mean()), "cal_full_sd": float(vals.std()),
                          "cal3_mean": float(cal_i_sub.mean()), "cal3_sd": float(cal_i_sub.std())}
                res["floors"][key] = fl
            row["cal"]["floor"] = key
        else:
            key = f"matched_{shape[0]}x{shape[1]}_" + "_".join(avail)
            if key not in res["floors"]:
                fl = _floor(pk, shape, avail, floor_draws, seed, ind_cache)
                fl["kind"] = "cluster-matched in-dist draws"
                res["floors"][key] = fl
            row["cal"]["floor"] = key
        if v != C.IN_DIST:
            fl = res["floors"][row["cal"]["floor"]]
            for c in ("cal_full", "cal3"):
                sd = fl[f"{c}_sd"] or 1e-12
                row["cal"][f"{c}_z_vs_floor"] = float((row["cal"][f"{c}_mean"] - fl[f"{c}_mean"]) / sd)
                row["cal"][f"{c}_ratio_to_floor"] = float(row["cal"][f"{c}_mean"] / fl[f"{c}_mean"])
                if f"{c}_p95" in fl:
                    row["cal"][f"{c}_above_floor_p95"] = bool(row["cal"][f"{c}_mean"] > fl[f"{c}_p95"])
        # ---- z bias ----
        m = rows["variate"] == v
        zt = {}
        for p in ("omega_m", "sigma_8", "S8"):
            z = rows[f"z_{p}"][m]
            sims = rows["sim_id"][m]
            ok = np.isfinite(z)
            zt[p] = {"mean_z": float(np.mean(z[ok])), "mean_z_ci95": K.cluster_ci(z[ok], sims[ok], n_boot=n_boot, seed=seed),
                     "mean_z_ci68": K.cluster_ci(z[ok], sims[ok], n_boot=n_boot, seed=seed, q=(16, 84)),
                     "sd_z": float(np.std(z[ok]))}
            for t in K.THRESH:
                pr = float(np.mean(np.abs(z[ok]) > t))
                zt[p][f"P_absz_gt_{t:g}"] = pr
                zt[p][f"P_absz_gt_{t:g}_excess"] = pr - zr[p][t]
                zt[p][f"P_absz_gt_{t:g}_ci95"] = K.cluster_ci((np.abs(z[ok]) > t).astype(float), sims[ok],
                                                              n_boot=n_boot, seed=seed)
        row["z"] = zt
        res["variates"][v] = row
        k0 = row[f"kl_{pk['kl']}"]
        print(f"[kl] {pk['key']} {v}: KL[{pk['kl']}] median {k0['median']:.3f}"
              + (f" AUROC {k0['auroc']:.3f} [{k0['auroc_ci95'][0]:.3f},{k0['auroc_ci95'][1]:.3f}] power5 {k0['power_at_5pct']:.3f}"
                 if "auroc" in k0 else "")
              + f" | cal_full {row['cal']['cal_full_mean']:.4f}+-{row['cal']['cal_full_sd']:.4f}"
              + (f" (z vs floor {row['cal'].get('cal_full_z_vs_floor', float('nan')):+.1f})" if v != C.IN_DIST else "")
              + f" | z_S8 {zt['S8']['mean_z']:+.3f} [{zt['S8']['mean_z_ci95'][0]:+.3f},{zt['S8']['mean_z_ci95'][1]:+.3f}]"
              + f"  ({time.time() - t0:.0f}s)", flush=True)
    res["in_dist_P_absz"] = zr
    return res


# =============================================================================================
# P5.7 b_g + IA framing
# =============================================================================================
def bgia(pk, variates, n_boot, seed):
    names, lo, hi = C.scaler_box(pk["exp"])
    width = dict(zip(names, hi - lo))
    out = {"pack": pk["exp"], "box": {n: [float(a), float(c)] for n, a, c in zip(names, lo, hi)}}
    R = len(pk["repeats"])
    a, bb = C.GB_PAIR
    if a in variates and b_ok(variates, bb):
        per = []
        for j in range(R):
            A, B = K.moments(pk, a, j), K.moments(pk, bb, j)
            ka = {(int(s), int(u)): i for i, (s, u) in enumerate(zip(A["sim_ids"], A["aug_ids"]))}
            kb = {(int(s), int(u)): i for i, (s, u) in enumerate(zip(B["sim_ids"], B["aug_ids"]))}
            common = sorted(set(ka) & set(kb))
            ia, ib = np.array([ka[k] for k in common]), np.array([kb[k] for k in common])
            I = K.moments(pk, C.IN_DIST, j)
            pn = [str(x) for x in A["params"]]
            rec = {"n_a": len(ka), "n_b": len(kb), "n_paired": len(common), "per_param": {}}
            for p in ("omega_m", "sigma_8", "w0"):
                k = pn.index(p)
                dz = B["z"][ib, k] - A["z"][ia, k]
                dth = (B["mean"][ib, k] - A["mean"][ia, k]) * width[p]
                rec["per_param"][p] = {"dz_mean": float(dz.mean()), "dz_se": float(dz.std(ddof=1) / np.sqrt(dz.size)),
                                       "dtheta_mean": float(dth.mean()),
                                       "sigma_post_indist": float((I["std"][:, k] * width[p]).mean())}
            per.append(rec)
        pair = {"paired_all_repeats": all(r["n_paired"] == r["n_a"] == r["n_b"] for r in per), "per_repeat": per,
                "summary": {}}
        for p in ("omega_m", "sigma_8", "w0"):
            dz = np.array([r["per_param"][p]["dz_mean"] for r in per])
            dt = np.array([r["per_param"][p]["dtheta_mean"] for r in per])
            sp = np.mean([r["per_param"][p]["sigma_post_indist"] for r in per])
            pair["summary"][p] = {"dz_mean_over_repeats": float(dz.mean()), "dz_sd_over_repeats": float(dz.std()),
                                  "dtheta_mean_over_repeats": float(dt.mean()), "dtheta_sd_over_repeats": float(dt.std()),
                                  "sigma_post_indist": float(sp), "dtheta_over_sigma": float(dt.mean() / sp)}
        # S8 paired dz from the rows (S8 z from the sample dumps)
        rows = K.rows(pk, [a, bb])
        dzs = []
        for r in pk["repeats"]:
            ma = (rows["variate"] == a) & (rows["repeat"] == r)
            mb = (rows["variate"] == bb) & (rows["repeat"] == r)
            za = dict(zip(rows["file"][ma], rows["z_S8"][ma]))
            zb = dict(zip(rows["file"][mb], rows["z_S8"][mb]))
            com = sorted(set(za) & set(zb))
            dzs.append(np.mean([zb[f] - za[f] for f in com]))
        pair["summary"]["S8"] = {"dz_mean_over_repeats": float(np.mean(dzs)), "dz_sd_over_repeats": float(np.std(dzs))}
        out["gb_pair"] = pair
        print(f"[bgia] {pk['key']} gb pair paired={pair['paired_all_repeats']}: " + "; ".join(
            f"{p} dz {x['dz_mean_over_repeats']:+.3f}+-{x['dz_sd_over_repeats']:.3f}"
            + (f" dtheta {x['dtheta_mean_over_repeats']:+.4f} ({x['dtheta_over_sigma']:+.2f} sigma)" if "dtheta_over_sigma" in x else "")
            for p, x in pair["summary"].items()), flush=True)
    # A_IA ranges from the stored theta0s (moments theta0 = scaled truth)
    ia = names.index("a_ia")
    ib = names.index("b_ia")
    aia = {}
    for v in variates:
        th = K.moments(pk, v, 0)["theta0"]
        a_raw = th[:, ia] * width["a_ia"] + lo[ia]
        rec = {"raw_min": float(np.nanmin(a_raw)), "raw_max": float(np.nanmax(a_raw)), "raw_median": float(np.nanmedian(a_raw))}
        if v in C.NLA_FAMILY:
            rec["convention"] = "A_IA^total (stored directly)"
            A = a_raw
        else:
            rec["convention"] = "raw NLA-M prefactor -> converted"
            A = K.nla_m_effective(a_raw, th[:, ib] * width["b_ia"] + lo[ib])
        rec.update({"A_total_min": float(np.nanmin(A)), "A_total_max": float(np.nanmax(A)),
                    "A_total_median": float(np.nanmedian(A)),
                    "A_total_p5": float(np.nanpercentile(A, 5)), "A_total_p95": float(np.nanpercentile(A, 95))})
        aia[v] = rec
        print(f"[bgia] {pk['key']} {v}: a_ia raw [{rec['raw_min']:.2f},{rec['raw_max']:.2f}] -> A_total "
              f"[{rec['A_total_min']:.2f},{rec['A_total_max']:.2f}] median {rec['A_total_median']:.2f} ({rec['convention']})",
              flush=True)
    out["a_ia"] = aia
    out["gbk2_per_event_bg"] = ("BLOCKED: per-event b_g_bin1..6 are not in the npz (theta is the 9-vector); needs a "
                                "data-h5 dump of gower_vdqk2_nla_m test files (orchestrator)")
    return out


def b_ok(variates, v):
    return v in variates


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", choices=sorted(C.PACKS), required=True)
    ap.add_argument("--sections", nargs="+", default=["detector", "kl", "bgia"], choices=["detector", "kl", "bgia"])
    ap.add_argument("--variates", nargs="+", default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--floor-draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    pk = C.PACKS[a.pack]
    variates = a.variates or [v for v in C.present(pk) if v in C.present(pk, "summaries")]
    print(f"[tables] {a.pack}: variates {variates}", flush=True)
    od = K.out_dir("tables")
    if "detector" in a.sections:
        r = detector_matrix(pk, variates, a.n_boot, a.seed)
        r["variate_list"] = variates
        json.dump(r, open(os.path.join(od, f"{a.pack}_detector.json"), "w"), indent=1, default=float)
    if "kl" in a.sections:
        r = kl_axis(pk, variates, a.n_boot, a.seed, a.floor_draws)
        r["variate_list"] = variates
        json.dump(r, open(os.path.join(od, f"{a.pack}_klaxis.json"), "w"), indent=1, default=float)
    if "bgia" in a.sections:
        r = bgia(pk, variates, a.n_boot, a.seed)
        r["variate_list"] = variates
        json.dump(r, open(os.path.join(od, f"{a.pack}_bgia.json"), "w"), indent=1, default=float)
    print(f"[tables] wrote {od}/{a.pack}_*.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
