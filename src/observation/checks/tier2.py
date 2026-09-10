"""Tier-2: OOD score -> posterior-bias probability tables, from the production misspecification
matrix of the NPE encoder pack ``gower_npe_finetune_nla_m_bgp_z8_ens1`` (5 independently seeded
encoders x 7 Gower variates).

Per (variate, encoder repeat, event) row:
  z_omega_m, z_sigma_8      (theta0 - posterior mean)/posterior std   from misspec_posterior_moments
  z_S8                      same for S_8 = sigma_8 sqrt(Omega_m/0.3), from the FULL sample dump
  knn_p                     that encoder's summary-space kNN right-tail p vs its ID null (summaries npz)
  meanp                     5-encoder mean-p combiner, recalibrated (phase6 npz; event-level)
  kl                        cross-encoder symmetric diag-Gaussian KL disagreement (event-level)

Two detector axes are tabulated: ``meanp`` (kNN family) and ``kl``. For each we bin the score,
and per bin report -- with COSMOLOGY-block bootstrap CIs (rows are ~8 correlated augmentations
per cosmology) -- E[z], E[|z|], and P(|z| > t) for t in {0.3, 0.5, 1.0} for Omega_m, sigma_8, S_8,
next to the in-distribution reference value of P(|z| > t) in the SAME bin (a calibrated posterior
already has P(|z|>1) ~ 0.32; only the EXCESS is evidence of bias). Also the exceedance version
P(|z| > t | score >= s) and the inverse reading s*(t, q): the smallest score cut at which
P(|z| > t | score >= s*) >= q.

Pooling: all variates enter with EQUAL weight per event (the mixture weights are a choice, not a
measurement; per-variate tables are written alongside). The tables are calibrated on the NPE
compressor's posteriors and applied as a PROXY to the NLE arms' posteriors (DECISIONS.md).
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

PARAMS_OF_INTEREST = ("omega_m", "sigma_8", "S8")

# ADOPTED Tier-2 KL statistic (user decision 2026-09-10; DECISIONS T2-6, artifacts/KL_SUBSET_STUDY.md).
# The production detector was the FULL 9-param diagonal-Gaussian KL. It is replaced by the
# full-COVARIANCE symmetric KL over (Omega_m, sigma_8): the diagonal form discards the
# Omega_m-sigma_8 correlation (rho ~ -0.84), which is where the kappa=2 galaxy-bias signal lives --
# every single-dimension AUROC for that variate is ~0.5 while the correlated statistic reaches
# 0.661. It beats the old detector on EVERY variate that carries signal with no loss anywhere
# (nla +0.009, nla_z +0.019, gbk2 +0.124). Set to "full" to restore the previous behaviour.
ADOPTED_KL = "om_s8_fullcov"

# ...but the winner is PACK-DEPENDENT, measured 2026-09-10 (KL_SUBSET_STUDY.md sections 2 and 3).
# On the 2-pt (bandpower) pack the FULL 9-param diagonal KL is best and every variant is worse
# (nla 0.929 vs 0.917 for om_s8_fullcov, nla_z 0.936 vs 0.932), because the kappa=2 galaxy-bias
# signal that om_s8_fullcov recovers on the maps is simply ABSENT from the bandpowers -- b_g
# cancels there, so no estimator can find it (2-pt gbk2 AUROC = 0.500 under every variant).
# Bias tables are built per pack, so the two may legitimately differ; what must never differ is a
# table and the observation reading looked up in it (DECISIONS T2-8), which the stamp enforces.
ADOPTED_KL_BY_PACK = {
    "gower_npe_finetune_band_nla_m_bgp_ens9": "full",     # M17, 2-pt only
}


def adopted_kl_for(base_dir_or_experiment: str) -> str:
    """The Tier-2 KL statistic this pack is scored with. Pack-specific, else :data:`ADOPTED_KL`."""
    import os as _os
    name = _os.path.basename(str(base_dir_or_experiment).rstrip("/"))
    return ADOPTED_KL_BY_PACK.get(name, ADOPTED_KL)
THRESHOLDS = (0.3, 0.5, 1.0)
VARIATES_DEFAULT = ("gower_bgp_nla_m", "gower_nla", "gower_nla_z", "gower_vd", "gower_gb1p0", "gower_gb1p3", "gower_gbk2")
IN_DIST = "gower_bgp_nla_m"


def _s8(om, s8):
    return s8 * np.sqrt(om / 0.3)


def _scaler_box(experiment: str):
    """(names, lo, hi) of the min-max box the NPE pack's samples live in (SCALED [0,1] space)."""
    import eval as _eval  # registers every config family
    cfg, _ = _eval.load_config(experiment)
    from src.ml.utils import _build_cosmo_preset_scaler
    names = list(cfg.cosmo_param_names)
    preset = dict(getattr(cfg, "scaler_options", {}).get("cosmo", {}).get("preset_overrides", {}) or {})
    from src.ml.data.constants import COSMO_PARAM_PRESET_MINMAX
    box = {**COSMO_PARAM_PRESET_MINMAX, **preset}
    sc = _build_cosmo_preset_scaler(box, names)
    return names, np.asarray(sc.min, dtype=np.float64), np.asarray(sc.max, dtype=np.float64)


def load_rows(base_dir: str, experiment: str = "gower_npe_finetune_nla_m_bgp_z8_ens1",
              variates: Sequence[str] = VARIATES_DEFAULT, repeats: Sequence[int] = (0, 1, 2, 3, 4),
              phase6_dir: Optional[str] = None, with_s8: bool = True, max_events_s8: Optional[int] = None,
              match_template: str = "ncosmo300_{r}", kl_params: str = ADOPTED_KL) -> Dict[str, np.ndarray]:
    """Join everything by (variate, test_files basename). Returns column arrays.

    ``match_template`` names the per-repeat filename tag; it is ``ncosmo300_{r}`` for the
    field-level pack AND for the 2-pt M17 pack (confirmed from the misspec job log -- do NOT infer
    it from ``match_num_cosmo``).

    ``kl_params`` selects which ensemble-disagreement statistic fills the ``kl`` column:
    ``full`` (the production default, byte-identical to reading ``kl_score`` off the npz) or any
    key of ``src.ml.eval.kl_subsets.ALL_SCORES``. See ``artifacts/KL_SUBSET_STUDY.md``; a table
    built with one choice must never be read against an observation scored with another.

    Summaries are OPTIONAL: a KL-axis-only pack (no kNN summary extraction, e.g. M17) yields NaN
    ``knn_score``/``knn_p``/``meanp`` and a printed warning rather than a crash.
    """
    names, lo, hi = _scaler_box(experiment)
    iom, is8 = names.index("omega_m"), names.index("sigma_8")
    cols: Dict[str, List] = {k: [] for k in ("variate", "repeat", "file", "sim_id", "z_omega_m", "z_sigma_8", "z_S8",
                                             "knn_score", "knn_p", "kl", "meanp")}
    for v in variates:
        mdir = os.path.join(base_dir, "misspec", v)
        sdir = os.path.join(base_dir, "summaries", v)
        # event-level: KL disagreement + phase6 meanp
        kl_by_file, meanp_by_file = {}, {}
        for f in glob.glob(os.path.join(mdir, "misspec_repeat_disagreement_*.npz")):
            d = np.load(f)
            if d["mu"].shape[0] >= 5:   # the 5-encoder file (a 2-repeat one may also exist)
                files_d = [os.path.basename(x) for x in d["test_files"]]
                if kl_params == "full":
                    kl_vals = d["kl_score"]          # byte-identical to the production path
                else:
                    from src.ml.eval.kl_subsets import load_variate, subset_scores
                    rec = load_variate(base_dir, v, match_template, repeats, names, lo, hi,
                                       max_events_s8=max_events_s8)
                    kl_vals = subset_scores(rec, names)[kl_params]
                    files_d = [os.path.basename(x) for x in rec["files"]]
                kl_by_file = dict(zip(files_d, kl_vals))
        if phase6_dir:
            p6 = os.path.join(phase6_dir, f"multiencoder_ood_{v}.npz")
            if os.path.exists(p6):
                d = np.load(p6, allow_pickle=True)
                meanp_by_file = dict(zip([os.path.basename(x) for x in d["test_files"]], d["combined_p"]))
        for r in repeats:
            m = np.load(os.path.join(mdir, "misspec_posterior_moments_%s.npz" % match_template.format(r=r)))
            files = [os.path.basename(x) for x in m["test_files"]]
            z = m["z"]
            zS8 = np.full(len(files), np.nan)
            if with_s8:
                sp = os.path.join(mdir, "misspec_posterior_samples_%s.npz" % match_template.format(r=r))
                if os.path.exists(sp):
                    d = np.load(sp, mmap_mode="r")
                    S = d["samples"]                       # [S,N,D] scaled
                    th = np.asarray(d["theta0s"])          # [N,D] scaled
                    sfiles = [os.path.basename(x) for x in d["test_files"]]
                    n = S.shape[1] if max_events_s8 is None else min(S.shape[1], max_events_s8)
                    om = np.asarray(S[:, :n, iom]) * (hi[iom] - lo[iom]) + lo[iom]
                    s8 = np.asarray(S[:, :n, is8]) * (hi[is8] - lo[is8]) + lo[is8]
                    S8 = _s8(om, s8)
                    om0 = th[:n, iom] * (hi[iom] - lo[iom]) + lo[iom]
                    s80 = th[:n, is8] * (hi[is8] - lo[is8]) + lo[is8]
                    S80 = _s8(om0, s80)
                    zs = (S80 - S8.mean(0)) / S8.std(0)
                    zmap = dict(zip(sfiles[:n], zs))
                    zS8 = np.array([zmap.get(f, np.nan) for f in files])
            spath = os.path.join(sdir, "summaries_%s.npz" % match_template.format(r=r))
            if os.path.exists(spath):
                s = np.load(spath)
                sfiles = [os.path.basename(x) for x in s["test_files"]]
                knn_s = dict(zip(sfiles, s["ood_knn_score"]))
                knn_p = dict(zip(sfiles, s["ood_knn_p"]))
            else:
                # KL-axis-only pack (no summary extraction was run): NaN the kNN family rather
                # than crash, so the KL table can still be built.
                print("[tier2] no summaries at %s -- knn/meanp columns will be NaN" % spath, flush=True)
                knn_s, knn_p = {}, {}
            for i, f in enumerate(files):
                cols["variate"].append(v)
                cols["repeat"].append(r)
                cols["file"].append(f)
                cols["sim_id"].append(int(m["sim_ids"][i]))
                cols["z_omega_m"].append(float(z[i, iom]))
                cols["z_sigma_8"].append(float(z[i, is8]))
                cols["z_S8"].append(float(zS8[i]))
                cols["knn_score"].append(float(knn_s.get(f, np.nan)))
                cols["knn_p"].append(float(knn_p.get(f, np.nan)))
                cols["kl"].append(float(kl_by_file.get(f, np.nan)))
                cols["meanp"].append(float(meanp_by_file.get(f, np.nan)))
    out = {k: np.asarray(vv) for k, vv in cols.items()}
    return out


def _boot_cosmo(values: np.ndarray, sim_ids: np.ndarray, stat, n_boot: int = 300, seed: int = 0):
    """Bootstrap ``stat(values)`` by resampling COSMOLOGY blocks."""
    rng = np.random.default_rng(seed)
    sims = np.unique(sim_ids)
    idx_by = {s: np.where(sim_ids == s)[0] for s in sims}
    est = []
    for _ in range(n_boot):
        pick = rng.choice(sims, size=len(sims), replace=True)
        idx = np.concatenate([idx_by[s] for s in pick])
        est.append(stat(values[idx]))
    est = np.asarray(est)
    return float(np.nanpercentile(est, 16)), float(np.nanpercentile(est, 84))


def bias_table(rows: Dict[str, np.ndarray], score: str, *, edges: Optional[Sequence[float]] = None,
               n_bins: int = 6, params: Sequence[str] = PARAMS_OF_INTEREST, thresholds: Sequence[float] = THRESHOLDS,
               in_dist: str = IN_DIST, n_boot: int = 200, seed: int = 0, lower_is_ood: bool = True) -> Dict:
    """Bin ``score`` (a p-value: LOW = OOD; or KL: HIGH = OOD, pass lower_is_ood=False) and tabulate
    the bias statistics per bin (pooled over variates, equal weight per row) with the in-dist
    reference in the same bin, plus exceedance P(|z| > t | more-OOD-than s)."""
    s = rows[score]
    ok = np.isfinite(s)
    if edges is None:
        if score in ("knn_p", "meanp"):
            edges = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0000001]
        else:
            edges = list(np.nanquantile(s[ok], np.linspace(0, 1, n_bins + 1)))
            edges[-1] += 1e-9
    edges = list(edges)
    which = np.digitize(s, edges) - 1
    is_id = rows["variate"] == in_dist
    table = {"score": score, "edges": edges, "lower_is_ood": lower_is_ood, "bins": [], "params": list(params),
             "thresholds": list(thresholds)}
    for b in range(len(edges) - 1):
        sel = ok & (which == b)
        row = {"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n_rows": int(sel.sum()),
               "n_cosmologies": int(len(np.unique(rows["sim_id"][sel]))) if sel.any() else 0,
               "n_in_dist_rows": int((sel & is_id).sum()),
               "variate_mix": {v: int((sel & (rows["variate"] == v)).sum()) for v in np.unique(rows["variate"])},
               "per_param": {}}
        for p in params:
            z = rows[f"z_{p}"]
            zz = z[sel]
            fin = np.isfinite(zz)
            zz, sid = zz[fin], rows["sim_id"][sel][fin]
            zid = z[sel & is_id]
            zid = zid[np.isfinite(zid)]
            entry = {"n": int(len(zz))}
            if len(zz) >= 5:
                entry["mean_z"] = float(zz.mean())
                entry["mean_z_ci68"] = _boot_cosmo(zz, sid, np.mean, n_boot=n_boot, seed=seed)
                entry["mean_abs_z"] = float(np.abs(zz).mean())
                entry["mean_abs_z_ci68"] = _boot_cosmo(np.abs(zz), sid, np.mean, n_boot=n_boot, seed=seed)
                for t in thresholds:
                    pr = float(np.mean(np.abs(zz) > t))
                    ci = _boot_cosmo((np.abs(zz) > t).astype(float), sid, np.mean, n_boot=n_boot, seed=seed)
                    ref = float(np.mean(np.abs(zid) > t)) if len(zid) >= 5 else float("nan")
                    entry[f"P_absz_gt_{t:g}"] = {"pooled": pr, "ci68": ci, "in_dist_same_bin": ref,
                                                 "excess": (pr - ref) if np.isfinite(ref) else float("nan")}
            row["per_param"][p] = entry
        table["bins"].append(row)
    # exceedance + inverse reading
    exc = {}
    for p in params:
        z = rows[f"z_{p}"]
        fin = ok & np.isfinite(z)
        cuts = np.nanquantile(s[fin], np.linspace(0.02, 0.98, 25))
        ex_rows = []
        for c in cuts:
            m = fin & ((s <= c) if lower_is_ood else (s >= c))
            if m.sum() < 20:
                continue
            ex_rows.append({"cut": float(c), "n": int(m.sum()),
                            **{f"P_absz_gt_{t:g}": float(np.mean(np.abs(z[m]) > t)) for t in thresholds},
                            "mean_abs_z": float(np.abs(z[m]).mean())})
        inverse = {}
        for t in thresholds:
            for q in (0.5, 0.68, 0.9):
                hit = [r for r in ex_rows if r[f"P_absz_gt_{t:g}"] >= q]
                inverse[f"s_star(t={t:g},q={q})"] = (max(r["cut"] for r in hit) if lower_is_ood else min(r["cut"] for r in hit)) if hit else None
        exc[p] = {"exceedance": ex_rows, "inverse": inverse}
    table["exceedance"] = exc
    return table


def lookup(table: Dict, score_value: float) -> Dict:
    """The row of ``table`` an observed score falls in (the 'compare against the tabulated
    scores' step of the protocol)."""
    edges = table["edges"]
    b = int(np.digitize([score_value], edges)[0] - 1)
    b = min(max(b, 0), len(table["bins"]) - 1)
    return table["bins"][b]


def format_markdown(table: Dict) -> str:
    lines = [f"### detector `{table['score']}` ({'low' if table['lower_is_ood'] else 'high'} = more OOD)", ""]
    hdr = "| bin | n rows (cosmo) | in-dist rows | " + " | ".join(
        f"{p}: E[z], P(>{t:g}) [excess]" for p in table["params"] for t in table["thresholds"]) + " |"
    lines += [hdr, "|" + "---|" * (3 + len(table["params"]) * len(table["thresholds"]))]
    for r in table["bins"]:
        cells = [f"[{r['lo']:.3g}, {r['hi']:.3g})", f"{r['n_rows']} ({r['n_cosmologies']})", str(r["n_in_dist_rows"])]
        for p in table["params"]:
            e = r["per_param"].get(p, {})
            for t in table["thresholds"]:
                k = f"P_absz_gt_{t:g}"
                if k in e:
                    cells.append(f"{e['mean_z']:+.2f}, {e[k]['pooled']:.2f} [{e[k]['excess']:+.2f}]")
                else:
                    cells.append("-")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    for p, ex in table["exceedance"].items():
        inv = {k: (f"{v:.3g}" if v is not None else "n/a") for k, v in ex["inverse"].items()}
        lines.append(f"- **{p}** inverse reading (score cut for P(|z|>t) >= q): " + ", ".join(f"{k}: {v}" for k, v in inv.items()))
    return "\n".join(lines)
