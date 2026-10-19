"""Shared loaders + cluster-aware statistics for the VDQ misspec tables/figures.

Rows of every Gower table are ~8 (gb slices: 2) correlated augmentations per cosmology, so every
interval here is a COSMOLOGY bootstrap (resample sim_ids, take all their rows on both sides;
paired when the two sides are re-simulations of the same cosmologies), never a row bootstrap
(memory cluster-aware-stats-gower-mocks).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vdq_config as C  # noqa: E402

PARAMS9 = ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"]
KEY3 = ["omega_m", "sigma_8", "w0"]
SUBSET_KEY = "sigma_8__omega_m__w0"          # misspec json TARP subset key for (Om, s8, w)
THRESH = (0.3, 0.5, 1.0)

# NLA-M effective-amplitude conversion (gower_aia_vs_kl.py; src/cosmology/nla.py:kappa_ia_nla_m,
# f_red from src/KiDS/systematics.py, per-bin halo masses kids-legacy-sbi massdep_means.txt)
F_RED = np.array([0.15, 0.20, 0.17, 0.24, 0.19, 0.03])
LOG10_M = np.array([11.69224599, 12.45768451, 12.76280679, 12.93435839, 13.08446641, 13.21675774])
LOG10_M_PIVOT = 13.5


def nla_m_effective(a_ia, b_ia):
    """Raw NLA-M (a_ia, b_ia) -> A_IA^total, averaged over the six tomographic bins."""
    a = np.asarray(a_ia, dtype=np.float64)[:, None]
    b = np.asarray(b_ia, dtype=np.float64)[:, None]
    return (a * F_RED[None, :] * 10.0 ** ((LOG10_M[None, :] - LOG10_M_PIVOT) * b)).mean(axis=1)


def b(x):
    return os.path.basename(str(x))


def load_npz(p):
    with np.load(p, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def load_json(p):
    with open(p) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------------------------
# per-pack file access
# ---------------------------------------------------------------------------------------------
def summ(pk, v, j):
    """Member-0 summaries npz of variate v for the pack's j-th repeat (X-1 resolution)."""
    from src.observation.cluster_modes import _representative_summaries
    f = _representative_summaries(os.path.join(pk["summaries"], v), pk["summ_matches"][j])
    if f is None:
        raise FileNotFoundError(f"no summaries {v} {pk['summ_matches'][j]}")
    return load_npz(f)


def moments(pk, v, j):
    return load_npz(os.path.join(pk["misspec"], v, f"misspec_posterior_moments_{pk['matches'][j]}.npz"))


def samples_path(pk, v, j):
    return os.path.join(pk["misspec"], v, f"misspec_posterior_samples_{pk['matches'][j]}.npz")


def results(pk, v, j):
    return load_json(os.path.join(pk["misspec"], v, f"misspec_evaluation_results_{pk['matches'][j]}.json"))


def tarp_ci(pk, v, j):
    return load_json(os.path.join(pk["misspec"], v, f"misspec_tarp_credible_intervals_{pk['matches'][j]}.json"))


def disagreement(pk, v):
    """The ALL-repeat cross-repeat disagreement npz (kl_score = FULL diagonal KL)."""
    n = len(pk["repeats"])
    for f in sorted(glob.glob(os.path.join(pk["misspec"], v, "misspec_repeat_disagreement_*.npz"))):
        d = load_npz(f)
        if d["mu"].shape[0] == n:
            d["files"] = np.array([b(x) for x in d["test_files"]])
            return d
    raise FileNotFoundError(f"no {n}-repeat disagreement npz for {v}")


def kl_event(pk, v, which):
    """Per-event cross-repeat KL: (files, sim_ids, values). which = 'full' or a kl_subsets key."""
    names, lo, hi = C.scaler_box(pk["exp"])
    if which == "full":
        d = disagreement(pk, v)
        files, vals = d["files"], np.asarray(d["kl_score"], float)
    else:
        from src.ml.eval.kl_subsets import load_variate, subset_scores
        rec = load_variate(pk["base"], v, pk["match"], pk["repeats"], names, lo, hi, want_s8=False)
        files, vals = np.array([b(x) for x in rec["files"]]), np.asarray(subset_scores(rec, names)[which], float)
    sims = np.array([int(str(f).split("_")[1]) for f in files])
    return files, sims, vals


def rows(pk, variates=None, rebuild=False):
    """Tier-2 rows (variate, repeat, file, sim_id, z_*, knn_*, kl[adopted]) for the pack, cached
    at PACKS[pk]['rows_cache'] (the SAME cache scripts/tier2_tables.py reads), with the
    obs-identical `meanp` (meanp_knn.py) joined event-level."""
    variates = list(variates or C.present(pk))
    cache = pk["rows_cache"]
    r = None
    if os.path.exists(cache) and not rebuild:
        with np.load(cache, allow_pickle=False) as d:
            r = {("file" if k == "files" else k): d[k] for k in d.files}
        if not set(variates) <= set(np.unique(r["variate"]).tolist()):
            r = None
    if r is None:
        from src.observation.checks.tier2 import load_rows
        r = load_rows(pk["base"], experiment=pk["exp"], variates=variates, repeats=pk["repeats"],
                      phase6_dir=None, match_template=pk["match"], kl_params=pk["kl"])
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, **{("files" if k == "file" else k): v for k, v in r.items()})
    keep = np.isin(r["variate"], variates)
    r = {k: v[keep] for k, v in r.items()}
    if os.path.exists(pk["meanp_npz"]):
        d = load_npz(pk["meanp_npz"])
        key = {(str(v), b(f)): float(x) for v, f, x in zip(d["variate"], d["test_files"], d["meanp_knn_recal"])}
        r["meanp"] = np.array([key.get((str(v), str(f)), np.nan) for v, f in zip(r["variate"], r["file"])])
    return r


# ---------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------
def auroc(s_null, s_q):
    """P(score_q > score_null) (+0.5 ties); higher score = more OOD. NaNs dropped."""
    s_null = np.asarray(s_null, float)
    s_q = np.asarray(s_q, float)
    s_null, s_q = s_null[np.isfinite(s_null)], s_q[np.isfinite(s_q)]
    if not (s_null.size and s_q.size):
        return np.nan
    rk = rankdata(np.concatenate([s_null, s_q]))
    n0, n1 = s_null.size, s_q.size
    return float((rk[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def cosmo_draws(sim_null, sim_q, n_boot, rng):
    """Paired cosmology bootstrap index sets: resample the null's cosmologies, take all their rows
    on both sides (the variates are re-simulations of the in-dist test cosmologies)."""
    uniq = np.unique(sim_null)
    idx_n = {g: np.flatnonzero(sim_null == g) for g in uniq}
    idx_q = {g: np.flatnonzero(sim_q == g) for g in uniq}
    out = []
    for _ in range(n_boot):
        dr = rng.choice(uniq, len(uniq), replace=True)
        rn = np.concatenate([idx_n[g] for g in dr])
        rq = [idx_q[g] for g in dr if idx_q[g].size]
        out.append((rn, np.concatenate(rq) if rq else np.array([], int)))
    return out


def cluster_ci(values, sims, stat=np.mean, n_boot=1000, seed=0, q=(2.5, 97.5)):
    """Cosmology-bootstrap percentile interval of stat(values)."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    uniq = np.unique(sims)
    idx = {g: np.flatnonzero(sims == g) for g in uniq}
    est = []
    for _ in range(n_boot):
        dr = rng.choice(uniq, len(uniq), replace=True)
        est.append(stat(values[np.concatenate([idx[g] for g in dr])]))
    return [float(np.nanpercentile(est, q[0])), float(np.nanpercentile(est, q[1]))]


def matched_draw(sim_null, n_cos, rpc, rng):
    """Row indices of a cluster-matched draw from the null: n_cos cosmologies x rpc rows each."""
    uniq = np.unique(sim_null)
    pick = rng.choice(uniq, size=min(n_cos, len(uniq)), replace=False)
    sel = []
    for g in pick:
        ii = np.flatnonzero(sim_null == g)
        sel.append(rng.choice(ii, size=min(rpc, ii.size), replace=False))
    return np.concatenate(sel)


def shape_of(sims):
    n_cos = len(np.unique(sims))
    return n_cos, int(round(len(sims) / max(n_cos, 1)))


def fmt_ci(x, ci, d=3):
    return f"{x:.{d}f} [{ci[0]:.{d}f}, {ci[1]:.{d}f}]"


def obs(pk):
    return load_json(pk["obs_json"])


def out_dir(*parts):
    p = os.path.join(C.OUT_MISSPEC, *parts)
    os.makedirs(p, exist_ok=True)
    return p
