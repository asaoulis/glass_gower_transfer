"""Parameter-subset decomposition of the cross-encoder ensemble-disagreement KL (Tier-2 OOD axis).

The diagonal-Gaussian symmetric KL is a sum over parameters, so the per-event KL splits exactly into
per-parameter and per-subset terms. The per-dimension terms are the same in scaled and physical
space. S_8 is nonlinear, so its moments come from the physical-space sample dumps. AUROC CIs are
cosmology-block bootstraps.
"""
from __future__ import annotations

import os
import re
from typing import Dict, Optional, Sequence, Tuple

import glob

import numpy as np

from .ensemble_discrepancies import diag_gaussian_symmetric_kl
from .ood import auroc

# Subset name -> parameter names (None == every dimension in the pack).
SUBSETS: Dict[str, Optional[Tuple[str, ...]]] = {
    "full": None,
    "om_s8": ("omega_m", "sigma_8"),
    "S8": ("S8",),
    "om_s8_w0": ("omega_m", "sigma_8", "w0"),
}

# Variants that need the per-event posterior covariance from the moments npz, not a diagonal slice.
FULLCOV_SUBSETS: Dict[str, Tuple[str, ...]] = {
    "om_s8_fullcov": ("omega_m", "sigma_8"),
}

# Interpretive blocks of the 9-param vector (used by the per-dimension report).
PARAM_BLOCKS: Dict[str, Tuple[str, ...]] = {
    "cosmology": ("omega_m", "sigma_8", "w0"),
    "weak_nuisance": ("mnu", "h", "ns", "ombh2"),
    "IA": ("a_ia", "b_ia"),
}


def kl_per_dim(mu: np.ndarray, var: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Per-dimension contribution to the mean pairwise symmetric diag-Gaussian KL.

    mu and var are [K, N, D] means and diagonal variances over K members. Returns [N, D], whose sum
    over the last axis is diag_gaussian_symmetric_kl.
    """
    mu = np.asarray(mu, dtype=np.float64)
    var = np.maximum(np.asarray(var, dtype=np.float64), eps)
    k, n, d = mu.shape
    if k < 2:
        return np.zeros((n, d), dtype=np.float64)

    out = np.zeros((n, d), dtype=np.float64)
    count = 0
    for i in range(k):
        for j in range(i + 1, k):
            vi, vj = var[i], var[j]
            dmu2 = (mu[i] - mu[j]) ** 2
            kl_ij = 0.5 * (np.log(vj / vi) + (vi + dmu2) / vj - 1.0)
            kl_ji = 0.5 * (np.log(vi / vj) + (vj + dmu2) / vi - 1.0)
            out += 0.5 * (kl_ij + kl_ji)
            count += 1
    return out / float(count)


def sym_kl_fullcov(mu: np.ndarray, cov: np.ndarray, *, jitter: float = 1e-12) -> np.ndarray:
    """Mean pairwise symmetric KL between full-covariance Gaussians, [N].

    mu [K, N, D], cov [K, N, D, D]. With d = mu_i - mu_j the symmetric KL is

        0.5 * ( tr(S_j^-1 S_i) + tr(S_i^-1 S_j) + d^T (S_i^-1 + S_j^-1) d - 2D ) / 2

    which reduces to diag_gaussian_symmetric_kl for diagonal covariances.
    """
    mu = np.asarray(mu, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    k, n, d = mu.shape
    if k < 2:
        return np.zeros(n, dtype=np.float64)
    eye = np.eye(d)[None, None, :, :] * jitter
    inv = np.linalg.inv(cov + eye)

    out = np.zeros(n, dtype=np.float64)
    count = 0
    for i in range(k):
        for j in range(i + 1, k):
            dmu = mu[i] - mu[j]
            tr = (np.einsum("nab,nba->n", inv[j], cov[i])
                  + np.einsum("nab,nba->n", inv[i], cov[j]))
            quad = (np.einsum("na,nab,nb->n", dmu, inv[j], dmu)
                    + np.einsum("na,nab,nb->n", dmu, inv[i], dmu))
            out += 0.5 * (tr + quad - 2.0 * d) * 0.5
            count += 1
    return out / float(count)


def fullcov_moments(moments_path: str, params: Sequence[str], want: Sequence[str]):
    """(mean[N,k], cov[N,k,k], basenames) for the `want` parameters of one repeat's moments npz."""
    idx = [list(params).index(p) for p in want]
    with np.load(moments_path, allow_pickle=False) as d:
        mean = np.asarray(d["mean"], dtype=np.float64)[:, idx]
        cov = np.asarray(d["cov"], dtype=np.float64)[:, idx][:, :, idx]
        files = [os.path.basename(str(x)) for x in d["test_files"]]
    return mean, cov, files


def self_check(mu: np.ndarray, var: np.ndarray, kl_reference: np.ndarray, *, rtol: float = 1e-5) -> float:
    """Assert the decomposition reproduces the stored kl_score. Returns the max relative error."""
    got = kl_per_dim(mu, var).sum(axis=-1)
    ref = np.asarray(kl_reference, dtype=np.float64)
    denom = np.maximum(np.abs(ref), 1e-12)
    err = float(np.max(np.abs(got - ref) / denom))
    if not err <= rtol:
        raise AssertionError(
            "kl_per_dim(...).sum(-1) does not reproduce the stored kl_score: max rel err %.3e" % err)
    return err


# --- S_8, the one subset that is not a slice ---

def s8_from_scaled(samples: np.ndarray, names: Sequence[str], lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """S_8 = sigma_8 sqrt(omega_m / 0.3) from scaled samples [S, N, D] -> [S, N]."""
    names = list(names)
    iom, is8 = names.index("omega_m"), names.index("sigma_8")
    om = np.asarray(samples[..., iom], dtype=np.float64) * (hi[iom] - lo[iom]) + lo[iom]
    s8 = np.asarray(samples[..., is8], dtype=np.float64) * (hi[is8] - lo[is8]) + lo[is8]
    return s8 * np.sqrt(om / 0.3)


def s8_moments(samples_path: str, names: Sequence[str], lo: np.ndarray, hi: np.ndarray,
               *, max_events: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, list]:
    """Per-event S_8 mean and variance for one repeat's sample dump.

    Returns (mu[N], var[N], test_file_basenames) in physical S_8 units.
    """
    with np.load(samples_path, mmap_mode="r") as d:
        S = d["samples"]                                   # [S, N, D] scaled
        files = [os.path.basename(str(x)) for x in d["test_files"]]
        n = S.shape[1] if max_events is None else min(S.shape[1], int(max_events))
        S8 = s8_from_scaled(np.asarray(S[:, :n, :]), names, lo, hi)   # [S, n]
    return S8.mean(axis=0), S8.var(axis=0), files[:n]


# --- discriminating power ---

def sim_ids_from_files(files: Sequence[str]) -> np.ndarray:
    """Cosmology id per row, parsed from `output_<sim_id>_...h5` basenames."""
    out = []
    for f in files:
        m = re.search(r"output_(\d+)_", os.path.basename(str(f)))
        out.append(int(m.group(1)) if m else -1)
    return np.asarray(out, dtype=np.int64)


def auroc_with_block_ci(score_id: np.ndarray, sim_id: np.ndarray,
                        score_ood: np.ndarray, sim_ood: np.ndarray,
                        *, n_boot: int = 500, seed: int = 0,
                        quantiles: Tuple[float, float] = (0.16, 0.84)):
    """AUROC(query=OOD) with a cosmology-block bootstrap CI.

    Rows are correlated augmentations of one cosmology, so both pools are resampled per cosmology.
    Returns dict(auroc, lo, hi, n_id, n_ood, n_cosmo_id, n_cosmo_ood).
    """
    score_id = np.asarray(score_id, dtype=np.float64)
    score_ood = np.asarray(score_ood, dtype=np.float64)
    sim_id = np.asarray(sim_id)
    sim_ood = np.asarray(sim_ood)

    ok_i = np.isfinite(score_id)
    ok_o = np.isfinite(score_ood)
    score_id, sim_id = score_id[ok_i], sim_id[ok_i]
    score_ood, sim_ood = score_ood[ok_o], sim_ood[ok_o]

    res = {"n_id": int(score_id.size), "n_ood": int(score_ood.size),
           "n_cosmo_id": int(np.unique(sim_id).size), "n_cosmo_ood": int(np.unique(sim_ood).size)}
    if score_id.size == 0 or score_ood.size == 0:
        res.update(auroc=float("nan"), lo=float("nan"), hi=float("nan"))
        return res

    res["auroc"] = float(auroc(score_id, score_ood))

    rng = np.random.default_rng(seed)
    sims_i, sims_o = np.unique(sim_id), np.unique(sim_ood)
    by_i = {s: np.where(sim_id == s)[0] for s in sims_i}
    by_o = {s: np.where(sim_ood == s)[0] for s in sims_o}
    draws = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        pi = np.concatenate([by_i[s] for s in rng.choice(sims_i, size=sims_i.size, replace=True)])
        po = np.concatenate([by_o[s] for s in rng.choice(sims_o, size=sims_o.size, replace=True)])
        draws[b] = auroc(score_id[pi], score_ood[po])
    res["lo"] = float(np.quantile(draws, quantiles[0]))
    res["hi"] = float(np.quantile(draws, quantiles[1]))
    return res


def auroc_rowlevel_ci(score_id: np.ndarray, score_ood: np.ndarray, *, n_boot: int = 500, seed: int = 0,
                      quantiles: Tuple[float, float] = (0.16, 0.84)):
    """Row-level bootstrap CI."""
    score_id = np.asarray(score_id, dtype=np.float64)
    score_ood = np.asarray(score_ood, dtype=np.float64)
    score_id = score_id[np.isfinite(score_id)]
    score_ood = score_ood[np.isfinite(score_ood)]
    if score_id.size == 0 or score_ood.size == 0:
        return {"auroc": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    draws = np.array([
        auroc(rng.choice(score_id, score_id.size, replace=True),
              rng.choice(score_ood, score_ood.size, replace=True))
        for _ in range(int(n_boot))])
    return {"auroc": float(auroc(score_id, score_ood)),
            "lo": float(np.quantile(draws, quantiles[0])),
            "hi": float(np.quantile(draws, quantiles[1]))}


# --- loading one variate's scores from a misspec pack directory ---

ALL_SCORES = list(SUBSETS) + list(FULLCOV_SUBSETS)


def disagreement_npz(mdir: str, n_repeats: int):
    """The disagreement file built from all n_repeats members, as (path, k), or None."""
    best = None
    for f in sorted(glob.glob(os.path.join(mdir, "misspec_repeat_disagreement_*.npz"))):
        with np.load(f, mmap_mode="r") as d:
            k = int(d["mu"].shape[0])
        if k >= n_repeats and (best is None or k > best[1]):
            best = (f, k)
    return best


def params_of_pack(mdir: str, match_template: str, repeats):
    """Parameter names, read from a moments npz because the 2-pt pack may differ."""
    for r in repeats:
        p = os.path.join(mdir, "misspec_posterior_moments_%s.npz" % match_template.format(r=r))
        if os.path.exists(p):
            with np.load(p, allow_pickle=False) as d:
                return [str(x) for x in d["params"]]
    raise FileNotFoundError("no moments npz under %s for template %r" % (mdir, match_template))


def load_variate(base_dir: str, variate: str, match_template: str, repeats, names, lo, hi,
                 *, max_events_s8=None, want_s8=True):
    """-> dict(files, sim_ids, per_dim [N,D], kl_full [N], kl_S8 [N]) for one variate."""
    mdir = os.path.join(base_dir, "misspec", variate)
    found = disagreement_npz(mdir, len(repeats))
    if found is None:
        raise FileNotFoundError("no %d-repeat disagreement npz under %s" % (len(repeats), mdir))
    path, _k = found
    with np.load(path, allow_pickle=False) as d:
        mu, var = np.asarray(d["mu"]), np.asarray(d["var"])
        files = [os.path.basename(str(x)) for x in d["test_files"]]
        kl_stored = np.asarray(d["kl_score"], dtype=np.float64)

    rel = self_check(mu, var, kl_stored)
    per_dim = kl_per_dim(mu, var)                 # [N, D]

    kl_s8 = np.full(len(files), np.nan)
    if want_s8:
        mus, vars_, common = [], [], None
        for r in repeats:
            sp = os.path.join(mdir, "misspec_posterior_samples_%s.npz" % match_template.format(r=r))
            if not os.path.exists(sp):
                mus = []
                break
            m8, v8, sf = s8_moments(sp, names, lo, hi, max_events=max_events_s8)
            idx = {f: i for i, f in enumerate(sf)}
            mus.append((m8, v8, idx))
            common = set(idx) if common is None else (common & set(idx))
        if mus and common:
            order = [f for f in files if f in common]
            sel = np.array([files.index(f) for f in order])
            M = np.stack([m8[[idx[f] for f in order]] for m8, _v, idx in mus])[:, :, None]
            V = np.stack([v8[[idx[f] for f in order]] for _m, v8, idx in mus])[:, :, None]
            kl_s8[sel] = diag_gaussian_symmetric_kl(M, V)

    fullcov = {}
    for sub, want in FULLCOV_SUBSETS.items():
        vals = np.full(len(files), np.nan)
        if all(p in names for p in want):
            mus, common = [], None
            for r in repeats:
                mp = os.path.join(mdir, "misspec_posterior_moments_%s.npz" % match_template.format(r=r))
                if not os.path.exists(mp):
                    mus = []
                    break
                mean, cov, mf = fullcov_moments(mp, names, want)
                idx = {f: i for i, f in enumerate(mf)}
                mus.append((mean, cov, idx))
                common = set(idx) if common is None else (common & set(idx))
            if mus and common:
                order = [f for f in files if f in common]
                sel = np.array([files.index(f) for f in order])
                M = np.stack([mean[[idx[f] for f in order]] for mean, _c, idx in mus])
                C = np.stack([cov[[idx[f] for f in order]] for _m, cov, idx in mus])
                vals[sel] = sym_kl_fullcov(M, C)
        fullcov[sub] = vals

    return {"variate": variate, "files": np.array(files), "fullcov": fullcov, "sim_ids": sim_ids_from_files(files),
            "per_dim": per_dim, "kl_full": per_dim.sum(axis=-1), "kl_S8": kl_s8,
            "kl_stored": kl_stored, "selfcheck_rel_err": rel, "source": os.path.basename(path)}


def subset_scores(rec, names):
    """Every named subset's per-event score for one variate."""
    out = dict(rec.get("fullcov", {}))
    for sub, params in SUBSETS.items():
        if sub == "S8":
            out[sub] = rec["kl_S8"]
        elif params is None:
            out[sub] = rec["kl_full"]
        else:
            idx = [names.index(p) for p in params if p in names]
            if len(idx) != len(params):
                out[sub] = np.full(rec["kl_full"].shape, np.nan)
            else:
                out[sub] = rec["per_dim"][:, idx].sum(axis=-1)
    return out


# --- non-parametric discrepancies estimated from the samples ---

def knn_kl(x: np.ndarray, y: np.ndarray, *, k: int = 1, eps: float = 1e-12) -> float:
    """Wang-Kulkarni-Verdu k-NN estimator of KL(p_x || p_y) from samples.

        KL = (d/n) sum_i log(nu_i / rho_i) + log(m / (n - 1))

    with rho_i the distance from x_i to its k-th nearest neighbour in X (excluding itself) and nu_i
    the distance to its k-th nearest in Y.
    """
    from scipy.spatial import cKDTree
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = x.shape
    m = y.shape[0]
    if n < k + 2 or m < k + 1:
        return float("nan")
    rho = cKDTree(x).query(x, k=k + 1)[0][:, k]
    nu = cKDTree(y).query(x, k=k)[0]
    if nu.ndim > 1:
        nu = nu[:, k - 1]
    ok = (rho > eps) & (nu > eps)
    if ok.sum() < 8:
        return float("nan")
    return float((d / ok.sum()) * np.sum(np.log(nu[ok] / rho[ok])) + np.log(m / (n - 1.0)))


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Energy distance 2 E|X-Y| - E|X-X'| - E|Y-Y'|.

    Not invariant under per-dimension rescaling, so compare it only within one scaling convention.
    """
    from scipy.spatial.distance import cdist
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(2.0 * cdist(x, y).mean() - cdist(x, x).mean() - cdist(y, y).mean())
