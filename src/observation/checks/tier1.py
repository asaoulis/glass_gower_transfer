"""Tier-1 pre-model data checks of ONE observation against the nla_m mock cloud.

  1a  two-point in-distribution   whitened EE bandpowers: Mahalanobis + kNN conformal p-values
                                  against a held-out half of the cloud; per-(spectrum, band)
                                  robust z-scores; per-spectrum chi^2 PTEs.
  1b  B-mode null                 chi^2 of the BB bandpowers against the mock-BB mean/covariance
                                  (PTE from the EMPIRICAL null of the same statistic on held-out
                                  mocks, plus the analytic chi^2 PTE for reference), the same
                                  statistic against ZERO, and per-tomo-auto PTEs.
  1c  pre-CNN E-map OOD           whitened hand-crafted map summaries: kNN conformal p (primary,
                                  Sun+22 / Bates+23), Mahalanobis p (global cross-check), a PCA-k
                                  kNN variant, and per-group kNN p-values.

Null calibration: the cloud is split by COSMOLOGY (sim_id) into a FIT half (mean/covariance/
whitener/kNN reference) and a NULL half (empirical p-values). Rows of the null half are ~8-80
correlated augmentations per cosmology; that reduces the effective size of the null (p-value
resolution), not the validity of the marginal p for a NEW observation drawn from the same
mixture. A mock-as-real observation must have its own sim_id EXCLUDED from the cloud (see
``exclude_sim_ids``), otherwise its augmentation siblings make it look in-distribution.

Caveat stated up front: every statistic here is MARGINAL over the prior (the cloud spans it while
the observation sits at one unknown cosmology), so a pass is a weak statement and a fail a strong
one. The conditional version (cond_knn) needs theta the observation does not have.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.ml.eval.ood import TrainWhitener, empirical_pvalues, knn_scores, mahalanobis_scores
from .summaries import GROUPS, group_mask

N_SPECTRA = 21


def spectrum_labels(nbins: int = 6) -> List[str]:
    labs = []
    for i in range(nbins):
        for j in range(nbins):
            if i >= j:
                labs.append(f"{i + 1}-{j + 1}")
    return labs


@dataclass
class Cloud:
    sim_ids: np.ndarray
    aug_ids: np.ndarray
    bandpowers: Optional[np.ndarray] = None     # [N,21,B]
    bb: Optional[np.ndarray] = None             # [N,21,B]
    emap: Optional[np.ndarray] = None           # [N,D]
    emap_names: Optional[np.ndarray] = None
    files: Optional[np.ndarray] = None

    @classmethod
    def from_npz(cls, baked_npz: Optional[str] = None, raw_npz: Optional[str] = None) -> "Cloud":
        """bandpowers + emap from the baked-store npz; bb (and, if absent above, bandpowers/emap)
        from the raw-store npz. The two need not cover the same files (bb has its own cloud)."""
        from ..reference import load_reference
        base = load_reference(baked_npz) if baked_npz else load_reference(raw_npz)
        c = cls(sim_ids=base["sim_ids"], aug_ids=base["aug_ids"], files=base.get("files"),
                bandpowers=base.get("bandpowers"), emap=base.get("emap"), emap_names=base.get("emap_names"))
        if raw_npz:
            raw = load_reference(raw_npz)
            c.bb = raw.get("bb")
            c.bb_sim_ids = raw["sim_ids"]
            if c.bandpowers is None:
                c.bandpowers = raw.get("bandpowers")
            if c.emap is None:
                c.emap = raw.get("emap")
                c.emap_names = raw.get("emap_names")
        else:
            c.bb_sim_ids = c.sim_ids
        return c


def cosmology_split(sim_ids: np.ndarray, frac_fit: float = 0.5, seed: int = 0,
                    exclude: Sequence[int] = ()) -> tuple:
    """Index arrays (fit, null) split by unique sim_id; ``exclude`` sim_ids are dropped from both."""
    rng = np.random.default_rng(seed)
    sims = np.array(sorted(set(sim_ids.tolist()) - set(int(s) for s in exclude)))
    rng.shuffle(sims)
    n_fit = max(1, int(round(frac_fit * len(sims))))
    fit_sims = set(sims[:n_fit].tolist())
    null_sims = set(sims[n_fit:].tolist())
    fit = np.array([i for i, s in enumerate(sim_ids) if int(s) in fit_sims])
    null = np.array([i for i, s in enumerate(sim_ids) if int(s) in null_sims])
    return fit, null


def _robust_z(x_obs: np.ndarray, x_cloud: np.ndarray) -> np.ndarray:
    med = np.median(x_cloud, axis=0)
    mad = np.median(np.abs(x_cloud - med), axis=0) * 1.4826
    mad = np.where(mad > 0, mad, np.std(x_cloud, axis=0) + 1e-30)
    return (x_obs - med) / mad


def _shrink_whitener(x_fit: np.ndarray, shrinkage: Optional[float] = None) -> TrainWhitener:
    """TrainWhitener on a (Ledoit-Wolf-)shrunk covariance when d is not << n."""
    n, d = x_fit.shape
    if shrinkage is None:
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(x_fit)
            cov = lw.covariance_
        except Exception:
            cov = np.cov(x_fit, rowvar=False)
    else:
        cov = np.cov(x_fit, rowvar=False)
        cov = (1 - shrinkage) * cov + shrinkage * np.eye(d) * np.trace(cov) / d
    cov = np.atleast_2d(cov) + 1e-10 * np.eye(d)
    L = np.linalg.cholesky(cov)
    return TrainWhitener(mean=x_fit.mean(axis=0), chol_inv=np.linalg.inv(L), eps=1e-10)


def _score_block(x_fit: np.ndarray, x_null: np.ndarray, x_obs: np.ndarray, *, k: int = 10,
                 pca: Optional[int] = None) -> Dict:
    """Whiten on fit, score null + obs (Mahalanobis, kNN), return per-obs scores + empirical p.

    Columns are first put in fit-half scatter units: the raw bandpowers are ~1e-9, so any absolute
    ridge in the whitener would otherwise dominate their covariance and the "whitened" statistic
    would silently degrade to a Euclidean one weighted by the largest-amplitude bands (found on
    the first production report, 2026-09-08). Scale-free inputs (the MAD-standardised map
    summaries) are unaffected."""
    sd = np.asarray(x_fit, dtype=np.float64).std(0)
    sd = np.where(sd > 0, sd, 1.0)
    x_fit, x_null, x_obs = (np.asarray(a, dtype=np.float64) / sd for a in (x_fit, x_null, x_obs))
    w = _shrink_whitener(x_fit)
    zf, zn, zo = w(x_fit), w(x_null), w(x_obs)
    if pca:
        # PCA of the whitened fit cloud (directions of largest residual variance after whitening are
        # ~isotropic; PCA here just truncates to the leading k subspace for a lower-d kNN)
        u, s, vt = np.linalg.svd(zf - zf.mean(0), full_matrices=False)
        P = vt[:pca].T
        zf, zn, zo = zf @ P, zn @ P, zo @ P
    out = {}
    m_null, m_obs = mahalanobis_scores(zn), mahalanobis_scores(zo)
    kn_null = knn_scores(zf, zn, k=k)
    kn_obs = knn_scores(zf, zo, k=k)
    out["mahalanobis"] = {"obs": m_obs.tolist(), "p": empirical_pvalues(m_null, m_obs).tolist(),
                          "null_median": float(np.median(m_null)), "null_p95": float(np.quantile(m_null, 0.95))}
    out["knn"] = {"obs": kn_obs.tolist(), "p": empirical_pvalues(kn_null, kn_obs).tolist(),
                  "null_median": float(np.median(kn_null)), "null_p95": float(np.quantile(kn_null, 0.95)),
                  "k": k, "null_scores": kn_null.tolist()}
    out["n_fit"], out["n_null"], out["dim"] = int(len(zf)), int(len(zn)), int(zf.shape[1])
    return out


# ----------------------------------------------------------------------------------------------
# 1a  two-point
# ----------------------------------------------------------------------------------------------
def twopoint_check(cloud: Cloud, obs_bp: np.ndarray, *, seed: int = 0, exclude_sim_ids: Sequence[int] = (),
                   k: int = 10) -> Dict:
    """``obs_bp``: [M,21,B] (M labels)."""
    X = cloud.bandpowers.reshape(len(cloud.bandpowers), -1)
    O = np.asarray(obs_bp).reshape(len(obs_bp), -1)
    fit, null = cosmology_split(cloud.sim_ids, seed=seed, exclude=exclude_sim_ids)
    res = _score_block(X[fit], X[null], O, k=k)
    # per-entry robust z (vs the whole cloud) and per-spectrum chi^2 PTE from the fit covariance,
    # with the null-half empirical distribution of the same per-spectrum statistic
    nb = cloud.bandpowers.shape[-1]
    z = _robust_z(O, X).reshape(len(O), N_SPECTRA, nb)
    per_spec = []
    labs = spectrum_labels()
    for s in range(N_SPECTRA):
        sd_s = cloud.bandpowers[fit, s, :].std(0) + 1e-30          # scatter units (see _score_block)
        xs_fit = cloud.bandpowers[fit, s, :] / sd_s
        xs_null = cloud.bandpowers[null, s, :] / sd_s
        xo = np.asarray(obs_bp)[:, s, :] / sd_s
        mu = xs_fit.mean(0)
        cov = np.cov(xs_fit, rowvar=False) + 1e-8 * np.eye(nb)
        ci = np.linalg.inv(cov)
        chi2_null = np.einsum("ni,ij,nj->n", xs_null - mu, ci, xs_null - mu)
        chi2_obs = np.einsum("ni,ij,nj->n", xo - mu, ci, xo - mu)
        per_spec.append({"spectrum": labs[s], "chi2": chi2_obs.tolist(),
                         "pte_empirical": empirical_pvalues(chi2_null, chi2_obs).tolist(),
                         "null_chi2_median": float(np.median(chi2_null))})
    res.update({"robust_z": z.tolist(), "per_spectrum": per_spec, "spectrum_labels": labs,
                "n_cosmologies_fit": int(len(set(cloud.sim_ids[fit].tolist()))),
                "n_cosmologies_null": int(len(set(cloud.sim_ids[null].tolist())))})
    return res


# ----------------------------------------------------------------------------------------------
# 1b  B-mode null
# ----------------------------------------------------------------------------------------------
def bmode_check(cloud: Cloud, obs_bb: np.ndarray, *, seed: int = 0, exclude_sim_ids: Sequence[int] = ()) -> Dict:
    """``obs_bb``: [M,21,B]. chi^2 against the mock-BB mean/cov (empirical + analytic PTE), the
    same statistic against zero, per-tomo-auto PTEs, and per-entry robust z."""
    from scipy.stats import chi2 as _chi2
    B = cloud.bb.reshape(len(cloud.bb), -1)
    O = np.asarray(obs_bb).reshape(len(obs_bb), -1)
    sim_ids = getattr(cloud, "bb_sim_ids", cloud.sim_ids)
    fit, null = cosmology_split(sim_ids, seed=seed, exclude=exclude_sim_ids)
    d = B.shape[1]
    n_fit_rows = len(fit)
    n_fit_cosmo = len(set(sim_ids[fit].tolist()))
    # Work in SCATTER-NORMALISED units. BB bandpowers are ~1e-9, their covariance ~1e-18, and an
    # absolute ridge (the 1e-12*I first used here) swamped it -- the "chi^2" then read 3-5 for
    # 168 dof (found on the T report, 2026-09-08). Dividing every entry by its fit-half scatter
    # makes the ridge negligible (1e-8 on a unit diagonal) and the statistic the chi^2 it claims.
    sd = B[fit].std(0) + 1e-30
    B = B / sd
    O = O / sd
    mu = B[fit].mean(0)
    cov = np.cov(B[fit], rowvar=False) + 1e-8 * np.eye(d)
    ci = np.linalg.inv(cov)
    # Hartlap factor with the CONSERVATIVE count (cosmologies, not rows) -- reported, and applied
    # only to the analytic PTE; the empirical PTE needs no correction.
    hartlap = (n_fit_cosmo - d - 2) / (n_fit_cosmo - 1) if n_fit_cosmo > d + 2 else float("nan")

    def chi2_of(x, centre):
        r = x - centre
        return np.einsum("ni,ij,nj->n", r, ci, r)

    chi2_null = chi2_of(B[null], mu)
    chi2_obs = chi2_of(O, mu)
    chi2_null0 = chi2_of(B[null], 0.0)
    chi2_obs0 = chi2_of(O, 0.0)
    out = {
        "dim": int(d), "n_fit_rows": int(n_fit_rows), "n_fit_cosmologies": int(n_fit_cosmo),
        "n_null_rows": int(len(null)), "hartlap_cosmo": hartlap,
        "chi2_vs_mockmean": chi2_obs.tolist(),
        "pte_empirical": empirical_pvalues(chi2_null, chi2_obs).tolist(),
        "pte_analytic_hartlap": (_chi2.sf(chi2_obs * hartlap, d).tolist() if np.isfinite(hartlap) else None),
        "null_chi2_median": float(np.median(chi2_null)), "null_chi2_p95": float(np.quantile(chi2_null, 0.95)),
        "chi2_vs_zero": chi2_obs0.tolist(),
        "pte_vs_zero_empirical": empirical_pvalues(chi2_null0, chi2_obs0).tolist(),
        "mock_bb_mean_over_scatter": float(np.mean(np.abs(mu) / (np.sqrt(np.diag(cov)) + 1e-30))),
        "robust_z": _robust_z(O, B).reshape(len(O), N_SPECTRA, -1).tolist(),
        "spectrum_labels": spectrum_labels(),
    }
    # per tomo auto-spectrum (i-i) PTEs over the bands
    labs = spectrum_labels()
    autos = [s for s, l in enumerate(labs) if l.split("-")[0] == l.split("-")[1]]
    nb = cloud.bb.shape[-1]
    per_auto = []
    Bs = B.reshape(len(B), N_SPECTRA, -1)            # scatter-normalised, as above
    Os = O.reshape(len(O), N_SPECTRA, -1)
    for s in autos:
        xs_fit = Bs[fit, s, :]
        xs_null = Bs[null, s, :]
        m = xs_fit.mean(0)
        c = np.cov(xs_fit, rowvar=False) + 1e-8 * np.eye(nb)
        cinv = np.linalg.inv(c)
        cn = np.einsum("ni,ij,nj->n", xs_null - m, cinv, xs_null - m)
        co = np.einsum("ni,ij,nj->n", Os[:, s, :] - m, cinv, Os[:, s, :] - m)
        per_auto.append({"spectrum": labs[s], "chi2": co.tolist(), "pte_empirical": empirical_pvalues(cn, co).tolist()})
    out["per_auto"] = per_auto
    return out


# ----------------------------------------------------------------------------------------------
# 1c  pre-CNN E-map OOD
# ----------------------------------------------------------------------------------------------
def emap_check(cloud: Cloud, obs_emap: np.ndarray, *, seed: int = 0, exclude_sim_ids: Sequence[int] = (),
               k: int = 10, pca: int = 30, groups: Sequence[str] = GROUPS) -> Dict:
    """``obs_emap``: [M,D]. Robust-standardise on the cloud, then whitened kNN / Mahalanobis
    (full-d and PCA-k) plus per-group kNN p-values."""
    E = np.asarray(cloud.emap, dtype=np.float64)
    O = np.asarray(obs_emap, dtype=np.float64)
    names = [str(n) for n in cloud.emap_names]
    finite = np.isfinite(E).all(0) & np.isfinite(O).all(0)
    # drop constant / non-finite columns
    sd = E[:, finite].std(0)
    keep = np.where(finite)[0][sd > 0]
    E, O = E[:, keep], O[:, keep]
    names_k = [names[i] for i in keep]
    med = np.median(E, 0)
    mad = np.median(np.abs(E - med), 0) * 1.4826
    mad = np.where(mad > 0, mad, E.std(0))
    Es, Os = (E - med) / mad, (O - med) / mad
    fit, null = cosmology_split(cloud.sim_ids, seed=seed, exclude=exclude_sim_ids)
    out = {"full": _score_block(Es[fit], Es[null], Os, k=k),
           f"pca{pca}": _score_block(Es[fit], Es[null], Os, k=k, pca=min(pca, Es.shape[1])),
           "groups": {}, "n_dims_used": int(len(keep)),
           "n_cosmologies_fit": int(len(set(cloud.sim_ids[fit].tolist()))),
           "n_cosmologies_null": int(len(set(cloud.sim_ids[null].tolist())))}
    for g in groups:
        m = group_mask(names_k, [g])
        if m.sum() >= 2:
            out["groups"][g] = _score_block(Es[fit][:, m], Es[null][:, m], Os[:, m], k=k)
            out["groups"][g]["dim"] = int(m.sum())
    out["robust_z"] = dict(zip(names_k, np.asarray(Os).T.tolist()))
    return out


def power_check(cloud_id: Cloud, cloud_ood: Cloud, *, which: str = "emap", seed: int = 0, k: int = 10,
                pca: int = 30, max_ood: int = 2000) -> Dict:
    """AUROC of each Tier-1 statistic at separating a KNOWN-OOD cloud from the in-distribution null
    (the power measurement GATE 1 requires). Uses the same fit/null split machinery."""
    from src.ml.eval.ood import auroc
    fit, null = cosmology_split(cloud_id.sim_ids, seed=seed)
    # The variate stores re-process the SAME Gower simulations as the in-distribution store, so an
    # OOD row whose cosmology (and density field) sits in the FIT half has an unfairly close
    # neighbour there (measured: AUROC 0.23 for the VD variate before this filter). Score only OOD
    # rows from cosmologies of the NULL half -- the same unseen-cosmology footing as the null
    # events themselves (and paired with them when the sim ids coincide). Unrelated OOD clouds
    # (no shared sim ids) fall back to all rows.
    null_ids = set(np.asarray(cloud_id.sim_ids)[null].tolist())
    q_ok = np.array([int(i) in null_ids for i in np.asarray(cloud_ood.sim_ids)], dtype=bool)
    if not q_ok.any():
        q_ok[:] = True
    q_idx = np.where(q_ok)[0][:max_ood]
    if which == "emap":
        E = np.asarray(cloud_id.emap, dtype=np.float64)
        Q = np.asarray(cloud_ood.emap, dtype=np.float64)[q_idx]
        finite = np.isfinite(E).all(0) & np.isfinite(Q).all(0)
        sd = E[:, finite].std(0)
        keep = np.where(finite)[0][sd > 0]
        E, Q = E[:, keep], Q[:, keep]
        med = np.median(E, 0)
        mad = np.median(np.abs(E - med), 0) * 1.4826
        mad = np.where(mad > 0, mad, E.std(0))
        E, Q = (E - med) / mad, (Q - med) / mad
    else:
        E = np.asarray(cloud_id.bandpowers, dtype=np.float64).reshape(len(cloud_id.bandpowers), -1)
        Q = np.asarray(cloud_ood.bandpowers, dtype=np.float64).reshape(len(cloud_ood.bandpowers), -1)[q_idx]
        sd = E[fit].std(0)                       # scatter units before whitening (see _score_block)
        sd = np.where(sd > 0, sd, 1.0)
        E, Q = E / sd, Q / sd
    res = {}
    for tag, p in (("full", None), (f"pca{pca}", min(pca, E.shape[1]))):
        w = _shrink_whitener(E[fit])
        zf, zn, zq = w(E[fit]), w(E[null]), w(Q)
        if p:
            u, s, vt = np.linalg.svd(zf - zf.mean(0), full_matrices=False)
            P = vt[:p].T
            zf, zn, zq = zf @ P, zn @ P, zq @ P
        res[tag] = {"auroc_knn": float(auroc(knn_scores(zf, zn, k=k), knn_scores(zf, zq, k=k))),
                    "auroc_mahalanobis": float(auroc(mahalanobis_scores(zn), mahalanobis_scores(zq))),
                    "n_null": int(len(zn)), "n_ood": int(len(zq)),
                    "n_ood_cosmologies": int(len(np.unique(np.asarray(cloud_ood.sim_ids)[q_idx]))),
                    "ood_restricted_to_null_cosmologies": bool(q_ok.sum() < len(q_ok))}
    return res


def run_tier1(cloud: Cloud, obs: Dict[str, np.ndarray], labels: Sequence[str], *, seed: int = 0,
              exclude_sim_ids: Sequence[int] = ()) -> Dict:
    """``obs``: {'bandpowers': [M,21,B], 'bb': [M,21,B], 'emap': [M,D]} for M labels."""
    out = {"labels": list(labels), "seed": seed, "n_excluded_sim_ids": int(len(exclude_sim_ids))}
    if cloud.bandpowers is not None and "bandpowers" in obs:
        out["twopoint"] = twopoint_check(cloud, obs["bandpowers"], seed=seed, exclude_sim_ids=exclude_sim_ids)
    if cloud.bb is not None and "bb" in obs:
        out["bmodes"] = bmode_check(cloud, obs["bb"], seed=seed, exclude_sim_ids=exclude_sim_ids)
    if cloud.emap is not None and "emap" in obs:
        out["emap"] = emap_check(cloud, obs["emap"], seed=seed, exclude_sim_ids=exclude_sim_ids)
    return out


def verdict(res: Dict, alpha: float = 0.01) -> Dict[str, List[str]]:
    """Pre-registered pass/fail per label: KiDS-Legacy-style PTE > alpha on the B-null; right-tail
    p > alpha on the 2-pt kNN and the E-map kNN. Returns {label: [failed tests]}."""
    labels = res["labels"]
    fails = {l: [] for l in labels}
    for i, l in enumerate(labels):
        if "twopoint" in res and res["twopoint"]["knn"]["p"][i] < alpha:
            fails[l].append("twopoint_knn")
        if "twopoint" in res and res["twopoint"]["mahalanobis"]["p"][i] < alpha:
            fails[l].append("twopoint_mahalanobis")
        if "bmodes" in res and res["bmodes"]["pte_empirical"][i] < alpha:
            fails[l].append("bmode_null")
        if "emap" in res and res["emap"]["full"]["knn"]["p"][i] < alpha:
            fails[l].append("emap_knn")
    return fails
