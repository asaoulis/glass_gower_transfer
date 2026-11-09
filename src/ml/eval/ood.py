"""Summary-space out-of-distribution diagnostics for model misspecification.

Works on small arrays of compressed summaries z (and scaled theta): Mahalanobis, kNN (Sun et al.
2022) and theta-conditional kNN scores, empirical p-values against the held-out null, MMD and
classifier two-sample tests. Summaries are per model and never pooled across models. Extraction
lives in summaries.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


# --- whitening (train cloud only) ---
@dataclass
class TrainWhitener:
    """Affine whitening fit on the train summary cloud: x -> L^{-1}(x - mean)."""

    mean: np.ndarray
    chol_inv: np.ndarray  # [D, D], inverse Cholesky factor of the covariance
    eps: float = 1e-8

    @classmethod
    def fit(cls, z_train: np.ndarray, eps: float = 1e-8) -> "TrainWhitener":
        z = np.asarray(z_train, dtype=np.float64)
        mean = z.mean(axis=0)
        cov = np.cov(z - mean, rowvar=False)
        cov = np.atleast_2d(cov) + eps * np.eye(z.shape[1])
        L = np.linalg.cholesky(cov)
        return cls(mean=mean, chol_inv=np.linalg.inv(L), eps=eps)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        return (z - self.mean) @ self.chol_inv.T


# --- per-event scores ---
def mahalanobis_scores(zw_query: np.ndarray) -> np.ndarray:
    """Mahalanobis distance in the whitened frame (= Euclidean norm)."""
    return np.sqrt((np.asarray(zw_query) ** 2).sum(axis=1))


def _knn_workers() -> int:
    import os
    env = os.environ.get("OOD_KNN_WORKERS")
    if env:
        return int(env)
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, min(32, n))


def knn_scores(zw_train: np.ndarray, zw_query: np.ndarray, k: int = 10, *, exclude_self: bool = False) -> np.ndarray:
    """Mean Euclidean distance to the k nearest train points (whitened frame).

    exclude_self drops the zero self-distance when the query is the train cloud.
    """
    if cKDTree is None:
        raise ImportError("knn_scores needs scipy")
    tree = cKDTree(np.asarray(zw_train, dtype=np.float64))
    kk = k + 1 if exclude_self else k
    # The parallel query is bit-identical to the serial one. Threads follow the CPU allocation, capped at 32.
    d, _ = tree.query(np.asarray(zw_query, dtype=np.float64), k=kk, workers=_knn_workers())
    d = np.atleast_2d(d)
    if exclude_self:
        d = d[:, 1:]
    return d.mean(axis=1)


class ThetaKNNRegressor:
    """E[z | theta] by k-nearest-neighbour averaging in (scaled) theta space, fit on the train set."""

    def __init__(self, theta_train: np.ndarray, z_train: np.ndarray, k: int = 50, dims: Optional[Sequence[int]] = None):
        if cKDTree is None:
            raise ImportError("ThetaKNNRegressor needs scipy")
        self.dims = list(dims) if dims is not None else list(range(np.asarray(theta_train).shape[1]))
        self.theta = np.asarray(theta_train, dtype=np.float64)[:, self.dims]
        self.z = np.asarray(z_train, dtype=np.float64)
        self.k = int(k)
        self.tree = cKDTree(self.theta)

    def predict(self, theta_query: np.ndarray, *, exclude_self: bool = False) -> np.ndarray:
        q = np.asarray(theta_query, dtype=np.float64)[:, self.dims]
        kk = self.k + 1 if exclude_self else self.k
        _, idx = self.tree.query(q, k=kk)
        idx = np.atleast_2d(idx)
        if exclude_self:
            idx = idx[:, 1:]
        return self.z[idx].mean(axis=1)


@dataclass
class ConditionalResidualModel:
    """Residual r = z - E[z|theta] on the train cloud, whitened by the train residual covariance."""

    regressor: ThetaKNNRegressor
    whitener: TrainWhitener
    residual_train_w: np.ndarray = field(repr=False)

    @classmethod
    def fit(cls, theta_train: np.ndarray, z_train: np.ndarray, *, k_theta: int = 50,
            theta_dims: Optional[Sequence[int]] = None) -> "ConditionalResidualModel":
        reg = ThetaKNNRegressor(theta_train, z_train, k=k_theta, dims=theta_dims)
        r_train = np.asarray(z_train, dtype=np.float64) - reg.predict(theta_train, exclude_self=True)
        wh = TrainWhitener.fit(r_train)
        return cls(regressor=reg, whitener=wh, residual_train_w=wh(r_train))

    def residual_w(self, theta: np.ndarray, z: np.ndarray) -> np.ndarray:
        r = np.asarray(z, dtype=np.float64) - self.regressor.predict(theta)
        return self.whitener(r)

    def scores(self, theta: np.ndarray, z: np.ndarray, *, k: int = 10) -> Dict[str, np.ndarray]:
        rw = self.residual_w(theta, z)
        return {
            "cond_mahalanobis": mahalanobis_scores(rw),
            "cond_knn": knn_scores(self.residual_train_w, rw, k=k),
        }


# --- calibration and dataset-level statistics ---
def empirical_pvalues(null_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Right-tail empirical p-value of each score against the null sample (higher score = more OOD)."""
    null = np.sort(np.asarray(null_scores, dtype=np.float64))
    s = np.asarray(scores, dtype=np.float64)
    n_ge = null.size - np.searchsorted(null, s, side="left")
    return (1.0 + n_ge) / (1.0 + null.size)


def auroc(scores_null: np.ndarray, scores_query: np.ndarray) -> float:
    """AUROC for 'query is OOD' with higher score = more OOD (rank-based, tie-aware)."""
    from scipy.stats import rankdata

    a = np.asarray(scores_null, dtype=np.float64)
    b = np.asarray(scores_query, dtype=np.float64)
    ranks = rankdata(np.concatenate([a, b]))
    rb = ranks[a.size:].sum()
    return float((rb - b.size * (b.size + 1) / 2.0) / (a.size * b.size))


def _rbf_mmd2(x: np.ndarray, y: np.ndarray, sigma2: float) -> float:
    def k(a, b):
        d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
        return np.exp(-np.maximum(d2, 0.0) / (2.0 * sigma2))
    n, m = x.shape[0], y.shape[0]
    kxx = k(x, x); kyy = k(y, y); kxy = k(x, y)
    # Unbiased estimator.
    return float((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2.0 * kxy.mean())


def mmd_permutation_test(x: np.ndarray, y: np.ndarray, *, n_perm: int = 200, max_n: int = 1500,
                         seed: int = 0, sigma2: Optional[float] = None) -> Dict[str, float]:
    """RBF-MMD^2 two-sample test with a permutation p-value (median-heuristic bandwidth on the pooled set)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if x.shape[0] > max_n:
        x = x[rng.choice(x.shape[0], max_n, replace=False)]
    if y.shape[0] > max_n:
        y = y[rng.choice(y.shape[0], max_n, replace=False)]
    pooled = np.concatenate([x, y])
    if sigma2 is None:
        sub = pooled[rng.choice(pooled.shape[0], min(2000, pooled.shape[0]), replace=False)]
        d2 = (sub * sub).sum(1)[:, None] + (sub * sub).sum(1)[None, :] - 2.0 * sub @ sub.T
        sigma2 = float(np.median(d2[np.triu_indices(sub.shape[0], 1)]))
        sigma2 = sigma2 if np.isfinite(sigma2) and sigma2 > 0 else 1.0
    stat = _rbf_mmd2(x, y, sigma2)
    n = x.shape[0]
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(pooled.shape[0])
        null[i] = _rbf_mmd2(pooled[p[:n]], pooled[p[n:]], sigma2)
    pval = float((1.0 + (null >= stat).sum()) / (1.0 + n_perm))
    return {"mmd2": stat, "mmd_pvalue": pval, "mmd_null_mean": float(null.mean()), "mmd_null_std": float(null.std()), "sigma2": sigma2}


def c2st_auroc(x: np.ndarray, y: np.ndarray, *, seed: int = 0, n_splits: int = 5) -> Dict[str, float]:
    """Classifier two-sample test: cross-validated AUROC of x against y (0.5 = indistinguishable).

    Returns the AUROC of a logistic regression and of a histogram gradient boosting model.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X = np.concatenate([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)])
    yv = np.concatenate([np.zeros(len(x)), np.ones(len(y))])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = {}
    for name, clf in (("c2st_auroc_linear", LogisticRegression(max_iter=2000)),
                      ("c2st_auroc_gbm", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed))):
        prob = cross_val_predict(clf, X, yv, cv=cv, method="predict_proba")[:, 1]
        out[name] = float(roc_auc_score(yv, prob))
    return out


# --- the suite: one model, one query set ---
@dataclass
class OODReference:
    """Everything fit on one model's train cloud and its held-out null, reused for every query set.

    The conditional model is built lazily per tuple of theta dims, on exactly the dims the query
    can supply.
    """

    whitener: TrainWhitener
    z_train: np.ndarray = field(repr=False)
    z_id: np.ndarray = field(repr=False)
    zw_train: np.ndarray = field(repr=False)
    zw_id: np.ndarray = field(repr=False)
    null_scores: Dict[str, np.ndarray] = field(repr=False)
    k: int = 10
    k_theta: int = 50
    theta_train: Optional[np.ndarray] = field(default=None, repr=False)
    theta_id: Optional[np.ndarray] = field(default=None, repr=False)
    n_train: int = 0
    _cond: Dict[tuple, ConditionalResidualModel] = field(default_factory=dict, repr=False)
    _cond_null: Dict[tuple, Dict[str, np.ndarray]] = field(default_factory=dict, repr=False)

    @classmethod
    def fit(cls, z_train: np.ndarray, z_id: np.ndarray, *, theta_train: Optional[np.ndarray] = None,
            theta_id: Optional[np.ndarray] = None, k: int = 10, k_theta: int = 50,
            max_train: Optional[int] = None, seed: int = 0) -> "OODReference":
        rng = np.random.default_rng(seed)
        z_train = np.asarray(z_train, dtype=np.float64)
        if theta_train is not None:
            theta_train = np.asarray(theta_train, dtype=np.float64)
        if max_train is not None and z_train.shape[0] > max_train:
            sel = rng.choice(z_train.shape[0], max_train, replace=False)
            z_train = z_train[sel]
            theta_train = theta_train[sel] if theta_train is not None else None
        wh = TrainWhitener.fit(z_train)
        zw_train = wh(z_train)
        z_id = np.asarray(z_id, dtype=np.float64)
        zw_id = wh(z_id)
        null = {"mahalanobis": mahalanobis_scores(zw_id), "knn": knn_scores(zw_train, zw_id, k=k)}
        return cls(whitener=wh, z_train=z_train, z_id=z_id, zw_train=zw_train, zw_id=zw_id, null_scores=null,
                   k=k, k_theta=k_theta, theta_train=theta_train,
                   theta_id=None if theta_id is None else np.asarray(theta_id, dtype=np.float64),
                   n_train=z_train.shape[0])

    def _conditional(self, dims: Sequence[int]):
        if self.theta_train is None or self.theta_id is None:
            return None, None
        key = tuple(int(d) for d in dims)
        if key not in self._cond:
            fin_tr = np.isfinite(self.theta_train[:, list(key)]).all(axis=1)
            cm = ConditionalResidualModel.fit(self.theta_train[fin_tr], self.z_train[fin_tr], k_theta=self.k_theta, theta_dims=list(key))
            fin_id = np.isfinite(self.theta_id[:, list(key)]).all(axis=1)
            null = cm.scores(self.theta_id[fin_id], self.z_id[fin_id], k=self.k)
            self._cond[key] = cm
            self._cond_null[key] = null
        return self._cond[key], self._cond_null[key]

    def _query_dims(self, theta_query: Optional[np.ndarray], theta_dims: Optional[Sequence[int]]):
        if theta_query is None or self.theta_train is None:
            return None
        th = np.asarray(theta_query, dtype=np.float64)
        dims = list(theta_dims) if theta_dims is not None else list(range(th.shape[1]))
        # Drop dims the query cannot supply (NaN-filled params).
        dims = [d for d in dims if np.isfinite(th[:, d]).all()]
        return dims or None

    def score(self, z_query: np.ndarray, theta_query: Optional[np.ndarray] = None,
              theta_dims: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
        zw = self.whitener(z_query)
        s = {"mahalanobis": mahalanobis_scores(zw), "knn": knn_scores(self.zw_train, zw, k=self.k)}
        dims = self._query_dims(theta_query, theta_dims)
        if dims is not None:
            cm, _ = self._conditional(dims)
            if cm is not None:
                s.update(cm.scores(np.asarray(theta_query, dtype=np.float64), np.asarray(z_query, dtype=np.float64), k=self.k))
        return s

    def evaluate(self, z_query: np.ndarray, theta_query: Optional[np.ndarray] = None, *,
                 theta_dims: Optional[Sequence[int]] = None, two_sample: bool = True,
                 n_perm: int = 200, seed: int = 0) -> Dict:
        """Per-event scores and p-values, and dataset-level statistics, for one query set."""
        scores = self.score(z_query, theta_query, theta_dims)
        dims = self._query_dims(theta_query, theta_dims)
        null = dict(self.null_scores)
        cm = None
        if dims is not None:
            cm, cnull = self._conditional(dims)
            if cnull is not None:
                null.update(cnull)
        per_event: Dict[str, np.ndarray] = {}
        dataset: Dict[str, float] = {"n_query": int(np.asarray(z_query).shape[0]), "n_null": int(self.zw_id.shape[0]),
                                     "n_train": int(self.n_train), "theta_dims": list(dims) if dims is not None else None}
        for name, s in scores.items():
            p = empirical_pvalues(null[name], s)
            per_event[f"{name}_score"] = s
            per_event[f"{name}_p"] = p
            dataset[f"{name}_mean_p"] = float(np.mean(p))
            dataset[f"{name}_frac_p_lt_0p05"] = float(np.mean(p < 0.05))
            dataset[f"{name}_auroc"] = auroc(null[name], s)
            dataset[f"{name}_median_score_ratio"] = float(np.median(s) / max(np.median(null[name]), 1e-12))
        if two_sample:
            zw = self.whitener(z_query)
            dataset.update(mmd_permutation_test(self.zw_id, zw, n_perm=n_perm, seed=seed))
            try:
                dataset.update(c2st_auroc(self.zw_id, zw, seed=seed))
            except Exception as e:
                dataset["c2st_error"] = repr(e)
            if cm is not None and "cond_knn_score" in per_event:
                fin_id = np.isfinite(self.theta_id[:, dims]).all(axis=1)
                rw_id = cm.residual_w(self.theta_id[fin_id], self.z_id[fin_id])
                rw_q = cm.residual_w(np.asarray(theta_query, dtype=np.float64), np.asarray(z_query, dtype=np.float64))
                m = mmd_permutation_test(rw_id, rw_q, n_perm=n_perm, seed=seed)
                dataset.update({f"cond_{k}": v for k, v in m.items()})
                try:
                    dataset.update({f"cond_{k}": v for k, v in c2st_auroc(rw_id, rw_q, seed=seed).items()})
                except Exception as e:
                    dataset["cond_c2st_error"] = repr(e)
        return {"per_event": per_event, "dataset": dataset}
