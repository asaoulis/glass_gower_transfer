class DataScaler:
    def __init__(self):
        self.min = None
        self.max = None
        self.mean = None
        self.std = None

    def fit_minmax(self, X):
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)
    
    def transform_minmax(self, X):
        return (X - self.min) / (self.max - self.min)
    
    def inverse_transform_minmax(self, X):
        return X * (self.max - self.min) + self.min

    def fit_standard(self, X):
        self.mean = X.mean()
        self.std = X.std()
    
    def transform_standard(self, X):
        return (X - self.mean) / self.std
    
    def inverse_transform_standard(self, X):
        return X * self.std + self.mean

# New, abstracted scaler classes with transform()/inverse_transform()
# These are backend-agnostic (NumPy or PyTorch) and convert parameters on the fly

import numpy as np
import torch
from typing import Optional, Union

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_backend(x: ArrayLike, arr: ArrayLike):
    if torch.is_tensor(x):
        return torch.as_tensor(arr, dtype=x.dtype, device=x.device)
    return np.asarray(arr)


class BaseScaler:
    def fit(self, X: ArrayLike):
        raise NotImplementedError

    def transform(self, X: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def inverse_transform(self, X: ArrayLike) -> ArrayLike:
        raise NotImplementedError


class MinMaxScaler(BaseScaler):
    def __init__(self, parameter_names= None):
        self.parameter_names = parameter_names
        self.min: Optional[np.ndarray] = None
        self.max: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike):
        Xn = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
        self.min = Xn.min(axis=0)
        self.max = Xn.max(axis=0)
        span = self.max - self.min
        span[span == 0] = 1.0
        self.max = self.min + span

    def transform(self, X: ArrayLike) -> ArrayLike:
        if self.min is None or self.max is None:
            return X
        min_v = _to_backend(X, self.min)
        max_v = _to_backend(X, self.max)
        denom = max_v - min_v
        if torch.is_tensor(denom):
            denom = torch.clamp(denom, min=1e-12)
        else:
            denom = np.clip(denom, 1e-12, None)
        return (X - min_v) / denom

    def inverse_transform(self, X: ArrayLike) -> ArrayLike:
        if self.min is None or self.max is None:
            return X
        min_v = _to_backend(X, self.min)
        max_v = _to_backend(X, self.max)
        return X * (max_v - min_v) + min_v


class StandardScaler(BaseScaler):
    """Scalar standard scaler: stores a single mean/std for the whole array.

    This preserves the original behaviour used for data scalers.
    """

    def __init__(self):
        self.mean: Optional[float] = None
        self.std: Optional[float] = None

    def fit(self, X: ArrayLike):
        Xn = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
        self.mean = float(Xn.mean())
        std = float(Xn.std())
        if std == 0 or not np.isfinite(std):
            std = 1.0
        self.std = std

    def transform(self, X: ArrayLike) -> ArrayLike:
        if self.mean is None or self.std is None:
            return X
        return (X - float(self.mean)) / float(self.std)

    def inverse_transform(self, X: ArrayLike) -> ArrayLike:
        if self.mean is None or self.std is None:
            return X
        return X * float(self.std) + float(self.mean)


class PerDimStandardScaler(BaseScaler):
    """Per-feature standard scaler (mean/std) for embeddings.

    For 1D inputs it behaves as a scalar standardiser; for 2D or higher,
    statistics are computed along axis=0.
    """

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike):
        Xn = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
        if Xn.ndim <= 1:
            mean = Xn.mean()
            std = Xn.std()
        else:
            mean = Xn.mean(axis=0)
            std = Xn.std(axis=0)
        std = np.where((std == 0) | ~np.isfinite(std), 1.0, std)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def transform(self, X: ArrayLike) -> ArrayLike:
        if self.mean is None or self.std is None:
            return X
        mean_v = _to_backend(X, self.mean)
        std_v = _to_backend(X, self.std)
        if torch.is_tensor(std_v):
            std_v = torch.clamp(std_v, min=1e-12)
        else:
            std_v = np.clip(std_v, 1e-12, None)
        return (X - mean_v) / std_v

    def inverse_transform(self, X: ArrayLike) -> ArrayLike:
        if self.mean is None or self.std is None:
            return X
        mean_v = _to_backend(X, self.mean)
        std_v = _to_backend(X, self.std)
        return X * std_v + mean_v


class WhitenPCAScaler(BaseScaler):
    """Standardise per-dim (fit-split mean/std), rotate into the PCA basis, keep top-k.

    Direct port of the validated ``Whitener`` prototype in
    ``.claude/runs/training-runs/nle-overconfidence-research/artifacts/diagnostics/fix_prototype_pipeline.py``
    (lines 68-85), extended with a versioned ``save``/``load`` persistence format (with
    fit-source metadata so cross-run mismatches are detectable) and an ``inverse_transform``
    for BaseScaler-contract completeness / debugging.

    Recipe (must match the prototype byte-for-byte on the same data, up to per-PC sign):
      1. mean/std over axis 0 (torch unbiased std, matching ``z.std(0)``).
      2. covariance of the standardised data via ``np.cov`` (ddof=1).
      3. ``torch.linalg.eigh`` of that covariance, eigenvalues sorted descending.
      4. keep the top-``k`` eigenvectors (``components``); whiten each PC by ``sqrt(eigenvalue)``.

    ``k == input_dim`` is the pure-whiten mode (no truncation); ``k < input_dim`` also truncates.
    The transform is a fixed affine map, so once fit on the pretrain (GLASS) train split and
    persisted, it is REUSED unchanged on the finetune (Gower) fidelity and at eval — this is the
    non-negotiable fix for the per-run-refit failure mode (research Finding C3).
    """

    VERSION = "whiten_pca_v1"

    def __init__(self, k: Optional[int] = None):
        self.k = k
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.components: Optional[np.ndarray] = None  # (input_dim, k)
        self.scale: Optional[np.ndarray] = None       # (k,)
        self.input_dim: Optional[int] = None
        self.explained_variance_ratio: Optional[np.ndarray] = None
        # Fit-source metadata (informational; enables mismatch detection across runs).
        self.fit_source_experiment: Optional[str] = None
        self.fit_repeat_match: Optional[str] = None
        self.fit_n_train_samples: Optional[int] = None

    def fit(self, X: ArrayLike, k: Optional[int] = None):
        Xt = X if torch.is_tensor(X) else torch.as_tensor(np.asarray(X))
        Xt = Xt.detach().cpu().float()
        if Xt.ndim != 2:
            raise ValueError(f"WhitenPCAScaler.fit expects a 2D array, got shape {tuple(Xt.shape)}.")
        n, dim = Xt.shape
        k = int(k) if k is not None else (int(self.k) if self.k is not None else dim)
        if not (1 <= k <= dim):
            raise ValueError(f"WhitenPCAScaler: k={k} must satisfy 1 <= k <= input_dim={dim}.")

        mean = Xt.mean(0)
        std = Xt.std(0)  # unbiased (correction=1), matches the prototype's z.std(0)
        std = torch.where((std == 0) | ~torch.isfinite(std), torch.ones_like(std), std)
        x = (Xt - mean) / std
        cov = torch.from_numpy(np.cov(x.numpy().T)).float()
        evals, evecs = torch.linalg.eigh(cov)
        order = torch.argsort(evals, descending=True)
        evals_sorted = evals[order]
        W = evecs[:, order][:, :k]              # (dim, k)
        scale = evals_sorted[:k].clamp(min=1e-12).sqrt()  # whiten each PC to unit variance
        evr = (evals_sorted[:k] / evals_sorted.clamp(min=0).sum().clamp(min=1e-12))

        self.k = k
        self.input_dim = int(dim)
        self.mean = mean.numpy().astype(np.float32)
        self.std = std.numpy().astype(np.float32)
        self.components = W.numpy().astype(np.float32)
        self.scale = scale.numpy().astype(np.float32)
        self.explained_variance_ratio = evr.numpy().astype(np.float32)
        self.fit_n_train_samples = int(n)
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        if self.components is None:
            return X
        mean = _to_backend(X, self.mean)
        std = _to_backend(X, self.std)
        W = _to_backend(X, self.components)
        scale = _to_backend(X, self.scale)
        x = (X - mean) / std
        return (x @ W) / scale

    def inverse_transform(self, X: ArrayLike) -> ArrayLike:
        """Lossy inverse (the truncated null space is zeroed)."""
        if self.components is None:
            return X
        mean = _to_backend(X, self.mean)
        std = _to_backend(X, self.std)
        W = _to_backend(X, self.components)
        scale = _to_backend(X, self.scale)
        x = (X * scale) @ (W.T if torch.is_tensor(W) else W.T)
        return x * std + mean

    def state(self) -> dict:
        return {
            "kind": self.VERSION,
            "input_dim": self.input_dim,
            "k": self.k,
            "mean": torch.as_tensor(self.mean) if self.mean is not None else None,
            "std": torch.as_tensor(self.std) if self.std is not None else None,
            "components": torch.as_tensor(self.components) if self.components is not None else None,
            "scale": torch.as_tensor(self.scale) if self.scale is not None else None,
            "explained_variance_ratio": (
                torch.as_tensor(self.explained_variance_ratio)
                if self.explained_variance_ratio is not None else None
            ),
            "fit_source_experiment": self.fit_source_experiment,
            "fit_repeat_match": self.fit_repeat_match,
            "fit_n_train_samples": self.fit_n_train_samples,
        }

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state(), path)

    @classmethod
    def load(cls, path: str) -> "WhitenPCAScaler":
        st = torch.load(path, map_location="cpu")
        kind = st.get("kind")
        if kind != cls.VERSION:
            raise ValueError(
                f"WhitenPCAScaler.load: unexpected file kind '{kind}' at {path} "
                f"(expected '{cls.VERSION}')."
            )
        obj = cls(k=st.get("k"))
        obj.input_dim = st.get("input_dim")

        def _np(x):
            if x is None:
                return None
            if torch.is_tensor(x):
                return x.detach().cpu().numpy().astype(np.float32)
            return np.asarray(x, dtype=np.float32)

        obj.mean = _np(st.get("mean"))
        obj.std = _np(st.get("std"))
        obj.components = _np(st.get("components"))
        obj.scale = _np(st.get("scale"))
        obj.explained_variance_ratio = _np(st.get("explained_variance_ratio"))
        obj.fit_source_experiment = st.get("fit_source_experiment")
        obj.fit_repeat_match = st.get("fit_repeat_match")
        obj.fit_n_train_samples = st.get("fit_n_train_samples")
        return obj


class LogNormalScaler(BaseScaler):
    """
    Applies a log10 transform followed by standard scaling, and inverts accordingly.
    Values <= 0 are clamped to a small epsilon before logging.
    """
    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)
        self.mean: Optional[float] = None
        self.std: Optional[float] = None

    def _log(self, X: ArrayLike) -> ArrayLike:
        if torch.is_tensor(X):
            return torch.log10(torch.clamp(X, min=self.eps))
        return np.log10(np.clip(X, self.eps, None))

    def _pow10(self, X: ArrayLike) -> ArrayLike:
        if torch.is_tensor(X):
            return 10.0 ** X
        return np.power(10.0, X)

    def fit(self, X: ArrayLike):
        Xlog = self._log(X)
        Xn = Xlog.detach().cpu().numpy() if torch.is_tensor(Xlog) else np.asarray(Xlog)
        self.mean = float(Xn.mean())
        std = float(Xn.std())
        if std == 0 or not np.isfinite(std):
            std = 1.0
        self.std = std

    def transform(self, X: ArrayLike) -> ArrayLike:
        if self.mean is None or self.std is None:
            return X
        Z = self._log(X)
        return (Z - float(self.mean)) / float(self.std)

    def inverse_transform(self, X: ArrayLike) -> ArrayLike:
        if self.mean is None or self.std is None:
            return X
        Z = X * float(self.std) + float(self.mean)
        return self._pow10(Z)