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
    # numpy path
    return np.asarray(arr)


class BaseScaler:
    def fit(self, X: ArrayLike):
        raise NotImplementedError

    def transform(self, X: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def inverse_transform(self, X: ArrayLike) -> ArrayLike:
        raise NotImplementedError


class MinMaxScaler(BaseScaler):
    def __init__(self):
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
        mean_v = float(self.mean)
        std_v = float(self.std) if self.std != 0 else 1.0
        return (X - mean_v) / std_v

    def inverse_transform(self, X: ArrayLike) -> ArrayLike:
        if self.mean is None or self.std is None:
            return X
        return X * float(self.std) + float(self.mean)


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