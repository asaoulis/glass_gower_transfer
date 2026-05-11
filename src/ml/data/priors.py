import torch
import numpy as np
from scipy.stats import gaussian_kde
from torch.distributions import Distribution, constraints
from torch.distributions import Normal, Uniform, TransformedDistribution
from torch.distributions.transforms import ExpTransform
import math
import os
import pickle
import torch
import numpy as np

try:
    from sbi.utils import MultipleIndependent, RestrictedPrior
except Exception:  # pragma: no cover
    MultipleIndependent = None
    RestrictedPrior = None


class S8OmegaCH2OmegaBH2HPriorToModelParams(Distribution):
    """Joint prior in model parameter space implied by a flat box in (S8, ωc, ωb, h).

    Base (flat) parameters:
      - S8 in [s8_low, s8_high]
      - ωc ≡ Ωc h^2 in [oc_low, oc_high]
      - ωb ≡ Ωb h^2 in [ob_low, ob_high]
      - h in [h_low, h_high]

    Model parameters (output / log_prob input) are a 4-vector containing:
      {"sigma_8", "omega_m", "ombh2", "h"} in the provided `parameter_order`.

    Mapping:
      Ωm = (ωc + ωb) / h^2
      sigma_8 = S8 / sqrt(Ωm / pivot_omega_m)

    The induced density in (sigma_8, Ωm, ωb, h) includes the Jacobian
      |det ∂(S8, ωc, ωb, h) / ∂(sigma_8, Ωm, ωb, h)| = h^2 * sqrt(Ωm / pivot).
    """

    arg_constraints = {}
    support = constraints.real
    has_rsample = False

    def __init__(
        self,
        *,
        s8_low: float = 0.5,
        s8_high: float = 1.0,
        oc_low: float = 0.051,
        oc_high: float = 0.22,
        ob_low: float = 0.022,
        ob_high: float = 0.0228,
        h_low: float = 0.64,
        h_high: float = 0.78,
        pivot_omega_m: float = 0.3,
        parameter_order=("sigma_8", "omega_m", "ombh2", "h"),
    ):
        super().__init__(validate_args=False)

        parameter_order = tuple(parameter_order)
        expected = {"sigma_8", "omega_m", "ombh2", "h"}
        if set(parameter_order) != expected or len(parameter_order) != 4:
            raise ValueError(
                "S8OmegaCH2OmegaBH2HPriorToModelParams: parameter_order must contain "
                f"exactly {sorted(expected)}; got {parameter_order}."
            )

        self.parameter_order = parameter_order
        self._idx = {name: i for i, name in enumerate(parameter_order)}

        self.s8_low = torch.tensor(float(s8_low), dtype=torch.float32)
        self.s8_high = torch.tensor(float(s8_high), dtype=torch.float32)
        self.oc_low = torch.tensor(float(oc_low), dtype=torch.float32)
        self.oc_high = torch.tensor(float(oc_high), dtype=torch.float32)
        self.ob_low = torch.tensor(float(ob_low), dtype=torch.float32)
        self.ob_high = torch.tensor(float(ob_high), dtype=torch.float32)
        self.h_low = torch.tensor(float(h_low), dtype=torch.float32)
        self.h_high = torch.tensor(float(h_high), dtype=torch.float32)

        self.pivot_omega_m = torch.tensor(float(pivot_omega_m), dtype=torch.float32)
        if float(pivot_omega_m) <= 0:
            raise ValueError("pivot_omega_m must be > 0")

        self._event_shape = torch.Size([4])
        self._batch_shape = torch.Size([])

        # Exact normalisation for base uniform box
        vol = (
            (self.s8_high - self.s8_low)
            * (self.oc_high - self.oc_low)
            * (self.ob_high - self.ob_low)
            * (self.h_high - self.h_low)
        )
        self._log_base_const = -torch.log(vol)

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def batch_shape(self):
        return self._batch_shape

    def _to_device_bounds(self, device):
        return (
            self.s8_low.to(device),
            self.s8_high.to(device),
            self.oc_low.to(device),
            self.oc_high.to(device),
            self.ob_low.to(device),
            self.ob_high.to(device),
            self.h_low.to(device),
            self.h_high.to(device),
            self.pivot_omega_m.to(device),
            self._log_base_const.to(device),
        )

    def sample(self, sample_shape=torch.Size()):
        device = self.s8_low.device
        u = torch.rand(sample_shape + torch.Size([4]), device=device, dtype=torch.float32)

        s8_low, s8_high, oc_low, oc_high, ob_low, ob_high, h_low, h_high, pivot, _ = self._to_device_bounds(device)

        S8 = s8_low + (s8_high - s8_low) * u[..., 0]
        oc = oc_low + (oc_high - oc_low) * u[..., 1]
        ob = ob_low + (ob_high - ob_low) * u[..., 2]
        h = h_low + (h_high - h_low) * u[..., 3]

        omega_m = (oc + ob) / (h**2)
        sigma8 = S8 / torch.sqrt(omega_m / pivot)

        out = torch.empty(sample_shape + self.event_shape, device=device, dtype=torch.float32)
        out[..., self._idx["sigma_8"]] = sigma8
        out[..., self._idx["omega_m"]] = omega_m
        out[..., self._idx["ombh2"]] = ob
        out[..., self._idx["h"]] = h
        return out

    def log_prob(self, value):
        x = value
        device = x.device
        s8_low, s8_high, oc_low, oc_high, ob_low, ob_high, h_low, h_high, pivot, log_base_const = self._to_device_bounds(device)

        sigma8 = x[..., self._idx["sigma_8"]]
        omega_m = x[..., self._idx["omega_m"]]
        ob = x[..., self._idx["ombh2"]]
        h = x[..., self._idx["h"]]

        # Back-transform to box variables
        S8 = sigma8 * torch.sqrt(omega_m / pivot)
        oc = omega_m * (h**2) - ob

        in_box = (
            (S8 >= s8_low) & (S8 <= s8_high)
            & (oc >= oc_low) & (oc <= oc_high)
            & (ob >= ob_low) & (ob <= ob_high)
            & (h >= h_low) & (h <= h_high)
            & (omega_m > 0.0)
            & (sigma8 > 0.0)
        )

        out = torch.full_like(S8, float("-inf"))
        if not torch.any(in_box):
            return out

        # log |Jacobian| = log(h^2 * sqrt(omega_m/pivot))
        log_j = 2.0 * torch.log(h) + 0.5 * (torch.log(omega_m) - torch.log(pivot))

        out[in_box] = (log_base_const + log_j)[in_box]
        return out

    def to(self, device):
        self.s8_low = self.s8_low.to(device)
        self.s8_high = self.s8_high.to(device)
        self.oc_low = self.oc_low.to(device)
        self.oc_high = self.oc_high.to(device)
        self.ob_low = self.ob_low.to(device)
        self.ob_high = self.ob_high.to(device)
        self.h_low = self.h_low.to(device)
        self.h_high = self.h_high.to(device)
        self.pivot_omega_m = self.pivot_omega_m.to(device)
        self._log_base_const = self._log_base_const.to(device)
        return self


class TruncatedNormal1D(Distribution):
    """1D truncated normal distribution.

    This is a minimal Distribution implementation used for analytic priors.
    It uses rejection sampling for `.sample()` and an exact normalization
    constant for `.log_prob()`.
    """

    arg_constraints = {}
    has_rsample = False

    def __init__(self, loc, scale, low, high, max_tries: int = 10_000):
        super().__init__(validate_args=False)

        self.loc = torch.as_tensor(loc, dtype=torch.float32).reshape(1)
        self.scale = torch.as_tensor(scale, dtype=torch.float32).reshape(1)
        self.low = torch.as_tensor(low, dtype=torch.float32).reshape(1)
        self.high = torch.as_tensor(high, dtype=torch.float32).reshape(1)
        self.max_tries = int(max_tries)

        self.base = Normal(self.loc, self.scale)

        self._event_shape = torch.Size([1])
        self._batch_shape = torch.Size([])

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def batch_shape(self):
        return self._batch_shape

    def sample(self, sample_shape=torch.Size()):
        n = int(torch.tensor(sample_shape).prod()) if len(sample_shape) else 1
        out = torch.empty((n,), dtype=torch.float32)
        filled = 0
        tries = 0

        # Oversample in chunks to keep it reasonably fast.
        while filled < n and tries < self.max_tries:
            tries += 1
            k = max(256, 2 * (n - filled))
            x = self.base.sample((k,))  # [k, 1]
            x = x[..., 0]
            mask = (x >= self.low.item()) & (x <= self.high.item())
            accepted = x[mask]
            if accepted.numel() == 0:
                continue
            take = min(accepted.numel(), n - filled)
            out[filled:filled + take] = accepted[:take]
            filled += take

        if filled < n:
            raise RuntimeError(
                "TruncatedNormal1D.sample: rejection sampler exhausted; "
                "check bounds/scale." 
            )

        return out.reshape(sample_shape + self.event_shape)

    def log_prob(self, value):
        # value expected shape [..., 1]
        x = value[..., 0]

        # Outside support -> -inf
        valid = (x >= self.low.to(x.device)[0]) & (x <= self.high.to(x.device)[0])
        logp = torch.full_like(x, float("-inf"))

        # Normalization constant Z = Phi((b-m)/s) - Phi((a-m)/s)
        a = (self.low.to(x.device)[0] - self.loc.to(x.device)[0]) / self.scale.to(x.device)[0]
        b = (self.high.to(x.device)[0] - self.loc.to(x.device)[0]) / self.scale.to(x.device)[0]

        # torch Normal.cdf exists; fall back to erf if needed.
        try:
            Z = (Normal(0.0, 1.0).cdf(b) - Normal(0.0, 1.0).cdf(a)).clamp_min(1e-12)
        except Exception:
            def _phi(z):
                return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
            Z = (_phi(b) - _phi(a)).clamp_min(1e-12)

        lp_valid = self.base.log_prob(x.unsqueeze(-1))[..., 0] - torch.log(Z)
        logp[valid] = lp_valid[valid]
        return logp

    def to(self, device):
        self.loc = self.loc.to(device)
        self.scale = self.scale.to(device)
        self.low = self.low.to(device)
        self.high = self.high.to(device)
        self.base = Normal(self.loc, self.scale)
        return self


def build_log_uniform(low: float, high: float) -> TransformedDistribution:
    """Return a distribution with log(x) ~ Uniform(log(low), log(high))."""
    low_t = torch.tensor([float(low)], dtype=torch.float32)
    high_t = torch.tensor([float(high)], dtype=torch.float32)
    base = Uniform(torch.log(low_t), torch.log(high_t))
    return TransformedDistribution(base, [ExpTransform()])


def build_gower_paper_known_priors():
    """Analytic priors from the paper, in *physical* parameter units."""
    return {
        # w ~ N(-1, 1/3) truncated to [-1, -1/3]
        "w0": TruncatedNormal1D(loc=-1.0, scale=1.0 / 3.0, low=-1.0, high=-1. / 3.0),
        # log(mnu) ~ U[log(0.06), log(0.14)] after dropping the first 192 runs
        "mnu": build_log_uniform(0.06, 0.14),
        # Gaussian priors
        "h": Normal(torch.tensor([0.7022]), torch.tensor([0.0245])),
        "ns": Normal(torch.tensor([0.9649]), torch.tensor([0.0063])),
        "ombh2": Normal(torch.tensor([0.02237]), torch.tensor([0.00015])),
        # IA priors
        "a_ia": Uniform(torch.tensor([4.48]), torch.tensor([7.0])),
        "b_ia": Uniform(torch.tensor([0.28]), torch.tensor([0.6])),
    }


def build_s8_box_known_priors(
    *,
    pivot_omega_m: float = 0.3,
    cosmo_parameter_order=("sigma_8", "omega_m", "ombh2", "h"),
):
    """Analytic priors in *physical* units with flat boxes in (S8, ωc, ωb) and h.

    Requested prior specification:
      - S8 in [0.5, 1.0]
      - ωc ≡ Ωc h^2 in [0.051, 0.255]
      - ωb ≡ Ωb h^2 (= ombh2) in [0.019, 0.026]
      - ΩK (= omega_k) fixed to 0.0
      - ns in [0.84, 1.10]
      - h in [0.64, 0.82]
      - w0 flat in [-1, -1/3]
      - mnu flat in [0.06, 0.14]
      - (a_ia, b_ia) joint multivariate normal using (mu_*, cov) defined below.

    Notes
    -----
    - The (sigma_8, omega_m, ombh2, h) prior is returned as a *joint* 4D
      distribution because it is not factorisable when you impose flat boxes
      in (S8, ωc, ωb).
    - Joint blocks are keyed by a tuple of parameter names.
    """

    cosmo_parameter_order = tuple(cosmo_parameter_order)
    expected = {"sigma_8", "omega_m", "ombh2", "h"}
    if set(cosmo_parameter_order) != expected or len(cosmo_parameter_order) != 4:
        raise ValueError(
            "build_s8_box_known_priors: cosmo_parameter_order must be a permutation "
            f"of {sorted(list(expected))}; got {cosmo_parameter_order!r}"
        )

    ia_base = MultivariateNormal(
        torch.tensor(mean, dtype=torch.float32),
        torch.tensor(cov, dtype=torch.float32),
    )

    cosmo_joint = S8OmegaCH2OmegaBH2HPriorToModelParams(
        s8_low=0.5,
        s8_high=0.9,
        oc_low = 0.051,
        oc_high = 0.18,
        ob_low = 0.022,
        ob_high = 0.0228,
        h_low = 0.64,
        h_high= 0.78,
        pivot_omega_m=float(pivot_omega_m),
        parameter_order=cosmo_parameter_order,
    )

    return {
        cosmo_parameter_order: cosmo_joint,
        "ns": Uniform(torch.tensor([0.948]), torch.tensor([0.984])),
        "w0": Uniform(torch.tensor([-1.0]), torch.tensor([-1.0 / 3.0])),
        "mnu": Uniform(torch.tensor([0.06]), torch.tensor([0.14])),
        ("a_ia", "b_ia"): ia_base,
    }
class TorchKDE1D(Distribution):
    arg_constraints = {}
    support = constraints.unit_interval
    has_rsample = False

    def __init__(self, samples, min_val, max_val):
        super().__init__(validate_args=False)

        samples = np.asarray(samples, dtype=np.float64)

        self.min = float(min_val)
        self.max = float(max_val)

        span = self.max - self.min
        if span == 0:
            span = 1.0

        samples_norm = (samples - self.min) / span

        self.kde = gaussian_kde(samples_norm)

        self._event_shape = torch.Size([1])
        self._batch_shape = torch.Size([])

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def batch_shape(self):
        return self._batch_shape

    def sample(self, sample_shape=torch.Size()):
        n = int(torch.tensor(sample_shape).prod()) if len(sample_shape) else 1

        s = self.kde.resample(n)[0]
        s = np.clip(s, 0.0, 1.0)

        x = torch.tensor(s, dtype=torch.float32)

        return x.reshape(sample_shape + self.event_shape)

    def log_prob(self, value):
        x = value[..., 0].detach().cpu().numpy().reshape(-1)
        lp = np.log(self.kde(x))
        return torch.tensor(lp, dtype=torch.float32).reshape(value.shape[:-1])

    def to(self, device):
        return self

from torch.distributions import MultivariateNormal


def build_scaled_joint_gaussian(names, mean, cov, scaler):

    name_to_idx = {
        n: i for i, n in enumerate(scaler.parameter_names)
    }

    mins = []
    maxs = []

    for n in names:
        idx = name_to_idx[n]
        mins.append(scaler.min[idx])
        maxs.append(scaler.max[idx])

    base = MultivariateNormal(
        torch.tensor(mean, dtype=torch.float32),
        torch.tensor(cov, dtype=torch.float32),
    )

    return ScaledMVNDistribution(
        base,
        mins,
        maxs,
    )
mu_AIA = 5.74
mu_beta = 0.44

sigma_AIA = 0.29
sigma_beta = 0.03
rho = -0.59

cov = [
    [sigma_AIA**2, rho * sigma_AIA * sigma_beta],
    [rho * sigma_AIA * sigma_beta, sigma_beta**2],
]

mean = [mu_AIA, mu_beta]

class ScaledDistribution(Distribution):
    arg_constraints = {}
    support = constraints.unit_interval
    has_rsample = False

    def __init__(self, base_dist, min_val, max_val):
        super().__init__(validate_args=False)

        self.base = base_dist
        self.min = float(min_val)
        self.max = float(max_val)
        self.span = self.max - self.min

        if self.span == 0:
            self.span = 1.0

        self._event_shape = torch.Size([1])
        self._batch_shape = torch.Size([])

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def batch_shape(self):
        return self._batch_shape

    def sample(self, sample_shape=torch.Size()):
        x = self.base.sample(sample_shape)

        x = (x - self.min) / self.span

        return x

    def log_prob(self, value):
        x = value * self.span + self.min

        lp = self.base.log_prob(x)

        # Jacobian
        lp = lp + torch.log(torch.tensor(self.span))

        return lp

    def to(self, device):
        return self

from torch.distributions import Distribution, constraints
import torch


class ScaledMVNDistribution(Distribution):
    arg_constraints = {}
    support = constraints.real
    has_rsample = False

    def __init__(self, base_dist, mins, maxs):
        super().__init__(validate_args=False)

        self.base = base_dist

        self.mins = torch.tensor(mins, dtype=torch.float32)
        self.maxs = torch.tensor(maxs, dtype=torch.float32)

        self.spans = self.maxs - self.mins
        self.spans[self.spans == 0] = 1.0

        self.log_det = torch.log(self.spans).sum()

        self._event_shape = torch.Size([len(mins)])
        self._batch_shape = torch.Size([])

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def batch_shape(self):
        return self._batch_shape

    def sample(self, sample_shape=torch.Size()):
        x = self.base.sample(sample_shape)

        x = (x - self.mins) / self.spans

        return x

    def log_prob(self, value):

        x = value * self.spans + self.mins

        lp = self.base.log_prob(x)

        lp = lp + self.log_det

        return lp

    def to(self, device):
        self.mins = self.mins.to(device)
        self.maxs = self.maxs.to(device)
        self.spans = self.spans.to(device)
        self.log_det = self.log_det.to(device)
        return self


class ScaledJointDistribution(Distribution):
    """Scale an arbitrary joint Distribution into [0,1]^D using affine min/max.

    If x_phys in R^D and x_scaled = (x_phys - mins) / spans,
    then log p_scaled(x_scaled) = log p_phys(x_phys) + sum(log(spans)).
    """

    arg_constraints = {}
    support = constraints.unit_interval
    has_rsample = False

    def __init__(self, base_dist: Distribution, mins, maxs):
        super().__init__(validate_args=False)

        self.base = base_dist

        mins_t = torch.as_tensor(mins, dtype=torch.float32)
        maxs_t = torch.as_tensor(maxs, dtype=torch.float32)
        if mins_t.ndim != 1 or maxs_t.ndim != 1 or mins_t.shape != maxs_t.shape:
            raise ValueError("ScaledJointDistribution: mins/maxs must be 1D tensors with the same shape")

        self.mins = mins_t
        self.maxs = maxs_t
        self.spans = self.maxs - self.mins
        self.spans[self.spans == 0] = 1.0
        self.log_det = torch.log(self.spans).sum()

        self._event_shape = base_dist.event_shape
        self._batch_shape = base_dist.batch_shape

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def batch_shape(self):
        return self._batch_shape

    def sample(self, sample_shape=torch.Size()):
        x = self.base.sample(sample_shape)
        return (x - self.mins.to(x.device)) / self.spans.to(x.device)

    def log_prob(self, value):
        v = value
        # elementwise unit hypercube check
        in_cube = ((v >= 0.0) & (v <= 1.0)).all(dim=-1)
        out = torch.full(v.shape[:-1], float("-inf"), device=v.device, dtype=v.dtype)
        if not torch.any(in_cube):
            return out

        x = v * self.spans.to(v.device) + self.mins.to(v.device)
        lp = self.base.log_prob(x) + self.log_det.to(v.device)
        out[in_cube] = lp[in_cube]
        return out

    def to(self, device):
        self.mins = self.mins.to(device)
        self.maxs = self.maxs.to(device)
        self.spans = self.spans.to(device)
        self.log_det = self.log_det.to(device)
        if hasattr(self.base, "to"):
            self.base = self.base.to(device)
        return self

def build_kde_prior_from_df(
    df,
    columns,
    scaler,
    extra_priors=None,
):
    if MultipleIndependent is None:
        raise ModuleNotFoundError("build_kde_prior_from_df requires 'sbi' (MultipleIndependent).")
    dists = []

    name_to_idx = {
        n: i for i, n in enumerate(scaler.parameter_names)
    }

    for c in columns:
        idx = name_to_idx[c]

        min_v = scaler.min[idx]
        max_v = scaler.max[idx]

        d = TorchKDE1D(
            df[c].values,
            min_v,
            max_v,
        )

        dists.append(d)

    if extra_priors is not None:
        for name, dist in extra_priors.items():

            idx = name_to_idx[name]

            min_v = scaler.min[idx]
            max_v = scaler.max[idx]

            d = ScaledDistribution(
                dist,
                min_v,
                max_v,
            )

            dists.append(d)

    return MultipleIndependent(dists)

from src.cosmology.gower_street import GowerStPrior

def build_gower_st_prior(
    variables,
    scaler,
    csv_path,
    drop_first=192,
    n_samples=5000,
    extra_priors=None,
):

    gower_prior_builder = GowerStPrior.from_csv(
        csv_path,
        drop_first=drop_first,
    )

    res = gower_prior_builder.sample(n_samples)

    gower_prior = build_kde_prior_from_df(
        res,
        columns=variables,
        scaler=scaler,
        extra_priors=extra_priors,
    )

    return gower_prior

SAVE_PATH = "./data/gower_prior.pkl"

from ..models.custom_sbi import NeuralSplineFlow

def train_or_load_gower_prior(
    csv_path,
    variables,
    scaler,
    drop_first=192,
    device="cpu",
    epochs=2000,
    lr=1e-3,
    batch_size=128,
    val_fraction=0.2,
    patience=50,
    retrain=False,
    save_path=None,
):

    # -------------------------
    # load if exists
    # -------------------------
    
    if save_path is None:
        # Avoid collisions when training flows for different variable subsets.
        safe_name = "_".join([str(v) for v in variables])
        save_path = f"./data/gower_prior_{safe_name}.pkl"

    if os.path.exists(save_path) and not retrain:
        with open(save_path, "rb") as f:
            flow = pickle.load(f)
        flow.to(device)
        return flow

    # -------------------------
    # load dataset
    # -------------------------

    gower = GowerStPrior.from_csv(
        csv_path,
        drop_first=drop_first,
    )

    data = [gower.series_true[v] for v in variables]

    theta_np = np.stack(data, axis=1)

    # -------------------------
    # scale
    # -------------------------
    # Keep existing scaling if provided (e.g. preset min/max). Only fit if
    # min/max are missing.
    if getattr(scaler, "min", None) is None or getattr(scaler, "max", None) is None:
        scaler.fit(theta_np)

    theta_scaled = scaler.transform(theta_np)

    theta = torch.tensor(
        theta_scaled,
        dtype=torch.float32,
        device=device,
    )

    dim = theta.shape[1]

    # -------------------------
    # train / val split
    # -------------------------

    N = len(theta)

    perm = torch.randperm(N)

    n_val = int(val_fraction * N)

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    theta_train = theta[train_idx]
    theta_val = theta[val_idx]

    # -------------------------
    # build flow
    # -------------------------

    # flow = MaskedAutoregressiveFlow(
    #     features=dim,
    #     hidden_features=16,
    #     num_layers=5,
    #     num_blocks_per_layer=2,
    # ).to(device)
    flow = NeuralSplineFlow(
        features=dim,
        hidden_features=32,
        num_layers=4,
        num_blocks_per_layer=2,
    ).to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)

    # -------------------------
    # early stopping state
    # -------------------------

    best_val = np.inf
    best_state = None
    epochs_no_improve = 0

    # -------------------------
    # training loop
    # -------------------------

    for epoch in range(epochs):

        flow.train()

        perm = torch.randperm(len(theta_train), device=device)

        for i in range(0, len(theta_train), batch_size):

            idx = perm[i:i + batch_size]

            batch = theta_train[idx]

            loss = -flow.log_prob(batch).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # -----------------
        # validation
        # -----------------

        flow.eval()

        with torch.no_grad():

            val_loss = -flow.log_prob(theta_val).mean().item()

        print(
            f"epoch {epoch} "
            f"train {loss.item():.4f} "
            f"val {val_loss:.4f}"
        )

        # -----------------
        # best checkpoint
        # -----------------

        if val_loss < best_val:

            best_val = val_loss
            best_state = flow.state_dict()
            epochs_no_improve = 0

        else:

            epochs_no_improve += 1

        # -----------------
        # early stopping
        # -----------------

        if epochs_no_improve >= patience:

            print("Early stopping")
            break

    # -------------------------
    # restore best model
    # -------------------------

    if best_state is not None:
        flow.load_state_dict(best_state)

    # -------------------------
    # save
    # -------------------------

    os.makedirs("./data", exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(flow, f)

    return flow


import torch
import numpy as np
from torch.distributions import Distribution, constraints

class NFlowDistribution(Distribution):
    arg_constraints = {}
    support = constraints.real
    has_rsample = False

    def __init__(self, flow, dims):
        """
        Wraps an unconditional nflows.Flow into a standard PyTorch Distribution.
        
        Args:
            flow: The trained nflows Flow object.
            dims: Dimension of the flow space.
            mins: Min scaling bounds (for denormalizing the sample).
            maxs: Max scaling bounds (for denormalizing the sample).
        """
        super().__init__(validate_args=False)

        self.flow = flow
        self.dims = dims
        self.log_det = torch.tensor(0.0)  # If your flow includes scaling, you may need to set this to the appropriate log determinant of the scaling transform.
        self._event_shape = torch.Size([dims])
        self._batch_shape = torch.Size([])

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def batch_shape(self):
        return self._batch_shape

    def sample(self, sample_shape=torch.Size()):
        # Determine number of samples
        n = int(torch.tensor(sample_shape).prod()) if len(sample_shape) else 1

        # nflows returns tensor of shape (n, dims)
        x = self.flow.sample(n)

        return x.reshape(sample_shape + self.event_shape)

    def log_prob(self, x):
        """
        Compute log probability of `x` under the flow, handling batches
        and masking invalid inputs.
        """
        # Save original shape
        original_shape = x.shape
        batch_flat = x.reshape(-1, self.dims)

        # Mask invalid values outside [0,1]
        valid_mask = ((batch_flat >= 0.0) & (batch_flat <= 1.0)).all(dim=1)

        # Initialize log-probabilities with -inf for invalid entries
        logp_flat = torch.full((batch_flat.shape[0],), float('-inf'), device=x.device)

        logp_flat[valid_mask] = self.flow.log_prob(batch_flat[valid_mask])

        # Reshape back to original batch shape (excluding last dim)
        logp = logp_flat.reshape(original_shape[:-1])

        # Add any log-det Jacobian if needed
        logp = logp + self.log_det

        return logp

    def to(self, device):
        self.flow = self.flow.to(device)
        self.log_det = self.log_det.to(device)
        return self

def build_flow_with_extras_prior(
    flow,
    columns, # names of the variables the flow was trained on
    scaler,
    extra_priors=None, # dict e.g. {"a_ia": Uniform(...), "b_ia": Uniform(...)}
    return_restricted=False
):
    if MultipleIndependent is None:
        raise ModuleNotFoundError("build_flow_with_extras_prior requires 'sbi' (MultipleIndependent).")
    if return_restricted and RestrictedPrior is None:
        raise ModuleNotFoundError("build_flow_with_extras_prior(return_restricted=True) requires 'sbi' (RestrictedPrior).")
    name_to_idx = {n: i for i, n in enumerate(scaler.parameter_names)}
    
    # 2. Add the Flow as the first joint distribution block
    flow_dist = NFlowDistribution(
        flow=flow,
        dims=len(columns),
    )
    dists = [flow_dist]

    # 3. Scale and append the extra independent priors
    if extra_priors is not None:
        for name, dist in extra_priors.items():
            idx = name_to_idx[name]
            
            d = ScaledDistribution(
                dist,
                min_val=scaler.min[idx],
                max_val=scaler.max[idx],
            )
            dists.append(d)

    joint_prior = MultipleIndependent(dists)

    if not return_restricted:
        return joint_prior
    else: # only if we want to sample, does not work with sbi for some reason
        # 4. Define the restriction function
        # This checks if every dimension of a sample is within [0, 1]
        def is_within_unit_hypercube(theta):
            # theta shape: (batch_size, total_dims)
            return torch.all((theta >= 0) & (theta <= 1), dim=-1)

        # 5. Wrap in RestrictedPrior
        # 'rejection' is the default sampling method
        restricted_prior = RestrictedPrior(
            prior=joint_prior, 
            accept_reject_fn=is_within_unit_hypercube,
            sample_with='rejection'
        )
    # --- THE FIX: Add the missing .to() method ---
        def to_device(self, device):
            self._prior.to(device)
            self._device = torch.device(device)
            return self

        # Bind the function to our specific instance
        import types
        restricted_prior.to = types.MethodType(to_device, restricted_prior)
        # ---------------------------------------------

        return restricted_prior


class PermutedDistribution(Distribution):
    """Wrap a joint distribution and expose a permuted event dimension order.

    This is primarily used to build priors from convenient *blocks* (e.g.
    a 4D cosmology joint) while presenting an interface compatible with the
    rest of the codebase, which passes `theta` vectors in the order given by
    `params`.

    If `enforce_unit_hypercube` is True, `log_prob` returns `-inf` whenever
    any component lies outside [0, 1].
    """

    arg_constraints = {}
    support = constraints.unit_interval
    has_rsample = False

    def __init__(
        self,
        base: Distribution,
        *,
        base_order,
        target_order,
        enforce_unit_hypercube: bool = True,
    ):
        super().__init__(validate_args=False)

        self.base = base
        self.base_order = list(base_order)
        self.target_order = list(target_order)
        self.enforce_unit_hypercube = bool(enforce_unit_hypercube)

        if sorted(self.base_order) != sorted(self.target_order):
            raise ValueError(
                "PermutedDistribution: base_order and target_order must contain the same names. "
                f"Got base_order={self.base_order!r}, target_order={self.target_order!r}"
            )

        # Indices to map between orders.
        # - target -> base: value_target[..., idx_target_to_base] gives base-ordered vector.
        self._idx_target_to_base = torch.tensor(
            [self.target_order.index(n) for n in self.base_order],
            dtype=torch.long,
        )
        # - base -> target: sample_base[..., idx_base_to_target] gives target-ordered vector.
        self._idx_base_to_target = torch.tensor(
            [self.base_order.index(n) for n in self.target_order],
            dtype=torch.long,
        )

        self._event_shape = torch.Size([len(self.target_order)])
        self._batch_shape = base.batch_shape

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def batch_shape(self):
        return self._batch_shape

    def sample(self, sample_shape=torch.Size()):
        x_base = self.base.sample(sample_shape)
        return x_base[..., self._idx_base_to_target.to(x_base.device)]

    def log_prob(self, value):
        v = value
        if v.shape[-1] != len(self.target_order):
            raise ValueError(
                f"PermutedDistribution.log_prob: expected last dim {len(self.target_order)}, got {v.shape[-1]}"
            )

        if self.enforce_unit_hypercube:
            in_cube = ((v >= 0.0) & (v <= 1.0)).all(dim=-1)
            out = torch.full(v.shape[:-1], float("-inf"), device=v.device, dtype=v.dtype)
            if not torch.any(in_cube):
                return out

            v_base = v[..., self._idx_target_to_base.to(v.device)]
            lp = self.base.log_prob(v_base)
            out[in_cube] = lp[in_cube]
            return out

        v_base = v[..., self._idx_target_to_base.to(v.device)]
        return self.base.log_prob(v_base)

    def to(self, device):
        self._idx_target_to_base = self._idx_target_to_base.to(device)
        self._idx_base_to_target = self._idx_base_to_target.to(device)
        if hasattr(self.base, "to"):
            maybe = self.base.to(device)
            # Some distributions (notably from sbi) mutate in-place and return None.
            if maybe is not None:
                self.base = maybe
        return self

def build_analytic_prior(
    params,
    scaler,
    *,
    pivot_omega_m: float = 0.3,
    return_restricted: bool = False,
):
    """Build an analytic prior in *scaled* space ([0,1]^D).

    This mirrors the *shape* of `build_flow_with_extras_prior` but replaces the
    flow block with the analytic transformed prior implied by a flat box in
    (S8, ωc≡Ωc h^2, ωb≡Ωb h^2, h).

    Parameters
    ----------
    params : sequence[str]
        Parameter names (used to set block ordering).
    scaler : object
        Must provide `parameter_names`, `min`, `max` (as in MinMaxScaler).

    Returns
    -------
    torch.distributions.Distribution
        A joint distribution over scaled parameters in **the same order as**
        `params`.
    """

    if MultipleIndependent is None:
        raise ModuleNotFoundError("build_analytic_prior requires 'sbi' (MultipleIndependent).")

    params = list(params)
    name_to_idx = {n: i for i, n in enumerate(scaler.parameter_names)}

    cosmo_set = {"omega_m", "sigma_8", "ombh2", "h"}
    cosmo_block = [p for p in params if p in cosmo_set]
    if set(cosmo_block) != cosmo_set:
        missing = sorted(list(cosmo_set - set(cosmo_block)))
        raise ValueError(f"build_analytic_prior: missing required cosmo params {missing} in params={params}")

    # Reuse the defaults in build_s8_box_known_priors (physical-space priors)
    # and request the cosmo joint block in the same order as `params`.
    phys_priors = build_s8_box_known_priors(
        pivot_omega_m=float(pivot_omega_m),
        cosmo_parameter_order=tuple(cosmo_block),
    )

    phys_cosmo = phys_priors[tuple(cosmo_block)]

    cosmo_mins = [float(scaler.min[name_to_idx[p]]) for p in cosmo_block]
    cosmo_maxs = [float(scaler.max[name_to_idx[p]]) for p in cosmo_block]
    scaled_cosmo = ScaledJointDistribution(phys_cosmo, cosmo_mins, cosmo_maxs)

    # Remaining priors (in physical space) then scaled into [0,1].
    one_d_priors = {
        "ns": phys_priors["ns"],
        "w0": phys_priors["w0"],
        "mnu": phys_priors["mnu"],
    }

    ia_names = [p for p in params if p in {"a_ia", "b_ia"}]
    if len(ia_names) == 1:
        raise ValueError(
            "build_analytic_prior: if using IA params, both 'a_ia' and 'b_ia' must be present. "
            f"Got ia_names={ia_names!r} in params={params!r}"
        )
    ia_block = tuple(ia_names) if len(ia_names) == 2 else None

    dists = [scaled_cosmo]
    internal_order = list(cosmo_block)

    used = set(cosmo_block)
    used.update(ia_names)

    for p in params:
        if p in used:
            continue
        if p not in one_d_priors:
            raise ValueError(
                f"build_analytic_prior: no analytic prior specified for parameter {p!r}. "
                f"Handled: {sorted(list(cosmo_set | set(one_d_priors) | {'a_ia','b_ia'}))}"
            )
        idx = name_to_idx[p]
        dists.append(ScaledDistribution(one_d_priors[p], scaler.min[idx], scaler.max[idx]))
        internal_order.append(p)

    if ia_block is not None:
        # build_s8_box_known_priors stores IA as (a_ia, b_ia); permute if needed
        ia_base_ab = phys_priors[("a_ia", "b_ia")]

        if ia_block == ("a_ia", "b_ia"):
            ia_base = ia_base_ab
        elif ia_block == ("b_ia", "a_ia"):
            perm = torch.tensor([1, 0], dtype=torch.long)
            mean_perm = ia_base_ab.mean[perm]
            cov_perm = ia_base_ab.covariance_matrix[perm][:, perm]
            ia_base = MultivariateNormal(mean_perm, cov_perm)
        else:
            raise ValueError(
                f"build_analytic_prior: unexpected IA ordering {ia_block!r}"
            )

        ia_mins = [float(scaler.min[name_to_idx[p]]) for p in ia_block]
        ia_maxs = [float(scaler.max[name_to_idx[p]]) for p in ia_block]
        dists.append(ScaledMVNDistribution(ia_base, ia_mins, ia_maxs))
        internal_order.extend(list(ia_block))

    joint_prior = MultipleIndependent(dists)

    # Expose prior in `params` order (important for inference/MCMC).
    base_to_wrap = joint_prior
    if return_restricted:
        if RestrictedPrior is None:
            raise ModuleNotFoundError(
                "build_analytic_prior(return_restricted=True) requires 'sbi' (RestrictedPrior)."
            )

        def is_within_unit_hypercube(theta):
            return torch.all((theta >= 0) & (theta <= 1), dim=-1)

        restricted_prior = RestrictedPrior(
            prior=joint_prior,
            accept_reject_fn=is_within_unit_hypercube,
            sample_with="rejection",
        )

        # Provide .to() like in build_flow_with_extras_prior
        import types

        def to_device(self, device):
            self._prior.to(device)
            self._device = torch.device(device)
            return self

        restricted_prior.to = types.MethodType(to_device, restricted_prior)
        base_to_wrap = restricted_prior

    return PermutedDistribution(
        base_to_wrap,
        base_order=internal_order,
        target_order=params,
        enforce_unit_hypercube=True,
    )