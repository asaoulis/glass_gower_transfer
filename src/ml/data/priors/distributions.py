import math

import numpy as np
import torch
from scipy.stats import gaussian_kde
from torch.distributions import Distribution, constraints
from torch.distributions import MultivariateNormal, Normal, TransformedDistribution, Uniform
from torch.distributions.transforms import ExpTransform


class S8OmegaCH2OmegaBH2HPriorToModelParams(Distribution):
    """Joint prior in model parameter space implied by a flat box in (S8, ωc, ωb, h)."""

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
    """1D truncated normal distribution."""

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

        while filled < n and tries < self.max_tries:
            tries += 1
            k = max(256, 2 * (n - filled))
            x = self.base.sample((k,))[..., 0]
            mask = (x >= self.low.item()) & (x <= self.high.item())
            accepted = x[mask]
            if accepted.numel() == 0:
                continue
            take = min(accepted.numel(), n - filled)
            out[filled:filled + take] = accepted[:take]
            filled += take

        if filled < n:
            raise RuntimeError(
                "TruncatedNormal1D.sample: rejection sampler exhausted; check bounds/scale."
            )

        return out.reshape(sample_shape + self.event_shape)

    def log_prob(self, value):
        x = value[..., 0]

        valid = (x >= self.low.to(x.device)[0]) & (x <= self.high.to(x.device)[0])
        logp = torch.full_like(x, float("-inf"))

        a = (self.low.to(x.device)[0] - self.loc.to(x.device)[0]) / self.scale.to(x.device)[0]
        b = (self.high.to(x.device)[0] - self.loc.to(x.device)[0]) / self.scale.to(x.device)[0]

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
    """Analytic priors from the paper, in physical parameter units."""
    return {
        "w0": TruncatedNormal1D(loc=-1.0, scale=1.0 / 3.0, low=-1.0, high=-1.0 / 3.0),
        "mnu": build_log_uniform(0.06, 0.14),
        "h": Normal(torch.tensor([0.7022]), torch.tensor([0.0245])),
        "ns": Normal(torch.tensor([0.9649]), torch.tensor([0.0063])),
        "ombh2": Normal(torch.tensor([0.02237]), torch.tensor([0.00015])),
        "a_ia": Uniform(torch.tensor([4.48]), torch.tensor([7.0])),
        "b_ia": Uniform(torch.tensor([0.28]), torch.tensor([0.6])),
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
        return (x - self.min) / self.span

    def log_prob(self, value):
        x = value * self.span + self.min
        lp = self.base.log_prob(x)
        return lp + torch.log(torch.tensor(self.span, device=lp.device, dtype=lp.dtype))

    def to(self, device):
        return self


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
        return (x - self.mins) / self.spans

    def log_prob(self, value):
        x = value * self.spans + self.mins
        return self.base.log_prob(x) + self.log_det.to(x.device)

    def to(self, device):
        self.mins = self.mins.to(device)
        self.maxs = self.maxs.to(device)
        self.spans = self.spans.to(device)
        self.log_det = self.log_det.to(device)
        return self


class ScaledJointDistribution(Distribution):
    """Scale an arbitrary joint Distribution into [0,1]^D using affine min/max."""

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


class NFlowDistribution(Distribution):
    arg_constraints = {}
    support = constraints.real
    has_rsample = False

    def __init__(self, flow, dims, max_tries: int = 10_000):
        super().__init__(validate_args=False)

        self.flow = flow
        self.dims = dims
        self.max_tries = int(max_tries)
        self.log_det = torch.tensor(0.0)
        self._event_shape = torch.Size([dims])
        self._batch_shape = torch.Size([])

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def batch_shape(self):
        return self._batch_shape

    def sample(self, sample_shape=torch.Size()):
        # REJECTION-RESAMPLE to the flow's own declared support. The empirical Gower flow is a
        # normalising flow over the SCALED ([0,1]^d) parameter box, so it is unbounded and leaks
        # ~1 % of draws outside the box -- while `log_prob` below hard-masks those to -inf. Without
        # this loop `sample()` and `log_prob()` disagree, and the joint prior assigns ZERO density
        # to ~1 % of the samples it drew itself (measured 0.9901 finite on 20k draws, identical at
        # kappa=1/kappa=2 and with no b_g params at all -- i.e. this is a property of the flow, not
        # of any analytic marginal). Mirrors TruncatedNormal1D.sample.
        # NOTE: `log_prob` is deliberately left untouched, so it remains unnormalised by
        # log Z ~ -0.01 nats (a constant: it cancels in MCMC and is negligible for dMI).
        n = int(torch.tensor(sample_shape).prod()) if len(sample_shape) else 1
        if n == 0:  # e.g. sample_shape=(0,); the pre-fix code returned an empty tensor here
            return torch.empty(sample_shape + self.event_shape)

        chunks = []
        filled = 0
        tries = 0
        while filled < n and tries < self.max_tries:
            tries += 1
            k = max(256, 2 * (n - filled))
            x = self.flow.sample(k).reshape(-1, self.dims)
            accepted = x[((x >= 0.0) & (x <= 1.0)).all(dim=-1)]
            if accepted.numel() == 0:
                continue
            take = min(accepted.shape[0], n - filled)
            chunks.append(accepted[:take])
            filled += take

        if filled < n:
            raise RuntimeError(
                "NFlowDistribution.sample: rejection sampler exhausted after "
                f"{tries} tries ({filled}/{n} accepted); the flow is placing almost no mass "
                "inside the [0, 1]^d scaled box it was trained on."
            )

        x = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        return x.reshape(sample_shape + self.event_shape)

    def log_prob(self, x):
        original_shape = x.shape
        batch_flat = x.reshape(-1, self.dims)
        valid_mask = ((batch_flat >= 0.0) & (batch_flat <= 1.0)).all(dim=1)
        logp_flat = torch.full((batch_flat.shape[0],), float("-inf"), device=x.device)
        # GUARD: nflows cannot evaluate an EMPTY batch -- a coupling transform does
        # `transform_params.reshape(b, d, -1)` and with b=0 that raises
        # "cannot reshape tensor of 0 elements into shape [0, 1, -1]". The vectorised slice
        # sampler proposes in unconstrained space, so a call in which EVERY chain sits outside
        # the scaled [0,1]^d box is rare but inevitable -- it killed the VD `_hf` r0 Stage-B job
        # (1351808) in its MCMC eval after all 150 training epochs had completed.
        # Those rows are already -inf by construction, so skipping the flow call changes no
        # finite value on any existing path.
        if valid_mask.any():
            logp_flat[valid_mask] = self.flow.log_prob(batch_flat[valid_mask])
        logp = logp_flat.reshape(original_shape[:-1])
        return logp + self.log_det.to(x.device)

    def to(self, device):
        self.flow = self.flow.to(device)
        self.log_det = self.log_det.to(device)
        return self


class PermutedDistribution(Distribution):
    """Wrap a joint distribution and expose a permuted event dimension order."""

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

        self._idx_target_to_base = torch.tensor(
            [self.target_order.index(n) for n in self.base_order],
            dtype=torch.long,
        )
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
            if maybe is not None:
                self.base = maybe
        return self
