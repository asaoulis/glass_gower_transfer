import torch
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sbi.utils import process_prior
import torch
import numpy as np
from scipy.stats import gaussian_kde
from torch.distributions import Distribution, constraints
import torch
import numpy as np
from scipy.stats import gaussian_kde
from torch.distributions import Distribution, constraints
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
from sbi.utils import MultipleIndependent, process_prior

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

from sbi.utils import MultipleIndependent


def build_kde_prior_from_df(
    df,
    columns,
    scaler,
    extra_priors=None,
):
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