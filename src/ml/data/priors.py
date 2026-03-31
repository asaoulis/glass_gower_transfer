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
import os
import pickle
import torch
import numpy as np
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
    retrain=False
):

    # -------------------------
    # load if exists
    # -------------------------
    
    if os.path.exists(SAVE_PATH) and not retrain:
        with open(SAVE_PATH, "rb") as f:
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

    with open(SAVE_PATH, "wb") as f:
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

from sbi.utils import MultipleIndependent,RestrictedPrior

def build_flow_with_extras_prior(
    flow,
    columns, # names of the variables the flow was trained on
    scaler,
    extra_priors=None, # dict e.g. {"a_ia": Uniform(...), "b_ia": Uniform(...)}
    return_restricted=False
):
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