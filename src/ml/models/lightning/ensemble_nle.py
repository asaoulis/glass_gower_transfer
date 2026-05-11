from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from tqdm import tqdm

from sbi import utils as sbi_utils

from .estimators import PatchedLikelihoodEstimator
from .utils import _move_nested_to_device


class _EnsembleLikelihoodModel(nn.Module):
    """Averages member log-likelihoods for use inside MCMC potentials."""

    def __init__(self, members: list[pl.LightningModule], reduction: str = "logmeanexp"):
        super().__init__()
        if reduction not in {"mean_log_prob", "logmeanexp"}:
            raise ValueError("reduction must be one of {'mean_log_prob', 'logmeanexp'}")
        self.members = nn.ModuleList(members)
        self.reduction = reduction

    def log_prob(self, x, theta):
        log_probs = [m.forward(x, cond=theta) for m in self.members]
        stacked = torch.stack(log_probs, dim=0)

        if self.reduction == "logmeanexp":
            return torch.logsumexp(stacked, dim=0) - np.log(len(self.members))

        return stacked.mean(dim=0)


class EnsembleLikelihoodNDELightningModule(pl.LightningModule):
    """Evaluation-time ensemble for likelihood NDEs."""

    def __init__(self, members: list[pl.LightningModule]):
        super().__init__()
        if not members:
            raise ValueError(
                "EnsembleLikelihoodNDELightningModule requires at least one member."
            )
        self.members = nn.ModuleList(members)

        first = members[0]
        self.test_dataloader = getattr(first, "test_dataloader", None)
        self.loss_name = getattr(first, "loss_name", "log_prob")
        self.conditioning_dim = getattr(first, "conditioning_dim", None)
        self.inference_dim = getattr(first, "inference_dim", None)

    def _resolve_device(self):
        try:
            return self.device
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _validate_member_loaders(self):
        if self.test_dataloader is None:
            raise ValueError("EnsembleLikelihoodNDELightningModule.test_dataloader is None")

        for idx, m in enumerate(self.members):
            loader = getattr(m, "test_dataloader", None)
            if loader is None:
                raise ValueError(f"Ensemble member {idx} has no test_dataloader")
            if len(loader.dataset) != len(self.test_dataloader.dataset):
                raise ValueError(
                    "Ensemble members have different test dataset lengths; "
                    "cannot safely ensemble their likelihoods."
                )

    def to(self, *args, **kwargs):  # type: ignore[override]
        super().to(*args, **kwargs)
        for m in self.members:
            m.to(*args, **kwargs)
        return self

    def eval(self):  # type: ignore[override]
        super().eval()
        for m in self.members:
            m.eval()
        return self

    @torch.no_grad()
    def compute_avg_log_prob(self):
        self._validate_member_loaders()
        target_device = self._resolve_device()

        all_log_probs = []
        for batch in self.test_dataloader:
            x, theta = batch
            x = _move_nested_to_device(x, target_device)
            theta = _move_nested_to_device(theta, target_device)
            batch_lps = []
            for m in self.members:
                m.to(target_device)
                m.eval()
                batch_lps.append(m.forward(x, cond=theta))
            all_log_probs.append(torch.stack(batch_lps, dim=0).mean(dim=0).reshape(-1))

        return float(-torch.cat(all_log_probs, dim=0).mean().item())

    def build_posterior_object(self, prior=None, fixed_parameters=None, reduction: str = "logmeanexp"):
        first = self.members[0]
        device = self._resolve_device()

        for m in self.members:
            m.eval()
            if (
                hasattr(m, "model")
                and hasattr(m.model, "embedding_net")
                and hasattr(m.model.embedding_net, "only_return_mu")
            ):
                m.model.embedding_net.only_return_mu = True

        if prior is None:
            prior = sbi_utils.BoxUniform(
                low=0 * torch.ones(first.conditioning_dim, device=device),
                high=1.0 * torch.ones(first.conditioning_dim, device=device),
                device=device,
            )

        ensemble_likelihood = _EnsembleLikelihoodModel(
            list(self.members), reduction=reduction
        )
        likelihood_estimator = PatchedLikelihoodEstimator(
            model=ensemble_likelihood,
            prior=prior,
            input_shape=(first.inference_dim,),
            condition_shape=(first.conditioning_dim,),
            fixed_parameters=fixed_parameters,
        )
        return likelihood_estimator

    @torch.no_grad()
    def generate_samples(
        self,
        num_samples=2_000,
        num_jobs=36,
        backend="loky",
        prior=None,
        reduction: str = "logmeanexp",
        **mcmc_kwargs,
    ):
        posterior = self.build_posterior_object(
            prior=prior,
            reduction=reduction,
            fixed_parameters=mcmc_kwargs.pop("fixed_parameters", None),
        )

        posterior.to("cpu")
        posterior.prior.to("cpu")

        jobs = Parallel(
            n_jobs=num_jobs,
            backend=backend,
            return_as="generator",
        )(
            delayed(posterior.sample_single_batch)(
                num_samples,
                test_data,
                test_cosmo,
                dict(mcmc_kwargs),
            )
            for test_data, test_cosmo in self.test_dataloader
        )

        results = list(
            tqdm(
                jobs,
                total=len(self.test_dataloader),
                desc="Sampling ensemble batches",
            )
        )

        theta0s, samples = zip(*results)
        theta0s = torch.cat(theta0s, dim=0)
        samples = torch.cat(samples, dim=1)
        return theta0s, samples
