"""Evaluation-time ensemble of NLE LightningModules.

Averages member log-likelihoods inside the MCMC potential and samples with a joblib pool of
per-batch work items (_sample_shard).
"""
from __future__ import annotations

import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from tqdm import tqdm

try:  # joblib >= 1.3; the cluster env is 1.5.x, but never make the import a hard failure.
    from joblib import parallel_config
except ImportError:  # pragma: no cover
    from joblib import parallel_backend as _parallel_backend

    def parallel_config(backend="loky", **kwargs):
        return _parallel_backend(backend, **kwargs)

from sbi import utils as sbi_utils

from .estimators import PatchedLikelihoodEstimator
from .ensemble_flow import build_grouped_ensemble_likelihood
from .utils import _move_nested_to_device


def _sample_shard(posterior, num_samples, test_data, test_cosmo, mcmc_kwargs,
                  seed=None, threads=None):
    """One joblib work item: draw num_samples per event for one batch of events.

    Module-level so loky can pickle it. Thread count and seeds are set inside the worker: loky
    overrides thread limits in its children, and the slice sampler draws from the global numpy
    and torch RNGs of reused worker processes.
    """
    if threads:
        torch.set_num_threads(int(threads))
    if seed is not None:
        np.random.seed(int(seed) % (2 ** 31 - 1))
        torch.manual_seed(int(seed))
    return posterior.sample_single_batch(num_samples, test_data, test_cosmo, dict(mcmc_kwargs))


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
        per_member = [[] for _ in self.members]
        for batch in self.test_dataloader:
            x, theta = batch
            x = _move_nested_to_device(x, target_device)
            theta = _move_nested_to_device(theta, target_device)
            batch_lps = []
            for i, m in enumerate(self.members):
                m.to(target_device)
                m.eval()
                lp = m.forward(x, cond=theta)
                batch_lps.append(lp)
                per_member[i].append(lp.reshape(-1).detach())
            all_log_probs.append(torch.stack(batch_lps, dim=0).mean(dim=0).reshape(-1))

        # Report per member: a mean of log-densities hides one degenerate member.
        member_means = [float(-torch.cat(v, dim=0).mean().item()) for v in per_member]
        print(
            "[ensemble-nle] per-member mean -log p(z|theta): "
            + ", ".join(f"m{i}={v:.4f}" for i, v in enumerate(member_means)),
            flush=True,
        )

        return float(-torch.cat(all_log_probs, dim=0).mean().item())

    def build_posterior_object(
        self,
        prior=None,
        fixed_parameters=None,
        reduction: str = "logmeanexp",
        fast: bool = True,
    ):
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

        # Fast path: all members in one vectorised forward (ensemble_flow.py), else the serial loop.
        ensemble_likelihood = None
        if fast:
            ensemble_likelihood = build_grouped_ensemble_likelihood(
                list(self.members), reduction=reduction
            )
            if ensemble_likelihood is None:
                print(
                    "[EnsembleLikelihoodNDE] fast path unavailable for these members; "
                    "falling back to the serial ensemble likelihood."
                )
        if ensemble_likelihood is None:
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
        fast: bool = True,
        mcmc_workers: int | None = None,
        mcmc_threads: int | None = None,
        mcmc_seed: int | None = None,
        **mcmc_kwargs,
    ):
        """Sample the ensemble posterior for every event in test_dataloader.

        Returns samples (num_samples, n_events, n_params), chain-major along the sample axis.
        mcmc_workers=K splits each event's sample budget into K shards of ceil(num_samples / K)
        draws, one process each, which leaves the target distribution unchanged. None keeps one
        work item per batch. mcmc_threads sets torch threads and loky's inner_max_num_threads per
        worker. mcmc_seed seeds shard i with mcmc_seed + i for numpy and torch.
        """
        posterior = self.build_posterior_object(
            prior=prior,
            reduction=reduction,
            fixed_parameters=mcmc_kwargs.pop("fixed_parameters", None),
            fast=fast,
        )

        posterior.to("cpu")
        posterior.prior.to("cpu")

        batches = list(self.test_dataloader)
        n_shards = max(1, int(mcmc_workers or 1))
        # Every shard draws the same (rounded-up) count, so the concatenation is >= num_samples
        # and the truncation below drops only the tail of the last shard.
        per_shard = int(math.ceil(num_samples / n_shards))
        items = [(bi, si) for bi in range(len(batches)) for si in range(n_shards)]
        # Clamp n_jobs only when sharding: n_jobs=1 would switch joblib to its sequential backend.
        n_jobs = num_jobs
        if n_shards > 1 and num_jobs and num_jobs > 0:
            n_jobs = min(int(num_jobs), len(items))

        # joblib honours inner_max_num_threads only through parallel_config, so Parallel below
        # must not take backend=.
        cfg = {"backend": backend}
        if mcmc_threads:
            cfg["inner_max_num_threads"] = int(mcmc_threads)

        if n_shards > 1 or mcmc_threads or mcmc_seed is not None:
            print(f"[ensemble-nle] sampling {len(batches)} batch(es) x {n_shards} shard(s) "
                  f"= {len(items)} work items on n_jobs={n_jobs} "
                  f"(per-shard draws {per_shard}, threads/worker {mcmc_threads or 'default'})",
                  flush=True)

        with parallel_config(**cfg):
            jobs = Parallel(
                n_jobs=n_jobs,
                return_as="generator",
            )(
                delayed(_sample_shard)(
                    posterior,
                    per_shard,
                    batches[bi][0],
                    batches[bi][1],
                    dict(mcmc_kwargs),
                    None if mcmc_seed is None else int(mcmc_seed) + k,
                    mcmc_threads,
                )
                for k, (bi, si) in enumerate(items)
            )

            # Consumed INSIDE the context: with return_as="generator" the workers are only
            # created/driven while the generator is being drained.
            results = list(
                tqdm(
                    jobs,
                    total=len(items),
                    desc="Sampling ensemble batches",
                )
            )

        # Shards of one batch concatenate along the sample axis, batches along the event axis (dim 1).
        per_batch_samples: dict[int, list] = {}
        per_batch_theta0: dict[int, object] = {}
        for (bi, _si), (theta0, s) in zip(items, results):
            per_batch_samples.setdefault(bi, []).append(s)
            per_batch_theta0.setdefault(bi, theta0)

        theta0s = torch.cat([per_batch_theta0[bi] for bi in range(len(batches))], dim=0)
        samples = torch.cat(
            [torch.cat(per_batch_samples[bi], dim=0)[:num_samples] for bi in range(len(batches))],
            dim=1,
        )
        return theta0s, samples
