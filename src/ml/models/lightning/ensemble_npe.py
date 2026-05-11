from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from collections.abc import Mapping

from .utils import _move_nested_to_device


class EnsembleNDELightningModule(pl.LightningModule):
    """Evaluation-time ensemble of separately-trained NPE LightningModules."""

    def __init__(self, members: list[pl.LightningModule]):
        super().__init__()
        if not members:
            raise ValueError("EnsembleNDELightningModule requires at least one member.")
        self.members = nn.ModuleList(members)

        self.test_dataloader = getattr(members[0], "test_dataloader", None)
        self.loss_name = getattr(members[0], "loss_name", "log_prob")

    def _resolve_device(self):
        try:
            return self.device
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def _get_theta0s_from_loader(self):
        if self.test_dataloader is None:
            raise ValueError("EnsembleNDELightningModule.test_dataloader is None")
        theta0s = []
        for _, theta in self.members[0].test_dataloader:
            theta0s.append(theta)
        return torch.cat(theta0s, dim=0)

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
    def compute_avg_log_prob(self, reduction: str = "logmeanexp", return_log_probs: bool = False):
        if reduction not in {"logmeanexp", "mean_log_prob"}:
            raise ValueError("reduction must be one of {'mean_log_prob', 'logmeanexp'}")

        if self.test_dataloader is None:
            raise ValueError("EnsembleNDELightningModule.test_dataloader is None")

        target_device = self._resolve_device()
        all_log_probs = []

        for batch in self.test_dataloader:
            data_dict, theta = batch
            data_dict = _move_nested_to_device(data_dict, target_device)
            theta = _move_nested_to_device(theta, target_device)

            batch_lps = []
            for m in self.members:
                m.to(target_device)
                m.eval()
                batch_lps.append(m.forward(theta, cond=data_dict).reshape(-1))

            stacked = torch.stack(batch_lps, dim=0)
            if reduction == "logmeanexp":
                batch_lp = torch.logsumexp(stacked, dim=0) - np.log(len(self.members))
            else:
                batch_lp = stacked.mean(dim=0)
            all_log_probs.append(batch_lp)

        if len(all_log_probs) == 0:
            raise ValueError("Test dataloader produced no batches.")

        all_log_probs = torch.cat(all_log_probs, dim=0)
        if return_log_probs:
            return all_log_probs.detach().cpu()
        return float(-all_log_probs.mean().item())

    @torch.no_grad()
    def generate_samples(self, num_samples=10000, **kwargs):
        theta0s = self._get_theta0s_from_loader()

        for idx, m in enumerate(self.members):
            if getattr(m, "test_dataloader", None) is None:
                raise ValueError(f"Ensemble member {idx} has no test_dataloader")

        n = len(self.members)
        base = num_samples // n
        rem = num_samples % n
        counts = [base + (1 if i < rem else 0) for i in range(n)]

        target_device = self._resolve_device()

        parts = []
        for m, k in zip(self.members, counts):
            if k <= 0:
                continue
            m.to(target_device)
            m.eval()
            _, samp = m.generate_samples(num_samples=k, **kwargs)
            parts.append(samp)

        samples = torch.cat(parts, dim=0)
        perm = torch.randperm(samples.shape[0])
        samples = samples[perm]
        return theta0s, samples

    def build_posterior_object(self, prior=None):
        class _EnsemblePosterior:
            def __init__(self, posteriors, members):
                self._posteriors = posteriors
                self._members = members
                self.prior = getattr(posteriors[0], "prior", None)

            @staticmethod
            def _infer_device(x):
                if isinstance(x, Mapping):
                    for v in x.values():
                        if hasattr(v, "device"):
                            return v.device
                if hasattr(x, "device"):
                    return x.device
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")

            def to(self, *args, **kwargs):
                for p in self._posteriors:
                    if hasattr(p, "to"):
                        p.to(*args, **kwargs)
                    if hasattr(p, "prior") and hasattr(p.prior, "to"):
                        p.prior.to(*args, **kwargs)
                return self

            def eval(self):
                for p in self._posteriors:
                    if hasattr(p, "eval"):
                        p.eval()
                return self

            @torch.no_grad()
            def gen_samples(self, num_samples, x, use_latent=True, **kwargs):
                if num_samples <= 0:
                    raise ValueError("num_samples must be > 0")

                device = self._infer_device(x)
                self.to(device)
                self.eval()

                n = len(self._posteriors)
                base = num_samples // n
                rem = num_samples % n
                counts = [base + (1 if i < rem else 0) for i in range(n)]

                parts = []
                for p, k in zip(self._posteriors, counts):
                    if k <= 0:
                        continue
                    out = p.gen_samples(k, x=x, use_latent=use_latent, **kwargs)
                    if isinstance(out, tuple):
                        out = out[0]
                    parts.append(out)

                if not parts:
                    raise ValueError("No samples were generated (check num_samples and ensemble size).")

                samples = torch.cat(parts, dim=0)
                perm = torch.randperm(samples.shape[0], device=samples.device)
                return samples[perm]

        posteriors = [m.build_posterior_object(prior=prior) for m in self.members]
        return _EnsemblePosterior(posteriors=posteriors, members=self.members)
