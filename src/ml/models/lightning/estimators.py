from __future__ import annotations

from collections.abc import Mapping

import torch
from sbi.analysis import conditional_potential
from sbi.inference.posteriors import MCMCPosterior
from sbi.inference.potentials.likelihood_based_potential import (
    likelihood_estimator_based_potential,
)
from sbi.neural_nets.estimators import ConditionalDensityEstimator
from sbi.samplers.rejection import rejection
from sbi.utils.sbiutils import within_support

from .utils import (
    ConditionDict,
    _BatchableTransform,
    _move_nested_to_device,
)


class PatchedConditionalDensityEstimator(ConditionalDensityEstimator):
    def __init__(self, model, prior, input_shape=(1,), condition_shape=(1,)):
        super().__init__(model, input_shape=input_shape, condition_shape=condition_shape)
        self.prior = prior
        self.max_sampling_batch_size = 10_000

    def _check_condition_shape(self, condition):
        pass

    def _check_input_shape(self, input):
        pass

    def log_prob(self, x, y):
        return self.net.log_prob(x, y)

    def loss(self, x, y):
        return -self.net.log_prob(x, y).mean()

    def sample(self, num_samples, condition):
        return self.net.sample(num_samples, condition)

    def latent_sample(self, num_samples, condition):
        return self.net.flow.sample(num_samples, condition)

    def compress(self, data_dict):
        if hasattr(self.net, "encode"):
            return self.net.encode(data_dict)
        return self.net.embedding_net(data_dict)

    def latent_log_prob(self, x, y_emb):
        if hasattr(self.net, "latent_log_prob"):
            return self.net.latent_log_prob(x, y_emb)
        return self.net.flow.log_prob(x.unsqueeze(0), y_emb)

    @torch.no_grad()
    def gen_samples(self, num_samples, x, use_latent=True, **kwargs):
        if isinstance(x, dict):
            cond = ConditionDict(x)
        else:
            cond = x
        sampling_func = self.latent_sample if use_latent else self.sample
        samples = rejection.accept_reject_sample(
            proposal=sampling_func,
            accept_reject_fn=lambda theta: within_support(self.prior, theta),
            num_samples=num_samples,
            show_progress_bars=False,
            max_sampling_batch_size=self.max_sampling_batch_size,
            proposal_sampling_kwargs={"condition": cond},
            alternative_method="build_posterior(..., sample_with='mcmc')",
            num_xos=cond.shape[0],
        )[0]
        return samples


class PatchedLikelihoodEstimator(ConditionalDensityEstimator):
    """ConditionalDensityEstimator wrapper for neural likelihood estimation."""

    def __init__(
        self,
        model,
        prior,
        input_shape=(1,),
        condition_shape=(1,),
        fixed_parameters=None,
    ):
        super().__init__(model, input_shape=input_shape, condition_shape=condition_shape)
        self.prior = prior
        self.max_sampling_batch_size = 10_000
        self.fixed_parameters = fixed_parameters

    def _check_condition_shape(self, condition):
        pass

    def _check_input_shape(self, input):
        pass

    def log_prob(self, x, condition):
        if hasattr(x, "ndim") and x.ndim > 2 and x.shape[0] == 1:
            x = x.squeeze(0)
        return self.net.log_prob(x, condition)

    def sample(self, num_samples, condition):
        return self.net.sample(num_samples, condition)

    def loss(self, x, y):
        return -self.log_prob(x, y).mean()

    def log_likelihood(self, x, theta):
        return self.log_prob(x, theta)

    def sample_single_batch(self, num_samples, test_data, test_cosmo, mcmc_kwargs):
        x = test_data

        samples = self._gen_samples(
            num_samples=num_samples,
            x=x,
            use_latent=False,
            **mcmc_kwargs,
        )

        return test_cosmo, samples

    def gen_samples(self, num_samples, x, use_latent=True, num_jobs=10, **mcmc_kwargs):
        # Move to CPU for joblib multiprocessing.
        self.to("cpu")
        self.prior.to("cpu")

        if num_jobs > 1:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=num_jobs, backend="loky")(
                delayed(self._gen_samples)(
                    num_samples=num_samples,
                    x=x_single.unsqueeze(0).to("cpu"),
                    use_latent=use_latent,
                    **mcmc_kwargs,
                )
                for x_single in x
            )
            samples = torch.stack(results, dim=1)
        else:
            samples = self._gen_samples(
                num_samples=num_samples,
                x=x,
                use_latent=use_latent,
                **mcmc_kwargs,
            )
        return samples

    @torch.no_grad()
    def _gen_samples(self, num_samples: int, x, use_latent, **mcmc_kwargs):
        device = next(self.net.parameters()).device
        x_batch = _move_nested_to_device(x, device)

        method = mcmc_kwargs.pop("method", "slice_np_vectorized")
        num_chains = mcmc_kwargs.pop("num_chains", 4)
        thin = mcmc_kwargs.pop("thin", 1)
        warmup_steps = mcmc_kwargs.pop("warmup_steps", 500)
        show_progress_bars = mcmc_kwargs.pop("show_progress_bars", False)

        sample_shape = (num_samples,)

        while True:
            try:
                potential, tf = likelihood_estimator_based_potential(
                    self,
                    self.prior,
                    x_o=None,
                    enable_transform=True,
                )
                break
            except AssertionError as e:
                print("Error in check_transform, retrying...", e)

        prior_to_use = self.prior

        if self.fixed_parameters:
            total_dim = self.condition_shape[0]

            condition = torch.zeros(total_dim, device=device)
            fixed_indices = [idx for idx, _ in self.fixed_parameters]
            dims_to_sample = [i for i in range(total_dim) if i not in fixed_indices]

            for idx, val in self.fixed_parameters:
                condition[idx] = val

            potential, tf, prior_to_use = conditional_potential(
                potential_fn=potential,
                theta_transform=tf,
                prior=self.prior,
                condition=condition,
                dims_to_sample=dims_to_sample,
            )
            tf = _BatchableTransform(tf)

        posterior = MCMCPosterior(
            potential_fn=potential,
            proposal=prior_to_use,
            theta_transform=tf,
            method=method,
            num_chains=num_chains,
            num_workers=1,
            thin=thin,
            warmup_steps=warmup_steps,
            device=device,
            **mcmc_kwargs,
        )

        samples = posterior.sample_batched(
            sample_shape=sample_shape,
            x=x_batch,
            show_progress_bars=show_progress_bars,
        )

        return samples.cpu()
