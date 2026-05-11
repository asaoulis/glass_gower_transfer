from __future__ import annotations

import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from sbi import utils as sbi_utils

from .estimators import PatchedLikelihoodEstimator
from .npe import NDELightningModule


class LikelihoodNDELightningModule(NDELightningModule):
    """Neural likelihood LightningModule (likelihood p(x | theta))."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, cond=None):
        x_emb = self.model.embedding_net(x)
        x_emb = x_emb.unsqueeze(0)
        return self.model.flow.log_prob(x_emb, cond)

    def training_step(self, batch, batch_idx):
        x, theta = batch
        preds = self.forward(x, cond=theta)
        loss = self.compute_loss(preds, theta)
        self.log(
            f"train_{self.loss_name}",
            loss,
            prog_bar=True,
            sync_dist=self.is_distributed,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, theta = batch
        preds = self.forward(x, cond=theta)
        loss = self.compute_loss(preds, theta)
        self.log(
            f"val_{self.loss_name}",
            loss,
            prog_bar=True,
            sync_dist=self.is_distributed,
        )
        self.log_custom_evals(preds, theta)
        return loss

    def compute_avg_log_prob(self):
        predictions = []
        for batch in self.test_dataloader:
            batch = self.transfer_batch_to_device(batch, self.device, 0)
            data_dict, theta = batch
            predictions.append(self.forward(data_dict, theta).reshape(-1))
        all_log_probs = torch.cat(predictions, dim=0)
        avg_log_prob = -all_log_probs.mean().item()
        return avg_log_prob

    def generate_samples(
        self,
        num_samples=2_000,
        num_jobs=36,
        backend="loky",
        prior=None,
        fixed_parameters=None,
        **mcmc_kwargs,
    ):
        if fixed_parameters is None:
            fixed_parameters = mcmc_kwargs.pop("fixed_parameters", None)

        posterior = self.build_posterior_object(prior=prior, fixed_parameters=fixed_parameters)
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
                mcmc_kwargs,
            )
            for test_data, test_cosmo in self.test_dataloader
        )

        results = list(
            tqdm(
                jobs,
                total=len(self.test_dataloader),
                desc="Sampling batches",
            )
        )

        theta0s, samples = zip(*results)

        theta0s = torch.cat(theta0s, dim=0)
        samples = torch.cat(samples, dim=1)

        return theta0s, samples

    def build_posterior_object(self, prior=None, fixed_parameters=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        if hasattr(self.model, "embedding_net") and hasattr(
            self.model.embedding_net, "only_return_mu"
        ):
            self.model.embedding_net.only_return_mu = True

        if prior is None:
            prior = sbi_utils.BoxUniform(
                low=0 * torch.ones(self.conditioning_dim, device=device),
                high=1.0 * torch.ones(self.conditioning_dim, device=device),
                device=device,
            )

        likelihood_estimator = PatchedLikelihoodEstimator(
            model=self.model,
            prior=prior,
            input_shape=(self.inference_dim,),
            condition_shape=(self.conditioning_dim,),
            fixed_parameters=fixed_parameters,
        )
        return likelihood_estimator
