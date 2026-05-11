from __future__ import annotations

import torch

from .npe import NDELightningModule


class KLDRegularisedNDELightningModule(NDELightningModule):
    def __init__(
        self,
        *args,
        kl_weight: float = 1e-2,
        kl_min: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.kl_weight = kl_weight
        self.kl_min = kl_min

    def _latent_stats(self, data_dict):
        encoder = self.embedding_net
        mu, logvar = encoder.compress(data_dict)
        return mu, logvar

    @staticmethod
    def _reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _kl_divergence(self, mu, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl = kl.mean()
        if self.kl_min > 0.0:
            kl = torch.clamp(kl, min=self.kl_min)
        return kl

    def _shared_step(self, batch, stage: str):
        data_dict, theta = batch

        mu, logvar = self._latent_stats(data_dict)

        kl = self._kl_divergence(mu, logvar)

        if stage == "train":
            z = self._reparameterize(mu, logvar)
        else:
            z = mu

        log_prob = self.model.latent_log_prob(theta, z)
        nll = -log_prob.mean()

        total = nll + self.kl_weight * kl

        metrics = {
            f"{stage}_{self.loss_name}": nll,
            f"{stage}_kl": kl,
            f"{stage}_weighted_kl": self.kl_weight * kl,
        }
        for name, value in metrics.items():
            on_step = stage == "train"
            self.log(
                name,
                value,
                prog_bar=(name.endswith(self.loss_name)),
                on_step=on_step,
                on_epoch=True,
                sync_dist=self.is_distributed,
            )

        return total, log_prob, theta

    def training_step(self, batch, batch_idx):
        loss, log_prob, theta = self._shared_step(batch, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_prob, theta = self._shared_step(batch, stage="val")
        self.log_custom_evals(log_prob, theta)
        return loss

    def compute_avg_log_prob(self):
        predictions = []
        for batch in self.test_dataloader:
            batch = self.transfer_batch_to_device(batch, self.device, 0)
            data_dict, theta = batch
            mu, logvar = self._latent_stats(data_dict)
            log_prob = self.model.latent_log_prob(theta, mu)
            predictions.append(log_prob.reshape(-1))
        all_log_probs = torch.cat(predictions, dim=0)
        avg_log_prob = -all_log_probs.mean().item()
        return avg_log_prob

    def log_custom_evals(self, preds, y):
        if len(self.test_loss_values) > 0:
            self.log(
                "test_log_prob",
                self.test_loss_values.pop(),
                sync_dist=self.is_distributed,
            )
