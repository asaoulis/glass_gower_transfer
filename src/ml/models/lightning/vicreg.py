from __future__ import annotations

import torch
import torch.nn.functional as F

from .npe import NDELightningModule
from ...data.data_augmentations import RandomEBPatchAugment


class VICRegRegularisedNDELightningModule(NDELightningModule):
    """NPE + VICReg regularisation on the learned summary.

    Implements the variance-invariance-covariance (VICReg) regulariser of Williamson et al.
    "DES Y3: optimized wCDM simulation-based inference with weak-lensing map-level hybrid
    statistics" (arXiv:2606.11309), Section 3.4. The penalty is added to the VMIM/NPE NLL and
    trained jointly (L_total = L_VMIM + L_VIC), operating on the final compression summary z
    (the flow's conditioning context). Two views T, T' of the same input are produced by two
    independent flip / 180-deg-rotation augmentations of the E/B map patches (the
    rotational-augmentation analogue of the paper's "different rotations of the same cosmology");
    the band part of z is frozen and identical across views, so the invariance term is driven by
    the trainable map encoder.

    Paper loss (each weight set to unity in the paper):

        s(T, T') = (1/n) sum_i || t_i - t'_i ||^2                  (invariance / similarity)
        v(T)     = (1/d) sum_j max(0, gamma - sqrt(Var(t^j) + eps))  (variance hinge)
        c(T)     = (1/d) sum_{i != j} [Cov(T)]^2_{ij}               (covariance)
        L_VIC    = lambda * s + mu * [v(T) + v(T')] + nu * [c(T) + c(T')]

    Defaults: lambda = mu = nu = 1.0 (paper: "each set to unity"); gamma = 1.0, eps = 1e-4
    (standard VICReg, Bardes et al. 2022). All exposed as config keys (vicreg_*).
    """

    def __init__(
        self,
        *args,
        vicreg_sim_coeff: float = 1.0,
        vicreg_var_coeff: float = 1.0,
        vicreg_cov_coeff: float = 1.0,
        vicreg_gamma: float = 1.0,
        vicreg_eps: float = 1e-4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vicreg_sim_coeff = vicreg_sim_coeff
        self.vicreg_var_coeff = vicreg_var_coeff
        self.vicreg_cov_coeff = vicreg_cov_coeff
        self.vicreg_gamma = vicreg_gamma
        self.vicreg_eps = vicreg_eps
        self._augment = RandomEBPatchAugment()

    def _augmented_view(self, data_dict):
        """One independent flip/180-rot augmentation of the E/B map patches (batch-level, per
        side). Bandpowers and any non-EB entries pass through untouched. Returns a NEW dict; the
        original tensors are untouched (torch.flip / rot90 allocate new tensors)."""
        return self._augment(dict(data_dict))

    @staticmethod
    def _variance_term(z, gamma, eps):
        std = torch.sqrt(z.var(dim=0, unbiased=True) + eps)
        return torch.mean(F.relu(gamma - std))

    @staticmethod
    def _covariance_term(z):
        n, d = z.shape
        z = z - z.mean(dim=0, keepdim=True)
        cov = (z.T @ z) / (n - 1)
        off_diag_sq = cov.pow(2).sum() - torch.diagonal(cov).pow(2).sum()
        return off_diag_sq / d

    def _vicreg_loss(self, z1, z2):
        # Reductions in fp32 for numerical stability (z may be bf16 under amp autocast).
        z1 = z1.float()
        z2 = z2.float()
        if z1.shape[0] < 2:
            # Variance/covariance need >= 2 samples; skip the penalty on a degenerate batch.
            zero = z1.sum() * 0.0
            return zero, zero, zero, zero
        sim = ((z1 - z2) ** 2).sum(dim=1).mean()  # (1/n) sum_i || t_i - t'_i ||^2
        var = self._variance_term(z1, self.vicreg_gamma, self.vicreg_eps) + self._variance_term(
            z2, self.vicreg_gamma, self.vicreg_eps
        )
        cov = self._covariance_term(z1) + self._covariance_term(z2)
        weighted = (
            self.vicreg_sim_coeff * sim
            + self.vicreg_var_coeff * var
            + self.vicreg_cov_coeff * cov
        )
        return weighted, sim, var, cov

    def _shared_step(self, batch, stage: str):
        data_dict, theta = batch

        # Two augmented views -> two summaries (the band part is frozen+identical across views).
        z1 = self.model.encode(self._augmented_view(data_dict))
        z2 = self.model.encode(self._augmented_view(data_dict))

        log_prob = self.model.latent_log_prob(theta, z1)
        nll = -log_prob.mean()

        vic, sim, var, cov = self._vicreg_loss(z1, z2)
        total = nll + vic

        metrics = {
            f"{stage}_{self.loss_name}": nll,
            f"{stage}_vicreg": vic,
            f"{stage}_vicreg_sim": sim,
            f"{stage}_vicreg_var": var,
            f"{stage}_vicreg_cov": cov,
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

    def log_custom_evals(self, preds, y):
        if len(self.test_loss_values) > 0:
            self.log(
                "test_log_prob",
                self.test_loss_values.pop(),
                sync_dist=self.is_distributed,
            )
