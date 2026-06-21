from __future__ import annotations

import torch
import torch.nn.functional as F

from .npe import NDELightningModule


class VICRegRegularisedNDELightningModule(NDELightningModule):
    """NPE + VICReg regularisation on the learned summary, with same-cosmology invariance.

    Implements the variance-invariance-covariance (VICReg) regulariser of Williamson et al.
    "DES Y3: optimized wCDM simulation-based inference with weak-lensing map-level hybrid
    statistics" (arXiv:2606.11309), Section 3.4 (eq. 13). The penalty is added to the VMIM/NPE
    NLL and trained jointly (L_total = L_VMIM + L_VIC), operating on the final compression summary
    z = ``model.encode(...)`` (the flow's conditioning context). For the hybrid architecture z is
    the post-``hybrid_head`` embedding that linearly combines the (frozen) bandpower summary with
    the trainable map summary — so even with the band frozen, the FINAL embedding is regularised.

    Invariance view source (the redesign, research Option 2): the two summaries the invariance term
    pulls together are DIFFERENT REALISATIONS of the SAME cosmology θ, as in the paper, NOT two
    augmentations of one map. This is delivered by the :class:`MPerCosmoBatchSampler` train loader,
    which packs each batch with k distinct cosmologies × m realisations each and passes the
    per-sample integer cosmology id as a 3rd batch element. The invariance term then groups z by
    cosmology id and penalises within-cosmology spread. A SINGLE encoder forward per batch (no
    doubled map reads / no augmentation pass).

    Paper loss (each weight set to unity in the paper):

        s        = within-cosmology invariance: mean over cosmologies of the mean squared
                   distance of that cosmology's z-rows to their group mean (generalises eq.13's
                   pairwise || t_i - t'_i ||^2 to m>2, SupCon-style; for m=2 it equals the literal
                   pairwise term up to a constant folded into vicreg_sim_coeff).
        v(Z)     = (1/d) sum_j max(0, gamma - sqrt(Var(z^j) + eps))   (variance hinge, full batch)
        c(Z)     = (1/d) sum_{i != j} [Cov(Z)]^2_{ij}                 (covariance, full batch)
        L_VIC    = lambda * s + mu * v(Z) + nu * c(Z)

    Defaults: lambda = mu = nu = 1.0 (paper: "each set to unity"); gamma = 1.0, eps = 1e-4
    (standard VICReg, Bardes et al. 2022). All exposed as config keys (vicreg_*).

    Note: ``RandomEBPatchAugment`` (via ``augment_eb_patches``) stays ON as ordinary per-sample
    base-dataset jitter — it is now applied to genuinely different realisations and is independent
    of the VICReg pairing (it is NOT the source of the two views).
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

    @staticmethod
    def _invariance_term(z, cosmo_id):
        """Within-cosmology invariance (eq.13, SupCon-generalised to m>2).

        Group the rows of z by their integer cosmology id; for each cosmology with >= 2 members in
        the batch, compute the mean squared distance of its z-rows to their group mean (the
        within-cosmology spread); average over the non-singleton cosmologies. Returns a
        graph-connected 0 if no cosmology has >= 2 members in the batch.
        """
        cids = cosmo_id.detach().view(-1)
        terms = []
        for c in torch.unique(cids):
            mask = cids == c
            if int(mask.sum()) < 2:
                continue
            zg = z[mask]
            diff = zg - zg.mean(dim=0, keepdim=True)
            terms.append((diff ** 2).sum(dim=1).mean())
        if not terms:
            return z.sum() * 0.0
        return torch.stack(terms).mean()

    def _vicreg_loss(self, z, cosmo_id):
        # Reductions in fp32 for numerical stability (z may be bf16 under amp autocast).
        z = z.float()
        if z.shape[0] < 2:
            # Variance/covariance need >= 2 samples; skip the penalty on a degenerate batch.
            zero = z.sum() * 0.0
            return zero, zero, zero, zero
        sim = self._invariance_term(z, cosmo_id)
        var = self._variance_term(z, self.vicreg_gamma, self.vicreg_eps)
        cov = self._covariance_term(z)
        weighted = (
            self.vicreg_sim_coeff * sim
            + self.vicreg_var_coeff * var
            + self.vicreg_cov_coeff * cov
        )
        return weighted, sim, var, cov

    def _shared_step(self, batch, stage: str):
        # Train batches (MPerCosmoBatchSampler) carry the per-sample cosmology id as a 3rd element;
        # the val/test loaders are plain (2-tuple) so VIC terms are 0 there and val_log_prob stays
        # the clean NPE NLL (the checkpoint monitor metric).
        if len(batch) == 3:
            data_dict, theta, cosmo_id = batch
        else:
            data_dict, theta = batch
            cosmo_id = None

        # Single encoder forward -> the final post-hybrid_head summary z.
        z = self.model.encode(data_dict)

        log_prob = self.model.latent_log_prob(theta, z)
        nll = -log_prob.mean()

        if cosmo_id is not None:
            vic, sim, var, cov = self._vicreg_loss(z, cosmo_id)
        else:
            zero = z.sum() * 0.0
            vic = sim = var = cov = zero
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
