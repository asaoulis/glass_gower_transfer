from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from sbi import utils as sbi_utils

from .base import BaseLightningModule
from .estimators import PatchedConditionalDensityEstimator, PatchedLikelihoodEstimator
from .flows import _CondEmbeddingFlow
from .npe import NDELightningModule
from .utils import load_partial_weights


class JointVMIMNLELightningModule(BaseLightningModule):
    """Joint training of NPE (posterior) and NLE (likelihood) heads."""

    flow_type_map = NDELightningModule.flow_type_map

    def __init__(
        self,
        model,
        conditioning_dim,
        inference_dim,
        *,
        nle_weighting: float = 1.0,
        redundancy_dim: int = 0,
        lr=0.0001,
        scheduler_type="cosine",
        test_dataloader=None,
        flow_type="nsf",
        num_extra_blocks=None,
        flow_kwargs=None,
        **kwargs,
    ):
        super().__init__(model, loss_fn=None, lr=lr, scheduler_type=scheduler_type, **kwargs)

        self.embedding_net = model if model is not None else nn.Identity()
        self.conditioning_dim = int(conditioning_dim)
        self.inference_dim = int(inference_dim)
        self.redundancy_dim = int(redundancy_dim)

        self.nle_weighting = float(nle_weighting)

        self.build_flow = self.flow_type_map[flow_type]
        flow_kwargs = flow_kwargs or {}
        if "zuko" in str(flow_type).lower():
            self.flow_kwargs = dict(flow_kwargs)
        else:
            self.flow_kwargs = {"use_batch_norm": False, **dict(flow_kwargs)}

        self.test_dataloader = test_dataloader
        self.loss_name = "joint"

        self._test_npe_loss_values = []
        self._test_nle_loss_values = []

        self.set_up_model()

    def set_up_model(self):
        z_dataset = torch.randn(10, self.conditioning_dim)
        theta_dataset = torch.randn(10, self.inference_dim)

        hidden_features = self.flow_kwargs.pop("hidden_features", self.conditioning_dim)

        if "zuko" in str(self.build_flow).lower():
            npe_flow_kwargs = dict(self.flow_kwargs)
        else:
            npe_flow_kwargs = {"conditional_dim": self.conditioning_dim, **self.flow_kwargs}

        npe_flow = self.build_flow(
            theta_dataset,
            z_dataset,
            num_transforms=5,
            z_score_x=None,
            z_score_y=None,
            embedding_net=nn.Identity(),
            hidden_features=32,
            **npe_flow_kwargs,
        )

        if "zuko" in str(self.build_flow).lower():
            nle_flow_kwargs = dict(self.flow_kwargs)
        else:
            nle_flow_kwargs = {"conditional_dim": self.inference_dim, **self.flow_kwargs}

        nle_flow = self.build_flow(
            z_dataset,
            theta_dataset,
            num_transforms=5,
            z_score_x=None,
            z_score_y=None,
            embedding_net=nn.Identity(),
            hidden_features=hidden_features,
            **nle_flow_kwargs,
        )

        self.npe_flow = npe_flow
        self.nle_flow = nle_flow

        self.npe_model = _CondEmbeddingFlow(self.embedding_net, self.npe_flow)

    def compress(self, data_dict):
        return self.embedding_net(data_dict)

    def _encode_to_z(self, data_dict):
        z = self.compress(data_dict)
        if isinstance(z, tuple):
            z = z[0]
        return z

    def forward_npe(self, theta, data_dict):
        return self.npe_model.log_prob(theta, data_dict)

    def forward_nle(self, data_dict, theta):
        z = self._encode_to_z(data_dict)
        z = z.unsqueeze(0)
        return self.nle_flow.log_prob(z, theta)

    def compute_losses(self, data_dict, theta):
        lp_npe = self.forward_npe(theta, data_dict)
        lp_nle = self.forward_nle(data_dict, theta)
        npe_loss = -lp_npe.mean()
        nle_loss = -lp_nle.mean()
        total = npe_loss + self.nle_weighting * nle_loss
        return total, npe_loss, nle_loss

    def training_step(self, batch, batch_idx):
        data_dict, theta = batch
        total, npe_loss, nle_loss = self.compute_losses(data_dict, theta)
        self.log("train_npe_loss", npe_loss, prog_bar=True, sync_dist=self.is_distributed)
        self.log("train_nle_loss", nle_loss, prog_bar=False, sync_dist=self.is_distributed)
        self.log("train_joint_loss", total, prog_bar=False, sync_dist=self.is_distributed)
        return total

    def validation_step(self, batch, batch_idx):
        data_dict, theta = batch
        total, npe_loss, nle_loss = self.compute_losses(data_dict, theta)
        self.log("val_npe_loss", npe_loss, prog_bar=True, sync_dist=self.is_distributed)
        self.log("val_nle_loss", nle_loss, prog_bar=False, sync_dist=self.is_distributed)
        self.log("val_joint_loss", total, prog_bar=False, sync_dist=self.is_distributed)
        self.log_custom_evals((npe_loss.detach(), nle_loss.detach()), theta)
        return total

    def on_validation_epoch_end(self):
        if self.test_dataloader is None:
            return
        self.eval()
        with torch.no_grad():
            npe, nle = self.compute_avg_test_losses()
        if npe is not None:
            self._test_npe_loss_values.append(float(npe))
        if nle is not None:
            self._test_nle_loss_values.append(float(nle))

    def compute_avg_test_losses(self):
        npe_losses = []
        nle_losses = []
        for batch in self.test_dataloader:
            batch = self.transfer_batch_to_device(batch, self.device, 0)
            data_dict, theta = batch
            _, npe_loss, nle_loss = self.compute_losses(data_dict, theta)
            npe_losses.append(npe_loss.detach().item())
            nle_losses.append(nle_loss.detach().item())
        if not npe_losses:
            return None, None
        return float(np.mean(npe_losses)), float(np.mean(nle_losses))

    def log_custom_evals(self, preds, y):
        if len(self._test_npe_loss_values) > 0:
            self.log("test_npe_loss", self._test_npe_loss_values.pop(), sync_dist=self.is_distributed)
        if len(self._test_nle_loss_values) > 0:
            self.log("test_nle_loss", self._test_nle_loss_values.pop(), sync_dist=self.is_distributed)

    def build_posterior_object(self, prior=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval()
        if hasattr(self.embedding_net, "only_return_mu"):
            self.embedding_net.only_return_mu = True

        if prior is None:
            prior = sbi_utils.BoxUniform(
                low=0 * torch.ones(self.inference_dim, device=device),
                high=1.0 * torch.ones(self.inference_dim, device=device),
                device=device,
            )

        density_estimator = PatchedConditionalDensityEstimator(
            self.npe_model,
            prior,
            input_shape=(self.inference_dim,),
            condition_shape=(self.conditioning_dim,),
        )
        return density_estimator

    def build_likelihood_object(self, prior=None, fixed_parameters=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval()
        if hasattr(self.embedding_net, "only_return_mu"):
            self.embedding_net.only_return_mu = True

        if prior is None:
            prior = sbi_utils.BoxUniform(
                low=0 * torch.ones(self.conditioning_dim, device=device),
                high=1.0 * torch.ones(self.conditioning_dim, device=device),
                device=device,
            )

        class _NLEWrapper(nn.Module):
            def __init__(self, embedding_net, nle_flow):
                super().__init__()
                self.embedding_net = embedding_net if embedding_net is not None else nn.Identity()
                self.flow = nle_flow

            def log_prob(self, x, theta):
                z = self.embedding_net(x)
                if isinstance(z, tuple):
                    z = z[0]
                z = z.unsqueeze(0)
                return self.flow.log_prob(z, theta)

            def sample(self, shape, theta, **kwargs):
                return self.flow.sample(shape, theta, **kwargs)

        nle_model = _NLEWrapper(self.embedding_net, self.nle_flow)

        likelihood_estimator = PatchedLikelihoodEstimator(
            model=nle_model,
            prior=prior,
            input_shape=(self.conditioning_dim,),
            condition_shape=(self.inference_dim,),
            fixed_parameters=fixed_parameters,
        )
        return likelihood_estimator

    def _load_pretrained_npe_head(
        self, ckpt_path: str, freeze: bool = False, flow_prefix: str = "model.flow."
    ) -> None:
        print(
            f"[JointVMIMNLELightningModule] Loading pretrained NPE head from {ckpt_path}..."
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        src_state = checkpoint.get("state_dict", checkpoint)
        load_partial_weights(
            target_module=self.npe_flow,
            source_state_dict=src_state,
            prefix=flow_prefix,
            freeze=freeze,
            verbose=True,
        )

    def _load_pretrained_nle_head(
        self, ckpt_path: str, freeze: bool = False, flow_prefix: str = "model.flow."
    ) -> None:
        print(
            f"[JointVMIMNLELightningModule] Loading pretrained NLE head from {ckpt_path}..."
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        src_state = checkpoint.get("state_dict", checkpoint)
        load_partial_weights(
            target_module=self.nle_flow,
            source_state_dict=src_state,
            prefix=flow_prefix,
            freeze=freeze,
            verbose=True,
        )
