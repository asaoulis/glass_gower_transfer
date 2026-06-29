from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm

from sbi import utils as sbi_utils
from sbi.neural_nets.net_builders import build_maf, build_zuko_nsf

from src.ml.models.custom_sbi import build_maf_rqs, build_nsf

from .base import BaseLightningModule
from .estimators import PatchedConditionalDensityEstimator
from .flows import _CondEmbeddingFlow


class NDELightningModule(BaseLightningModule):
    """NPE-style neural density estimator LightningModule (posterior p(theta|x))."""

    flow_type_map = {
        "nsf": build_nsf,
        "maf": build_maf,
        "rqs": build_maf_rqs,
        "zuko_nsf": build_zuko_nsf,
    }

    def __init__(
        self,
        model,
        conditioning_dim,
        inference_dim,
        redundancy_dim=0,
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
        self.conditioning_dim = conditioning_dim
        self.inference_dim = inference_dim
        self.redundancy_dim = redundancy_dim
        self.build_flow = self.flow_type_map[flow_type]

        flow_kwargs = flow_kwargs or {}
        if "zuko" in str(flow_type).lower():
            self.flow_kwargs = dict(flow_kwargs)
        else:
            self.flow_kwargs = {
                "conditional_dim": self.conditioning_dim,
                "use_batch_norm": False,
                **dict(flow_kwargs),
            }

        self.test_dataloader = test_dataloader
        self.loss_name = "log_prob"
        self.set_up_model()
        self.test_loss_values = []

    def set_up_model(self):
        y_dataset = torch.randn(10, self.conditioning_dim)
        x_dataset = torch.randn(10, self.inference_dim)
        hidden_features = self.flow_kwargs.pop("hidden_features", self.conditioning_dim)

        flow = self.build_flow(
            x_dataset,
            y_dataset,
            num_transforms=5,
            z_score_x=None,
            z_score_y=None,
            embedding_net=nn.Identity(),
            hidden_features=hidden_features,
            **self.flow_kwargs,
        )
        self.flow = flow
        self.model = _CondEmbeddingFlow(self.embedding_net, self.flow)

    def compress(self, data_dict):
        return self.model.encode(data_dict)

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        print("Overwriting model weights from checkpoint:", checkpoint_path)
        state_dict = checkpoint["state_dict"]
        # Checkpoints saved from a torch.compile'd submodule carry an `_orig_mod.` prefix on the
        # compiled keys (e.g. ...shared_cnn.backbone._orig_mod.patch_embed.weight). When the model
        # is rebuilt UNcompiled (e.g. the embeddings pipeline / eval load a compile-trained hybrid),
        # a strict load_state_dict would reject every such key. Strip the compile artifact so the
        # keys align; keep the load strict so a genuine architecture mismatch still surfaces.
        if not any("_orig_mod." in k for k in self.state_dict()):
            state_dict = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict)

    def build_posterior_object(self, prior=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        if hasattr(self.model, "embedding_net") and hasattr(
            self.model.embedding_net, "only_return_mu"
        ):
            self.model.embedding_net.only_return_mu = True

        prior = sbi_utils.BoxUniform(
            low=0 * torch.ones(self.inference_dim, device=device),
            high=1.0 * torch.ones(self.inference_dim, device=device),
            device=device,
        )

        density_estimator = PatchedConditionalDensityEstimator(
            self.model,
            prior,
            input_shape=(self.inference_dim,),
            condition_shape=(self.conditioning_dim,),
        )
        return density_estimator

    @torch.no_grad()
    def get_representations_and_samples(
        self,
        *,
        num_samples: int = 10_000,
        batch_size: int = 8,
        prior=None,
        return_theta0s: bool = True,
        return_z: bool = True,
        return_representation: bool = True,
        **posterior_kwargs,
    ):
        """Return intermediate representations and posterior samples on the test set.

        Mirrors `generate_samples`, but additionally returns:
          - `representation`: encoder features before the final head
          - `z`: compressed outputs after applying the head (flow context)

        Implementation details
        ----------------------
        We avoid running the embedding net twice by:
          1) computing `rep = self.model.get_representation(data)`
          2) computing `z = self.model.embedding_net.head(rep)`
        then sampling using the posterior conditioned on `z`.

        Returns
        -------
        dict with keys among: theta0s, representation, z, samples
        where samples has shape [num_samples, N, D].
        """

        device = "cuda" if torch.cuda.is_available() else "cpu"

        posterior = self.build_posterior_object(prior=prior)

        theta0s_items = []
        rep_items = []
        z_items = []

        # Encode test set once.
        for data_dict, theta in tqdm(self.test_dataloader, desc="Encoding test set"):
            if isinstance(data_dict, dict):
                data_dict = {k: v.to(device) for k, v in data_dict.items()}
            else:
                data_dict = data_dict.to(device)

            if return_theta0s:
                theta0s_items.append(theta.detach().cpu())

            rep = self.model.get_representation(data_dict)
            if return_representation:
                rep_items.append(rep.detach().cpu())

            if return_z:
                emb = getattr(self.model, "embedding_net", None)
                head = getattr(emb, "head", None) if emb is not None else None
                if head is None:
                    raise AttributeError(
                        "get_representations_and_samples requires embedding_net.head to exist; "
                        "either use a KidsInferenceEncoder-style compressor or extend this method."
                    )
                z = head(rep)
                z_items.append(z.detach().cpu())
            else:
                # still need z for sampling
                emb = getattr(self.model, "embedding_net", None)
                head = getattr(emb, "head", None) if emb is not None else None
                if head is None:
                    raise AttributeError(
                        "get_representations_and_samples requires embedding_net.head to exist for sampling."
                    )
                z = head(rep)
                z_items.append(z.detach().cpu())

        out: dict[str, torch.Tensor] = {}
        if return_theta0s:
            out["theta0s"] = torch.cat(theta0s_items, dim=0)
        if return_representation:
            out["representation"] = torch.cat(rep_items, dim=0) if rep_items else torch.empty(0)

        z_conds = torch.cat(z_items, dim=0) if z_items else torch.empty(0)
        if return_z:
            out["z"] = z_conds

        # Sample from posterior conditioned on z.
        posterior.prior.to(device)
        posterior.to(device)
        samples = []
        for i in tqdm(range(0, len(z_conds), batch_size), desc="Generating samples"):
            z_batch = z_conds[i : i + batch_size].to(device)
            s = posterior.gen_samples(num_samples=num_samples, x=z_batch, **posterior_kwargs)
            if isinstance(s, tuple):
                s = s[0]
            samples.append(s.detach().cpu())
        out["samples"] = torch.cat(samples, dim=1) if samples else torch.empty(0)
        return out

    @torch.no_grad()
    def generate_samples(self, num_samples=10000, **kwargs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        posterior = self.build_posterior_object()

        theta0s, z_conds = [], []
        for data_dict, theta in tqdm(self.test_dataloader, desc="Encoding test set"):
            if isinstance(data_dict, dict):
                data_dict = {k: v.to(device) for k, v in data_dict.items()}
            else:
                data_dict = data_dict.to(device)
            z = posterior.compress(data_dict)
            theta0s.append(theta)
            z_conds.append(z.cpu())

        theta0s = torch.cat(theta0s, dim=0)
        z_conds = torch.cat(z_conds, dim=0)

        device = "cuda"
        posterior.prior.to(device)
        posterior.to(device)
        batch_size = 8
        samples = []
        for i in tqdm(range(0, len(z_conds), batch_size), desc="Generating samples"):
            z_batch = z_conds[i : i + batch_size].to(device)
            samples_i = posterior.gen_samples(num_samples=num_samples, x=z_batch)
            samples.append(samples_i.cpu())
        samples = torch.cat(samples, dim=1)
        return theta0s, samples

    def generate_samples_batched(self, test_dataloader, num_samples=10000):
        posterior = self.build_posterior_object()
        all_samples = []
        for batch in tqdm(test_dataloader, desc="Sampling"):
            y, x = batch
            x_samples = posterior.sample_batched(
                (num_samples,), x=y, show_progress_bars=False
            )
            all_samples.append(x_samples)
        all_samples = torch.cat(all_samples, dim=0)
        return all_samples

    def compute_loss(self, preds, y):
        return -preds.mean()

    def forward(self, x, cond=None):
        return self.model.log_prob(x, cond)

    def training_step(self, batch, batch_idx):
        data_dict, theta = batch
        preds = self.forward(theta, cond=data_dict)
        loss = self.compute_loss(preds, theta)
        self.log(
            f"train_{self.loss_name}",
            loss,
            prog_bar=True,
            sync_dist=self.is_distributed,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        data_dict, theta = batch
        preds = self.forward(theta, cond=data_dict)
        loss = self.compute_loss(preds, theta)
        self.log(
            f"val_{self.loss_name}",
            loss,
            prog_bar=True,
            sync_dist=self.is_distributed,
        )
        self.log_custom_evals(preds, theta)
        return loss

    def on_validation_epoch_end(self):
        if self.test_dataloader is None:
            return

        self.model.eval()
        with torch.no_grad():
            avg_log_prob = self.compute_avg_log_prob()
        if avg_log_prob is not None:
            self.test_loss_values.append(avg_log_prob)

    def compute_avg_log_prob(self):
        predictions = []
        for batch in self.test_dataloader:
            batch = self.transfer_batch_to_device(batch, self.device, 0)
            data_dict, theta = batch
            predictions.append(self.forward(theta, data_dict).reshape(-1))
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
