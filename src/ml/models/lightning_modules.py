import torch
import torch.nn as nn
import pytorch_lightning as pl
from copy import deepcopy
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    LambdaLR,
    ReduceLROnPlateau,
    SequentialLR,
)
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm
from types import MethodType
from functools import partial
from typing import Dict


def load_partial_weights(
    target_module: nn.Module,
    source_state_dict: Dict[str, torch.Tensor],
    prefix: str = "",
    freeze: bool = False,
    verbose: bool = True,
):
    """Safely load a subset of weights into target_module.

    - Optionally strips a prefix from keys in `source_state_dict`.
    - Only loads keys that exist in `target_module.state_dict()` and
      whose shapes match.
    - Optionally freezes the loaded module's parameters.
    """
    target_state = target_module.state_dict()
    loaded_weights = {}

    for k, v in source_state_dict.items():
        # Strip prefix if requested
        if prefix and k.startswith(prefix):
            local_key = k[len(prefix) :]
        elif not prefix:
            local_key = k
        else:
            continue  # key does not belong to this submodule

        # Match key and shape
        if local_key in target_state:
            if v.shape == target_state[local_key].shape:
                loaded_weights[local_key] = v
            elif verbose:
                print(
                    f"[load_partial_weights] Skipping {local_key}: shape mismatch "
                    f"{tuple(v.shape)} vs {tuple(target_state[local_key].shape)}"
                )

    missing, unexpected = target_module.load_state_dict(loaded_weights, strict=False)

    if verbose and loaded_weights:
        print(
            f"[load_partial_weights] Loaded {len(loaded_weights)} keys into {target_module.__class__.__name__}. "
            f"Missing: {len(missing)}, unexpected (ignored): {len(unexpected)}"
        )

    if freeze:
        for p in target_module.parameters():
            p.requires_grad = False
        # target_module.eval()
     # if we are loading and the model has .only_return_mu, we need to set it to True
    if hasattr(target_module, "only_return_mu"):
        target_module.only_return_mu = True


class BaseLightningModule(pl.LightningModule):
    def __init__(self, model, loss_fn, lr=0.0001, scheduler_type='cosine', element_names=None, optimizer_kwargs = {}, scheduler_kwargs= {}, freeze_CNN=False, **kwargs):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn  # Loss function is now dynamic
        self.lr = lr
        self.scheduler_type = scheduler_type
        self.best_checkpoints = []
        self.element_names = element_names if element_names is not None else []
        self.loss_name = ''
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        self.freeze_CNN = freeze_CNN

    def forward(self, x, cond):
        return self.model(x)

    def compute_loss(self, preds, y):
        """Generic loss computation method to be overridden if needed."""
        return self.loss_fn(preds, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x, cond=y)
        loss = self.compute_loss(preds, y)
        self.log(f"train_{self.loss_name}", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x, cond=y)
        loss = self.compute_loss(preds, y)
        self.log(f"val_{self.loss_name}", loss, prog_bar=True)
        self.log_custom_evals(preds, y)
        return loss
    
    def log_custom_evals(self, preds, y):
        pass

    def configure_optimizers(self):
        # paper is weight_decay=3.53e-7
        default_optimizer_kwargs = dict(weight_decay=3.53e-7, betas=(0.5, 0.999))
        optimizer_kwargs = {**default_optimizer_kwargs, **self.optimizer_kwargs}
        print(optimizer_kwargs)
        if self.freeze_CNN:
            for p in self.model.embedding_net.parameters():
                p.requires_grad = False
            from torch.optim import Adam
            optimizer = Adam(self.model.flow.parameters(), lr=self.lr, **optimizer_kwargs)
        else:
            optimizer = AdamW(self.model.parameters(), lr=self.lr, **optimizer_kwargs)

        # --- Scheduler selection ---
        sched_type = str(self.scheduler_type).lower()

        # Plateau keeps its own behaviour
        if sched_type == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.95,
                patience=10,
                threshold=1e-4,
                min_lr=1e-9,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": f"val_{self.loss_name}",
                },
            }

        # Try to get max_epochs from trainer (falls back to 500)
        max_epochs = getattr(self.trainer, "max_epochs", None) or 500
        warmup_frac = float(self.scheduler_kwargs.get("warmup_frac", 0.05))
        warmup_epochs = max(1, int(warmup_frac * max_epochs))
        cosine_epochs = max_epochs - warmup_epochs

        # Cyclic schedule: step-based warmup + CyclicLR
        if sched_type == "cyclic":
            total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
            if total_steps is not None and max_epochs > 0:
                steps_per_epoch = max(1, total_steps // max_epochs)
            else:
                steps_per_epoch = getattr(self.trainer, "num_training_batches", None) or 1

            warmup_steps = max(1, warmup_epochs * steps_per_epoch)

            def lr_lambda_warmup(step_idx: int):
                return min(float(step_idx + 1) / float(max(1, warmup_steps)), 1.0)

            warmup = LambdaLR(optimizer, lr_lambda=lr_lambda_warmup)

            cyclic_period_steps = int(self.scheduler_kwargs.get("cyclic_period_steps", 2000))
            half_period = max(1, cyclic_period_steps // 2)

            cyclic = CyclicLR(
                optimizer,
                base_lr=self.lr * float(self.scheduler_kwargs.get("cyclic_min_factor", 0.05)),
                max_lr=self.lr,
                step_size_up=half_period,
                step_size_down=cyclic_period_steps - half_period,
                mode="triangular",
                cycle_momentum=False,
            )

            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cyclic],
                milestones=[warmup_steps],
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        # Epoch-based warmup + cosine / exponential style decay (default path)
        def lr_lambda_warmup(epoch: int):
            return float(epoch + 1) / float(max(1, warmup_epochs))

        warmup = LambdaLR(optimizer, lr_lambda=lr_lambda_warmup)

        if sched_type in {"cosine", "cosine_simple"}:
            # New default: cosine decay to 1/5 of initial LR by end of training
            min_factor = float(self.scheduler_kwargs.get("min_factor", 0.2))
            eta_min = self.lr * min_factor
            # decay only over the post-warmup portion
            T_max = max(1, cosine_epochs)
            cosine = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
            main = cosine
        elif sched_type in {"exp", "exponential"}:
            # Simple exponential decay after warmup
            gamma = float(self.scheduler_kwargs.get("gamma", 0.98))

            def lr_lambda_exp(epoch: int):
                if epoch < warmup_epochs:
                    return 1.0
                t = epoch - warmup_epochs
                return gamma ** t

            main = LambdaLR(optimizer, lr_lambda=lr_lambda_exp)
        else:
            # Fallback: no second stage, just run warmup then keep LR constant
            main = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, main],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


class RegressionLightningModule(BaseLightningModule):
    def __init__(self, model, lr=0.0001, scheduler_type='cosine', batch_size=32, element_names=None, **kwargs):
        super().__init__(model, loss_fn=torch.nn.MSELoss(), lr=lr, scheduler_type=scheduler_type, batch_size=batch_size, element_names=element_names)
        self.loss_name = "loss"
    def log_r2_eval(self, preds, y):
        """Logs R² scores for each output element if applicable."""
        if not self.element_names:
            return
        
        preds_np = preds.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        for i, element in enumerate(self.element_names):
            r2 = r2_score(y_np[:, i], preds_np[:, i])
            self.log(f"R²_{element}", r2, prog_bar=False)
    def log_custom_evals(self, preds, y):
        self.log_r2_eval(preds, y)

    def compute_loss(self, preds, y):
        return torch.log(self.loss_fn(preds, y))  # Log-transformed MSE loss

class GaussianLightningModule(BaseLightningModule):
    def __init__(self, model, lr=0.0001, scheduler_type='cosine', batch_size=32, element_names=None, num_outputs=2, **kwargs):
        super().__init__(model, loss_fn=torch.nn.MSELoss(), lr=lr, scheduler_type=scheduler_type, batch_size=batch_size, element_names=element_names)
        self.loss_name = "loss"
        self.num_outputs = num_outputs
    def log_r2_eval(self, preds, y):
        """Logs R² scores for each output element if applicable."""
        if not self.element_names:
            return
        
        preds_np = preds.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        for i, element in enumerate(self.element_names):
            r2 = r2_score(y_np[:, i], preds_np[:, i])
            self.log(f"R²_{element}", r2, prog_bar=False)
    def log_custom_evals(self, preds, y):
        self.log_r2_eval(preds, y)

    def compute_loss(self, preds, y):
        y_NN = preds[:, :self.num_outputs]
        e_NN = preds[:, self.num_outputs:]
        loss1 = torch.mean((y_NN - y)**2,                axis=0)
        loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
        loss  = torch.mean(torch.log(loss1) + torch.log(loss2))
        return loss


from sbi.neural_nets.net_builders import build_nsf, build_maf, build_zuko_nsf
from torch.optim import Adam, AdamW
import torch
from sbi import utils as utils

import torch.nn as nn

class _CondEmbeddingFlow(nn.Module):
    """Wrapper that delegates to an embedding_net and a conditional flow.

    The embedding_net is responsible for turning high-dimensional data_dict
    into a fixed-size representation used as the flow condition.
    """
    def __init__(self, embedding_net: nn.Module, flow: nn.Module):
        super().__init__()
        self.embedding_net = embedding_net if embedding_net is not None else nn.Identity()
        self.flow = flow
    
    def encode(self, y):
        return self.embedding_net(y)
    
    def log_prob(self, x, y):
        """Return elementwise log p(x|y) for a batch.

        Keeps the original convention required by nflows: the underlying
        flow sees a leading dimension (typically num_chains) in addition to
        the batch dimension. We therefore unsqueeze x along dim=0 before
        calling into the flow. Any wrapper (e.g. MultipleFlow) must respect
        this convention.
        """
        y_emb = self.embedding_net(y)
        x = x.unsqueeze(0)  # [1, batch, dim_x]
        return self.flow.log_prob(x, y_emb)
    
    def sample(self, shape, y, **kwargs):
        y_emb = self.embedding_net(y)
        return self.flow.sample(shape, y_emb, **kwargs)
    
    def sample_batched(self, shape, y, **kwargs):
        y_emb = self.embedding_net(y)
        return self.flow.sample_batched(shape, y_emb, **kwargs)

class MultipleFlow(nn.Module):
    """Container for an ensemble of flow models with a single-flow-like API.

    Assumes each flow implements ``log_prob(x, y)`` and ``sample`` / ``sample_batched``
    with the same semantics as used in ``_CondEmbeddingFlow``. In particular,
    ``log_prob`` expects x to already have the leading nflows dimension
    (typically added via ``unsqueeze(0)`` in ``_CondEmbeddingFlow``) and
    returns a tensor whose leading dimension corresponds to that.
    """

    def __init__(self, flows: list[nn.Module]):
        super().__init__()
        if len(flows) == 0:
            raise ValueError("MultipleFlow requires at least one flow.")
        self.flows = nn.ModuleList(flows)

    def log_prob(self, x, y, **kwargs):
        """Return ensemble-averaged log_prob over flows for a batch.

        x: tensor with leading nflows dim, e.g. [1, batch, dim_x]. Each
        underlying flow is called with the same x, and we average the
        resulting log-probabilities across the ensemble dimension while
        preserving all original dimensions so that callers (e.g.
        ``_CondEmbeddingFlow`` and the Lightning modules) see the same
        shape as for a single flow.
        """
        # Each flow.log_prob returns something like [1, batch] (or
        # [1, batch, ...]); we stack along a new ensemble dim=0 and
        # average over that, then squeeze the ensemble dim back out.
        log_probs = [flow.log_prob(x, y, **kwargs) for flow in self.flows]
        stacked = torch.stack(log_probs, dim=0)  # [n_flows, 1, batch, ...]
        mean_lp = stacked.mean(dim=0)            # [1, batch, ...]
        return mean_lp

    def sample(self, shape, y, **kwargs):
        samples = [flow.sample(shape, y, **kwargs) for flow in self.flows]
        samples = torch.stack(samples, dim=0)  # [n_flows, *shape, dim_x]
        return samples.mean(dim=0)

    def sample_batched(self, shape, y, **kwargs):
        """Batched sampling if underlying flows expose ``sample_batched``.

        Falls back to ``sample`` for flows without ``sample_batched``.
        """
        samples = []
        for flow in self.flows:
            if hasattr(flow, "sample_batched"):
                samples.append(flow.sample_batched(shape, y, **kwargs))
            else:
                samples.append(flow.sample(shape, y, **kwargs))
        samples = torch.stack(samples, dim=0)
        return samples.mean(dim=0)

from sbi.neural_nets.estimators import ConditionalDensityEstimator
from sbi.samplers.rejection import rejection
from sbi.utils.sbiutils import within_support

from collections.abc import Mapping

class ConditionDict(dict):
    """
    Dict-like that additionally exposes a .shape so that .shape[0]
    returns the batch dimension of the first value.
    """
    def __init__(self, data: Mapping):
        if not isinstance(data, Mapping):
            raise TypeError("ConditionDict expects a mapping.")
        super().__init__(data)

    @property
    def shape(self):
        try:
            first_val = next(iter(self.values()))
        except StopIteration:
            raise ValueError("ConditionDict is empty; cannot infer .shape[0].")
        if not hasattr(first_val, "shape"):
            raise AttributeError("First value has no .shape attribute.")
        # Return a 1-tuple so code using .shape[0] works.
        return first_val.shape

    def copy(self):
        # Ensure copies keep the subclass (some libs call .copy())
        return ConditionDict(self)

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

    def gen_samples(self, num_samples, x):
        samples = rejection.accept_reject_sample(
            proposal=self.sample,
            accept_reject_fn=lambda theta: within_support(self.prior, theta),
            num_samples=num_samples,
            show_progress_bars=False,
            max_sampling_batch_size=self.max_sampling_batch_size,
            proposal_sampling_kwargs={"condition": ConditionDict(x)},
            alternative_method="build_posterior(..., sample_with='mcmc')",
        )[0]
        return samples


class NDELightningModule(BaseLightningModule):
    flow_type_map = {"nsf": build_nsf, "maf": build_maf, 'zuko_nsf': build_zuko_nsf}

    def __init__(
        self,
        model,
        conditioning_dim,
        inference_dim,
        redundancy_dim = 0,
        lr=0.0001,
        scheduler_type='cosine',
        test_dataloader=None,
        flow_type='nsf',
        num_extra_blocks=None,
        flow_kwargs= {},
        **kwargs,
    ):
        super().__init__(model, loss_fn=None, lr=lr, scheduler_type=scheduler_type, **kwargs)
        # Keep embedding net separate from the flow; expected to implement compress().
        self.embedding_net = model if model is not None else nn.Identity()
        self.conditioning_dim = conditioning_dim
        self.inference_dim = inference_dim
        self.redundancy_dim = redundancy_dim
        self.build_flow = self.flow_type_map[flow_type]
        if 'zuko' in flow_type:
            self.flow_kwargs = flow_kwargs
        else:
            self.flow_kwargs = {"conditional_dim": self.conditioning_dim, "use_batch_norm": False, **flow_kwargs}
        self.test_dataloader = test_dataloader
        self.loss_name = "log_prob"
        self.set_up_model()
        self.test_loss_values = []

    def set_up_model(self):
        """Builds the flow model and wraps it together with the embedding encoder.

        The flow itself works on latent representations; the encoder is
        responsible for compressing the high-dimensional data_dict into a
        fixed-size vector of dimension `conditioning_dim`.
        """
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
        """Return the latent representation used as condition for the flow.

        Delegates to the wrapper's encode(), which mirrors the internal
        behaviour of _CondEmbeddingFlow and keeps a single source of truth
        for how embeddings are computed from data_dict.
        """
        return self.model.encode(data_dict)

    def load_from_checkpoint(self, checkpoint_path):
        """Loads model weights from a given checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))  # Adjust device as needed
        print("Overwriting model weights from checkpoint:", checkpoint_path)
        self.load_state_dict(checkpoint['state_dict'])  # Ensure the key matches the saved checkpoint format
    
    def build_posterior_object(self):
        prior = utils.BoxUniform(low=0 * torch.ones(self.inference_dim), high=1. * torch.ones(self.inference_dim), device="cuda")
        density_estimator = PatchedConditionalDensityEstimator(self.model, prior)
        return density_estimator

    def generate_samples(self, dummy_loader, num_samples=10000):
        test_y, test_x = self.test_dataloader.dataset.tensors
        posterior = self.build_posterior_object()
        test_y = torch.tensor(test_y).to('cuda', dtype=torch.float32).unsqueeze(1)
        theta0s = []
        samples = []
        num_tarp_samples = len(test_x)
        for i in tqdm(range(num_tarp_samples), desc="Sampling", total=num_tarp_samples):
            x= test_x[i]
            y = test_y[i]
            try:
                x_samples = posterior.sample((num_samples,), x=y, show_progress_bars=False)
            except:
                pass
            theta0s.append(x)
            samples.append(x_samples)
        theta0s = torch.stack(theta0s)
        samples = torch.stack(samples).permute(1, 0, 2)
        return theta0s, samples
    

    def generate_samples_batched(self, test_dataloader, num_samples=10000):
        """
        Generates samples from the model using the test dataloader. Doesn't work!
        
        Args:
            test_dataloader (DataLoader): The test dataloader.
            num_samples (int): Number of samples to generate.
        
        Returns:
            torch.Tensor: Generated samples.
        """
        posterior = self.build_posterior_object()
        all_samples = []
        for batch in tqdm(test_dataloader, desc="Sampling"):
            y, x = batch
            x_samples = posterior.sample_batched((num_samples,), x=y, show_progress_bars=False)
            all_samples.append(x_samples)
        all_samples = torch.cat(all_samples, dim=0)
        return all_samples


    def compute_loss(self, preds, y):   
        """Uses log probability as the loss for density estimation."""
        return -preds.mean()  # Negative log-likelihood loss

    def forward(self, x, cond=None):
        """Evaluate log p(x | cond).

        cond is a high-dimensional data dict that will be compressed by
        the encoder via its compress()/forward method before feeding
        into the normalising flow.
        """
        return self.model.log_prob(x, cond)

    # Override steps to pass (theta|data) ordering correctly
    def training_step(self, batch, batch_idx):
        data_dict, theta = batch  # dataset yields (data, cosmo)
        preds = self.forward(theta, cond=data_dict)
        loss = self.compute_loss(preds, theta)
        self.log(f"train_{self.loss_name}", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data_dict, theta = batch
        preds = self.forward(theta, cond=data_dict)
        loss = self.compute_loss(preds, theta)
        self.log(f"val_{self.loss_name}", loss, prog_bar=True)
        self.log_custom_evals(preds, theta)
        return loss

    def on_validation_epoch_end(self):
        """Logs custom evaluation metrics at the end of each validation epoch."""
        if self.test_dataloader is None:
            return  # Skip if no test dataloader is provided

        self.model.eval()  # Ensure model is in eval mode
        with torch.no_grad():
            avg_log_prob = self.compute_avg_log_prob()
        if avg_log_prob is not None:
            self.test_loss_values.append(avg_log_prob)

    def compute_avg_log_prob(self):
        """Computes the average log probability over the test dataset."""
        predictions = []
        for batch in self.test_dataloader:
            batch = self.transfer_batch_to_device(batch, self.device, 0)
            data_dict, theta = batch
            predictions.append(self.forward(theta, data_dict))
        all_log_probs = torch.cat(predictions, dim=0)  # Collect predictions
        avg_log_prob = all_log_probs.mean().item()
        return avg_log_prob

    def log_custom_evals(self, preds, y):
        if len(self.test_loss_values) > 0:
            self.log("test_log_prob", self.test_loss_values.pop())

    def _load_pretrained_band_encoder(self, ckpt_path: str, freeze: bool, band_prefix: str = 'band_encoder.') -> None:
        """Load weights for a bandpower encoder inside the embedding_net.

        This assumes that `self.embedding_net` has an attribute that contains
        the desired band-encoder submodule (e.g. `band_encoder`), and that the
        checkpoint corresponds to a bandpower+NDE model where the band encoder
        weights live under `band_prefix` (e.g. 'band_encoder.').
        """
        print(f"[NDELightningModule] Loading pretrained band encoder from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        src_state = checkpoint.get("state_dict", checkpoint)

        # Heuristic: look for a candidate submodule on the embedding net
        band_module = None
        for attr in ("band_encoder", "band_model"):
            if hasattr(self.embedding_net, attr):
                band_module = getattr(self.embedding_net, attr)
                print(f"[NDELightningModule] Using embedding_net.{attr} as band encoder target.")
                break
        if band_module is None:
            print("[NDELightningModule] Warning: embedding_net has no band encoder submodule; skipping band loading.")
            return

        load_partial_weights(
            target_module=band_module,
            source_state_dict=src_state,
            prefix=band_prefix,
            freeze=freeze,
            verbose=True,
        )
    
    def _load_pretrained_flow(self, ckpt_path: str, freeze: bool, flow_prefix: str = 'model.flow.') -> None:
        """Load weights for the normalising flow from a given checkpoint."""
        print(f"[NDELightningModule] Loading pretrained flow from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        src_state = checkpoint.get("state_dict", checkpoint)

        load_partial_weights(
            target_module=self.flow,
            source_state_dict=src_state,
            prefix=flow_prefix,
            freeze=freeze,
            verbose=True,
        )

    def _load_pretrained_cnn_backbone(
        self,
        ckpt_path: str,
        freeze: bool,
        backbone_prefix: str = 'shared_cnn.backbone.',
        target_prefix: str = '',
    ) -> None:
        """Load weights for the CNN backbone used inside the embedding network.

        This is designed to support scenarios like:
          * Source: KidsO3NorthSouthEmbedding.shared_cnn.backbone
            (UNetO3StyleEncoder.backbone -> UNetStyleEncoder)
          * Target: KidsCombinedCNNTransformer.shared_cnn
            (a full UNetStyleEncoder without a .backbone attribute)

        In that case, `backbone_prefix` should point at the source backbone
        inside the checkpoint (e.g. 'shared_cnn.backbone.') while
        `target_prefix` can be left as the default empty string so that the
        keys are loaded directly into the target UNetStyleEncoder module.

        The actual loading is delegated to `load_partial_weights`, which only
        loads matching keys and can optionally freeze the loaded parameters.
        """
        print(f"[NDELightningModule] Loading pretrained CNN backbone from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        src_state = checkpoint.get("state_dict", checkpoint)

        # Locate the CNN module that will receive the backbone weights.
        cnn_module = None
        # 1) Embedding net itself might already be the CNN (with .backbone)
        if hasattr(self.embedding_net, "patch_encoder"):
            shared = getattr(self.embedding_net, "patch_encoder")
            # If the shared CNN exposes a .backbone, load into that
            if hasattr(shared, "shared_cnn"):
                cnn_module = shared.shared_cnn

        if cnn_module is None:
            print("[NDELightningModule] Warning: could not find a CNN backbone on embedding_net; skipping backbone loading.")
            return

        # Optionally re-map keys by changing their target_prefix. This is
        # handled by adjusting the prefix we strip and what we expect inside
        # cnn_module.state_dict(): we pre-strip backbone_prefix on the source
        # side, and if target_prefix is non-empty we add it back before
        # matching against cnn_module's keys.
        if target_prefix:
            # Build a temporary state dict with remapped keys
            remapped = {}
            for k, v in src_state.items():
                if backbone_prefix and k.startswith(backbone_prefix):
                    inner_key = k[len(backbone_prefix):]
                    remapped[target_prefix + inner_key] = v
            src_state = remapped
            # We already consumed backbone_prefix in the remap, so clear it
            backbone_prefix = ''

        load_partial_weights(
            target_module=cnn_module,
            source_state_dict=src_state,
            prefix=backbone_prefix,
            freeze=freeze,
            verbose=True,
        )



import torch
import torch.nn as nn

class KLDRegularisedNDELightningModule(NDELightningModule):
    def __init__(
        self,
        *args,
        kl_weight: float = 1e-4,  
        kl_min: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.kl_weight = kl_weight
        self.kl_min = kl_min

    def _latent_stats(self, data_dict):
        """Return (mu, logvar) from the encoder.

        Assumes the embedding_net is a KidsInferenceEncoder with
        KL-aware behaviour:
          * in train mode: forward(data) -> logvar
          * in eval  mode: forward(data) -> mu

        We temporarily toggle training mode on the encoder to obtain
        both mu and logvar without changing the global Lightning mode.
        """
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

        # 1) Get stats from encoder
        mu, logvar = self._latent_stats(data_dict)

        # 2) KL term
        kl = self._kl_divergence(mu, logvar)

        # 3) Reparameterize
        if stage == "train":
            z = self._reparameterize(mu, logvar)
        else:
            z = mu

        # 4) Flow loss
        theta_for_flow = theta.unsqueeze(0)
        log_prob = self.model.flow.log_prob(theta_for_flow, z)
        nll = -log_prob.mean()

        total = nll + self.kl_weight * kl

        metrics = {
            f"{stage}_{self.loss_name}": nll,
            f"{stage}_kl": kl,
            f"{stage}_weighted_kl": self.kl_weight * kl,
        }
        for name, value in metrics.items():
            on_step = stage == "train"
            self.log(name, value, prog_bar=(name.endswith(self.loss_name)), on_step=on_step, on_epoch=True)

        return total, log_prob, theta

    def training_step(self, batch, batch_idx):
        loss, log_prob, theta = self._shared_step(batch, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_prob, theta = self._shared_step(batch, stage="val")
        self.log_custom_evals(log_prob, theta)
        return loss


class EnsembleNDELightningModule(NDELightningModule):
    """NDELightningModule that uses an ensemble of N independent flows.

    The encoder / embedding network is shared; only the flow is replicated N
    times. During training and evaluation, the negative log-likelihood (NLL)
    is computed from the ensemble-averaged log-probability, implemented via
    ``MultipleFlow``.
    """

    def __init__(
        self,
        *args,
        num_flows: int = 4,
        **kwargs,
    ):
        self.num_flows = num_flows
        super().__init__(*args, **kwargs)

    def set_up_model(self):
        """Build ``num_flows`` separate flows and wrap them in MultipleFlow.

        The API matches ``NDELightningModule.set_up_model`` but uses a
        ``MultipleFlow`` container so that the rest of the code can treat the
        ensemble as a single flow object.
        """
        y_dataset = torch.randn(10, self.conditioning_dim)
        x_dataset = torch.randn(10, self.inference_dim)

        flows = []
        for _ in range(self.num_flows):
            flow = self.build_flow(
                x_dataset,
                y_dataset,
                num_transforms=5,
                z_score_x=None,
                z_score_y=None,
                embedding_net=nn.Identity(),
                hidden_features=self.conditioning_dim,
                **self.flow_kwargs,
            )
            flows.append(flow)

        self.flow = MultipleFlow(flows)
        # Reuse the same embedding network and wrapper API as the base class.
        self.model = _CondEmbeddingFlow(self.embedding_net, self.flow)