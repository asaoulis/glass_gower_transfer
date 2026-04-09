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
from torch.optim.lr_scheduler import LambdaLR, CyclicLR, SequentialLR, ExponentialLR

import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm
from types import MethodType
from functools import partial
from typing import Dict
from joblib import Parallel, delayed
from tqdm import tqdm
import torch

from typing import Dict
import torch
import torch.nn as nn


def load_partial_weights(
    target_module: nn.Module,
    source_state_dict: Dict[str, torch.Tensor],
    prefix: str = "",
    freeze: bool = False,
    verbose: bool = True,
    error_on_mismatch: bool = False,
):
    """
    Safely load a subset of weights into target_module.

    - Optionally strips a prefix from keys in `source_state_dict`.
    - Only loads keys that exist in `target_module.state_dict()` and
      whose shapes match.
    - Reports unused / missing / mismatched keys.
    - Can raise if weights are not fully consumed.
    """

    target_state = target_module.state_dict()

    loaded_weights = {}
    used_source_keys = set()
    skipped_shape = []
    skipped_missing = []

    # ------------------------
    # Match keys
    # ------------------------
    for k, v in source_state_dict.items():

        # Strip prefix if requested
        if prefix:
            if not k.startswith(prefix):
                continue
            local_key = k[len(prefix):]
        else:
            local_key = k

        # Check key exists
        if local_key not in target_state:
            skipped_missing.append(local_key)
            continue

        # Check shape
        if v.shape != target_state[local_key].shape:
            skipped_shape.append(
                (local_key, tuple(v.shape), tuple(target_state[local_key].shape))
            )
            continue

        loaded_weights[local_key] = v
        used_source_keys.add(k)

    # ------------------------
    # Load
    # ------------------------
    missing, unexpected = target_module.load_state_dict(
        loaded_weights, strict=False
    )

    # ------------------------
    # Unused source keys
    # ------------------------
    unused_source = set(source_state_dict.keys()) - used_source_keys

    # ------------------------
    # Reporting
    # ------------------------
    if verbose:

        print(
            f"[load_partial_weights] {target_module.__class__.__name__}"
        )
        print(f"  Loaded keys: {len(loaded_weights)}")
        print(f"  Missing in target after load: {len(missing)}")
        print(f"  Unexpected during load: {len(unexpected)}")
        print(f"  Unused source keys: {len(unused_source)}")
        print(f"  Shape mismatches: {len(skipped_shape)}")
        print(f"  Missing target keys: {len(skipped_missing)}")

        if len(loaded_weights) == 0:
            print("⚠️  WARNING: No weights were loaded!")

        if skipped_shape:
            print("⚠️  Shape mismatches:")
            for k, s1, s2 in skipped_shape[:10]:
                print(f"   {k}: {s1} vs {s2}")

        if unused_source and prefix:
            print(
                "⚠️  Some source keys not used — prefix may be wrong."
            )

    # ------------------------
    # Error mode
    # ------------------------
    if error_on_mismatch:

        problems = (
            len(unused_source)
            + len(skipped_shape)
            + len(skipped_missing)
            + len(missing)
        )

        if problems > 0:
            raise RuntimeError(
                f"load_partial_weights mismatch detected: "
                f"{len(loaded_weights)} loaded, "
                f"{len(unused_source)} unused, "
                f"{len(skipped_shape)} shape mismatch, "
                f"{len(missing)} missing"
            )

    # ------------------------
    # Freeze
    # ------------------------
    if freeze:
        for p in target_module.parameters():
            p.requires_grad = False

    # ------------------------
    # Special flag
    # ------------------------
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
        # Optional per-module learning rates for pretrained components.
        # These are expected to be dicts of {module_name: lr} and will be
        # populated from config in utils.build_model if used.
        self.pretrained_band_lrs = getattr(self, "pretrained_band_lrs", None)
        self.pretrained_backbone_lrs = getattr(self, "pretrained_backbone_lrs", None)
        # Will hold per-param-group "base" LRs used by schedulers
        self._optimizer_group_base_lrs = None
        # Flag to indicate if we are running in a distributed setting.
        # This is read from the config via utils.build_model and used to
        # control sync_dist in self.log calls.
        self.is_distributed = getattr(self.hparams, "is_distributed", False)

    def forward(self, x, cond):
        return self.model(x)

    def compute_loss(self, preds, y):
        """Generic loss computation method to be overridden if needed."""
        return self.loss_fn(preds, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x, cond=y)
        loss = self.compute_loss(preds, y)
        self.log(f"train_{self.loss_name}", loss, prog_bar=True, sync_dist=self.is_distributed)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x, cond=y)
        loss = self.compute_loss(preds, y)
        self.log(f"val_{self.loss_name}", loss, prog_bar=True, sync_dist=self.is_distributed)
        self.log_custom_evals(preds, y)
        return loss
    
    def log_custom_evals(self, preds, y):
        pass

    def _build_param_groups(self, base_optimizer_cls):
        """Build parameter groups, allowing per-module LRs for pretrained parts."""
        
        param_groups = []
        assigned_params = set()

        def add_module_group(mod_name: str, lr_value: float):
            try:
                module = self.model.get_submodule(mod_name)
            except AttributeError:
                # Fallback for older pytorch versions or simple getattr
                if not hasattr(self.model, mod_name):
                    print(f"Warning: Module {mod_name} not found.")
                    return
                module = getattr(self.model, mod_name)
                
            # Collect params for this module
            group_params = [p for p in module.parameters() if p.requires_grad]
            
            if not group_params:
                return

            param_groups.append({"params": group_params, "lr": lr_value})
            
            # Add these param objects to our set so we don't add them again later
            for p in group_params:
                assigned_params.add(p)

        if isinstance(self.pretrained_band_lrs, dict):
            for name, lr_val in self.pretrained_band_lrs.items():
                add_module_group(name, lr_val)

        if isinstance(self.pretrained_backbone_lrs, dict):
            for name, lr_val in self.pretrained_backbone_lrs.items():
                add_module_group(name, lr_val)

        base_params = [
            p for p in self.model.parameters() 
            if p.requires_grad and p not in assigned_params
        ]
        
        if base_params:
            param_groups.append({"params": base_params, "lr": self.lr})

        optimizer = base_optimizer_cls(param_groups, lr=self.lr, **self.optimizer_kwargs)
        # Cache the base LR for each param group so schedulers don't lose them
        self._optimizer_group_base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        return optimizer

    def configure_optimizers(self):
        # 1. Setup Optimizer
        default_optimizer_kwargs = dict(weight_decay=3.53e-7, betas=(0.5, 0.999))
        optimizer_kwargs = {**default_optimizer_kwargs, **self.optimizer_kwargs}
        
        # Logic to build param groups or single group
        if (isinstance(self.pretrained_band_lrs, dict) and self.pretrained_band_lrs) or \
        (isinstance(self.pretrained_backbone_lrs, dict) and self.pretrained_backbone_lrs):
            self.optimizer_kwargs = optimizer_kwargs
            optimizer = self._build_param_groups(AdamW)
        else:
            optimizer = AdamW(self.model.parameters(), lr=self.lr, **optimizer_kwargs)

        # CRITICAL: Capture the actual base LRs from the optimizer groups
        # This list corresponds to the 'max' LR each group should hit after warmup.
        base_lrs = [pg['lr'] for pg in optimizer.param_groups]

        # 2. Timing logic
        max_epochs = getattr(self.trainer, "max_epochs", 200)
        # Estimate total steps for the 'step' interval schedulers
        total_steps = self.trainer.estimated_stepping_batches
        if total_steps is None or total_steps <= 0:
            # Fallback if trainer hasn't attached yet
            steps_per_epoch = 1000 
            total_steps = steps_per_epoch * max_epochs
        else:
            steps_per_epoch = total_steps // max_epochs

        warmup_frac = float(self.scheduler_kwargs.get("warmup_frac", 0.05))
        warmup_steps = max(1, int(total_steps * warmup_frac))

        # 3. Build Schedulers
        # --- STAGE 1: Warmup (Common to all) ---
        # We use a lambda that scales from 0.1 to 1.0. 
        # LambdaLR applies this to the INITIAL LRs in the optimizer.
        warmup_start_factor = float(self.scheduler_kwargs.get("warmup_start_factor", 0.1))
        def warmup_lambda(step):
            return warmup_start_factor + (1.0 - warmup_start_factor) * min(1.0, step / warmup_steps)
        
        warmup_sched = LambdaLR(optimizer, lr_lambda=warmup_lambda)

        # --- STAGE 2: Main Schedule ---
        sched_type = str(self.scheduler_type).lower()

        if sched_type == "cyclic":
            cyclic_period_steps = int(self.scheduler_kwargs.get("cyclic_period_steps", 2000))
            min_factor = float(self.scheduler_kwargs.get("cyclic_min_factor", 0.05))
            
            main_sched = CyclicLR(
                optimizer,
                base_lr=[lr * min_factor for lr in base_lrs],
                max_lr=base_lrs,  # The peak is our original defined LRs
                step_size_up=cyclic_period_steps // 2,
                step_size_down=cyclic_period_steps // 2,
                mode="triangular",
                cycle_momentum=False
            )
            interval = "step"
            milestone = warmup_steps

        elif sched_type in ["exp", "exponential"]:
            gamma = float(self.scheduler_kwargs.get("gamma", 0.98))
            step_gamma = gamma ** (1 / steps_per_epoch)
            main_sched = ExponentialLR(optimizer, gamma=step_gamma)
            interval = "step"
            milestone = warmup_steps

        elif sched_type in ["cyclic_exp", "exp_cyclic"]:
            # Cyclic LR whose peak decays exponentially over time
            cyclic_period_steps = int(self.scheduler_kwargs.get("cyclic_period_steps", 2000))
            min_factor = float(self.scheduler_kwargs.get("cyclic_min_factor", 0.05))
            gamma = float(self.scheduler_kwargs.get("gamma", 0.98))
            # Convert epoch-wise gamma to step-wise, as you did above
            step_gamma = gamma ** (1 / steps_per_epoch)

            def combined_lambda(global_step: int):
                # Exponential envelope
                exp_factor = step_gamma ** global_step

                # Triangle wave in [0, 1]
                if cyclic_period_steps <= 0:
                    cyc = 1.0
                else:
                    phase = (global_step % cyclic_period_steps) / cyclic_period_steps  # [0, 1)
                    if phase < 0.5:
                        cyc01 = phase / 0.5           # ramp up 0 -> 1
                    else:
                        cyc01 = (1.0 - phase) / 0.5    # ramp down 1 -> 0
                    # Map to [min_factor, 1]
                    cyc = min_factor + (1.0 - min_factor) * cyc01

                return exp_factor * cyc

            # LambdaLR applies this factor to each group's *base* LR
            main_sched = LambdaLR(optimizer, lr_lambda=combined_lambda)
            interval = "step"
            milestone = warmup_steps

        else:
            # Fallback: Constant LR
            main_sched = LambdaLR(optimizer, lr_lambda=lambda x: 1.0)
            interval = "step"
            milestone = warmup_steps

        # 4. Chain them
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, main_sched],
            milestones=[milestone]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,
            },
        }
    def _load_pretrained_band_encoder(self, ckpt_path: str, freeze: bool, band_prefix: str = 'model.embedding_net.') -> str | None:
        """Load weights for a bandpower encoder inside the embedding_net.

        Returns the dotted submodule path (relative to ``self.model``)
        of the band encoder that was used, so that callers can set a
        dedicated learning rate for that module if desired. If no
        suitable submodule is found, returns ``None``.
        """
        print(f"[NDELightningModule] Loading pretrained band encoder from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        src_state = checkpoint.get("state_dict", checkpoint)

        # Heuristic: look for a candidate submodule on the embedding net
        band_module = None
        band_module_name = None
        for attr in ("band_encoder", "band_model"):
            if hasattr(self.embedding_net, attr):
                band_module = getattr(self.embedding_net, attr)
                band_module_name = f"embedding_net.{attr}"
                print(f"[NDELightningModule] Using {band_module_name} as band encoder target.")
                break

        if band_module is None:
            print("[NDELightningModule] Warning: embedding_net has no band encoder submodule; skipping band loading.")
            return None

        load_partial_weights(
            target_module=band_module,
            source_state_dict=src_state,
            prefix=band_prefix,
            freeze=freeze,
            verbose=True,
        )
        if freeze:
            self.embedding_net.freeze_band = True  # Set to eval mode if we're freezing, to disable dropout/batchnorm updates

        return band_module_name

    def _load_pretrained_flow(self, ckpt_path: str, freeze: bool, flow_prefix: str = 'model.flow.') -> None:
        """Load weights for the normalising flow from a given checkpoint."""
        print(f"[NDELightningModule] Loading pretrained flow from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        src_state = checkpoint.get("state_dict", checkpoint)

        load_partial_weights(
            target_module=self.model.flow,
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
    ) -> str | None:
        """Load weights for the CNN backbone used inside the embedding network.

        Returns the dotted submodule path (relative to ``self.model``)
        of the CNN module that received the weights, or ``None`` if
        no suitable module was found.
        """
        print(f"[NDELightningModule] Loading pretrained CNN backbone from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        src_state = checkpoint.get("state_dict", checkpoint)

        # Locate the CNN module that will receive the backbone weights.
        cnn_module = None
        cnn_module_name = None
        if hasattr(self.embedding_net, "patch_encoder"):
            shared = getattr(self.embedding_net, "patch_encoder")
            if hasattr(shared, "shared_cnn"):
                cnn_module = shared.shared_cnn
                cnn_module_name = "embedding_net.patch_encoder.shared_cnn"

        if cnn_module is None:
            print("[NDELightningModule] Warning: could not find a CNN backbone on embedding_net; skipping backbone loading.")
            return None

        if target_prefix:
            remapped = {}
            for k, v in src_state.items():
                if backbone_prefix and k.startswith(backbone_prefix):
                    inner_key = k[len(backbone_prefix):]
                    remapped[target_prefix + inner_key] = v
            src_state = remapped
            backbone_prefix = ''

        load_partial_weights(
            target_module=cnn_module,
            source_state_dict=src_state,
            prefix=backbone_prefix,
            freeze=freeze,
            verbose=True,
        )

        return cnn_module_name

    def _load_pretrained_embedding_net(
        self,
        ckpt_path: str,
        freeze: bool,
        patch_prefix: str = 'model.embedding_net.',
    ) -> str | None:
        """Load weights for the patch encoder used inside the embedding network.

        Returns the dotted submodule path (relative to ``self.model``)
        of the patch encoder module that received the weights, or ``None``
        if no suitable module was found.
        """
        print(f"[NDELightningModule] Loading pretrained patch encoder from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        src_state = checkpoint.get("state_dict", checkpoint)

        # Locate the patch encoder module.
        patch_module = self.embedding_net
        patch_module_name = "embedding_net"

        if patch_module is None:
            print("[NDELightningModule] Warning: embedding_net has no patch_encoder submodule; skipping patch encoder loading.")
            return None

        load_partial_weights(
            target_module=patch_module,
            source_state_dict=src_state,
            prefix=patch_prefix,
            freeze=freeze,
            verbose=True,
        )

        return patch_module_name


class RegressionLightningModule(BaseLightningModule):
    """Simple MSE regression on cosmological parameters.

    Follows the same data conventions as NDELightningModule:
    - Batches are (data_dict, theta) tuples.
    - The embedding network compresses data_dict into a latent vector.
    - A linear head maps the latent vector to cosmological parameter predictions.
    """

    def __init__(
        self,
        model,
        conditioning_dim,
        inference_dim,
        lr=0.0001,
        scheduler_type='exp',
        test_dataloader=None,
        **kwargs,
    ):
        # Pop NDE-specific kwargs that don't apply to regression
        kwargs.pop('flow_type', None)
        kwargs.pop('flow_kwargs', None)
        kwargs.pop('num_extra_blocks', None)
        kwargs.pop('redundancy_dim', None)
        kwargs.pop('num_flows', None)
        super().__init__(model, loss_fn=torch.nn.MSELoss(), lr=lr, scheduler_type=scheduler_type, **kwargs)
        self.loss_name = "loss"
        self.embedding_net = model if model is not None else nn.Identity()
        self.conditioning_dim = conditioning_dim
        self.inference_dim = inference_dim
        self.test_dataloader = test_dataloader
        self.test_loss_values = []

        # Linear regression head: latent -> cosmo params
        self.regression_head = nn.Linear(conditioning_dim, inference_dim)

    def forward(self, data_dict, cond=None):
        """Compress data_dict and predict cosmological parameters."""
        z = self.embedding_net(data_dict)
        # If encoder returns (mu, logvar) tuple, take only mu
        if isinstance(z, tuple):
            z = z[0]
        return self.regression_head(z)

    def compute_loss(self, preds, y):
        return self.loss_fn(preds, y)

    def training_step(self, batch, batch_idx):
        data_dict, theta = batch
        preds = self.forward(data_dict)
        loss = self.compute_loss(preds, theta)
        self.log(f"train_{self.loss_name}", loss, prog_bar=True, sync_dist=self.is_distributed)
        return loss

    def validation_step(self, batch, batch_idx):
        data_dict, theta = batch
        preds = self.forward(data_dict)
        loss = self.compute_loss(preds, theta)
        self.log(f"val_{self.loss_name}", loss, prog_bar=True, sync_dist=self.is_distributed)
        self.log_custom_evals(preds, theta)
        return loss

    def on_validation_epoch_end(self):
        if self.test_dataloader is None:
            return
        self.eval()
        with torch.no_grad():
            avg_loss = self._compute_avg_test_loss()
        if avg_loss is not None:
            self.test_loss_values.append(avg_loss)

    def _compute_avg_test_loss(self):
        losses = []
        for batch in self.test_dataloader:
            batch = self.transfer_batch_to_device(batch, self.device, 0)
            data_dict, theta = batch
            preds = self.forward(data_dict)
            loss = self.compute_loss(preds, theta)
            losses.append(loss.item())
        if not losses:
            return None
        return np.mean(losses)

    def log_custom_evals(self, preds, y):
        # Log test loss from previous epoch end
        if len(self.test_loss_values) > 0:
            self.log("test_loss", self.test_loss_values.pop(), sync_dist=self.is_distributed)
        # Log per-element R² scores and MSEs
        if self.element_names:
            # convert to f32 for numpy detach
            preds = preds.float()
            preds_np = preds.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            for i, element in enumerate(self.element_names):
                if i < preds_np.shape[-1] and i < y_np.shape[-1]:
                    r2 = r2_score(y_np[:, i], preds_np[:, i])
                    self.log(f"R2_{element}", r2, prog_bar=False, sync_dist=self.is_distributed)
                    self.log(f"MSE_{element}", np.mean((y_np[:, i] - preds_np[:, i])**2), prog_bar=False, sync_dist=self.is_distributed)

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        print("Overwriting model weights from checkpoint:", checkpoint_path)
        self.load_state_dict(checkpoint['state_dict'])

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
            self.log(f"R²_{element}", r2, prog_bar=False, sync_dist=self.is_distributed)
    def log_custom_evals(self, preds, y):
        self.log_r2_eval(preds, y)

    def compute_loss(self, preds, y):
        y_NN = preds[:, :self.num_outputs]
        e_NN = preds[:, self.num_outputs:]
        loss1 = torch.mean((y_NN - y)**2,                axis=0)
        loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
        loss  = torch.mean(torch.log(loss1) + torch.log(loss2))
        return loss


from sbi.neural_nets.net_builders import build_maf, build_zuko_nsf
from src.ml.models.custom_sbi import build_nsf, build_maf_rqs
from torch.optim import Adam, AdamW
from sbi import utils as utils

import torch.nn as nn

class _CondEmbeddingFlow(nn.Module):
    """Minimal original wrapper linking an embedding network and a flow.

    The embedding_net takes the conditioning data and returns a fixed-size
    representation used as context for the flow. All dtype handling is left
    to the caller / Trainer (no bf16-specific casting here).
    """

    def __init__(self, embedding_net: nn.Module, flow: nn.Module):
        super().__init__()
        self.embedding_net = embedding_net if embedding_net is not None else nn.Identity()
        self.flow = flow

    def encode(self, y):
        return self.embedding_net(y)
    
    def get_representation(self, y):
        return self.embedding_net.get_representation(y)

    def log_prob(self, x, y):
        y_emb = self.embedding_net(y)
        x = x.unsqueeze(0)
        return self.flow.log_prob(x, y_emb)

    def latent_log_prob(self, x, y_emb):
        x = x.unsqueeze(0)
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
from sbi.inference.posteriors import MCMCPosterior
from sbi.inference.potentials.likelihood_based_potential import likelihood_estimator_based_potential

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


class ConditionList(list):
    """List-like condition container with a tensor-like ``.shape`` attribute.

    This is used when each ensemble member requires its own conditioned input
    (e.g. member-specific scaled observations for NLE ensembles), while SBI
    utilities still expect ``x.shape[0]`` to return the batch dimension.
    """

    @property
    def shape(self):
        if len(self) == 0:
            raise ValueError("ConditionList is empty; cannot infer .shape[0].")
        first_val = self[0]
        if not hasattr(first_val, "shape"):
            raise AttributeError("First element has no .shape attribute.")
        return first_val.shape

    def copy(self):
        return ConditionList(self)


def _move_nested_to_device(x, device):
    """Move nested tensors/structures to ``device`` while preserving container type."""
    if hasattr(x, "to"):
        return x.to(device)
    if isinstance(x, Mapping):
        return type(x)({k: _move_nested_to_device(v, device) for k, v in x.items()})
    if isinstance(x, tuple):
        return tuple(_move_nested_to_device(v, device) for v in x)
    if isinstance(x, list):
        return type(x)([_move_nested_to_device(v, device) for v in x])
    return x


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

    # --- helpers to access encoder / latent flow ---
    def compress(self, data_dict):
        """Return latent representation used as condition for the flow.

        Delegates to the wrapped _CondEmbeddingFlow, which in turn calls
        the underlying embedding network. Keeps a single source of truth
        for how the conditioning representation is computed.
        """
        if hasattr(self.net, "encode"):
            return self.net.encode(data_dict)
        # Fallback: assume standard forward behaviour.
        return self.net.embedding_net(data_dict)

    def latent_log_prob(self, x, y_emb):
        """Log prob when conditions are already embedded (no re-encoding)."""
        if hasattr(self.net, "latent_log_prob"):
            return self.net.latent_log_prob(x, y_emb)
        # Fallback: flow expects (x, y_emb) directly.
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
            num_xos=cond.shape[0]
        )[0]
        return samples


class PatchedLikelihoodEstimator(ConditionalDensityEstimator):
    """ConditionalDensityEstimator wrapper for neural likelihood estimation.

    This mirrors PatchedConditionalDensityEstimator but exposes a
    log_likelihood-style API and a ``gen_samples`` method that uses
    ``MCMCPosterior.sample_batched`` under the hood.
    """

    def __init__(self, model, prior, input_shape=(1,), condition_shape=(1,), fixed_parameters=None):
        # ``model`` is expected to implement .log_prob(x, y)
        super().__init__(model, input_shape=input_shape, condition_shape=condition_shape)
        self.prior = prior
        self.max_sampling_batch_size = 10_000
        self.fixed_parameters = fixed_parameters

    def _check_condition_shape(self, condition):
        # Handled by underlying model / ConditionDict, no-op here.
        pass

    def _check_input_shape(self, input):
        # Handled externally, keep behaviour minimal.
        pass

    def log_prob(self, x, condition):
        # remove leading nflows dim if present, as the underlying model expects [batch, dim_x]
        if hasattr(x, "ndim") and x.ndim > 2 and x.shape[0] == 1:
            x = x.squeeze(0)
        return self.net.log_prob(x, condition)
    
    def sample(self, num_samples, condition):
        return self.net.sample(num_samples, condition)

    def loss(self, x, y):
        # Negative log-likelihood.
        return -self.log_prob(x, y).mean()

    def log_likelihood(self, x, theta):
        """Alias with explicit (x, theta) semantics used by MCMC potential fns."""
        return self.log_prob(x, theta)
    def sample_single_batch(
        self,
        num_samples,
        test_data,
        test_cosmo,
        mcmc_kwargs,
    ):
        x = test_data

        samples = self._gen_samples(
            num_samples=num_samples,
            x=x,
            use_latent=False,
            **mcmc_kwargs,
        )

        return test_cosmo, samples

    def gen_samples(self, num_samples, x, use_latent=True, num_jobs=10, **mcmc_kwargs):
        """Generate samples from the posterior p(theta | x) using MCMC.

        This uses sbi's MCMCPosterior with a likelihood_estimator_based_potential
        that wraps this likelihood estimator. The ``use_latent`` flag is
        currently unused but could be used in the future to switch between
        sampling in latent space vs. original parameter space if desired.
        """
        # use Parallel to wrap _gen_samples if num_jobs > 1, otherwise call directly
        # move everything to cpu
        self.to("cpu")
        self.prior.to("cpu")
        if num_jobs > 1:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=num_jobs, backend="loky")(
                delayed(self._gen_samples)(
                    num_samples=num_samples,
                    x=x_single.unsqueeze(0).to("cpu"),  # Each job gets a single data point
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
    def _gen_samples(
        self,
        num_samples: int,
        x,
        use_latent,
        **mcmc_kwargs,
    ):
        device = next(self.net.parameters()).device
        x_batch = _move_nested_to_device(x, device)

        method = mcmc_kwargs.pop("method", "slice_np_vectorized")
        num_chains = mcmc_kwargs.pop("num_chains", 4)
        thin = mcmc_kwargs.pop("thin", 1)
        warmup_steps = mcmc_kwargs.pop("warmup_steps", 500)
        show_progress_bars = mcmc_kwargs.pop("show_progress_bars", False)

        sample_shape = (num_samples,)

        # 1. Get the base potential and transform
        potential, tf = likelihood_estimator_based_potential(
            self,
            self.prior,
            x_o=None,
        )

        prior_to_use = self.prior

        # 2. If fixed_parameters were provided, apply the conditioning
        if self.fixed_parameters:
            # Assuming self.condition_shape[0] holds the total number of parameters
            total_dim = self.condition_shape[0]
            
            condition = torch.zeros(total_dim, device=device)
            fixed_indices = [idx for idx, _ in self.fixed_parameters]
            dims_to_sample = [i for i in range(total_dim) if i not in fixed_indices]
            
            # Populate the condition tensor with the fixed values
            for idx, val in self.fixed_parameters:
                condition[idx] = val
                
            # Overwrite potential, tf, and prior with the conditioned versions
            potential, tf, prior_to_use = conditional_potential(
                potential_fn=potential,
                theta_transform=tf,
                prior=self.prior,
                condition=condition,
                dims_to_sample=dims_to_sample,
            )

        # 3. Create the posterior with the (potentially restricted) variables
        posterior = MCMCPosterior(
            potential_fn=potential,
            proposal=prior_to_use, 
            theta_transform=tf,
            method=method,
            num_chains=num_chains,
            num_workers=1,          # <- IMPORTANT
            thin=thin,
            warmup_steps=warmup_steps,
            device=device,
            **mcmc_kwargs,
        )

        # 4. Sample
        samples = posterior.sample_batched(
            sample_shape=sample_shape,
            x=x_batch,
            show_progress_bars=show_progress_bars,
        )
        
        samples_cpu = samples.cpu()

        return samples_cpu

class NDELightningModule(BaseLightningModule):
    flow_type_map = {"nsf": build_nsf, "maf": build_maf, "rqs":build_maf_rqs, 'zuko_nsf': build_zuko_nsf}

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
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Adjust device as needed
        print("Overwriting model weights from checkpoint:", checkpoint_path)
        self.load_state_dict(checkpoint['state_dict'])  # Ensure the key matches the saved checkpoint format
    
    def build_posterior_object(self, prior=None):
        """Build a neural posterior object (NPE-style).

        By default this uses a BoxUniform prior on [0, 1] for each
        inference dimension, matching the previous behaviour. A custom
        prior can be passed in via the ``prior`` argument.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        if hasattr(self.model, "embedding_net") and hasattr(self.model.embedding_net, "only_return_mu"):
            self.model.embedding_net.only_return_mu = True

        if prior is None:
            prior = utils.BoxUniform(
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
    def generate_samples(self, num_samples=10000, **kwargs):
        """Generate posterior samples in two stages:

        1) Compress the full test set once with the encoder.
        2) Run sampling only on the latent representations using the flow.
        """
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

        # now move everything to cpu
        device = "cuda"
        posterior.prior.to(device)
        posterior.to(device)
        batch_size = 8
        samples = []
        for i in tqdm(range(0, len(z_conds), batch_size),
                    desc="Generating samples"):
            z_batch = z_conds[i:i + batch_size].to(device)   # [B, z_dim]

            samples_i = posterior.gen_samples(
                num_samples=num_samples,
                x=z_batch
            )                                                 # [num_samples, B, dim]
            samples.append(samples_i.cpu())
        samples = torch.cat(samples, dim=1)  # concat over datapoints
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
        self.log(f"train_{self.loss_name}", loss, prog_bar=True, sync_dist=self.is_distributed)
        return loss

    def validation_step(self, batch, batch_idx):
        data_dict, theta = batch
        preds = self.forward(theta, cond=data_dict)
        loss = self.compute_loss(preds, theta)
        self.log(f"val_{self.loss_name}", loss, prog_bar=True, sync_dist=self.is_distributed)
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
            predictions.append(self.forward(theta, data_dict).reshape(-1))
        all_log_probs = torch.cat(predictions, dim=0)  # Collect predictions
        avg_log_prob = -all_log_probs.mean().item()
        return avg_log_prob

    def log_custom_evals(self, preds, y):
        if len(self.test_loss_values) > 0:
            self.log("test_log_prob", self.test_loss_values.pop(), sync_dist=self.is_distributed)

import torch
import torch.nn as nn

class KLDRegularisedNDELightningModule(NDELightningModule):
    def __init__(
        self,
        *args,
        kl_weight: float = 1e-3,  
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
            self.log(name, value, prog_bar=(name.endswith(self.loss_name)), on_step=on_step, on_epoch=True, sync_dist=self.is_distributed)

        return total, log_prob, theta

    def training_step(self, batch, batch_idx):
        loss, log_prob, theta = self._shared_step(batch, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_prob, theta = self._shared_step(batch, stage="val")
        self.log_custom_evals(log_prob, theta)
        return loss

    def compute_avg_log_prob(self):
        """Computes the average log probability over the test dataset."""
        predictions = []
        for batch in self.test_dataloader:
            batch = self.transfer_batch_to_device(batch, self.device, 0)
            data_dict, theta = batch
            mu, logvar = self._latent_stats(data_dict)
            log_prob = self.model.latent_log_prob(theta, mu)
            predictions.append(log_prob.reshape(-1))
        all_log_probs = torch.cat(predictions, dim=0)  # Collect predictions
        avg_log_prob = -all_log_probs.mean().item()
        return avg_log_prob

    def log_custom_evals(self, preds, y):
        if len(self.test_loss_values) > 0:
            self.log("test_log_prob", self.test_loss_values.pop(), sync_dist=self.is_distributed)


class LikelihoodNDELightningModule(NDELightningModule):
    """Minimal subclass of NDELightningModule for neural likelihood training.

    Interprets the flow output as a likelihood p(x | theta) instead of a
    posterior p(theta | x). Posterior sampling is delegated to
    ``PatchedLikelihoodEstimator.gen_samples`` so that the API matches
    NDELightningModule.generate_samples (i.e. exposing ``gen_samples``).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, cond=None):
        """Evaluate log p(x | theta).

        In NLE, ``x`` corresponds to data and ``cond`` to parameters theta.
        """
        x_emb = self.model.embedding_net(x)
        x_emb = x_emb.unsqueeze(0)
        return self.model.flow.log_prob(x_emb, cond)

    def training_step(self, batch, batch_idx):
        x, theta = batch
        preds = self.forward(x, cond=theta)
        loss = self.compute_loss(preds, theta)
        self.log(f"train_{self.loss_name}", loss, prog_bar=True, sync_dist=self.is_distributed)
        return loss

    def validation_step(self, batch, batch_idx):
        x, theta = batch
        preds = self.forward(x, cond=theta)
        loss = self.compute_loss(preds, theta)
        self.log(f"val_{self.loss_name}", loss, prog_bar=True, sync_dist=self.is_distributed)
        self.log_custom_evals(preds, theta)
        return loss
    def compute_avg_log_prob(self):
        """Computes the average log probability over the test dataset."""
        predictions = []
        for batch in self.test_dataloader:
            batch = self.transfer_batch_to_device(batch, self.device, 0)
            data_dict, theta = batch
            predictions.append(self.forward(data_dict, theta).reshape(-1))
        all_log_probs = torch.cat(predictions, dim=0)  # Collect predictions
        avg_log_prob = -all_log_probs.mean().item()
        return avg_log_prob


    def generate_samples(
        self,
        num_samples=2_000,
        num_jobs=36,
        backend="loky",
        prior=None,
        **mcmc_kwargs,
    ):
        """
        Parallelize over batches using joblib, while each batch uses
        fast single-worker vectorized MCMC.
        """
        posterior = self.build_posterior_object(prior=prior)
        # move everything to cpu for joblib
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
        
        results = list(tqdm(
            jobs, 
            total=len(self.test_dataloader), 
            desc="Sampling batches"
        ))

        theta0s, samples = zip(*results)

        theta0s = torch.cat(theta0s, dim=0)
        samples = torch.cat(samples, dim=1)  # [num_samples, total_batch, dim]

        return theta0s, samples

    def build_posterior_object(self, prior=None, fixed_parameters=None):
        """Build and return a PatchedLikelihoodEstimator with gen_samples()."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        if hasattr(self.model, "embedding_net") and hasattr(self.model.embedding_net, "only_return_mu"):
            self.model.embedding_net.only_return_mu = True

        if prior is None:
            prior = utils.BoxUniform(
                low=0 * torch.ones(self.conditioning_dim, device=device),
                high=1.0 * torch.ones(self.conditioning_dim, device=device),
                device=device,
            )

        likelihood_estimator = PatchedLikelihoodEstimator(
            model=self.model,
            prior=prior,
            input_shape=(self.inference_dim,),
            condition_shape=(self.conditioning_dim,),
            fixed_parameters=fixed_parameters,  # <-- Pass fixed parameters to init
        )
        return likelihood_estimator

class EnsembleNDELightningModule(pl.LightningModule):
    """Evaluation-time ensemble of separately-trained NDELightningModules.

    Key constraints:
      - Each ensemble member is a full LightningModule (typically NDELightningModule)
        loaded independently (so embedding nets may differ).
      - We expose the subset of the NDELightningModule interface relied upon by
        evaluation utilities:
          * compute_avg_log_prob()
          * generate_samples(num_samples=..., **kwargs) -> (theta0s, samples)
      - Log prob is averaged over members.
      - Sampling is distributed approximately equally across members, then the
        resulting samples are concatenated and shuffled along the sample axis.

    This module is NOT meant for training.
    """

    def __init__(self, members: list[pl.LightningModule]):
        super().__init__()
        if not members:
            raise ValueError("EnsembleNDELightningModule requires at least one member.")
        self.members = nn.ModuleList(members)

        # IMPORTANT: do not overwrite member.test_dataloader; each member must
        # keep the loader/scalers from its own split.
        self.test_dataloader = getattr(members[0], "test_dataloader", None)
        self.loss_name = getattr(members[0], "loss_name", "log_prob")

    def _resolve_device(self):
        # Prefer Lightning's device if set; otherwise fall back to cuda if available.
        try:
            return self.device
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def _get_theta0s_from_loader(self):
        """Collect theta0s without invoking member.generate_samples() (avoids side-effects)."""
        if self.test_dataloader is None:
            raise ValueError("EnsembleNDELightningModule.test_dataloader is None")
        theta0s = []
        for _, theta in self.test_dataloader:
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
    def compute_avg_log_prob(self):
        vals = []
        for m in self.members:
            if hasattr(m, "compute_avg_log_prob"):
                vals.append(float(m.compute_avg_log_prob()))
            else:
                raise AttributeError("Ensemble member lacks compute_avg_log_prob")
        return float(np.mean(vals))

    @torch.no_grad()
    def generate_samples(self, num_samples=10000, **kwargs):
        # Collect theta0s once. We use member[0]'s test loader as the canonical
        # ordering. Other members must be built with the same underlying test set
        # (only scaling differs), so theta0s should match.
        theta0s = self._get_theta0s_from_loader()

        # Do NOT force all members to use the same loader: each member's
        # test_dataloader contains its own scaling based on its split_seed.
        # We only sanity-check that loaders exist and have consistent lengths.
        for idx, m in enumerate(self.members):
            if getattr(m, "test_dataloader", None) is None:
                raise ValueError(f"Ensemble member {idx} has no test_dataloader")
            if len(m.test_dataloader.dataset) != len(self.test_dataloader.dataset):
                raise ValueError(
                    "Ensemble members have different test dataset lengths; "
                    "cannot safely ensemble their posteriors."
                )

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

        samples = torch.cat(parts, dim=0)  # [num_samples, N, dim]
        perm = torch.randperm(samples.shape[0])
        samples = samples[perm]
        return theta0s, samples


class _EnsembleLikelihoodModel(nn.Module):
    """Averages member log-likelihoods for use inside MCMC potentials."""

    def __init__(self, members: list[pl.LightningModule], reduction: str = "logmeanexp"):
        super().__init__()
        if reduction not in {"mean_log_prob", "logmeanexp"}:
            raise ValueError("reduction must be one of {'mean_log_prob', 'logmeanexp'}")
        self.members = nn.ModuleList(members)
        self.reduction = reduction

    def log_prob(self, x, theta):
        # x can be a single batch tensor (shared for all members) or a
        # ConditionList containing one batch per member.
        if isinstance(x, ConditionList):
            if len(x) != len(self.members):
                raise ValueError(
                    f"ConditionList length ({len(x)}) does not match number of members ({len(self.members)})."
                )
            xs = x
        else:
            xs = [x] * len(self.members)

        log_probs = [m.forward(x_i, cond=theta) for m, x_i in zip(self.members, xs)]
        stacked = torch.stack(log_probs, dim=0)

        if self.reduction == "logmeanexp":
            return torch.logsumexp(stacked, dim=0) - np.log(len(self.members))

        # Default: arithmetic average of log-probabilities.
        return stacked.mean(dim=0)


class EnsembleLikelihoodNDELightningModule(pl.LightningModule):
    """Evaluation-time ensemble for likelihood NDEs.

    Unlike NPE ensembles, this class performs a single MCMC run against an
    ensemble-averaged likelihood by combining member log_prob evaluations
    inside the potential function.
    """

    def __init__(self, members: list[pl.LightningModule]):
        super().__init__()
        if not members:
            raise ValueError("EnsembleLikelihoodNDELightningModule requires at least one member.")
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

    def _iter_member_batches(self):
        self._validate_member_loaders()
        return zip(*(m.test_dataloader for m in self.members))

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
        for member_batches in self._iter_member_batches():
            batch_lps = []
            for m, batch in zip(self.members, member_batches):
                x_i, theta_i = batch
                x_i = _move_nested_to_device(x_i, target_device)
                theta_i = _move_nested_to_device(theta_i, target_device)
                m.to(target_device)
                m.eval()
                batch_lps.append(m.forward(x_i, cond=theta_i))
            all_log_probs.append(torch.stack(batch_lps, dim=0).mean(dim=0).reshape(-1))

        return float(-torch.cat(all_log_probs, dim=0).mean().item())

    def build_posterior_object(
        self,
        prior=None,
        fixed_parameters=None,
        reduction: str = "logmeanexp",
    ):
        first = self.members[0]
        device = self._resolve_device()

        for m in self.members:
            m.eval()
            if hasattr(m, "model") and hasattr(m.model, "embedding_net") and hasattr(m.model.embedding_net, "only_return_mu"):
                m.model.embedding_net.only_return_mu = True

        if prior is None:
            prior = utils.BoxUniform(
                low=0 * torch.ones(first.conditioning_dim, device=device),
                high=1.0 * torch.ones(first.conditioning_dim, device=device),
                device=device,
            )

        ensemble_likelihood = _EnsembleLikelihoodModel(list(self.members), reduction=reduction)
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
        prior=None,
        reduction: str = "logmeanexp",
        **mcmc_kwargs,
    ):
        posterior = self.build_posterior_object(
            prior=prior,
            reduction=reduction,
            fixed_parameters=mcmc_kwargs.pop("fixed_parameters", None),
        )

        target_device = self._resolve_device()
        posterior.to(target_device)
        posterior.prior.to(target_device)

        theta0s, samples = [], []
        total_batches = len(self.test_dataloader)
        for member_batches in tqdm(self._iter_member_batches(), total=total_batches, desc="Sampling ensemble batches"):
            x_per_member = []
            theta_ref = None
            for idx, (m, batch) in enumerate(zip(self.members, member_batches)):
                x_i, theta_i = batch
                x_per_member.append(_move_nested_to_device(x_i, target_device))
                if idx == 0:
                    theta_ref = theta_i

            theta0s.append(theta_ref)
            x_condition = ConditionList(x_per_member)
            samples_i = posterior._gen_samples(
                num_samples=num_samples,
                x=x_condition,
                use_latent=False,
                **mcmc_kwargs,
            )
            samples.append(samples_i.cpu())

        theta0s = torch.cat(theta0s, dim=0)
        samples = torch.cat(samples, dim=1)
        return theta0s, samples

class JointVMIMNLELightningModule(BaseLightningModule):
    """Joint training of NPE (posterior) and NLE (likelihood) heads.

    - Shared embedding network (encoder) producing latent representation z.
    - Two conditional density estimators (flows):
        * NPE head models p(theta | z)
        * NLE head models p(z | theta)

    Loss:
        L = L_npe + nle_weighting * L_nle

    Notes
    -----
    This module is intended for scenarios where a single encoder is trained
    jointly with both posterior and likelihood objectives.
    """

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
        scheduler_type='cosine',
        test_dataloader=None,
        flow_type='nsf',
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
        # Keep behaviour consistent with NDELightningModule regarding zuko flows.
        if 'zuko' in str(flow_type).lower():
            self.flow_kwargs = dict(flow_kwargs)
        else:
            self.flow_kwargs = {"use_batch_norm": False, **dict(flow_kwargs)}

        self.test_dataloader = test_dataloader
        self.loss_name = "joint"

        self._test_npe_loss_values = []
        self._test_nle_loss_values = []

        self.set_up_model()

    def set_up_model(self):
        """Build and store two flows plus wrappers.

        NPE: maps theta conditioned on z
        NLE: maps z conditioned on theta
        """
        # Dummy datasets for builder API.
        z_dataset = torch.randn(10, self.conditioning_dim)
        theta_dataset = torch.randn(10, self.inference_dim)

        hidden_features = self.flow_kwargs.pop("hidden_features", self.conditioning_dim)

        # NPE: x=theta, y=z
        if 'zuko' in str(self.build_flow).lower():
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

        # NLE: x=z, y=theta
        if 'zuko' in str(self.build_flow).lower():
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

        # Wrapper gives us a consistent encoder->flow interface for NPE.
        self.npe_model = _CondEmbeddingFlow(self.embedding_net, self.npe_flow)

    def compress(self, data_dict):
        """Return latent representation z from the shared encoder."""
        return self.embedding_net(data_dict)

    def _encode_to_z(self, data_dict):
        z = self.compress(data_dict)
        # If encoder returns (mu, logvar), take only mu for conditioning.
        if isinstance(z, tuple):
            z = z[0]
        return z

    def forward_npe(self, theta, data_dict):
        """log p(theta | z(data_dict))"""
        return self.npe_model.log_prob(theta, data_dict)

    def forward_nle(self, data_dict, theta):
        """log p(z(data_dict) | theta)"""
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
        # Log test losses from previous epoch end
        if len(self._test_npe_loss_values) > 0:
            self.log("test_npe_loss", self._test_npe_loss_values.pop(), sync_dist=self.is_distributed)
        if len(self._test_nle_loss_values) > 0:
            self.log("test_nle_loss", self._test_nle_loss_values.pop(), sync_dist=self.is_distributed)

    # --- Optional helpers mirroring NDE/NLE modules ---
    def build_posterior_object(self, prior=None):
        """Build an NPE-style posterior object for the NPE head."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval()
        if hasattr(self.embedding_net, "only_return_mu"):
            self.embedding_net.only_return_mu = True

        if prior is None:
            prior = utils.BoxUniform(
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
        """Build an NLE-style estimator object for the NLE head."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval()
        if hasattr(self.embedding_net, "only_return_mu"):
            self.embedding_net.only_return_mu = True

        if prior is None:
            prior = utils.BoxUniform(
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

    def _load_pretrained_npe_head(self, ckpt_path: str, freeze: bool = False, flow_prefix: str = "model.flow.") -> None:
        """Load weights for the NPE head flow from a checkpoint.

        By default accepts checkpoints saved from an NDELightningModule where the
        flow lived at state_dict key prefix ``model.flow.``.
        """
        print(f"[JointVMIMNLELightningModule] Loading pretrained NPE head from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        src_state = checkpoint.get("state_dict", checkpoint)
        load_partial_weights(
            target_module=self.npe_flow,
            source_state_dict=src_state,
            prefix=flow_prefix,
            freeze=freeze,
            verbose=True,
        )

    def _load_pretrained_nle_head(self, ckpt_path: str, freeze: bool = False, flow_prefix: str = "model.flow.") -> None:
        """Load weights for the NLE head flow from a checkpoint.

        By default accepts checkpoints saved from a LikelihoodNDELightningModule
        where the flow lived at state_dict key prefix ``model.flow.``.
        """
        print(f"[JointVMIMNLELightningModule] Loading pretrained NLE head from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        src_state = checkpoint.get("state_dict", checkpoint)
        load_partial_weights(
            target_module=self.nle_flow,
            source_state_dict=src_state,
            prefix=flow_prefix,
            freeze=freeze,
            verbose=True,
        )

