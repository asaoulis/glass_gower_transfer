from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR, ExponentialLR, LambdaLR, SequentialLR

from .utils import load_partial_weights


class BaseLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        loss_fn,
        lr=0.0001,
        scheduler_type="cosine",
        element_names=None,
        optimizer_kwargs=None,
        scheduler_kwargs=None,
        freeze_CNN=False,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.scheduler_type = scheduler_type
        self.best_checkpoints = []
        self.element_names = element_names if element_names is not None else []
        self.loss_name = ""
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.freeze_CNN = freeze_CNN

        # Optional per-module learning rates for pretrained components.
        self.pretrained_band_lrs = getattr(self, "pretrained_band_lrs", None)
        self.pretrained_backbone_lrs = getattr(self, "pretrained_backbone_lrs", None)

        self._optimizer_group_base_lrs = None

        # Used to control sync_dist in self.log calls.
        self.is_distributed = getattr(self.hparams, "is_distributed", False)

    def forward(self, x, cond):
        return self.model(x)

    def compute_loss(self, preds, y):
        return self.loss_fn(preds, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x, cond=y)
        loss = self.compute_loss(preds, y)
        self.log(
            f"train_{self.loss_name}",
            loss,
            prog_bar=True,
            sync_dist=self.is_distributed,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x, cond=y)
        loss = self.compute_loss(preds, y)
        self.log(
            f"val_{self.loss_name}",
            loss,
            prog_bar=True,
            sync_dist=self.is_distributed,
        )
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
                if not hasattr(self.model, mod_name):
                    print(f"Warning: Module {mod_name} not found.")
                    return
                module = getattr(self.model, mod_name)

            group_params = [p for p in module.parameters() if p.requires_grad]
            if not group_params:
                return

            param_groups.append({"params": group_params, "lr": lr_value})
            for p in group_params:
                assigned_params.add(p)

        if isinstance(self.pretrained_band_lrs, dict):
            for name, lr_val in self.pretrained_band_lrs.items():
                add_module_group(name, lr_val)

        if isinstance(self.pretrained_backbone_lrs, dict):
            for name, lr_val in self.pretrained_backbone_lrs.items():
                add_module_group(name, lr_val)

        base_params = [
            p
            for p in self.model.parameters()
            if p.requires_grad and p not in assigned_params
        ]

        if base_params:
            param_groups.append({"params": base_params, "lr": self.lr})

        optimizer = base_optimizer_cls(param_groups, lr=self.lr, **self.optimizer_kwargs)
        self._optimizer_group_base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        return optimizer

    def configure_optimizers(self):
        default_optimizer_kwargs = dict(weight_decay=3.53e-7, betas=(0.5, 0.999))
        optimizer_kwargs = {**default_optimizer_kwargs, **self.optimizer_kwargs}

        if (
            isinstance(self.pretrained_band_lrs, dict)
            and self.pretrained_band_lrs
            or isinstance(self.pretrained_backbone_lrs, dict)
            and self.pretrained_backbone_lrs
        ):
            self.optimizer_kwargs = optimizer_kwargs
            optimizer = self._build_param_groups(AdamW)
        else:
            optimizer = AdamW(self.model.parameters(), lr=self.lr, **optimizer_kwargs)

        base_lrs = [pg["lr"] for pg in optimizer.param_groups]

        max_epochs = getattr(self.trainer, "max_epochs", 200)
        total_steps = self.trainer.estimated_stepping_batches
        if total_steps is None or total_steps <= 0:
            steps_per_epoch = 1000
            total_steps = steps_per_epoch * max_epochs
        else:
            steps_per_epoch = total_steps // max_epochs

        warmup_frac = float(self.scheduler_kwargs.get("warmup_frac", 0.05))
        warmup_steps_override = self.scheduler_kwargs.get("warmup_steps", None)
        if warmup_steps_override is not None:
            warmup_steps = max(0, int(warmup_steps_override))
        else:
            warmup_steps = max(0, int(total_steps * warmup_frac))

        warmup_start_factor = float(self.scheduler_kwargs.get("warmup_start_factor", 0.1))

        def warmup_lambda(step):
            if warmup_steps <= 0:
                return 1.0
            return warmup_start_factor + (1.0 - warmup_start_factor) * min(
                1.0, step / warmup_steps
            )

        warmup_sched = LambdaLR(optimizer, lr_lambda=warmup_lambda)

        sched_type = str(self.scheduler_type).lower()

        if sched_type == "cyclic":
            cyclic_period_steps = int(self.scheduler_kwargs.get("cyclic_period_steps", 2000))
            min_factor = float(self.scheduler_kwargs.get("cyclic_min_factor", 0.05))

            main_sched = CyclicLR(
                optimizer,
                base_lr=[lr * min_factor for lr in base_lrs],
                max_lr=base_lrs,
                step_size_up=cyclic_period_steps // 2,
                step_size_down=cyclic_period_steps // 2,
                mode="triangular",
                cycle_momentum=False,
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
            cyclic_period_steps = int(self.scheduler_kwargs.get("cyclic_period_steps", 2000))
            min_factor = float(self.scheduler_kwargs.get("cyclic_min_factor", 0.05))
            gamma = float(self.scheduler_kwargs.get("gamma", 0.98))
            step_gamma = gamma ** (1 / steps_per_epoch)

            def combined_lambda(global_step: int):
                exp_factor = step_gamma**global_step

                if cyclic_period_steps <= 0:
                    cyc = 1.0
                else:
                    phase = (global_step % cyclic_period_steps) / cyclic_period_steps
                    if phase < 0.5:
                        cyc01 = phase / 0.5
                    else:
                        cyc01 = (1.0 - phase) / 0.5
                    cyc = min_factor + (1.0 - min_factor) * cyc01

                return exp_factor * cyc

            main_sched = LambdaLR(optimizer, lr_lambda=combined_lambda)
            interval = "step"
            milestone = warmup_steps

        else:
            main_sched = LambdaLR(optimizer, lr_lambda=lambda x: 1.0)
            interval = "step"
            milestone = warmup_steps

        if warmup_steps > 0:
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_sched, main_sched],
                milestones=[milestone],
            )
        else:
            scheduler = main_sched

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,
            },
        }

    def _load_pretrained_band_encoder(
        self,
        ckpt_path: str,
        freeze: bool,
        band_prefix: str = "model.embedding_net.",
    ) -> str | None:
        print(f"[NDELightningModule] Loading pretrained band encoder from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        src_state = checkpoint.get("state_dict", checkpoint)

        band_module = None
        band_module_name = None
        for attr in ("band_encoder", "band_model"):
            if hasattr(self.embedding_net, attr):
                band_module = getattr(self.embedding_net, attr)
                band_module_name = f"embedding_net.{attr}"
                print(f"[NDELightningModule] Using {band_module_name} as band encoder target.")
                break

        if band_module is None:
            print(
                "[NDELightningModule] Warning: embedding_net has no band encoder submodule; skipping band loading."
            )
            return None

        load_partial_weights(
            target_module=band_module,
            source_state_dict=src_state,
            prefix=band_prefix,
            freeze=freeze,
            verbose=True,
        )
        if freeze:
            self.embedding_net.freeze_band = True

        return band_module_name

    def _load_pretrained_flow(
        self,
        ckpt_path: str,
        freeze: bool,
        flow_prefix: str = "model.flow.",
        error_on_mismatch: bool = False,
    ) -> None:
        print(f"[NDELightningModule] Loading pretrained flow from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        src_state = checkpoint.get("state_dict", checkpoint)

        diag = load_partial_weights(
            target_module=self.model.flow,
            source_state_dict=src_state,
            prefix=flow_prefix,
            freeze=freeze,
            verbose=True,
        )
        # Guard (a): a warm start whose flow shapes don't match the checkpoint would otherwise be
        # silently skipped (load_partial_weights defaults to lenient), leaving flow layers random.
        # When the caller demands correctness (whitened embeddings), turn that into a hard failure.
        if error_on_mismatch and diag is not None:
            n_shape = len(diag.get("skipped_shape") or [])
            n_missing = len(diag.get("missing") or [])
            if n_shape > 0 or n_missing > 0:
                raise RuntimeError(
                    "Pretrained-flow warm start is incomplete: "
                    f"{n_shape} shape-mismatched key(s), {n_missing} flow key(s) left uninitialised "
                    f"(loaded {diag.get('loaded')}). This almost certainly means the finetune "
                    "embedding dimension (whitener k) disagrees with the pretrain flow the checkpoint "
                    f"was trained on. Checkpoint: {ckpt_path}. First mismatches: "
                    f"{(diag.get('skipped_shape') or [])[:5]}"
                )

    def _load_pretrained_cnn_backbone(
        self,
        ckpt_path: str,
        freeze: bool,
        backbone_prefix: str = "shared_cnn.backbone.",
        target_prefix: str = "",
    ) -> str | None:
        print(f"[NDELightningModule] Loading pretrained CNN backbone from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        src_state = checkpoint.get("state_dict", checkpoint)

        cnn_module = None
        cnn_module_name = None
        if hasattr(self.embedding_net, "patch_encoder"):
            shared = getattr(self.embedding_net, "patch_encoder")
            if hasattr(shared, "shared_cnn"):
                cnn_module = shared.shared_cnn
                cnn_module_name = "embedding_net.patch_encoder.shared_cnn"

        if cnn_module is None:
            print(
                "[NDELightningModule] Warning: could not find a CNN backbone on embedding_net; skipping backbone loading."
            )
            return None

        if target_prefix:
            remapped = {}
            for k, v in src_state.items():
                if backbone_prefix and k.startswith(backbone_prefix):
                    inner_key = k[len(backbone_prefix) :]
                    remapped[target_prefix + inner_key] = v
            src_state = remapped
            backbone_prefix = ""

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
        patch_prefix: str = "model.embedding_net.",
    ) -> str | None:
        print(f"[NDELightningModule] Loading pretrained patch encoder from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        src_state = checkpoint.get("state_dict", checkpoint)

        patch_module = self.embedding_net
        patch_module_name = "embedding_net"

        load_partial_weights(
            target_module=patch_module,
            source_state_dict=src_state,
            prefix=patch_prefix,
            freeze=freeze,
            verbose=True,
        )

        return patch_module_name


class RegressionLightningModule(BaseLightningModule):
    """Simple MSE regression on cosmological parameters."""

    def __init__(
        self,
        model,
        conditioning_dim,
        inference_dim,
        lr=0.0001,
        scheduler_type="exp",
        test_dataloader=None,
        **kwargs,
    ):
        kwargs.pop("flow_type", None)
        kwargs.pop("flow_kwargs", None)
        kwargs.pop("num_extra_blocks", None)
        kwargs.pop("redundancy_dim", None)
        kwargs.pop("num_flows", None)

        super().__init__(
            model,
            loss_fn=torch.nn.MSELoss(),
            lr=lr,
            scheduler_type=scheduler_type,
            **kwargs,
        )
        self.loss_name = "loss"
        self.embedding_net = model if model is not None else nn.Identity()
        self.conditioning_dim = conditioning_dim
        self.inference_dim = inference_dim
        self.test_dataloader = test_dataloader
        self.test_loss_values = []

        self.regression_head = nn.Linear(conditioning_dim, inference_dim)

    def forward(self, data_dict, cond=None):
        z = self.embedding_net(data_dict)
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
        if len(self.test_loss_values) > 0:
            self.log("test_loss", self.test_loss_values.pop(), sync_dist=self.is_distributed)

        if self.element_names:
            preds = preds.float()
            preds_np = preds.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            for i, element in enumerate(self.element_names):
                if i < preds_np.shape[-1] and i < y_np.shape[-1]:
                    r2 = r2_score(y_np[:, i], preds_np[:, i])
                    self.log(f"R2_{element}", r2, prog_bar=False, sync_dist=self.is_distributed)
                    self.log(
                        f"MSE_{element}",
                        np.mean((y_np[:, i] - preds_np[:, i]) ** 2),
                        prog_bar=False,
                        sync_dist=self.is_distributed,
                    )

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        print("Overwriting model weights from checkpoint:", checkpoint_path)
        self.load_state_dict(checkpoint["state_dict"])


class GaussianLightningModule(BaseLightningModule):
    def __init__(
        self,
        model,
        lr=0.0001,
        scheduler_type="cosine",
        batch_size=32,
        element_names=None,
        num_outputs=2,
        **kwargs,
    ):
        super().__init__(
            model,
            loss_fn=torch.nn.MSELoss(),
            lr=lr,
            scheduler_type=scheduler_type,
            batch_size=batch_size,
            element_names=element_names,
        )
        self.loss_name = "loss"
        self.num_outputs = num_outputs

    def log_r2_eval(self, preds, y):
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
        y_NN = preds[:, : self.num_outputs]
        e_NN = preds[:, self.num_outputs :]
        loss1 = torch.mean((y_NN - y) ** 2, axis=0)
        loss2 = torch.mean(((y_NN - y) ** 2 - e_NN**2) ** 2, axis=0)
        loss = torch.mean(torch.log(loss1) + torch.log(loss2))
        return loss
