
import torch
import torch.nn as nn
import pytorch_lightning as pl
from copy import deepcopy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR, LambdaLR, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm
from .custom_sbi import CustomSNPE_C, sample_batched
from types import MethodType
from functools import partial

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
            optimizer = Adam(self.model._transform.parameters(), lr=self.lr, **optimizer_kwargs)
        else:
            optimizer = AdamW(self.model.parameters(), lr=self.lr, **optimizer_kwargs)
        # optimizer = AdamW(self.model._transform.parameters(), lr=self.lr, **optimizer_kwargs)
        interval = "step"

        if self.scheduler_type == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, eta_min=1e-9)
        elif self.scheduler_type == 'cosine_2mult':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, eta_min=1e-9, T_mult=2)
        elif self.scheduler_type == 'cyclic':
            scheduler = CyclicLR(
                optimizer, base_lr=1e-9, max_lr=self.lr,
                step_size_up=1000, step_size_down=1000,
                cycle_momentum=False
            )
        elif self.scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',  # Adjust based on whether you're tracking loss ('min') or accuracy ('max')
                factor=0.95,  # Reduce LR by a factor of 0.9
                patience=10,  # Number of steps with no improvement before reducing LR
                threshold=1e-4,  # Minimum change to qualify as an improvement
                min_lr=1e-9
            )
            interval = "epoch"
        else:  # Default: Warm-up + Exponential Decay
            warmup_steps = self.scheduler_kwargs.get("warmup", 1000)
            gamma = self.scheduler_kwargs.get("gamma", 0.98)
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps  
                else:
                    return gamma ** (0.01 * (step - warmup_steps))

            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,
                "monitor": f"val_{self.loss_name}"
            }
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


from transfer_sbi.toy.custom_sbi import build_maf, build_maf_rqs, build_nsf
from torch.optim import Adam, AdamW
import torch
from sbi import utils as utils

import torch.nn as nn


class NDELightningModule(BaseLightningModule):
    flow_type_map = {"nsf": build_nsf, "maf": build_maf, "rqs": build_maf_rqs}

    def __init__(self, model, conditioning_dim, lr=0.0001, scheduler_type='cosine', test_dataloader=None, flow_type='nsf', num_extra_blocks=None, checkpoint_path=None,  **kwargs):
        super().__init__(model, loss_fn=None, lr=lr, scheduler_type=scheduler_type, **kwargs)
        embedding_net = model if model is not None else nn.Identity()
        self.conditioning_dim = conditioning_dim
        self.build_flow = self.flow_type_map[flow_type]  # Function to build the normalizing flow model
        self.flow_kwargs = {"conditional_dim": self.conditioning_dim}
        self.test_dataloader = test_dataloader
        self.loss_name = "log_prob"
        self.set_up_model(embedding_net)
        self.test_loss_values = []
        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

    def set_up_model(self, embedding_net):
        """Builds the flow model dynamically when training starts."""
        y_dataset, x_dataset = self.test_dataloader.dataset.tensors
        self.model = self.build_flow(x_dataset, y_dataset, num_transforms=4, z_score_x=None, z_score_y=None, 
                                     embedding_net=embedding_net, hidden_features=128, use_batch_norm=True, 
                                     **self.flow_kwargs)

    def load_from_checkpoint(self, checkpoint_path):
        """Loads model weights from a given checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))  # Adjust device as needed
        self.load_state_dict(checkpoint['state_dict'])  # Ensure the key matches the saved checkpoint format
    
    def build_posterior_object(self):
        y_dataset, x_dataset = self.test_dataloader.dataset.tensors
        dim = x_dataset.shape[1]
        prior = utils.BoxUniform(low=0 * torch.ones(dim), high=1. * torch.ones(dim), device="cuda")
        inference_method = CustomSNPE_C(prior=prior, device='cuda')
        inference_method.append_simulations(x_dataset[:10], y_dataset[:10])
        posterior_sbi = inference_method.build_posterior(self.model.to('cuda'))
        return posterior_sbi

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
        return self.model.log_prob(cond, x)

    # def predict_step(self, batch, batch_idx):
    #     batch = self.transfer_batch_to_device(batch, self.device, 0)
    #     x, cond = batch
    #     return self.forward(x, cond)

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
            predictions.append(self.forward(batch[0], batch[1]))
        all_log_probs = torch.cat(predictions, dim=0)  # Collect predictions
        avg_log_prob = all_log_probs.mean().item()
        return avg_log_prob

    def log_custom_evals(self, preds, y):
        if len(self.test_loss_values) > 0:
            self.log("test_log_prob", self.test_loss_values.pop())