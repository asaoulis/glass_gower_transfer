from nflows import flows, transforms
from transfer_sbi.toy.custom_sbi import build_maf, build_nsf, build_maf_rqs
from typing import NamedTuple
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
import torch
from sbi import utils as utils
from torch.optim import Adam, lr_scheduler

from transfer_sbi.toy.custom_sbi import CustomSNPE_C


class TrainConfig(NamedTuple):

    flow_type : str
    lr : float
    finetune_lr : float
    num_initial_blocks : int
    num_extra_blocks : int
    optimizer: torch.optim.Optimizer = None 
    resblocks: bool = False
    size: int = 400
    pretrain_size : int = 10000
    pretrain_wd : float = 0.01
    only_finetune_extra_blocks : bool = False
    conditioning_dim : int = 4
    # create a unique wandb name
    def wandb_name(self):
        return f"{self.flow_type}_{self.pretrain_size}pds_{self.only_finetune_extra_blocks}only_{self.size}ds_{self.num_initial_blocks}ib_{self.num_extra_blocks}eb_{self.lr}lr_{self.resblocks}res_{self.pretrain_wd}wd"

import torch.nn.init as init

def reinitialise_final_layer(layer, scale=1e-5):
    """Reinitialise the final layer of a MADE block."""
    if isinstance(layer, torch.nn.Linear) or 'MaskedLinear' in layer.__class__.__name__:
        init.uniform_(layer.weight, a=-scale, b=scale)  # Small random weights
        if layer.bias is not None:
            init.uniform_(layer.bias, a=-scale, b=scale)
def reinitialise_made_final_layers(model, scale=1e-5):
    """Reinitialise all 'final_layer' instances in a MADE block."""
    for name, module in model.named_modules():
        # if 'final_layer' in name:  # Check if the name matches 'final_layer'
            reinitialise_final_layer(module,scale)
     

import torch.nn as nn
from nflows.distributions.normal import StandardNormal
def prepare_identity_maf(test_maf, embeddings, bounds=(0., 1.), n_epochs=250, device='cpu'):
    distribution = utils.BoxUniform(low=bounds[0] * torch.ones(4), high=bounds[1] * torch.ones(4), device=device)
    distribution  = StandardNormal(shape=(4,))

    random_samples = distribution.sample(embeddings.shape[0]).to(device)
    embeddings = embeddings.to(device)
    neural_net = flows.Flow(test_maf._transform, distribution).to(device)
    opt =torch.optim.Adam(neural_net._transform.parameters(), lr=0.001, weight_decay=0.1)
    import torch.nn.functional as F

    def identity_loss_with_jacobian(model, x, y):
        z, _ = model._transform(x, y)
        loss_identity = F.mse_loss(z, x)  # Identity mapping loss
        # loss_jacobian = torch.mean((log_det) ** 2)  # Penalize deviations in log-det
        return loss_identity #+ 0.1 * loss_jacobian

    for epoch in range(n_epochs):
        opt.zero_grad()
        # Replace with the actual loss for fine-tuning (e.g., NLL)
        loss = identity_loss_with_jacobian(neural_net, random_samples, embeddings)
        loss.backward()
        opt.step()
        if epoch % 50 == 0:
            print(f"Fine-tune Epoch {epoch + 1}, Loss: {loss.item()}")
    return neural_net

class MAFPretrainFineTune(flows.Flow):
    flow_type_map = {"nsf": build_nsf, "maf": build_maf, "rqs": build_maf_rqs}
    def __init__(self, config : TrainConfig, bounds=(0, 1.0), embedding_net=None, device=None):
        super().__init__(None, None, None)
        self.config = config
        self.maf_pretrained = None
        self.maf_finetuned = None
        self.bounds = bounds

        self.cheap_x_dataset = None
        self.cheap_y_dataset = None
        self.device = device
        self.build_flow = self.flow_type_map[config.flow_type]

        self.embedding_net = embedding_net if embedding_net is not None else nn.Identity()
        self.conditioning_dim = config.conditioning_dim
        self.flow_kwargs = {"conditional_dim": self.conditioning_dim}

    def pretrain(self, cheap_x, cheap_y, test_dataloader=None, logger=None, lr=0.0001, scheduler = False, batch_size=128):
        self.cheap_x_dataset = cheap_x
        self.cheap_y_dataset = cheap_y
        prior = utils.BoxUniform(low=self.bounds[0] * torch.ones(4), high=self.bounds[1] * torch.ones(4), device=self.device)
        inference_method = CustomSNPE_C(prior=prior, device=self.device)
        neural_net = self.build_flow(cheap_x, cheap_y, num_transforms=self.config.num_initial_blocks, z_score_x=None, z_score_y=None, embedding_net=self.embedding_net, hidden_features=128, use_batch_norm=False, **self.flow_kwargs)
       # Implement training loop (omitted for brevity)
        inference_method.append_simulations(cheap_x, cheap_y)
        pretrain_opts = [AdamW(neural_net.parameters(), lr=lr, weight_decay=self.config.pretrain_wd)]
        if scheduler:
            # scheduler = lr_scheduler.StepLR(pretrain_opts[0], step_size=10, gamma=0.8) if scheduler else None
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(pretrain_opts[0], T_0=1000, eta_min=1e-9 )
            print('Using scheduler', scheduler, flush=True)
        self.maf_pretrained = inference_method.train(network=neural_net, optimizer=pretrain_opts, test_dataloader=test_dataloader, logger=logger, training_batch_size=batch_size, scheduler=scheduler)
        self.pretrain_opt = pretrain_opts[0]

        return self.maf_pretrained.state_dict()

    def finetune(self, train_x, train_y, test_dataloader, logger, state_dict = None):
        if self.maf_pretrained is None:
            raise ValueError("Model must be pretrained first.")
        if state_dict is not None:
            self.maf_pretrained.load_state_dict(state_dict)
        prior = utils.BoxUniform(low=self.bounds[0] * torch.ones(4), high=self.bounds[1] * torch.ones(4), device=self.device)
        inference_method = CustomSNPE_C(prior=prior, device=self.device)

        optimizer = []
        
        main_flow_opt = torch.optim.Adam(self.maf_pretrained.parameters(), lr=self.config.finetune_lr, weight_decay=self.config.pretrain_wd)
        
        if not self.config.only_finetune_extra_blocks:
            optimizer.append(main_flow_opt)
        embeddings = self.maf_pretrained._embedding_net(self.cheap_y_dataset.to(self.device)).detach().cpu()
        extra_blocks = build_maf(self.cheap_x_dataset, embeddings, num_transforms= self.config.num_extra_blocks, use_residual_blocks=True, z_score_x=None, z_score_y=None)

        if len(list(extra_blocks.parameters())) > 0:
            reinitialise_made_final_layers(extra_blocks, scale=1e-2) 

            extra_blocks = prepare_identity_maf(extra_blocks, embeddings, bounds=self.bounds, n_epochs=251, device=self.device)

            extra_opt = torch.optim.Adam(extra_blocks.parameters(), lr=self.config.lr)
            optimizer.append(extra_opt)

            
        new_flow = flows.Flow(transform = transforms.CompositeTransform((self.maf_pretrained._transform,extra_blocks._transform)), distribution = self.maf_pretrained._distribution, embedding_net=self.maf_pretrained._embedding_net)
        inference_method.append_simulations(train_x, train_y)
        scheduler = lr_scheduler.StepLR(main_flow_opt, step_size=10, gamma=0.8)
        self.maf_finetuned = inference_method.train(network=new_flow, optimizer=optimizer, test_dataloader=test_dataloader, logger=logger, stop_after_epochs=60, training_batch_size=10, scheduler=scheduler)
        posterior = inference_method.build_posterior(self.maf_finetuned)
        # Implement fine-tuning loop (omitted for brevity)
        return self.maf_finetuned, posterior
    
    def anneal(self, train_x, train_y, test_dataloader, logger, anneal_lr):
        prior = utils.BoxUniform(low=-1. * torch.ones(4), high=1. * torch.ones(4))
        inference_method = CustomSNPE_C(prior=prior)

        anneal_opt = torch.optim.Adam(self.maf_finetuned.parameters(), lr=anneal_lr)
        optimizer = [anneal_opt]
        inference_method.append_simulations(train_x, train_y)
        self.maf_finetuned = inference_method.train(network=self.maf_finetuned, optimizer=optimizer, test_dataloader=test_dataloader, logger=logger, stop_after_epochs=30)


def prep_test_dataloader(test_x_data, test_y_data, batch_size=32):
    test_x_data = torch.tensor(test_x_data).float()
    test_y_data = torch.tensor(test_y_data).float()
    dataset = torch.utils.data.TensorDataset(test_x_data, test_y_data)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return test_dataloader