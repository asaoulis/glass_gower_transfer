from nflows import flows, transforms
from transfer_sbi.toy.custom_sbi import build_maf, build_nsf, build_maf_rqs
from typing import NamedTuple
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
import torch
from sbi import utils as utils
from torch.optim import Adam

from transfer_sbi.toy.custom_sbi import CustomSNPE_C

class BackboneConfig(NamedTuple):
    flow_type: str
    lr: float
    num_blocks: int
    weight_decay: float
    use_resblocks: bool = False
    use_batch_norm: bool = False
    hidden_features: int = 128


class HeadConfig(NamedTuple):
    lr: float
    num_blocks: int
    weight_decay: float
    flow_builder: callable = None  # Optional override


class TrainConfig(NamedTuple):
    backbone: BackboneConfig
    head: HeadConfig
    finetune_lr: float
    only_finetune_extra_blocks: bool = False
    size: int = 400
    pretrain_size: int = 10000
    conditioning_dim: int = 4

    def wandb_name(self):
        return (
            f"{self.backbone.flow_type}_{self.pretrain_size}pds_"
            f"{self.only_finetune_extra_blocks}only_{self.size}ds_"
            f"{self.backbone.num_blocks}ib_{self.head.num_blocks}eb_"
            f"{self.head.lr}lr_{self.backbone.use_resblocks}res_"
            f"{self.backbone.weight_decay}wd"
        )

class TransferIndentityMAF(flows.Flow):
    flow_type_map = {"nsf": build_nsf, "maf": build_maf, "rqs": build_maf_rqs}

    def __init__(self, config: TrainConfig, bounds=(0, 1.0), embedding_net=None, prior=None, device=None):
        super().__init__(None, None, None)
        self.config = config
        self.bounds = bounds
        self.device = device
        self.embedding_net = embedding_net or nn.Identity()
        self.conditioning_dim = config.conditioning_dim
        self.flow_kwargs = {"conditional_dim": self.conditioning_dim}

        self.build_flow = self.flow_type_map[config.model.backbone.flow_type]
        self.build_extra_blocks = config.model.head.flow_builder or self.flow_type_map[config.model.backbone.flow_type]

        self.cheap_x_dataset = None
        self.cheap_y_dataset = None

        self.prior = prior or utils.BoxUniform(
            low=self.bounds[0] * torch.ones(4),
            high=self.bounds[1] * torch.ones(4),
            device=self.device
        )

    def pretrain(self, cheap_x, cheap_y, test_dataloader=None, logger=None, lr=None, batch_size=128):
        self.cheap_x_dataset = cheap_x
        self.cheap_y_dataset = cheap_y

        bcfg = self.config.model.backbone
        inference_method = CustomSNPE_C(prior=self.prior, device=self.device)

        neural_net = self.build_flow(
            cheap_x, cheap_y,
            num_transforms=bcfg.num_blocks,
            embedding_net=self.embedding_net,
            hidden_features=bcfg.hidden_features,
            use_residual_blocks=bcfg.use_resblocks,
            use_batch_norm=bcfg.use_batch_norm,
            z_score_x=None,
            z_score_y=True,
            **self.flow_kwargs
        )

        inference_method.append_simulations(cheap_x, cheap_y)

        opt = AdamW(neural_net.parameters(), lr=lr or bcfg.lr, weight_decay=bcfg.weight_decay)

        self.maf_pretrained = inference_method.train(
            network=neural_net,
            optimizer=[opt],
            test_dataloader=test_dataloader,
            logger=logger,
            training_batch_size=batch_size
        )
        self.pretrain_opt = opt
        return self.maf_pretrained.state_dict()

    def build_finetune_flow(self, train_x, train_y):
        if self.maf_pretrained is None:
            raise ValueError("Model must be pretrained first.")

        embeddings = self.maf_pretrained._embedding_net(self.cheap_y_dataset.to(self.device)).detach().cpu()
        hcfg = self.config.model.head
        bcfg = self.config.model.backbone

        optimizer = []

        if not self.config.only_finetune_extra_blocks:
            main_opt = torch.optim.Adam(
                self.maf_pretrained.parameters(),
                lr=self.config.finetune_lr,
                weight_decay=bcfg.weight_decay
            )
            optimizer.append(main_opt)

        extra_blocks = self.build_extra_blocks(
            self.cheap_x_dataset,
            embeddings,
            num_transforms=hcfg.num_blocks,
            use_residual_blocks=True,
            use_identity=False,
            z_score_x=None,
            z_score_y=None,
            **self.flow_kwargs
        )

        if len(list(extra_blocks.parameters())) > 0:
            reinitialise_made_final_layers(extra_blocks, scale=1e-2)
            head_opt = torch.optim.Adam(
                extra_blocks.parameters(),
                lr=hcfg.lr,
                weight_decay=hcfg.weight_decay
            )
            optimizer.append(head_opt)

        composite_transform = transforms.CompositeTransform((
            self.maf_pretrained._transform,
            extra_blocks._transform
        ))

        final_flow = flows.Flow(
            transform=composite_transform,
            distribution=self.maf_pretrained._distribution,
            embedding_net=self.maf_pretrained._embedding_net
        )

        return final_flow, optimizer

    def finetune(self, train_x, train_y, test_dataloader, logger, state_dict=None):
        if state_dict is not None:
            self.maf_pretrained.load_state_dict(state_dict)

        inference_method = CustomSNPE_C(prior=self.prior, device=self.device)
        inference_method.append_simulations(train_x, train_y)

        final_flow, optimizer = self.build_finetune_flow(train_x, train_y)

        self.maf_finetuned = inference_method.train(
            network=final_flow,
            optimizer=optimizer,
            test_dataloader=test_dataloader,
            logger=logger,
            stop_after_epochs=60,
            training_batch_size=10
        )

        posterior = inference_method.build_posterior(self.maf_finetuned)
        return self.maf_finetuned, posterior

