import sys
import torch

from config.default import get_default_config
from config.experiments import experiments
from src.ml.embeddings.embeddings_utils import load_pretrained_models, build_embedding_dataloaders, IdentityEmbedding, fit_nde_on_embeddings
from src.ml.utils import prepare_data_parameters
from src.ml.eval.utils import evaluate_best_checkpoint
from src.ml.models.lightning_modules import NDELightningModule, LikelihoodNDELightningModule
from src.ml.models.utils import create_run_name
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train_embeddings.py <target_experiment_name> <source_exp1> [<source_exp2> ...]")
        sys.exit(1)

    target_experiment = sys.argv[1]
    source_experiments = sys.argv[2:]

    # Load target experiment config and apply the same max_trainval_cosmos logic as in train.py
    if target_experiment not in experiments:
        raise ValueError(f"Experiment '{target_experiment}' not found in config.experiments.experiments.")

    target_exp_cfg = experiments[target_experiment]

    # Helper to build a cfg with all non-list values applied and a given max_trainval_cosmos
    def _build_target_cfg(n_cosmo=None):
        cfg = get_default_config()
        cfg.experiment_name = target_experiment
        for k, v in target_exp_cfg.items():
            if k == "max_trainval_cosmos":
                continue
            setattr(cfg, k, v)
        if n_cosmo is not None:
            cfg.max_trainval_cosmos = int(n_cosmo)
            cfg.match_string = f"ncosmo{n_cosmo}"
        # Default inference mode for embeddings is 'npe' unless experiment overrides it.
        if not hasattr(cfg, "inference_mode"):
            cfg.inference_mode = target_exp_cfg.get("inference_mode", "npe")
        return cfg

    max_tv = target_exp_cfg.get("max_trainval_cosmos", None)

    def _run_single(target_cfg_local):
        # Load source models (pretrained representation providers)
        models, dataset_quantities = load_pretrained_models(source_experiments)
        # loop over cfgs and sum data patterns
        target_cfg_local.dataset_quantities = dataset_quantities
        target_cfg_local.test_shape_noise_idx = [0]
        # Use *target* cfg to build dataloaders with correct max_trainval_cosmos and scalers
        scalers, train_loader, val_loader, test_loader = prepare_data_parameters(target_cfg_local)
        run_name = create_run_name(target_cfg_local, target_cfg_local.match_string)
        # Build embedding dataloaders on top of those loaders
        train_emb_loader, val_emb_loader, test_emb_loader = build_embedding_dataloaders(
            train_loader, val_loader, test_loader, models, base_cfg=target_cfg_local, wandb_run_name=run_name
        )

        # Embedding dimension is last dimension of one batch from train_emb_loader
        sample_batch = next(iter(train_emb_loader))[0]
        emb_dim = sample_batch.shape[-1]

        fit_nde_on_embeddings(emb_dim, train_emb_loader, val_emb_loader, test_emb_loader, target_cfg_local)

        # Custom model builder for evaluation: build embedding-based NDE from checkpoint
        def emb_model_builder(cfg, loader):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Identity embedding with the same embedding dimension
            embedding_net = IdentityEmbedding(emb_dim).to(device)

            conditioning_dim = emb_dim
            inference_dim = len(cfg.cosmo_param_names)

            inference_mode = getattr(cfg, "inference_mode", "npe").lower()
            LightningCls = NDELightningModule if inference_mode == "npe" else LikelihoodNDELightningModule
            if inference_mode == "nle":
                conditioning_dim, inference_dim = inference_dim, conditioning_dim
            model = LightningCls(
                embedding_net,
                conditioning_dim=conditioning_dim,
                inference_dim=inference_dim,
                lr=cfg.lr,
                flow_type=cfg.flow_type,
                scheduler_type=cfg.scheduler_type,
                element_names=["Omega", "sigma8"],
                test_dataloader=loader,
                optimizer_kwargs=cfg.optimizer_kwargs,
                num_extra_blocks=cfg.extra_blocks,
                freeze_CNN=False,
                scheduler_kwargs=cfg.scheduler_kwargs,
                flow_kwargs=cfg.flow_kwargs,
            )

            # If a checkpoint_path is set on cfg, load the weights
            checkpoint_path = getattr(cfg, "checkpoint_path", None)
            if checkpoint_path:
                ckpt = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(ckpt["state_dict"], strict=False)

            model.to(device)
            model.eval()
            return model

        # evaluate_best_checkpoint(
        #     target_cfg_local,
        #     test_emb_loader,
        #     scalers["cosmo"],
        #     reference_samples=None,
        #     model_builder=emb_model_builder,
        # )

    if isinstance(max_tv, (list, tuple)):
        for n_cosmo in max_tv:
            print(f"Running embedding experiment '{target_experiment}' with max_trainval_cosmos={n_cosmo}")
            cfg_copy = _build_target_cfg(n_cosmo)
            _run_single(cfg_copy)
    else:
        cfg_single = _build_target_cfg(max_tv)
        print(f"Running embedding experiment '{target_experiment}'")
        _run_single(cfg_single)
