ablation_experiments = {
    "ablation_glass_hybrid_OFF": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/glass_mocks_prior/output_*.h5",
        "model_type": "kids_o3_dual",
        "dataset_quantities": ["E_north", "E_south"],
        "model_kwargs": {
            "encoder_type": "unet_o3",
            "pool_types": ('avg', 'max', 'gem'),
            "patch_conditioning": ("side_info")
        },
        "latent_dim": 8 + 8,
        "epochs": 60,
        "batch_size": 100,
        "scheduler_type": "cyclic",
        "scheduler_kwargs": {'warmup': 2000, 'min_factor': 0.1, "cyclic_period_steps":6000},#, "gamma":0.98},
        "lr": 0.0004,
        "use_KL_loss": False,
        "flow_kwargs": {"hidden_features": 32},
        "project": "glass-pretraining",
        "cosmo_param_names": ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"],
        "repeat_indices": [1,2]
    },

    "finetune_ablation_glass_hybrid_OFF": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_o3_dual",
        "dataset_quantities": ["E_north", "E_south"],
        "model_kwargs": {
            "encoder_type": "unet_o3",
            "pool_types": ('avg', 'max', 'gem'),
            "patch_conditioning": ("side_info")
        },
        "latent_dim": 8 + 8,
        "scheduler_type": "exp",
        "scheduler_kwargs": {'warmup': 0,},
        "use_KL_loss": False,
        "flow_kwargs": {"hidden_features": 32},
        "project": "gower-finetuning",
        "cosmo_param_names": ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"],
        "match_num_cosmo": False,
        "repeat_indices": [0,1,2],
        "repeats": 3,

        "epochs": 25,
        "batch_size": 128,
        "lr": 1.e-5,
        "train_frac":0.65,
        "val_frac":0.25,
        "max_trainval_cosmos": [530,],
        "checkpoint_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/ablation_glass_hybrid_OFF/",
    },

    # NO SIDE INFO
    "ablation_glass_no_side": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/glass_mocks_prior/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south"],
        "model_kwargs": {
            "bandpower_type": "mlp",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 8,
            "map_kwargs":{
                "encoder_type": "unet_o3",
                "pool_types": ('avg', 'max', 'gem'),
                "patch_conditioning": None,
            },
            "bandpower_kwargs":{
                "hidden_multiple":32,
                "dropout": 0,
            }
        },
        "latent_dim": 8 + 8,
        "pretrained_band_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/glass_bandpower_mlp_9param/",
        "freeze_band": True,
        "epochs": 60,
        "batch_size": 100,
        "scheduler_type": "cyclic",
        "scheduler_kwargs": {'warmup': 2000, 'min_factor': 0.1, "cyclic_period_steps":6000},#, "gamma":0.98},
        "lr": 0.0004,
        "use_KL_loss": False,
        "flow_kwargs": {"hidden_features": 32},
        "project": "glass-pretraining",
        "cosmo_param_names": ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"],
        "repeat_indices": [1,2]

    },

    # Speed-optimized twin of ablation_glass_no_side (architectures/model-optimization task).
    # Identical architecture; adds ml_perf (encoder bf16 autocast + torch.compile the CNN
    # backbone) + a larger batch the freed memory allows. Measured A6000: ~2.4x training-loop
    # throughput vs the fp32 B=100 baseline (amp+compile 1.81x at B=100, 2.45x at B=200).
    "ablation_glass_no_side_fast": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/glass_mocks_prior/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south"],
        "model_kwargs": {
            "bandpower_type": "mlp",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 8,
            "map_kwargs":{
                "encoder_type": "unet_o3",
                "pool_types": ('avg', 'max', 'gem'),
                "patch_conditioning": None,
            },
            "bandpower_kwargs":{
                "hidden_multiple":32,
                "dropout": 0,
            }
        },
        "latent_dim": 8 + 8,
        "pretrained_band_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/glass_bandpower_mlp_9param/",
        "freeze_band": True,
        "epochs": 60,
        "batch_size": 224,  # sweet spot under the AMP+compile 48 GB budget (2.37x e2e; 256 regresses)
        "ml_perf": {"amp": True, "compile": "backbone", "tf32": False, "fused_adam": False},
        "persistent_workers": True,
        "prefetch_factor": 4,
        "pin_memory": True,
        "scheduler_type": "cyclic",
        "scheduler_kwargs": {'warmup': 2000, 'min_factor': 0.1, "cyclic_period_steps":6000},
        "lr": 0.0004,
        "use_KL_loss": False,
        "flow_kwargs": {"hidden_features": 32},
        "project": "glass-pretraining",
        "cosmo_param_names": ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"],
        "repeat_indices": [1,2]
    },

    "finetune_ablation_glass_no_side": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south"],
        "model_kwargs": {
            "bandpower_type": "mlp",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 8,
            "map_kwargs":{
                "encoder_type": "unet_o3",
                "pool_types": ('avg', 'max', 'gem'),
                "patch_conditioning": None,
            },
            "bandpower_kwargs":{
                "hidden_multiple":32,
                "dropout": 0,
            }
        },
        "latent_dim": 8 + 8,
        "pretrained_band_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/glass_bandpower_mlp_9param/",
        "scheduler_type": "exp",
        "scheduler_kwargs": {'warmup': 0,},
        "use_KL_loss": False,
        "flow_kwargs": {"hidden_features": 32},
        "project": "glass-pretraining",
        "cosmo_param_names": ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"],
        "match_num_cosmo": False,
        "repeat_indices": [0,1,2],
        "repeats": 3,
    
        "epochs": 15,
        "batch_size": 128,
        "lr": 1.e-5,
        "train_frac":0.65,
        "val_frac":0.25,
        "max_trainval_cosmos": [530,],
        "checkpoint_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/ablation_glass_no_side/",
    },

    # MAF NDE
    "ablation_glass_MAF": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/glass_mocks_prior/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "flow_type": "maf",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south"],
        "model_kwargs": {
            "bandpower_type": "mlp",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 8,
            "map_kwargs":{
                "encoder_type": "unet_o3",
                "pool_types": ('avg', 'max', 'gem'),
                "patch_conditioning": ("side_info")
            },
            "bandpower_kwargs":{
                "hidden_multiple":32,
                "dropout": 0,
            },
            "flow_type": "maf",
        },
        "latent_dim": 8 + 8,
        "pretrained_band_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/glass_bandpower_mlp_9param/",
        "freeze_band": True,
        "epochs": 60,
        "batch_size": 100,
        "scheduler_type": "cyclic",
        "scheduler_kwargs": {'warmup': 2000, 'min_factor': 0.1, "cyclic_period_steps":6000},#, "gamma":0.98},
        "lr": 0.0004,
        "use_KL_loss": False,
        "flow_kwargs": {"hidden_features": 32},
        "project": "glass-pretraining",
        "cosmo_param_names": ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"],
        "repeats": 3
    },
    "finetune_ablation_glass_MAF": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south"],
        "flow_type": "maf",
        "model_kwargs": {
            "bandpower_type": "mlp",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 8,
            "map_kwargs":{
                "encoder_type": "unet_o3",
                "pool_types": ('avg', 'max', 'gem'),
                "patch_conditioning": ("side_info")
            },
            "bandpower_kwargs":{
                "hidden_multiple":32,
                "dropout": 0,
            }
        },
        "latent_dim": 8 + 8,
        # "pretrained_band_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/glass_bandpower_mlp_9param/",
        "scheduler_type": "exp",
        "scheduler_kwargs": {'warmup': 0,},
        "use_KL_loss": False,
        "flow_kwargs": {"hidden_features": 32},
        "project": "glass-pretraining",
        "cosmo_param_names": ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"],
        "match_num_cosmo": False,
        "repeat_indices": [0,1,2],
        "repeats": 3,
    
        "epochs": 15,
        "batch_size": 128,
        "lr": 1.e-5,
        "train_frac":0.65,
        "val_frac":0.25,
        "max_trainval_cosmos": [530,],
        "checkpoint_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/ablation_glass_MAF/",
    },


    "ablation_glass_no_GEM": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/glass_mocks_prior/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south"],
        "model_kwargs": {
            "bandpower_type": "mlp",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 8,
            "map_kwargs":{
                "encoder_type": "unet_o3",
                "pool_types": ('avg',),
                "patch_conditioning": ("side_info")
            },
            "bandpower_kwargs":{
                "hidden_multiple":32,
                "dropout": 0,
            }
        },
        "latent_dim": 8 + 8,
        "pretrained_band_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/glass_bandpower_mlp_9param/",
        "freeze_band": True,
        "epochs": 60,
        "batch_size": 100,
        "scheduler_type": "cyclic",
        "scheduler_kwargs": {'warmup': 2000, 'min_factor': 0.1, "cyclic_period_steps":6000},#, "gamma":0.98},
        "lr": 0.0004,
        "use_KL_loss": False,
        "flow_kwargs": {"hidden_features": 32},
        "project": "glass-pretraining",
        "cosmo_param_names": ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"],
        "repeats": 3
    },

    "finetune_ablation_glass_no_GEM": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south"],
        "model_kwargs": {
            "bandpower_type": "mlp",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 8,
            "map_kwargs":{
                "encoder_type": "unet_o3",
                "pool_types": ('avg',),
                "patch_conditioning": ("side_info")
            },
            "bandpower_kwargs":{
                "hidden_multiple":32,
                "dropout": 0,
            }
        },
        "latent_dim": 8 + 8,
        "pretrained_band_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/glass_bandpower_mlp_9param/",
        "scheduler_type": "exp",
        "scheduler_kwargs": {'warmup': 0,},
        "use_KL_loss": False,
        "flow_kwargs": {"hidden_features": 32},
        "project": "glass-pretraining",
        "cosmo_param_names": ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"],
        "match_num_cosmo": False,
        "repeat_indices": [0,1,2],
        "repeats": 3,
    
        "epochs": 15,
        "batch_size": 128,
        "lr": 1.e-5,
        "train_frac":0.65,
        "val_frac":0.25,
        "max_trainval_cosmos": [530,],
        "checkpoint_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/ablation_glass_no_GEM/",
    },

    "ablation_glass_B_modes": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/glass_mocks_prior/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south", "B_north", "B_south"],
        "model_kwargs": {
            "bandpower_type": "mlp",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 8,
            "map_kwargs":{
                "encoder_type": "unet_o3",
                "pool_types": ('avg', 'max', 'gem'),
            "patch_conditioning": ("side_info")
            },
            "bandpower_kwargs":{
                "hidden_multiple":32,
                "dropout": 0,
            }
        },
        "latent_dim": 8 + 8,
        "pretrained_band_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/glass_bandpower_mlp_9param/",
        "freeze_band": True,
        "epochs": 60,
        "batch_size": 80,
        "scheduler_type": "cyclic",
        "scheduler_kwargs": {'warmup': 2000, 'min_factor': 0.1, "cyclic_period_steps":6000},#, "gamma":0.98},
        "lr": 0.0004,
        "use_KL_loss": False,
        "flow_kwargs": {"hidden_features": 32},
        "project": "glass-pretraining",
        "cosmo_param_names": ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"],
        "repeat_indices": [1,2]
    },

    "finetune_ablation_glass_B_modes": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south", "B_north", "B_south"],
        "model_kwargs": {
            "bandpower_type": "mlp",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 8,
            "map_kwargs":{
                "encoder_type": "unet_o3",
                "pool_types": ('avg', 'max', 'gem'),
            "patch_conditioning": ("side_info")
            },
            "bandpower_kwargs":{
                "hidden_multiple":32,
                "dropout": 0,
            }
        },
        "latent_dim": 8 + 8,
        "pretrained_band_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/glass_bandpower_mlp_9param/",
        "freeze_band": True,
        "scheduler_type": "exp",
        "scheduler_kwargs": {'warmup': 0,},
        "use_KL_loss": False,
        "flow_kwargs": {"hidden_features": 32},
        "project": "glass-pretraining",
        "cosmo_param_names": ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"],
        "match_num_cosmo": False,
        "repeat_indices": [0,1,2],
        "repeats": 3,
        "epochs": 15,
        "batch_size": 128,
        "lr": 1.e-5,
        "train_frac":0.65,
        "val_frac":0.25,
        "max_trainval_cosmos": [530,],
        "checkpoint_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/ablation_glass_B_modes/",
    },
    "ablation_glass_no_cyclic": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/glass_mocks_prior/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south"],
        "model_kwargs": {
            "bandpower_type": "mlp",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 8,
            "map_kwargs":{
                "encoder_type": "unet_o3",
                "pool_types": ('avg', 'max', 'gem'),
                "patch_conditioning": ("side_info")
            },
            "bandpower_kwargs":{
                "hidden_multiple":32,
                "dropout": 0,
            }
        },
        "latent_dim": 8 + 8,
        "pretrained_band_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/glass_bandpower_mlp_9param/",
        "freeze_band": True,
        "epochs": 60,
        "batch_size": 100,
        "scheduler_type": "exp",
        "scheduler_kwargs": {'warmup': 2000, "gamma": 0.98},#, "gamma":0.98},
        "lr": 0.0004,
        "use_KL_loss": False,
        "flow_kwargs": {"hidden_features": 32},
        "project": "glass-pretraining",
        "cosmo_param_names": ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"],
        "repeat_indices": [0,1,2]
    },
    "finetune_ablation_glass_no_cyclic": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south"],
        "model_kwargs": {
            "bandpower_type": "mlp",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 8,
            "map_kwargs":{
                "encoder_type": "unet_o3",
                "pool_types": ('avg', 'max', 'gem'),
                "patch_conditioning": ("side_info")
            },
            "bandpower_kwargs":{
                "hidden_multiple":32,
                "dropout": 0,
            }
        },
        "latent_dim": 8 + 8,
        "pretrained_band_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/glass_bandpower_mlp_9param/",
        "freeze_band": True,
        "scheduler_type": "exp",
        "scheduler_kwargs": {'warmup': 0,},
        "use_KL_loss": False,
        "flow_kwargs": {"hidden_features": 32},
        "project": "glass-pretraining",
        "cosmo_param_names": ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"],
        "match_num_cosmo": False,
        "repeat_indices": [0,1,2],
        "repeats": 3,
    
        "epochs": 15,
        "batch_size": 128,
        "lr": 1.e-5,
        "train_frac":0.65,
        "val_frac":0.25,
        "max_trainval_cosmos": [530,],
        "checkpoint_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/ablation_glass_no_cyclic/",
    },
    "ablation_glass_no_LINEARHEAD": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/glass_mocks_prior/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south"],
        "model_kwargs": {
            "bandpower_type": "mlp",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 8,
            "map_kwargs":{
                "encoder_type": "unet_o3",
                "pool_types": ('avg', 'max', 'gem'),
                "patch_conditioning": ("side_info")
            },
            "bandpower_kwargs":{
                "hidden_multiple":32,
                "dropout": 0,
            }
        },
        "latent_dim": 8 + 8,
        "pretrained_band_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/glass_bandpower_mlp_9param/",
        "freeze_band": True,
        "epochs": 60,
        "batch_size": 100,
        "scheduler_type": "cyclic",
        "scheduler_kwargs": {'warmup': 2000, 'min_factor': 0.1, "cyclic_period_steps":6000},#, "gamma":0.98},
        "lr": 0.0004,
        "use_KL_loss": False,
        "flow_kwargs": {"hidden_features": 32},
        "project": "glass-pretraining",
        "cosmo_param_names": ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"],
        "repeat_indices": [0,1,2]
    },
}
