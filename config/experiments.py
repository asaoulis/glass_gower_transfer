
experiments = {
    # baseline gower
    "default": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5"
    },
    "cnn_shared": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_o3_dual",
        "latent_dim": 256,
        "epochs": 200,
        "batch_size": 64,
    },
    "cnn_transform": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_combined_cnn_transformer",
        "latent_dim": 196,
        "epochs": 200,
        "batch_size": 64,
        # "model_kwargs": {
        #     "encoder_type": "unet_o3",
        # }
    },
    "fno_shared": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_fno_dual",
        "latent_dim": 256,
    },
    "cnn_transform_maf": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_combined_cnn_transformer",
        "flow_type": "maf",
    },
    "bandpower_mlp": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_bandpowers_mlp",
        "dataset_quantities": ["mixed_bandpowers"],
    },
    "bandpower_cnn": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_bandpowers_cnn1d",
        "dataset_quantities": ["mixed_bandpowers"],
    },
    "bandpower_cnn_64": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_bandpowers_cnn1d",
        "dataset_quantities": ["mixed_bandpowers"],
        "latent_dim": 64,
        "batch_size": 64,
    },
    "bandpower_cnn_64_KL": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_bandpowers_cnn1d",
        "dataset_quantities": ["mixed_bandpowers"],
        "latent_dim": 64,
        "batch_size": 64,
        "use_KL_loss": True,
    },
    "bandpower_cnn_8_KLDIV": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_bandpowers_cnn1d",
        "dataset_quantities": ["mixed_bandpowers"],
        "latent_dim": 8,
        "batch_size": 64,
        "use_KL_loss": True,
    },
    "bandpower_cnn_KL_redundancy": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_bandpowers_cnn1d",
        "dataset_quantities": ["mixed_bandpowers"],
        "latent_dim": 64,
        "batch_size": 64,
        "use_KL_loss": True,
        "redundancy_dim": 128,
    },
    "hybrid_bandpower_maps_o3":{
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south", "B_north", "B_south"],
        "model_kwargs": {
            "bandpower_type": "cnn",
            "map_encoder_type": "o3_dual",
        #     "transformer_kwargs": 
        #         { "hidden":12, "channels_per_map":6, "d_model":256, "n_heads":4, "n_layers":4, "n_queries":8, "dropout":0.1}
        },
        "latent_dim": 256,
    },
    "hybrid_bandpower_maps":{
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south", "B_north", "B_south"],
        "model_kwargs": {
            "bandpower_type": "cnn",
            "transformer_kwargs": 
                { "hidden":12, "channels_per_map":6, "d_model":256, "n_heads":4, "n_layers":4, "n_queries":8, "dropout":0.1}
        },
        "latent_dim": 256,
    },
    "hybrid_frozen_bandpower_maps":{
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south", "B_north", "B_south"],
        "model_kwargs": {
            "bandpower_type": "cnn",
            "transformer_kwargs": 
                { "hidden":12, "channels_per_map":6, "d_model":256, "n_heads":4, "n_layers":4, "n_queries":8, "dropout":0.1},
            "bandpower_latent_dim": 256,
        },
        "latent_dim": 512,
        "pretrained_band_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/bandpower_cnn/run_pretrain_kids_bandpowers_cnn1d_exp_0.0004__dsNone_0/checkpoint-epoch=31-val_log_prob=-2.8146.ckpt",
        "freeze_band": True,
    },
    "hybrid_frozen_bandpower_maps_small":{
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south", "B_north", "B_south"],
        "model_kwargs": {
            "bandpower_type": "cnn",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 64,
        },
        "latent_dim": 128 + 64,
        "pretrained_band_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/bandpower_cnn_64_KL/run_pretrain_kids_bandpowers_cnn1d_exp_0.0004__dsNone_0/checkpoint-epoch=71-val_log_prob=-2.9770.ckpt",
        "freeze_band": True,
        "epochs": 200,
        "batch_size": 64,
    },
    "hybrid_frozen_multpools_o3":{
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south", "B_north", "B_south"],
        "model_kwargs": {
            "bandpower_type": "cnn",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 64,
            "map_kwargs":{
                "encoder_type": "flex_o3",
                "pool_types": ('avg', 'max', 'gem')
            }
        },
        "latent_dim": 128 + 64,
        "pretrained_band_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/bandpower_cnn_64_KL/run_pretrain_kids_bandpowers_cnn1d_exp_0.0004__dsNone_0/checkpoint-epoch=71-val_log_prob=-2.9770.ckpt",
        "freeze_band": True,
        "epochs": 200,
        "batch_size": 64,
        "scheduler_kwargs": {'warmup': 5000, 'gamma': 0.998},
    },
    "hybrid_frozen_multpools_unet":{
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south", "B_north", "B_south"],
        "model_kwargs": {
            "bandpower_type": "cnn",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 64,
            "map_kwargs":{
                "encoder_type": "unet_o3",
                "pool_types": ('avg', 'max', 'gem')
            }
        }
    },
    "hybrid_frozen_multpools_unet_tinylatent":{
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south", "B_north", "B_south"],
        "model_kwargs": {
            "bandpower_type": "cnn",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 8,
            "map_kwargs":{
                "encoder_type": "unet_o3",
                "pool_types": ('avg', 'max', 'gem')
            }
        },
        "latent_dim": 32 + 8,
        "pretrained_band_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/bandpower_cnn_8_KL/run_bandpower_cnn_8_KL/pretrain_kids_bandpowers_cnn1d_exp_0.0004__dsNone_0/checkpoint-epoch=56-val_log_prob=-2.9690.ckpt",
        "freeze_band": True,
        "epochs": 200,
        "batch_size": 64,
        "scheduler_kwargs": {'warmup': 4000, 'gamma': 0.998},
        "lr": 0.0005,
        "use_KL_loss": True,
        "flow_kwargs": {"hidden_features": 128}
    },
    "hybrid_frozen_multpools_unet_large":{
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south", "B_north", "B_south"],
        "model_kwargs": {
            "bandpower_type": "cnn",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 64,
            "map_kwargs":{
                "encoder_type": "unet_o3",
                "pool_types": ('avg',)
            }
        },
        "latent_dim": 256 + 64,
        "pretrained_band_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/bandpower_cnn_64_KL/run_pretrain_kids_bandpowers_cnn1d_exp_0.0004__dsNone_0/checkpoint-epoch=71-val_log_prob=-2.9770.ckpt",
        "freeze_band": True,
        "epochs": 200,
        "batch_size": 64,
        "scheduler_kwargs": {'warmup': 1000, 'gamma': 0.999},
        "lr": 4.e-4,
    },
    "hybrid_frozen_multpools_unet_multihead":{
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south", "B_north", "B_south"],
        "model_kwargs": {
            "bandpower_type": "cnn",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 64,
            "map_kwargs":{
                "encoder_type": "unet_o3",
                "pool_types": ('avg', 'max', 'gem')
            }
        },
        "latent_dim": 128 + 64,
        "pretrained_band_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/bandpower_cnn_64_KL/run_pretrain_kids_bandpowers_cnn1d_exp_0.0004__dsNone_0/checkpoint-epoch=71-val_log_prob=-2.9770.ckpt",
        "freeze_band": True,
        "epochs": 200,
        "batch_size": 32,
        "scheduler_kwargs": {'warmup': 5000, 'gamma': 0.998},
        "lr": 4.e-4,
        "num_flow_heads": 4,
    },
    "hybrid_frozen_unet_cnntransformer":{
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south", "B_north", "B_south"],
        "model_kwargs": {
            "bandpower_type": "cnn",
            "map_encoder_type": "transformer",
            "bandpower_latent_dim": 64,
            "map_kwargs":{
                "encoder_type": "unet_o3",
            }
        },
        "latent_dim": 128 + 64,
        "pretrained_band_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/bandpower_cnn_64_KL/run_pretrain_kids_bandpowers_cnn1d_exp_0.0004__dsNone_0/checkpoint-epoch=71-val_log_prob=-2.9770.ckpt",
        "pretrained_backbone_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/hybrid_frozen_multpools_unet/run_hybrid_frozen_multpools_unet/pretrain_kids_hybrid_bandpowers_maps_exp_0.0004__dsNone_0/checkpoint-epoch=104-val_log_prob=-3.4624.ckpt",
        "freeze_band": True,
        "epochs": 200,
        "batch_size": 64,
        "scheduler_kwargs": {'warmup': 5000, 'gamma': 0.999},
        "lr": 4.e-5
    },
    "hybrid_frozen_unet_cnntransformer_pretrainedcnn":{
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south", "B_north", "B_south"],
        "model_kwargs": {
            "bandpower_type": "cnn",
            "map_encoder_type": "transformer",
            "bandpower_latent_dim": 64,
            "map_kwargs":{
                "encoder_type": "unet_o3",
            }
        },
        "latent_dim": 128 + 64,
        "pretrained_band_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/bandpower_cnn_64_KL/run_pretrain_kids_bandpowers_cnn1d_exp_0.0004__dsNone_0/checkpoint-epoch=71-val_log_prob=-2.9770.ckpt",
        "pretrained_backbone_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/hybrid_frozen_multpools_unet/run_hybrid_frozen_multpools_unet/pretrain_kids_hybrid_bandpowers_maps_exp_0.0004__dsNone_0/checkpoint-epoch=104-val_log_prob=-3.4624.ckpt",
        "freeze_band": True,
        "epochs": 200,
        "batch_size": 64,
        "scheduler_kwargs": {'warmup': 5000, 'gamma': 0.999},
        "lr": 4.e-5
    },
    "hybrid_frozen_bandpower_maps_small_lr":{
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south", "B_north", "B_south"],
        "model_kwargs": {
            "bandpower_type": "cnn",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 64,
        },
        "latent_dim": 128 + 64,
        "pretrained_band_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/bandpower_cnn_64_KL/run_pretrain_kids_bandpowers_cnn1d_exp_0.0004__dsNone_0/checkpoint-epoch=71-val_log_prob=-2.9770.ckpt",
        "freeze_band": True,
        "epochs": 200,
        "batch_size": 64,
        "lr": 1.e-4,
        "scheduler_kwargs": {'warmup': 5000, 'gamma': 0.998},
    },
    "hybrid_pre_bandpower_maps_small_lr":{
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south", "B_north", "B_south"],
        "model_kwargs": {
            "bandpower_type": "cnn",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 64,
        },
        "latent_dim": 128 + 64,
        "pretrained_band_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/bandpower_cnn_KL_redundancy/run_pretrain_kids_bandpowers_cnn1d_exp_0.0004__dsNone_0/checkpoint-epoch=48-val_log_prob=-2.9504.ckpt",
        "freeze_band": True,
        "epochs": 200,
        "batch_size": 64,
        "lr": 4.e-4,
        "scheduler_kwargs": {'warmup': 5000, 'gamma': 0.995},
        "load_pretrained_flow": True,
    },
    # pretrain gower
    "glass_bandpower_cnn": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/glass_mocks/output_*.h5",
        "model_type": "kids_bandpowers_cnn1d",
        "dataset_quantities": ["mixed_bandpowers"],
        "latent_dim": 8,
        "batch_size": 128,
        "use_KL_loss": True,
        "flow_kwargs": {"hidden_features": 256},
        "project": "glass-pretraining",
        "model_kwargs": {
            "channels":(64, 128, 128, 256)
        }
    },
    "glass_bandpower_mlp": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/glass_mocks/output_*.h5",
        "model_type": "kids_bandpowers_mlp",
        "dataset_quantities": ["mixed_bandpowers"],
        "project": "glass-pretraining",
        "batch_size": 128,
        "latent_dim": 8,
        "use_KL_loss": True,
        "flow_kwargs": {"hidden_features": 128},
        "model_kwargs": {
            "hidden_multiple":16,
        },
        "epochs": 120,
    },
    "glass_hybrid_bandpower_maps":{
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/glass_gower_prior/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south", "B_north", "B_south"],
        "model_kwargs": {
            "bandpower_type": "cnn",
            "map_encoder_type": "o3_dual",
        #     "transformer_kwargs": 
        #         { "hidden":12, "channels_per_map":6, "d_model":256, "n_heads":4, "n_layers":4, "n_queries":8, "dropout":0.1}
        },
        "latent_dim": 512,
        "project": "glass-pretraining",
    },
    "glass_cnn_shared": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/glass_gower_prior/output_*.h5",
        "model_type": "kids_o3_dual",
        "latent_dim": 512,
        "project": "glass-pretraining",

    },
    "glass_kids_combined_cnn_transformer": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/glass_gower_prior/output_*.h5",
        "model_type": "kids_combined_cnn_transformer",
        "latent_dim": 256,
        "project": "glass-pretraining"
    },
# finetune gower
    "finetune_cnn_shared": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_o3_dual",
        "latent_dim": 512,
        "checkpoint_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/glass_cnn_shared/",
        "lr": 0.00005
    },
}
