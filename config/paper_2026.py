"""Experiment rows behind the 2026 KiDS-Legacy-like mock paper posteriors.

hybrid_patches_16_9param is the high-fidelity-only hybrid trained on Gower Street mocks.
finetune_hybrid_16_9param_ensemble is the 9-member ensemble that finetunes the GLASS-pretrained
hybrid on Gower Street mocks.
"""

paper_2026_experiments = {
    "hybrid_patches_16_9param": {
        "data_patterns": "/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south"],
        "model_kwargs": {
            "bandpower_type": "mlp",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 8,
            "map_kwargs": {
                "encoder_type": "unet_o3",
                "pool_types": ('avg', 'max', 'gem'),
                "patch_conditioning": ("side_info")
            },
            "bandpower_kwargs": {
                "hidden_multiple": 32,
                "dropout": 0,
            }
        },
        "latent_dim": 8 + 8,
        "pretrained_band_ckpt_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/bandpower_mlp_lat8_9param/",
        "freeze_band": True,
        "epochs": 100,
        "batch_size": 100,
        "scheduler_type": "cyclic",
        "scheduler_kwargs": {'warmup': 500, 'min_factor': 0.1, "cyclic_period_steps": 2500},
        "lr": 0.0001,
        "use_KL_loss": False,
        "flow_kwargs": {"hidden_features": 32},
        "max_trainval_cosmos": [100, 120],
        "cosmo_param_names": ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"],
        "repeats": 3,
    },
    "finetune_hybrid_16_9param_ensemble": {
        "data_patterns": "/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["mixed_bandpowers", "E_north", "E_south"],
        "model_kwargs": {
            "bandpower_type": "mlp",
            "map_encoder_type": "o3_dual",
            "bandpower_latent_dim": 8,
            "map_kwargs": {
                "encoder_type": "unet_o3",
                "pool_types": ('avg', 'max', 'gem'),
                "patch_conditioning": ("side_info")
            },
            "bandpower_kwargs": {
                "hidden_multiple": 32,
                "dropout": 0,
            }
        },
        "latent_dim": 8 + 8,
        "epochs": 10,
        "batch_size": 128,
        "lr": 1.e-5,
        "scheduler_type": "exp",
        "scheduler_kwargs": {'warmup': 0},
        "use_KL_loss": False,
        "flow_kwargs": {"hidden_features": 32},
        "checkpoint_path": "/share/gpu5/asaoulis/transfer_models/checkpoints/glass_hybrid_patches_16_9param/",
        "max_trainval_cosmos": [10, 15, 20, 30, 40, 60, 80],
        "train_frac": 0.6,
        "val_frac": 0.3,
        "project": "gower-finetuning",
        "match_num_cosmo": False,
        "cosmo_param_names": ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"],
        "ensemble_repeats": 9,
        "repeats": 3,
    },
}
