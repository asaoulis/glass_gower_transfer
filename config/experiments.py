
experiments = {
    "default": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_full_only_mocks/output_*.h5"
    },
    "cnn_shared": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_full_only_mocks/output_*.h5",
        "model_type": "kids_o3_dual",
        "latent_dim": 512,
    },
    "cnn_transform": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_full_only_mocks/output_*.h5",
        "model_type": "kids_combined_cnn_transformer",
    },
    "cnn_transform_maf": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_full_only_mocks/output_*.h5",
        "model_type": "kids_combined_cnn_transformer",
        "flow_type": "maf",
    },
    "bandpower_mlp": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_full_only_mocks/output_*.h5",
        "model_type": "kids_bandpowers_mlp",
        "dataset_quantities": ["bandpowers"],
    },
    "bandpower_cnn": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_full_only_mocks/output_*.h5",
        "model_type": "kids_bandpowers_cnn1d",
        "dataset_quantities": ["bandpowers"],
    },
    "hybrid_bandpower_maps":{
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_full_only_mocks/output_*.h5",
        "model_type": "kids_hybrid_bandpowers_maps",
        "dataset_quantities": ["bandpowers", "E_north", "E_south", "B_north", "B_south"],
        "model_kwargs": {
            "bandpower_type": "cnn",
            "transformer_kwargs": 
                { "hidden":12, "channels_per_map":6, "d_model":256, "n_heads":4, "n_layers":4, "n_queries":8, "dropout":0.1}
        },
        "latent_dim": 512,
    }
}
