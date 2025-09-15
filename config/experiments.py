
experiments = {
    "default": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_full_only_mocks/output_*.h5"
    },
    "cnn_transform": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_full_only_mocks/output_*.h5",
        "model_type": "kids_combined_cnn_transformer",
    },
    "bandpower_mlp": {
        "data_patterns":"/share/gpu5/asaoulis/transfer_datasets/gower_full_only_mocks/output_*.h5",
        "model_type": "kids_bandpowers_mlp",
        "dataset_quantities": ["bandpowers"],
    }
}
