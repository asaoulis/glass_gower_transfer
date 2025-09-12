import ml_collections

def get_default_config():
    config = ml_collections.ConfigDict()
    # Training
    config.lr = 0.0004
    config.epochs = 400
    config.batch_size = 32
    config.val_batch_size = None  # defaults to batch_size if None
    config.test_batch_size = None # defaults to val_batch_size if None
    config.latent_dim = 512
    config.extra_blocks = 0
    config.checkpoint_path = None
    config.scheduler_type = 'exp'
    config.optimizer_kwargs = {'weight_decay': 0.01, 'betas': (0.9, 0.999)}
    config.scheduler_kwargs = {'warmup': 250, 'gamma': 0.99}
    config.model_type = "kids_o3_dual"
    config.model_kwargs = {
        "hidden": 12,
        "channels_per_map": 6,
        "d_model": 512,
        "n_heads": 4,
        "n_layers": 4,
        "n_queries": 8,
    }
    config.freeze_cnn = False

    # Data loading (new dataset interface)
    config.data_patterns = "/share/gpu5/asaoulis/transfer_datasets/gower_full_only_mocks/output_*.h5"
    config.dataset_nested_keys = None
    # Alternatively, specify simple quantity names and we will build dataset_nested_keys
    # Valid options (defaults): E_north, E_south, B_north, B_south, bandpowers, bandpower_ls, cls
    config.dataset_quantities = ["E_north", "E_south", "B_north", "B_south",]

    config.cosmo_param_names = [
        "omega_m", "s8"
    ]
    config.train_frac = 0.8
    config.val_frac = 0.1
    config.test_frac = 0.1
    config.split_seed = 42
    config.shuffle_train = True
    config.num_workers = 0
    config.pin_memory = False
    config.stack_groups = False

    # Optional: limit training dataset size (None => no trimming)
    config.dataset_size = None

    # Scaling options
    # - data: per-key standard scaling (keys=None => scale all keys in dataset_nested_keys)
    # - cosmo: min/max scaling for cosmological parameter vectors
    config.scaler_options = {
        'data': {'type': 'standard', 'keys': None},
        'cosmo': {'type': 'minmax'},
    }


    # Misc/legacy fields kept for compatibility with older code paths
    config.dataset_name = "illustris"
    config.dataset_suite = "LH"
    config.scaling_dataset = None
    config.repeats = 6
    config.experiment_name = None
    config.data_seed = None
    config.log_normal_dataset_path = None
    config.unpaired = False
    config.match_string = None

    return config