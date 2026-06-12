import ml_collections

def get_default_config():
    config = ml_collections.ConfigDict()
    # Training
    config.training_mode = "nde"
    config.lr = 0.0004
    config.epochs = 80
    config.batch_size = 32
    config.val_batch_size = None  # defaults to batch_size if None
    config.test_batch_size = None # defaults to val_batch_size if None
    config.latent_dim = 256
    config.extra_blocks = 0
    config.match_num_cosmo = True
    config.checkpoint_path = None
    config.pretrained_band_ckpt_path = None
    config.freeze_band = False
    config.scheduler_type = 'exp'
    config.optimizer_kwargs = {'weight_decay': 0.01, 'betas': (0.9, 0.999)}
    config.scheduler_kwargs = {'warmup': 250, 'gamma': 0.9999}
    config.flow_type = 'nsf'  # Options: 'nsf', 'maf', 'zuko_nsf'
    config.model_type = "kids_o3_dual"
    config.model_kwargs = {
        "hidden": 12,
        "d_model": 256,
        "n_heads": 4,
        "n_layers": 4,
        "n_queries": 8,
    }
    config.freeze_cnn = False
    config.flow_kwargs = {}

    # Data loading (new dataset interface)
    config.data_patterns = "/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5"
    config.dataset_nested_keys = None
    # Alternatively, specify simple quantity names and we will build dataset_nested_keys
    # Valid options (defaults): E_north, E_south, B_north, B_south, bandpowers, bandpower_ls, cls
    config.dataset_quantities = ["mixed_bandpowers", "E_north", "E_south", "B_north", "B_south",]
    # Which smoothing variant the logical E/B map names resolve to in the HDF5
    # ('pixelised_results/{E|B}_{eb_map_variant}/{north,south}'). None => legacy bare
    # 'pixelised_results/{E|B}/...' groups (old datasets). For datasets written by the
    # multi-variant simulator set e.g. "fwhm8", "fwhm6_lcut1024" or "fwhm6_lmin76_lcut1024".
    # Explicit variant-tagged quantity names (e.g. "E_fwhm8_north") bypass this and resolve directly.
    config.eb_map_variant = None
    config.base_path = "/share/gpu5/asaoulis/transfer_models"
    config.cosmo_param_names = [
        "omega_m", "sigma_8"
    ]
    config.test_shape_noise_idx = None
    config.max_trainval_cosmos = None  # scalar / None, not a list
    config.split_on_source_experiments = False
    config.train_frac = 0.8
    config.val_frac = 0.1
    config.test_frac = 0.1
    config.split_seed = 42
    config.N_extra_test_cosmologies = None  # int or None
    # Opt-in fixed test set (EXTRA functionality; None => off, normal split unchanged).
    # Path to a JSON lock-file of Gower sim_ids (e.g.
    # "config/fixed_test_sets/gower_test_ids.json") OR an explicit list of ints. When set,
    # exactly those sim_ids present on disk are FORCED into the test split (shared across
    # all experiments) and removed from the train/val pool. Ids absent on disk are ignored;
    # if forcing would empty train/val it falls back to the normal split (with a warning).
    config.fixed_test_sim_ids = None
    config.shuffle_train = True
    config.num_workers = None
    config.pin_memory = False
    config.stack_groups = False
    config.augment_eb_patches = True

    # Optional: limit training dataset size (None => no trimming)
    config.dataset_size = None

    # Scaling options
    # - data: per-key standard scaling (keys=None => scale all keys in dataset_nested_keys)
    # - cosmo: min/max scaling for cosmological parameter vectors
    # config.scaler_options = {
    #     'data': {'type': 'standard', 'keys': None},
    #     'cosmo': {'type': 'minmax'},
    # }
    config.scaler_options = {
            'data': {'type': 'standard', 'keys': None},
            'cosmo': {'type': 'preset'},
    }
    config.project = "kids-transfer-tests"
    config.train_val_selection_strategy = "random"
    config.train_val_selection_cosmo_params = None
    config.train_val_stratified_bins = "auto"

    # Misc/legacy fields kept for compatibility with older code paths
    config.dataset_name = "illustris"
    config.dataset_suite = "LH"
    config.scaling_dataset = None
    # Number of train repeats (int). Use `repeat_indices` to run a subset.
    config.repeats = 1
    # Optional explicit repeat indices (tuple/list of ints). If set, overrides `repeats`.
    config.repeat_indices = None
    config.experiment_name = None
    config.data_seed = None
    config.log_normal_dataset_path = None
    config.unpaired = False
    config.match_string = ""
    config.pretrained_band_match_string = ""
    config.num_flow_heads = 1
    config.load_pretrained_flow = False
    config.run_training = True # for embedding eval pattern

    # Embeddings pipeline options
    # If False, do not normalise/standardise the precomputed embedding vectors
    # before training the NDE on embeddings.
    config.scale_embeddings = True
    config.accumulate_grad_batches = 1
    config.inference_mode = "npe"

    return config