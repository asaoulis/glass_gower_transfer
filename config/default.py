"""Default values for every field of an experiment config.

Experiment rows are override dicts layered onto get_default_config(), one setattr per key.
The trailing comment on a field says what it controls and what None means.
"""
import os

import ml_collections

def get_default_config():
    config = ml_collections.ConfigDict()
    # Training
    config.training_mode = "nde"
    config.lr = 0.0004
    config.epochs = 80
    config.val_check_interval = None  # None = once per epoch, float in (0, 1] = fraction of an epoch, int = every N batches
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
    config.patch_aux_weight = 0.0  # weight of the map-only auxiliary VMIM head, 0 = off
    config.patch_aux_flow_kwargs = {}
    config.freeze_backbone = False
    config.backbone_prefix = 'shared_cnn.backbone.'
    # Cached-embedding head training (src/ml/embeddings), all off by default
    config.embedding_cut = None            # for example 'hybrid_pre_head'
    config.reuse_embedding_cache = False   # reuse the embedding cache during training runs too
    config.embedding_cache_name = None     # share one cache dir across head-variant targets
    config.embedding_head_type = None      # for example 'hybrid_features'
    config.embedding_head_kwargs = {}

    # Training-speed options, all off by default
    config.ml_perf = {
        "amp": False,            # bf16 autocast on the map encoder only, the flow stays fp32
        "compile": "none",       # "none" | "backbone" | "reduce-overhead" | "default"
        "tf32": False,           # TF32 matmul/cudnn (free when amp off; no-op with amp)
        "fused_adam": False,     # AdamW(fused=True)
    }
    config.prefetch_factor = None      # DataLoader prefetch_factor (workers>0)
    config.persistent_workers = None   # keep workers alive across epochs (None=auto)
    config.precision = None            # None = fit_model default
    config.preload_data = False        # cache every item in RAM at dataset build (small vector stores such as the HOS sidecars)
    config.exclude_files = None         # basenames dropped from train/val/test after the cosmology split (JSON list or list)
    config.include_files_regex = None   # keep only files whose basename matches (re.search), applied before the cosmology split

    # Data loading
    config.data_patterns = "/share/gpu5/asaoulis/transfer_datasets/gower_mocks/output_*.h5"
    config.dataset_nested_keys = None
    # Alternative to dataset_nested_keys: E_north, E_south, B_north, B_south, bandpowers, bandpower_ls, cls
    config.dataset_quantities = ["mixed_bandpowers", "E_north", "E_south", "B_north", "B_south",]
    # Smoothing tag of the E/B groups, 'pixelised_results/{E,B}_<tag>/...'. None = bare 'E'/'B' groups.
    config.eb_map_variant = None
    # Checkpoint root. GLASS_MODELS_ROOT overrides it for local runs.
    config.base_path = os.environ.get("GLASS_MODELS_ROOT",
                                      "/share/gpu5/asaoulis/transfer_models")
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
    # JSON lock-file path or list of Gower sim_ids forced into the test split. None = normal split.
    config.fixed_test_sim_ids = None
    # Keep only the first N test cosmologies by sorted sim_id, after train/val selection. None = all.
    config.N_test_cosmologies = None
    config.emb_test_batch_size = None  # embeddings test-loader batch size, None = 32
    config.shuffle_train = True
    config.num_workers = None
    config.pin_memory = False
    config.stack_groups = False
    config.augment_eb_patches = True
    # Cyclic RA roll of the patches. Exact only for a full-RA footprint (Euclid), keep False for KiDS.
    config.augment_eb_ra_roll = False
    config.n_tomo_bins = 6  # input channels = n_tomo_bins (E only) or 2 * n_tomo_bins (E+B), 6 for KiDS, 13 for Euclid
    config.scaler_fit_max_obs = 1000  # number of files the data scalers are fit on
    config.scaler_fit_max_obs_by_key = None  # per-key override of scaler_fit_max_obs, for example {'mixed_bandpowers': 1000}
    config.log_throughput = False  # print a per-epoch throughput line
    # 'self' = per-bin footprint standardisation of every E/B map at load time. None = off.
    config.eb_noise_norm = None

    config.dataset_size = None

    config.scaler_fit_seed = 0  # seed of the scaler-fit file subsample

    config.scaler_options = {
            'data': {'type': 'standard', 'keys': None},
            'cosmo': {'type': 'preset'},
    }
    config.project = "kids-transfer-tests"
    config.train_val_selection_strategy = "random"
    config.train_val_selection_cosmo_params = None
    config.train_val_stratified_bins = "auto"

    config.repeats = 1
    config.repeat_indices = None  # explicit repeat indices, overrides repeats
    config.experiment_name = None
    config.match_string = ""
    config.pretrained_band_match_string = ""
    config.num_flow_heads = 1
    config.load_pretrained_flow = False
    config.run_training = True # for embedding eval pattern
    config.run_evaluation = True # embeddings: set False to skip post-training (MCMC) evaluation
    config.sampling_num_jobs = None  # MCMC pool size for ensemble-NLE sampling, None = generate_samples default

    # Embeddings pipeline options
    config.scale_embeddings = True  # standardise embedding vectors before training the NDE on them
    # {'k': int}: WhitenPCAScaler fit once on the pretrain split and reused downstream, replaces scale_embeddings
    config.whiten_embeddings = None
    # Abort if finetune epoch-0 val NLL exceeds the pretrain best by this many nats. None or <=0 disables.
    config.whiten_warmstart_max_gap_nats = 12.0
    config.accumulate_grad_batches = 1
    config.inference_mode = "npe"

    return config
