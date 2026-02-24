from config.default import get_default_config
from config.experiments import experiments
from src.ml.eval.utils import evaluate_best_checkpoint, parse_results, load_best_model_and_build_posterior
from src.ml.eval.ensemble import evaluate_best_ensemble
from pathlib import Path
import numpy as np
import torch

print(torch.cuda.is_available())

def load_config(experiment_name: str):
    """Load config in a way consistent with train.py."""
    experiment_config = experiments[experiment_name]

    config = get_default_config()
    config.experiment_name = experiment_name

    # Set all non-list values directly on the config, but skip max_trainval_cosmos
    for key, val in experiment_config.items():
        if key == "max_trainval_cosmos":
            continue
        setattr(config, key, val)

    return config, experiment_config

from src.ml.utils import prepare_data_parameters

# Base config used only for data_parameters (like in your old script)
experiments_to_evaluate = [
    # "bandpower_mlp_representation_varying_sizes_bs128",
    # "hybrid_frozen_representation_varying_sizes",
    # "finetune_hybrid_representation_E_mode",
    # "hybrid_frozen_representation"
    "hybrid_frozen_representation_cyclic_varying_sizes"
]

for experiment_name in experiments_to_evaluate:
    if experiment_name not in experiments:
        print(f"Experiment '{experiment_name}' not found in config.experiments, skipping.")
        continue

    config, experiment_config = load_config(experiment_name)

    # Handle max_trainval_cosmos similarly to train.py
    max_tv = experiment_config.get("max_trainval_cosmos", None)

    if isinstance(max_tv, (list, tuple)):
        # Multiple cosmos: evaluate each separately
        for n_cosmo in max_tv:
            cfg = get_default_config()
            # Reapply all non-list values again to fresh config
            cfg.experiment_name = config.experiment_name
            cfg.test_shape_noise_idx = [0]
            for key, val in experiment_config.items():
                if key == "max_trainval_cosmos":
                    continue
                setattr(cfg, key, val)

            cfg.max_trainval_cosmos = int(n_cosmo)
            # This string is what you said you want for matching paths
            cfg.match_string = f"ncosmo{n_cosmo}"
            print(f"Evaluating '{experiment_name}' with max_trainval_cosmos={n_cosmo}, match_string={cfg.match_string}", flush=True)
            data_parameters = prepare_data_parameters(cfg)

            evaluate_best_checkpoint(cfg, data_parameters[0]["cosmo"], data_parameters[3])
    else:
        # Single or no max_trainval_cosmos
        config.test_shape_noise_idx = [0]
        data_parameters = prepare_data_parameters(config)
        if max_tv is not None:
            config.max_trainval_cosmos = int(max_tv)
            config.match_string = f"ncosmo{int(max_tv)}"
        else:
            # If you want a default match_string when ncosmo is not used, set it here if needed
            config.match_string = ""

        print(f"Evaluating '{experiment_name}' with max_trainval_cosmos={getattr(config, 'max_trainval_cosmos', None)}, match_string={config.match_string}")
        evaluate_best_checkpoint(cfg, data_parameters[0]["cosmo"], data_parameters[3])
