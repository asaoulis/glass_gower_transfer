from config.default import get_default_config
from config.experiments import experiments
from config.ablations import ablation_experiments
from config.kids_legacy import kids_legacy_experiments
from src.ml.eval.utils import evaluate_best_checkpoint
from pathlib import Path
import numpy as np
import torch
from copy import copy

experiments.update(ablation_experiments)  # Combine experiments and ablations into a single dict
experiments.update(kids_legacy_experiments)  # KiDS-Legacy NLA-M configs

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
from src.ml.models.utils import apply_repeat_config

# Base config used only for data_parameters
experiments_to_evaluate = [
    # "bandpower_mlp_lat8_2param",
    # "hybrid_patches_16_2param",
    # "finetune_hybrid_16_2param",
    # "glass_hybrid_patches_16_9param",
    # "finetune_hybrid_16_9param",
    # "finetune_hybrid_16_2param_ensemble_stratify",
    # "finetune_hybrid_16_9param_ensemble",
    # "finetune_hybrid_16_9param_ensemble_stratify",
    # "finetune_hybrid_16_9param_ensemble_kcenter_sampling",
    # "bandpower_mlp_lat8_9param",
    # "hybrid_patches_16_9param",
    # "embeddings_pretrained_flow_9param",
    # "finetune_ablation_glass_hybrid_OFF",
    # "finetune_ablation_glass_no_side",
    # "finetune_ablation_glass_no_GEM",
    # "finetune_ablation_glass_MAF",
    # "finetune_hybrid_16_9param",
    # "finetune_ablation_glass_no_cyclic"
    # "finetune_ablation_glass_B_modes"
    # vicreg-nle-first-test (superseded by the z8 production run):
    # "gower_npe_finetune_nla_m_base",
    # "gower_npe_finetune_nla_m_vicreg",
    # PRODUCTION eval (run on v100 as jobs finish; evaluate_best_checkpoint skips repeats/runs with no
    # checkpoint yet, so re-running is safe): main-variate Gower NPE ens9 (all 5 repeats) + the GLASS
    # sub-variate encoder-finetunes (NPE-style compressors). Each writes evaluation_results.json /
    # ensemble_evaluation_results_*.json + the P0 posterior_samples.npz / ensemble_posterior_samples_*.npz.
    "gower_npe_finetune_nla_m_z8",
    "glass_encoder_finetune_nla_z_z8",
    "glass_encoder_finetune_nla_z8",
    "glass_encoder_finetune_no_vd_z8",
]

# --- Model-misspecification eval (one model × many variate datasets) ------------------------
# When True, eval.py ONLY runs the misspec driver (src/ml/eval/misspec.py): the production NPE
# r0 ensemble evaluated on the TEST split of every Gower variate dataset with the ORIGINAL
# nla_m training scalers injected (never refit). Outputs land under
# checkpoints/<exp>/misspec/<variate>/. Set False to restore the standard list eval above.
RUN_MISSPEC = True
if RUN_MISSPEC:
    from src.ml.eval.misspec import run_misspecification_eval

    run_misspecification_eval(base_experiment="gower_npe_finetune_nla_m_z8", repeat_index=0)
    experiments_to_evaluate = []  # skip the standard loop below

for experiment_name in experiments_to_evaluate:
    if experiment_name not in experiments:
        print(f"Experiment '{experiment_name}' not found in config.experiments, skipping.")
        continue

    config, experiment_config = load_config(experiment_name)

    # Handle max_trainval_cosmos similarly to train.py
    max_tv = experiment_config.get("max_trainval_cosmos", None)

    # Keep evaluation consistent with training: evaluate across repeats. Production configs use
    # `repeat_indices` (not `repeats`); fall back to range(repeats) for legacy configs.
    repeats = getattr(config, "repeats", 1)
    repeat_idxs = list(getattr(config, "repeat_indices", None) or range(repeats))

    if isinstance(max_tv, (list, tuple)):
        # Multiple cosmos: evaluate each separately
        for n_cosmo in max_tv:
            cfg = get_default_config()
            cfg.experiment_name = config.experiment_name
            cfg.test_shape_noise_idx = [0]

            for key, val in experiment_config.items():
                if key == "max_trainval_cosmos":
                    continue
                setattr(cfg, key, val)

            cfg.max_trainval_cosmos = int(n_cosmo)
            cfg.match_num_cosmo = True  # Ensure match_string includes n_cosmo
            for i in repeat_idxs:
                cfg_copy = copy(cfg)  # Avoid mutating cfg across repeats
                # Apply the exact repeat match_string logic used by train_model
                repeat_match, _ = apply_repeat_config(cfg_copy, i)
                cfg_copy.match_string = repeat_match

                print(
                    f"Evaluating '{experiment_name}' ncosmo={n_cosmo} repeat={i} match_string={cfg_copy.match_string}",
                    flush=True,
                )

                scalers, _, _, test_loader = prepare_data_parameters(cfg_copy)
                evaluate_best_checkpoint(cfg_copy, test_loader, scalers["cosmo"])
    else:
        # Single or no max_trainval_cosmos
        config.test_shape_noise_idx = [0]
        if max_tv is not None:
            config.max_trainval_cosmos = int(max_tv)

        for i in repeat_idxs:
            repeat_match, _ = apply_repeat_config(config, i)
            config.match_string = repeat_match

            print(
                f"Evaluating '{experiment_name}' max_trainval_cosmos={getattr(config, 'max_trainval_cosmos', None)} repeat={i} match_string={config.match_string}",
                flush=True,
            )

            scalers, _, _, test_loader = prepare_data_parameters(config)
            evaluate_best_checkpoint(config, test_loader, scalers["cosmo"])
