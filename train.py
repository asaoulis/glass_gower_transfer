import sys
from config.default import get_default_config
from config.experiments import experiments
from config.ablations import ablation_experiments
from config.kids_legacy import kids_legacy_experiments
from config.kids_legacy_counts import kids_legacy_counts_experiments
from config.kids_legacy_novd import kids_legacy_novd_experiments
from config.kids_legacy_dn import kids_legacy_dn_experiments
from src.ml.models.utils import train_model
# import os
# os.environ["MASTER_ADDR"] = "127.0.0.1"
# os.environ["MASTER_PORT"] = "29500"  # or any free port
import os

experiments.update(ablation_experiments)  # Merge ablation experiments into main experiments dict
experiments.update(kids_legacy_experiments)  # Merge KiDS-Legacy NLA-M configs
experiments.update(kids_legacy_counts_experiments)  # Merge counts-normalisation rerun configs
experiments.update(kids_legacy_novd_experiments)  # Merge NO-VD production suite configs
experiments.update(kids_legacy_dn_experiments)  # dual-normalisation arm-comparison suite

# ... rest of your code ...
def retrieve_first_list_from_experiments(experiments):
    list_key = None
    list_values = None
    for key, value in experiments.items():
        if isinstance(value, list):
            list_key = key
            list_values = value
            break
    return list_key, list_values

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python camels_regress.py <experiment_name>")
        sys.exit(1)

    experiment_name = sys.argv[1]
    if experiment_name not in experiments:
        print(f"Error: Experiment '{experiment_name}' not found.")
        sys.exit(1)

    experiment_config = experiments[experiment_name]

    config = get_default_config()
    config.experiment_name = experiment_name

    # Set all non-list values directly on the config, but skip max_trainval_cosmos
    for key, val in experiment_config.items():
        if key == "max_trainval_cosmos":
            continue
        setattr(config, key, val)

    # Submit-time override: REPEAT_INDICES env (set by the gatekeeper from run_remote --repeat-indices)
    # OVERRIDES config.repeat_indices, so the SAME experiment name can be split across multiple parallel
    # jobs (e.g. "0,1" and "2,3") while keeping all repeats under one checkpoints/<exp>/ dir — required
    # for correct per-repeat band loading + ensemble loading. No config edit/re-sync per split.
    _ri = os.environ.get("REPEAT_INDICES", "").strip()
    if _ri:
        config.repeat_indices = [int(x) for x in _ri.split(",") if x.strip() != ""]
        print(f"[train.py] REPEAT_INDICES override -> repeat_indices={config.repeat_indices}")

    # Handle max_trainval_cosmos separately so it can be either scalar or list in experiments.py
    max_tv = experiment_config.get("max_trainval_cosmos", None)
    if isinstance(max_tv, (list, tuple)):
        for n_cosmo in max_tv:
            print(f"Running experiment '{experiment_name}' with max_trainval_cosmos={n_cosmo}")
            config.max_trainval_cosmos = int(n_cosmo)
            train_model(config)
    else:
        if max_tv is not None:
            config.max_trainval_cosmos = int(max_tv)
        print(f"Running experiment '{experiment_name}'")
        train_model(config)
