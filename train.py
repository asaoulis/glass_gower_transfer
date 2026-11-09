"""Train one named experiment: python train.py <experiment_name>.

Merges every config module into the registry, builds the config from config.default plus the
experiment's override dict, and calls train_model once per max_trainval_cosmos value.
"""
import sys
from config.default import get_default_config
from config.experiments import experiments
from config.paper_2026 import paper_2026_experiments
from config.kids_legacy import kids_legacy_experiments
from config.kids_legacy_novd import kids_legacy_novd_experiments
from config.euclid import euclid_experiments
from config.kids_legacy_bgp import kids_legacy_bgp_experiments
from config.kids_legacy_vdq import kids_legacy_vdq_experiments
from config.kids_legacy_hos import kids_legacy_hos_experiments
from src.ml.models.utils import train_model
import os

experiments.update(paper_2026_experiments)
experiments.update(kids_legacy_experiments)
experiments.update(kids_legacy_novd_experiments)
experiments.update(euclid_experiments)
experiments.update(kids_legacy_bgp_experiments)
experiments.update(kids_legacy_vdq_experiments)
experiments.update(kids_legacy_hos_experiments)

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
        print("Usage: python train.py <experiment_name>")
        sys.exit(1)

    experiment_name = sys.argv[1]
    if experiment_name not in experiments:
        print(f"Error: Experiment '{experiment_name}' not found.")
        sys.exit(1)

    experiment_config = experiments[experiment_name]

    config = get_default_config()
    config.experiment_name = experiment_name

    for key, val in experiment_config.items():
        if key == "max_trainval_cosmos":
            continue
        setattr(config, key, val)

    # REPEAT_INDICES (env) overrides config.repeat_indices so one experiment can run as several
    # parallel jobs under one checkpoint dir.
    _ri = os.environ.get("REPEAT_INDICES", "").strip()
    if _ri:
        config.repeat_indices = [int(x) for x in _ri.split(",") if x.strip() != ""]
        print(f"[train.py] REPEAT_INDICES override -> repeat_indices={config.repeat_indices}")

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
