import sys
from config.default import get_default_config
from config.experiments import experiments
from src.train.utils import train_model

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
    list_key, list_values = retrieve_first_list_from_experiments(experiment_config)

    for value in list_values:
        config = get_default_config()
        config.experiment_name = experiment_name
        # Set the non-list values
        for key, val in experiment_config.items():
            if key != list_key:
                setattr(config, key, val)
        # Set the list value
        setattr(config, list_key, value)
        print(f"Running experiment '{experiment_name}' with {list_key}={value}")
        train_model(config)
