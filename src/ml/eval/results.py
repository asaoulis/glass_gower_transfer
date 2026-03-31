import os
import json
import numpy as np
from pathlib import Path

def flatten_dict(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def unflatten_dict(d, sep="."):
    result = {}
    for k, v in d.items():
        keys = k.split(sep)
        cur = result
        for key in keys[:-1]:
            cur = cur.setdefault(key, {})
        cur[keys[-1]] = v
    return result

import os

def parse_results(experiment_name, base_path="/share/gpu0/asaoulis/cmd/checkpoints"):
    experiment_path = os.path.join(base_path, f"{experiment_name}")
    # check if experiment path exists
    run_folders = [
        os.path.join(experiment_path, d)
        for d in os.listdir(experiment_path)
        if os.path.isdir(os.path.join(experiment_path, d))
    ]

    results = {}
    aggregated_results = {}

    # Collect raw ensemble metrics (flattened) keyed by match_string
    ensemble_results = {}

    # ---- Parse ensemble results ----
    # Files are saved as: ensemble_evaluation_results_{match_string}.json
    # where match_string itself may contain underscores (e.g. ncosmo30_0).
    prefix = "ensemble_evaluation_results_"
    suffix = ".json"
    ensemble_result_paths = Path(experiment_path).glob(f"{prefix}*.json")
    for ensemble_result in ensemble_result_paths:
        name = ensemble_result.name
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        match_string = name[len(prefix) : -len(suffix)]

        with open(ensemble_result, "r") as f:
            data = json.load(f)

        metrics = data.get("metrics", data)
        ensemble_results[match_string] = flatten_dict(metrics)

    # ---- Parse individual runs ----
    for run_folder in run_folders:
        results_file = os.path.join(run_folder, "evaluation_results.json")
        if not os.path.exists(results_file):
            continue

        with open(results_file, "r") as f:
            data = json.load(f)

        run_name = os.path.basename(run_folder)
        metrics = flatten_dict(data["metrics"])
        results[run_name] = metrics

    # ---- Aggregate individual runs by base run name ----
    for run_name, metrics in results.items():
        base_name = (
            "_".join(run_name.split("_")[:-1])
            if run_name.split("_")[-1].isdigit()
            else run_name
        )

        aggregated_results.setdefault(base_name, {})

        for metric, value in metrics.items():
            aggregated_results[base_name].setdefault(metric, [])
            aggregated_results[base_name][metric].append(value)

    # ---- Aggregate ensemble results by match_string stem (drop trailing _repeat_idx) ----
    aggregated_ensemble_results = {}
    for match_string, metrics in ensemble_results.items():
        base_match = (
            "_".join(match_string.split("_")[:-1])
            if match_string.split("_")[-1].isdigit()
            else match_string
        )
        aggregated_ensemble_results.setdefault(base_match, {})
        for metric, value in metrics.items():
            aggregated_ensemble_results[base_match].setdefault(metric, [])
            aggregated_ensemble_results[base_match][metric].append(value)

    def _aggregate_metric_values(values):
        """Aggregate a list of metric values.

        - If all values are numeric -> return {mean, stderr}
        - Else -> return values as-is (useful for strings/objects)
        """
        # Fast path: attempt numeric aggregation
        try:
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                return {"mean": np.nan, "stderr": np.nan}
            if len(arr) == 1:
                return {"mean": float(arr[0]), "stderr": 0.0}
            mean = float(np.mean(arr))
            stderr = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
            return {"mean": mean, "stderr": stderr}
        except (TypeError, ValueError):
            return values

    # ---- Build final results: ensemble (aggregated) + individual runs (aggregated) ----
    final_results = {}

    # Ensemble results: aggregated across repeats
    for base_match, metrics in aggregated_ensemble_results.items():
        final_results[f"ensemble_{base_match}"] = {}
        for metric, values in metrics.items():
            final_results[f"ensemble_{base_match}"][metric] = _aggregate_metric_values(values)

    # Individual run results: aggregated across repeats
    for base_name, metrics in aggregated_results.items():
        final_results[base_name] = {}
        for metric, values in metrics.items():
            final_results[base_name][metric] = _aggregate_metric_values(values)

    return final_results