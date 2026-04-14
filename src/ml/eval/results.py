import os
import json
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt

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


def parse_results(
    experiment_name,
    base_path="/share/gpu0/asaoulis/cmd/checkpoints",
    include_tarp_curves=False,
):
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

    # Optional: collect per-run and ensemble TARP curve payloads
    tarp_curves_by_run = {}
    ensemble_tarp_curves = {}

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

        if include_tarp_curves:
            intervals_path = Path(experiment_path) / f"ensemble_tarp_credible_intervals_{match_string}.json"
            if intervals_path.exists():
                with open(intervals_path, "r") as f:
                    interval_data = json.load(f)
                ensemble_tarp_curves[match_string] = flatten_dict(interval_data)

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

        if include_tarp_curves:
            intervals_path = os.path.join(run_folder, "tarp_credible_intervals.json")
            if os.path.exists(intervals_path):
                with open(intervals_path, "r") as f:
                    interval_data = json.load(f)
                tarp_curves_by_run[run_name] = flatten_dict(interval_data)

    def _strip_repeat_from_name(name):
        """Drop repeat/ensemble markers from names for stable aggregation.

        Supports names where repeat index is not necessarily the final token,
        e.g. "pretrain_ncosmo20_0_finetune_hybrid_16_8param".
        Also removes an optional immediate "_ens{j}" marker.
        """
        # Prefer removing repeat right after ncosmo token when present.
        name_no_repeat = re.sub(r"(ncosmo\d+)_\d+(?=(?:_ens\d+)?(?:_|$))", r"\1", name, count=1)
        name_no_repeat_or_ens = re.sub(r"_ens\d+(?=_|$)", "", name_no_repeat, count=1)
        if name_no_repeat_or_ens != name:
            return name_no_repeat_or_ens

        # Fallback to legacy behaviour when no ncosmo marker is present.
        parts = name.split("_")
        if parts and parts[-1].isdigit():
            return "_".join(parts[:-1])
        return name

    # ---- Aggregate individual runs by base run name ----
    for run_name, metrics in results.items():
        base_name = _strip_repeat_from_name(run_name)

        aggregated_results.setdefault(base_name, {})

        for metric, value in metrics.items():
            aggregated_results[base_name].setdefault(metric, [])
            aggregated_results[base_name][metric].append(value)

    # ---- Aggregate ensemble results by normalized match_string stem ----
    aggregated_ensemble_results = {}
    for match_string, metrics in ensemble_results.items():
        base_match = _strip_repeat_from_name(match_string)
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

    if not include_tarp_curves:
        return final_results

    return {
        "metrics": final_results,
        "raw_metrics": {
            "runs": results,
            "ensembles": ensemble_results,
        },
        "tarp_curves": {
            "runs": tarp_curves_by_run,
            "ensembles": ensemble_tarp_curves,
        },
    }


def _derive_x_from_run_name(run_name, default_value=None):
    match = re.search(r"ncosmo(\d+)", run_name)
    if match:
        return int(match.group(1))
    return default_value


def _resolve_tarp_curve_keys(curve_type):
    if curve_type == "bootstrap":
        return "ecp_bootstrap", True
    if curve_type == "mean":
        return "ecp", False
    raise ValueError("curve_type must be either 'bootstrap' or 'mean'")


def _aggregate_curve(ecp_value, use_bootstrap):
    arr = np.asarray(ecp_value, dtype=float)
    if use_bootstrap:
        if arr.ndim != 2:
            raise ValueError("Expected bootstrapped ECP to have shape (num_bootstrap, num_alpha)")
        return arr.mean(axis=0), arr.std(axis=0, ddof=0)

    if arr.ndim != 1:
        raise ValueError("Expected ECP to be a 1D array when curve_type='mean'")
    return arr, np.zeros_like(arr)


def _resolve_plot_title(metric_key, plotting_name_conversions=None):
    if plotting_name_conversions is None:
        return metric_key

    if metric_key in plotting_name_conversions:
        return plotting_name_conversions[metric_key]

    if metric_key.endswith(".calibration_error"):
        base_key = metric_key[: -len(".calibration_error")]
        if base_key in plotting_name_conversions:
            return plotting_name_conversions[base_key]
    else:
        cal_key = f"{metric_key}.calibration_error"
        if cal_key in plotting_name_conversions:
            return plotting_name_conversions[cal_key]

    return metric_key


def _resolve_calibration_error_key(metric_key):
    if metric_key.endswith(".calibration_error"):
        return metric_key
    return f"{metric_key}.calibration_error"


def plot_tarp_calibration_curves(
    tarp_curves,
    metric_key,
    run_filter="",
    sigma_levels=(1,),
    curve_type="bootstrap",
    x_from_run_name=False,
    run_metrics=None,
    ax=None,
    title=None,
    plotting_name_conversions=None,
    figsave=None,
):
    """Plot one or more TARP calibration curves from parsed curve payloads.

    Args:
        tarp_curves: dict from parse_results(..., include_tarp_curves=True)["tarp_curves"]["runs"]
        metric_key: key prefix like "tarp.full", "tarp.per_param.omega_m", or
            "tarp.subsets.sigma_8__omega_m__w0"
        run_filter: optional substring to select runs.
        sigma_levels: iterable of k values for k-sigma uncertainty bands.
        curve_type: "bootstrap" to use ecp_bootstrap; "mean" to use ecp.
        x_from_run_name: if True, append ncosmo size from run name to labels.
        ax: optional matplotlib axis.
        title: optional figure title.
        figsave: optional path to save figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    if metric_key.endswith(".calibration_error"):
        metric_key = metric_key[: -len(".calibration_error")]

    ce_key = _resolve_calibration_error_key(metric_key)

    ecp_key, use_bootstrap = _resolve_tarp_curve_keys(curve_type)
    alpha_key = f"{metric_key}.credible_intervals"
    ecp_path = f"{metric_key}.{ecp_key}"

    plotted = 0
    for run_name, run_curves in tarp_curves.items():
        if run_filter and run_filter not in run_name:
            continue
        if alpha_key not in run_curves or ecp_path not in run_curves:
            continue

        alpha = np.asarray(run_curves[alpha_key], dtype=float)
        mean_ecp, std_ecp = _aggregate_curve(run_curves[ecp_path], use_bootstrap=use_bootstrap)
        try:
            label = f"Repeat {run_name.split('_')[2]}"  # default label is first token of run name
        except:
            label = run_name
        if x_from_run_name:
            ncosmo = _derive_x_from_run_name(run_name)
            if ncosmo is not None:
                label = f"{run_name} (N={ncosmo})"

        if run_metrics is not None and run_name in run_metrics:
            ce_value = run_metrics[run_name].get(ce_key)
            if ce_value is not None:
                label = f"{label} (CE={float(ce_value):.4g})"

        ax.plot(alpha, mean_ecp, linewidth=2, label=label)
        for k in sigma_levels:
            ax.fill_between(alpha, mean_ecp - k * std_ecp, mean_ecp + k * std_ecp, alpha=0.2)
        plotted += 1

    ax.plot([0, 1], [0, 1], ls="--", color="k", label="Ideal case")
    ax.set_xlabel("Credibility Level")
    ax.set_ylabel("Expected Coverage")
    if title is None:
        title = _resolve_plot_title(metric_key, plotting_name_conversions)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)

    if plotted > 0:
        ax.legend()

    if figsave:
        fig.savefig(figsave, dpi=300, transparent=True)

    return ax


def plot_tarp_calibration_suite(
    tarp_curves,
    metric_keys,
    run_filter="",
    sigma_levels=(1,),
    curve_type="bootstrap",
    ncols=2,
    figsize=(10, 8),
    run_metrics=None,
    plotting_name_conversions=None,
):
    """Plot a list of TARP calibration metric keys in a grid."""
    metric_keys = list(metric_keys)
    if len(metric_keys) == 0:
        raise ValueError("metric_keys must contain at least one key")

    nrows = int(np.ceil(len(metric_keys) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for i, metric_key in enumerate(metric_keys):
        plot_tarp_calibration_curves(
            tarp_curves=tarp_curves,
            metric_key=metric_key,
            run_filter=run_filter,
            sigma_levels=sigma_levels,
            curve_type=curve_type,
            run_metrics=run_metrics,
            ax=axes_flat[i],
            title=_resolve_plot_title(metric_key, plotting_name_conversions),
            plotting_name_conversions=plotting_name_conversions,
        )

    for j in range(len(metric_keys), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    return fig, axes