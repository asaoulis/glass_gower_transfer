"""Quantitative diagnostics of the empirical/theory bandpower ratios.

The core acceptance test: per (spectrum, bandpower) the median ratio must agree with
unity to within ``ok_frac`` (default 5%); ``warn_frac`` (default 10%) is a WARNING;
beyond that is an ERROR (``!!!``).  We also surface a few trend diagnostics that make a
systematic problem obvious (ℓ-dependence, auto-vs-cross offset, per-tomo-bin offset),
and emit both a machine-readable ``report.json`` and a human ``report.md``.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Dict, List

import numpy as np

from . import config as cfg


def summarise_ratios(ensemble: dict) -> dict:
    """Per-(spectrum, band) summary statistics over the file ensemble.

    Returns arrays of shape ``(n_spectra, nbands)`` plus the bandpower ℓ centres.
    ``abs_dev`` = ``|median - 1|`` (the quantity the thresholds act on); ``scatter``
    = the mean over files of ``|ratio - 1|`` (realisation spread, not a bias).
    """
    ratios = ensemble["ratios"]  # (n_files, n_spectra, nbands)
    q10, q16, q50, q84, q90 = np.percentile(ratios, [10, 16, 50, 84, 90], axis=0)
    return {
        "labels": list(ensemble["labels"]),
        "cls": np.asarray(ensemble["cls"]),
        "median": q50,
        "q16": q16,
        "q84": q84,
        "q10": q10,
        "q90": q90,
        "abs_dev": np.abs(q50 - 1.0),
        "scatter": np.mean(np.abs(ratios - 1.0), axis=0),
        "n_files": int(ratios.shape[0]),
        "n_spectra": int(ratios.shape[1]),
        "nbands": int(ratios.shape[2]),
    }


def classify(stats: dict, ok_frac: float = cfg.OK_FRAC, warn_frac: float = cfg.WARN_FRAC):
    """Classify each (spectrum, band) by ``|median - 1|`` into OK / WARNING / ERROR."""
    dev = stats["abs_dev"]
    status = np.full(dev.shape, cfg.STATUS_OK, dtype=object)
    status[dev > ok_frac] = cfg.STATUS_WARNING
    status[dev > warn_frac] = cfg.STATUS_ERROR
    counts = {
        cfg.STATUS_OK: int(np.sum(status == cfg.STATUS_OK)),
        cfg.STATUS_WARNING: int(np.sum(status == cfg.STATUS_WARNING)),
        cfg.STATUS_ERROR: int(np.sum(status == cfg.STATUS_ERROR)),
    }
    return status, counts


def overall_status(counts: Dict[str, int]) -> int:
    """Map the worst classification present to an exit code."""
    if counts.get(cfg.STATUS_ERROR, 0) > 0:
        return cfg.EXIT_ERRORS
    if counts.get(cfg.STATUS_WARNING, 0) > 0:
        return cfg.EXIT_WARNINGS
    return cfg.EXIT_PASS


def detect_trends(stats: dict, nbins: int = cfg.NBINS) -> dict:
    """Lightweight trend diagnostics that make a systematic discrepancy obvious."""
    median = stats["median"]                # (n_spectra, nbands)
    labels = stats["labels"]
    cls = stats["cls"]
    logl = np.log(np.asarray(cls, dtype=float))

    # (a) ℓ-dependence: slope of (median ratio) vs log-ℓ per spectrum; positive slope
    #     => high-ℓ pull (the expected non-linear / pixel-window signature).
    slopes = np.array([np.polyfit(logl, median[s], 1)[0] for s in range(len(labels))])
    lowest_band = median[:, 0]
    highest_band = median[:, -1]
    ell_trend = {
        "per_spectrum_slope": {labels[s]: float(slopes[s]) for s in range(len(labels))},
        "mean_slope": float(np.mean(slopes)),
        "mean_low_ell_ratio": float(np.mean(lowest_band)),
        "mean_high_ell_ratio": float(np.mean(highest_band)),
        "note": "slope = d(median ratio)/d(ln l); +ve => high-l pull",
    }

    # (b) auto (i==j) vs cross (i>j) spectra mean |median-1|.
    auto_idx, cross_idx = [], []
    for s, lbl in enumerate(labels):
        a, b = lbl.replace("S", "").split("-")
        (auto_idx if a == b else cross_idx).append(s)
    abs_dev = stats["abs_dev"]
    auto_cross = {
        "auto_mean_abs_dev": float(np.mean(abs_dev[auto_idx])) if auto_idx else None,
        "cross_mean_abs_dev": float(np.mean(abs_dev[cross_idx])) if cross_idx else None,
    }

    # (c) per-tomographic-bin systematic offset: mean (median-1) over every spectrum &
    #     band that involves each bin (a consistent per-bin sign flags a per-bin issue).
    per_bin = {}
    for bin_id in range(1, nbins + 1):
        sel = [s for s, lbl in enumerate(labels)
               if str(bin_id) in lbl.replace("S", "").split("-")]
        if sel:
            per_bin[f"S{bin_id}"] = {
                "mean_signed_dev": float(np.mean(median[sel] - 1.0)),
                "mean_abs_dev": float(np.mean(np.abs(median[sel] - 1.0))),
            }

    return {"ell_dependence": ell_trend, "auto_vs_cross": auto_cross, "per_tomo_bin": per_bin}


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _flag(status: str) -> str:
    return {cfg.STATUS_OK: "ok", cfg.STATUS_WARNING: "WARN",
            cfg.STATUS_ERROR: "!!! ERROR"}[status]


def write_report(stats: dict, status, counts: dict, trends: dict, out_dir: str,
                 meta: dict) -> dict:
    """Write ``report.json`` (machine) and ``report.md`` (human) into ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    labels, cls = stats["labels"], list(map(float, stats["cls"]))
    nbands = stats["nbands"]
    exit_code = overall_status(counts)
    verdict = ("PASS" if exit_code == cfg.EXIT_PASS
               else "WARNINGS" if exit_code == cfg.EXIT_WARNINGS else "ERRORS")

    # ---- JSON (every bin) ----
    bins = []
    for s, lbl in enumerate(labels):
        for b in range(nbands):
            bins.append({
                "spectrum": lbl, "band": b, "ell": cls[b],
                "median": float(stats["median"][s, b]),
                "q16": float(stats["q16"][s, b]), "q84": float(stats["q84"][s, b]),
                "q10": float(stats["q10"][s, b]), "q90": float(stats["q90"][s, b]),
                "abs_dev": float(stats["abs_dev"][s, b]),
                "scatter": float(stats["scatter"][s, b]),
                "status": str(status[s, b]),
            })
    report = {
        "verdict": verdict, "exit_code": exit_code, "counts": counts,
        "meta": {**meta, "git_rev": _git_rev(), "n_files": stats["n_files"]},
        "trends": trends, "bins": bins,
    }
    json_path = os.path.join(out_dir, "report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # ---- Markdown (human) ----
    worst = np.unravel_index(int(np.argmax(stats["abs_dev"])), stats["abs_dev"].shape)
    lines = [
        f"# Theory-vs-empirical bandpower validation — **{verdict}**",
        "",
        f"- data: `{meta.get('data_dir')}`  (sim-type: {meta.get('sim_type')})",
        f"- files loaded: {stats['n_files']}  (failed: {meta.get('n_failed', 0)})",
        f"- mixing matrix: `{meta.get('mixing_matrix')}`  |  nonlinear: "
        f"`{meta.get('nonlinear') or 'OFF'}`",
        f"- thresholds: OK <= {meta.get('ok_frac')*100:.0f}% | "
        f"WARNING <= {meta.get('warn_frac')*100:.0f}% | ERROR beyond",
        f"- git rev: `{_git_rev()}`",
        "",
    ]
    if meta.get("caveat"):
        lines += [f"> ⚠️ **CAVEAT:** {meta['caveat']}", ""]
    lines += [
        f"## Summary ({stats['n_spectra']}x{nbands} = "
        f"{stats['n_spectra']*nbands} bins)",
        f"- OK: {counts[cfg.STATUS_OK]}  |  "
        f"WARNING: {counts[cfg.STATUS_WARNING]}  |  "
        f"**ERROR: {counts[cfg.STATUS_ERROR]}**",
        f"- worst bin: {labels[worst[0]]} band {worst[1]} "
        f"(median={stats['median'][worst]:.4f}, "
        f"|dev|={stats['abs_dev'][worst]*100:.1f}%)",
        "",
    ]

    # trends
    ed = trends["ell_dependence"]
    ac = trends["auto_vs_cross"]
    lines += [
        "## Trends",
        f"- ℓ-dependence: mean slope d(ratio)/d(ln ℓ) = {ed['mean_slope']:+.4f} "
        f"(low-ℓ mean {ed['mean_low_ell_ratio']:.3f} -> high-ℓ mean "
        f"{ed['mean_high_ell_ratio']:.3f})",
        f"- auto vs cross mean |dev|: auto={ac['auto_mean_abs_dev']}, "
        f"cross={ac['cross_mean_abs_dev']}",
        "- per-tomo-bin mean signed dev: "
        + ", ".join(f"{k} {v['mean_signed_dev']:+.3f}"
                    for k, v in trends["per_tomo_bin"].items()),
        "",
    ]

    # offending bins
    lines.append("## WARNING / ERROR bins")
    flagged = [(s, b) for s in range(len(labels)) for b in range(nbands)
               if status[s, b] != cfg.STATUS_OK]
    if not flagged:
        lines.append("None — every bin within the OK threshold. ✅")
    else:
        # ERRORs first (worst dev first), then WARNINGs.
        flagged.sort(key=lambda sb: (status[sb] != cfg.STATUS_ERROR,
                                     -stats["abs_dev"][sb]))
        lines.append("")
        lines.append("| spectrum | band | ℓ | median | 16-84% | |dev| | status |")
        lines.append("|---|---|---|---|---|---|---|")
        for s, b in flagged:
            lines.append(
                f"| {labels[s]} | {b} | {cls[b]:.0f} | "
                f"{stats['median'][s, b]:.4f} | "
                f"[{stats['q16'][s, b]:.3f}, {stats['q84'][s, b]:.3f}] | "
                f"{stats['abs_dev'][s, b]*100:.1f}% | {_flag(status[s, b])} |"
            )
    lines.append("")

    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    return {"json": json_path, "md": md_path, "exit_code": exit_code, "verdict": verdict}


def run_diagnostics(ensemble: dict, out_dir: str, meta: dict,
                    ok_frac: float = cfg.OK_FRAC, warn_frac: float = cfg.WARN_FRAC,
                    nbins: int = cfg.NBINS) -> dict:
    """Full diagnostic pass: summarise -> classify -> trends -> write report."""
    stats = summarise_ratios(ensemble)
    status, counts = classify(stats, ok_frac, warn_frac)
    trends = detect_trends(stats, nbins=nbins)
    meta = {**meta, "ok_frac": ok_frac, "warn_frac": warn_frac}
    out = write_report(stats, status, counts, trends, out_dir, meta)
    out["counts"] = counts
    out["stats"] = stats
    return out
