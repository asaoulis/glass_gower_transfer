"""Analyse sweep JSONL -> paired b_g shifts, M2 linear/quadratic fits, leaderboard tables.

Model per statistic x and triplet t:  s_t(b) = x_t(b)/x_t(1) - 1, fitted with
    s(b) = alpha*(b-1) + beta*(b^2-1)
Exactly determined by the (0.5, 1.0, 1.5) triplet:
    s(0.5) = -0.5a - 0.75b ;  s(1.5) = +0.5a + 1.25b
    => beta = (s(1.5)+s(0.5))/0.5 ,  alpha = (s(1.5)-s(0.5)) - 2*beta
H1 (counts-noise convexity) predicts alpha ~ 0, beta > 0 for variance-like stats; first-order
signal leakage predicts alpha != 0. Usage:
    python scripts/replay_shear_maps.py  (sweep) -> sweep.jsonl
    python -m scripts.shear_replay.analyse sweep.jsonl --out report.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = ("0.5", "1", "1.5")


def load_rows(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _numeric_keys(rows):
    keys = set()
    for r in rows[:50]:
        for k, v in r["arms"]["1"].items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                keys.add(k)
    return sorted(keys)


def paired_shifts(rows):
    """{candidate: {stat: {'s05': [...], 's15': [...], 'alpha': [...], 'beta': [...]}}}"""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        cand = r["candidate"]
        arms = r["arms"]
        if not all(a in arms for a in ARMS):
            continue
        for k, v1 in arms["1"].items():
            if not isinstance(v1, (int, float)) or not np.isfinite(v1) or v1 == 0:
                continue
            v05, v15 = arms["0.5"].get(k), arms["1.5"].get(k)
            if v05 is None or v15 is None:
                continue
            s05, s15 = v05 / v1 - 1.0, v15 / v1 - 1.0
            beta = (s15 + s05) / 0.5
            alpha = (s15 - s05) - 2.0 * beta
            d = out[cand][k]
            d["s05"].append(s05)
            d["s15"].append(s15)
            d["alpha"].append(alpha)
            d["beta"].append(beta)
    return out


def within_scatter(rows, arm="1"):
    """Within-cosmology fractional scatter of every stat, from the reference arm's 4 cat_idx."""
    by_sim = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["candidate"] != rows[0]["candidate"]:
            continue
        for k, v in r["arms"][arm].items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                by_sim[k][(r.get("tag", ""), r["sim_id"])][r["cat_idx"]] = v
    scat = {}
    for k, sims in by_sim.items():
        fr = []
        for sim, vals in sims.items():
            v = np.array(list(vals.values()), dtype=float)
            if v.size >= 2 and np.abs(v.mean()) > 0:
                fr.append(v.std(ddof=1) / np.abs(v.mean()))
        if fr:
            scat[k] = float(np.mean(fr))
    return scat


def summarise(shifts, scatter, stats_filter=None):
    """Per candidate x stat: mean shift +- sem, alpha/beta, and shift in within-scatter units."""
    tab = []
    for cand, stats in shifts.items():
        for k, d in stats.items():
            if stats_filter and not any(f in k for f in stats_filter):
                continue
            s05 = np.array(d["s05"], dtype=float)
            s15 = np.array(d["s15"], dtype=float)
            al = np.array(d["alpha"], dtype=float)
            be = np.array(d["beta"], dtype=float)
            n = s05.size
            if n == 0:
                continue
            sc = scatter.get(k, np.nan)
            row = {
                "candidate": cand, "stat": k, "n": n,
                "s05_mean": s05.mean(), "s05_sem": s05.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
                "s15_mean": s15.mean(), "s15_sem": s15.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
                "alpha": al.mean(), "alpha_sem": al.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
                "beta": be.mean(), "beta_sem": be.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
                "within_scatter": sc,
                "s15_over_scatter": s15.mean() / sc if sc and np.isfinite(sc) and sc > 0 else np.nan,
            }
            tab.append(row)
    return tab


DECISION_STATS = ("E_north_std_all", "E_south_std_all", "D2_Bstd_b0", "D2_Bstd_b5",
                  "D4_hilo_ratio_b0", "D6b_randstd_b0", "D7_slope_b0", "D10_bp_mean",
                  "cnt_varN_rel_b0", "cnt_noiseterm_b0")


def to_markdown(tab, title, order_key="s15_over_scatter"):
    lines = [f"## {title}", "",
             "| candidate | stat | n | s(0.5) | s(1.5) | alpha | beta | within | s15/within |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(tab, key=lambda r: (r["stat"], abs(r.get(order_key) or 0))):
        lines.append(
            f"| {r['candidate']} | {r['stat']} | {r['n']} "
            f"| {r['s05_mean']:+.4%}±{r['s05_sem']:.4%} | {r['s15_mean']:+.4%}±{r['s15_sem']:.4%} "
            f"| {r['alpha']:+.4f}±{r['alpha_sem']:.4f} | {r['beta']:+.4f}±{r['beta_sem']:.4f} "
            f"| {r['within_scatter']:.4%} | {r['s15_over_scatter']:+.2f} |")
    return "\n".join(lines)


def paired_probe_summary(rows):
    """Aggregate the triplet-level M1/M4/D8 probes per candidate."""
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k, v in r.get("paired", {}).items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                agg[r["candidate"]][k].append(v)
    out = {}
    for cand, d in agg.items():
        out[cand] = {k: {"mean": float(np.mean(v)),
                         "sem": float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None,
                         "n": len(v)}
                     for k, v in d.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stats", default=None,
                    help="comma-separated substrings; default = the decision set")
    ap.add_argument("--all-stats", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.jsonl)
    print(f"[analyse] {len(rows)} rows, "
          f"{len(set(r['candidate'] for r in rows))} candidates, "
          f"{len(set((r.get('tag', ''), r['sim_id'], r['cat_idx']) for r in rows))} triplets")
    shifts = paired_shifts(rows)
    scatter = within_scatter(rows)
    filt = None if args.all_stats else \
        (tuple(args.stats.split(",")) if args.stats else DECISION_STATS)
    tab = summarise(shifts, scatter, stats_filter=filt)
    md = to_markdown(tab, "Paired b_g shifts (per candidate x stat)")
    probes = paired_probe_summary(rows)
    md += "\n\n## Triplet probes (M1 closure, M4 xcorr, D8)\n```json\n"
    md += json.dumps(probes, indent=1, default=float)[:20000]
    md += "\n```\n"
    if args.out:
        Path(args.out).write_text(md)
        print(f"[analyse] wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
