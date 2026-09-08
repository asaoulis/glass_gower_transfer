#!/usr/bin/env python
"""Tier-2: build the OOD-score -> bias-probability tables (from the local misspec/summaries npz) and,
optionally, read an observation's scalar scores against them.

    PYTHONPATH=. python scripts/tier2_tables.py --base-dir ml-checkpoints/gower_npe_finetune_nla_m_bgp_z8_ens1 \
        --phase6-dir .claude/runs/eval-and-viz/production-model-misspec/artifacts/phase6 --out-dir <dir> \
        [--rows-cache rows.npz] [--obs-score obs_score_A.json ...]

Outputs: BIAS_TABLES.md, bias_tables.json, bias_curves.png (P(|z|>t | more-OOD-than s) vs s, with the
in-distribution baseline), per_variate.json, and -- with --obs-score -- obs_reading_<label>.json/.md
with the table row each observed detector value falls in. The observation's scalars are NOT
unblinding information (a detector score carries no location).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402


def _curves_figure(rows, tables, out_png, params=("omega_m", "sigma_8", "S8"), thresholds=(0.3, 0.5, 1.0)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dets = list(tables)
    fig, axes = plt.subplots(len(params), len(dets), figsize=(4.6 * len(dets), 3.2 * len(params)), squeeze=False)
    in_dist = rows["variate"] == "gower_bgp_nla_m"
    for j, det in enumerate(dets):
        t = tables[det]
        for i, p in enumerate(params):
            ax = axes[i, j]
            ex = t["exceedance"][p]["exceedance"]
            cuts = [r["cut"] for r in ex]
            for th in thresholds:
                ax.plot(cuts, [r[f"P_absz_gt_{th:g}"] for r in ex], marker=".", label=f"P(|z|>{th:g})")
                base = float(np.mean(np.abs(rows[f"z_{p}"][in_dist & np.isfinite(rows[f"z_{p}"])]) > th))
                ax.axhline(base, ls=":", color="0.5", lw=0.8)
            ax.set_title(f"{det}: {p}", fontsize=10)
            ax.set_xlabel(("score cut s (rows with score <= s)" if t["lower_is_ood"] else "score cut s (rows with score >= s)"), fontsize=8)
            if det in ("meanp", "knn_p"):
                ax.set_xscale("log")
            ax.set_ylim(0, 1)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Tier-2: P(|z| > t | detector more OOD than s); dotted = in-distribution baseline", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", default="ml-checkpoints/gower_npe_finetune_nla_m_bgp_z8_ens1")
    ap.add_argument("--phase6-dir", default=".claude/runs/eval-and-viz/production-model-misspec/artifacts/phase6")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rows-cache", default=None, help="npz of the joined rows (built if missing)")
    ap.add_argument("--obs-score", nargs="*", default=[], help="obs_score_<label>.json files from eval --mode obs-score")
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args(argv)

    from src.observation.checks.tier2 import bias_table, format_markdown, load_rows, lookup
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = None
    if args.rows_cache and os.path.exists(args.rows_cache):
        with np.load(args.rows_cache, allow_pickle=False) as d:
            rows = {("file" if k == "files" else k): d[k] for k in d.files}
    if rows is None:
        rows = load_rows(args.base_dir, phase6_dir=args.phase6_dir)
        if args.rows_cache:
            np.savez(args.rows_cache, **{("files" if k == "file" else k): v for k, v in rows.items()})
    tables = {"meanp": bias_table(rows, "meanp", n_boot=args.n_boot),
              "knn_p": bias_table(rows, "knn_p", n_boot=args.n_boot),
              "kl": bias_table(rows, "kl", lower_is_ood=False, n_boot=args.n_boot)}
    with open(out / "bias_tables.json", "w") as fh:
        json.dump(tables, fh, default=float)
    md = ["# Tier-2 bias tables (pooled over variates, equal weight per row)\n",
          "Rows = (variate, encoder repeat, event) of `gower_npe_finetune_nla_m_bgp_z8_ens1` x 7 Gower variates; "
          "z = (truth - posterior mean)/posterior std; S8 from the full sample dumps; CI68 = cosmology-block bootstrap; "
          "'excess' = P(|z|>t) minus the in-distribution value in the SAME bin. A calibrated posterior has "
          "P(|z|>0.3/0.5/1) = 0.76/0.62/0.32, so only the excess is evidence of bias.\n"]
    for k, t in tables.items():
        md += [format_markdown(t), ""]
    per_var = {}
    md.append("## Per-variate composition (all rows)\n")
    md.append("| variate | n | E[z_om] | E[z_s8] | E[z_S8] | P(|z_S8|>0.5) | P(|z_S8|>1) | median knn_p | median meanp | median kl |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for v in np.unique(rows["variate"]):
        m = rows["variate"] == v
        d = {"n": int(m.sum()), "mean_z_omega_m": float(np.nanmean(rows["z_omega_m"][m])),
             "mean_z_sigma_8": float(np.nanmean(rows["z_sigma_8"][m])), "mean_z_S8": float(np.nanmean(rows["z_S8"][m])),
             "P_absz_S8_gt_0.5": float(np.nanmean(np.abs(rows["z_S8"][m]) > 0.5)),
             "P_absz_S8_gt_1": float(np.nanmean(np.abs(rows["z_S8"][m]) > 1)),
             "median_knn_p": float(np.nanmedian(rows["knn_p"][m])), "median_meanp": float(np.nanmedian(rows["meanp"][m])),
             "median_kl": float(np.nanmedian(rows["kl"][m]))}
        per_var[str(v)] = d
        md.append(f"| {v} | {d['n']} | {d['mean_z_omega_m']:+.2f} | {d['mean_z_sigma_8']:+.2f} | {d['mean_z_S8']:+.2f} | "
                  f"{d['P_absz_S8_gt_0.5']:.2f} | {d['P_absz_S8_gt_1']:.2f} | {d['median_knn_p']:.3f} | {d['median_meanp']:.3f} | {d['median_kl']:.3f} |")
    md.append("\n**Floor (stated, not hidden):** `gower_vd`, `gower_gb1p0`, `gower_gb1p3` sit at chance for both detectors "
              "(AUROC ~0.5 in the production matrix); the bias they DO produce (rows above) is the 'invisible bias' budget "
              "that no score can bound.")
    with open(out / "per_variate.json", "w") as fh:
        json.dump(per_var, fh, indent=2)
    _curves_figure(rows, tables, str(out / "bias_curves.png"))

    # observation readings
    for f in args.obs_score:
        with open(f) as fh:
            sc = json.load(fh)
        label = sc.get("label", os.path.basename(f))
        reading = {"label": label, "scores": {k: sc.get(k) for k in ("meanp_raw", "meanp_recalibrated", "kl")}}
        lines = [f"## Observation {label}: Tier-2 reading\n"]
        for det, key in (("meanp", "meanp_recalibrated"), ("kl", "kl")):
            val = sc.get(key)
            if val is None:
                continue
            row = lookup(tables[det], float(val))
            reading[det] = {"value": val, "bin": [row["lo"], row["hi"]], "per_param": row["per_param"], "n_rows": row["n_rows"]}
            lines.append(f"- **{det}** = {val:.4g} -> bin [{row['lo']:.3g}, {row['hi']:.3g}) (n={row['n_rows']} rows, "
                         f"{row['n_cosmologies']} cosmologies):")
            for p in ("omega_m", "sigma_8", "S8"):
                e = row["per_param"].get(p, {})
                if "mean_z" in e:
                    lines.append(f"    - {p}: E[z] = {e['mean_z']:+.2f} (CI68 {e['mean_z_ci68'][0]:+.2f}..{e['mean_z_ci68'][1]:+.2f}); "
                                 + "; ".join(f"P(|z|>{t:g}) = {e[f'P_absz_gt_{t:g}']['pooled']:.2f} [excess {e[f'P_absz_gt_{t:g}']['excess']:+.2f}]"
                                             for t in (0.3, 0.5, 1.0)))
        per_enc = sc.get("per_encoder", {})
        if per_enc:
            lines.append("- per-encoder kNN p: " + ", ".join(f"{k}: {v.get('ood_knn_p', float('nan')):.3f}" for k, v in per_enc.items()))
        with open(out / f"obs_reading_{label}.json", "w") as fh:
            json.dump(reading, fh, indent=2, default=float)
        with open(out / f"obs_reading_{label}.md", "w") as fh:
            fh.write("\n".join(lines) + "\n")
        md += [""] + lines
    with open(out / "BIAS_TABLES.md", "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"wrote {out}/BIAS_TABLES.md, bias_tables.json, bias_curves.png, per_variate.json"
          + (f" and {len(args.obs_score)} observation readings" if args.obs_score else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
