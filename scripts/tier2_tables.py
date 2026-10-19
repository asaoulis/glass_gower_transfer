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


def _curves_figure(rows, tables, out_png, params=("omega_m", "sigma_8", "S8"), thresholds=(0.3, 0.5, 1.0),
                   in_dist_name="gower_bgp_nla_m"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dets = list(tables)
    fig, axes = plt.subplots(len(params), len(dets), figsize=(4.6 * len(dets), 3.2 * len(params)), squeeze=False)
    in_dist = rows["variate"] == in_dist_name
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
    ap.add_argument("--match-template", default="ncosmo300_{r}",
                    help="per-repeat filename tag; ncosmo300_{r} for BOTH the field-level pack and "
                         "the 2-pt M17 pack (confirmed from the misspec job log)")
    ap.add_argument("--kl-params", default=None,
                    help="which ensemble-disagreement statistic fills the kl column: full (default, "
                         "the production detector) or a variant from src.ml.eval.kl_subsets "
                         "(om_s8, S8, om_s8_w0, om_s8_fullcov). See artifacts/KL_SUBSET_STUDY.md. "
                         "The choice is STAMPED into bias_tables.json and the markdown header: a "
                         "table must never be read against an observation scored differently.")
    # Pack / variate plumbing (2026-10-19, vdq-unblinding-rerun P1.2). Defaults = the previous
    # hardcoded behaviour (old no-VD BGP field pack, its 7 variates, repeats 0-4), so a call
    # without these flags is byte-identical to before.
    ap.add_argument("--experiment", default="gower_npe_finetune_nla_m_bgp_z8_ens1",
                    help="pack name used for the scaler box (S8 from samples, KL subsets)")
    ap.add_argument("--variates", nargs="+", default=None,
                    help="variate dirs under misspec/ and summaries/ (default: tier2.VARIATES_DEFAULT)")
    ap.add_argument("--in-dist", default=None, help="in-distribution variate (default: tier2.IN_DIST)")
    ap.add_argument("--repeats", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                    help="encoder repeats of the pack (field VDQ pack: 0 2 3 4 5)")
    ap.add_argument("--meanp-source", choices=("phase6", "obs_knn"), default="phase6",
                    help="phase6 = the multiencoder_ood combined_p (cond_knn mean-p, needs true theta; "
                         "mock-side only); obs_knn = the OBS-IDENTICAL encoder-mean of the unconditional "
                         "kNN p recalibrated against the _idtest null (artifacts/vdq_misspec/meanp_knn.py), "
                         "the statistic obs-score's meanp_recalibrated actually is")
    ap.add_argument("--meanp-npz", default=None, help="meanp_knn.py output (required for --meanp-source obs_knn)")
    args = ap.parse_args(argv)

    from src.observation.checks.tier2 import (IN_DIST, VARIATES_DEFAULT, adopted_kl_for, bias_table,
                                              format_markdown, load_rows, lookup)
    variates = list(args.variates) if args.variates else list(VARIATES_DEFAULT)
    in_dist = args.in_dist or IN_DIST
    if in_dist not in variates:
        raise SystemExit(f"--in-dist {in_dist} is not in --variates {variates}")
    if args.meanp_source == "obs_knn" and not args.meanp_npz:
        raise SystemExit("--meanp-source obs_knn needs --meanp-npz (artifacts/vdq_misspec/meanp_knn.py output)")
    if args.kl_params is None:
        args.kl_params = adopted_kl_for(args.base_dir)
        print("[tier2-tables] kl_params not given -> adopted statistic for this pack: %s"
              % args.kl_params, flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = None
    if args.rows_cache and os.path.exists(args.rows_cache):
        with np.load(args.rows_cache, allow_pickle=False) as d:
            rows = {("file" if k == "files" else k): d[k] for k in d.files}
        # a cache may hold MORE variates than requested (one cache, per-variate + pooled tables):
        # keep exactly --variates, and refuse a request the cache cannot satisfy
        have = set(str(v) for v in np.unique(rows["variate"]))
        if not set(variates) <= have:
            raise SystemExit(f"rows cache {args.rows_cache} lacks {sorted(set(variates) - have)} -- rebuild it")
        keep = np.isin(rows["variate"], variates)
        if not keep.all():
            rows = {k: v[keep] for k, v in rows.items()}
            print(f"[tier2-tables] rows cache filtered to {variates} ({int(keep.sum())} rows)", flush=True)
    if rows is None:
        rows = load_rows(args.base_dir, experiment=args.experiment, variates=variates, repeats=list(args.repeats),
                         phase6_dir=(args.phase6_dir if args.meanp_source == "phase6" else None),
                         match_template=args.match_template, kl_params=args.kl_params)
        if args.rows_cache:
            np.savez(args.rows_cache, **{("files" if k == "file" else k): v for k, v in rows.items()})
    if args.meanp_source == "obs_knn":
        # event-level join on (variate, test-file basename): the same value on every repeat row of
        # an event, exactly like the phase6 column it replaces
        with np.load(args.meanp_npz, allow_pickle=False) as d:
            key = {(str(v), os.path.basename(str(f))): float(x)
                   for v, f, x in zip(d["variate"], d["test_files"], d["meanp_knn_recal"])}
        rows["meanp"] = np.array([key.get((str(v), str(f)), np.nan) for v, f in zip(rows["variate"], rows["file"])])
        miss = int(np.sum(~np.isfinite(rows["meanp"])))
        print(f"[tier2-tables] meanp <- obs-identical meanp_knn_recal from {args.meanp_npz} "
              f"({miss}/{len(rows['meanp'])} rows unmatched)", flush=True)
    tables = {}
    for det, kw in (("meanp", {}), ("knn_p", {}), ("kl", {"lower_is_ood": False})):
        if not np.any(np.isfinite(rows[det])):
            # a KL-axis-only pack (no summary extraction / no phase6) -> that detector has no data
            print("[tier2-tables] %s is entirely NaN -- skipping its table" % det, flush=True)
            continue
        tables[det] = bias_table(rows, det, n_boot=args.n_boot, in_dist=in_dist, **kw)
    provenance = {"base_dir": args.base_dir, "match_template": args.match_template,
                  "kl_params": args.kl_params, "phase6_dir": args.phase6_dir,
                  "n_boot": args.n_boot, "detectors": sorted(tables),
                  "experiment": args.experiment, "variates": variates, "in_dist": in_dist,
                  "repeats": list(args.repeats), "meanp_source": args.meanp_source,
                  "meanp_npz": args.meanp_npz,
                  "variates_in_rows": sorted(str(v) for v in np.unique(rows["variate"]))}
    with open(out / "bias_tables.json", "w") as fh:
        json.dump({"_provenance": provenance, **tables}, fh, default=float)
    md = ["# Tier-2 bias tables (pooled over variates, equal weight per row)\n",
          "**Detector provenance — `kl_params = %s`, `match_template = %s`, pack `%s`.** An "
          "observation reading is only valid against a table built with the SAME `kl_params`.\n"
          % (args.kl_params, args.match_template, os.path.basename(args.base_dir.rstrip("/"))),
          # The pack name here was hardcoded to the FIELD pack whatever --base-dir said, so a
          # band-pack table misattributed its own rows -- and this is the line a reader trusts.
          "Rows = (variate, encoder repeat, event) of `%s` x %d Gower variates (%s; in-dist `%s`; "
          "meanp source `%s`); "
          % (os.path.basename(args.base_dir.rstrip("/")), len(np.unique(rows["variate"])),
             ", ".join(f"`{v}`" for v in np.unique(rows["variate"])), in_dist, args.meanp_source) +
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
    _curves_figure(rows, tables, str(out / "bias_curves.png"), in_dist_name=in_dist)

    # observation readings
    for f in args.obs_score:
        with open(f) as fh:
            sc = json.load(fh)
        label = sc.get("label", os.path.basename(f))
        # The KL column of the TABLE is built with `kl_params`; the observation must be looked up
        # with the SAME statistic or it lands in the wrong bin. obs_score writes every variant,
        # so select `kl_<kl_params>` (== `kl_adopted` when obs-score chose the same one) rather
        # than the bare `kl`, which is always the FULL DIAGONAL value. Fixed 2026-09-18: the
        # previous code used sc["kl"] unconditionally -- e.g. 0.194 (full) instead of 0.115
        # (om_s8_fullcov) for the same observation, overstating the predicted bias.
        kl_key = f"kl_{args.kl_params}"
        if kl_key not in sc and sc.get("kl_params") == args.kl_params and "kl_adopted" in sc:
            kl_key = "kl_adopted"
        if kl_key not in sc:
            print(f"[tier2-tables] WARNING: {label}: no '{kl_key}' in the obs-score json; the kl "
                  f"reading is SKIPPED rather than read against a mismatched statistic "
                  f"(available: {[k for k in sc if k.startswith('kl')]})", flush=True)
        reading = {"label": label, "kl_params": args.kl_params,
                   "scores": {k: sc.get(k) for k in ("meanp_raw", "meanp_recalibrated", kl_key)}}
        lines = [f"## Observation {label}: Tier-2 reading\n",
                 f"(detector provenance: `kl_params = {args.kl_params}`, read from `{kl_key}`)\n"]
        for det, key in (("meanp", "meanp_recalibrated"), ("kl", kl_key)):
            val = sc.get(key)
            if val is None:
                continue
            if det not in tables:
                # the pack has no table for this detector (e.g. a KL-only pack without summaries):
                # SAY so in the reading rather than crash or silently drop it
                lines.append(f"- **{det}** = {val:.4g} -> **no {det} table for this pack** (detector not tabulated)")
                reading[det] = {"value": val, "bin": None, "note": "no table for this pack"}
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
