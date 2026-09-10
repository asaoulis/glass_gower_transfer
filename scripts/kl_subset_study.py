#!/usr/bin/env python
"""Does restricting the Tier-2 ensemble-disagreement KL to a PARAMETER SUBSET improve its OOD
discriminating power?

    PYTHONPATH=. python scripts/kl_subset_study.py \
        --base-dir ml-checkpoints/gower_npe_finetune_nla_m_bgp_z8_ens1 \
        --experiment gower_npe_finetune_nla_m_bgp_z8_ens1 \
        --out-dir .claude/runs/eval-and-viz/unblinding-prep/artifacts/kl_subsets

Points at ANY misspec pack directory, so the same code answers the question for the field-level
encoder pack and for the 2-pt (M17 band) pack -- only ``--base-dir``/``--match-template`` change.
Nothing here re-evaluates a model: the KL is a sum over the parameter axis, so every subset is an
exact slice of the saved ``mu``/``var`` (see ``src/ml/eval/kl_subsets``). ``S8`` is the exception
and is built from the per-repeat sample dumps.

Outputs
    kl_subset_rows.npz     per-variate per-event scores for every subset + the per-dim matrix
    kl_subset_auroc.json   AUROC (cosmology-block CI) for the per-dim and subset tables
    KL_SUBSETS_<tag>.md    the two tables plus the controls

Discriminating power is AUROC against the pack's OWN in-distribution variate. Never compare a raw
KL value between packs: an ens9 pack's mixture posteriors are broader, so its KL scale differs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from src.ml.eval.kl_subsets import (  # noqa: E402
    ALL_SCORES, FULLCOV_SUBSETS, PARAM_BLOCKS, SUBSETS, auroc_rowlevel_ci, auroc_with_block_ci,
    load_variate, params_of_pack, subset_scores,
)

from src.ml.eval.ensemble_discrepancies import diag_gaussian_symmetric_kl  # noqa: E402

VARIATES_DEFAULT = ("gower_bgp_nla_m", "gower_nla", "gower_nla_z", "gower_vd",
                    "gower_gb1p0", "gower_gb1p3", "gower_gbk2")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--experiment", required=True, help="config name, for the min/max scaler box")
    ap.add_argument("--match-template", default="ncosmo300_{r}")
    ap.add_argument("--repeats", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--variates", nargs="+", default=list(VARIATES_DEFAULT))
    ap.add_argument("--in-dist", default="gower_bgp_nla_m")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default=None, help="suffix for the markdown report")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-events-s8", type=int, default=None)
    ap.add_argument("--no-s8", action="store_true")
    ap.add_argument("--rowlevel-ci", action="store_true",
                    help="also report the i.i.d. row-level CI (for reproducing a recorded number)")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or Path(args.base_dir).name

    from src.observation.checks.tier2 import _scaler_box
    names, lo, hi = _scaler_box(args.experiment)
    mdir0 = os.path.join(args.base_dir, "misspec", args.in_dist)
    pack_names = params_of_pack(mdir0, args.match_template, args.repeats)
    if pack_names != list(names):
        print("[kl-subsets] NOTE: pack params %s differ from the config order %s; using the PACK order"
              % (pack_names, list(names)), flush=True)
        # reorder the scaler box to the pack's parameter order
        order = [list(names).index(p) for p in pack_names]
        lo, hi, names = lo[order], hi[order], pack_names

    recs, scores = {}, {}
    for v in args.variates:
        try:
            rec = load_variate(args.base_dir, v, args.match_template, args.repeats, names, lo, hi,
                               max_events_s8=args.max_events_s8, want_s8=not args.no_s8)
        except FileNotFoundError as e:
            print("[kl-subsets] SKIP %s: %s" % (v, e), flush=True)
            continue
        recs[v] = rec
        scores[v] = subset_scores(rec, names)
        n_s8 = int(np.isfinite(rec["kl_S8"]).sum())
        print("[kl-subsets] %-18s N=%5d  ncosmo=%4d  S8-defined=%5d  selfcheck=%.2e  (%s)"
              % (v, rec["kl_full"].size, len(np.unique(rec["sim_ids"])), n_s8,
                 rec["selfcheck_rel_err"], rec["source"]), flush=True)

    if args.in_dist not in recs:
        raise SystemExit("in-distribution variate %r not loaded" % args.in_dist)
    idr = recs[args.in_dist]
    ood = [v for v in args.variates if v in recs and v != args.in_dist]

    res = {"base_dir": args.base_dir, "experiment": args.experiment, "params": list(names),
           "match_template": args.match_template, "repeats": list(args.repeats),
           "in_dist": args.in_dist, "n_boot": args.n_boot, "seed": args.seed,
           "param_blocks": {k: list(v) for k, v in PARAM_BLOCKS.items()},
           "per_dim": {}, "subsets": {}, "contributions": {}}

    # --- per-dimension: contribution share + single-dim AUROC ------------------------------
    for v in ood:
        r = recs[v]
        share = r["per_dim"].mean(axis=0) / max(r["per_dim"].mean(axis=0).sum(), 1e-30)
        res["contributions"][v] = {p: float(s) for p, s in zip(names, share)}
        res["per_dim"][v] = {}
        for i, p in enumerate(names):
            res["per_dim"][v][p] = auroc_with_block_ci(
                idr["per_dim"][:, i], idr["sim_ids"], r["per_dim"][:, i], r["sim_ids"],
                n_boot=args.n_boot, seed=args.seed)
    res["contributions"][args.in_dist] = {
        p: float(s) for p, s in zip(names, idr["per_dim"].mean(axis=0) /
                                    max(idr["per_dim"].mean(axis=0).sum(), 1e-30))}

    # --- subsets ---------------------------------------------------------------------------
    for v in ood:
        res["subsets"][v] = {}
        for sub in ALL_SCORES:
            cell = auroc_with_block_ci(scores[args.in_dist][sub], idr["sim_ids"],
                                       scores[v][sub], recs[v]["sim_ids"],
                                       n_boot=args.n_boot, seed=args.seed)
            if args.rowlevel_ci:
                cell["rowlevel"] = auroc_rowlevel_ci(scores[args.in_dist][sub], scores[v][sub],
                                                     n_boot=args.n_boot, seed=args.seed)
            res["subsets"][v][sub] = cell

    # --- pooled equal-weight-per-event column (never the headline) --------------------------
    res["pooled"] = {}
    for sub in ALL_SCORES:
        q = np.concatenate([scores[v][sub] for v in ood])
        qs = np.concatenate([recs[v]["sim_ids"] for v in ood])
        res["pooled"][sub] = auroc_with_block_ci(scores[args.in_dist][sub], idr["sim_ids"], q, qs,
                                                 n_boot=args.n_boot, seed=args.seed)

    with open(out / "kl_subset_auroc.json", "w") as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(
        out / "kl_subset_rows.npz",
        **{"%s__%s" % (v, s): scores[v][s] for v in scores for s in ALL_SCORES},
        **{"%s__per_dim" % v: recs[v]["per_dim"] for v in recs},
        **{"%s__sim_ids" % v: recs[v]["sim_ids"] for v in recs},
        params=np.array(names))

    _write_markdown(out / ("KL_SUBSETS_%s.md" % tag), res, recs, ood, names, args)
    print("[kl-subsets] wrote %s" % (out / ("KL_SUBSETS_%s.md" % tag)), flush=True)
    return res


def _cell(d):
    if d is None or not np.isfinite(d.get("auroc", np.nan)):
        return "--"
    return "%.3f (%.3f-%.3f)" % (d["auroc"], d["lo"], d["hi"])


def _write_markdown(path, res, recs, ood, names, args):
    L = []
    L.append("# KL parameter-subset study -- `%s`\n" % os.path.basename(args.base_dir))
    L.append("Pack `%s`, repeats %s, in-distribution variate `%s`. AUROC of each score against the "
             "in-distribution pool, higher = better separation; parentheses are 16-84%% "
             "**cosmology-block** bootstrap intervals (%d resamples) -- rows are ~8 correlated "
             "augmentations per cosmology, so row-level intervals would be far too tight.\n"
             % (args.experiment, list(args.repeats), args.in_dist, args.n_boot))
    L.append("Parameters (from the pack's own moments npz): `%s`.\n" % ", ".join(names))

    L.append("\n## Event counts\n")
    L.append("| variate | events | cosmologies | S8-defined |")
    L.append("|---|---|---|---|")
    for v in [args.in_dist] + ood:
        r = recs[v]
        L.append("| `%s` | %d | %d | %d |" % (v, r["kl_full"].size, len(np.unique(r["sim_ids"])),
                                              int(np.isfinite(r["kl_S8"]).sum())))

    L.append("\n## Per-dimension AUROC (which parameters carry the detection)\n")
    L.append("| parameter | " + " | ".join("`%s`" % v for v in ood) + " |")
    L.append("|---|" + "---|" * len(ood))
    for p in names:
        L.append("| `%s` | " % p + " | ".join(_cell(res["per_dim"][v][p]) for v in ood) + " |")

    L.append("\n### Mean share of the full KL carried by each parameter\n")
    L.append("| parameter | " + " | ".join("`%s`" % v for v in [args.in_dist] + ood) + " |")
    L.append("|---|" + "---|" * (len(ood) + 1))
    for p in names:
        L.append("| `%s` | " % p + " | ".join("%.3f" % res["contributions"][v][p]
                                              for v in [args.in_dist] + ood) + " |")

    L.append("\n## Subset AUROC\n")
    L.append("| subset | " + " | ".join("`%s`" % v for v in ood) + " | pooled |")
    L.append("|---|" + "---|" * (len(ood) + 1))
    for sub in ALL_SCORES:
        L.append("| `%s` | " % sub + " | ".join(_cell(res["subsets"][v][sub]) for v in ood)
                 + " | " + _cell(res["pooled"][sub]) + " |")
    L.append("\n`full` is the current production detector. A subset is an improvement only where "
             "its interval sits above `full`'s; read this as a trade-off matrix, not a winner.\n")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
