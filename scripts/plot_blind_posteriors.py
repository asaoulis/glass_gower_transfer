#!/usr/bin/env python
"""Half-blind posterior plots (Tier 3) + the corner-plot battery. Reads raw observation posteriors
ONLY through src.blind.standardise and writes standardised npz + figures. Never prints a location.

    PYTHONPATH=. python scripts/plot_blind_posteriors.py \
        --blind-root /data/alex/unblinding/blind_store --out-dir <dir> \
        [--flagship-experiment gower_nle_finetune_nla_m_bgp_z8_r0_ens9] [--flagship-label A] \
        [--matched-mocks nla_m=/data/alex/variate_samples/<exp>/external/gower_match_nla_m/external_posterior_samples_gower_ncosmo300_0.npz:<exp> ...] \
        [--params omega_m sigma_8 S8 w0 ...] [--prior gower]

Figures (all standardised; axes in sigma units of SOME posterior, never physical):
  plotA_<label>.png       every arm x repeat of one label standardised to N(0,1) per parameter
                          (Plot A: shapes / degeneracy directions only; maximally blind)
  plotB_<label>.png       flagship posterior in its OWN frame vs the matched near-fiducial mocks
                          standardised with the flagship's std and their OWN mean (Plot B: SIZE and
                          degeneracies comparable, location hidden)
  plotBprime_<label>.png  every arm in the FLAGSHIP frame (mean AND std of the flagship): inter-arm
                          offsets visible in sigma units (user-approved arm-consistency check)
  battery/<label>_<arm>_corner.png    per-arm full-parameter corner (self frame)
  battery/<label>_shared_corner.png   all arms over-plotted on the shared parameters (flagship frame)
  labels_overplot.png     the 2-3 blinded labels' flagship posteriors in label-A's frame (blinds A/B/C)
Colours and label order are randomised per run so nothing about provenance is encoded.
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

LABELS_TEX = {"omega_m": r"$\Omega_{\rm m}$", "sigma_8": r"$\sigma_8$", "S8": r"$S_8$", "w0": r"$w_0$",
              "mnu": r"$\sum m_\nu$", "h": r"$h$", "ns": r"$n_s$", "ombh2": r"$\Omega_b h^2$",
              "a_ia": r"$A_{\rm IA}$", "b_ia": r"$\beta_{\rm IA}$", "b_z": r"$B_{\rm IA}$"}


def _arm_key(experiment: str) -> str:
    """Short arm name from the experiment name (nla_m / nla_m_nobgp / nla / nla_z / vd / k2)."""
    e = experiment
    if "bgpk2" in e:
        return "k2"
    if "_vd_" in e:
        return "vd"
    if "nla_z" in e:
        return "nla_z"
    if "nla_m_bgp" in e:
        return "nla_m"
    if "nla_m_z8" in e:
        return "nla_m_nobgp"
    if "nla_bgp" in e:
        return "nla"
    return e


def _tex(p: str) -> str:
    if p in LABELS_TEX:
        return LABELS_TEX[p]
    if p.startswith("b_g_bin"):
        return r"$b_g^{(%s)}$" % p[len("b_g_bin"):]
    return p.replace("_", " ")     # never feed a raw underscore to mathtext


def _chain(z, names, params, name, **kw):
    import pandas as pd
    from chainconsumer import Chain
    cols = [_tex(p) for p in params]
    idx = [list(names).index(p) for p in params]
    df = pd.DataFrame(z[:, idx], columns=cols)
    return Chain(samples=df, parameters=cols, name=name, **kw)


def _plot(chains, out_png, title=None, figsize=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from chainconsumer import ChainConsumer, PlotConfig
    c = ChainConsumer()
    for ch in chains:
        c.add_chain(ch)
    c.set_plot_config(PlotConfig(flip=False, tick_font_size=10, label_font_size=12, serif=True, usetex=False,
                                 legend_kwargs={"loc": "upper right"}))
    fig = c.plotter.plot(figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blind-root", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--flagship-experiment", default="gower_nle_finetune_nla_m_bgp_z8_r0_ens9")
    ap.add_argument("--flagship-label", default=None, help="label whose flagship frame is the reference (default: first)")
    ap.add_argument("--prior", default="gower", help="prior tag of the runs to plot")
    ap.add_argument("--use", default="auto", choices=["auto", "pooled", "repeats"],
                    help="headline posterior per arm: the POOLED run (final analysis; auto = pooled when present, "
                         "else the per-repeat runs), or the per-repeat runs only")
    ap.add_argument("--params", nargs="+", default=["omega_m", "sigma_8", "S8", "w0"])
    ap.add_argument("--matched-mocks", nargs="*", default=[], metavar="ARM=PATH:EXPERIMENT",
                    help="matched near-fiducial mock dumps per arm (Plot B)")
    ap.add_argument("--mock-events", type=int, nargs="+", default=[0, 1, 2], help="events of the mock dump to overlay")
    ap.add_argument("--seed", type=int, default=None, help="colour/order randomisation seed (default: random)")
    ap.add_argument("--max-samples", type=int, default=20000)
    args = ap.parse_args(argv)

    from src.blind import BLIND_ROOT, STANDARDISED_ROOT
    from src.blind.standardise import (list_raw_runs, standardise_in_real_frame, standardise_self,
                                       write_standardised)
    root = args.blind_root or BLIND_ROOT
    out = Path(args.out_dir)
    (out / "battery").mkdir(parents=True, exist_ok=True)
    std_dir = Path(STANDARDISED_ROOT)
    all_runs = [r for r in list_raw_runs(root) if r["prior"] == args.prior]
    if not all_runs:
        raise SystemExit(f"no raw runs with prior={args.prior} under {root}")
    labels = sorted({r["label"] for r in all_runs})

    def _tag(r):
        return f"{r['arm'] or _arm_key(r['experiment'])} {'POOLED' if r['pooled'] else r['match']}"

    def headline_runs(label):
        """One posterior per arm: the pooled one when it exists (the final analysis), else the repeats."""
        lr = [r for r in all_runs if r["label"] == label]
        if args.use == "repeats":
            return [r for r in lr if not r["pooled"]]
        out = []
        for arm in sorted({r["arm"] or _arm_key(r["experiment"]) for r in lr}):
            ar = [r for r in lr if (r["arm"] or _arm_key(r["experiment"])) == arm]
            pooled = [r for r in ar if r["pooled"]]
            if pooled:
                out += pooled
            elif args.use == "auto":
                out += [r for r in ar if not r["pooled"]]
        return out

    runs = [r for label in labels for r in headline_runs(label)]
    if not runs:
        raise SystemExit(f"no headline runs (--use {args.use}) with prior={args.prior} under {root}")
    rng = np.random.default_rng(args.seed)
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf", "#7f7f7f"]
    rng.shuffle(palette)
    manifest = {"prior": args.prior, "use": args.use, "labels": labels, "figures": [], "standardised": [],
                "headline": [{"label": r["label"], "arm": r["arm"], "pooled": r["pooled"], "match": r["match"]} for r in runs]}
    flag_label = args.flagship_label or labels[0]

    def flagship_run(label):
        pooled = [r for r in runs if r["label"] == label and r["pooled"] and r["experiment"] == args.flagship_experiment]
        cands = pooled or [r for r in runs if r["label"] == label and r["experiment"] == args.flagship_experiment]
        return cands[0] if cands else None

    for label in labels:
        lruns = [r for r in runs if r["label"] == label]
        # ---- Plot A: everything in its own frame ------------------------------------------
        chains = []
        for i, r in enumerate(lruns):
            st = standardise_self(r["path"], r["experiment"])
            p = write_standardised(st, str(std_dir / label / r["experiment"]), f"self_{args.prior}_{r['match']}")
            manifest["standardised"].append(p)
            z = st.z[:args.max_samples]
            arm = r["arm"] or _arm_key(r["experiment"])
            chains.append(_chain(z, st.names, [q for q in args.params if q in st.names],
                                 _tag(r), color=palette[i % len(palette)], linewidth=1.2))
            _plot([_chain(z, st.names, [q for q in st.names], _tag(r), color=palette[i % len(palette)])],
                  str(out / "battery" / f"{label}_{arm}_{r['match']}_corner.png"),
                  title=f"label {label} | arm {arm} | {'POOLED' if r['pooled'] else r['match']} | self-standardised (all params)")
        _plot(chains, str(out / f"plotA_{label}.png"), title=f"Plot A: label {label}, each posterior standardised to N(0,1)")
        manifest["figures"].append(str(out / f"plotA_{label}.png"))

        # ---- Plot B-prime: all arms in the flagship frame ----------------------------------
        fr = flagship_run(label)
        if fr is not None:
            chains = []
            for i, r in enumerate(lruns):
                st = standardise_in_real_frame(r["path"], r["experiment"], fr["path"], fr["experiment"], subtract="real")
                p = write_standardised(st, str(std_dir / label / r["experiment"]), f"flagframe_{args.prior}_{r['match']}")
                manifest["standardised"].append(p)
                chains.append(_chain(st.z[:args.max_samples], st.names, [q for q in args.params if q in st.names],
                                     _tag(r), color=palette[i % len(palette)], linewidth=1.2))
            _plot(chains, str(out / f"plotBprime_{label}.png"),
                  title=f"Plot B': label {label}, all arms in the flagship frame (offsets in sigma of the flagship)")
            manifest["figures"].append(str(out / f"plotBprime_{label}.png"))
            # shared over-plot on the common parameter set, all params
            shared = None
            for r in lruns:
                st = standardise_in_real_frame(r["path"], r["experiment"], fr["path"], fr["experiment"], subtract="real")
                shared = list(st.names) if shared is None else [n for n in shared if n in st.names]
            chains = []
            for i, r in enumerate(lruns):
                st = standardise_in_real_frame(r["path"], r["experiment"], fr["path"], fr["experiment"], subtract="real")
                chains.append(_chain(st.z[:args.max_samples], st.names, shared, _tag(r),
                                     color=palette[i % len(palette)]))
            _plot(chains, str(out / "battery" / f"{label}_shared_corner.png"),
                  title=f"label {label}: all arms, shared parameters, flagship frame")

        # ---- Plot D: seed spread -- every repeat in the POOLED frame of its arm ------------
        # Diagnostic for the final (pooled) posterior: pooling can only ADD spread across seeds, so
        # the per-repeat offsets/widths in sigma of the pooled posterior show how much of the pooled
        # width is seed disagreement. Blind-safe: everything is relative to the pooled location.
        for pr in [r for r in lruns if r["pooled"]]:
            arm = pr["arm"]
            reps = [r for r in all_runs if r["label"] == label and not r["pooled"]
                    and (r["arm"] or _arm_key(r["experiment"])) == arm]
            if not reps:
                continue
            stp = standardise_self(pr["path"], pr["experiment"])
            chains = [_chain(stp.z[:args.max_samples], stp.names, [q for q in args.params if q in stp.names],
                             f"{arm} POOLED", color="black", linewidth=2.0)]
            for i, r in enumerate(sorted(reps, key=lambda r: r["match"])):
                st = standardise_in_real_frame(r["path"], r["experiment"], pr["path"], pr["experiment"], subtract="real")
                p = write_standardised(st, str(std_dir / label / r["experiment"]), f"pooledframe_{args.prior}_{r['match']}")
                manifest["standardised"].append(p)
                chains.append(_chain(st.z[:args.max_samples], st.names, [q for q in args.params if q in st.names],
                                     f"{arm} {r['match']}", color=palette[i % len(palette)], linewidth=0.9))
            _plot(chains, str(out / f"plotD_{label}_{arm}.png"),
                  title=f"Plot D: label {label}, arm {arm}: repeats in the POOLED frame (seed spread in sigma of the pooled)")
            manifest["figures"].append(str(out / f"plotD_{label}_{arm}.png"))

        # ---- Plot B: flagship vs matched mocks (own mean, flagship std) ------------------
        if fr is not None and args.matched_mocks:
            st_real = standardise_self(fr["path"], fr["experiment"])
            chains = [_chain(st_real.z[:args.max_samples], st_real.names, [q for q in args.params if q in st_real.names],
                             f"label {label} (flagship{' POOLED' if fr['pooled'] else ''})", color="black", linewidth=2.0)]
            for i, tok in enumerate(args.matched_mocks):
                arm, rest = tok.split("=", 1)
                mpath, mexp = rest.rsplit(":", 1)
                for ev in args.mock_events:
                    st = standardise_in_real_frame(mpath, mexp, fr["path"], fr["experiment"], subtract="own", event=ev)
                    chains.append(_chain(st.z[:args.max_samples], st.names, [q for q in args.params if q in st.names],
                                         f"mock {arm} ev{ev}", color=palette[(i + 1) % len(palette)], linewidth=0.8))
            _plot(chains, str(out / f"plotB_{label}.png"),
                  title=f"Plot B: label {label} flagship (own frame) vs matched mocks (own mean, flagship std)")
            manifest["figures"].append(str(out / f"plotB_{label}.png"))

    # ---- labels over-plot: flagship posterior of every label in label-A's frame ---------------
    fa = flagship_run(flag_label)
    if fa is not None and len(labels) > 1:
        chains = []
        order = list(labels)
        rng.shuffle(order)
        for i, label in enumerate(order):
            fr = flagship_run(label)
            if fr is None:
                continue
            st = standardise_in_real_frame(fr["path"], fr["experiment"], fa["path"], fa["experiment"], subtract="real")
            chains.append(_chain(st.z[:args.max_samples], st.names, [q for q in args.params if q in st.names],
                                 f"label {label}", color=palette[i % len(palette)]))
        _plot(chains, str(out / "labels_overplot.png"), title=f"flagship posteriors of all labels in label {flag_label}'s frame")
        manifest["figures"].append(str(out / "labels_overplot.png"))

    with open(out / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(manifest['figures'])} figures + battery under {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
