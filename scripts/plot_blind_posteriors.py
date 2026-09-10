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
Colours are pinned by ARM (src.viz.style; --palette picks the option); only the label order of
labels_overplot.png is randomised, so nothing about a label's provenance is encoded.
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

import re  # noqa: E402

import numpy as np  # noqa: E402

from src.viz import style as S  # noqa: E402


def _arm_key(experiment: str) -> str:
    """Short arm name from the experiment name (band / nla_m / nla_m_nobgp / nla / nla_z / vd / k2)."""
    e = experiment
    # ⚠️ `band` MUST be tested first: the M16 2-pt experiment name
    # `gower_nle_finetune_band_nla_m_bgp_k8_r0_ens9_e150` contains "nla_m_bgp", so any later
    # ordering silently labels the 2-pt arm as the flagship map arm.
    if "_band_" in e:
        return "band"
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
    return S.label(p)


def _rep(match: str) -> str:
    """'ncosmo300_1' / 'ncosmoNone_3' -> 'r1' / 'r3' (a legend name that is LaTeX-safe)."""
    m = re.search(r"_(\d+)$", match or "")
    return f"r{m.group(1)}" if m else S.tex(match or "")


def _chain(z, names, params, name, role="outline", colour=None, **kw):
    """One chain for `_plot`: (name, samples, TeX columns, Chain kwargs by ROLE)."""
    cols = [_tex(p) for p in params]
    idx = [list(names).index(p) for p in params]
    return (name, np.asarray(z)[:, idx], cols, S.chain_kwargs(role, colour, name=PALETTE, **kw))


NO_TITLES = False


def _plot(chains, out_png, title=None, figsize=None, formats=("png", "pdf"), battery=False):
    """The paper's corner (src.viz.style.plot_chains) on standardised chains: fixed +-4 sigma
    extents on every panel, smooth=10 on <=4-param corners (cell 11), none on full corners."""
    samples = {n: z for n, z, _, _ in chains}
    columns = {n: c for n, _, c, _ in chains}
    kwargs = {n: k for n, _, _, k in chains}
    n_par = max(len(c) for c in columns.values())
    extents = S.standardised_extents({c for cs in columns.values() for c in cs})
    fig = S.plot_chains(samples, columns, plotting_kwargs=kwargs, extents=extents,
                        figsize=figsize or (S.FIGSIZE_CORNER if n_par <= 4 else S.FIGSIZE_CORNER_FULL),
                        smooth=S.CORNER_SMOOTH if n_par <= 4 else None, prune_ticks=n_par <= 4,
                        legend_loc="upper right", palette_name=PALETTE, savefig=None, close=False)
    if title and (battery or not NO_TITLES):
        with S.context(PALETTE):
            fig.suptitle(S.tex(title), fontsize=13, y=1.0)
    S.save(fig, out_png, formats=formats)


PALETTE = S.DEFAULT_PALETTE


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
    ap.add_argument("--no-titles", action="store_true", help="no suptitles on the headline plots A/B/B'/D (captions live in the figure-set README)")
    ap.add_argument("--palette", default=S.DEFAULT_PALETTE, choices=list(S.PALETTES),
                    help="colour option from src.viz.style (arms keep their identity in every option)")
    args = ap.parse_args(argv)
    global PALETTE, NO_TITLES
    PALETTE = args.palette
    NO_TITLES = args.no_titles

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
        arm = S.arm_name(r["arm"] or _arm_key(r["experiment"]))
        return f"{arm}, pooled" if r["pooled"] else f"{arm}, {_rep(r['match'])}"

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
    armc = lambda r: S.arm_colour(r["arm"] or _arm_key(r["experiment"]), PALETTE)   # noqa: E731
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
                                 _tag(r), "outline", armc(r)))
            _plot([_chain(z, st.names, [q for q in st.names], _tag(r), "flagship", armc(r))],
                  str(out / "battery" / f"{label}_{arm}_{r['match']}_corner.png"),
                  title=f"label {label}: {_tag(r)}, self-standardised (all parameters)", formats=("png",), battery=True)
        _plot(chains, str(out / f"plotA_{label}.png"), title=f"Plot A: label {label}, each posterior standardised to N(0,1)")
        manifest["figures"].append(str(out / f"plotA_{label}.png"))

        # ---- Plot B-prime: all arms in the flagship frame ----------------------------------
        fr = flagship_run(label)
        if fr is None:
            # Plots B and B' are the flagship-frame figures, so without a flagship run they are
            # simply absent -- and an absent figure looks the same as a figure nobody asked for.
            # Say so: `--flagship-experiment` defaults to the r0 experiment name, so a prior whose
            # r0 has not landed yet (or a mistyped name) silently produced a battery with no Plot B.
            print("  ! label %s: no run matching --flagship-experiment %s under prior %r "
                  "(available: %s) -- Plots B and B' skipped"
                  % (label, args.flagship_experiment, args.prior,
                     ", ".join(sorted({r["experiment"] for r in lruns})) or "none"))
        if fr is not None:
            chains = []
            for i, r in enumerate(lruns):
                st = standardise_in_real_frame(r["path"], r["experiment"], fr["path"], fr["experiment"], subtract="real")
                p = write_standardised(st, str(std_dir / label / r["experiment"]), f"flagframe_{args.prior}_{r['match']}")
                manifest["standardised"].append(p)
                is_flag = r["experiment"] == fr["experiment"] and r["pooled"] == fr["pooled"]
                chains.append(_chain(st.z[:args.max_samples], st.names, [q for q in args.params if q in st.names],
                                     _tag(r), "flagship" if is_flag else "outline", armc(r)))
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
                chains.append(_chain(st.z[:args.max_samples], st.names, shared, _tag(r), "outline", armc(r)))
            _plot(chains, str(out / "battery" / f"{label}_shared_corner.png"),
                  title=f"label {label}: all arms, shared parameters, flagship frame", formats=("png",), battery=True)

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
                             f"{S.arm_name(arm)}, pooled", "real")]
            reps = sorted(reps, key=lambda r: r["match"])
            ramp = S.sequential(max(len(reps), 2))
            for i, r in enumerate(reps):
                st = standardise_in_real_frame(r["path"], r["experiment"], pr["path"], pr["experiment"], subtract="real")
                p = write_standardised(st, str(std_dir / label / r["experiment"]), f"pooledframe_{args.prior}_{r['match']}")
                manifest["standardised"].append(p)
                chains.append(_chain(st.z[:args.max_samples], st.names, [q for q in args.params if q in st.names],
                                     f"{S.arm_name(arm)}, {_rep(r['match'])}", "outline", ramp[i], linewidth=1.1))
            chains = chains[1:] + chains[:1]          # repeats underneath, the pooled posterior on top
            _plot(chains, str(out / f"plotD_{label}_{arm}.png"),
                  title=f"Plot D: label {label}, arm {arm}: repeats in the POOLED frame (seed spread in sigma of the pooled)")
            manifest["figures"].append(str(out / f"plotD_{label}_{arm}.png"))

        # ---- Plot B: flagship vs matched mocks (own mean, flagship std) ------------------
        if fr is not None and args.matched_mocks:
            st_real = standardise_self(fr["path"], fr["experiment"])
            chains = [_chain(st_real.z[:args.max_samples], st_real.names, [q for q in args.params if q in st_real.names],
                             f"label {label}, flagship{', pooled' if fr['pooled'] else ''}", "real")]
            for i, tok in enumerate(args.matched_mocks):
                arm, rest = tok.split("=", 1)
                mpath, mexp = rest.rsplit(":", 1)
                for ev in args.mock_events:
                    st = standardise_in_real_frame(mpath, mexp, fr["path"], fr["experiment"], subtract="own", event=ev)
                    first = ev == args.mock_events[0]
                    chains.append(_chain(st.z[:args.max_samples], st.names, [q for q in args.params if q in st.names],
                                         f"matched mocks, {S.arm_name(arm)}" if first else f"matched mock {ev + 1}, {S.arm_name(arm)}",
                                         "mock", S.role_colour("mock", PALETTE) if i == 0 else S.arm_colour(arm, PALETTE),
                                         show_label_in_legend=first))
            chains = chains[1:] + chains[:1]          # mocks underneath, the observation on top
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
                                 f"label {label}", "outline", S.label_colour(label, PALETTE)))
        _plot(chains, str(out / "labels_overplot.png"), title=f"flagship posteriors of all labels in label {flag_label}'s frame")
        manifest["figures"].append(str(out / "labels_overplot.png"))

    with open(out / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(manifest['figures'])} figures + battery under {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
