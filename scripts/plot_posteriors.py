#!/usr/bin/env python
"""Chainconsumer corner-plot overlays from saved posterior-sample npz files.

Consumes the eval sample-dump schema (src/ml/eval/utils.py:_save_posterior_samples,
also written by gen_samples.py):
  samples    [S, N, D]  scaled [0,1]
  theta0s    [N, D]     scaled [0,1]
  test_files [N]        basenames output_<sim>_out<o>_rot<r>_<n>.h5   (optional but expected)
  sim_ids    [N], aug_ids [N]                                          (optional)

Test points are matched ACROSS input files by the full test-file basename — NOT by
(sim_id, aug_id): aug_id is only the TRAILING noise index, so e.g. out0_rot0_0 and
out1_rot0_0 of one cosmology both have aug_id=0 — only the basename is unique. Files
without `test_files` fall back to positional matching (requires equal N everywhere;
warned loudly).

Samples/truths are inverse-transformed to physical units with the preset Gower box
(src/ml/data/constants.py:COSMO_PARAM_PRESET_MINMAX — the same box used to scale cosmo
params for training), then a derived S8 = sigma_8*sqrt(omega_m/0.3) column is inserted
and ombh2 is displayed as ombh2*100. Contours are drawn with shading OFF (lines only) by
default, per-chain colors, and a Truth marker (chainconsumer >= 1.x pandas API).

Input modes:
  A) explicit files:   --samples r0.npz r1.npz r2.npz [--labels r0 r1 r2]
  B) experiment dirs:  --experiments exp_r0 exp_r1 exp_r2 \
                       --checkpoints-root /share/gpu5/.../transfer_models/checkpoints \
                       --pattern 'samples_kids_s8_analytic_*.npz'
     (each matched file becomes one chain; label = experiment name, or exp:stem when a
      pattern matches several files in one experiment dir)

Examples:
  python scripts/plot_posteriors.py --samples m0.npz m1.npz m2.npz --out plots/
  python scripts/plot_posteriors.py \
      --experiments gower_nle_finetune_nla_m_z8_r0_ens9 gower_nle_finetune_nla_m_z8_r1_ens9 \
      --checkpoints-root $MODELS_ROOT/checkpoints --pattern 'samples_kids_s8_analytic_*.npz' \
      --out $MODELS_ROOT/plots/nle_z8 --max-points 4
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

# Repo root on sys.path so `src.ml.data.constants` resolves regardless of the CWD the
# script is launched from (locally or in the cluster sbatch).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PARAM_NAMES = ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"]

# Display labels (mirrors ml_experiments_9param_saved.ipynb; \mathrm so usetex is optional).
LABELS = {
    "omega_m": r"$\Omega_\mathrm{m}$",
    "sigma_8": r"$\sigma_8$",
    "w0": r"$w_0$",
    "mnu": r"$m_\nu$",
    "h": r"$h$",
    "ns": r"$n_\mathrm{s}$",
    "ombh2": r"$\Omega_\mathrm{b} h^2 \times 10^{-2}$",
    "a_ia": r"$A_\mathrm{IA}$",
    "b_ia": r"$\beta_\mathrm{IA}$",
}
S8_LABEL = r"$S_8$"

# Paul Tol muted palette — avoids a scienceplots dependency on cluster nodes.
DEFAULT_COLORS = ["#332288", "#CC6677", "#117733", "#DDCC77", "#88CCEE", "#AA4499", "#44AA99"]


def _parse_overrides(pairs):
    """--override a_ia=-6,6 -> {'a_ia': (-6.0, 6.0)}"""
    out = {}
    for p in pairs or []:
        name, _, box = p.partition("=")
        lo, _, hi = box.partition(",")
        if not (name and lo and hi):
            raise SystemExit(f"--override expects name=min,max ; got {p!r}")
        out[name] = (float(lo), float(hi))
    return out


def _preset_box(param_names, overrides):
    from src.ml.data.constants import COSMO_PARAM_PRESET_MINMAX

    preset = dict(COSMO_PARAM_PRESET_MINMAX)
    preset.update(overrides)
    missing = [p for p in param_names if p not in preset]
    if missing:
        raise SystemExit(f"no preset min/max for params: {missing}")
    mins = np.array([preset[p][0] for p in param_names], dtype=np.float64)
    maxs = np.array([preset[p][1] for p in param_names], dtype=np.float64)
    return mins, maxs


def _load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    for key in ("samples", "theta0s"):
        if key not in d:
            raise SystemExit(f"{path}: missing required key '{key}' (has {sorted(d)})")
    if d["samples"].ndim != 3 or d["theta0s"].ndim != 2:
        raise SystemExit(
            f"{path}: bad shapes samples={d['samples'].shape} theta0s={d['theta0s'].shape}"
        )
    return d


def _resolve_inputs(args):
    """Return ordered [(label, npz_dict), ...]."""
    entries = []
    if args.samples:
        labels = args.labels or [os.path.splitext(os.path.basename(p))[0] for p in args.samples]
        if len(labels) != len(args.samples):
            raise SystemExit("--labels must match --samples in length")
        for lab, p in zip(labels, args.samples):
            entries.append((lab, _load_npz(p)))
    elif args.experiments:
        root = args.checkpoints_root
        if not root:
            raise SystemExit("--experiments mode requires --checkpoints-root")
        for exp in args.experiments:
            hits = sorted(glob.glob(os.path.join(root, exp, args.pattern)))
            if not hits:
                raise SystemExit(f"no files matching {args.pattern!r} under {os.path.join(root, exp)}")
            for h in hits:
                stem = os.path.splitext(os.path.basename(h))[0]
                lab = exp if len(hits) == 1 else f"{exp}:{stem}"
                entries.append((lab, _load_npz(h)))
    else:
        raise SystemExit("provide either --samples or --experiments")
    if len(entries) < 1:
        raise SystemExit("no input sample files resolved")
    return entries


def _common_points(entries, max_points, sim_ids_filter, point_files, select_random=None, seed=0):
    """Return [(key_basename_or_index, {label: point_index}), ...] for the points to plot.

    Selection precedence: --point-files (exact) > --select-random (seeded, without
    replacement) > first --max-points of the deterministic ordering.
    """
    have_files = all("test_files" in d for _, d in entries)
    if not have_files:
        ns = {lab: d["theta0s"].shape[0] for lab, d in entries}
        if len(set(ns.values())) != 1:
            raise SystemExit(f"positional fallback needs equal N across files; got {ns}")
        n = next(iter(ns.values()))
        print(
            "[plot] WARNING: some inputs lack 'test_files'; matching points POSITIONALLY "
            "(only valid if all files share the same test split/order).",
            flush=True,
        )
        if point_files:
            raise SystemExit("--point-files requires 'test_files' in every input npz")
        idxs = list(range(n))
        if select_random:
            rng = np.random.default_rng(seed)
            idxs = sorted(rng.choice(n, size=min(int(select_random), n), replace=False).tolist())
            print(f"[plot] randomly selected {len(idxs)} of {n} points (seed={seed}): {idxs}", flush=True)
        elif max_points:
            idxs = idxs[:max_points]
        return [(f"idx{i}", {lab: i for lab, _ in entries}) for i in idxs]

    index_maps = []
    for lab, d in entries:
        files = [os.path.basename(str(f)) for f in d["test_files"]]
        index_maps.append((lab, {f: i for i, f in enumerate(files)}))
    common = set(index_maps[0][1])
    for _, m in index_maps[1:]:
        common &= set(m)
    if not common:
        raise SystemExit("no common test-file basenames across the input sample files")

    def sim_of(f):
        m = re.search(r"output_(\d+)_", f)
        return int(m.group(1)) if m else -1

    ordered = sorted(common, key=lambda f: (sim_of(f), f))
    if point_files:
        missing = [f for f in point_files if f not in common]
        if missing:
            raise SystemExit(f"requested --point-files not present in all inputs: {missing}")
        ordered = list(point_files)
    elif sim_ids_filter:
        keep = set(sim_ids_filter)
        ordered = [f for f in ordered if sim_of(f) in keep]
        if not ordered:
            raise SystemExit(f"no common points for sim_ids {sorted(keep)}")
    if not point_files and select_random:
        rng = np.random.default_rng(seed)
        pick = sorted(rng.choice(len(ordered), size=min(int(select_random), len(ordered)), replace=False).tolist())
        ordered = [ordered[i] for i in pick]
        print(f"[plot] randomly selected {len(ordered)} of the common test points (seed={seed})", flush=True)
    elif not point_files and max_points and len(ordered) > max_points:
        print(f"[plot] plotting first {max_points} of {len(ordered)} common test points", flush=True)
        ordered = ordered[:max_points]
    return [(f, {lab: m[f] for lab, m in index_maps}) for f in ordered]


def _to_display(phys, param_names):
    """[*, D] physical -> [*, D+1] display: insert S8 after omega_m; ombh2 -> ombh2*100."""
    i_om, i_s8 = param_names.index("omega_m"), param_names.index("sigma_8")
    s8 = phys[..., i_s8] * np.sqrt(phys[..., i_om] / 0.3)
    out = np.concatenate(
        [phys[..., : i_om + 1], s8[..., None], phys[..., i_om + 1 :]], axis=-1
    )
    cols = [LABELS[p] for p in param_names]
    cols = cols[: i_om + 1] + [S8_LABEL] + cols[i_om + 1 :]
    if "ombh2" in param_names:
        j = cols.index(LABELS["ombh2"])
        out[..., j] = out[..., j] * 100.0
    return out, cols


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_argument_group("inputs")
    src.add_argument("--samples", nargs="+", help="npz sample files (one chain each)")
    src.add_argument("--labels", nargs="+", help="chain labels for --samples")
    src.add_argument("--experiments", nargs="+", help="experiment dir names under --checkpoints-root")
    src.add_argument("--checkpoints-root", help="e.g. /share/gpu5/asaoulis/transfer_models/checkpoints")
    src.add_argument("--pattern", default="samples_kids_s8_analytic_*.npz",
                     help="npz basename glob inside each experiment dir")
    sel = ap.add_argument_group("point selection")
    sel.add_argument("--max-points", type=int, default=4, help="max test points to plot (default 4)")
    sel.add_argument("--select-random", type=int, default=None,
                     help="pick N test points at random (seeded) instead of the first --max-points")
    sel.add_argument("--seed", type=int, default=0, help="rng seed for --select-random")
    sel.add_argument("--sim-ids", type=int, nargs="+", help="only points from these cosmologies")
    sel.add_argument("--point-files", nargs="+", help="exact test-file basenames to plot")
    fig = ap.add_argument_group("figure")
    fig.add_argument("--out", required=True, help="output directory for PNGs")
    fig.add_argument("--prefix", default="posterior_", help="PNG filename prefix")
    fig.add_argument("--param-names", nargs="+", default=PARAM_NAMES,
                     help="parameter order of the D axis (default: the 9-param set)")
    fig.add_argument("--params", nargs="+",
                     help="subset to plot: parameter NAMES (omega_m sigma_8 S8 w0 ...) or "
                          "display labels (default: all + derived S8)")
    fig.add_argument("--colors", nargs="+", help="per-chain colors")
    fig.add_argument("--alpha", type=float, default=0.8)
    fig.add_argument("--smooth", type=int, default=None, help="chainconsumer smoothing override")
    fig.add_argument("--figsize", type=float, default=10.0)
    fig.add_argument("--usetex", action="store_true", help="LaTeX text rendering (needs system latex)")
    fig.add_argument("--summarise", action="store_true", help="marginal summary titles")
    fig.add_argument("--shade", action="store_true",
                     help="filled contours (default OFF: lines only)")
    fig.add_argument("--override", nargs="+", metavar="P=MIN,MAX",
                     help="preset-box overrides, e.g. a_ia=-6,6 for nla/nla_z runs")
    fig.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from chainconsumer import Chain, ChainConfig, ChainConsumer, PlotConfig, Truth

    entries = _resolve_inputs(args)
    param_names = list(args.param_names)
    mins, maxs = _preset_box(param_names, _parse_overrides(args.override))
    span = maxs - mins

    D = entries[0][1]["theta0s"].shape[1]
    if D != len(param_names):
        raise SystemExit(
            f"data has D={D} params but --param-names lists {len(param_names)}: {param_names}"
        )

    # --params accepts raw parameter names as well as display labels.
    if args.params:
        name_to_label = {**LABELS, "S8": S8_LABEL, "s8": S8_LABEL}
        args.params = [name_to_label.get(p, p) for p in args.params]

    points = _common_points(entries, args.max_points, args.sim_ids, args.point_files,
                            select_random=args.select_random, seed=args.seed)
    colors = args.colors or DEFAULT_COLORS
    if len(entries) > len(colors):
        raise SystemExit(f"{len(entries)} chains but only {len(colors)} colors; pass --colors")

    # Cross-file truth consistency: the same test point must carry the same theta0 in every
    # input (catches a silently-diverged split, especially in positional-fallback mode).
    for key, idx_by_label in points:
        t0 = entries[0][1]["theta0s"][idx_by_label[entries[0][0]]]
        for lab, d in entries[1:]:
            dt = np.max(np.abs(d["theta0s"][idx_by_label[lab]] - t0))
            if dt > 1e-3:
                print(f"[plot] WARNING: theta0 mismatch at point {key} between '{entries[0][0]}' and "
                      f"'{lab}' (max |d(scaled)|={dt:.3g}) — inputs may not share a test split!",
                      flush=True)

    os.makedirs(args.out, exist_ok=True)
    written = []
    for key, idx_by_label in points:
        c = ChainConsumer()
        truth_display = None
        display_cols = None
        for z, (lab, d) in enumerate(entries):
            i = idx_by_label[lab]
            phys = d["samples"][:, i, :].astype(np.float64) * span + mins
            disp, cols = _to_display(phys, param_names)
            display_cols = cols
            df = pd.DataFrame(disp, columns=cols)
            if args.params:
                keep = [cl for cl in cols if cl in set(args.params)]
                df = df[keep]
            c.add_chain(
                Chain(
                    samples=df,
                    name=lab,
                    color=colors[z],
                    linestyle="-",
                    alpha=args.alpha,
                    shade=bool(args.shade),
                    bar_shade=bool(args.shade),
                    zorder=z,
                )
            )
            if truth_display is None:
                t_phys = d["theta0s"][i].astype(np.float64) * span + mins
                t_disp, _ = _to_display(t_phys[None, :], param_names)
                truth_display = dict(zip(cols, t_disp[0]))

        if args.smooth is not None:
            c.set_override(ChainConfig(smooth=args.smooth))
        c.set_plot_config(
            PlotConfig(
                flip=False,
                tick_font_size=14,
                label_font_size=14,
                serif=True,
                usetex=bool(args.usetex),
                max_ticks=3,
                summarise=bool(args.summarise),
                legend_kwargs={"loc": "upper right"},
            )
        )
        plot_cols = args.params or display_cols
        c.add_truth(Truth(location={k: truth_display[k] for k in plot_cols}, color="black", label="Truth"))
        c.plotter.plot(figsize=(args.figsize, args.figsize))

        stem = re.sub(r"\.h5$", "", str(key))
        out_png = os.path.join(args.out, f"{args.prefix}{stem}.png")
        plt.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
        plt.close("all")
        written.append(out_png)
        print(f"[plot] wrote {out_png}", flush=True)

    print(f"[plot] done: {len(written)} figure(s) in {args.out}", flush=True)


if __name__ == "__main__":
    main()
