#!/usr/bin/env python
"""Bundle Euclid Stage-I (2-pt) and Stage-II (hybrid) posteriors into ONE portable .npz.

The eval writes, per run folder, a `posterior_samples.npz` holding EVERY test cosmology at
10 000 draws (~250 MB/run) — too heavy to pass around. This packs the four runs into a single
file by keeping:

  * full 10 000-draw posteriors for a SUBSAMPLE of cosmologies (float16, plotting precision), and
  * float64 per-cosmology summaries (mean / std / 16-50-84 quantiles / 68 % width) for the FULL
    test set, so aggregate constraining-power statistics lose nothing.

⭐ VERIFIED on the real eval output: all four runs share the SAME 2041-cosmology test set, in the
same order (`apply_repeat_config` varies `split_seed`, but the test carve-out does not depend on
it). So every comparison below is paired per cosmology, and r0 vs r1 differ only by training
initialisation, not by test data. The intersections are kept as a cheap guard in case a future
config changes the split.

Usage:
    python scripts/build_euclid_posterior_bundle.py --root <fetched-eval-dir> \
        --out euclid_posterior_bundle.npz [--n-show 120]
"""
import argparse
import json
import os

import numpy as np

ARMS = {  # label -> (experiment dir, human name)
    "band": ("euclid_band", "2-pt (bandpowers)"),
    "hybrid": ("euclid_hybrid_z8_resnet_imgmatch", "Stage II (hybrid)"),
}
REPEATS = (0, 1)
RUN_TMPL = "pretrain_ncosmoNone_{i}"
PARAMS = ["omega_m", "sigma_8", "w0"]


# ⚠️ `_save_posterior_samples` dumps the flow's own coordinates, i.e. min-max SCALED [0, 1]
# space -- the rescale to physical units happens later, inside run_evaluation_on_samples. So the
# raw npz must be inverted here or every number downstream is meaningless (w0 would come out
# positive). Boxes come from src.ml.data.constants.COSMO_PARAM_PRESET_MINMAX, the same preset the
# training scalers used; the literals are a fallback so this script also runs outside the repo.
_FALLBACK_BOX = {
    "omega_m": (0.1232301212, 0.4967720335),
    "sigma_8": (0.4197701053, 1.28618999),
    "w0": (-1.00973808, -0.3391160133),
}


def _boxes(params):
    try:
        import sys, os as _os
        sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        from src.ml.data.constants import COSMO_PARAM_PRESET_MINMAX as PRESET
    except Exception:
        PRESET = _FALLBACK_BOX
    lo, hi = [], []
    for p in params:
        if p not in PRESET:
            raise SystemExit(f"no preset min/max for '{p}' — cannot unscale")
        a, b = PRESET[p]
        lo.append(a)
        hi.append(b)
    return np.asarray(lo, float), np.asarray(hi, float)


def _unscale(x, lo, hi):
    """inverse_transform_minmax: X * (max - min) + min  (src/ml/data/scaling.py)."""
    return x * (hi - lo) + lo


def _load_run(root, exp, rep):
    run = os.path.join(root, exp, RUN_TMPL.format(i=rep))
    npz = np.load(os.path.join(run, "posterior_samples.npz"), allow_pickle=False)
    samples = npz["samples"]                      # [S, N, D] as written by _save_posterior_samples
    if samples.ndim != 3:
        raise ValueError(f"{run}: expected 3-D samples, got {samples.shape}")
    samples = np.transpose(samples, (1, 0, 2))    # -> [N, S, D]
    theta = npz["theta0s"]
    lo, hi = _boxes(PARAMS)                       # scaled [0,1] -> physical units
    samples = _unscale(samples.astype(np.float64), lo, hi)
    theta = _unscale(np.asarray(theta, dtype=np.float64), lo, hi)
    sim_ids = npz["sim_ids"] if "sim_ids" in npz.files else np.arange(len(theta))
    metrics = {}
    jpath = os.path.join(run, "evaluation_results.json")
    if os.path.exists(jpath):
        with open(jpath) as f:
            metrics = json.load(f)
    return samples, theta.astype(np.float64), sim_ids.astype(np.int64), metrics


def _s8_summaries(samples, theta, params):
    """S8 = sigma_8 sqrt(Om/0.3) needs the JOINT draws, so derive it here at full precision."""
    if not {"omega_m", "sigma_8"} <= set(params):
        return {}
    io, i8 = params.index("omega_m"), params.index("sigma_8")
    s8 = samples[:, :, i8] * np.sqrt(samples[:, :, io] / 0.3)           # [N, S]
    q16, q50, q84 = np.quantile(s8, [0.16, 0.50, 0.84], axis=1)
    return {
        "s8_mean_all": s8.mean(axis=1).astype(np.float64),
        "s8_std_all": s8.std(axis=1).astype(np.float64),
        "s8_w68_all": (q84 - q16).astype(np.float64),
        "s8_q50_all": q50.astype(np.float64),
        "s8_true_all": (theta[:, i8] * np.sqrt(theta[:, io] / 0.3)).astype(np.float64),
    }


def _summaries(samples):
    """Per-cosmology posterior summaries, float64, over the FULL test set. samples: [N, S, D]."""
    q16, q50, q84 = np.quantile(samples, [0.16, 0.50, 0.84], axis=1)
    return {
        "mean_all": samples.mean(axis=1).astype(np.float64),
        "std_all": samples.std(axis=1).astype(np.float64),
        "q16_all": q16.astype(np.float64),
        "q50_all": q50.astype(np.float64),
        "q84_all": q84.astype(np.float64),
        "w68_all": (q84 - q16).astype(np.float64),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir holding <exp>/<run>/posterior_samples.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-show", type=int, default=120,
                    help="cosmologies kept at full draw resolution (float16)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runs, payload, meta_runs = {}, {}, {}
    for arm, (exp, human) in ARMS.items():
        for rep in REPEATS:
            key = f"{arm}_r{rep}"
            s, th, ids, metrics = _load_run(args.root, exp, rep)
            runs[key] = (s, th, ids)
            meta_runs[key] = {
                "arm": arm, "label": human, "repeat": rep, "experiment": exp,
                "n_test": int(s.shape[0]), "n_draws": int(s.shape[1]),
                "metrics": metrics.get("metrics", metrics),
            }
            for k, v in _summaries(s).items():
                payload[f"{key}__{k}"] = v
            for k, v in _s8_summaries(s, th, PARAMS).items():
                payload[f"{key}__{k}"] = v
            payload[f"{key}__theta_all"] = th
            payload[f"{key}__simids_all"] = ids
            print(f"  {key}: N={s.shape[0]} S={s.shape[1]} D={s.shape[2]}")

    # Plotted subsample, chosen per repeat from that repeat's band/hybrid shared split (in
    # practice all four runs share one test set — see the module docstring). Ranking the full
    # ~2000-cosmology test set by distance from fiducial is what makes the plotted examples
    # realistic: a hard box on (S8, w, Omega_m) selects only ~1-2% of the broad Gower prior.
    i_om, i_s8, i_w = (PARAMS.index(x) for x in ("omega_m", "sigma_8", "w0"))

    for rep in REPEATS:
        b_ids = runs[f"band_r{rep}"][2]
        h_ids = runs[f"hybrid_r{rep}"][2]
        common = np.intersect1d(b_ids, h_ids)
        if len(common) == 0:
            raise SystemExit(f"repeat {rep}: band and hybrid share no cosmologies — split mismatch")

        th = runs[f"band_r{rep}"][1]
        pos_b = {int(v): i for i, v in enumerate(b_ids)}
        th_c = th[[pos_b[int(v)] for v in common]]
        om, s8v, w0v = th_c[:, i_om], th_c[:, i_s8], th_c[:, i_w]
        S8 = s8v * np.sqrt(om / 0.3)
        dist = np.sqrt(((S8 - 0.80) / 0.05) ** 2
                       + ((w0v + 1.0) / 0.15) ** 2
                       + ((om - 0.30) / 0.05) ** 2)

        n_show = min(args.n_show, len(common))
        order = np.argsort(dist)[:n_show]       # closest to fiducial first
        show = common[order]
        in_box = ((S8[order] > 0.75) & (S8[order] < 0.85)
                  & (np.abs(w0v[order] + 1.0) < 0.15)
                  & (om[order] > 0.24) & (om[order] < 0.36))

        payload[f"show_sim_ids_r{rep}"] = show
        payload[f"show_S8_r{rep}"] = S8[order].astype(np.float64)
        payload[f"show_in_fiducial_box_r{rep}"] = in_box
        payload[f"show_fiducial_distance_r{rep}"] = dist[order].astype(np.float64)
        print(f"  repeat {rep}: {len(common)} shared cosmologies -> {n_show} plotted, "
              f"{int(in_box.sum())} inside the fiducial box; "
              f"S8 {S8[order].min():.3f}-{S8[order].max():.3f}, "
              f"w {w0v[order].min():.2f}..{w0v[order].max():.2f}")

        for arm in ARMS:
            key = f"{arm}_r{rep}"
            s, th_a, ids = runs[key]
            pos = {int(v): i for i, v in enumerate(ids)}
            sel = np.array([pos[int(v)] for v in show])
            payload[f"{key}__samples"] = s[sel].astype(np.float16)   # plotting precision
            payload[f"{key}__theta0s"] = th_a[sel]
            payload[f"{key}__sim_ids"] = ids[sel]

    payload["meta"] = json.dumps({
        "params": PARAMS,
        "param_labels": {"omega_m": r"$\Omega_\mathrm{m}$", "sigma_8": r"$\sigma_8$", "w0": r"$w$"},
        "runs": meta_runs,
        "n_show": int(args.n_show),
        "repeats": list(REPEATS),
        "selection": ("plotted cosmologies are chosen PER REPEAT from that repeat's shared "
                      "band/hybrid test split, ranked by normalised distance from "
                      "(S8=0.80, w=-1.0, Omega_m=0.30) and kept closest-first"),
        "notes": ("samples are float16 (plotting); *_all summaries are float64 over the full "
                  "test set. band r_i and hybrid r_i share a test split; r0 and r1 do not."),
    })

    np.savez_compressed(args.out, **payload)
    print(f"wrote {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
