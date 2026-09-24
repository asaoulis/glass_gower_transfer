#!/usr/bin/env python
"""Bundle the KiDS paper-era (mean-norm) NPE posteriors into portable .npz files, one per model.

Companion to `build_euclid_posterior_bundle.py`, with the same key layout, but the inputs here are
the paper's own `gen_samples.py` dumps (`.tch` = `(samples, theta0s)`), not an eval
`posterior_samples.npz`. Two models are packed, each into its own file:

  * transfer  -- `finetune_hybrid_16_9param_ensemble_stratify`, match `ncosmo60_0`: the 9-member
                 NPE ensemble pretrained on GLASS and finetuned on N=60 Gower Street cosmologies
                 (the paper's multifidelity result);
  * gower     -- `hybrid_patches_16_9param`, match `ncosmo400_`: the high-fidelity-only NPE
                 trained directly on N=400 Gower Street cosmologies.

Both are evaluated on held-out GOWER STREET mocks. The paper-era Gower mocks were deleted from
the cluster, so neither model can be re-sampled; these dumps are the only posteriors left.

⚠️ The two dumps do NOT share one test set. gen_samples.py drew them over the model's native test
split PLUS `N_extra_test_cosmologies=130` taken from outside that model's own train/val set, and
N=400 vs N=60 leave different cosmologies outside. They overlap on a subset of observations (the
exact 9-D theta match below). Rows are unique in 9-D theta, because the intrinsic-alignment
parameters are redrawn per augmentation, so an exact theta match identifies the same observation.
The plotted subsample is drawn ONLY from those shared rows, so the two bundles are paired
there; the `*_all` summaries cover each model's full dump.

Usage (defaults point at the paper-review mirror):
    python scripts/build_kids_paper_posterior_bundle.py --out-dir <dir> [--n-show 0] [--n-draws 10000]
"""
import argparse
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(REPO, ".claude/runs/paper-review/last-runs-and-plots/artifacts/samples")
EVAL_MIRROR = os.path.join(REPO, ".claude/runs/paper-review/last-runs-and-plots/artifacts/eval_mirror")
GOWER_CSV = os.path.join(REPO, "ml-checkpoints/gower_st/PKDGRAV3_on_DiRAC_DES_330.csv")

# key -> (experiment, match string, human label, bundle file name)
MODELS = {
    "transfer": ("finetune_hybrid_16_9param_ensemble_stratify", "ncosmo60_0",
                 "Transfer NPE, 9-member ensemble (GLASS-pretrained, finetuned on N=60 Gower, "
                 "stratified train/val selection)",
                 "kids_paper_npe_glass_transfer_bundle.npz"),
    "gower": ("hybrid_patches_16_9param", "ncosmo400_",
              "High-fidelity-only NPE (trained on N=400 Gower)",
              "kids_paper_npe_gower_only_bundle.npz"),
}
# paper-era 9-parameter order (config `cosmo_param_names`)
PARAMS = ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"]
PARAM_LABELS = {
    "omega_m": r"$\Omega_\mathrm{m}$", "sigma_8": r"$\sigma_8$", "w0": r"$w_0$",
    "mnu": r"$m_\nu$", "h": r"$h$", "ns": r"$n_\mathrm{s}$", "ombh2": r"$\Omega_\mathrm{b}h^2$",
    "a_ia": r"$A_\mathrm{IA}$", "b_ia": r"$\beta_\mathrm{IA}$",
}
# Gower prior CSV column for each N-body parameter (the IA pair is drawn per mock, not per sim)
CSV_COLS = {"omega_m": "Omega_m", "sigma_8": "sigma_8", "w0": "w", "mnu": "m_nu",
            "h": "little_h", "ns": "n_s", "ombh2": "Omega_b little_h^2"}


def _boxes(params):
    import sys
    sys.path.insert(0, REPO)
    from src.ml.data.constants import COSMO_PARAM_PRESET_MINMAX as PRESET
    lo = np.array([PRESET[p][0] for p in params], float)
    hi = np.array([PRESET[p][1] for p in params], float)
    return lo, hi


def _unscale(x, lo, hi):
    """inverse_transform_minmax: X * (max - min) + min  (src/ml/data/scaling.py)."""
    return x * (hi - lo) + lo


def _load_tch(path):
    import torch
    samples, theta = torch.load(path, weights_only=False, map_location="cpu")
    samples = np.asarray(samples, dtype=np.float64)          # [S, N, D] as gen_samples writes
    theta = np.asarray(theta, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[1:] != theta.shape:
        raise ValueError(f"{path}: samples {samples.shape} vs theta0s {theta.shape}")
    return np.transpose(samples, (1, 0, 2)), theta          # -> [N, S, D]


def _gower_sim_ids(theta_phys):
    """Recover the Gower Street serial number of every row by matching the 7 N-body parameters.

    The dumps carry no ids. The match doubles as a units check: if the preset boxes used to
    unscale were not the ones the paper-era scaler used, nothing would match to float32 precision.
    """
    import csv
    with open(GOWER_CSV) as f:
        rows = list(csv.reader(f))
    header = rows[1]
    body = [r for r in rows[2:] if r and r[0].strip()]
    serial = np.array([int(r[0]) for r in body])
    idx = [PARAMS.index(p) for p in CSV_COLS]
    tab = np.array([[float(r[header.index(c)]) for c in CSV_COLS.values()] for r in body])
    lo, hi = _boxes(list(CSV_COLS))
    # compare in scaled units so every parameter carries the same weight
    d = np.abs((theta_phys[:, None, idx] - tab[None]) / (hi - lo)).max(axis=2)   # [N, n_sims]
    best = d.argmin(axis=1)
    resid = d[np.arange(len(d)), best]
    if resid.max() > 1e-5:
        raise SystemExit(f"Gower sim-id match failed: worst scaled residual {resid.max():.2e}")
    return serial[best], float(resid.max())


def _key(theta_row):
    """Exact-match key for one observation: its 9-D theta, as stored (float32, scaled)."""
    return tuple(np.float32(theta_row).tolist())


def _summaries(samples):
    q16, q50, q84 = np.quantile(samples, [0.16, 0.50, 0.84], axis=1)
    return {"mean_all": samples.mean(axis=1), "std_all": samples.std(axis=1),
            "q16_all": q16, "q50_all": q50, "q84_all": q84, "w68_all": q84 - q16}


def _s8_summaries(samples, theta):
    io, i8 = PARAMS.index("omega_m"), PARAMS.index("sigma_8")
    s8 = samples[:, :, i8] * np.sqrt(samples[:, :, io] / 0.3)
    q16, q50, q84 = np.quantile(s8, [0.16, 0.50, 0.84], axis=1)
    return {"s8_mean_all": s8.mean(axis=1), "s8_std_all": s8.std(axis=1),
            "s8_w68_all": q84 - q16, "s8_q50_all": q50,
            "s8_true_all": theta[:, i8] * np.sqrt(theta[:, io] / 0.3)}


def _paper_evals(exp, match):
    """Per-repeat eval.py metrics for the model, where the mirror has them (provenance only)."""
    root = os.path.join(EVAL_MIRROR, exp)
    out = {}
    if not os.path.isdir(root):
        return out
    for run in sorted(os.listdir(root)):
        j = os.path.join(root, run, "evaluation_results.json")
        if match.rstrip("_") in run and os.path.exists(j):
            with open(j) as f:
                d = json.load(f)
            m = d["metrics"]
            out[run] = {"best_checkpoint": os.path.basename(d.get("best_checkpoint", "")),
                        "test_log_prob": m.get("test_log_prob"), "fom": m.get("fom"),
                        "width_68": {p: m[p]["width_68"] for p in PARAMS if p in m}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", default=SAMPLES_DIR)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-show", type=int, default=0,
                    help="observations kept at full draw resolution (0 = one per shared cosmology)")
    ap.add_argument("--n-draws", type=int, default=10000,
                    help="draws kept per plotted observation (the dumps hold 25 000)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    lo, hi = _boxes(PARAMS)

    runs = {}
    for key, (exp, match, human, _) in MODELS.items():
        path = os.path.join(args.samples_dir, f"{exp}_{match}_longsamples.tch")
        s, th = _load_tch(path)
        tkeys = [_key(r) for r in th]                     # keyed in the dump's own scaled space
        if len(set(tkeys)) != len(tkeys):
            raise SystemExit(f"{key}: theta0s rows are not unique -- exact-theta pairing invalid")
        s, th = _unscale(s, lo, hi), _unscale(th, lo, hi)
        ids, resid = _gower_sim_ids(th)
        runs[key] = (s, th, ids, os.path.basename(path), resid, tkeys)
        print(f"  {key}: N={s.shape[0]} S={s.shape[1]} D={s.shape[2]}  "
              f"{len(np.unique(ids))} Gower sims (id match residual {resid:.1e})")

    # --- rows shared by every model (exact 9-D theta) -> the paired plotted subsample ---------
    keysets = {k: {t: i for i, t in enumerate(v[5])} for k, v in runs.items()}
    shared = set.intersection(*(set(ks) for ks in keysets.values()))
    first = next(iter(runs))
    shared_rows = np.array(sorted(keysets[first][t] for t in shared))     # row idx in `first`
    th = runs[first][1][shared_rows]
    sid = runs[first][2][shared_rows]
    i_om, i_s8, i_w = (PARAMS.index(x) for x in ("omega_m", "sigma_8", "w0"))
    S8 = th[:, i_s8] * np.sqrt(th[:, i_om] / 0.3)
    dist = np.sqrt(((S8 - 0.80) / 0.05) ** 2 + ((th[:, i_w] + 1.0) / 0.15) ** 2
                   + ((th[:, i_om] - 0.30) / 0.05) ** 2)
    # one observation per Gower sim first (lowest row), closest-to-fiducial first; the remaining
    # augmentation siblings only fill in if --n-show asks for more than there are sims
    order = np.lexsort((shared_rows, dist))
    seen, primary, extra = set(), [], []
    for i in order:
        (extra if sid[i] in seen else primary).append(i)
        seen.add(sid[i])
    n_show = args.n_show or len(primary)
    pick = np.array((primary + extra)[:n_show])
    show_keys = [runs[first][5][shared_rows[i]] for i in pick]
    in_box = ((S8[pick] > 0.75) & (S8[pick] < 0.85) & (np.abs(th[pick, i_w] + 1.0) < 0.15)
              & (th[pick, i_om] > 0.24) & (th[pick, i_om] < 0.36))
    print(f"  shared: {len(shared)} observations over {len(np.unique(sid))} Gower sims -> "
          f"{len(pick)} plotted ({int(in_box.sum())} inside the fiducial box)")

    for key, (exp, match, human, fname) in MODELS.items():
        s, th_all, ids, src, resid, tkeys = runs[key]
        payload = {}
        for k, v in _summaries(s).items():
            payload[f"{key}__{k}"] = v.astype(np.float64)
        for k, v in _s8_summaries(s, th_all).items():
            payload[f"{key}__{k}"] = np.asarray(v, np.float64)
        payload[f"{key}__theta_all"] = th_all
        payload[f"{key}__simids_all"] = ids.astype(np.int64)
        payload[f"{key}__in_shared_all"] = np.array([t in shared for t in tkeys])

        sel = np.array([keysets[key][t] for t in show_keys])
        # float32, NOT the Euclid bundle's float16: in physical units the narrow-prior parameters
        # sit on a coarse f16 grid (ns ~76 distinct values per posterior, ombh2 ~57 across the
        # whole box), which shows up as combs in a 9-parameter corner plot
        payload[f"{key}__samples"] = s[sel, :args.n_draws].astype(np.float32)
        payload[f"{key}__theta0s"] = th_all[sel]
        payload[f"{key}__sim_ids"] = ids[sel].astype(np.int64)
        payload[f"{key}__rows"] = sel.astype(np.int64)
        payload["show_S8"] = S8[pick]
        payload["show_in_fiducial_box"] = in_box
        payload["show_fiducial_distance"] = dist[pick]

        # A match string without a repeat index (`ncosmo400_`) resolves to whichever repeat
        # gen_samples picked, so name it: the eval run whose mean 68 % widths the dump reproduces.
        evals = _paper_evals(exp, match)
        matched = None
        if evals:
            w = payload[f"{key}__w68_all"].mean(axis=0)[:3]
            rel = {run: float(np.abs(w / np.array([e["width_68"][p] for p in PARAMS[:3]]) - 1).max())
                   for run, e in evals.items()}
            best = min(rel, key=rel.get)
            matched = {"run": best, "max_rel_width68_diff_Om_s8_w0": rel[best]}
            print(f"  {key}: dump reproduces eval run {best} (max rel width diff {rel[best]:.3f})")

        payload["meta"] = json.dumps({
            "model": key, "label": human, "experiment": exp, "match_string": match,
            "source_dump": src, "estimator": "NPE (normalising flow, direct posterior samples)",
            "params": PARAMS, "param_labels": PARAM_LABELS,
            "units": "physical (unscaled from the COSMO_PARAM_PRESET_MINMAX min-max boxes)",
            "n_test": int(s.shape[0]), "n_draws_all": int(s.shape[1]),
            "n_draws_samples": int(min(args.n_draws, s.shape[1])),
            "n_gower_sims": int(len(np.unique(ids))), "simid_match_residual": resid,
            "paired_with": [k for k in MODELS if k != key],
            "n_shared_observations": len(shared),
            "selection": ("plotted observations are the rows shared (exact 9-D theta) with the "
                          "other bundle, one per Gower sim, ranked by normalised distance from "
                          "(S8=0.80, w=-1.0, Omega_m=0.30), closest first; identical rows and "
                          "order in both bundles"),
            "test_set": ("held-out Gower Street mocks: the model's native test split plus 130 "
                         "extra cosmologies from outside its train/val set, shape-noise index "
                         "[0,0]; ~4 augmentations per Gower sim, each with its own (a_ia, b_ia)"),
            "paper_eval_json": evals,
            "matched_eval_run": matched,
            "notes": ("KiDS-Legacy-like mocks, paper-era 'mean' shear normalisation, NLA-M IA, "
                      "9 inferred parameters. samples are float32; *_all summaries "
                      "are float64 over all draws of the full dump. `simids_all` are Gower "
                      "Street serial numbers recovered by matching the 7 N-body parameters."),
        })
        out = os.path.join(args.out_dir, fname)
        np.savez_compressed(out, **payload)
        print(f"wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
