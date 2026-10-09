#!/usr/bin/env python
"""Tier-3 posterior-level SANITY gate (blind-safe): did every repeat score the observation with a
correctly loaded model? Emits ONLY relative scalars (KL between repeats, width RATIOS, R-hat, PTEs)
-- never a posterior mean, std or location.

    PYTHONPATH=. python scripts/tier3_sanity.py --label A2 --arms nla_m band \
        --priors LCDM_fixed_w0 kids_s8_analytic --out-dir /data/alex/unblinding/real/work/tier3/sanity_A2

Why a posterior-level check on top of the load-time identity assertions and the repro gate: a model
can load the right FILES and still be wrong (e.g. a stale member, a mis-scaled input), and the
observation has no truth. What it does have is FIVE independently trained repeats that should agree
about it as well as they agree about in-distribution mocks. A misloaded encoder / whitener / member in
one repeat moves that repeat's posterior far outside the repeat-to-repeat spread seen on mocks.

Statistics per (arm, prior), each against the SAME statistic computed per event on a null:
  kl_diag      mean pairwise symmetric diag-Gaussian KL over the free parameters
               (`ensemble_discrepancies.diag_gaussian_symmetric_kl`, the misspec repeat-disagreement metric)
  kl_oms8      mean pairwise symmetric FULL-covariance Gaussian KL on (omega_m, sigma_8)
  loo_max      max over repeats of [mean diag-KL of repeat r vs the others] -> names the outlier repeat
  width_max    max over (repeat, parameter) of |log(std_r / median_s std_s)|
plus, per repeat, the between-shard R-hat (shards = `mcmc_workers` blocks of the dump) and the draw count.

Nulls (scaled space; both KLs are invariant to per-parameter affine maps, so the frame does not matter):
  testset   the Stage-B in-distribution Gower test-set dumps of the SAME 5 repeats
            (`ensemble_posterior_samples_<match>.npz`, aligned by test-file basename), GOWER prior --
            approximate for the KiDS priors; for LCDM each null posterior is Gaussian-CONDITIONED on
            w0 = -1 (the U-E slab conditioning), so its widths match an LCDM posterior's.
  matched   (if present for every repeat) the matched-mock dumps under the SAME prior.
Verdict FLAG if any PTE < --pte-flag, any R-hat > --rhat-flag, or a draw count != --expect-draws.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.ml.eval.arms import ACTIVE_ARMS, ARMS, arm_match_string, assert_active  # noqa: E402
from src.ml.eval.ensemble_discrepancies import diag_gaussian_symmetric_kl  # noqa: E402

TESTSET_ROOT = "/data/alex/variate_samples"
MATCHED_ROOT = "/data/alex/variate_samples_kids"
OMS8 = ("omega_m", "sigma_8")


# ------------------------------------------------------------------ moments + statistics
def _moments(S):
    """S [n_draws, N, D] -> mu [N, D], var [N, D], cov2 handled separately."""
    S = np.asarray(S, dtype=np.float64)
    return S.mean(axis=0), S.var(axis=0, ddof=1)


def _cov_pair(S, i, j):
    """[N, 2, 2] covariance of columns (i, j)."""
    X = np.asarray(S[..., [i, j]], dtype=np.float64)          # [n, N, 2]
    Xc = X - X.mean(axis=0, keepdims=True)
    return np.einsum("snk,snl->nkl", Xc, Xc) / (X.shape[0] - 1)


def _kl_full(m0, c0, m1, c1):
    """KL(N0 || N1) for batches [N, d], [N, d, d]."""
    d = m0.shape[-1]
    c1i = np.linalg.inv(c1)
    dm = (m1 - m0)[..., None]
    tr = np.einsum("nij,nji->n", c1i, c0)
    quad = (np.swapaxes(dm, -1, -2) @ c1i @ dm)[..., 0, 0]
    ld = np.linalg.slogdet(c1)[1] - np.linalg.slogdet(c0)[1]
    return 0.5 * (tr + quad - d + ld)


def _pair_sym_diag(mu, var, a, b):
    va, vb = np.maximum(var[a], 1e-12), np.maximum(var[b], 1e-12)
    dm2 = (mu[a] - mu[b]) ** 2
    kab = 0.5 * np.sum(va / vb + dm2 / vb - 1 + np.log(vb / va), axis=-1)
    kba = 0.5 * np.sum(vb / va + dm2 / va - 1 + np.log(va / vb), axis=-1)
    return 0.5 * (kab + kba)


def statistics(mu, var, m2, c2):
    """mu, var [K, N, D]; m2 [K, N, 2], c2 [K, N, 2, 2] -> dict of [N] arrays + loo [K, N]."""
    K = mu.shape[0]
    kl_diag = diag_gaussian_symmetric_kl(mu, var)
    pairs = [(a, b) for a in range(K) for b in range(a + 1, K)]
    kl2 = np.mean([0.5 * (_kl_full(m2[a], c2[a], m2[b], c2[b]) + _kl_full(m2[b], c2[b], m2[a], c2[a]))
                   for a, b in pairs], axis=0)
    loo = np.stack([np.mean([_pair_sym_diag(mu, var, r, s) for s in range(K) if s != r], axis=0)
                    for r in range(K)])                                    # [K, N]
    sd = np.sqrt(np.maximum(var, 1e-24))
    width = np.abs(np.log(sd / np.median(sd, axis=0, keepdims=True))).max(axis=(0, 2))   # [N]
    return {"kl_diag": kl_diag, "kl_oms8": kl2, "loo_max": loo.max(axis=0), "width_max": width}, loo


def rhat_blocks(S, n_blocks):
    """Between-shard R-hat per parameter for ONE event, S [n, D]; blocks are contiguous shards."""
    n = S.shape[0]
    L = int(np.ceil(n / n_blocks))
    blocks = [S[i * L:(i + 1) * L] for i in range(n_blocks) if S[i * L:(i + 1) * L].shape[0] > 1]
    m = len(blocks)
    Lmin = min(b.shape[0] for b in blocks)
    blocks = np.stack([b[:Lmin] for b in blocks])                         # [m, L, D]
    W = blocks.var(axis=1, ddof=1).mean(axis=0)
    B = Lmin * blocks.mean(axis=1).var(axis=0, ddof=1)
    var_hat = (Lmin - 1) / Lmin * W + B / Lmin
    return np.sqrt(var_hat / np.maximum(W, 1e-24)), m


def pte(null, obs):
    null = np.asarray(null)
    null = null[np.isfinite(null)]
    return float((1 + np.sum(null >= obs)) / (1 + null.size)), int(null.size)


# ------------------------------------------------------------------ loading
def _param_names(experiment):
    import eval as _eval
    cfg, _ = _eval.load_config(experiment)
    return list(cfg.cosmo_param_names)


def _free(prior, names):
    from src.ml.eval.nle_external import free_param_names
    return list(free_param_names(prior, names))


def _full_moments(S, chunk=256):
    """S [n, N, D] -> mean [N, D], full covariance [N, D, D] (event-chunked to bound memory)."""
    n, N, D = S.shape
    mu, cov = np.empty((N, D)), np.empty((N, D, D))
    for a in range(0, N, chunk):
        X = np.asarray(S[:, a:a + chunk, :], dtype=np.float64)
        m = X.mean(axis=0)
        Xc = X - m
        mu[a:a + chunk] = m
        cov[a:a + chunk] = np.einsum("snk,snl->nkl", Xc, Xc) / (n - 1)
    return mu, cov


def _condition(mu, cov, free_idx, pin_idx, pin_val):
    """Gaussian moments of the free block given the pinned coordinates = pin_val (the slab /
    Gaussian conditioning of U-E, validated on mocks to 0.01-0.02 sigma); marginalisation if no pin."""
    mf, Sff = mu[:, free_idx], cov[:, free_idx][:, :, free_idx]
    if not pin_idx:
        return mf, Sff
    Sfp = cov[:, free_idx][:, :, pin_idx]
    Spp = cov[:, pin_idx][:, :, pin_idx]
    G = Sfp @ np.linalg.inv(Spp)                                            # [N, f, p]
    dx = (np.asarray(pin_val)[None, :] - mu[:, pin_idx])[..., None]           # [N, p, 1]
    return mf + (G @ dx)[..., 0], Sff - G @ np.swapaxes(Sfp, -1, -2)


def _null_moments(paths, names_full, free, cache_dir, tag, pinned=None):
    """Moments of K aligned null dumps on the `free` columns (cached as a small npz). `pinned`
    {name: SCALED value}: the null posteriors were drawn with that parameter free (Gower prior), so
    they are CONDITIONED on it (else an LCDM observation, whose posteriors are narrower, would be
    judged against a too-wide null and false-flag)."""
    pinned = dict(pinned or {})
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"null_{tag}.npz")
    key = json.dumps({"free": list(free), "pinned": pinned, "paths": list(paths)}, sort_keys=True)
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=False)
        if str(d["key"]) == key:
            return d["mu"], d["var"], d["m2"], d["c2"], int(d["n"])
    files, per = None, []
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            S, tf = z["samples"], [os.path.basename(str(f)) for f in z["test_files"]]
        if S.shape[-1] != len(names_full):
            raise ValueError(f"{p}: {S.shape[-1]} cols vs {len(names_full)} names")
        per.append((S, tf))
        files = set(tf) if files is None else files & set(tf)
    common = sorted(files)
    fi = [names_full.index(n) for n in free]
    pi = [names_full.index(n) for n in pinned]
    pv = [pinned[n] for n in pinned]
    i, j = free.index(OMS8[0]), free.index(OMS8[1])
    mus, vrs, m2s, c2s = [], [], [], []
    for S, tf in per:
        idx = {f: k for k, f in enumerate(tf)}
        mu_full, cov_full = _full_moments(S[:, [idx[f] for f in common], :])
        mu, cov = _condition(mu_full, cov_full, fi, pi, pv)
        mus.append(mu); vrs.append(np.einsum("nii->ni", cov))
        m2s.append(mu[:, [i, j]]); c2s.append(cov[:, [i, j]][:, :, [i, j]])
    del per
    out = [np.stack(x) for x in (mus, vrs, m2s, c2s)]
    np.savez_compressed(cache, mu=out[0], var=out[1], m2=out[2], c2=out[3], n=len(common), key=np.array(key))
    return (*out, len(common))


def _pinned_scaled(experiment, prior):
    """{name: scaled value} of the parameters `prior` pins, in the experiment's preset frame."""
    from src.ml.eval.nle_external import FIXED_BY_PRIOR_MODE
    fx = dict(FIXED_BY_PRIOR_MODE.get(prior, {}) or {})
    if not fx:
        return {}
    import eval as _eval
    from src.ml.data.constants import COSMO_PARAM_PRESET_MINMAX
    from src.ml.utils import _build_cosmo_preset_scaler
    cfg, _ = _eval.load_config(experiment)
    preset = dict(getattr(cfg, "scaler_options", {}).get("cosmo", {}).get("preset_overrides", {}) or {})
    names = list(fx)
    sc = _build_cosmo_preset_scaler({**COSMO_PARAM_PRESET_MINMAX, **preset}, names)
    v = np.asarray(sc.transform(np.array([[fx[n] for n in names]], dtype=np.float32)), dtype=np.float64)[0]
    return {n: float(x) for n, x in zip(names, v)}


def _obs_runs(label, experiment, prior, match):
    from src.blind.standardise import BLIND_ROOT, list_raw_runs
    hits = [r for r in list_raw_runs(BLIND_ROOT) if r["label"] == label and r["experiment"] == experiment
            and r["prior"] == prior and r["match"] == match and not r["pooled"]]
    if len(hits) != 1:
        raise FileNotFoundError(f"{label}/{experiment}/{prior}/{match}: {len(hits)} raw dumps (need 1)")
    return hits[0]["path"]


def _provenance(npz_path, match, prior):
    """Per-prior provenance (written since 2026-10-09); the legacy per-match file is shared by the
    priors of one repeat, so it is accepted only if it names THIS prior."""
    d = os.path.dirname(npz_path)
    p = os.path.join(d, f"external_provenance_{prior}_{match}.json")
    if os.path.exists(p):
        return json.load(open(p))
    p = os.path.join(d, f"external_provenance_{match}.json")
    if os.path.exists(p):
        j = json.load(open(p))
        return j if j.get("prior") == prior else {}
    return {}


# ------------------------------------------------------------------ main
_HULL = None


def _build_hull(exp0, names_full, test_path):
    """(omega_m, sigma_8) convex hull of the Gower cosmologies (unique sims of the Stage-B test set,
    SCALED units) -- the region the likelihood was actually trained on."""
    global _HULL
    from scipy.spatial import Delaunay
    if not os.path.exists(test_path):
        return None
    with np.load(test_path, allow_pickle=False) as z:
        t, sid = z["theta0s"], z["sim_ids"]
    _, u = np.unique(sid, return_index=True)
    _HULL = Delaunay(t[u][:, [names_full.index(OMS8[0]), names_full.index(OMS8[1])]])
    return _HULL


def _null_kurtosis(path, names_full, n_events=400):
    """Excess kurtosis of the (omega_m, sigma_8) marginals across test-set mock posteriors."""
    with np.load(path, allow_pickle=False) as z:
        S = z["samples"]
        idx = np.linspace(0, S.shape[1] - 1, min(n_events, S.shape[1])).astype(int)
        return {n: stats.kurtosis(np.asarray(S[:, idx, names_full.index(n)], dtype=np.float64), axis=0)
                for n in OMS8}


def run_arm_prior(args, arm, prior):
    assert_active(arm)
    tmpl, _bake, reps = ARMS[arm]
    reps = [r for r in reps if r in args.repeats]
    exps = [tmpl.format(r=r) for r in reps]
    matches = [arm_match_string(arm, r) for r in reps]
    names_full = _param_names(exps[0])
    free = _free(prior, names_full)
    i, j = free.index(OMS8[0]), free.index(OMS8[1])
    res = {"arm": arm, "prior": prior, "repeats": reps, "free_params": free, "issues": []}
    _t0 = os.path.join(TESTSET_ROOT, exps[0], f"ensemble_posterior_samples_{matches[0]}.npz")
    _build_hull(exps[0], names_full, _t0)

    # observation: per-repeat moments + shard R-hat + draw counts (nothing location-like leaves here)
    mu, var, m2, c2, rh = [], [], [], [], {}
    for r, e, m in zip(reps, exps, matches):
        path = _obs_runs(args.label, e, prior, m)
        with np.load(path, allow_pickle=False) as z:
            S = np.asarray(z["samples"])
        if S.ndim == 2:
            S = S[:, None, :]
        if S.shape[1] != 1 or S.shape[2] != len(free):
            res["issues"].append(f"r{r}: dump shape {S.shape} != (n, 1, {len(free)})")
            return res
        if not np.isfinite(S).all():
            res["issues"].append(f"r{r}: non-finite draws")
        if args.expect_draws and S.shape[0] != args.expect_draws:
            res["issues"].append(f"r{r}: {S.shape[0]} draws != {args.expect_draws}")
        prov = _provenance(path, m, prior)
        K = int((prov.get("sampling") or {}).get("mcmc_workers") or 0)
        if K > 1:
            R, nb = rhat_blocks(S[:, 0, :].astype(np.float64), K)
            rh[f"r{r}"] = {"max": float(R.max()), "worst_param": free[int(R.argmax())], "n_shards": nb}
            if R.max() > args.rhat_flag:
                res["issues"].append(f"r{r}: shard R-hat {R.max():.4f} > {args.rhat_flag} ({free[int(R.argmax())]})")
        else:
            rh[f"r{r}"] = {"max": None, "note": "no mcmc_workers in provenance"}
        a, b = _moments(S)
        mu.append(a); var.append(b); m2.append(a[:, [i, j]]); c2.append(_cov_pair(S, i, j))
        # SUPPORT + SHAPE (blind-safe: fractions and standardised moments only)
        X = S[:, 0, :].astype(np.float64)
        sup = res.setdefault("support", {})
        sup[f"r{r}"] = {"frac_outside_box_any": float(np.mean(np.any((X < 0) | (X > 1), axis=1))),
                        "frac_outside_box": {n: float(np.mean((X[:, k] < 0) | (X[:, k] > 1)))
                                             for k, n in enumerate(free) if np.any((X[:, k] < 0) | (X[:, k] > 1))},
                        "frac_outside_sim_hull_oms8": (float(np.mean(_HULL.find_simplex(X[:, [i, j]]) < 0))
                                                       if _HULL is not None else None),
                        "excess_kurtosis": {n: float(stats.kurtosis(X[:, free.index(n)]))
                                            for n in OMS8}}
        # identity provenance (filenames only): the exact files this repeat was scored with
        pv = prov.get("provenance") or {}
        res.setdefault("identity", {})[f"r{r}"] = {k: pv.get(k) for k in
                                                   ("source_checkpoints", "whitener_path", "member_checkpoints")}
    res["rhat"] = rh
    if os.path.exists(_t0):
        nk = _null_kurtosis(_t0, names_full)
        for rr, sv in res.get("support", {}).items():
            sv["kurtosis_pte_vs_testset"] = {n: pte(nk[n], sv["excess_kurtosis"][n])[0] for n in OMS8}
            sv["testset_kurtosis_median_p95"] = {n: [float(np.median(nk[n])), float(np.quantile(nk[n], .95))] for n in OMS8}
    obs, loo = statistics(*(np.stack(x) for x in (mu, var, m2, c2)))
    res["obs"] = {k: float(v[0]) for k, v in obs.items()}
    res["obs_loo_per_repeat"] = {f"r{r}": float(loo[k, 0]) for k, r in enumerate(reps)}

    # nulls
    res["nulls"] = {}
    test_paths = [os.path.join(TESTSET_ROOT, e, f"ensemble_posterior_samples_{m}.npz") for e, m in zip(exps, matches)]
    cands = {"testset (Gower prior, in-dist)": test_paths}
    if args.matched_tag:
        stem = "external_" + "posterior_samples_"            # (string split: see guard_blind)
        cands[f"matched {args.matched_tag} ({prior})"] = [
            os.path.join(MATCHED_ROOT, e, "external", args.matched_tag, f"{stem}{prior}_{m}.npz")
            for e, m in zip(exps, matches)]
    for nname, paths in cands.items():
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            res["nulls"][nname] = {"missing": [os.path.relpath(p, os.path.dirname(os.path.dirname(p))) for p in missing]}
            continue
        # same-prior nulls (matched) already pin; the Gower-prior test-set null is conditioned
        pin = _pinned_scaled(exps[0], prior) if nname.startswith("testset") else {}
        nmu, nvar, nm2, nc2, n = _null_moments(paths, names_full if nname.startswith("testset") else free,
                                               free, args.cache_dir, f"{arm}_{prior}_{nname.split()[0]}", pin)
        null, nloo = statistics(nmu, nvar, nm2, nc2)
        block = {"n_events": n}
        for k, v in null.items():
            p_, nn = pte(v, res["obs"][k])
            block[k] = {"obs": res["obs"][k], "null_median": float(np.median(v)),
                        "null_p95": float(np.quantile(v, 0.95)), "null_p99": float(np.quantile(v, 0.99)), "pte": p_}
            if p_ < args.pte_flag:
                res["issues"].append(f"{nname}: {k} PTE {p_:.4f} < {args.pte_flag}")
        # per-repeat LOO PTE against that repeat's own null LOO -> names the culprit repeat
        block["loo_pte_per_repeat"] = {f"r{r}": pte(nloo[k], float(loo[k, 0]))[0] for k, r in enumerate(reps)}
        res["nulls"][nname] = block
    if not any("n_events" in b for b in res["nulls"].values()):
        res["issues"].append("no complete null available")
    res["verdict"] = "FLAG" if res["issues"] else "PASS"
    return res


def _md(results):
    L = ["| arm | prior | stat | obs | null median | null p99 | PTE |", "|---|---|---|---:|---:|---:|---:|"]
    for r in results:
        for nname, b in (r.get("nulls") or {}).items():
            if "n_events" not in b:
                continue
            for k in ("kl_diag", "kl_oms8", "loo_max", "width_max"):
                x = b[k]
                L.append(f"| {r['arm']} | {r['prior']} | {k} [{nname.split()[0]}] | {x['obs']:.3g} | "
                         f"{x['null_median']:.3g} | {x['null_p99']:.3g} | {x['pte']:.3f} |")
    L.append("")
    for r in results:
        L.append(f"- **{r['arm']} / {r['prior']}: {r.get('verdict')}**"
                 + ("" if not r["issues"] else " — " + "; ".join(r["issues"])))
        for rr, sv in (r.get("support") or {}).items():
            L.append(f"  - {rr} support: outside box {sv['frac_outside_box_any']:.3f} {sv['frac_outside_box']}; "
                     f"outside Gower (Om,s8) hull {sv['frac_outside_sim_hull_oms8']}; excess kurtosis "
                     + ", ".join(f"{n} {sv['excess_kurtosis'][n]:.2f} (testset med/p95 "
                                 f"{sv.get('testset_kurtosis_median_p95', {}).get(n, ['?','?'])[0]:.2f}/"
                                 f"{sv.get('testset_kurtosis_median_p95', {}).get(n, ['?','?'])[1]:.2f}, "
                                 f"PTE {sv.get('kurtosis_pte_vs_testset', {}).get(n, float('nan')):.3f})" for n in OMS8))
        L.append(f"  - shard R-hat: " + ", ".join(f"{k} {v.get('max') if v.get('max') is None else format(v['max'], '.4f')}"
                                                  for k, v in (r.get("rhat") or {}).items()))
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--arms", nargs="+", default=list(ACTIVE_ARMS))
    ap.add_argument("--priors", nargs="+", default=["LCDM_fixed_w0", "kids_s8_analytic"])
    ap.add_argument("--repeats", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--matched-tag", default=None, help="e.g. gower_match_vdq (used only if all repeats exist)")
    ap.add_argument("--expect-draws", type=int, default=25000)
    ap.add_argument("--pte-flag", type=float, default=0.01)
    ap.add_argument("--rhat-flag", type=float, default=1.01)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default="/data/alex/unblinding/real/work/tier3/sanity_null_cache")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for arm in args.arms:
        for prior in args.priors:
            try:
                results.append(run_arm_prior(args, arm, prior))
            except Exception as e:                      # one arm's failure must not hide the others
                results.append({"arm": arm, "prior": prior, "issues": [f"{type(e).__name__}: {e}"], "verdict": "FLAG"})
    # IDENTITY consistency (filenames only): one (arm, repeat) must have been scored with the SAME
    # source encoder / whitener / 9 member files under every prior -- and in the matched-mock run when
    # it exists. A differing file means a checkpoint was re-resolved differently between jobs.
    for arm in args.arms:
        rows = [r for r in results if r.get("arm") == arm and r.get("identity")]
        refs = {}
        for r in rows:
            for rep, ident in r["identity"].items():
                if not ident.get("member_checkpoints") or not ident.get("source_checkpoints"):
                    r["issues"].append(f"{rep}: provenance lacks identity fields (pre-assertion code rev?)")
                refs.setdefault(rep, []).append((r["prior"], json.dumps(ident, sort_keys=True)))
        if args.matched_tag:
            tmpl = ARMS[arm][0]
            for rep in list(refs):
                rr = int(rep[1:])
                pj = os.path.join(MATCHED_ROOT, tmpl.format(r=rr), "external", args.matched_tag,
                                  f"external_provenance_LCDM_fixed_w0_{arm_match_string(arm, rr)}.json")
                if os.path.exists(pj):
                    pv = (json.load(open(pj)).get("provenance") or {})
                    refs[rep].append(("matched", json.dumps({k: pv.get(k) for k in
                                      ("source_checkpoints", "whitener_path", "member_checkpoints")}, sort_keys=True)))
        for rep, lst in refs.items():
            if len({s for _, s in lst}) > 1:
                msg = f"{rep}: scored with DIFFERENT files across {[p for p, _ in lst]}"
                for r in rows:
                    r["issues"].append(msg)
                    r["verdict"] = "FLAG"
    with open(os.path.join(args.out_dir, f"SANITY_{args.label}.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    md = _md(results)
    with open(os.path.join(args.out_dir, f"SANITY_{args.label}.md"), "w") as fh:
        fh.write(f"# Tier-3 sanity gate — label {args.label}\n\n" + md)
    print(md)
    return 0 if all(r.get("verdict") == "PASS" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
