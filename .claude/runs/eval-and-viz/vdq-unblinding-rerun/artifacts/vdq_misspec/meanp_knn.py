#!/usr/bin/env python
"""P5.3 -- the OBS-IDENTICAL mean-p column for the VDQ Tier-2 tables.

The real observation's `meanp_recalibrated` (src/observation/cluster_modes.py:run_obs_score) is

    meanp_raw  = mean over the pack's encoders (member 0 per repeat) of the UNCONDITIONAL kNN p
                 (`ood_knn_p`, k=10, whitener fit on that encoder's `_train` cloud, p vs its
                 `_idtest` null)
    recal      = empirical_pvalues(-null, -meanp_raw),  null = `_idtest_meanp_null` (the same
                 encoder-mean over the `_idtest` rows)

The phase6 `combined_p` the 09-24 tables used is a DIFFERENT detector (encoder-mean of the
CONDITIONAL kNN p, needs the true theta, half-A recalibrated). This script builds the statistic
the observation actually is, per mock row, so a table and its reading share one detector (D3).

Rows
  * OOD variates: in-job `ood_knn_p` of each repeat's member-0 summaries (verified here to equal
    a local recomputation from `ood_knn_score` against the local `_idtest` null -- i.e. the local
    `_train/_idtest` are the ones the job used), mean over encoders, recalibrated against the
    full `_idtest` null (the obs's own null).
  * In-distribution variate (== the `_idtest` files): LEAVE-ONE-COSMOLOGY-OUT. For cosmology c
    the null is `_idtest` minus c's rows at BOTH stages (per-encoder p and the mean-p recal), so a
    row is never scored against itself or its sibling augmentations.

GATE (plumbing): the obs json's own per-encoder p through `meanp_from_per_encoder` with the full
null must reproduce `meanp_raw` and `meanp_recalibrated` to 4 d.p. and `n_idtest_null`. A failure
means the local `_train/_idtest` are not what the obs was scored against: STOP, do not tune.

    python meanp_knn.py --pack field [--variates ...] [--out <npz>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vdq_config as C  # noqa: E402

from src.ml.eval.ood import TrainWhitener, empirical_pvalues, knn_scores  # noqa: E402
from src.observation.cluster_modes import _idtest_meanp_null, _representative_summaries  # noqa: E402

K = 10          # == eval.py --ood-k default == cluster_modes --score-knn-k default


def _b(x):
    return os.path.basename(str(x))


def meanp_from_per_encoder(p_enc, null_meanp):
    """(raw, recal) for per-encoder p of shape [..., R] against a mean-p null (LOW = OOD)."""
    raw = np.mean(np.asarray(p_enc, dtype=np.float64), axis=-1)
    recal = empirical_pvalues(-np.asarray(null_meanp, dtype=np.float64), -np.atleast_1d(raw))
    return raw, recal


def idtest_scores(sdir, summ_matches):
    """Per encoder: kNN score of every `_idtest` row vs that encoder's whitened `_train` cloud,
    exactly as `_idtest_meanp_null` / `OODReference.fit` compute it. Rows aligned by file."""
    files0, sims0, augs0, S = None, None, None, []
    for m in summ_matches:
        ftr = _representative_summaries(os.path.join(sdir, "_train"), m)
        fid = _representative_summaries(os.path.join(sdir, "_idtest"), m)
        dtr, did = np.load(ftr), np.load(fid)
        wh = TrainWhitener.fit(np.asarray(dtr["z"], dtype=np.float64))
        s = knn_scores(wh(np.asarray(dtr["z"], dtype=np.float64)), wh(np.asarray(did["z"], dtype=np.float64)), k=K)
        files = [_b(x) for x in did["test_files"]]
        if files0 is None:
            files0, sims0, augs0 = files, np.asarray(did["sim_ids"], int), np.asarray(did["aug_ids"], int)
            S.append(s)
        else:
            pos = {f: i for i, f in enumerate(files)}
            assert set(pos) == set(files0), f"_idtest files differ across encoders ({m})"
            S.append(s[[pos[f] for f in files0]])
    return files0, sims0, augs0, np.stack(S, axis=1)          # [N, R]


def variate_rows(sdir, v, summ_matches, null_scores, files_id):
    """In-job member-0 ood_knn_p per encoder for variate v, aligned on common files; also
    re-derives each p from ood_knn_score vs the local null (identity check)."""
    per, maxdev, ndev, ntot = [], 0.0, 0, 0
    for j, m in enumerate(summ_matches):
        f = _representative_summaries(os.path.join(sdir, v), m)
        if f is None:
            raise FileNotFoundError(f"no summaries for {v} {m} in {sdir}")
        d = np.load(f)
        p_job = np.asarray(d["ood_knn_p"], dtype=np.float64)
        p_loc = empirical_pvalues(null_scores[:, j], np.asarray(d["ood_knn_score"], dtype=np.float64))
        maxdev = max(maxdev, float(np.max(np.abs(p_job - p_loc))))
        ndev += int(np.sum(np.abs(p_job - p_loc) > 0))
        ntot += len(p_job)
        per.append({_b(x): (p, int(s), int(a)) for x, p, s, a in zip(d["test_files"], p_job, d["sim_ids"], d["aug_ids"])})
    common = sorted(set.intersection(*(set(x) for x in per)))
    P = np.array([[x[f][0] for x in per] for f in common])
    sims = np.array([per[0][f][1] for f in common])
    augs = np.array([per[0][f][2] for f in common])
    # deviation in RANK STEPS of the null (1/(n+1)); <=1 step on a few rows = a null re-encoded on a
    # different device (gb1p0 summaries ran on CPU), not a different detector
    steps = maxdev * (null_scores.shape[0] + 1)
    return common, sims, augs, P, {"max_abs_dev_p": maxdev, "max_rank_steps": float(steps),
                                   "frac_cells_dev": ndev / max(ntot, 1)}


def loco_in_dist(S_id, sims):
    """Leave-one-cosmology-out per-encoder p, raw mean-p and recal for every `_idtest` row."""
    n, R = S_id.shape
    P = np.full((n, R), np.nan)
    raw = np.full(n, np.nan)
    recal = np.full(n, np.nan)
    for c in np.unique(sims):
        q, nn = sims == c, sims != c
        Pn = np.stack([empirical_pvalues(S_id[nn, j], S_id[nn, j]) for j in range(R)], axis=1)
        Pq = np.stack([empirical_pvalues(S_id[nn, j], S_id[q, j]) for j in range(R)], axis=1)
        null_c = Pn.mean(axis=1)
        r_, rc = meanp_from_per_encoder(Pq, null_c)
        P[q], raw[q], recal[q] = Pq, r_, rc
    return P, raw, recal


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", choices=sorted(C.PACKS), required=True)
    ap.add_argument("--variates", nargs="+", default=None, help="default: every VARIATES dir present locally")
    ap.add_argument("--out", default=None, help="default: vdq_config PACKS[pack]['meanp_npz']")
    ap.add_argument("--gate-only", action="store_true")
    a = ap.parse_args(argv)
    pk = C.PACKS[a.pack]
    sdir = pk["summaries"]
    out = a.out or pk["meanp_npz"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rep = {"pack": pk["exp"], "repeats": pk["repeats"], "summ_matches": pk["summ_matches"], "k": K}

    # ---- the null, two ways (must agree) -------------------------------------------------
    files_id, sims_id, augs_id, S_id = idtest_scores(sdir, pk["summ_matches"])
    P_id_full = np.stack([empirical_pvalues(S_id[:, j], S_id[:, j]) for j in range(S_id.shape[1])], axis=1)
    order = np.argsort(files_id)                        # _idtest_meanp_null sorts the common files
    mine = P_id_full.mean(axis=1)[order]
    null = _idtest_meanp_null(sdir, pk["summ_matches"], k=K)
    assert null.shape == mine.shape and np.array_equal(null, mine), "local null != _idtest_meanp_null"
    rep["n_null"] = int(null.size)
    rep["n_null_cosmologies"] = int(len(np.unique(sims_id)))

    # ---- GATE: the observation's own per-encoder p -> its recorded meanp -------------------
    with open(pk["obs_json"]) as fh:
        obs = json.load(fh)
    p_obs = [obs["per_encoder"][m.replace("_m0", "")]["ood_knn_p"] for m in pk["summ_matches"]]
    raw_o, rec_o = meanp_from_per_encoder(np.array(p_obs), null)
    gate = {"obs_label": obs["label"], "per_encoder_p": p_obs,
            "meanp_raw": float(raw_o), "meanp_raw_json": obs["meanp_raw"],
            "meanp_recal": float(rec_o[0]), "meanp_recal_json": obs["meanp_recalibrated"],
            "n_null": int(null.size), "n_null_json": obs.get("n_idtest_null")}
    gate["pass"] = bool(round(gate["meanp_raw"], 4) == round(gate["meanp_raw_json"], 4)
                        and round(gate["meanp_recal"], 4) == round(gate["meanp_recal_json"], 4)
                        and gate["n_null"] == gate["n_null_json"])
    rep["gate"] = gate
    print(f"[gate] {a.pack}: raw {gate['meanp_raw']:.6f} vs json {gate['meanp_raw_json']:.6f}; recal "
          f"{gate['meanp_recal']:.6f} vs json {gate['meanp_recal_json']:.6f}; n_null {gate['n_null']} vs "
          f"{gate['n_null_json']} -> {'PASS' if gate['pass'] else 'FAIL'}", flush=True)
    if not gate["pass"]:
        with open(out.replace(".npz", "_report.json"), "w") as fh:
            json.dump(rep, fh, indent=2, default=float)
        print("[gate] FAIL: local _train/_idtest are not what the observation was scored against -- STOP")
        return 2
    if a.gate_only:
        return 0

    # ---- rows ------------------------------------------------------------------------------
    variates = a.variates or [v for v in C.VARIATES if os.path.isdir(os.path.join(sdir, v))]
    cols = {k: [] for k in ("variate", "test_files", "sim_ids", "aug_ids", "meanp_raw", "meanp_knn_recal")}
    P_all = []
    rep["variates"] = {}
    for v in variates:
        if v == C.IN_DIST:
            P, raw, recal = loco_in_dist(S_id, sims_id)
            files, sims, augs = files_id, sims_id, augs_id
            # the in-dist variate summaries must be exactly the _idtest files
            fv = _representative_summaries(os.path.join(sdir, v), pk["summ_matches"][0])
            fv_files = set(_b(x) for x in np.load(fv)["test_files"])
            assert fv_files == set(files_id), "in-dist variate files != _idtest files"
            # in-job p of the in-dist variate (self-included null) for comparison
            _, _, _, Pjob, dev = variate_rows(sdir, v, pk["summ_matches"], S_id, files_id)
            info = {"loco": True, "identity": dev,
                    "selfnull_frac_recal_lt_0.05": float(np.mean(
                        meanp_from_per_encoder(Pjob, null)[1] < 0.05))}
        else:
            files, sims, augs, P, dev = variate_rows(sdir, v, pk["summ_matches"], S_id, files_id)
            raw, recal = meanp_from_per_encoder(P, null)
            info = {"loco": False, "identity": dev}
        info.update({"n_rows": len(files), "n_cosmologies": int(len(np.unique(sims))),
                     "median_meanp_recal": float(np.median(recal)),
                     "frac_recal_lt_0.05": float(np.mean(recal < 0.05)),
                     "frac_recal_lt_0.01": float(np.mean(recal < 0.01))})
        rep["variates"][v] = info
        print(f"[rows] {v}: n={info['n_rows']} ({info['n_cosmologies']} cos), identity: <= {dev['max_rank_steps']:.2f} "
              f"rank steps on {dev['frac_cells_dev']:.2%} of (row, encoder) cells, "
              f"median recal {info['median_meanp_recal']:.3f}, frac<0.05 {info['frac_recal_lt_0.05']:.3f}"
              + (f" (self-null {info['selfnull_frac_recal_lt_0.05']:.3f})" if info["loco"] else ""), flush=True)
        if dev["max_rank_steps"] > 1.01 or dev["frac_cells_dev"] > 0.05:
            print(f"[rows] WARNING {v}: in-job ood_knn_p differs from the local recomputation beyond one rank "
                  f"step on a few rows ({dev}) -- the job's _train/_idtest are not the local ones", flush=True)
        cols["variate"] += [v] * len(files)
        cols["test_files"] += list(files)
        cols["sim_ids"] += list(sims)
        cols["aug_ids"] += list(augs)
        cols["meanp_raw"] += list(raw)
        cols["meanp_knn_recal"] += list(recal)
        P_all.append(P)
    np.savez(out, **{k: np.asarray(v) for k, v in cols.items()}, p_enc=np.concatenate(P_all),
             null_meanp=null, repeats=np.asarray(pk["repeats"]))
    with open(out.replace(".npz", "_report.json"), "w") as fh:
        json.dump(rep, fh, indent=2, default=float)
    print(f"[write] {out} ({len(cols['variate'])} rows) + _report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
