"""Pooled posteriors: the equal-weight mixture q_pool(theta | x) = (1/R) sum_r q_r(theta | x).

The pooled sample set is the per-repeat draws concatenated along the draw axis, written in the
_save_posterior_samples schema and scored by the standard evaluation. test_log_prob is not
poolable from samples and is written as null. Metrics are gated on finite truth, so observation
dumps print no sample statistic.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


# --- loading and alignment ---
@dataclass
class Member:
    """One repeat's posterior dump."""
    path: str
    samples: np.ndarray          # [S, N, D]
    theta0s: np.ndarray          # [N, D]
    test_files: Optional[np.ndarray]
    sim_ids: Optional[np.ndarray]
    aug_ids: Optional[np.ndarray]

    @property
    def n_draws(self):
        return int(self.samples.shape[0])

    @property
    def n_rows(self):
        return int(self.theta0s.shape[0])


def load_member(path):
    d = np.load(path, allow_pickle=False)
    if "samples" not in d.files or "theta0s" not in d.files:
        raise ValueError("%s is not a posterior dump (keys: %s)" % (path, list(d.files)))
    get = lambda k: np.asarray(d[k]) if k in d.files else None      # noqa: E731
    return Member(path=path, samples=np.asarray(d["samples"]), theta0s=np.asarray(d["theta0s"]),
                  test_files=get("test_files"), sim_ids=get("sim_ids"), aug_ids=get("aug_ids"))


def _aligned(a: Member, b: Member):
    """Are two members row-aligned? -> (bool, how, detail).

    test_files is the primary key. theta0s is the fallback for dumps without test_files, compared
    with equal_nan=True because observation dumps carry an all-NaN truth vector.
    """
    if a.n_rows != b.n_rows:
        return False, "n_rows", "%d vs %d" % (a.n_rows, b.n_rows)
    if a.test_files is not None and b.test_files is not None:
        if not np.array_equal(a.test_files, b.test_files):
            return False, "test_files", "same length, different files or order"
        return True, "test_files", "%d rows" % a.n_rows
    if not np.array_equal(a.theta0s, b.theta0s, equal_nan=True):
        return False, "theta0s", "truth vectors differ"
    return True, "theta0s(equal_nan)", "%d rows (no test_files on at least one dump)" % a.n_rows


def _git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


# --- pooling ---
def pool_sample_dumps(member_npzs: Sequence[str], out_path: str, *,
                      draws_per_member: Optional[int] = None,
                      provenance: Optional[dict] = None,
                      rng_seed: int = 0,
                      compress: bool = True):
    """Concatenate row-aligned posterior dumps along the draw axis -> one pooled dump.

    member_npzs are the per-repeat dumps in recording order. draws_per_member subsamples each
    member to that many draws first, which leaves the mixture unchanged and adds Monte-Carlo noise.
    Returns the written payload's metadata dict.
    """
    if len(member_npzs) < 2:
        raise ValueError("pooling needs >= 2 members, got %d" % len(member_npzs))
    members = [load_member(p) for p in member_npzs]

    ref = members[0]
    for m in members[1:]:
        ok, how, detail = _aligned(ref, m)
        if not ok:
            raise ValueError(
                "members are NOT row-aligned (%s: %s)\n  %s\n  %s\n"
                "Pooling misaligned rows silently mixes different observations into one posterior."
                % (how, detail, ref.path, m.path))
        if m.samples.shape[2] != ref.samples.shape[2]:
            raise ValueError("member dimensionality differs (%d vs %d): %s vs %s -- these are not "
                             "the same parameter vector" % (ref.samples.shape[2],
                                                            m.samples.shape[2], ref.path, m.path))
    align_how = _aligned(ref, members[1])[1]

    counts = [m.n_draws for m in members]
    if draws_per_member is None and len(set(counts)) != 1:
        raise ValueError(
            "members have DIFFERENT draw counts %s. An equal-weight mixture requires equal S -- "
            "pooling as-is would silently down-weight the short member (a job that died early "
            "reweights the science). Pass draws_per_member=%d to subsample them to a common size, "
            "deliberately." % (counts, min(counts)))

    rng = np.random.default_rng(rng_seed)
    per = [draws_per_member if draws_per_member is not None else m.n_draws for m in members]
    if draws_per_member is not None:
        for i, m in enumerate(members):
            if draws_per_member > m.n_draws:
                raise ValueError("draws_per_member=%d exceeds member %d's %d draws (%s)"
                                 % (draws_per_member, i, m.n_draws, m.path))

    # Preallocate and fill, freeing each member as it is copied: np.concatenate holds inputs and output at once.
    total = int(sum(per))
    pooled = np.empty((total, ref.n_rows, ref.samples.shape[2]), dtype=ref.samples.dtype)
    owner = np.empty(total, dtype=np.int8)
    at = 0
    for i, m in enumerate(members):
        s = m.samples
        if draws_per_member is not None:
            # Draws within a member are exchangeable, so a subset without replacement is unbiased.
            s = s[np.sort(rng.choice(m.n_draws, size=draws_per_member, replace=False))]
        pooled[at:at + per[i]] = s
        owner[at:at + per[i]] = i
        at += per[i]
        m.samples = None
        del s
    assert at == total, "fill loop wrote %d of %d draws" % (at, total)

    payload = {"samples": pooled, "theta0s": ref.theta0s, "member_of_draw": owner}
    for key in ("test_files", "sim_ids", "aug_ids"):
        val = getattr(ref, key)
        if val is not None:
            payload[key] = val

    has_truth = bool(np.isfinite(ref.theta0s).all())
    meta = {
        "n_members": len(members),
        "members": [os.path.basename(m.path) for m in members],
        "member_paths": [m.path for m in members],
        "draws_per_member": [int(x) for x in per],
        "draws_total": int(pooled.shape[0]),
        "n_rows": ref.n_rows, "n_dims": int(pooled.shape[2]),
        "row_alignment_key": align_how,
        "has_truth": has_truth,
        "subsampled": draws_per_member is not None,
        "rng_seed": rng_seed if draws_per_member is not None else None,
        "git_rev": _git_rev(),
        "note": ("equal-weight mixture over members; member_of_draw records the owner of every "
                 "draw. test_log_prob is NOT poolable from samples (needs the member flows)."),
    }
    if provenance:
        meta.update(provenance)
    payload["pooled_provenance"] = np.array(json.dumps(meta))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # zlib over several GB of high-entropy float32 costs minutes for a few percent. The gates turn it off.
    (np.savez_compressed if compress else np.savez)(out_path, **payload)
    # Shapes and provenance only: never a sample statistic when there is no truth.
    print("[pool] wrote %s  (%d members x %d draws = %d, N=%d, D=%d, aligned on %s, truth=%s)"
          % (out_path, meta["n_members"], meta["draws_per_member"][0], meta["draws_total"],
             meta["n_rows"], meta["n_dims"], align_how, has_truth), flush=True)
    return meta


# --- scoring ---
def _frame_for(experiments: Sequence[str]):
    """-> (cosmo_param_names, preset_minmax) shared by every member. Raises if the members differ."""
    from src.ml.data.constants import COSMO_PARAM_PRESET_MINMAX
    from config.paper_2026 import paper_2026_experiments
    from config.experiments import experiments as base_experiments
    from config.kids_legacy import kids_legacy_experiments
    from config.kids_legacy_bgp import kids_legacy_bgp_experiments
    from config.kids_legacy_vdq import kids_legacy_vdq_experiments
    from config.kids_legacy_hos import kids_legacy_hos_experiments
    from config.kids_legacy_novd import kids_legacy_novd_experiments

    exps = dict(base_experiments)
    for extra in (paper_2026_experiments, kids_legacy_experiments,
                  kids_legacy_novd_experiments,
                  kids_legacy_bgp_experiments, kids_legacy_vdq_experiments,
                  kids_legacy_hos_experiments):
        exps.update(extra)
    frames = []
    for name in experiments:
        if name not in exps:
            raise KeyError("experiment %r is not in the merged config dicts" % name)
        cfg = exps[name]
        names = list(cfg.get("cosmo_param_names") or [])
        if not names:
            from config.default import get_default_config
            names = list(get_default_config().cosmo_param_names)
        preset = dict(COSMO_PARAM_PRESET_MINMAX)
        preset.update((cfg.get("scaler_options") or {}).get("cosmo", {}).get("preset_overrides") or {})
        missing = [p for p in names if p not in preset]
        if missing:
            raise ValueError("%s: no prior box for %s" % (name, missing))
        frames.append((tuple(names), tuple(sorted((k, tuple(map(float, v)))
                                                  for k, v in preset.items() if k in names))))
    if len(set(frames)) != 1:
        raise ValueError(
            "members do not share a parameter frame -- cosmo_param_names or preset_overrides "
            "differ across %s. Pooling them would produce a posterior in no member's frame."
            % list(experiments))
    names = list(frames[0][0])
    return names, {k: list(v) for k, v in frames[0][1]}


def score_pooled_dump(npz_path, experiments: Sequence[str], out_prefix, *,
                      prior_mode: Optional[str] = None, num_prior_samples: int = 20_000):
    """Run run_evaluation_on_samples, the production eval code, on a pooled dump.

    Metrics are gated on finite truth.
    """
    import torch
    from src.ml.utils import _build_cosmo_preset_scaler
    from .evaluate_models import run_evaluation_on_samples
    from .utils import _pop_credible_intervals, _to_json_compatible

    names, preset = _frame_for(experiments)
    # A pinning prior mode samples D-1 dimensions while theta0s keeps D, so score the free columns.
    full_names = list(names)
    pinned = False
    if prior_mode:
        from .nle_external import free_param_names
        names = list(free_param_names(prior_mode, full_names))
        pinned = len(names) != len(full_names)
    scaler = _build_cosmo_preset_scaler(preset, names)

    d = np.load(npz_path, allow_pickle=False)
    samples = torch.as_tensor(np.asarray(d["samples"]))
    theta0s = torch.as_tensor(np.asarray(d["theta0s"]))
    if pinned and theta0s.shape[-1] == len(full_names):
        theta0s = theta0s[..., [full_names.index(n) for n in names]]
    if samples.shape[2] != len(names):
        raise ValueError("pooled dump is %d-D but the experiment frame names %d parameters"
                         % (samples.shape[2], len(names)))

    if not bool(torch.isfinite(theta0s).all()):
        # An observation has no truth, so every metric is skipped.
        print("[pool] no finite truth -> metrics skipped (this is the observation path)", flush=True)
        return {"metrics_json": None, "has_truth": False}

    # Pass the experiment's preset_overrides to either builder: the b_g prior width depends on them.
    overrides = {k: v for k, v in preset.items() if k in names}
    if pinned:
        # build_prior_for_mode returns the full-dimensional prior, which is wrong here.
        prior = None
    elif prior_mode:
        from .nle_external import build_prior_for_mode
        prior, _ = build_prior_for_mode(prior_mode, names, preset_overrides=overrides)
    else:
        from .utils import build_gower_prior
        prior = build_gower_prior(names, preset_overrides=overrides)

    metrics = run_evaluation_on_samples(theta0s, samples, scaler, prior=prior,
                                        compute_calibration=True,
                                        prior_num_samples=num_prior_samples)
    metrics["test_log_prob"] = None
    metrics["test_log_prob_unavailable_reason"] = (
        "the pooled posterior is a mixture; its density is logsumexp_r log q_r(theta|x) - log R, "
        "which requires the member flows and cannot be recovered from pooled samples")

    intervals = _pop_credible_intervals(metrics)
    mpath = "%s_evaluation_results.json" % out_prefix
    with open(mpath, "w") as fh:
        json.dump(_to_json_compatible({"pooled": True, "experiments": list(experiments),
                                       "metrics": metrics}), fh, indent=2)
    print("[pool] wrote %s" % mpath, flush=True)
    out = {"metrics_json": mpath, "has_truth": True}
    if intervals:
        ipath = "%s_tarp_credible_intervals.json" % out_prefix
        with open(ipath, "w") as fh:
            json.dump(_to_json_compatible(intervals), fh, indent=2)
        print("[pool] wrote %s" % ipath, flush=True)
        out["tarp_json"] = ipath
    return out


# --- gates: run the pooling path on data whose answer is known ---
DETERMINISTIC_METRIC_KEYS = ("mse", "bias", "std_dev", "width_68", "width_95")


def _ecp_from_f(f, num_alpha_bins=100):
    """The ECP curve for a set of credibilities, as at the end of tarp._get_tarp_coverage_single."""
    h, alpha = np.histogram(np.asarray(f), density=True, bins=num_alpha_bins, range=(0, 1))
    return np.concatenate([[0.0], np.cumsum(h) * (alpha[1] - alpha[0])])


def gate_pool_of_one(member_npz, experiment, workdir, *, production_json=None, atol=1e-6,
                     rows=256):
    """G1: pooling one member is the identity and rescoring reproduces the production numbers.

    The concatenation check runs on the first `rows` rows. Only DETERMINISTIC_METRIC_KEYS are
    compared to production_json, within atol. FoM and TARP carry Monte-Carlo noise and are reported.
    Returns the report dict with its verdict.
    """
    src_full = np.load(member_npz, allow_pickle=False)
    n_slice = min(int(rows), int(src_full["theta0s"].shape[0]))
    sliced = os.path.join(workdir, "gate_member_slice.npz")
    np.savez(sliced, samples=np.asarray(src_full["samples"][:, :n_slice, :]),
             theta0s=np.asarray(src_full["theta0s"][:n_slice]),
             **{k: np.asarray(src_full[k][:n_slice]) for k in ("test_files", "sim_ids", "aug_ids")
                if k in src_full.files})
    src = np.load(sliced, allow_pickle=False)["samples"]

    out = os.path.join(workdir, "gate_pool_of_one.npz")
    meta = pool_sample_dumps([sliced, sliced], out, provenance={"gate": "pool_of_one",
                                                               "rows_used": n_slice},
                             compress=False)

    got = np.load(out, allow_pickle=False)["samples"]
    half = got.shape[0] // 2
    identical = bool(np.array_equal(got[:half], src) and np.array_equal(got[half:], src))
    report = {"gate": "pool_of_one", "concatenation_exact": identical, "meta": meta}
    print("[gate G1a] concatenation is the identity: %s" % ("PASS" if identical else "FAIL"),
          flush=True)
    if not identical:
        report["verdict"] = "FAIL"
        return report

    # Score the single member, as production did, not the doubled pool.
    scored = score_pooled_dump(member_npz, [experiment],
                               os.path.join(workdir, "gate_pool_of_one"))
    report["scored"] = scored
    if not production_json or not os.path.exists(production_json):
        print("[gate G1b] no production json to compare against -- reported, not asserted",
              flush=True)
        report["verdict"] = "PASS(partial)"
        return report

    with open(production_json) as fh:
        prod = json.load(fh).get("metrics", {})
    with open(scored["metrics_json"]) as fh:
        ours = json.load(fh).get("metrics", {})
    diffs, worst = {}, 0.0
    for key, node in prod.items():
        if not isinstance(node, dict):
            continue
        for mk in DETERMINISTIC_METRIC_KEYS:
            if mk in node and isinstance(ours.get(key), dict) and mk in ours[key]:
                d = abs(float(node[mk]) - float(ours[key][mk]))
                diffs["%s.%s" % (key, mk)] = d
                worst = max(worst, d)
    report["worst_deterministic_abs_diff"] = worst
    report["n_deterministic_compared"] = len(diffs)
    ok = bool(diffs) and worst <= atol
    report["verdict"] = "PASS" if ok else "FAIL"
    print("[gate G1b] %d deterministic metrics vs production, worst |diff| = %.3g -> %s"
          % (len(diffs), worst, report["verdict"]), flush=True)
    for k in ("fom", "fom_dim_normalized"):
        if k in prod and k in ours:
            print("    %-20s production %.5f  ours %.5f (Monte-Carlo, not asserted)"
                  % (k, float(prod[k]), float(ours[k])), flush=True)
    return report


def gate_tarp_identity(pooled_npz, experiment, fcache_npz, *, set_name="full", n_ref=8, seed=20260908):
    """G2: the pooled dump's TARP agrees with the f-identity.

    fcache_npz holds the per-reference-draw credibilities, f_i(pooled) = mean_r f_i(r), for
    set_name. ours averages n_ref TARP passes on the pooled draws. Returns a dict with the max
    absolute ECP difference, the reference-draw floor and the verdict.
    """
    from src.ml.eval.tarp import get_tarp_coverage

    d = np.load(pooled_npz, allow_pickle=False)
    samples = np.asarray(d["samples"], dtype=np.float32)
    theta = np.asarray(d["theta0s"], dtype=np.float32)
    # No transpose: get_tarp_coverage wants (n_samples, n_sims, n_dims), the layout used here.
    assert samples.shape[1] == theta.shape[0], (
        "pooled dump is (S=%d, N=%d, D=%d) but theta has %d rows -- axis order is wrong"
        % (samples.shape + (theta.shape[0],)))
    # bootstrap=False: the f-identity side is a plain ECP, so the pooled side must be too.
    curves = []
    for _k in range(n_ref):
        e, alpha = get_tarp_coverage(samples, theta, num_alpha_bins=100,
                                     bootstrap=False, seed=None)
        curves.append(np.asarray(e))
    ours = np.mean(curves, axis=0)

    cached = np.load(fcache_npz, allow_pickle=False)
    keys = [k for k in cached.files if k.endswith("__" + set_name)]
    if not keys:
        raise ValueError("f-cache %s has no '%s' set (keys: %s)"
                         % (fcache_npz, set_name, sorted(cached.files)[:6]))
    f_pool = np.mean([cached[k] for k in keys], axis=0)          # [n_ref, n_rows]
    theirs = np.mean([_ecp_from_f(f_pool[k]) for k in range(f_pool.shape[0])], axis=0)

    delta = float(np.abs(ours - theirs).max())

    # The sides use different reference draws, so the floor is the gap between two halves of the cache.
    per_ref = np.array([_ecp_from_f(f_pool[k]) for k in range(f_pool.shape[0])])
    rng = np.random.default_rng(0)
    n_a = max(1, min(f_pool.shape[0] - 1, n_ref))
    floor = np.array([
        np.abs(per_ref[i[:n_a]].mean(0) - per_ref[i[n_a:]].mean(0)).max()
        for i in (rng.permutation(per_ref.shape[0]) for _ in range(300))])
    thresh = float(np.percentile(floor, 90))
    verdict = "PASS" if delta <= thresh else "FAIL"
    print("[gate G2] pooled-npz TARP vs f-identity curve: max |dECP| = %.4f over %d rows; "
          "reference-draw floor %.4f +- %.4f (90th pct %.4f) -> %s"
          % (delta, theta.shape[0], floor.mean(), floor.std(ddof=1), thresh, verdict), flush=True)
    return {"gate": "tarp_identity", "max_abs_ecp_diff": delta, "floor_p90": thresh,
            "floor_mean": float(floor.mean()), "verdict": verdict,
            "n_rows": int(theta.shape[0]), "set": set_name}
