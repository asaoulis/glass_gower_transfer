"""eval.py --mode pool: build and score the pooled (repeat-concatenated) posterior.

The launcher passes only bare names, so arm and experiment names are mapped to paths here.
Example: eval.py --mode pool --pool-arm nla_m --repeat-indices 0 1 2. Outputs go under
checkpoints/pooled/<label>/, never inside a member's directory.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence


def _base_path(args):
    override = getattr(args, "base_path", None)
    if override:
        return override
    from config.default import get_default_config
    return get_default_config().base_path


def _member_experiments(args) -> List[str]:
    """-> the per-repeat experiment names, from --pool-arm or --experiments."""
    from .arms import arm_experiments

    if getattr(args, "pool_arm", None):
        pairs = arm_experiments(args.pool_arm, getattr(args, "repeat_indices", None))
        return [exp for _r, exp in pairs]
    exps = list(getattr(args, "experiments", None) or [])
    if len(exps) < 2:
        raise SystemExit("--mode pool needs >= 2 members: pass --pool-arm ARM (with optional "
                         "--repeat-indices) or --experiments EXP1 EXP2 ...")
    return exps


def _match_for(args, experiment: str, index: int) -> str:
    """The eval match_string for one member.

    From the arm table when an arm is named. Otherwise the repeat is read from the experiment
    name's `_r<N>_` field.
    """
    from .arms import arm_match_string
    import re

    if getattr(args, "pool_arm", None):
        reps = getattr(args, "repeat_indices", None)
        from .arms import ARMS
        repeat = int(reps[index]) if reps else int(ARMS[args.pool_arm][2][index])
        return arm_match_string(args.pool_arm, repeat)

    m = re.search(r"_r(\d+)_", experiment)
    if not m:
        raise SystemExit("cannot infer the repeat index from experiment %r -- use --pool-arm, "
                         "whose table carries the match strings explicitly" % experiment)
    r = int(m.group(1))
    # Both forms exist on disk. _resolve_member_npz probes the other one.
    return "ncosmoNone_%d" % r if "_hf" in experiment or "bgpk2" in experiment else "ncosmo300_%d" % r


def _resolve_member_npz(base_path: str, experiment: str, match: str, *,
                        data_tag: Optional[str], prior_mode: Optional[str]) -> str:
    """The dump a member wrote, for either family. Probes both match forms before giving up."""
    exp_dir = os.path.join(base_path, "checkpoints", experiment)
    cands = []
    if data_tag:
        d = os.path.join(exp_dir, "external", data_tag)
        prior = prior_mode or "gower"
        cands.append(os.path.join(d, "external_posterior_samples_%s_%s.npz" % (prior, match)))
        alt = "ncosmo300_" if match.startswith("ncosmoNone_") else "ncosmoNone_"
        cands.append(os.path.join(d, "external_posterior_samples_%s_%s.npz"
                                  % (prior, alt + match.rsplit("_", 1)[1])))
    else:
        cands.append(os.path.join(exp_dir, "ensemble_posterior_samples_%s.npz" % match))
        alt = "ncosmo300_" if match.startswith("ncosmoNone_") else "ncosmoNone_"
        cands.append(os.path.join(exp_dir, "ensemble_posterior_samples_%s.npz"
                                  % (alt + match.rsplit("_", 1)[1])))
    for c in cands:
        if os.path.exists(c):
            return c
    raise SystemExit("no posterior dump for %s (match %s). Looked for:\n  %s"
                     % (experiment, match, "\n  ".join(cands)))


def _label(args, experiments: Sequence[str]) -> str:
    if getattr(args, "pool_arm", None):
        return args.pool_arm
    pref = os.path.commonprefix(list(experiments)).rstrip("_r")
    return pref or "pooled"


def run_pool_mode(args):
    from .pooling import gate_pool_of_one, gate_tarp_identity, pool_sample_dumps, score_pooled_dump

    base_path = _base_path(args)
    experiments = _member_experiments(args)
    data_tag = getattr(args, "data_tag", None)
    prior_mode = getattr(args, "prior_mode", None)
    # The mock path takes its prior from the experiment's own boxes, so prior_mode needs a data tag.
    prior_mode = prior_mode if data_tag else None

    matches = [_match_for(args, exp, i) for i, exp in enumerate(experiments)]
    members = [_resolve_member_npz(base_path, exp, m, data_tag=data_tag, prior_mode=prior_mode)
               for exp, m in zip(experiments, matches)]

    label = _label(args, experiments)
    out_dir = os.path.join(base_path, "checkpoints", "pooled", label)
    if data_tag:
        out_dir = os.path.join(out_dir, "external", data_tag)
    os.makedirs(out_dir, exist_ok=True)

    print("[pool] %d members -> %s" % (len(members), out_dir), flush=True)
    for exp, m, npz in zip(experiments, matches, members):
        print("    %-58s %-16s %s" % (exp, m, os.path.basename(npz)), flush=True)

    if getattr(args, "pool_check", False):
        return _run_gates(args, base_path, experiments, matches, members, out_dir, prior_mode)

    stem = "pooled_posterior_samples" + (("_" + prior_mode) if prior_mode else "")
    out_npz = os.path.join(out_dir, stem + ".npz")
    meta = pool_sample_dumps(
        members, out_npz,
        draws_per_member=getattr(args, "pool_draws", None),
        provenance={"experiments": list(experiments), "match_strings": matches,
                    "arm": getattr(args, "pool_arm", None), "data_tag": data_tag,
                    "prior_mode": prior_mode, "label": label},
    )

    prefix = os.path.join(out_dir, "pooled" + (("_" + prior_mode) if prior_mode else ""))
    scored = score_pooled_dump(out_npz, experiments, prefix, prior_mode=prior_mode)

    with open(os.path.join(out_dir, "pooled_provenance.json"), "w") as fh:
        json.dump({"pool": meta, "scored": scored}, fh, indent=2)
    print("[pool] done", flush=True)
    return {"pooled_npz": out_npz, "meta": meta, "scored": scored}


def _run_gates(args, base_path, experiments, matches, members, out_dir, prior_mode):
    """Run the gates on data whose answer is known."""
    from .pooling import gate_pool_of_one

    work = os.path.join(out_dir, "gates")
    os.makedirs(work, exist_ok=True)
    exp0, match0, npz0 = experiments[0], matches[0], members[0]
    prod_json = os.path.join(base_path, "checkpoints", exp0,
                             "ensemble_evaluation_results_%s.json" % match0)
    report = {"members": members, "experiments": list(experiments)}
    report["G1"] = gate_pool_of_one(npz0, exp0, work,
                                    production_json=prod_json if not prior_mode else None)

    path = os.path.join(work, "pool_check_report.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print("[pool-check] wrote %s -> %s" % (path, report["G1"].get("verdict")), flush=True)
    if report["G1"].get("verdict", "").startswith("FAIL"):
        raise SystemExit("pool gate FAILED -- do not use any pooled posterior from this revision")
    return report
