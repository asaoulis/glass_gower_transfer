"""Rebuild the cross-repeat KL file for a misspec variate whose repeats ran as SEPARATE jobs.

misspec.py only writes `misspec_repeat_disagreement_*` when all repeats run in one job. This
re-derives it offline from the per-repeat moments npz with the SAME function the job uses
(`misspec._compute_repeat_disagreement`; mu = posterior mean, var = std^2), as validated on the
field pack 2026-10-08 (rebuilt == stored: mu exact, var 7e-9, kl rel 8e-8).

    python rebuild_kl.py --dir <pack>/misspec/<variate> --matches ncosmo300_0 ... [--check]

--check: the dir already holds an in-job file; rebuild into a temp dir and compare instead.
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, "/home/alex/work/glass_gower_transfer")
from src.ml.eval.misspec import _compute_repeat_disagreement  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--matches", nargs="+", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    name = os.path.basename(os.path.normpath(a.dir))
    per = {}
    for m in a.matches:
        d = np.load(os.path.join(a.dir, f"misspec_posterior_moments_{m}.npz"))
        per[m] = {"test_files": [str(f) for f in d["test_files"]],
                  "mu": d["mean"].astype(np.float64), "var": d["std"].astype(np.float64) ** 2}
    tag = "_".join(sorted(a.matches))
    out = tempfile.mkdtemp() if a.check else a.dir
    pay = _compute_repeat_disagreement(name, per, out)
    if a.check:
        ref = np.load(os.path.join(a.dir, f"misspec_repeat_disagreement_{tag}.npz"))
        new = np.load(os.path.join(out, f"misspec_repeat_disagreement_{tag}.npz"))
        assert list(ref["test_files"]) == list(new["test_files"]), "test_files differ"
        rel = np.max(np.abs(new["kl_score"] - ref["kl_score"]) / np.maximum(np.abs(ref["kl_score"]), 1e-12))
        with open(os.path.join(a.dir, f"misspec_repeat_disagreement_{tag}.json")) as f:
            jr = json.load(f)
        print(f"[check] {name}: kl_mean in-job {jr['kl_mean']:.6f} rebuilt {pay['kl_mean']:.6f}; "
              f"max rel dev kl_score {rel:.2e}; n {jr['n_events']} vs {pay['n_events']}")


if __name__ == "__main__":
    main()
