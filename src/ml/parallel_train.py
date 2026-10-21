"""Single-node CPU training waves: many small trainings at once on one CPU node (`eval.py --mode hos-train`).

A wave is a list of task dicts (config/kids_legacy_hos.py:HOS_TRAIN_WAVES):
  {"kind": "train" | "embed", "exp": <experiment>, "repeat": r, "group": <store tag>, "after": [<exp>...],
   "env": {...}, "sources": [...] (embed only)}
`train` runs `python train.py <exp>` with REPEAT_INDICES=r; `embed` runs `python train_embeddings.py <exp> <sources>`.
Each task gets its own disjoint slice of the job's CPU affinity (taskset), OMP/MKL threads = slice size, no GPU, and its
own log file. Scheduling: list order is priority; a task starts when a slot is free, every `after` task has finished
with rc 0 (a failed dependency skips it), and the first task of its `group` has built the store's preload cache (its
log shows `[H5CosmoDataset] preloaded`, or it exited) -- so N concurrent jobs do not all build the same cache.
"""
import json
import os
import subprocess
import sys
import time

_READY = "[H5CosmoDataset] preloaded"


def _cmd(t, py):
    if t["kind"] == "train":
        return [py, "train.py", t["exp"]]
    if t["kind"] == "embed":
        return [py, "train_embeddings.py", t["exp"], *t["sources"]]
    raise ValueError(f"unknown task kind {t['kind']!r}")


def _label(t):
    return t["exp"] if t["kind"] == "embed" else f"{t['exp']}_r{t['repeat']}"


def run_wave(tasks, *, repo, log_dir, parallel=None, cores_per=None, stagger_s=20.0, poll_s=10.0, log=print):
    """Run `tasks` on this node. Returns {label: rc} (rc None = skipped: a dependency failed)."""
    cpus = sorted(os.sched_getaffinity(0))
    if cores_per is None:
        cores_per = max(1, len(cpus) // max(1, parallel or len(tasks)))
    n_slots = min(parallel or len(cpus) // cores_per, len(cpus) // cores_per, len(tasks))
    if n_slots < 1:
        raise ValueError(f"{len(cpus)} cpus cannot fit one slot of {cores_per}")
    os.makedirs(log_dir, exist_ok=True)
    log(f"[ptrain] {len(tasks)} tasks, {n_slots} slots x {cores_per} cores of {len(cpus)}; logs {log_dir}")

    labels = [_label(t) for t in tasks]
    assert len(set(labels)) == len(labels), "duplicate task labels"
    by_exp = {}
    for t, lab in zip(tasks, labels):
        by_exp.setdefault(t["exp"], []).append(lab)
    rc, running, slots = {}, {}, [None] * n_slots
    leader = {}                                     # group -> label of the task that builds its preload cache
    pending = list(range(len(tasks)))
    last_start = 0.0

    def group_ready(t, lab):
        g = t.get("group")
        if g is None or g not in leader or leader[g] == lab:
            return True
        lead = leader[g]
        if lead in rc:
            return True
        try:
            with open(os.path.join(log_dir, f"{lead}.log"), errors="replace") as fh:
                return _READY in fh.read()
        except OSError:
            return False

    while pending or running:
        for lab, (p, s, t0, fh) in list(running.items()):
            if p.poll() is not None:
                fh.close()
                rc[lab] = p.returncode
                slots[s] = None
                del running[lab]
                log(f"[ptrain] done {lab} rc={p.returncode} ({(time.time() - t0) / 60:.1f} min)")
        for i in list(pending):
            t, lab = tasks[i], labels[i]
            deps = [d for e in t.get("after", []) for d in by_exp.get(e, [])]   # outside this wave = already done
            if any(rc.get(d, 0) != 0 for d in deps if d in rc):
                rc[lab] = None
                pending.remove(i)
                log(f"[ptrain] skip {lab}: dependency failed ({deps})")
                continue
            if any(d not in rc for d in deps) or not group_ready(t, lab):
                continue
            if None not in slots or time.time() - last_start < stagger_s:
                break
            s = slots.index(None)
            core = ",".join(str(c) for c in cpus[s * cores_per:(s + 1) * cores_per])
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(cores_per),
                   "MKL_NUM_THREADS": str(cores_per), "PYTHONPATH": repo, **t.get("env", {})}
            if t["kind"] == "train":
                env["REPEAT_INDICES"] = str(t["repeat"])
            fh = open(os.path.join(log_dir, f"{lab}.log"), "w")
            p = subprocess.Popen(["taskset", "-c", core, *_cmd(t, sys.executable)], cwd=repo, env=env,
                                 stdout=fh, stderr=subprocess.STDOUT)
            running[lab] = (p, s, time.time(), fh)
            slots[s] = lab
            leader.setdefault(t.get("group"), lab)
            pending.remove(i)
            last_start = time.time()
            log(f"[ptrain] start {lab} on cpus {core} env {t.get('env') or {}}")
        time.sleep(poll_s)
    with open(os.path.join(log_dir, "summary.json"), "w") as fh:
        json.dump(rc, fh, indent=1)
    log(f"[ptrain] ALL DONE: {sum(v == 0 for v in rc.values())}/{len(rc)} ok, "
        f"failed {[k for k, v in rc.items() if v not in (0, None)]}, skipped {[k for k, v in rc.items() if v is None]}")
    return rc
