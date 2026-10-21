"""Local hyperparameter / epoch sweep driver for the HOS hybrid (config/kids_legacy_hos.py).

  python scripts/hos_sweep.py run --spec <spec.json> [--parallel 5] [--cores-per 8] [--core-offset 0]
  python scripts/hos_sweep.py collect --spec <spec.json> [--out <csv>]
  python scripts/hos_sweep.py run-trial <trial.json>          (internal; one subprocess per trial)

Spec JSON:
  {"stage": "A1", "base": "glass" | "gower" | "band_side" | "glassx" | "gowerx",
   "groups": [<hos_x group keys>]             (glassx / gowerx only; per trial override "groups"),
   "common": {<top-level config overrides>}, "common_hos": {<HOS MLP map_kwargs overrides>},
   "common_mk": {<merged into model_kwargs>}, "common_opt": {<merged into optimizer_kwargs, e.g. weight_decay>},
   "glass_ckpt": "<checkpoint dir>"            (gower base only; per-repeat '_{r}' resolution as in production),
   "trials": [{"id": "h256_d3", "hos": {...}, "top": {...}, "mk": {...}, "opt": {...}}, ...]}
Each trial trains `hossw_<stage>_<id>` exactly like train.py (get_default_config + row + overrides, the
max_trainval_cosmos list loop, train_model) under $GLASS_MODELS_ROOT (default /data/alex/hos_models) with
WANDB_MODE=offline. Trials are resumable: a trial with a DONE marker is skipped. Each run dir gets a val_history.csv
(every validation: epoch, step, all val_* and train_* metrics) written by a callback added in the trial process.
`collect` writes one CSV row per (trial, run): best val_log_prob (= NLL, lower is better) + its epoch from the
best-checkpoint filename, the val curve summary from val_history.csv, wall time, and the trial's hyperparameters.
"""
import argparse
import copy
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.environ.get("GLASS_MODELS_ROOT", "/data/alex/hos_models")
PY = sys.executable


def _trial_name(stage, tid):
    return f"hossw_{stage}_{tid}"


def _logdir(stage):
    d = os.path.join(MODELS, "logs", "sweep", stage)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------- one trial (subprocess)
def run_trial(path):
    sys.path.insert(0, REPO)
    os.chdir(REPO)                                  # the fixed-test lock path is repo-relative
    import pytorch_lightning as pl

    class ValHistory(pl.Callback):
        def on_validation_end(self, trainer, module):
            if trainer.sanity_checking:
                return
            ck = [c for c in trainer.callbacks if isinstance(c, pl.callbacks.ModelCheckpoint)]
            d = ck[0].dirpath if ck else os.getcwd()
            os.makedirs(d, exist_ok=True)
            m = {k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith(("val_", "train_"))}
            row = {"epoch": trainer.current_epoch, "step": trainer.global_step, "time": time.time(), **m}
            f = os.path.join(d, "val_history.csv")
            new = not os.path.exists(f)
            with open(f, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)

    _init = pl.Trainer.__init__

    def _patched(self, *a, **kw):
        kw["callbacks"] = list(kw.get("callbacks") or []) + [ValHistory()]
        _init(self, *a, **kw)
    pl.Trainer.__init__ = _patched

    from config.default import get_default_config
    from config.kids_legacy_hos import hos_glass_row, hos_gower_row, band_side_row, hosx_glass_row, hosx_gower_row
    from src.ml.models.utils import train_model

    t = json.load(open(path))
    if t["base"] == "glass":
        row = hos_glass_row(hos_kwargs=t.get("hos"))
    elif t["base"] == "gower":
        row = hos_gower_row(glass_ckpt=t["glass_ckpt"], hos_kwargs=t.get("hos"))
    elif t["base"] == "band_side":
        row = band_side_row()
    elif t["base"] == "glassx":                     # hos_x store, feature set = union of hos_x group keys
        row = hosx_glass_row(t["groups"], hos_kwargs=t.get("hos"))
    elif t["base"] == "gowerx":
        row = hosx_gower_row(t["glass_ckpt"], t["groups"], hos_kwargs=t.get("hos"))
    else:
        raise ValueError(t["base"])
    row.update(copy.deepcopy(t.get("top") or {}))
    if t.get("mk"):                                 # merged INTO model_kwargs (e.g. band_dropout_p)
        row["model_kwargs"] = {**row["model_kwargs"], **t["mk"]}
    config = get_default_config()
    if t.get("opt"):                                # merged INTO the default optimizer_kwargs (e.g. weight_decay)
        row["optimizer_kwargs"] = {**dict(config.optimizer_kwargs), **t["opt"]}
    config.experiment_name = t["name"]
    for k, v in row.items():
        if k != "max_trainval_cosmos":
            setattr(config, k, v)
    max_tv = row.get("max_trainval_cosmos")
    t0 = time.time()
    if isinstance(max_tv, (list, tuple)):
        for n in max_tv:
            config.max_trainval_cosmos = int(n)
            train_model(config)
    else:
        if max_tv is not None:
            config.max_trainval_cosmos = int(max_tv)
        train_model(config)
    with open(os.path.join(MODELS, "checkpoints", t["name"], "DONE"), "w") as fh:
        json.dump({"wall_s": time.time() - t0, "trial": t}, fh)


# ---------------------------------------------------------------- scheduler
def _expand(spec):
    out = []
    for tr in spec["trials"]:
        hos = {**(spec.get("common_hos") or {}), **(tr.get("hos") or {})}
        top = {**(spec.get("common") or {}), **(tr.get("top") or {})}
        mk = {**(spec.get("common_mk") or {}), **(tr.get("mk") or {})}
        opt = {**(spec.get("common_opt") or {}), **(tr.get("opt") or {})}
        out.append(dict(name=_trial_name(spec["stage"], tr["id"]), id=tr["id"], stage=spec["stage"], base=spec["base"],
                        hos=hos or None, top=top, mk=mk or None, opt=opt or None,
                        glass_ckpt=tr.get("glass_ckpt", spec.get("glass_ckpt")),
                        **({"groups": tr.get("groups", spec.get("groups"))} if spec["base"] in ("glassx", "gowerx")
                           else {})))
    return out


def run(spec_path, parallel, cores_per, core_offset, cpu=False):
    spec = json.load(open(spec_path))
    trials = _expand(spec)
    todo = [t for t in trials if not os.path.exists(os.path.join(MODELS, "checkpoints", t["name"], "DONE"))]
    ld = _logdir(spec["stage"])
    print(f"[sweep {spec['stage']}] {len(trials)} trials, {len(todo)} to run, parallel {parallel}", flush=True)
    assert parallel * cores_per + core_offset <= 60, "local node cap: <= 60 cores"
    slots = {i: None for i in range(parallel)}
    queue = list(todo)
    env = {**os.environ, "GLASS_MODELS_ROOT": MODELS, "WANDB_MODE": "offline",
           "WANDB_DIR": os.path.join(MODELS, "wandb"), "OMP_NUM_THREADS": str(cores_per if cpu else min(4, cores_per)),
           "PYTHONPATH": REPO}
    if cpu:     # tiny models are GPU kernel-launch bound (~141 ms/step) but ~44 ms/step on 8 CPU threads
        env["CUDA_VISIBLE_DEVICES"] = ""
    while queue or any(slots.values()):
        for i, p in slots.items():
            if p is not None and p[0].poll() is not None:
                rc = p[0].returncode
                print(f"[sweep {spec['stage']}] done {p[1]} rc={rc} ({time.time() - p[2]:.0f}s)", flush=True)
                slots[i] = None
            if slots[i] is None and queue:
                t = queue.pop(0)
                tp = os.path.join(ld, f"{t['id']}.json")
                json.dump(t, open(tp, "w"), indent=1)
                c0 = core_offset + i * cores_per
                cmd = ["taskset", "-c", f"{c0}-{c0 + cores_per - 1}", PY, os.path.abspath(__file__), "run-trial", tp]
                log = open(os.path.join(ld, f"{t['id']}.log"), "w")
                slots[i] = (subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT), t["id"], time.time())
                print(f"[sweep {spec['stage']}] start {t['id']} on cores {c0}-{c0 + cores_per - 1}", flush=True)
        time.sleep(5)
    print(f"[sweep {spec['stage']}] ALL DONE", flush=True)


# ---------------------------------------------------------------- collector
_CK = re.compile(r"epoch=(\d+)-val_[a-z_]+=(-?[0-9.]+)\.ckpt$")


def collect(spec_path, out=None):
    spec = json.load(open(spec_path))
    rows = []
    for t in _expand(spec):
        d = os.path.join(MODELS, "checkpoints", t["name"])
        done = os.path.join(d, "DONE")
        wall = json.load(open(done))["wall_s"] if os.path.exists(done) else None
        for rd in sorted(glob.glob(os.path.join(d, "*/"))):
            cks = [(int(m.group(1)), float(m.group(2))) for f in os.listdir(rd) for m in [_CK.search(f)] if m]
            if not cks:
                continue
            be, bv = min(cks, key=lambda x: x[1])
            hist = os.path.join(rd, "val_history.csv")
            last_v = n_val = first_v = None
            if os.path.exists(hist):
                H = list(csv.DictReader(open(hist)))
                vals = [float(h["val_log_prob"]) for h in H if h.get("val_log_prob")]
                if vals:
                    n_val, last_v, first_v = len(vals), vals[-1], vals[0]
            rows.append(dict(stage=t["stage"], trial=t["id"], run=os.path.basename(rd.rstrip("/")), best_val=bv,
                             best_epoch=be, last_val=last_v, first_val=first_v, n_val=n_val, done=os.path.exists(done),
                             wall_s=wall, hos=json.dumps(t["hos"]), top=json.dumps(t["top"]), mk=json.dumps(t["mk"]),
                             opt=json.dumps(t["opt"])))
    out = out or os.path.join(MODELS, "logs", "sweep", f"sweep_{spec['stage']}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["stage"])
        w.writeheader()
        w.writerows(rows)
    print(f"[collect] {len(rows)} runs -> {out}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--spec", required=True); r.add_argument("--parallel", type=int, default=5)
    r.add_argument("--cores-per", type=int, default=8); r.add_argument("--core-offset", type=int, default=0)
    r.add_argument("--cpu", action="store_true", help="train on CPU (hide the GPU); OMP threads = cores-per")
    c = sub.add_parser("collect"); c.add_argument("--spec", required=True); c.add_argument("--out", default=None)
    t = sub.add_parser("run-trial"); t.add_argument("path")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.spec, a.parallel, a.cores_per, a.core_offset, cpu=a.cpu)
    elif a.cmd == "collect":
        collect(a.spec, a.out)
    else:
        run_trial(a.path)
