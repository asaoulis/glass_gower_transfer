"""Cluster-side entry points for the unblinding front end, dispatched from ``eval.py --mode ...``.

Why they live behind ``eval.py``: the gatekeeper's ``eval-submit`` / ``eval-cpu-submit`` pass
``--args`` tokens straight through, so every cluster-side need becomes an ``eval.py`` mode with
BARE NAMES mapped to paths in code (token charset ``[A-Za-z0-9][A-Za-z0-9_.-]*`` or
``--[a-z-]*``, no ``/ * =``). No gatekeeper change, no bootstrap re-run.

Roots (overridable by env for LOCAL tests):
  UNBLIND_DATASETS_ROOT   default /share/gpu5/asaoulis/transfer_datasets   (baked obs stores go here)
  UNBLIND_DATASETS_ROOT2  default /share/gpu4/asaoulis/transfer_datasets   (saved sim catalogues)
  UNBLIND_MODELS_ROOT     default config.base_path (/share/gpu5/asaoulis/transfer_models)

Mode ``observe-build``
    --catalogue-store NAME [--catalogue-root gpu4|gpu5] (--catalogue-index N | --catalogue-file BASENAME)
    --obs-label L --obs-store NAME [--bake-arms sc8a1 a1] [--variants production|full]
    [--fidelity] [--exact-rng] [--jitter-floor] [--rng-seed N] [--column-map BASENAME] [--kind auto|sim|h5|fits]
  Builds  MODELS_ROOT/unblinding/<obs-store>/observation_<L>.h5 (+ provenance, fidelity JSON) and
  bakes   DATASETS_ROOT/<obs-store>_<arm>/output_<id>_out0_rot0_0.h5 for each arm, so a sampling job
  can use ``--data-store <obs-store>_<arm> --data-tag <L>``.
  With --fidelity the sibling ``output_*.h5`` of a SIM catalogue (same store, same block name) is the
  reference and the report is written next to the observation.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def datasets_root(which: str = "gpu5") -> str:
    from src.ml.eval.misspec import _GPU4, _GPU5
    if which == "gpu4":
        return os.environ.get("UNBLIND_DATASETS_ROOT2", _GPU4)
    return os.environ.get("UNBLIND_DATASETS_ROOT", _GPU5)


def models_root() -> str:
    env = os.environ.get("UNBLIND_MODELS_ROOT")
    if env:
        return env
    from config.default import get_default_config
    return get_default_config().base_path


_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _bare(name: str, what: str) -> str:
    if not _NAME_RE.match(name or ""):
        raise SystemExit(f"{what} must be a bare name (got {name!r})")
    return name


def add_observe_args(parser) -> None:
    g = parser.add_argument_group("observe-build")
    g.add_argument("--catalogue-store", default=None, help="bare dataset dir holding catalogues/ (sim) or the raw file")
    g.add_argument("--catalogue-root", default="gpu4", choices=["gpu4", "gpu5"])
    g.add_argument("--catalogue-index", type=int, default=None, help="index into sorted catalogues/catalogue_*.h5")
    g.add_argument("--catalogue-file", default=None, help="basename of the catalogue file inside the store")
    g.add_argument("--column-map", default=None, help="basename of a JSON column map inside the store (external catalogues)")
    g.add_argument("--kind", default="auto", choices=["auto", "sim", "h5", "fits"])
    g.add_argument("--obs-label", default=None, help="opaque label (A/B/C)")
    g.add_argument("--obs-store", default=None, help="bare name prefix for the baked stores under DATASETS_ROOT")
    g.add_argument("--bake-arms", nargs="+", default=["sc8a1", "a1"])
    g.add_argument("--variants", default="production", choices=["production", "full", "a1only"])
    g.add_argument("--fidelity", action="store_true")
    g.add_argument("--exact-rng", action="store_true")
    g.add_argument("--jitter-floor", action="store_true")
    g.add_argument("--rng-seed", type=int, default=20260908)
    g.add_argument("--m-bias-source", default="auto", choices=["auto", "given", "zero"])


def _resolve_catalogue(args) -> tuple:
    store = _bare(args.catalogue_store, "--catalogue-store")
    root = datasets_root(args.catalogue_root)
    store_dir = os.path.join(root, store)
    if args.catalogue_file:
        path = os.path.join(store_dir, "catalogues", _bare(args.catalogue_file, "--catalogue-file"))
        if not os.path.exists(path):
            path = os.path.join(store_dir, _bare(args.catalogue_file, "--catalogue-file"))
    else:
        cats = sorted(glob.glob(os.path.join(store_dir, "catalogues", "catalogue_*.h5")))
        if not cats:
            raise SystemExit(f"no catalogues/catalogue_*.h5 under {store_dir}")
        idx = int(args.catalogue_index or 0)
        path = cats[idx]
    if not os.path.exists(path):
        raise SystemExit(f"catalogue not found: {path}")
    mock = None
    m = re.search(r"catalogue_(\d+_out\d+_rot\d+_\d+)\.h5$", os.path.basename(path))
    if m:
        cand = os.path.join(store_dir, f"output_{m.group(1)}.h5")
        mock = cand if os.path.exists(cand) else None
    cmap = None
    if args.column_map:
        cmap = os.path.join(store_dir, _bare(args.column_map, "--column-map"))
    return path, mock, cmap


def run_observe_build(args) -> int:
    from src.observation import (Geometry, MapVariants, apply_c_terms, apply_m_bias, apply_weights,
                                 bake_observation, build_observation, compare_observation_to_mock,
                                 load_catalogue)
    from src.observation.fidelity import format_report, jitter_floor
    from src.observation.geometry import master_postproc_rng

    label = _bare(args.obs_label, "--obs-label")
    obs_store = _bare(args.obs_store, "--obs-store")
    cat_path, mock_path, cmap_path = _resolve_catalogue(args)
    column_map = json.load(open(cmap_path)) if cmap_path else None
    print(f"[observe-build] catalogue={cat_path}\n[observe-build] sibling mock={mock_path}", flush=True)

    cat = load_catalogue(cat_path, column_map=column_map, kind=args.kind)
    cat = apply_weights(cat)
    m_bias = apply_m_bias(cat, source=args.m_bias_source)
    cat = apply_c_terms(cat)
    attrs = cat.provenance.get("sim_attrs", {})
    geometry = Geometry.from_catalogue_attrs(attrs) if attrs else Geometry.production()
    variants = MapVariants.named(args.variants)
    print(f"[observe-build] n_gal={cat.n_gal:,} per-bin={cat.counts_per_bin(geometry.nbins)} "
          f"geometry={geometry.name} variants={args.variants}", flush=True)

    def rng_factory():
        if args.exact_rng:
            r = master_postproc_rng(attrs)
            if r is None:
                raise SystemExit("--exact-rng needs a sim catalogue written under --rng-seed")
            return r
        return np.random.default_rng(int(args.rng_seed))

    out_dir = Path(models_root()) / "unblinding" / obs_store
    out_dir.mkdir(parents=True, exist_ok=True)
    obs = build_observation(cat, m_bias, out_path=str(out_dir / f"observation_{label}.h5"), label=label,
                            geometry=geometry, variants=variants, rng_seed=args.rng_seed, rng=rng_factory(),
                            extra_provenance={"cli": {k: v for k, v in vars(args).items() if v is not None}})

    rc = 0
    if args.fidelity:
        if mock_path is None:
            print("[observe-build] --fidelity requested but no sibling output_*.h5 found; skipping", flush=True)
        else:
            floor = None
            if args.jitter_floor:
                floor = jitter_floor(cat, m_bias, geometry=geometry, variants=variants, rng_factory=rng_factory,
                                     workdir=str(out_dir / f"_floor_{label}"))
            rep = compare_observation_to_mock(obs, mock_path, floor=floor)
            rep.update({"floor": floor, "mock": mock_path, "exact_rng": bool(args.exact_rng)})
            print(format_report(rep), flush=True)
            with open(out_dir / f"observation_{label}_fidelity.json", "w") as fh:
                json.dump(rep, fh, indent=2, default=str)
            rc = 0 if rep["pass"] else 2

    baked = {}
    for arm in args.bake_arms:
        store_dir = os.path.join(datasets_root("gpu5"), f"{obs_store}_{arm}")
        try:
            baked[arm] = bake_observation(obs, store_dir, arm=arm, label=label, overwrite=True)
            print(f"[observe-build] baked {arm}: {baked[arm]}", flush=True)
        except (KeyError, RuntimeError) as ex:
            print(f"[observe-build] bake {arm} SKIPPED: {ex}", flush=True)
    with open(out_dir / f"observation_{label}_stores.json", "w") as fh:
        json.dump({"label": label, "observation": obs, "baked": baked,
                   "data_store_names": {arm: f"{obs_store}_{arm}" for arm in baked}}, fh, indent=2)
    return rc
