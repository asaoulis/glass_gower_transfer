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
    --obs-label L --obs-store NAME [--bake-arms sc8a1 a0_tagged] [--variants production|full]
    [--fidelity] [--exact-rng] [--jitter-floor] [--rng-seed N] [--column-map BASENAME] [--kind auto|sim|h5|fits]
  Builds  MODELS_ROOT/checkpoints/unblinding/<obs-store>/observation_<L>.h5 (+ provenance, fidelity JSON;
          under checkpoints/ so `run_remote.py fetch --exp unblinding --rel <obs-store>` can pull it) and
  bakes   DATASETS_ROOT/<obs-store>_<L>_<arm>/output_<id>_out0_rot0_0.h5 for each arm, so a sampling job
  can use ``--data-store <obs-store>_<L>_<arm> --data-tag <L>`` (scripts/sample_observation.py).
  With --fidelity the sibling ``output_*.h5`` of a SIM catalogue (same store, same block name) is the
  reference and the report is written next to the observation.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from src.observation.bake import ARM_BAKES

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


def unblinding_root() -> Path:
    """MODELS_ROOT/checkpoints/unblinding -- the fetchable home of every non-blind unblinding
    artefact (`fetch --exp unblinding --rel <subdir>`)."""
    return Path(models_root()) / "checkpoints" / "unblinding"


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
    g.add_argument("--bake-arms", nargs="+", default=["sc8a1", "a0_tagged"], choices=list(ARM_BAKES))
    g.add_argument("--variants", default="production", choices=["production", "full", "a1only"])
    g.add_argument("--fidelity", action="store_true")
    g.add_argument("--exact-rng", action="store_true")
    g.add_argument("--jitter-floor", action="store_true")
    g.add_argument("--rng-seed", type=int, default=20260908)
    g.add_argument("--m-bias-source", default="auto", choices=["auto", "given", "zero", "fiducial"])
    g.add_argument("--weights-mode", default="ignore", choices=["ignore", "lensfit"],
                   help="lensfit = weighted estimator (needs the protected weights patch on the cluster checkout)")



# ---------------------------------------------------------------------------------------------
# observe-strip: a CLEAN in-distribution control -- one held-out flagship mock, stripped of its
# truth, dropped into an observation store. GATE 0b's catalogue (glass_paired_bg1p0_cats, a
# pre-BGP GLASS mock at fixed b_g) is a legitimate known-OOD control for the Gower nla_m BGP
# cloud; Tier 1 / GATE 3b / GATE 4 also need a mock-as-real that IS in distribution.
# ---------------------------------------------------------------------------------------------
def add_strip_args(parser) -> None:
    g = parser.add_argument_group("observe-strip")
    g.add_argument("--strip-store", default=None, help="bare BAKED store (gpu5) to take the mock from")
    g.add_argument("--strip-root", default="gpu5", choices=["gpu4", "gpu5"])
    g.add_argument("--strip-index", type=int, default=None,
                   help="index into the SORTED list of candidate files (fixed-lock test cosmologies only); "
                        "default: a random draw made INSIDE the job and written only to the truthkey (an "
                        "explicit index is a name anyone can resolve to a Gower CSV row)")
    g.add_argument("--strip-lock", default="gower_test_ids",
                   help="fixed test lock (config/fixed_test_sets/<name>.json) restricting the candidates")
    g.add_argument("--strip-arm", default="sc8a1", help="which arm store the baked mock corresponds to")
    g.add_argument("--strip-raw-store", default=None,
                   help="bare RAW store carrying the same basename: copies cls (EE/BB) and re-bins bb_bandpowers")
    g.add_argument("--strip-raw-root", default="gpu4", choices=["gpu4", "gpu5"])


def run_obs_strip(args) -> int:
    """Copy one held-out mock into ``<obs-store>_<label>_<arm>/output_<obs_id>_out0_rot0_0.h5``
    with an EMPTY ``cosmo_dict`` and an ``observation/`` provenance group; the source basename,
    sim id and the stripped cosmo_dict go to the ``_truthkey/`` sidecar only. Stdout never names
    the chosen file."""
    import h5py
    from src.observation.bake import baked_filename, obs_id_for
    from src.observation.build import _git_rev, bb_bandpowers_from_cls
    from src.observation.geometry import Geometry
    from src.ml.data.data_selection import extract_cosmo_index

    label = _bare(args.obs_label, "--obs-label")
    obs_store = _bare(args.obs_store, "--obs-store")
    src_store = _bare(args.strip_store, "--strip-store")
    arm = _bare(args.strip_arm, "--strip-arm")
    src_dir = os.path.join(datasets_root(args.strip_root), src_store)
    files = sorted(glob.glob(os.path.join(src_dir, "output_*.h5")))
    if not files:
        raise SystemExit(f"no output_*.h5 under {src_dir}")
    lock_ids = None
    if args.strip_lock and args.strip_lock != "none":
        lock_path = REPO / "config" / "fixed_test_sets" / f"{_bare(args.strip_lock, '--strip-lock')}.json"
        with open(lock_path) as fh:
            lock = json.load(fh)
        ids = lock.get("sim_ids", lock) if isinstance(lock, dict) else lock
        lock_ids = {int(i) for i in ids}
        files = [f for f in files if extract_cosmo_index(f) in lock_ids]
    if not files:
        raise SystemExit("no candidate files after the lock filter")
    if args.strip_index is None:
        idx = int(np.random.default_rng().integers(len(files)))   # unseeded on purpose: not reproducible from the CLI
        how = "random"
    else:
        if not 0 <= args.strip_index < len(files):
            raise SystemExit(f"--strip-index out of range: {len(files)} candidates")
        idx, how = int(args.strip_index), "cli"
    src = files[idx]
    print(f"[observe-strip] label={label} source_store={src_store} arm={arm} candidates={len(files)} "
          f"(lock={args.strip_lock}) index={'<random, in truthkey>' if how == 'random' else idx}", flush=True)

    store_dir = os.path.join(datasets_root("gpu5"), f"{obs_store}_{label}_{arm}")
    os.makedirs(store_dir, exist_ok=True)
    dst = os.path.join(store_dir, baked_filename(label))
    truth = {"label": label, "source_store": src_store, "source_file": os.path.basename(src),
             "sim_id": int(extract_cosmo_index(src)), "candidate_index": idx, "index_source": how, "cosmo_dict": {}}
    prov = {"label": label, "kind": "stripped_mock", "arm": arm, "source_store": src_store,
            "git_rev": _git_rev(), "obs_id": int(obs_id_for(label)), "lock": args.strip_lock,
            "raw_store": args.strip_raw_store or ""}
    g = Geometry.production()
    tmp = dst + ".tmp"
    with h5py.File(src, "r") as fi, h5py.File(tmp, "w") as fo:
        for key in fi:
            if key == "cosmo_dict":
                continue
            fi.copy(key, fo)
        for a, v in fi.attrs.items():
            if a not in ("sim_id", "cosmo", "cosmology"):
                fo.attrs[a] = v
        if "cosmo_dict" in fi:
            truth["cosmo_dict"] = {k: (fi["cosmo_dict"][k][()].tolist() if hasattr(fi["cosmo_dict"][k][()], "tolist")
                                       else str(fi["cosmo_dict"][k][()])) for k in fi["cosmo_dict"]}
        fo.create_group("cosmo_dict")                       # EMPTY: the observation contract
        if args.strip_raw_store:
            raw = os.path.join(datasets_root(args.strip_raw_root), _bare(args.strip_raw_store, "--strip-raw-store"),
                               os.path.basename(src))
            with h5py.File(raw, "r") as fr:
                cls = np.asarray(fr["cls_results/full/cls"])          # (nbins, nbins, >=2, n_ell)
                bls = np.asarray(fr["cls_results/full/bandpower_ls"]) if "bandpower_ls" in fr["cls_results/full"] else None
            cr = fo.require_group("cls_results/full")
            if bls is not None and "bandpower_ls" not in cr:
                cr.create_dataset("bandpower_ls", data=bls)            # the bake drops it; Tier-1/notebook want it
            if "cls" in cr:
                del cr["cls"]
            cr.create_dataset("cls", data=cls[:, :, :2])
            cut = cls[:, :, :, g.lower_lscale:g.upper_lscale + 1]
            if "bb_bandpowers" in cr:
                del cr["bb_bandpowers"]
            cr.create_dataset("bb_bandpowers",
                              data=bb_bandpowers_from_cls(cut, g.nbins, g.lower_lscale, g.upper_lscale, g.nbands))
        og = fo.require_group("observation")   # a source that is itself an observation bake keeps its group
        for k, v in prov.items():
            og.attrs[k] = v
    os.replace(tmp, dst)

    out_dir = unblinding_root() / obs_store
    out_dir.mkdir(parents=True, exist_ok=True)
    tk_dir = out_dir / "_truthkey"
    tk_dir.mkdir(exist_ok=True)
    with open(tk_dir / f"observation_{label}_truthkey.json", "w") as fh:
        json.dump(truth, fh, indent=2, default=str)
    with open(out_dir / f"observation_{label}_provenance.json", "w") as fh:
        json.dump(prov, fh, indent=2)
    # a fetchable copy (DATASETS_ROOT is outside `fetch`'s reach; Tier 1 runs locally on this file)
    shutil.copyfile(dst, out_dir / f"observation_{label}_baked.h5")
    with open(out_dir / f"observation_{label}_stores.json", "w") as fh:
        json.dump({"label": label, "observation": None, "kind": "stripped_mock", "baked": {arm: dst},
                   "baked_copy": str(out_dir / f"observation_{label}_baked.h5"),
                   "data_store_names": {arm: f"{obs_store}_{label}_{arm}"}}, fh, indent=2)
    print(f"[observe-strip] wrote {dst} (+ fetchable copy, provenance, truthkey sidecar under {out_dir})", flush=True)
    return 0

def add_reference_args(parser) -> None:
    g = parser.add_argument_group("obs-reference")
    g.add_argument("--ref-store", default=None, help="bare BAKED store name (gpu5): bandpowers + E-map summaries")
    g.add_argument("--ref-raw-store", default=None, help="bare RAW store name: BB bandpowers from cls (+ sc8 E if --ref-raw-emap)")
    g.add_argument("--ref-raw-root", default="gpu4", choices=["gpu4", "gpu5"])
    g.add_argument("--ref-raw-variant", default="sc8_fwhm4_lmin56_lcut1400",
                   help="E_<variant>/noise_std_<variant> groups to replicate the bake from a RAW store")
    g.add_argument("--ref-raw-emap", action="store_true", help="also extract E-map summaries from the raw store")
    g.add_argument("--ref-max-files", type=int, default=None, help="cap on the baked store (whole cosmologies)")
    g.add_argument("--ref-max-raw-files", type=int, default=4000, help="cap on the raw store (whole cosmologies)")
    g.add_argument("--ref-out", default=None, help="bare output name under MODELS_ROOT/unblinding/reference/")
    g.add_argument("--ref-workers", type=int, default=16)


def run_obs_reference(args) -> int:
    """ONE pass producing the Tier-1 reference clouds: reference_baked.npz (bandpowers + emap over
    the baked store) and reference_raw.npz (BB bandpowers [+ emap] over a raw-store subset)."""
    import json
    from .reference import extract_reference
    out_name = _bare(args.ref_out, "--ref-out")
    out_dir = unblinding_root() / "reference" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    if args.ref_store:
        store = os.path.join(datasets_root("gpu5"), _bare(args.ref_store, "--ref-store"))
        paths = sorted(glob.glob(os.path.join(store, "output_*.h5")))
        print(f"[obs-reference] baked store {store}: {len(paths)} files", flush=True)
        summary["baked"] = extract_reference(paths, str(out_dir / "reference_baked.npz"), workers=args.ref_workers,
                                             max_files=args.ref_max_files, eb_variant=None, noise_norm=None,
                                             want=("bandpowers", "emap"))
        print(f"[obs-reference] baked: {summary['baked']}", flush=True)
    if args.ref_raw_store:
        store = os.path.join(datasets_root(args.ref_raw_root), _bare(args.ref_raw_store, "--ref-raw-store"))
        paths = sorted(glob.glob(os.path.join(store, "output_*.h5")))
        print(f"[obs-reference] raw store {store}: {len(paths)} files", flush=True)
        want = ("bandpowers", "bb") + (("emap",) if args.ref_raw_emap else ())
        summary["raw"] = extract_reference(paths, str(out_dir / "reference_raw.npz"), workers=args.ref_workers,
                                           max_files=args.ref_max_raw_files, eb_variant=args.ref_raw_variant,
                                           noise_norm="rand", want=want)
        print(f"[obs-reference] raw: {summary['raw']}", flush=True)
    with open(out_dir / "reference_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return 0


def add_score_args(parser) -> None:
    g = parser.add_argument_group("obs-score")
    g.add_argument("--score-store", default=None, help="bare BAKED sc8a1 observation store (gpu5), e.g. obs_A_sc8a1")
    g.add_argument("--score-label", default=None, help="opaque label; names the BLIND subdir")
    g.add_argument("--score-base", default="gower_npe_finetune_nla_m_bgp_z8_ens1", help="the NPE encoder pack")
    g.add_argument("--score-repeats", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    g.add_argument("--score-num-samples", type=int, default=4000)
    g.add_argument("--score-skip-summaries", action="store_true")
    g.add_argument("--score-skip-misspec", action="store_true")
    g.add_argument("--score-knn-k", type=int, default=10, help="k of the summary-space kNN (summaries.py default)")


BLIND_SUBDIR = "unblinding_blind"     # under checkpoints/<exp>/ -- denied to the agent by guard_blind



def _idtest_meanp_null(sdir: str, matches, *, k: int = 10) -> np.ndarray:
    """Null distribution of the encoder-averaged kNN p over the common ID held-out files."""
    from src.ml.eval.ood import TrainWhitener, empirical_pvalues, knn_scores
    per = {}
    for m in matches:
        ftr = os.path.join(sdir, "_train", f"summaries_{m}.npz")
        fid = os.path.join(sdir, "_idtest", f"summaries_{m}.npz")
        if not (os.path.exists(ftr) and os.path.exists(fid)):
            raise FileNotFoundError(f"missing _train/_idtest summaries for {m}")
        dtr, did = np.load(ftr), np.load(fid)
        wh = TrainWhitener.fit(np.asarray(dtr["z"], dtype=np.float64))
        null = knn_scores(wh(np.asarray(dtr["z"], dtype=np.float64)), wh(np.asarray(did["z"], dtype=np.float64)), k=k)
        p = empirical_pvalues(null, null)
        per[m] = dict(zip([str(x) for x in did["test_files"]], p))
    common = sorted(set.intersection(*(set(d) for d in per.values())))
    if not common:
        raise RuntimeError("no common ID held-out files across encoders")
    return np.mean(np.stack([[per[m][f] for f in common] for m in matches]), axis=0)


def run_obs_score(args) -> int:
    """Score ONE observation store through both Tier-2 detectors with the 5-encoder NPE pack:

      (i)  summary-space kNN: ``summaries.run_summary_extraction`` with the observation as an ad-hoc
           variate (``test_id_source='all'``) -> per-encoder knn p vs that encoder's ID null;
      (ii) cross-encoder KL: ``misspec.run_misspecification_eval`` over the 5 repeats (its
           test-id pool falls back to ALL on-disk ids since obs ids 9001+ are in no lock) ->
           misspec_repeat_disagreement (kl_score).

    Every per-observation intermediate (summary vectors z, posterior samples/moments) is an
    UNBLINDING ARTEFACT and is written under ``checkpoints/<base>/unblinding_blind/<label>/``.
    Only SCALARS (knn score/p per encoder, mean-p raw + recalibrated against the ID null, kl) go
    to ``MODELS_ROOT/unblinding/<store>/obs_score_<label>.json``.
    """
    import json
    import numpy as np
    from src.ml.eval.ood import empirical_pvalues
    store = _bare(args.score_store, "--score-store")
    label = _bare(args.score_label, "--score-label")
    base = _bare(args.score_base, "--score-base")
    patterns = os.path.join(datasets_root("gpu5"), store, "output_*.h5")
    variate = [{"name": f"obs_{label}", "patterns": patterns, "exclude_params": []}]
    ckpt_root = os.path.join(models_root(), "checkpoints", base)
    blind_root = os.path.join(BLIND_SUBDIR, label)
    reps = [int(r) for r in args.score_repeats]
    match_template = getattr(args, "score_match_template", None) or "ncosmo300_{r}"
    matches = [match_template.format(r=r) for r in reps]

    if not args.score_skip_summaries:
        from src.ml.eval.summaries import run_summary_extraction
        run_summary_extraction(base, repeat_indices=reps, variates=variate, test_id_source="all",
                               out_subdir=os.path.join(blind_root, "summaries"), run_ood=True)
    if not args.score_skip_misspec:
        from src.ml.eval.misspec import run_misspecification_eval
        run_misspecification_eval(base, repeat_indices=reps, variates=variate, num_samples=args.score_num_samples,
                                  out_subdir=os.path.join(blind_root, "misspec"), test_id_source="heldout")

    # ---- scalars only -------------------------------------------------------------------------
    out = {"label": label, "store": store, "base": base, "repeats": reps, "per_encoder": {}, "notes": []}
    sdir = os.path.join(ckpt_root, blind_root, "summaries")
    pv = []
    for m in matches:
        f = os.path.join(sdir, f"obs_{label}", f"summaries_{m}.npz")
        if os.path.exists(f):
            d = np.load(f)
            out["per_encoder"][m] = {k: float(d[k][0]) for k in ("ood_knn_score", "ood_knn_p", "ood_mahalanobis_score",
                                                                  "ood_mahalanobis_p") if k in d.files}
            pv.append(float(d["ood_knn_p"][0]))
    if pv:
        out["meanp_raw"] = float(np.mean(pv))
        # Recalibrate the mean-p statistic against the ID-test null of the same encoders. The
        # _idtest npz carries the held-out summary vectors z (not p-values), so the per-event
        # kNN p of every ID event is rebuilt here exactly as OODReference.fit did it (whitener on
        # the train cloud, kNN k=10 vs the train cloud, empirical p vs the ID null itself) and
        # averaged over encoders per test file (the fixed lock makes the ID split common).
        try:
            null_meanp = _idtest_meanp_null(sdir, matches, k=int(args.score_knn_k))
        except Exception as ex:  # recalibration is a bonus; the raw mean-p is the record
            null_meanp = None
            out["notes"].append(f"mean-p recalibration unavailable: {type(ex).__name__}: {ex}")
        if null_meanp is not None and null_meanp.size:
            # LOW mean-p = OOD, so the recalibrated p is the LEFT tail
            out["meanp_recalibrated"] = float(empirical_pvalues(-null_meanp, np.array([-out["meanp_raw"]]))[0])
            out["n_idtest_null"] = int(len(null_meanp))
    mdir = os.path.join(ckpt_root, blind_root, "misspec", f"obs_{label}")
    for f in glob.glob(os.path.join(mdir, "misspec_repeat_disagreement_*.npz")):
        d = np.load(f)
        if d["mu"].shape[0] == len(reps):
            out["kl"] = float(d["kl_score"][0])
            # Every parameter-subset variant of the SAME cross-encoder disagreement, as scalars.
            # Each is a disagreement MAGNITUDE, not a posterior location, so it carries the same
            # blind status the full KL already has (DECISIONS T2-4). Emitting them all here means
            # a later `--kl-params` switch never requires re-running a real-data observation
            # (DECISIONS T2-8). Purely additive: a failure never breaks the reading.
            try:
                from src.ml.eval.kl_subsets import (ALL_SCORES, PARAM_BLOCKS, kl_per_dim,
                                                    load_variate, subset_scores)
                from src.observation.checks.tier2 import _scaler_box
                names, lo, hi = _scaler_box(base)
                rec = load_variate(os.path.join(ckpt_root, blind_root), f"obs_{label}",
                                   match_template, reps, names, lo, hi)
                sc = subset_scores(rec, names)
                for sub in ALL_SCORES:
                    v = sc[sub]
                    out[f"kl_{sub}"] = float(v[0]) if np.isfinite(v[0]) else None
                out["kl_per_dim"] = {p: float(x) for p, x in zip(names, rec["per_dim"][0])}
                # The statistic the Tier-2 tables are built with. `kl` above stays the FULL
                # diagonal value so earlier obs_score jsons remain comparable; `kl_adopted` is
                # the one a bias-table lookup must use, and `kl_params` names it so a reading
                # can never be matched against a differently-built table (DECISIONS T2-8).
                from src.observation.checks.tier2 import adopted_kl_for
                adopted = adopted_kl_for(base)        # pack-specific: 2-pt uses the full KL
                out["kl_params"] = adopted
                out["kl_adopted"] = out.get(f"kl_{adopted}")
                out["kl_param_blocks"] = {b: sum(out["kl_per_dim"][p] for p in ps if p in out["kl_per_dim"])
                                          for b, ps in PARAM_BLOCKS.items()}
            except Exception as ex:      # a diagnostic must never take the reading down
                out["notes"].append(f"subset KLs unavailable: {type(ex).__name__}: {ex}")
    out["notes"].append("posterior moments/samples and summary vectors live under "
                        f"checkpoints/{base}/{blind_root}/ (BLIND); this file carries scalars only")
    res_dir = unblinding_root() / store
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / f"obs_score_{label}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[obs-score] {json.dumps({k: v for k, v in out.items() if k != 'notes'})}", flush=True)
    return 0


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
    cat = apply_weights(cat, mode=getattr(args, "weights_mode", "ignore"))
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

    out_dir = unblinding_root() / obs_store
    out_dir.mkdir(parents=True, exist_ok=True)
    obs = build_observation(cat, m_bias, out_path=str(out_dir / f"observation_{label}.h5"), label=label,
                            geometry=geometry, variants=variants, rng_seed=args.rng_seed, rng=rng_factory(),
                            extra_provenance={"cli": {k: v for k, v in vars(args).items() if v is not None}},
                            weights=cat.estimator_weights)

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
        # store name = <obs-store>_<label>_<arm>, the contract scripts/sample_observation.py assumes
        store_dir = os.path.join(datasets_root("gpu5"), f"{obs_store}_{label}_{arm}")
        try:
            baked[arm] = bake_observation(obs, store_dir, arm=arm, label=label, overwrite=True)
            print(f"[observe-build] baked {arm}: {baked[arm]}", flush=True)
        except (KeyError, RuntimeError) as ex:
            print(f"[observe-build] bake {arm} SKIPPED: {ex}", flush=True)
    with open(out_dir / f"observation_{label}_stores.json", "w") as fh:
        json.dump({"label": label, "observation": obs, "baked": baked,
                   "data_store_names": {arm: f"{obs_store}_{label}_{arm}" for arm in baked}}, fh, indent=2)
    return rc
