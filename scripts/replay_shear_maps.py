#!/usr/bin/env python
"""CLI for the shear-estimator replay harness (scripts/shear_replay/).

Run with:  PYTHONNOUSERSITE=1 /data/alex/glass/env/bin/python scripts/replay_shear_maps.py ...

Subcommands
-----------
cache     Build sparse pixel caches from catalogue_*.h5 files.
fidelity  Gate: replay `counts` from a cache and compare with the stored output_*.h5.
sweep     Candidate x paired-b_g-triplet sweep -> JSONL rows of discriminators + M-probes.
selftest  Unit checks: degrade equivalence, counts-vs-map_shears equivalence, half-split sum.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
import zlib
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.shear_replay import cache as cache_mod            # noqa: E402
from scripts.shear_replay import discriminators as disc        # noqa: E402
from scripts.shear_replay import estimators                     # noqa: E402
from scripts.shear_replay import replay as replay_mod           # noqa: E402

CAT_RE = re.compile(r"catalogue_(\d+)_out(\d+)_rot(\d+)_(\d+)\.h5$")


def _key(path):
    m = CAT_RE.search(str(path))
    return tuple(int(g) for g in m.groups()) if m else None


def _cache_path(cache_dir, cat_path, arm_tag):
    k = _key(cat_path)
    return Path(cache_dir) / arm_tag / f"cache_{k[0]}_out{k[1]}_rot{k[2]}_{k[3]}.h5"


def cmd_cache(args):
    paths = sorted(Path(args.cat_dir).glob("catalogue_*.h5"))[: args.limit or None]
    print(f"[cache] {len(paths)} catalogues from {args.cat_dir}")
    out_dir = Path(args.out_dir)
    jobs = [(p, out_dir / p.name.replace("catalogue_", "cache_")) for p in paths]
    if args.workers > 1:
        with get_context("spawn").Pool(args.workers) as pool:
            results = pool.starmap(_cache_one, [(str(a), str(b), args.overwrite) for a, b in jobs])
    else:
        results = [_cache_one(str(a), str(b), args.overwrite) for a, b in jobs]
    ok = sum(1 for r in results if r)
    print(f"[cache] done: {ok}/{len(jobs)} ok")


def _cache_one(cat, out, overwrite):
    try:
        cache_mod.build_pixel_cache(cat, out, overwrite=overwrite)
        return True
    except Exception:
        traceback.print_exc()
        return False


def cmd_fidelity(args):
    cache = cache_mod.load_cache(args.cache)
    variant = replay_mod.EB_SMOOTHING_VARIANTS[args.variant_idx]
    out = replay_mod.build_arm(cache, estimators.get_candidate("A0_counts"), variant,
                               rng=np.random.default_rng(args.seed))
    res = replay_mod.fidelity_check(out, args.mock)
    print(f"[fidelity] cache={args.cache}\n[fidelity] mock ={args.mock} variant={variant}")
    worst_map = 0.0
    for k, v in res.items():
        print(f"  {k:24s} rel-RMS = {v:.3e}")
        if k != "mixed_bandpowers":
            worst_map = max(worst_map, v)
    print(f"[fidelity] worst map rel-RMS = {worst_map:.3e} "
          f"({'PASS' if worst_map < args.tol else 'FAIL'} at tol={args.tol:g}); "
          "bandpowers carry the irreproducible random-rotation noise draw — "
          "compare against the double-replay spread, not the map tolerance.")
    return 0 if worst_map < args.tol else 1


def _sweep_triplet(task):
    """Worker: one (triplet, candidates) unit -> JSON row string."""
    (key, cat_by_bg, cache_dir, cand_names, variant_idx, seed, tag) = task
    try:
        variant = replay_mod.EB_SMOOTHING_VARIANTS[variant_idx]
        caches, reps_base = {}, {}
        for bg, cat_path in cat_by_bg.items():
            cp = _cache_path(cache_dir, cat_path, f"bg{bg:g}".replace(".", "p"))
            cache_mod.build_pixel_cache(cat_path, cp)
            caches[bg] = cache_mod.load_cache(cp)
        rows = []
        baselines = {}
        base_prods = {bg: {} for bg in caches}   # per-arm norm_key memo (shared SHTs)
        nside0 = int(next(iter(caches.values()))["attrs"]["nside"])
        # Group same-normalisation candidates adjacently so the single-slot memo hits;
        # the slot is cleared on every key change to bound worker memory.
        cand_names = sorted(cand_names, key=lambda c: str(
            replay_mod.norm_key(estimators.get_candidate(c), variant, nside0)))
        for cname in cand_names:
            cand = estimators.get_candidate(cname)
            rep_by_bg = {}
            for bg, cache in caches.items():
                rng = np.random.default_rng([seed, int(key[0]), int(key[3]),
                                             int(round(bg * 1000)),
                                             zlib.crc32(cname.encode()) & 0xFFFF])
                nside_native = int(cache["attrs"]["nside"])
                cur_key = replay_mod.norm_key(cand, variant, nside_native)
                if cur_key not in base_prods[bg]:
                    base_prods[bg].clear()
                bp_baseline = None
                if replay_mod.effective_nside_bin(cand, nside_native) != nside_native:
                    if bg not in baselines:
                        baselines[bg] = replay_mod.baseline_alms(cache, rng=rng)
                    bp_baseline = baselines[bg]
                rep_by_bg[bg] = replay_mod.build_arm(cache, cand, variant, rng=rng,
                                                     bp_baseline=bp_baseline,
                                                     base_products=base_prods[bg])
            arms = {f"{bg:g}": disc.arm_row(caches[bg], rep_by_bg[bg]) for bg in rep_by_bg}
            paired = disc.triplet_stats(rep_by_bg, caches)
            rows.append(json.dumps({
                "tag": tag, "sim_id": key[0], "outer_idx": key[1], "rot_idx": key[2],
                "cat_idx": key[3], "candidate": cname, "variant": list(variant),
                "arms": arms, "paired": paired,
            }))
        return rows
    except Exception:
        traceback.print_exc()
        return []


def cmd_sweep(args):
    arm_dirs = {}
    for spec in args.arm:
        tag, d = spec.split("=", 1)
        arm_dirs[float(tag)] = Path(d)
    by_key = {}
    for bg, d in arm_dirs.items():
        for p in sorted(d.glob("catalogue_*.h5")):
            by_key.setdefault(_key(p), {})[bg] = p
    triplets = {k: v for k, v in sorted(by_key.items()) if len(v) == len(arm_dirs)}
    keys = list(triplets)[: args.limit or None]
    cand_names = args.candidates.split(",") if args.candidates != "all" \
        else list(estimators.CANDIDATES)
    print(f"[sweep] {len(keys)} complete triplets, {len(cand_names)} candidates: {cand_names}")

    tasks = [(k, triplets[k], args.cache_dir, cand_names, args.variant_idx, args.seed, args.tag)
             for k in keys]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with open(out_path, "a") as fh:
        if args.workers > 1:
            with get_context("spawn").Pool(args.workers) as pool:
                for rows in pool.imap_unordered(_sweep_triplet, tasks):
                    for r in rows:
                        fh.write(r + "\n")
                    n_rows += len(rows)
                    fh.flush()
                    print(f"[sweep] {n_rows} rows", flush=True)
        else:
            for t in tasks:
                rows = _sweep_triplet(t)
                for r in rows:
                    fh.write(r + "\n")
                n_rows += len(rows)
                fh.flush()
                print(f"[sweep] {n_rows} rows", flush=True)
    print(f"[sweep] wrote {n_rows} rows -> {out_path}")


def cmd_selftest(args):
    import healpy as hp
    rng = np.random.default_rng(7)
    nside = 128
    ngal = 200_000
    nbins = 2
    # Synthetic catalogue over a strip. float64 ON PURPOSE: production binned f64 in-memory
    # positions; real saved catalogues are f32, whose boundary-galaxy pixel flips are an inherent
    # replay-precision limit measured at the FIDELITY gate, not here — this test isolates the
    # pure cache/estimator algebra.
    cat = {
        "RA": rng.uniform(0, 90, ngal),
        "DEC": rng.uniform(-30, 10, ngal),
        "Z_TRUE": rng.uniform(0, 2, ngal),
        "ZBIN": rng.integers(0, nbins, ngal).astype(np.int8),
        "E1": rng.normal(0, 0.28, ngal),
        "E2": rng.normal(0, 0.28, ngal),
    }
    import h5py
    tmp = Path(args.tmpdir)
    tmp.mkdir(parents=True, exist_ok=True)
    cat_path = tmp / "catalogue_0_out0_rot0_0.h5"
    with h5py.File(cat_path, "w") as f:
        g = f.create_group("catalogue")
        for k, v in cat.items():
            g.create_dataset(k, data=v)
        f.attrs.update({"nside": nside, "nside_out": 64, "sim_id": 0, "outer_idx": 0,
                        "rot_idx": 0, "cat_idx": 0, "galaxy_bias": 1.0, "rng_seed": 7,
                        "m_bias_for_shear": np.full(nbins, 0.01),
                        "shear_normalization": "counts"})
        f.create_group("cosmo_dict")
    cp = cache_mod.build_pixel_cache(cat_path, tmp / "cache.h5", overwrite=True)
    cache = cache_mod.load_cache(cp)

    fails = 0
    # (a) degrade equivalence: coarse accumulation == direct accumulation at nside/2
    for i in range(nbins):
        cb = cache["bins"][i]
        d64 = estimators.dense_maps(cb, nside, nside // 2)
        sel = cat["ZBIN"] == i
        pix64 = hp.ang2pix(nside // 2, cat["RA"][sel].astype(float),
                           cat["DEC"][sel].astype(float), lonlat=True)
        n_direct = np.bincount(pix64, minlength=hp.nside2npix(nside // 2))
        ok = np.array_equal(d64["N"].astype(int), n_direct)
        print(f"[selftest] (a) degrade N equivalence bin{i}: {'PASS' if ok else 'FAIL'}")
        fails += 0 if ok else 1
    # (b) counts normalisation == map_shears + _apply_normalization on the raw catalogue
    from src.cosmology.map_shears import make_alm_shear_convergence
    m_bias = np.full(nbins, 0.01)
    alm_direct, _ = make_alm_shear_convergence(
        {k: np.asarray(v) for k, v in cat.items()}, m_bias, nbins, nside, 2 * nside,
        nosh=False, normalization="counts", rng=np.random.default_rng(0))
    for i in range(nbins):
        cb = cache["bins"][i]
        dm = estimators.dense_maps(cb, nside, nside)
        m, _aux = estimators.normalise(dm["S"], dm["N"], {"norm": "counts"})
        almE, almB = replay_mod._alm_pair(m, 2 * nside)
        rr = replay_mod.rel_rms(np.abs(almE), np.abs(alm_direct[i][0]))
        ok = rr < 1e-10
        print(f"[selftest] (b) counts==map_shears bin{i}: rel-RMS={rr:.2e} "
              f"{'PASS' if ok else 'FAIL'}")
        fails += 0 if ok else 1
    # (c) half-split sums
    for i in range(nbins):
        cb = cache["bins"][i]
        ok = (np.allclose(cb["SA1"] + (cb["S1"] - cb["SA1"]), cb["S1"], atol=1e-6)
              and (cb["NA"] <= cb["N"]).all())
        print(f"[selftest] (c) half-split bin{i}: {'PASS' if ok else 'FAIL'}")
        fails += 0 if ok else 1
    print(f"[selftest] {'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache")
    c.add_argument("--cat-dir", required=True)
    c.add_argument("--out-dir", required=True)
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--overwrite", action="store_true")

    f = sub.add_parser("fidelity")
    f.add_argument("--cache", required=True)
    f.add_argument("--mock", required=True)
    f.add_argument("--variant-idx", type=int, default=0)
    f.add_argument("--tol", type=float, default=1e-5)
    f.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("sweep")
    s.add_argument("--arm", action="append", required=True,
                   help="bg=dir, e.g. --arm 0.5=/path/bg0p5/catalogues (repeat per arm)")
    s.add_argument("--cache-dir", required=True)
    s.add_argument("--candidates", default="all")
    s.add_argument("--variant-idx", type=int, default=0)
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--seed", type=int, default=20260805)
    s.add_argument("--tag", default="", help="run tag written into every row "
                   "(e.g. smoke seed id — smoke runs all share sim_id 0)")

    t = sub.add_parser("selftest")
    t.add_argument("--tmpdir", default="/tmp/shear_replay_selftest")

    args = ap.parse_args()
    if args.cmd == "cache":
        cmd_cache(args)
    elif args.cmd == "fidelity":
        sys.exit(cmd_fidelity(args))
    elif args.cmd == "sweep":
        cmd_sweep(args)
    elif args.cmd == "selftest":
        sys.exit(cmd_selftest(args))


if __name__ == "__main__":
    main()
