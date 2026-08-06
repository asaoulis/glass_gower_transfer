#!/usr/bin/env python
"""Pre-bake a compact ML training store from the raw glass mock h5 files.

⭐ THE STANDARD remote-training data-prep step: map training is I/O-bound, so pre-bake with this
tool, put the compact store on /share/gpu5 (the l40s node's LOCAL disk), then `train --gpu l40s`.
Reading maps NON-local over NFS (e.g. /share/gpu4) is ~4.5x slower (measured 30 vs 135 smp/s). See
glass CLAUDE.md → "Data → Fast out-of-core training" and .claude/cluster/README_cluster.md.

The raw mocks (~63 MB each, float64) carry 8 E/B smoothing variants x {north,south}, but a
training run reads only ONE E variant (north+south) + bandpowers + cosmo params. This tool
extracts just those, downcasts the maps to float16/float32, and writes them under the *bare*
`pixelised_results/E/{north,south}` group names (so the loader reads them with eb_map_variant=None).

It also DROPS truncated/corrupt source files (some downloads are partial) so downstream
dataloaders never choke on them.

Schemas (--format):
  * h5            : one small HDF5 per sample, bare groups, maps at --dtype (default).
  * h5_compressed : same, chunked + lzf|gzip compression (--compression) [Phase 1.5 proxy].

Usage:
  /data/alex/glass/env/bin/python scripts/prebake_maps.py \
      --src-glob '/data/alex/glass_mocks/*.h5' --out-dir /data/alex/glass_mocks_f16 \
      --eb-variant fwhm8 --dtype float16 [--limit-cosmos N] [--workers 8]

Shear-estimator arms (dual-normalisation stores only). Which arm a training run sees is decided
HERE -- by which prebaked store `data_patterns` points at -- not by the model config:

  A0_counts   --eb-variant <tag>                             (baseline; default, unchanged)
  A1_wht_rand --eb-variant <tag>      --noise-norm rand      (DEPLOYED)
  A3s8        --eb-variant sc8_<tag>
  A3s8_A1     --eb-variant sc8_<tag>  --noise-norm rand
  B1_selfstd  any of the above + the loader-side config knob eb_noise_norm='self'

The per-(variant, bin, patch) `noise_std*` scalar groups are always carried through verbatim, so
an arm can be re-baked later without touching the raw store.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np

_SIMID_RE = re.compile(r"output_(\d+)_")

# what a training sample needs
MAP_SIDES = ("north", "south")
BANDPOWER_PATH = ("cls_results", "full", "mixed_bandpowers")


def _simid(path):
    m = _SIMID_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


def _bake_one(args):
    src, out_dir, eb_variant, dtype, fmt, compression, clevel, keep_tag, noise_norm = args
    base = os.path.basename(src)
    dst = os.path.join(out_dir, base)
    if os.path.exists(dst):
        return ("skip", src, 0.0, 0)
    t0 = time.perf_counter()
    npdt = np.dtype(dtype)
    e_group = f"E_{eb_variant}" if eb_variant else "E"
    ns_group = f"noise_std_{eb_variant}" if eb_variant else "noise_std"
    try:
        with h5py.File(src, "r") as f:
            pix = f["pixelised_results"]
            # Noise-meter scalars (~288 B) for the A1 rescale. Carried through verbatim so the
            # estimator stays ablatable from the compact store; absent in pre-dual-norm mocks.
            noise_std = {}
            for g in pix.keys():
                if g.startswith("noise_std"):
                    noise_std[g] = {k: pix[g][k][()] for k in pix[g].keys()}
            maps = {}
            for side in MAP_SIDES:
                m = pix[e_group][side][()]
                if noise_norm != "none":
                    # A1_wht_rand: divide each tomographic bin by the std of its matched
                    # random-rotation noise map. Applied BEFORE the cast so float16 sees an
                    # O(1) map rather than the raw shear amplitude.
                    if ns_group not in noise_std:
                        raise KeyError(f"--noise-norm {noise_norm} needs {ns_group}")
                    key = "all" if noise_norm == "rand" else side
                    sd = np.asarray(noise_std[ns_group][key], dtype=np.float64)
                    if sd.shape[0] != m.shape[0] or not np.all(sd > 0):
                        raise ValueError(f"bad {ns_group}/{key}: shape={sd.shape} min={sd.min()}")
                    m = m.astype(np.float64) / sd[:, None, None]
                maps[side] = m.astype(npdt)
            bp = f["cls_results"]["full"]["mixed_bandpowers"][()].astype(np.float32)
            cosmo = {k: f["cosmo_dict"][k][()] for k in f["cosmo_dict"].keys()}
    except Exception as ex:  # truncated / corrupt / missing group
        return ("bad", f"{src}: {type(ex).__name__}", 0.0, 0)

    ck = dict(compression=compression, compression_opts=(clevel if compression == "gzip" else None)) \
        if fmt == "h5_compressed" and compression else {}
    tmp = dst + ".tmp"
    try:
        with h5py.File(tmp, "w") as g:
            # keep_tag preserves the source variant group name (E_<variant>) so a config with
            # eb_map_variant=<variant> reads it; otherwise write the bare 'E' group.
            out_egroup = (f"E_{eb_variant}" if (keep_tag and eb_variant) else "E")
            pe = g.create_group("pixelised_results").create_group(out_egroup)
            for side in MAP_SIDES:
                arr = maps[side]
                kw = dict(ck)
                if kw:
                    kw["chunks"] = (1,) + arr.shape[1:]  # chunk per tomo-bin channel
                pe.create_dataset(side, data=arr, **kw)
            for gname, members in noise_std.items():
                ng = g["pixelised_results"].create_group(gname)
                for k, v in members.items():
                    ng.create_dataset(k, data=v)
            if noise_norm != "none":
                g["pixelised_results"].create_dataset(
                    "prebake_noise_norm", data=f"{noise_norm}:{ns_group}",
                    dtype=h5py.string_dtype(encoding="utf-8"))
            cf = g.create_group("cls_results").create_group("full")
            cf.create_dataset("mixed_bandpowers", data=bp)
            cg = g.create_group("cosmo_dict")
            for k, v in cosmo.items():
                cg.create_dataset(k, data=v)
        os.replace(tmp, dst)
    except Exception as ex:
        if os.path.exists(tmp):
            os.remove(tmp)
        return ("bad", f"{src} (write): {type(ex).__name__}: {ex}", 0.0, 0)
    return ("ok", dst, time.perf_counter() - t0, os.path.getsize(dst))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-glob", default="/data/alex/glass_mocks/*.h5")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eb-variant", default="fwhm8", help="source E/B smoothing group tag")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32", "float64"])
    ap.add_argument("--format", default="h5", choices=["h5", "h5_compressed"])
    ap.add_argument("--compression", default=None, choices=[None, "lzf", "gzip"])
    ap.add_argument("--clevel", type=int, default=4, help="gzip level")
    ap.add_argument("--limit-cosmos", type=int, default=None, help="use only first N distinct sim ids")
    ap.add_argument("--keep-variant-tag", action="store_true",
                    help="write the tagged group E_<variant> (for smoke data matching a config "
                         "with eb_map_variant set) instead of the bare 'E' group")
    ap.add_argument("--noise-norm", default="none", choices=["none", "rand", "rand_patch"],
                    help="shear estimator arm. 'none' (default) = A0_counts, byte-identical to "
                         "the legacy output. 'rand' = A1_wht_rand: divide each tomographic bin by "
                         "noise_std_<variant>/all (both patches pooled). 'rand_patch' = the same "
                         "with the per-patch (north/south) std. Combine with "
                         "--eb-variant sc8_<tag> for the A3s8 / A3s8_A1 arms.")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    files = sorted(glob.glob(args.src_glob))
    if not files:
        print(f"no files match {args.src_glob}", file=sys.stderr)
        sys.exit(1)
    if args.limit_cosmos:
        keep = set(sorted({_simid(p) for p in files})[: args.limit_cosmos])
        files = [p for p in files if _simid(p) in keep]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[prebake] {len(files)} src files -> {args.out_dir}  dtype={args.dtype} "
          f"format={args.format} compression={args.compression}")

    work = [(p, args.out_dir, args.eb_variant, args.dtype, args.format,
             args.compression, args.clevel, args.keep_variant_tag, args.noise_norm)
            for p in files]
    t0 = time.perf_counter()
    ok = bad = skip = 0
    total_bytes = 0
    bad_list = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_bake_one, w) for w in work]
        for i, fut in enumerate(as_completed(futs), 1):
            status, msg, _dt, nbytes = fut.result()
            if status == "ok":
                ok += 1
                total_bytes += nbytes
            elif status == "skip":
                skip += 1
            else:
                bad += 1
                bad_list.append(msg)
            if i % 100 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}  ok={ok} skip={skip} bad={bad}  "
                      f"{(time.perf_counter()-t0):.0f}s", flush=True)

    print(f"[prebake] DONE ok={ok} skip={skip} bad={bad} in {(time.perf_counter()-t0):.0f}s")
    if total_bytes:
        print(f"[prebake] wrote {total_bytes/1e9:.2f} GB  ({total_bytes/max(ok,1)/1e6:.2f} MB/sample)")
    if bad_list:
        print(f"[prebake] {len(bad_list)} bad files (dropped):")
        for b in bad_list[:20]:
            print("   ", b)


if __name__ == "__main__":
    main()
