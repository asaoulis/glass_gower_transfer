"""Noise-texture-robust higher-order statistics (HOS) of the baked sc8a1 E maps + the per-mock sidecar bake.

Selected by task eval-and-viz/robust-hos-selection (RECOMMENDATIONS.md): the tomographic third moments
<dk_i dk_j dk_l>, i <= j <= l (56 per scale), are the only map statistics that add information beyond 2-pt AND are
insensitive to the real catalogue's noise texture (paired real-noise vs mock-noise skies; peaks / MFs / Betti / PH /
PDF / l1 / scattering all fail). Shape noise is sign-symmetric whatever its variance texture, so odd polynomial
moments see the texture only through its correlation with the signal; distinct-bin triples have no noise term.

Spec (R1, `hos/vector`, 112 numbers):
  input   the STORED map of a baked store (`pixelised_results/E/{north,south}`, (6, H, W), f16, already divided by
          the per-mock random-rotation noise meter: the sc8a1 convention) -> cast to float64 BEFORE smoothing
          (smoothing f16 destroys the moments).
  smooth  anisotropic plate-carree Gaussian on the full patch rectangle, extra FWHM sqrt(t^2 - 4^2) arcmin for
          nominal total t in (8, 16): sigma_pix = (s / PIX, s / (PIX cos dec_c)), PIX = nside-512 resolution,
          dec_c = the patch centre latitude; scipy gaussian_filter(mode="reflect", truncate=4.0).
  pool    footprint pixels (KiDS frac > 0.5 on the patch grid, config/hos_assets/, scripts/build_hos_mask.py),
          N + S concatenated; subtract the pooled mean per bin.
  moment  mean(D_i * D_j * D_l) over the pooled pixels, triples in lexicographic order i <= j <= l.
  names   s{t}/m3x3/b{i}b{j}b{l} (1-based bins), scales 8 then 16.
Superset (`hos_ext/moments`, (4, 77)): per scale t in (4, 8, 12, 16): m2 (6) + x2 (15, i<j) + m3x3 (56), names in
`hos_ext/names`; t = 4 is the stored map unsmoothed. Enables R2 (distinct triples), scale cuts and a map-level
2-pt control without a re-bake.
Identical (to summation order) to robust-hos-selection `hos_extract.battery` on meter-normalised maps.
"""
import hashlib
import json
import os
import re
import time

import numpy as np
from scipy.ndimage import gaussian_filter

from src.KiDS.simulation_config import named_patches

PIX = 6.870972823634812            # arcmin, hp.nside2resol(512)
SIDES = ("north", "south")
DEC_C = {s: float(named_patches[s][1]) for s in SIDES}
assert DEC_C == {"north": 0.0, "south": -31.0}, DEC_C
NB = 6
SCALES_R1 = (8, 16)
SCALES_EXT = (4, 8, 12, 16)
TRIPLES = [(i, j, l) for i in range(NB) for j in range(i, NB) for l in range(j, NB)]
PAIRS = [(i, j) for i in range(NB) for j in range(i + 1, NB)]
R1_NAMES = [f"s{t}/m3x3/b{i+1}b{j+1}b{l+1}" for t in SCALES_R1 for (i, j, l) in TRIPLES]
EXT_NAMES = ([f"m2/b{i+1}" for i in range(NB)] + [f"x2/b{i+1}b{j+1}" for i, j in PAIRS]
             + [f"m3x3/b{i+1}b{j+1}b{l+1}" for i, j, l in TRIPLES])
R2_INDEX = [k for k, (i, j, l) in enumerate(TRIPLES) if i < j < l]          # distinct-bin triples within one scale
MASK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "config", "hos_assets",
                         "kids_patch_mask_frac_nside512.npz")
FOOT_COUNTS = {"north": 42904, "south": 47602}
MAP_CONVENTION = "stored sc8a1 E (divided by noise_std_*/all at bake), float64 before smoothing"
_SIMID_RE = re.compile(r"output_(\d+)_")


def load_footprint(path=MASK_PATH):
    """{side: bool (H, W)} footprint (frac > 0.5) + the asset's sha256."""
    path = os.path.normpath(path)
    with open(path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    d = np.load(path)
    foot = {s: np.asarray(d[s]) > 0.5 for s in SIDES}
    got = {s: int(foot[s].sum()) for s in SIDES}
    assert got == FOOT_COUNTS, f"footprint pixel counts {got} != {FOOT_COUNTS}"
    return foot, sha


def extra_fwhm(t):
    return 0.0 if t == 4 else float(np.sqrt(t ** 2 - 16.0))


def smooth(x, fwhm_arcmin, side):
    """x (..., H, W) float64 -> anisotropic Gaussian smoothing on the plate-carree grid (reflect boundary)."""
    assert x.dtype == np.float64, f"smooth needs float64 input, got {x.dtype}"
    if fwhm_arcmin <= 0:
        return x
    s = fwhm_arcmin / (2 * np.sqrt(2 * np.log(2)))
    sig = (0,) * (x.ndim - 2) + (s / PIX, s / (PIX * np.cos(np.radians(DEC_C[side]))))
    return gaussian_filter(x, sig, mode="reflect", truncate=4.0)


def _pooled(maps, foot, t):
    """(NB, npix) footprint-mean-subtracted pooled N+S field at nominal scale t."""
    fw = extra_fwhm(t)
    D = np.concatenate([(smooth(maps[s], fw, s) if fw > 0 else maps[s])[:, foot[s]] for s in SIDES], axis=1)
    return D - D.mean(axis=1, keepdims=True)


def _third(D):
    return np.array([(D[i] * D[j] * D[l]).mean() for i, j, l in TRIPLES])


def third_moments(maps, foot, scales=SCALES_R1):
    """maps {side: (6, H, W) float64} -> (len(scales), 56) third moments."""
    return np.stack([_third(_pooled(maps, foot, t)) for t in scales])


def moments_ext(maps, foot, scales=SCALES_EXT):
    """(len(scales), 77): m2 (6) + x2 (15) + m3x3 (56) per scale."""
    rows = []
    for t in scales:
        D = _pooled(maps, foot, t)
        m2 = (D * D).mean(axis=1)
        x2 = np.array([(D[i] * D[j]).mean() for i, j in PAIRS])
        rows.append(np.concatenate([m2, x2, _third(D)]))
    return np.stack(rows)


def r1_from_ext(ext):
    """R1 vector (112) from the superset (4, 77): m3x3 at scales 8 and 16."""
    off = NB + len(PAIRS)
    return np.concatenate([ext[SCALES_EXT.index(t), off:] for t in SCALES_R1])


def read_stored_maps(f, egroup="E"):
    maps = {s: f["pixelised_results"][egroup][s][()].astype(np.float64) for s in SIDES}
    return maps


# ---------------------------------------------------------------- sidecar bake
def _git_rev():
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__)), timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def shard_of(path, nshards=20):
    m = _SIMID_RE.search(os.path.basename(path))
    return (int(m.group(1)) if m else 0) % nshards


def bake_one(src, dst, foot, mask_sha, ext=True, rev="unknown"):
    """Write the sidecar of one baked mock. Never drops a file: on a read error the sidecar carries a NaN vector
    and `hos/_error` (counted as 'bad'). Returns ('ok'|'skip'|'bad', dst, message)."""
    import h5py
    if os.path.exists(dst):
        return ("skip", dst, "")
    err = ""
    vec = np.full(len(R1_NAMES), np.nan)
    exm = np.full((len(SCALES_EXT), len(EXT_NAMES)), np.nan)
    bp, cosmo, noise = None, {}, {}
    try:
        with h5py.File(src, "r") as f:
            maps = read_stored_maps(f)
            for s in SIDES:
                assert maps[s].shape == (NB,) + foot[s].shape, f"{s} shape {maps[s].shape} vs mask {foot[s].shape}"
            if "cls_results" in f:
                bp = f["cls_results"]["full"]["mixed_bandpowers"][()]
            cosmo = {k: f["cosmo_dict"][k][()] for k in f["cosmo_dict"].keys()}
            for g in f["pixelised_results"].keys():
                if g.startswith("noise_std"):
                    noise[g] = {k: f["pixelised_results"][g][k][()] for k in f["pixelised_results"][g].keys()}
        if ext:
            exm = moments_ext(maps, foot)
            vec = r1_from_ext(exm)
        else:
            vec = third_moments(maps, foot).ravel()
    except Exception as ex:   # truncated / corrupt / missing group: keep the basename, flag it
        err = f"{type(ex).__name__}: {ex}"
    tmp = dst + ".tmp"
    with h5py.File(tmp, "w") as g:
        hg = g.create_group("hos")
        hg.create_dataset("vector", data=vec)
        hg.attrs["names"] = json.dumps(R1_NAMES)
        pv = hg.create_group("_provenance")
        pv.attrs.update(dict(mask_sha256=mask_sha, git_rev=rev, map_convention=MAP_CONVENTION,
                             scales=json.dumps(SCALES_R1), smoothing="anisotropic plate-carree Gaussian, extra FWHM "
                             "sqrt(t^2-16), reflect, truncate 4", source=os.path.basename(src)))
        if err:
            hg.create_dataset("_error", data=err, dtype=h5py.string_dtype(encoding="utf-8"))
        if ext:
            xg = g.create_group("hos_ext")
            xg.create_dataset("moments", data=exm)
            xg.attrs["names"] = json.dumps(EXT_NAMES)
            xg.attrs["scales"] = json.dumps(SCALES_EXT)
        if bp is not None:
            g.create_group("cls_results").create_group("full").create_dataset("mixed_bandpowers", data=bp)
        cg = g.create_group("cosmo_dict")
        for k, v in cosmo.items():
            cg.create_dataset(k, data=v)
        if noise:
            pr = g.create_group("pixelised_results")
            for gname, members in noise.items():
                ng = pr.create_group(gname)
                for k, v in members.items():
                    ng.create_dataset(k, data=v)
    os.replace(tmp, dst)
    return ("bad" if err else "ok", dst, err)


def _job(args):
    src, dst, ext, rev = args
    foot, sha = load_footprint()
    return bake_one(src, dst, foot, sha, ext=ext, rev=rev)


def run_bake(src_glob, out_dir, workers=16, shard=None, nshards=20, limit=None, ext=True, min_free_gb=20.0,
             archive=True, log=print):
    """Bake every file of `src_glob` into the FLAT dir `out_dir/<basename>` (resumable: skips existing). The flat
    layout mirrors the bake, so `collect_paths` (sorted full paths) orders files exactly like the bake and
    `split_by_cosmology` reproduces the frozen band's split. `shard` restricts to sim_id % nshards == shard.
    Writes manifest.json + hos_table.npz, and (archive=True) per-shard transfer archives
    `<out_dir>_tar/s{NN}.tar` (one `fetch` each; the flat dir itself is too large for one foreground fetch)."""
    import glob
    import shutil
    from concurrent.futures import ProcessPoolExecutor
    files = sorted(glob.glob(src_glob))
    if shard is not None:
        files = [f for f in files if shard_of(f, nshards) == shard]
    if limit:
        files = files[:int(limit)]
    os.makedirs(out_dir, exist_ok=True)
    rev = _git_rev()
    jobs = []
    for f in files:
        jobs.append((f, os.path.join(out_dir, os.path.basename(f)), ext, rev))
    log(f"[hos-bake] {len(jobs)} files from {src_glob} -> {out_dir} (workers {workers}, ext {ext}, rev {rev})")
    counts = {"ok": 0, "skip": 0, "bad": 0}
    bad = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (st, dst, msg) in enumerate(ex.map(_job, jobs, chunksize=8)):
            counts[st] += 1
            if st == "bad":
                bad.append((os.path.basename(dst), msg))
            if (i + 1) % 2000 == 0 or i + 1 == len(jobs):
                log(f"[hos-bake] {i+1}/{len(jobs)} {counts} {time.time() - t0:.0f}s")
                if shutil.disk_usage(out_dir).free / 1e9 < min_free_gb:
                    log(f"[hos-bake] STOPPED by disk guard (< {min_free_gb} GB free)")
                    break
    write_manifest(out_dir, src_glob, log=log)
    if archive:
        write_shard_archives(out_dir, nshards, log=log)
    log(f"[hos-bake] DONE ok={counts['ok']} skip={counts['skip']} bad={counts['bad']} total={len(jobs)}")
    for b in bad[:20]:
        log(f"[hos-bake] bad {b[0]}: {b[1]}")
    return counts


def write_shard_archives(out_dir, nshards=20, log=print):
    """`<out_dir>_tar/s{NN}.tar`: uncompressed tar of the sidecars with sim_id % nshards == NN (arcname = basename)
    plus manifest.json / hos_table.npz in s00. Rebuilt from scratch (atomic per archive)."""
    import glob
    import tarfile
    tdir = out_dir.rstrip("/") + "_tar"
    os.makedirs(tdir, exist_ok=True)
    fs = sorted(glob.glob(os.path.join(out_dir, "output_*.h5")))
    groups = {}
    for f in fs:
        groups.setdefault(shard_of(f, nshards), []).append(f)
    for k in range(nshards):
        dst = os.path.join(tdir, f"s{k:02d}.tar")
        tmp = dst + ".tmp"
        with tarfile.open(tmp, "w") as tf:
            for f in groups.get(k, []):
                tf.add(f, arcname=os.path.basename(f))
            if k == 0:
                for extra in ("manifest.json", "hos_table.npz"):
                    if os.path.exists(os.path.join(out_dir, extra)):
                        tf.add(os.path.join(out_dir, extra), arcname=extra)
        os.replace(tmp, dst)
    log(f"[hos-bake] archives: {nshards} shards in {tdir} ({len(fs)} sidecars)")


def write_manifest(out_dir, src_glob="", log=print):
    """manifest.json (counts, sim ids, mask hash, names) + hos_table.npz (basenames, R1 vectors, ext, key cosmo)."""
    import glob
    import h5py
    fs = sorted(glob.glob(os.path.join(out_dir, "output_*.h5")))
    keys = ("omega_m", "sigma_8", "w0", "h", "ns", "ombh2", "mnu", "a_ia", "b_ia")
    V, X, C, names, nbad = [], [], [], [], 0
    for f in fs:
        with h5py.File(f, "r") as g:
            V.append(g["hos/vector"][()])
            X.append(g["hos_ext/moments"][()] if "hos_ext" in g else np.full((len(SCALES_EXT), len(EXT_NAMES)), np.nan))
            C.append([float(g["cosmo_dict"][k][()]) if k in g["cosmo_dict"] else np.nan for k in keys])
            nbad += int("_error" in g["hos"])
        names.append(os.path.basename(f))
    _, sha = load_footprint()
    sims = sorted({int(m.group(1)) for n in names for m in [_SIMID_RE.search(n)] if m})
    man = dict(n_files=len(fs), n_bad=nbad, n_sim_ids=len(sims), src_glob=src_glob, mask_sha256=sha,
               names=R1_NAMES, ext_names=EXT_NAMES, ext_scales=list(SCALES_EXT), map_convention=MAP_CONVENTION)
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    np.savez(os.path.join(out_dir, "hos_table.npz"), basenames=np.array(names), hos=np.array(V), ext=np.array(X),
             cosmo=np.array(C), cosmo_keys=np.array(keys))
    log(f"[hos-bake] manifest: {len(fs)} sidecars, {nbad} bad, {len(sims)} sim ids")
