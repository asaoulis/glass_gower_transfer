"""Local CLI for the HOS sidecar bake (src/ml/data/hos.py). The cluster entry point is
`eval.py --mode hos-bake --data-store <gpu5 store>`; this wrapper bakes any local glob (fixtures, validation).

  python scripts/prebake_hos.py --src-glob '<baked store>/output_*.h5' --out-dir <dir> [--workers 16] [--no-ext] [--variant x]
Output layout: <out-dir>/<basename> (flat, like the bake) + manifest.json, hos_table.npz; <out-dir>_tar/sNN.tar.
"""
import argparse

from src.ml.data.hos import run_bake

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-glob", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--nshards", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-ext", action="store_true")
    ap.add_argument("--variant", choices=["r1ext", "x"], default="r1ext",
                    help="'x' also writes the hos_x superset group (src/ml/data/hos_x.py)")
    a = ap.parse_args()
    c = run_bake(a.src_glob, a.out_dir, workers=a.workers, shard=a.shard, nshards=a.nshards, limit=a.limit,
                 ext=not a.no_ext, variant=a.variant)
    raise SystemExit(0 if c["bad"] == 0 else 1)
