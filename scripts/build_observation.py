#!/usr/bin/env python
"""Build an OBSERVATION HDF5 (the mock schema, no cosmology) from a galaxy catalogue.

    PYTHONPATH=. python scripts/build_observation.py --catalogue <cat.h5|cat.fits> \
        --out-dir <dir> --label A [--geometry auto|production|smoke] [--variants production|full|a1only] \
        [--column-map spec.json] [--m-bias-source auto|given|zero --m-bias m1,...,m6] \
        [--rng-seed N] [--fidelity-mock output_*.h5 [--exact-rng] [--jitter-floor]]

Outputs <out-dir>/observation_<label>.h5, observation_<label>_provenance.json and (with
--fidelity-mock) observation_<label>_fidelity.json. Never reads or writes a cosmology.

The three deferred real-data corrections are exposed as flags that currently accept only their
identity value (weights 'ignore', c-terms 'global_mean_only', m-bias auto/given/zero) so the
provenance records the choice explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from src.observation import (Geometry, MapVariants, apply_c_terms, apply_m_bias, apply_weights,  # noqa: E402
                             build_observation, compare_observation_to_mock, load_catalogue)
from src.observation.fidelity import format_report, jitter_floor  # noqa: E402
from src.observation.geometry import master_postproc_rng  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", required=True, help="opaque label (e.g. A/B/C); appears in the filename only")
    ap.add_argument("--kind", default="auto", choices=["auto", "sim", "h5", "fits"])
    ap.add_argument("--column-map", default=None, help="JSON file mapping ra/dec/e1/e2/zbin/... for external catalogues")
    ap.add_argument("--nbins", type=int, default=6)
    ap.add_argument("--geometry", default="auto", choices=["auto", "production", "smoke"],
                    help="auto = from a sim catalogue's attrs, else production")
    ap.add_argument("--variants", default="production", choices=["production", "full", "a1only"])
    ap.add_argument("--m-bias-source", default="auto", choices=["auto", "given", "zero", "fiducial"])
    ap.add_argument("--m-bias", default=None, help="comma-separated per-bin m (with --m-bias-source given)")
    ap.add_argument("--weights-mode", default="ignore", choices=["ignore", "lensfit"],
                    help="lensfit = pass the catalogue weights to the estimator (needs the protected weights patch)")
    ap.add_argument("--normalization", default="counts", choices=["counts", "mean"],
                    help="bandpower-branch normalisation; 'mean' (paper Eq. 11) is a cross-check and loads the KiDS mask")
    ap.add_argument("--mask-data-dir", default=None, help="data dir for load_kids_mask (required with --normalization mean)")
    ap.add_argument("--c-terms-mode", default="global_mean_only")
    ap.add_argument("--rng-seed", type=int, default=20260908)
    ap.add_argument("--fidelity-mock", default=None, help="stored output_*.h5 to compare against (P0 gate)")
    ap.add_argument("--exact-rng", action="store_true",
                    help="with --fidelity-mock on a SIM catalogue written under --rng-seed: reuse the mock's "
                         "own random-rotation stream so the shape-noise debias is bit-reproducible")
    ap.add_argument("--jitter-floor", action="store_true",
                    help="also measure the float32-storage floor (two extra builds) and gate against it")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    column_map = json.load(open(args.column_map)) if args.column_map else None
    cat = load_catalogue(args.catalogue, column_map=column_map, nbins=args.nbins, kind=args.kind)
    cat = apply_weights(cat, mode=args.weights_mode, nbins=args.nbins)
    m_given = None if args.m_bias is None else [float(x) for x in args.m_bias.split(",")]
    m_bias = apply_m_bias(cat, m_bias=m_given, source=args.m_bias_source)
    cat = apply_c_terms(cat, mode=args.c_terms_mode)

    attrs = cat.provenance.get("sim_attrs", {})
    if args.geometry == "auto":
        geometry = Geometry.from_catalogue_attrs(attrs) if attrs else Geometry.production()
    else:
        geometry = Geometry.named(args.geometry)
    variants = MapVariants.named(args.variants)

    def rng_factory():
        if args.exact_rng:
            r = master_postproc_rng(attrs)
            if r is None:
                raise SystemExit("--exact-rng needs a sim catalogue written under --rng-seed (attr rng_seed >= 0)")
            return r
        return np.random.default_rng(int(args.rng_seed))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask = None
    if args.normalization == "mean":
        if not args.mask_data_dir:
            raise SystemExit("--normalization mean needs --mask-data-dir (load_kids_mask)")
        from src.KiDS.simulation_config import load_kids_mask
        mask = load_kids_mask(args.mask_data_dir)
    out = build_observation(cat, m_bias, out_path=str(out_dir / f"observation_{args.label}.h5"),
                            label=args.label, geometry=geometry, variants=variants,
                            rng_seed=args.rng_seed, rng=rng_factory(), verbose=not args.quiet,
                            extra_provenance={"cli": vars(args)}, weights=cat.estimator_weights,
                            normalization=args.normalization, mask=mask)
    print(f"observation: {out}")

    if args.fidelity_mock:
        floor = None
        if args.jitter_floor:
            floor = jitter_floor(cat, m_bias, geometry=geometry, variants=variants, rng_factory=rng_factory,
                                 workdir=str(out_dir / f"_fidelity_floor_{args.label}"))
        rep = compare_observation_to_mock(out, args.fidelity_mock, floor=floor)
        rep["floor"] = floor
        rep["mock"] = os.path.abspath(args.fidelity_mock)
        rep["exact_rng"] = bool(args.exact_rng)
        print(format_report(rep))
        if floor:
            print("floor (half-ulp float32 jitter):")
            for k, v in floor.items():
                print(f"  {k:40s} {v:.3e}")
        with open(out_dir / f"observation_{args.label}_fidelity.json", "w") as fh:
            json.dump(rep, fh, indent=2, default=str)
        return 0 if rep["pass"] else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
