"""CLI driver: theory-vs-empirical bandpower validation over a folder of mocks.

Run from the repo root (so ``src`` is importable), e.g. on the cluster:

    python -u -m src.validation.run_validation \
        --data-dir /share/gpu5/asaoulis/transfer_datasets/glass_theory_test_clean \
        --sim-type glass --out-dir <MODELS_ROOT>/validation/glass_theory_test_clean \
        --n-jobs 64 --mixing-matrix /share/gpu5/asaoulis/KiDS_Legacy_mixing_matrix_mask.npy

It loads ``mixed_bandpowers`` from every matching ``output_*.h5``, computes the matching
per-cosmology theory bandpowers, forms the empirical/theory ratios, writes the plot set
and ``report.{json,md}``, prints a concise summary, and exits with:
    0 = PASS (all bins <= ok)   3 = WARNINGS only   4 = at least one ERROR
    2 = could-not-run (no files / load failure).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from . import config as cfg
from . import diagnostics, plots
from .ratios import DEFAULT_NESTED_KEYS, compute_ensemble_ratios
from .theory import load_mixing_matrix


def _normalise_nonlinear(value: str) -> str:
    if value is None or value.lower() in ("none", "off", ""):
        return ""
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="folder of output_*.h5 mocks")
    p.add_argument("--out-dir", required=True, help="where to write plots + report")
    p.add_argument("--sim-type", default="glass", choices=["glass", "gower"],
                   help="informational metadata recorded in the report")
    p.add_argument("--include", nargs="*", default=None,
                   help="only files whose basename contains ANY of these substrings")
    p.add_argument("--exclude", nargs="*", default=None,
                   help="drop files whose basename contains ANY of these substrings")
    p.add_argument("--max-sims", type=int, default=None,
                   help="cap number of files (random subset, seeded)")
    p.add_argument("--n-jobs", type=int, default=os.cpu_count() or 8)
    p.add_argument("--mixing-matrix", default=cfg.MIXING_MATRIX_PATH,
                   help="path to the KiDS mixing-matrix .npy")
    p.add_argument("--no-mixing", action="store_true",
                   help="skip the mixing matrix (use raw EE theory; for legacy data)")
    p.add_argument("--empirical-key", default="mixed_bandpowers",
                   choices=["mixed_bandpowers", "bandpowers"],
                   help="which saved bandpower field to use as the empirical vector")
    p.add_argument("--nonlinear", default=cfg.NONLINEAR,
                   help="CAMB NonLinear mode for the splined theory (e.g. NonLinear_lens; "
                        "'none' to disable). Ignored by shell_projection mode.")
    p.add_argument("--theory-mode", default=cfg.THEORY_MODE,
                   choices=["shell_projection", "splined"],
                   help="theory recipe: shell_projection (Theory_B, matches the sim's "
                        "discrete multi-plane lensing; default) or splined (Theory_A, "
                        "continuous CAMB SplinedSourceWindow)")
    p.add_argument("--data-dir-nz", default=cfg.DATA_DIR,
                   help="data dir holding the tomographic n(z)")
    p.add_argument("--ratio-ylim", nargs=2, type=float, default=(0.7, 1.3),
                   help="y-limits for the ratio plots")
    p.add_argument("--thresholds", nargs=2, type=float, default=(cfg.OK_FRAC, cfg.WARN_FRAC),
                   metavar=("OK", "WARN"), help="OK and WARNING fractional thresholds")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    ok_frac, warn_frac = args.thresholds
    nonlinear = _normalise_nonlinear(args.nonlinear)

    mixing_matrix = None
    mixing_path = None
    if not args.no_mixing:
        mixing_path = args.mixing_matrix
        print(f"[validation] loading mixing matrix {mixing_path}")
        mixing_matrix = load_mixing_matrix(mixing_path, lmax=cfg.LMAX)

    eff_nl = ("NonLinear_both (shell mode)" if args.theory_mode == "shell_projection"
              else (nonlinear or "OFF"))
    print(f"[validation] computing ensemble ratios (n_jobs={args.n_jobs}, "
          f"theory_mode={args.theory_mode}, empirical_key={args.empirical_key}, "
          f"effective_nonlinear={eff_nl}, nz_dir={args.data_dir_nz})")
    try:
        ensemble = compute_ensemble_ratios(
            args.data_dir, DEFAULT_NESTED_KEYS, cosmo_params=None,
            include=args.include, exclude=args.exclude, mixing_matrix=mixing_matrix,
            nonlinear=nonlinear, theory_mode=args.theory_mode, data_dir=args.data_dir_nz,
            empirical_key=args.empirical_key,
            max_sims=args.max_sims, n_jobs=args.n_jobs,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[validation] CANNOT RUN: {e}")
        return 2

    os.makedirs(args.out_dir, exist_ok=True)

    # one example overlay (theory + empirical for the first loaded file)
    example_theory = example_empirical = None
    try:
        from src.ml.data.data_loading import unpack_data
        from .theory import compute_bandpower_theory_from_cosmo_vec
        from .ratios import load_numeric_cosmo_vec
        # cosmo_params=[] so unpack_data skips cosmo_dict (the NLA-era string ia_model would
        # otherwise raise); read the numeric cosmo vector robustly instead.
        data, _ = unpack_data(ensemble["files"][0], DEFAULT_NESTED_KEYS, [],
                              as_torch=False, return_names=True)
        cosmo_vec = load_numeric_cosmo_vec(ensemble["files"][0])
        example_empirical = np.asarray(data[args.empirical_key])
        example_theory = compute_bandpower_theory_from_cosmo_vec(
            cosmo_vec, mixing_matrix=mixing_matrix, nonlinear=nonlinear,
            theory_mode=args.theory_mode, data_dir=args.data_dir_nz)
    except Exception as e:  # noqa: BLE001 - the example overlay is best-effort
        print(f"[validation] (skipping loglog example: {e})")

    written = plots.save_all_plots(
        ensemble, args.out_dir, ratio_ylim=tuple(args.ratio_ylim),
        ok_frac=ok_frac, warn_frac=warn_frac,
        example_theory=example_theory, example_empirical=example_empirical,
    )
    print(f"[validation] wrote {len(written)} plots to {args.out_dir}")

    caveat = None
    if args.no_mixing:
        caveat = ("NO MIXING MATRIX APPLIED (--no-mixing): empirical mixed_bandpowers are "
                  "mask-suppressed (~f_sky), so ratios collapse toward f_sky and the "
                  "verdict is NOT a physical theory test. Use --mixing-matrix for the real run.")
    # Record the EFFECTIVE non-linear model actually used by the active theory_mode.
    # shell_projection inherits NonLinear_both from the sim's get_camb_matter_cls and IGNORES the
    # --nonlinear knob (which only applies to the splined path); reporting the raw knob alone was
    # misleading (precision-logbook H9).
    effective_nonlinear = (
        "NonLinear_both (via get_camb_matter_cls; shell mode ignores --nonlinear)"
        if args.theory_mode == "shell_projection" else (nonlinear or "OFF")
    )
    meta = {
        "data_dir": args.data_dir, "sim_type": args.sim_type,
        "mixing_matrix": mixing_path, "nonlinear": nonlinear,
        "effective_nonlinear": effective_nonlinear,
        "theory_mode": args.theory_mode,
        "empirical_key": args.empirical_key, "n_failed": ensemble["n_failed"],
        "include": args.include, "exclude": args.exclude, "caveat": caveat,
    }
    result = diagnostics.run_diagnostics(
        ensemble, args.out_dir, meta, ok_frac=ok_frac, warn_frac=warn_frac)

    counts = result["counts"]
    print(f"\n[validation] {result['verdict']}: "
          f"OK={counts[cfg.STATUS_OK]} WARNING={counts[cfg.STATUS_WARNING]} "
          f"ERROR={counts[cfg.STATUS_ERROR]}  "
          f"(loaded {ensemble['n_loaded']}, failed {ensemble['n_failed']})")
    print(f"[validation] report: {result['md']}")
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
