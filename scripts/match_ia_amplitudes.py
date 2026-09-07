#!/usr/bin/env python3
"""Solve for the IA parameters that give the SAME effective alignment amplitude in every IA model.

WHY
---
The production variate corners compare `nla_m` / `nla` / `nla_z` posteriors on mocks whose IA
nuisances were independent draws from wide, model-specific priors (`a_ia ~ U[4.48,7]` + `b_ia ~
U[0.28,0.6]` for nla_m, `a_ia ~ U[-6,6]` for nla, plus `b_z ~ N(-3.7,4.3)` for nla_z). Two arms'
panels therefore differ because the model differs AND because the realised alignment differs. Only
the first is the claim. This script picks, per model, the parameter values that put all three at the
SAME effective amplitude, so the remaining posterior differences are the model.

WHAT "EFFECTIVE AMPLITUDE" MEANS HERE
-------------------------------------
Every IA model in `src/cosmology/nla.py` multiplies the same physical prefactor

    f_prefactor(z) = -C1 * rho_cr * Omega_m / D(z)          (`nla.nla_amplitude` at a_ia = 1)

by a model-specific, dimensionless factor. Dividing it out leaves exactly that factor:

    A_eff,i(nla_m) = a_ia * f_red_i * (10^{M_i} / 10^{13.5})^{b_ia}
    A_eff,i(nla)   = a_ia
    A_eff,i(nla_z) = a_ia + b_z * (<a>_i / 0.769 - 1)

for tomographic bin i. Because the prefactor cancels, **the answer does not depend on cosmology or
redshift** -- the script asserts this by re-running at two different cosmologies.

The amplitudes are NOT re-derived here: they are read out of
`src/cosmology/systematics.py:NLASystematics.intrinsic_alignment_amplitude`, the very method the
simulator calls, evaluated at delta = 1. That file is PROTECTED and is only ever imported.

REDUCING 6 BINS TO 1 NUMBER
---------------------------
nla_m's factor varies across bins (through `f_red` and the halo mass) and nla_z's tilts with
redshift, so "the" amplitude of a model is a weighted mean over the 6 bins. Two reductions are
reported:

  * `n_arcmin2`-weighted -- weights each bin by its galaxy density (`src/KiDS/tomo.py:n_arcmin2`),
    i.e. "how many galaxies actually carry this alignment";
  * unweighted           -- the plain mean over bins.

⚠️ This weighting is an OFFLINE arithmetic choice for collapsing 6 numbers into 1. It changes
NOTHING in the simulator: the fiducial n(z) and `n_arcmin2` are used exactly as they are, and the
per-bin spread of each model survives into the mocks as a genuine feature of that model. The two
reductions are printed side by side precisely so the choice is auditable.

USAGE
-----
    python scripts/match_ia_amplitudes.py [--data-dir kids-legacy-sbi/data] [--reduction n_weighted]

Prints a markdown table and the `IA_PRIOR_OVERRIDES['matched']` block to paste into
`master_kids_legacy_simulator.py`.
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.cosmology.systematics import NLASystematics          # PROTECTED - imported, never edited
from src.KiDS.systematics import f_red, load_massdep_priors   # PROTECTED - imported, never edited
from src.KiDS.tomo import calculate_tomo_nz, n_arcmin2, nbins
from src.KiDS.simulation_config import n_los_chi, zmax, zmin

# Prior midpoints of the nla_m forward prior = the anchor point (IA_PRIOR_SPECS['nla_m']).
ANCHOR_A_IA = 0.5 * (4.48 + 7.0)     # 5.74
ANCHOR_B_IA = 0.5 * (0.28 + 0.6)     # 0.44
# nla_z is 2 free parameters against 1 constraint. We FIX b_z at the Wright et al. prior mean and
# solve for a_ia. b_z = 0 would also satisfy the constraint but collapses nla_z exactly onto plain
# nla (`nla_z_effective_amplitude` reduces to a_ia), destroying the redshift tilt the variate exists
# to test -- so the tilt is kept and the BIN-AVERAGED amplitude is what is matched.
NLA_Z_B_Z = -3.7
A_PIV = 0.769                        # nla.nla_z_effective_amplitude default


class _DuckCosmo:
    """Minimal stand-in for the CAMB-backed cosmology.

    `intrinsic_alignment_amplitude` only ever touches `.omega_m` and `.ef(z)` (through
    `linear_growth_factor`). The prefactor it contributes is common to all three models and cancels
    in every ratio -- `--check-cosmology-independence` proves that rather than assuming it.
    """

    def __init__(self, omega_m=0.315, h=0.674):
        self.omega_m = float(omega_m)
        self.h = float(h)

    def ef(self, z):
        z = np.asarray(z, dtype=float)
        return np.sqrt(self.omega_m * (1.0 + z) ** 3 + (1.0 - self.omega_m))


def _sys(model, cosmo, **ia):
    """An NLASystematics carrying only what `intrinsic_alignment_amplitude` reads.

    The dispatch key is `nla['model']` and `f_red` is passed IN the dict (not imported by the
    class) -- both exactly as `master_kids_legacy_simulator.py:994` assembles `ia_params`.
    """
    nla = dict(ia)
    nla["model"] = model
    nla.setdefault("f_red", f_red)
    return NLASystematics(shear_bias=None, nla=nla, cosmo=cosmo)


def per_bin_A_eff(cosmo, avg_a, log10_M_eff, z_eff=0.5):
    """-> {model: A_eff[nbins]}, with the shared prefactor divided out."""
    unit = _sys("nla", cosmo, a_ia=1.0).intrinsic_alignment_amplitude(z_eff, 0)
    if not np.isfinite(unit) or unit == 0:
        raise RuntimeError("degenerate NLA unit prefactor: %r" % unit)

    out = {}
    out["nla_m"] = np.array([
        _sys("nla_m", cosmo, a_ia=ANCHOR_A_IA, b_ia=ANCHOR_B_IA,
             log10_M_eff=log10_M_eff).intrinsic_alignment_amplitude(z_eff, i) / unit
        for i in range(nbins)])
    out["nla"] = np.array([
        _sys("nla", cosmo, a_ia=1.0).intrinsic_alignment_amplitude(z_eff, i) / unit
        for i in range(nbins)])                     # == 1 in every bin, by construction
    out["nla_z"] = np.array([
        _sys("nla_z", cosmo, a_ia=0.0, b_z=1.0,
             avg_a=avg_a).intrinsic_alignment_amplitude(z_eff, i) / unit
        for i in range(nbins)])                     # == (<a>_i/a_piv - 1), the b_z basis vector
    return out


def reduce_bins(x, weights=None):
    return float(np.average(np.asarray(x, dtype=float), weights=weights))


def solve(cosmo, avg_a, log10_M_eff, weights, z_eff=0.5):
    """-> (anchor, solved a_ia for nla, solved a_ia for nla_z, per-bin table)."""
    A = per_bin_A_eff(cosmo, avg_a, log10_M_eff, z_eff=z_eff)

    anchor = reduce_bins(A["nla_m"], weights)                 # target Abar
    a_ia_nla = anchor                                         # A_eff = a_ia, exactly
    # A_eff,i(nla_z) = a_ia + b_z * basis_i  =>  Abar = a_ia + b_z * <basis>
    basis_bar = reduce_bins(A["nla_z"], weights)
    a_ia_nla_z = anchor - NLA_Z_B_Z * basis_bar
    return anchor, a_ia_nla, a_ia_nla_z, A, basis_bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(REPO, "kids-legacy-sbi", "data"))
    ap.add_argument("--reduction", choices=["n_weighted", "unweighted"], default="n_weighted")
    ap.add_argument("--z-eff", type=float, default=0.5)
    args = ap.parse_args()

    # --- fiducial per-bin inputs, used exactly as the simulator has them -----------------------
    means, _cov = load_massdep_priors(args.data_dir)
    log10_M_eff = np.asarray(means, dtype=float)              # prior MEAN (the pin)
    los_z = np.linspace(zmin, zmax, n_los_chi)
    # shift_nz=False: the dz draw lives inside the `if shift_nz` branch, so this touches no RNG.
    tomo_nz = calculate_tomo_nz(args.data_dir, n_los_chi, los_z, shift_nz=False)
    a_of_z = 1.0 / (1.0 + los_z)
    avg_a = np.array([np.average(a_of_z, weights=tomo_nz[i]) for i in range(nbins)])

    print("Fiducial per-bin inputs (unmodified):")
    print("  f_red        %s" % np.array2string(np.asarray(f_red), precision=4))
    print("  log10_M_eff  %s   (massdep prior mean)" % np.array2string(log10_M_eff, precision=4))
    print("  <a>_i        %s" % np.array2string(avg_a, precision=5))
    print("  n_arcmin2    %s" % np.array2string(np.asarray(n_arcmin2), precision=4))

    weights = np.asarray(n_arcmin2) if args.reduction == "n_weighted" else None
    cosmo = _DuckCosmo()
    anchor, a_nla, a_nla_z, A, basis_bar = solve(cosmo, avg_a, log10_M_eff, weights, args.z_eff)

    # --- cosmology independence (the prefactor really does cancel) -----------------------------
    c2 = _DuckCosmo(omega_m=0.25, h=0.72)
    anchor2, a_nla2, a_nla_z2, _A2, _b2 = solve(c2, avg_a, log10_M_eff, weights, z_eff=1.3)
    drift = max(abs(anchor - anchor2), abs(a_nla - a_nla2), abs(a_nla_z - a_nla_z2))
    assert drift < 1e-10, "prefactor did not cancel: drift=%.3e" % drift
    print("\n  [check] cosmology/z independence: max drift %.2e  OK" % drift)

    # --- both reductions, side by side ---------------------------------------------------------
    print("\nReduction sensitivity (the offline 6->1 choice; nothing in the sim changes):")
    for tag, w in (("n_arcmin2-weighted", np.asarray(n_arcmin2)), ("unweighted", None)):
        an, an_nla, an_nlaz, _, _ = solve(cosmo, avg_a, log10_M_eff, w, args.z_eff)
        print("  %-20s  anchor Abar = %+.6f   a_ia(nla) = %+.6f   a_ia(nla_z) = %+.6f"
              % (tag, an, an_nla, an_nlaz))

    # --- the per-bin table ---------------------------------------------------------------------
    A_nla_z_solved = a_nla_z + NLA_Z_B_Z * A["nla_z"]
    A_nla_solved = np.full(nbins, a_nla)
    print("\n| bin | A_eff nla_m | A_eff nla | A_eff nla_z | n_arcmin2 |")
    print("|---|---|---|---|---|")
    for i in range(nbins):
        print("| %d | %+.5f | %+.5f | %+.5f | %.4f |"
              % (i + 1, A["nla_m"][i], A_nla_solved[i], A_nla_z_solved[i], n_arcmin2[i]))
    print("| **mean (%s)** | **%+.5f** | **%+.5f** | **%+.5f** | |"
          % (args.reduction,
             reduce_bins(A["nla_m"], weights), reduce_bins(A_nla_solved, weights),
             reduce_bins(A_nla_z_solved, weights)))

    # --- A2 verification gate ------------------------------------------------------------------
    bars = [reduce_bins(A["nla_m"], weights), reduce_bins(A_nla_solved, weights),
            reduce_bins(A_nla_z_solved, weights)]
    rel = max(abs(b - bars[0]) for b in bars) / max(abs(bars[0]), 1e-30)
    assert rel < 1e-10, "matched amplitudes disagree: rel=%.3e" % rel
    assert np.all(np.isfinite(A["nla_m"])), "non-finite nla_m amplitude"
    assert abs(a_nla) < 6.0 and abs(a_nla_z) < 6.0, \
        "solved a_ia outside the U[-6,6] forward prior: %.4f / %.4f" % (a_nla, a_nla_z)
    print("\n  [check] three models agree to %.2e relative  OK" % rel)
    print("  [check] solved a_ia inside the U[-6,6] forward prior  OK")

    # --- the block to paste --------------------------------------------------------------------
    print("\n--- IA_PRIOR_OVERRIDES['matched'] (paste into master_kids_legacy_simulator.py) ---")
    print("    'matched': {")
    print("        'nla_m': {'a_ia': ('uniform', %.6f, %.6f),\n"
          "                  'b_ia': ('uniform', %.6f, %.6f)},"
          % (ANCHOR_A_IA, ANCHOR_A_IA, ANCHOR_B_IA, ANCHOR_B_IA))
    print("        'nla':   {'a_ia': ('uniform', %.6f, %.6f)}," % (a_nla, a_nla))
    print("        'nla_z': {'a_ia': ('uniform', %.6f, %.6f),\n"
          "                  'b_z':  ('uniform', %.6f, %.6f)},"
          % (a_nla_z, a_nla_z, NLA_Z_B_Z, NLA_Z_B_Z))
    print("    },")
    return 0


if __name__ == "__main__":
    sys.exit(main())
