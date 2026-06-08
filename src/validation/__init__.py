"""Theory-vs-empirical validation of the KiDS-Legacy mock pseudo-Cl bandpowers.

This package productionises the ad-hoc utilities that previously lived in
``sims_analysis.ipynb`` and ``theory_tests_systematics.py``.  It loads the
bandpower-level results saved by the master simulator (``mixed_bandpowers``),
computes the corresponding per-cosmology *theory* bandpower vector, and compares
the two via the ratio ``empirical / theory`` per (tomographic-bin-pair, bandpower).

Public entry points:
    - :func:`src.validation.theory.compute_bandpower_theory_from_cosmo_vec`
    - :func:`src.validation.ratios.compute_ensemble_ratios`
    - :func:`src.validation.diagnostics.run_diagnostics`
    - the CLI driver ``python -m src.validation.run_validation``

Nothing here edits the protected physics in ``src/cosmology`` / ``src/KiDS`` — it
only *imports* those modules read-only to reproduce the theory recipe.
"""

from . import config  # noqa: F401
