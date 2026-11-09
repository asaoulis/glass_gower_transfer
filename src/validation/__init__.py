"""Theory-vs-empirical validation of the KiDS-Legacy mock pseudo-Cl bandpowers.

It compares the saved mixed_bandpowers with per-cosmology theory as the ratio empirical / theory
per (tomographic bin pair, bandpower). The CLI driver is python -m src.validation.run_validation.
"""

from . import config  # noqa: F401
