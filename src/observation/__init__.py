"""Turn an EXTERNAL galaxy catalogue (real KiDS-Legacy data, a blinded copy of it, or a saved mock
catalogue) into the HDF5 *observation* the trained models consume.

Design invariant (shared with ``src/ml/eval/nle_external.py``):

    **The mock HDF5 schema IS the observation contract.**

An observation is a file with the same ``cls_results/`` + ``pixelised_results/`` groups a mock has
and an EMPTY ``cosmo_dict`` group. Nothing in this package ever reads, infers or writes a cosmology.

The package reproduces ``master_kids_legacy_simulator.py``'s per-mock observable-building sequence
by CALLING the protected physics functions (``src/cosmology/``) in the production order — it never
re-implements an estimator. ``scripts/shear_replay`` is an independent re-implementation used only
as a cross-check (see ``fidelity.py``).

Modules
-------
catalogue_io   external catalogue -> the simulator's structured-array contract (+ provenance), and
               the three DEFERRED treatment hooks (weights / m-bias / c-terms) as single calls.
build          the master call sequence -> ``observation_<label>.h5`` (+ JSON provenance sidecar).
fidelity       rel-RMS comparison of a built observation against a stored ``output_*.h5``.
geometry       the fixed survey/sim geometry (production vs smoke) and the map-variant presets.
"""
from .geometry import Geometry, MapVariants  # noqa: F401
from .catalogue_io import Catalogue, load_catalogue, apply_weights, apply_m_bias, apply_c_terms  # noqa: F401
from .build import build_observation  # noqa: F401
from .fidelity import compare_observation_to_mock, rel_rms  # noqa: F401
from .bake import bake_observation, baked_filename, obs_id_for, ARM_BAKES  # noqa: F401
