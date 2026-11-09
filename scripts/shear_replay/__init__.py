"""Offline shear-estimator replay harness.

It replays --save-catalogues catalogues through candidate shear-map estimators: cache.py builds
the pixel cache, estimators.py the shear map, replay.py the patches and bandpowers, and
discriminators.py the per-mock scalars. The protected downstream functions are imported, never re-implemented.
"""
