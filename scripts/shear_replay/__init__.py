"""Offline shear-estimator replay harness (task kids-preparation/improved-shear-processing).

Replays raw galaxy catalogues (written by ``master_kids_legacy_simulator.py --save-catalogues``)
through CANDIDATE shear-map estimators, without touching the protected physics in
``src/cosmology/`` — the protected filter/patch/Cl functions are *imported*, never re-implemented,
so the downstream of every candidate is byte-for-byte the production downstream.

Pipeline: catalogue.h5 --(cache.py)--> sparse per-pixel moment cache --(estimators.py)-->
normalised full-sky shear map --(replay.py)--> E/B patches + bandpowers --(discriminators.py)-->
per-mock scalars --> JSONL rows (sweep over candidates × paired b_g triplets).

Run everything with ``PYTHONNOUSERSITE=1 /data/alex/glass/env/bin/python`` (numba/numpy pin).
"""
