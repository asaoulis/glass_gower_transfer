"""Shim: the base experiment dict now lives in config/archive/legacy_experiments.py.

These are pre-BGP rows (bandpower-MLP era). They are kept importable so existing entry points and
old runs keep working; do not add to them. New experiments belong in config/kids_legacy_bgp.py.
"""
from config.archive.legacy_experiments import experiments  # noqa: F401
