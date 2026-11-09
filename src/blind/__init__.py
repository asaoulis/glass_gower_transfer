"""Blind-analysis containment for the real-data posteriors.

Every raw observation posterior lives under the blind store (UNBLIND_BLIND_ROOT), as
<label>/<experiment>/external_posterior_samples_<prior>_<match>.npz. src.blind.standardise is the
only reader and hands out standardised samples only. .claude/hooks/guard_blind.py is a tripwire on
the obvious channels, not a sandbox.
"""
import os

BLIND_ROOT = os.environ.get("UNBLIND_BLIND_ROOT", "/data/alex/unblinding/blind_store")
STANDARDISED_ROOT = os.environ.get("UNBLIND_STANDARDISED_ROOT", "/data/alex/unblinding/standardised")
