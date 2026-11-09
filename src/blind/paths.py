"""Locations of the blind store and the standardised store, each overridable by an environment variable."""
import os

BLIND_ROOT = os.environ.get("UNBLIND_BLIND_ROOT", "/data/alex/unblinding/blind_store")
STANDARDISED_ROOT = os.environ.get("UNBLIND_STANDARDISED_ROOT", "/data/alex/unblinding/standardised")
