"""Blind-analysis containment for the real-data posteriors.

Rules (mechanically enforced by ``.claude/hooks/guard_blind.py`` + the layout below):

  * Every RAW posterior of an observation (real data or a mock treated as real) lives under ONE
    directory, the BLIND STORE (``UNBLIND_BLIND_ROOT``, default ``/data/alex/unblinding/blind_store``),
    mirroring the cluster tree ``<label>/<experiment>/external_posterior_samples_<prior>_<match>.npz``.
    Cluster-side, the equivalent is ``checkpoints/<exp>/external/<label>/`` and
    ``checkpoints/<exp>/unblinding_blind/<label>/`` (obs-score intermediates).
  * ``src.blind.standardise`` is the ONLY reader. It returns/persists STANDARDISED samples only
    -- per-parameter ``(x - mean)/std`` in a frame it computes and never exposes -- so a location or
    scale can never be read back from what it hands out.
  * The truth-key sidecars of mock-as-real observations (``_truthkey/``) are denied as well.
  * The notebook's unstandardised-contour cell is gated by ``UNBLIND = False``; the agent never
    flips it, and executes the notebook only on mocks.

This is a tripwire on the obvious channels (Read/cat/python-open/np.load/h5py of the blind paths,
``ssh hypatia-glass view|logs`` of the cluster blind dirs), not a sandbox.
"""
import os

BLIND_ROOT = os.environ.get("UNBLIND_BLIND_ROOT", "/data/alex/unblinding/blind_store")
STANDARDISED_ROOT = os.environ.get("UNBLIND_STANDARDISED_ROOT", "/data/alex/unblinding/standardised")
