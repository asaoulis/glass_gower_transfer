"""Blind-analysis containment for the real-data posteriors.

Every raw observation posterior lives under the blind store (UNBLIND_BLIND_ROOT), as
<label>/<experiment>/external_posterior_samples_<prior>_<match>.npz. src.blind.standardise is the
only reader and hands out standardised samples only. The guard_blind hook is a tripwire on the
obvious channels, not a sandbox. The store locations live in src.blind.paths.
"""
