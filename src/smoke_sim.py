"""Reduced-cost SMOKE backend for the KiDS-Legacy GLASS simulator.

Produces the SAME ``{param_dict, shells, matter, cosmo}`` contract as
``src.cosmology.sim_utils.prepare_glass_backend`` but tiny and fast (~15 s of CAMB), so that
``master_kids_legacy_simulator.py --simulator-type smoke`` can exercise the *entire* real
dataset-generation pipeline (CAMB->GLASS cls -> lognormal matter shells -> kappa/gamma -> IA ->
galaxy sampling -> shear maps -> bandpowers + pixelised maps -> HDF5) locally in <60 s. It is a
pre-flight to catch import / wiring / function-call / schema breakage before a costly cluster
MPI run.

This is NOT a physically meaningful mock: nside 256, ~7 shells to z~2, low CAMB accuracy,
tiny n_eff, a single augmentation and a non-physical bandpower ell-range. Use the outputs for
plumbing validation ONLY.

The module lives OUTSIDE ``src/cosmology/`` (the protected forward model). It only *calls* the
protected physics (``parameters.build_cosmology``, ``camb_matter_power.get_camb_matter_cls``) and
the public ``glass`` API exactly as the production backend does -- it never edits them.
"""
from __future__ import annotations


# Reduced-scale knobs. ``nside``/``lmax``/the ell-range/``nside_out`` are consumed by the smoke
# branch in master_kids_legacy_simulator.py; the z-grid + n_eff drive the helpers below. Keep
# ``lmax == 2*nside`` and ``upper_lscale < lmax`` (the bandpower slice must fit the spectrum).
SMOKE_CONFIG = {
    "nside": 256,
    "lmax": 512,
    "zmin": 0.0,
    "zmax": 2.0,           # full tomographic range so all 6 bins populate
    "dx": 700.0,           # Mpc/h -> ~7 shells (z_eff up to ~1.7)
    "n_los_chi": 100,      # line-of-sight integration grid (prod 1000)
    "lower_lscale": 50,
    "upper_lscale": 480,   # < lmax=512
    "nbands": 8,
    "nside_out": 128,      # pixelised map output nside (prod 512)
    # The real n(z) integrates to ~1.5 gal/arcmin^2 per bin -> production counts. Scale it
    # right down for the smoke (this is the "very low n_eff" knob).
    "n_eff_scale": 1.0e-3,
    # Fixed cosmology + nuisance params (no Gower prior CSV dependency locally). Recorded
    # verbatim into the HDF5 cosmo_dict.
    "fixed_param_dict": {
        "sigma_8": 0.80,
        "omega_m": 0.30,
        "h": 0.6736,
        "a_ia": 5.0,
        "b_ia": 0.4,
    },
}


def prepare_smoke_backend(rng, cfg=SMOKE_CONFIG, ia_prior_spec=None):
    """Tiny, real CAMB->GLASS cls -> lognormal matter shells.

    Mirrors ``prepare_glass_backend`` (same glass calls) but at smoke scale and in-process
    (the protected subprocess wrapper hardcodes cluster paths). Returns the same contract::

        {"param_dict": dict, "shells": list[RadialWindow], "matter": list[np.ndarray], "cosmo"}

    ``ia_prior_spec`` (when given) makes the smoke sample the per-IA-model nuisance params from the
    same spec as production, so the smoke param_dict carries exactly that model's IA params (a_ia
    plus b_ia / b_z / b_src) instead of the NLA-M defaults baked into ``fixed_param_dict``.
    """
    import glass
    from src.cosmology import parameters, camb_matter_power

    param_dict = dict(cfg["fixed_param_dict"])
    if ia_prior_spec is not None:
        from src.cosmology.sim_utils import sample_ia_params
        for _k in ("a_ia", "b_ia", "b_z", "b_src"):
            param_dict.pop(_k, None)
        param_dict.update(sample_ia_params(ia_prior_spec, rng))

    # Real CAMB cosmology (tuned only via small lmax / few shells; build_cosmology is protected).
    cosmo, pars = parameters.build_cosmology(param_dict)

    # Real CAMB -> GLASS angular matter spectra for a handful of shells.
    shells, glass_cls = camb_matter_power.get_camb_matter_cls(
        pars, cfg["lmax"], cfg["zmin"], cfg["zmax"], cfg["dx"]
    )

    # Lognormal matter fields -- identical pipeline to the production GLASS backend.
    glass_cls_disc = glass.discretized_cls(glass_cls, nside=cfg["nside"], lmax=cfg["lmax"], ncorr=1)
    fields = glass.lognormal_fields(shells)
    gls = glass.solve_gaussian_spectra(fields, glass_cls_disc)
    matter = list(glass.generate(fields, gls, cfg["nside"], ncorr=1, rng=rng))

    return {"param_dict": param_dict, "shells": shells, "matter": matter, "cosmo": cosmo}
