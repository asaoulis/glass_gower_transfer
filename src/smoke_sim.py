"""Reduced-cost smoke backend for the GLASS simulator.

prepare_smoke_backend returns the same {param_dict, shells, matter, cosmo} contract as
src.cosmology.sim_utils.prepare_glass_backend with a handful of shells, a low lmax and a
scaled-down n(z), with a fixed cosmology so no Gower prior CSV is needed.
"""
from __future__ import annotations


# Reduced-scale knobs read by the smoke branch of the master script. Keep lmax below 3 * nside.
SMOKE_CONFIG = {
    "nside": 256,
    "lmax": 512,
    "zmin": 0.0,
    "zmax": 2.0,           # full tomographic range so all 6 bins populate
    "dx": 700.0,           # Mpc/h -> ~7 shells (z_eff up to ~1.7)
    "n_los_chi": 100,      # line-of-sight integration grid (prod 1000)
    "lower_lscale": 56,
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
    """Real CAMB angular spectra for a few shells and log-normal matter fields at smoke scale.

    Uses the same glass calls as prepare_glass_backend, in-process. Returns the backend dict.
    With ia_prior_spec, the IA params are sampled from that spec instead of fixed_param_dict.
    """
    import glass
    from src.cosmology import parameters, camb_matter_power

    param_dict = dict(cfg["fixed_param_dict"])
    if ia_prior_spec is not None:
        from src.cosmology.sim_utils import sample_ia_params
        for _k in ("a_ia", "b_ia", "b_z", "b_src"):
            param_dict.pop(_k, None)
        param_dict.update(sample_ia_params(ia_prior_spec, rng))

    cosmo, pars = parameters.build_cosmology(param_dict)

    shells, glass_cls = camb_matter_power.get_camb_matter_cls(
        pars, cfg["lmax"], cfg["zmin"], cfg["zmax"], cfg["dx"]
    )

    glass_cls_disc = glass.discretized_cls(glass_cls, nside=cfg["nside"], lmax=cfg["lmax"], ncorr=1)
    fields = glass.lognormal_fields(shells)
    gls = glass.solve_gaussian_spectra(fields, glass_cls_disc)
    matter = list(glass.generate(fields, gls, cfg["nside"], ncorr=1, rng=rng))

    return {"param_dict": param_dict, "shells": shells, "matter": matter, "cosmo": cosmo}
