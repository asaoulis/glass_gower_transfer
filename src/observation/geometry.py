"""Survey / simulation geometry and map-variant presets for observation building.

Two sources of truth are mirrored here and CHECKED against, never copied silently:
  * ``src/KiDS/simulation_config.py`` (production nside, ell range, bands, patches);
  * ``master_kids_legacy_simulator.py`` (the A3s8 smoothed-counts branch constants and the
    ``patch_noise_std`` / ``eb_variant_tag`` helpers).

The master module imports cleanly in a plain (non-MPI) process (~4 s, verified 2026-09-08), so the
helpers are IMPORTED from it. The small constants are also vendored below so that a process that
cannot import the master (e.g. a stripped eval node) still works; ``check_master_parity()`` asserts
the vendored copies match the master's and is run by the P0 gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------------------------------
# Vendored constants (mirrors of master_kids_legacy_simulator.py; parity-checked)
# ----------------------------------------------------------------------------------------------
A3S8_FWHM_ARCMIN = 8.0                                  # smoothed-counts denominator FWHM
A3S8_VARIANTS: Tuple[Tuple[float, int, int], ...] = ((4.0, 56, 1400), (8.0, 56, 1024))
A3S8_MAP_DTYPE = np.float16
EB_SMOOTHING_VARIANTS_PROD: Tuple[Tuple[float, int, int], ...] = ()   # 'sc8only' preset
EB_SMOOTHING_VARIANTS_FULL: Tuple[Tuple[float, int, int], ...] = ((4.0, 56, 1400), (8.0, 56, 1400), (8.0, 56, 1024))
TAPER_START_FRAC = 0.95


def eb_variant_tag(fwhm_v, lmin_v, lcut_v) -> str:
    """HDF5 key suffix for one (fwhm, lmin, lcut) smoothing variant (mirror of master's)."""
    return (f"fwhm{fwhm_v:g}"
            + ("" if lmin_v is None else f"_lmin{int(lmin_v)}")
            + ("" if lcut_v is None else f"_lcut{int(lcut_v)}"))


def patch_noise_std(rand_E_maps, patches, nside_out, ang, patch_names):
    """Per-(patch, bin) std of the filtered random-rotation E map, + the pooled 'all'.
    Mirror of ``master_kids_legacy_simulator.patch_noise_std`` (parity-checked)."""
    from src.cosmology.pixelise_maps import get_patch_values
    per_patch = get_patch_values(rand_E_maps, patches, nside_out, ang)
    out = {}
    flat = []
    for patch_idx, patch_name in enumerate(patch_names):
        p = np.asarray(per_patch[patch_idx], dtype=np.float64)   # (nbins, H, W)
        out[patch_name] = p.reshape(p.shape[0], -1).std(axis=1)
        flat.append(p.reshape(p.shape[0], -1))
    out["all"] = np.concatenate(flat, axis=1).std(axis=1)
    return out


RNG_STREAM_POSTPROC = 2   # mirror of master's stream index (parity-checked)


def block_postproc_rng(seed: int, sim_num: int, outer_idx: int, rot_idx: int):
    """Vendored mirror of ``master_kids_legacy_simulator.build_block_rngs(...)[2]`` -- the
    random-rotation (noise-only alm) stream of one (sim, outer, rot) block -- WITHOUT the master's
    side effect of seeding the global numpy RNG. Kept here so nothing in ``src/observation`` needs
    to import the master module (mpi4py, glass.ext.camb) at runtime."""
    ss = np.random.SeedSequence([int(seed), int(sim_num), int(outer_idx), int(rot_idx)])
    children = ss.spawn(4)
    return np.random.default_rng(children[RNG_STREAM_POSTPROC])


def master_postproc_rng(attrs: dict):
    """Rebuild the EXACT random-rotation RNG a saved SIM catalogue's mock was post-processed with,
    from its file attrs (``rng_seed, sim_id, outer_idx, rot_idx``). Only possible for a catalogue
    written under ``--rng-seed`` (attr ``rng_seed`` >= 0); returns None otherwise. Used by the
    fidelity gate to make the shape-noise debias term bit-reproducible."""
    seed = int(attrs.get("rng_seed", -1))
    if seed < 0:
        return None
    return block_postproc_rng(seed, int(attrs.get("sim_id", 0)),
                              int(attrs.get("outer_idx", 0)), int(attrs.get("rot_idx", 0)))


def check_master_parity() -> dict:
    """Assert the vendored constants/helpers match the master simulator. Returns the values."""
    import master_kids_legacy_simulator as m  # noqa: WPS433 (deliberate late import)
    assert float(m.A3S8_FWHM_ARCMIN) == A3S8_FWHM_ARCMIN, (m.A3S8_FWHM_ARCMIN, A3S8_FWHM_ARCMIN)
    assert tuple(tuple(v) for v in m.A3S8_VARIANTS) == A3S8_VARIANTS, (m.A3S8_VARIANTS, A3S8_VARIANTS)
    assert m.A3S8_MAP_DTYPE is A3S8_MAP_DTYPE
    assert tuple(tuple(v) for v in m.EB_SMOOTHING_VARIANTS) == EB_SMOOTHING_VARIANTS_PROD, (
        m.EB_SMOOTHING_VARIANTS, EB_SMOOTHING_VARIANTS_PROD)
    for f, l, c in [(4.0, 56, 1400), (8.0, None, None), (8.0, 56, None)]:
        assert m.eb_variant_tag(f, l, c) == eb_variant_tag(f, l, c)
    # patch_noise_std: numerical parity on a random map
    rng = np.random.default_rng(0)
    import healpy as hp
    nside_out = 16
    maps = rng.normal(size=(2, hp.nside2npix(nside_out)))
    patches = [(12, -31, 90, 11), (-178, 0, 112, 10)]
    a = m.patch_noise_std(maps, patches, nside_out, 0, ["south", "north"])
    b = patch_noise_std(maps, patches, nside_out, 0, ["south", "north"])
    for k in a:
        assert np.allclose(a[k], b[k]), k
    # block RNG: the vendored postproc stream must equal the master's (first 1000 draws)
    state0 = np.random.get_state()
    _, _, mp = m.build_block_rngs(7, 3, 1, 2)
    np.random.set_state(state0)     # undo the master's global-RNG side effect
    vp = block_postproc_rng(7, 3, 1, 2)
    assert np.array_equal(mp.random(1000), vp.random(1000)), "block RNG parity"
    return {"A3S8_FWHM_ARCMIN": A3S8_FWHM_ARCMIN, "A3S8_VARIANTS": A3S8_VARIANTS,
            "EB_SMOOTHING_VARIANTS": EB_SMOOTHING_VARIANTS_PROD}


# ----------------------------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Geometry:
    """Everything the observable-building sequence needs to know about the sky/ell grid."""
    name: str
    nbins: int
    nside: int
    lmax: int
    nside_out: int
    lower_lscale: int
    upper_lscale: int
    nbands: int
    patch_names: Tuple[str, ...]
    patches: Tuple[Tuple[float, float, float, float], ...]
    ang: float = 0.0     # patch rotation: real data and rot-0 mocks live in the survey frame

    @classmethod
    def production(cls) -> "Geometry":
        from src.KiDS import simulation_config as sc
        return cls(name="production", nbins=int(sc.nbins), nside=int(sc.nside), lmax=int(sc.lmax),
                   nside_out=512, lower_lscale=int(sc.lower_lscale),
                   upper_lscale=int(sc.upper_lscale), nbands=int(sc.nbands),
                   patch_names=tuple(sc.named_patches.keys()),
                   patches=tuple(tuple(v) for v in sc.named_patches.values()))

    @classmethod
    def smoke(cls) -> "Geometry":
        from src.KiDS import simulation_config as sc
        from src.smoke_sim import SMOKE_CONFIG as S
        return cls(name="smoke", nbins=int(sc.nbins), nside=int(S["nside"]), lmax=int(S["lmax"]),
                   nside_out=int(S["nside_out"]), lower_lscale=int(S["lower_lscale"]),
                   upper_lscale=int(S["upper_lscale"]), nbands=int(S["nbands"]),
                   patch_names=tuple(sc.named_patches.keys()),
                   patches=tuple(tuple(v) for v in sc.named_patches.values()))

    @classmethod
    def named(cls, name: str) -> "Geometry":
        if name == "production":
            return cls.production()
        if name == "smoke":
            return cls.smoke()
        raise ValueError(f"unknown geometry {name!r} (production|smoke)")

    @classmethod
    def from_catalogue_attrs(cls, attrs: dict, fallback: str = "production") -> "Geometry":
        """Pick the geometry a SIM catalogue was produced under (its ``nside``/``nside_out`` attrs)."""
        nside = int(attrs.get("nside", -1))
        if nside == cls.production().nside:
            return cls.production()
        if nside == cls.smoke().nside:
            return cls.smoke()
        return cls.named(fallback)

    def as_dict(self) -> dict:
        return {"name": self.name, "nbins": self.nbins, "nside": self.nside, "lmax": self.lmax,
                "nside_out": self.nside_out, "lower_lscale": self.lower_lscale,
                "upper_lscale": self.upper_lscale, "nbands": self.nbands,
                "patch_names": list(self.patch_names), "patches": [list(p) for p in self.patches],
                "ang": self.ang}


@dataclass(frozen=True)
class MapVariants:
    """Which map products to build.

    ``a3s8``: the smoothed-counts E products ``E_sc8_<tag>`` + ``noise_std_sc8_<tag>`` (what the
    flagship BGP arms train on, after ``prebake --eb-variant sc8_<tag> --noise-norm rand``).
    ``eb``: the plain counts-normalised ``E_<tag>``/``B_<tag>`` products (the a1/nobgp arms and the
    B-map diagnostics). ``noise_std_for_eb``: which eb variants also get a ``noise_std_<tag>`` meter
    (None = all of them, mirroring master's ``NOISE_STD_VARIANTS=None``).
    """
    a3s8: Tuple[Tuple[float, int, int], ...] = A3S8_VARIANTS
    a3s8_fwhm_arcmin: float = A3S8_FWHM_ARCMIN
    eb: Tuple[Tuple[float, int, int], ...] = EB_SMOOTHING_VARIANTS_PROD
    noise_std_for_eb: Optional[Tuple[Tuple[float, int, int], ...]] = None

    @classmethod
    def named(cls, name: str) -> "MapVariants":
        if name in ("production", "sc8only"):
            return cls()
        if name == "full":          # sc8 arms + the three counts variants (B maps for diagnostics)
            return cls(eb=EB_SMOOTHING_VARIANTS_FULL)
        if name == "a1only":        # counts variants only (the nobgp / a1 arm); no sc8 branch
            return cls(a3s8=(), eb=((4.0, 56, 1400),))
        raise ValueError(f"unknown map-variant preset {name!r} (production|full|a1only)")

    def as_dict(self) -> dict:
        return {"a3s8": [list(v) for v in self.a3s8], "a3s8_fwhm_arcmin": self.a3s8_fwhm_arcmin,
                "eb": [list(v) for v in self.eb],
                "noise_std_for_eb": (None if self.noise_std_for_eb is None
                                     else [list(v) for v in self.noise_std_for_eb])}
