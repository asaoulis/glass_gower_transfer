from abc import ABC, abstractmethod
from .nla import (
    gamma_ia_density_weight,
    kappa_ia_nla,
    kappa_ia_nla_m,
    nla_z_effective_amplitude,
)
import numpy as np
import healpy as hp
import glass.shapes
import glass.lensing

class BaseSystematics(ABC):

    def apply_intrinsic_alignments(self, delta, kappa, z_eff, tomo):
        return kappa

    def intrinsic_alignment_amplitude(self, z_eff, tomo):
        """Per-bin scalar f such that the IA convergence is kappa_IA = f * delta (0 = no IA).

        Every IA model in this codebase contributes a convergence that is a SCALAR multiple of
        the matter shell `delta`; exposing the scalar lets the simulator exploit the linearity
        of `glass.lensing.from_convergence` (map2alm -> almxfl -> alm2map_spin, all linear) and
        compute the spin-2 SHT of kappa and delta ONCE per shell, forming each tomographic
        shear as gamma_i = G(kappa) + f_i * G(delta) instead of one SHT per bin.
        A subclass whose IA convergence is NOT delta-proportional must not use this seam:
        it should raise here (or the simulator loop must revert to per-bin transforms).
        """
        return 0.0

    def apply_intrinsic_alignments_shear(self, delta, gamma, z_eff, tomo, s=None):
        """Shear-level IA correction. Default: unchanged.

        Used by IA models whose contribution is NOT a pure convergence (so it cannot be folded
        into `apply_intrinsic_alignments`): the restricted-TATT/NLA-k density-weighting term is a
        real-space product of the density and tidal fields and must be added to the shear directly.
        `s` optionally passes in the precomputed unit-amplitude tidal shear
        from_convergence(delta, shear=True) so callers that already have it avoid a second SHT.
        """
        return gamma

    def effective_mask(self, tomo, shell_idx, base_mask, *, mask_rot_angle=0, rotator=None):
        """Mask the galaxies are sampled from for (tomo, shell). Default: unchanged.

        The shell index / rotation args are accepted for parity with the variable-depth
        override (which multiplies in a per-(tomo, shell) depth map rotated to match the
        footprint); the base model ignores them.
        """
        return base_mask

    def sample_ellipticity(self, tomo, lon, lat, count, sigma_e, *, rng):
        """Draw intrinsic galaxy ellipticities. Default: scalar per-bin dispersion.

        ``lon``/``lat`` are accepted for parity with the variable-depth override (which reads a
        per-pixel dispersion); the base model ignores them.
        """
        return glass.shapes.ellipticity_intnorm(count, sigma_e, rng=rng)

    def apply_shear_bias(self, tomo, shear, lat, *, lon=None):
        return shear.real, shear.imag

    def __repr__(self):
        return f"{self.__class__.__name__}()"

class NoSystematics(BaseSystematics):
    pass

class NLASystematics(BaseSystematics):

    def __init__(self, *, shear_bias, nla, cosmo):
        self.shear_bias = shear_bias
        self.nla = nla
        self.cosmo = cosmo

    def intrinsic_alignment_amplitude(self, z_eff, tomo):
        # delta enters every kappa_ia_* purely multiplicatively, so evaluating at delta=1.0
        # extracts the exact scalar f with kappa_IA = f * delta (bit-identical to the map path).
        model = self.nla.get('model', 'nla_m')
        if model == 'nla_m':
            return kappa_ia_nla_m(
                1.0,
                z_eff,
                self.nla['f_red'][tomo],
                self.cosmo,
                self.nla['a_ia'],
                self.nla['b_ia'],
                self.nla['log10_M_eff'][tomo]
            )
        if model == 'nla':
            # Whole-population single-amplitude NLA (f_red=1, no mass term).
            return kappa_ia_nla(1.0, z_eff, self.cosmo, self.nla['a_ia'])
        if model == 'nla_z':
            # NLA with per-bin redshift-dependent effective amplitude.
            a_eff = nla_z_effective_amplitude(
                self.nla['a_ia'], self.nla['b_z'], self.nla['avg_a'][tomo]
            )
            return kappa_ia_nla(1.0, z_eff, self.cosmo, a_eff)
        if model == 'tatt':
            # Restricted-TATT/NLA-k: the NLA part is convergence-additive here; the
            # density-weighting term is added at the shear level (apply_intrinsic_alignments_shear).
            return kappa_ia_nla(1.0, z_eff, self.cosmo, self.nla['a_ia'])
        raise ValueError(f"Unknown IA model: {model!r}")

    def apply_intrinsic_alignments(self, delta, kappa, z_eff, tomo):
        return kappa + delta * self.intrinsic_alignment_amplitude(z_eff, tomo)

    def apply_intrinsic_alignments_shear(self, delta, gamma, z_eff, tomo, s=None):
        if self.nla.get('model', 'nla_m') != 'tatt':
            return gamma
        # s = unit-amplitude NLA intrinsic shear (spin-2 tidal field of delta). The NLA part
        # f_NLA*s is already in `gamma` (added via the convergence above); here we add the
        # restricted-TATT density-weighting term f_NLA * b_src * (delta * s).
        if s is None:
            s, = glass.lensing.from_convergence(delta, shear=True)
        return gamma + gamma_ia_density_weight(
            delta, s, z_eff, self.cosmo, self.nla['a_ia'], self.nla['b_src']
        )

    def apply_shear_bias(self, tomo, shear, lat, *, lon=None):
        # hemisphere split
        north = np.abs(lat) < 15
        south = ~north
        c_bias = np.zeros_like(shear)
        c_bias[north] = self.shear_bias['c1_north'][tomo] + 1j*self.shear_bias['c2_north'][tomo]
        c_bias[south] = self.shear_bias['c1_south'][tomo] + 1j*self.shear_bias['c2_south'][tomo]
        m = self.shear_bias['m_bias'][tomo]
        E1 = (1 + m)*shear.real + c_bias.real
        E2 = (1 + m)*shear.imag + c_bias.imag
        return E1, E2


class VariableDepthSystematics(NLASystematics):
    """NLA systematics + KiDS-Legacy variable depth (VD).

    Adds the three VD seams on top of NLA (IA + additive c-bias N/S split are inherited
    unchanged). Behaviour matches the reference ``sample_galaxy_catalogue_vd``:

    - ``effective_mask``: galaxies are sampled from ``base_mask * var_depth_mask[tomo, shell]``
      (the depth map rotated by the same ``lon_exact`` mask-rotation as ``base_mask``).
    - ``sample_ellipticity``: per-pixel intrinsic dispersion via ``vd_shapes`` (not a scalar).
    - ``apply_shear_bias``: per-galaxy multiplicative bias ``m_bias_vd_realised[tomo][vd_bin]``
      (vd_bin from digitising the VD tracer at the galaxy pixel against that tomo bin's own
      edges — ``vd_trace_edges`` is (nbins, 11) since the 2026-08 recalibration), the inherited additive
      c-bias, plus the residual-PSF term ``alpha_{1,2}*psf_bias_map_{1,2}[gal_pix]``.

    All per-galaxy lookups (vd_bin, vd_shapes, psf) use the SURVEY-frame ``lon``/``lat`` (the
    galaxy positions un-rotated by the mask Z-rotation), so the VD/PSF maps are footprint-tied.
    """

    def __init__(
        self,
        *,
        shear_bias,
        nla,
        cosmo,
        var_depth_mask,
        vd_shapes,
        vd_map,
        vd_trace_edges,
        n_vardepth_bins,
        nside,
        m_bias_vd_realised,
        alpha_1_realised,
        alpha_2_realised,
        psf_bias_map_1,
        psf_bias_map_2,
    ):
        super().__init__(shear_bias=shear_bias, nla=nla, cosmo=cosmo)
        self.var_depth_mask = var_depth_mask
        self.vd_shapes = vd_shapes
        self.vd_map = vd_map
        self.vd_trace_edges = vd_trace_edges
        self.n_vardepth_bins = n_vardepth_bins
        self.nside = nside
        self.m_bias_vd_realised = m_bias_vd_realised
        self.alpha_1_realised = alpha_1_realised
        self.alpha_2_realised = alpha_2_realised
        self.psf_bias_map_1 = psf_bias_map_1
        self.psf_bias_map_2 = psf_bias_map_2
        # Per-shell cache of the (un-rotated) var_depth_mask[tomo, shell] full maps, so the
        # interp-heavy __getitem__ runs once per (tomo, shell) instead of per augmentation.
        self._cache_shell = None
        self._vd_mask_cache = {}

    def effective_mask(self, tomo, shell_idx, base_mask, *, mask_rot_angle=0, rotator=None):
        if shell_idx != self._cache_shell:
            self._cache_shell = shell_idx
            self._vd_mask_cache = {}
        if tomo not in self._vd_mask_cache:
            self._vd_mask_cache[tomo] = np.asarray(self.var_depth_mask[tomo, shell_idx])
        vdm = self._vd_mask_cache[tomo]
        if mask_rot_angle != 0:
            if rotator is None:
                raise ValueError(
                    "VariableDepthSystematics.effective_mask needs a rotator for a "
                    "nonzero mask_rot_angle"
                )
            vdm = rotator.rotate_map_longitude_exact(vdm, [mask_rot_angle, 0, 0], False)
        return base_mask * vdm

    def sample_ellipticity(self, tomo, lon, lat, count, sigma_e, *, rng):
        return self.vd_shapes.sample(tomo, lon, lat, rng=rng)

    def apply_shear_bias(self, tomo, shear, lat, *, lon=None):
        gal_pix = hp.ang2pix(self.nside, lon, lat, lonlat=True)

        vd_bin = np.clip(
            np.digitize(self.vd_map[tomo][gal_pix], self.vd_trace_edges[tomo]) - 1,
            0, self.n_vardepth_bins - 1,
        )
        gal_m_bias = self.m_bias_vd_realised[tomo][vd_bin]

        north = np.abs(lat) < 15
        south = ~north
        c_bias = np.zeros_like(shear)
        c_bias[north] = self.shear_bias['c1_north'][tomo] + 1j*self.shear_bias['c2_north'][tomo]
        c_bias[south] = self.shear_bias['c1_south'][tomo] + 1j*self.shear_bias['c2_south'][tomo]

        E1 = (1 + gal_m_bias)*shear.real + c_bias.real + self.alpha_1_realised[tomo]*self.psf_bias_map_1[gal_pix]
        E2 = (1 + gal_m_bias)*shear.imag + c_bias.imag + self.alpha_2_realised[tomo]*self.psf_bias_map_2[gal_pix]
        return E1, E2
