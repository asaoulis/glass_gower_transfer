import numpy as np
from scipy import integrate

def linear_extend(x, y, xmin, xmax, nmin, nmax, nfit):
    if xmin < x.min():
        xf = x[:nfit]
        yf = y[:nfit]
        p = np.polyfit(xf, yf, 1)
        xnew = np.linspace(xmin, x.min(), nmin, endpoint=False)
        ynew = np.polyval(p, xnew)
        x = np.concatenate((xnew, x))
        y = np.concatenate((ynew, y))
    if xmax > x.max():
        xf = x[-nfit:]
        yf = y[-nfit:]
        p = np.polyfit(xf, yf, 1)
        xnew = np.linspace(x.max(), xmax, nmax, endpoint=True)
        # skip the first point as it is just the xmax
        xnew = xnew[1:]
        ynew = np.polyval(p, xnew)
        x = np.concatenate((x, xnew))
        y = np.concatenate((y, ynew))
    return x, y

def growth_integrand(a, cosmo):
    if a == 0:
        return 0.0
    # For standard cosmologies, (x * E(1/x-1))^-3 approaches x^(3/2) or x^3 near x=0, so integral converges.
    return (a * cosmo.ef(1.0/a - 1.0))**-3.0

def growth_integral_value(a_val, cosmo):
    result, error = integrate.quad(growth_integrand, 0, a_val, args=(cosmo,))
    return result

def linear_growth_factor(z, cosmo):
    integral_at_z0 = growth_integral_value(1.0, cosmo)

    if isinstance(z, (list, np.ndarray)):
        z_arr = np.asarray(z)
        results = np.empty_like(z_arr, dtype=float)
        for i, z_val in enumerate(z_arr):
            if z_val < 0:
                results[i] = np.nan
                continue
            a_val = 1.0 / (1.0 + z_val)
            integral_at_a = growth_integral_value(a_val, cosmo)
            results[i] = (cosmo.ef(z_val) * integral_at_a) / (cosmo.ef(0) * integral_at_z0)
        return results
    else:
        if z < 0:
            return np.nan
        a = 1.0 / (1.0 + z)
        integral_at_a = growth_integral_value(a, cosmo)
        return (cosmo.ef(z) * integral_at_a) / (cosmo.ef(0) * integral_at_z0)
    
# C1 * rho_cr(z=0) in the IA normalisation, in (Mpc/h)^3 / M_sun units; the standard NLA
# value C1 = 5e-14 (h^2 M_sun^-1 Mpc^3) * rho_cr(0) ~ 0.0134 (see e.g. Bridle & King 2007).
RHO_C1 = 0.0134


def nla_amplitude(zeff, cosmo, a_ia):
    """Plain-NLA effective IA amplitude f_NLA = -A_IA * C1*rho_cr * Omega_m / D(z).

    This is the scalar that multiplies the matter overdensity delta to give the NLA
    intrinsic-alignment convergence (whole population, no red-fraction / mass weighting). It is
    shared by `kappa_ia_nla` and the TATT density-weighting term so the NLA normalisation is
    defined in exactly one place.
    """
    inverse_linear_growth = 1.0 / linear_growth_factor(zeff, cosmo)
    return -a_ia * RHO_C1 * cosmo.omega_m * inverse_linear_growth


def kappa_ia_nla(delta, zeff, cosmo, a_ia):
    """NLA (Bridle & King 2007) IA contribution to convergence: kappa_IA = delta * f_NLA.

    Whole-population single-amplitude NLA (Wright et al. 2025, sec 2.4.1) — i.e. NLA-M with
    f_red = 1 and no halo-mass term. Also used as the NLA part of the NLA-z (per-bin effective
    `a_ia`) and the restricted-TATT/NLA-k models.
    """
    return delta * nla_amplitude(zeff, cosmo, a_ia)


def nla_z_effective_amplitude(a_ia, b_z, avg_a, a_piv=0.769):
    """NLA-z per-bin effective amplitude (Wright et al. 2025, eq. 7).

    A_eff^(i) = A_IA + B_IA * (<a>^(i) / a_piv - 1), where <a>^(i) is the N(z)-weighted average
    scale factor in tomographic bin i and a_piv ~ 0.769 (the z=0.3 pivot). With B_IA (`b_z`) = 0
    this reduces exactly to plain NLA.
    """
    return a_ia + b_z * (avg_a / a_piv - 1.0)


def gamma_ia_density_weight(delta, s, zeff, cosmo, a_ia, b_src):
    """Restricted-TATT / NLA-k density-weighting shear correction (Wright et al. 2025, sec 2.4.3).

    The Blazek et al. (2019) tidal-alignment-with-density-weighting term is the real-space product
    C1d * (delta * s_ij) with C1d = b_src * C1. Because the unit-amplitude NLA intrinsic shear is
    exactly s = from_convergence(delta) (the spin-2 tidal field sourced by treating delta as a
    convergence), the restricted-TATT intrinsic shear is gamma_NLA * (1 + b_src * delta). This
    function returns ONLY the extra term beyond NLA, gamma_dw = f_NLA * b_src * (delta * s), to be
    ADDED to the already-computed shear (which contains gamma_NLA = f_NLA * s). The torquing term
    is fixed off, matching the restricted NLA-k model. `delta` here is the projected shell
    overdensity (the same field used as the NLA tidal source).
    """
    return nla_amplitude(zeff, cosmo, a_ia) * b_src * delta * s


def kappa_ia_nla_m(delta, zeff, red_f, cosmo, a_ia, b_ia, log10_M,
                   log10_M_pivot=13.5 #solar masses / h
    ):

    # c1 = 5e-14 / cosmo.h**2  # Solar masses per cubic Mpc
    # rho_c1 = c1 * cosmo.rho_c_z(0.0)

    rho_c1 = RHO_C1

    prefactor = -a_ia * rho_c1 * cosmo.omega_m
    inverse_linear_growth = 1.0 / linear_growth_factor(zeff, cosmo)
    mass_term = (10**(log10_M)/10**(log10_M_pivot))**(b_ia)

    f_nla = (
        red_f * prefactor * inverse_linear_growth * mass_term
    )
    return delta * f_nla
