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
    
def kappa_ia_nla_m(delta, zeff, red_f, cosmo, a_ia, b_ia, log10_M,
                   log10_M_pivot=13.5 #solar masses / h
    ):
    
    # c1 = 5e-14 / cosmo.h**2  # Solar masses per cubic Mpc
    # rho_c1 = c1 * cosmo.rho_c_z(0.0)

    rho_c1 = 0.0134

    prefactor = -a_ia * rho_c1 * cosmo.omega_m
    inverse_linear_growth = 1.0 / linear_growth_factor(zeff, cosmo)
    mass_term = (10**(log10_M)/10**(log10_M_pivot))**(b_ia)

    f_nla = (
        red_f * prefactor * inverse_linear_growth * mass_term
    )
    return delta * f_nla
