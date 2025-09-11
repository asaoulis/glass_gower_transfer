import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import camb
import glass
import levinpower


# --------------------
# Kernel Functions
# --------------------


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

def extrapolate_section(k, z, Pk, kmin, kmax, nmin, nmax, npoint):
    # load current values
    nz = len(z)
    # load other current values
    # extrapolate
    P_out = []
    for i in range(nz):
        Pi = Pk[i, :]
        logk, logp = linear_extend(np.log(k), np.log(Pi), np.log(
            kmin), np.log(kmax), nmin, nmax, npoint)
        P_out.append(np.exp(logp))
    
    return np.exp(logk), np.array(P_out)

def compute_los_integrals(zb, results, n_chi=1000):
    """Compute line-of-sight integration in redshift and comoving distance."""
    z_vals = np.linspace(zb[0], zb[-1], n_chi)
    chi_vals = np.asarray(results.angular_diameter_distance(z_vals) * (1 + z_vals))
    return z_vals, chi_vals


def make_kernels(pars, z_distance, chi_distance, zbins):
    """Generate integration kernels for each redshift shell."""
    kernels = []

    for i in range(len(zbins) - 1):
        z_range = z_distance[((zbins[i + 1] >= z_distance) & (z_distance > zbins[i]))]
        chi_range = chi_distance[((zbins[i + 1] >= z_distance) & (z_distance > zbins[i]))]
        w = pars.DarkEnergy.w + pars.DarkEnergy.wa * np.divide(z_range, z_range + 1)

        # E_of_z      = np.sqrt(config['omega_m']*(1+z_range)**3 + config['omega_k']*(1+z_range)**2 + config['omega_lambda']*(1+z_range)**(3*(1+w)))
        omega_de = 1 + pars.omk - pars.omegam - pars.omeganu
        E_of_z = np.sqrt(
            pars.omegam * (1 + z_range) ** 3
            + pars.omk * (1 + z_range) ** 2
            + omega_de * (1 + z_range) ** (3 * (1 + w))
        )

        k = np.zeros(len(z_distance))
        k[((zbins[i + 1] >= z_distance) & (z_distance > zbins[i]))] = np.divide(
            chi_range**2, E_of_z
        )
        k /= np.trapezoid(k, z_distance)
        kernels.append(k)

    return np.array(kernels)


def gen_log_space(limit, n, offset=1):
    """Generate roughly log-spaced ell values starting at offset."""
    result = [offset]
    if n > 1:
        ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))
    while len(result) < n:
        next_value = result[-1] * ratio
        if next_value - result[-1] >= 1:
            result.append(next_value)
        else:
            result.append(result[-1] + 1)
            ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))
    return np.array(list(map(lambda x: round(x) - 1, result)), dtype=np.uint64)


# --------------------
# Main Wrapper
# --------------------

def setup_levin_power(zb, z_grid, chi_grid, extended_k, extended_pk, results, pars,
                      ell_limber=30000, ell_nonlimber=1600,
                      ell_max=1600, ell_min=2, n_ell=1600,
                      N_nonlimber=200, N_limber=100, Ninterp=1000,
                      max_number_subintervals=30):
    """
    Initialize LevinPower with provided cosmology, shells, and power spectrum.

    Returns:
        ws: Shell windows
        lp: LevinPower object
    """
    # Shells / Window functions
    ws = glass.shells.tophat_windows(zb)

    # Line-of-sight integration
    los_z_integration, los_chi_integration = compute_los_integrals(zb, results)

    # Kernel construction
    kernels = make_kernels(pars, los_z_integration, los_chi_integration, zb)

    # ell array
    ell = list(map(int, gen_log_space(ell_max, int(n_ell)) + ell_min))

    # LevinPower initialization
    lp = levinpower.LevinPower(
        False,
        number_count=len(ws),
        z_bg=z_grid,
        chi_bg=chi_grid,
        chi_cl=los_chi_integration,
        kernel=kernels.T,
        k_pk=extended_k,
        z_pk=z_grid,
        pk=extended_pk.flatten(),
        boxy=True,
    )

    # Set levin integration parameters
    lp.set_parameters(
        ELL_limber=ell_limber,
        ELL_nonlimber=ell_nonlimber,
        max_number_subintervals=max_number_subintervals,
        minell=int(ell_min),
        maxell=int(ell_limber),
        N_nonlimber=N_nonlimber,
        N_limber=N_limber,
        Ninterp=Ninterp,
    )
    return ws, lp, ell
