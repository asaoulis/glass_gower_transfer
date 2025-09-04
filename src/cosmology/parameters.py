import numpy as np
import camb
from cosmology import Cosmology

# --------------------
# Constants and Defaults
# --------------------

DEFAULT_PARAMS = {
    "h": 0.6736,
    "omch2": 0.12,
    "ombh2": 0.022,
    "s8": 0.8,
    "w0": -1.0,
    "wa": 0.0,
    "ns": 0.965,
    "A_baryon": 3.10,
    "omega_k": 0.0,
    "mnu": 0.06,
    "alpha": 0.5,
    "fid_As": 2.1e-9,
    "kmax_transfer": 1.2,
    "k_per_logint": 5,
    "a_0": 0.98,
    "a_1": -0.12,
    "seed_num": 12,
    "nside": 1024,
    "n_ell": 20,
    "lmin": 0,
    "lmax": 300,
    "zmin": 0.0,
    "zmid": 2.0,
    "nz_mid": 50,
    "zmax": 6.0,
    "nz": 256,

}
# --------------------
# Core Functions
# --------------------

def compute_omegas(params):
    h = params["h"]
    Omega_c = params["omch2"] / h**2
    Omega_b = params["ombh2"] / h**2
    Omega_m = Omega_c + Omega_b
    return Omega_c, Omega_b, Omega_m

def compute_eta(params):
    return params["a_0"] + params["a_1"] * params["A_baryon"]

def s8_to_As(params):
    h = params["h"]
    s8 = params["s8"]
    ombh2 = params["ombh2"]
    omch2 = params["omch2"]
    omega_k = params["omega_k"]
    mnu = params["mnu"]
    fid_As = params["fid_As"]
    alpha = params["alpha"]
    ns = params["ns"]
    w0 = params["w0"]
    wa = params["wa"]
    kmax = params["kmax_transfer"]
    k_per_logint = params["k_per_logint"]

    Omega_c, Omega_b, Omega_m = compute_omegas(params)
    sigma8 = s8 / ((Omega_m / 0.3) ** alpha)

    p = camb.CAMBparams(WantTransfer=True, Want_CMB=False, Want_CMB_lensing=False, DoLensing=False,
                        NonLinear="NonLinear_none", WantTensors=False, WantVectors=False, WantCls=False,
                        WantDerivedParameters=False, want_zdrag=False, want_zstar=False)
    
    p.set_accuracy(DoLateRadTruncation=True)
    p.Transfer.high_precision = False
    p.Transfer.accurate_massive_neutrino_transfers = False
    p.Transfer.kmax = kmax
    p.Transfer.k_per_logint = k_per_logint
    p.Transfer.PK_redshifts = np.array([0.0])
    
    p.set_cosmology(H0=h * 100, ombh2=ombh2, omch2=omch2, omk=omega_k, mnu=mnu)
    p.set_dark_energy(w=w0, wa=wa)
    p.set_initial_power(camb.initialpower.InitialPowerLaw(As=fid_As, ns=ns))
    p.Reion = camb.reionization.TanhReionization()
    p.Reion.Reionization = False

    results = camb.get_results(p)
    fid_sigma8 = results.get_sigma8()[-1]

    As = fid_As * (sigma8 / fid_sigma8) ** 2

    return sigma8, As, Omega_c, Omega_b, Omega_m

def build_cosmology(params):
    # Compute derived quantities
    params = {**DEFAULT_PARAMS, **params}
    sigma8, As, Omega_c, Omega_b, Omega_m = s8_to_As(params)
    eta = compute_eta(params)

    z_grid = np.linspace(params["zmin"], params["zmax"], params["nz"])

    pars = camb.set_params(H0=params["h"] * 100,
                           ombh2=params["ombh2"],
                           omch2=params["omch2"],
                           HMCode_A_baryon=params["A_baryon"],
                           HMCode_eta_baryon=eta,
                           As=As,
                           ns=params["ns"],
                           w=params["w0"],
                           wa=params["wa"],
                           halofit_version='mead',
                           neutrino_hierarchy='normal',
                           DoLensing=False,
                           NonLinear=camb.model.NonLinear_both,
                           lmax=3000)

    pars.set_matter_power(redshifts=z_grid, kmax=20.0)
    cosmo = Cosmology.from_camb(pars)

    return cosmo, pars
