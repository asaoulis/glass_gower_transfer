from .levin import extrapolate_section
from cosmology import Cosmology
import camb
import glass
def update_pars_config(old_pars):
    pars = old_pars.copy()
    # What to compute
    pars.WantCls = True
    pars.WantTransfer = False   # CHANGED
    pars.WantScalars = True
    pars.WantTensors = False
    pars.WantVectors = False
    pars.WantDerivedParameters = True
    pars.Want_cl_2D_array = True
    pars.Want_CMB = True
    pars.Want_CMB_lensing = True

    # Lensing & nonlinear
    pars.DoLensing = True       # CHANGED
    pars.NonLinear = camb.model.NonLinear_both
    tr = pars.Transfer

    tr.high_precision = True   # flipped
    tr.accurate_massive_neutrinos = True # flipped

    tr.kmax = 5.0               # CHANGED, was 0.9
    tr.k_per_logint = 0
    tr.PK_num_redshifts = 1     # CHANGED
    tr.PK_redshifts = [0.0]    # CHANGED
    pars.validate()
    return pars

def get_camb_matter_cls(pars, lmax, zmin, zmax, dx):
    cosmo = Cosmology.from_camb(pars)

    zb = glass.shells.distance_grid(cosmo, zmin, zmax, dx=dx)
    # ws = glass.shells.tophat_windows(zb)
    ws = glass.shells.linear_windows(zb)
    updated_pars = update_pars_config(pars)
    camb_cls = glass.ext.camb.matter_cls(updated_pars, lmax, ws, limber=False, limber_lmin=100)
    return ws, camb_cls