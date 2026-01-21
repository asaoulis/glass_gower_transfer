from abc import ABC, abstractmethod
from .nla import kappa_ia_nla_m
import numpy as np

class BaseSystematics(ABC):

    def apply_intrinsic_alignments(self, delta, kappa, z_eff, tomo):
        return kappa

    def apply_shear_bias(self, tomo, shear, lat):
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

    def apply_intrinsic_alignments(self, delta, kappa, z_eff, tomo):
        kappa_ia = kappa_ia_nla_m(
            delta,
            z_eff,
            self.nla['f_red'][tomo],
            self.cosmo,
            self.nla['a_ia'],
            self.nla['b_ia'],
            self.nla['log10_M_eff'][tomo]
        )
        return kappa + kappa_ia

    def apply_shear_bias(self, tomo, shear, lat):
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
