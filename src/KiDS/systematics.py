from pathlib import Path

import numpy as np


m_bias = np.array([-0.022869, -0.015966, -0.011331, 0.019870, 0.029506, 0.044535], dtype=np.float64)
m_bias_unc = np.array([0.005630, 0.005900, 0.007111, 0.006773, 0.007598, 0.008902], dtype=np.float64)

c_1_bias_north = np.array([3.372, 8.941, 4.523, 4.722, 6.658, 4.224], dtype=np.float64) * 1e-4
c_1_bias_north_unc = np.array([1.528, 1.442, 1.747, 1.713, 1.887, 2.252], dtype=np.float64) * 1e-4
c_2_bias_north = np.array([7.941, 8.852, 4.533, 5.368, 5.532, 10.26], dtype=np.float64) * 1e-4
c_2_bias_north_unc = np.array([1.442, 1.642, 1.777, 1.665, 1.890, 2.400], dtype=np.float64) * 1e-4

c_1_bias_south = np.array([-3.398, -9.536, -4.755, -4.532, -6.117, -3.717], dtype=np.float64) * 1e-4
c_1_bias_south_unc = np.array([1.626, 1.519, 1.835, 1.653, 1.910, 2.151], dtype=np.float64) * 1e-4
c_2_bias_south = np.array([-8.002, -6.026, -4.766, -5.152, -5.082, -9.027], dtype=np.float64) * 1e-4
c_2_bias_south_unc = np.array([1.572, 1.590, 1.731, 1.594, 1.834, 2.282], dtype=np.float64) * 1e-4

f_red = np.array([0.15, 0.2, 0.17, 0.24, 0.19, 0.03], dtype=np.float64)
sigma_e = np.array([0.2772, 0.2716, 0.2899, 0.2619, 0.2802, 0.3002], dtype=np.float64)


def load_massdep_priors(data_dir: str | Path):
	data_dir = Path(data_dir)
	massdep_means = np.loadtxt(data_dir / "priors" / "massdep_means.txt")
	massdep_cov = np.loadtxt(data_dir / "priors" / "massdep_cov.txt")
	return massdep_means[2:], massdep_cov[2:, 2:]
