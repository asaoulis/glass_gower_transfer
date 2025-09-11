import numpy as np

nbins = 6

ztomo = [
    (0.10, 0.42),
    (0.42, 0.58),
    (0.58, 0.71),
    (0.71, 0.90),
    (0.90, 1.14),
    (1.14, 2.00)
]

ztomo_label = [
    ('0.10', '0.42'),
    ('0.42', '0.58'),
    ('0.58', '0.71'),
    ('0.71', '0.90'),
    ('0.90', '1.14'),
    ('1.14', '2.00')
]

# Galaxy density in each tomographic bin
n_arcmin2 = np.array([1.7698, 1.6494, 1.4974, 1.4578, 1.3451, 1.0682]) # per arcmin^2

def calculate_tomo_nz(data_dir, n_los_chi, los_z_integration, shift_nz=True):
    tomo_nz = np.zeros((nbins, n_los_chi))
        
    if shift_nz:
        dz_biases = np.loadtxt(f'{data_dir}/nofzs/dz/Nz_biases.txt')
        dz_cov = np.loadtxt(f'{data_dir}/nofzs/dz/Nz_covariance.txt')
        shift_dz_realised = np.random.multivariate_normal(mean=dz_biases, cov=dz_cov, size=1)[0]
            

    for i in range(nbins):
        hdu = np.loadtxt(f'{data_dir}/nofzs/nz/BLINDSHAPES_KIDS_Legacy_NS_shear_noSG_noWeiCut_newCut_blindABC_A1_rmcol_filt_lab_filt_lab_filt_PSF_RAD_calc_filt_ZB{str(ztomo_label[i][0]).split(".")[0]}p{str(ztomo_label[i][0]).split(".")[1]}t{str(ztomo_label[i][1]).split(".")[0]}p{str(ztomo_label[i][1]).split(".")[1]}_calib_goldwt_Nz.ascii').T
        z = hdu[0]
        if shift_nz:
            z_shifted = z - shift_dz_realised[i]
            n_z = np.interp(z, z_shifted, hdu[1])
            zmid = z[:-1] + 0.5*(z[1:] - z[:-1])
            dndz_interpolated = np.interp(los_z_integration, zmid, n_arcmin2[i]*n_z[:-1]/np.trapezoid((n_z[:-1]), zmid))
            tomo_nz[i] = np.clip(dndz_interpolated, 0, None)    
        else:
            zmid = z[:-1] + 0.5*(z[1:] - z[:-1])
            dndz_interpolated = np.interp(los_z_integration, zmid, n_arcmin2[i]*hdu[1][:-1]/np.trapezoid(hdu[1][:-1], zmid))
            tomo_nz[i] = np.clip(dndz_interpolated, 0, None)
    return tomo_nz