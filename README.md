# Euclid DR3 forecasting   

Our end goal is to quantify the difference in constraining power in the cosmological parameters (Omega_m, sigma_8, w_0 and w_a) between performing angular power spectra SBI analysis and full field-level SBI analysis. In order to do so we wish to adap the KiDS methodology present in this repository. 

## Logic
The pipeline presented in this repository has 3 effective moving parts:
- master_kids_legacy_simulator.py: generates the LogNormal simulations used to perform the SBI from a given configuration file.
- train.py: trains the neuronal posterior estimation networks used to learn the parameters.
- eval.py: tests the obtained posterior at different conditional points.

## Euclid DR3 details
In order to simulate DR3 catalogs we will use the following specs all else the same to the current KiDS modelling:
- redshift distributions: euclid_nzs_dr3.txt
- footprint: 14.000 sqr degree reactangle along the equator
- galaxy number density: 6.2 galaxies per square arcminute
- ellipticity dispersion: 0.26
We want to run our forecast assuming Planck 2018 cosmology as the ground truth.

## Obstacles
1. The simulation script doesn't consider wa just w0 as a free parameter. We need to include this parameter an assign it a prior.
2. The fact that the Euclid DR3 area is much larger than that of KiDS introduces many problems. First the theory angular power spectrum used to generate the maps from which the LogNormal fields are sampled needs to be computed beyond the Limber approximation. This is specially true for the thin n(z) distributions of DR3. We reuse the existing non-Limber `prepare_glass_backend` path (`src/cosmology/sim_utils.py`, via the subprocess-isolated CAMB call in `src/cosmology/mpi_camb.py`) rather than approximating with Limber, since it's already fully generic and already run at scale for KiDS.
3. Similarly, the neuronal posterior estimator networks in train.py operate in cartesian space, different from the curved space where our simulations live. For KiDS we divided the footprint in two patches, north and south, to reduce projection effects. However, we are hoping to be able to get away with just one patch for this exercise. Alternatively, we could split the North and South hemispheres of our equator band into two bands. Then rotate each band to be centered at the equator to minimzied errors. This would use the current code infrastructure used for KiDS.
4. Eval doesn't save the samples of the posterior which we need for plotting.

## Change Log:
- master_euclid_simulator: copy of KiDS simulator with Euclid nz and footprint
- cosmology/parameter.py: add ppf model to allow phanton crossing as a result of sampling w0/wa
- cosmology/pixelise_maps.py: Added get_recentred_patch_values() which rotates the patches of the Euclid mask to equator.
- cosmology/gower_street.py: added wa to params
- cofig/experiments/py: added Euclid experiment to learn only w0, wa, Omega_m, sigma_8