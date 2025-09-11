import os
import numpy as np
import healpy as hp
import glass
import time
from abc import ABC, abstractmethod
from .nla import kappa_ia_nla_m
import healpy as hp
import numpy as np

def rotate_mask_array(mask, nside, rot_deg, delta_shift=0.0, flip=False):
    """
    Rotate a healpy mask array and return another mask array of same shape.

    Parameters
    ----------
    mask : ndarray (shape = 12*nside^2)
        Boolean or float mask array.
    nside : int
        Healpix nside.
    rot_deg : float
        Rotation angle (about x-axis here), in degrees.
    delta_shift : float, optional
        Extra shift in degrees to add to rotation (default 0).
    flip : bool, optional
        If True, apply phi → π − phi symmetry flip.

    Returns
    -------
    rotated_mask : ndarray
        Rotated mask array (same shape as input).
    """
    npix = hp.nside2npix(nside)

    # Get the indices of masked pixels
    mask_pix = np.where(mask > 0)[0]

    # Define rotation
    rot_angles = [rot_deg + delta_shift, 0, 0]
    rot = hp.Rotator(rot=rot_angles, deg=True)

    # Convert to spherical angles (radians)
    alpha, delta = hp.pix2ang(nside, mask_pix)

    # Apply rotation
    rot_alpha, rot_delta = rot(alpha, delta)

    if flip:
        rot_alpha = np.pi - rot_alpha

    # Back to pixel indices
    rotated_pix = hp.ang2pix(nside, rot_alpha, rot_delta)

    # Build rotated mask with same shape
    rotated_mask = np.zeros(npix, dtype=mask.dtype)
    rotated_mask[rotated_pix] = mask[mask_pix]

    return rotated_mask



class BaseSimulator(ABC):
    """
    Abstract base class for lensing catalogue simulation.
    Subclasses must implement get_matter_fields() to yield (index, delta_i).
    """
    def __init__(
        self,
        cosmo,
        los_z_integration,
        tomo_nz,
        galaxy_bias,
        shear_bias_params,
        nla_params,
        sigma_e,
        mask,
        nside,
        nbins,
        rng = None,
        debug=True
    ):
        # cosmology & geometry
        self.cosmo = cosmo
        self.los_z_integration = los_z_integration
        self.tomo_nz = tomo_nz

        # galaxy bias (separate)
        self.galaxy_bias = galaxy_bias
        # shear bias: dict with keys m_bias, c1_north, c2_north, c1_south, c2_south
        self.shear_bias = shear_bias_params
        # NLA parameters: dict with keys f_red, a_ia, b_ia, log10_M_eff
        self.nla = nla_params

        self.sigma_e = sigma_e
        self.mask = mask

        # resolution & bins
        self.nside = nside
        self.nbins = nbins

        # convergence engine
        self.convergence = glass.lensing.MultiPlaneConvergence(self.cosmo)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.ws = []
        self.debug = debug
        self.row_dtype = np.empty(0, dtype=[('RA', float), ('DEC', float),
                                        ('Z_TRUE', float), ('ZBIN', int),
                                        ('E1', float), ('E2', float)]).dtype

    @abstractmethod
    def get_matter_fields(self):
        """
        Yield tuples (i, delta_i) for each shell index and density contrast map.
        """
        pass
    def run(self, rotation_angles=[0], num_shape_noise_realisations=4):
        """
        Generate galaxy catalogues with rotations and shape-noise realizations.
        Preallocates storage so that the first dimension indexes each augmentation.
        """
        n_rot = len(rotation_angles)
        n_aug = n_rot * num_shape_noise_realisations

        # Preallocate a list for each augmentation
        all_catalogues_lists = [[] for _ in range(n_aug)]

        n_shells = len(self.ws)
        # mask = np.where(self.mask > 0)[0]

        for i, delta in self.get_matter_fields():
            if self.debug:
                print(f"Processing shell {i+1}/{n_shells}...", flush=True)
                start_time = time.time()

            # Lens planes
            self.convergence.add_window(delta, self.ws[i])
            kappa = self.convergence.kappa

            for tomo in range(self.nbins):
                z_vals, dndz = glass.shells.restrict(
                    self.los_z_integration,
                    self.tomo_nz[tomo],
                    self.ws[i]
                )
                ngal = np.trapezoid(dndz, z_vals)
                z_eff = np.average(self.los_z_integration, weights=self.tomo_nz[tomo])

                # Intrinsic alignments
                kappa_ia = kappa_ia_nla_m(
                    delta,
                    z_eff,
                    self.nla['f_red'][tomo],
                    self.cosmo,
                    self.nla['a_ia'],
                    self.nla['b_ia'],
                    self.nla['log10_M_eff'][tomo]
                )
                kappa += kappa_ia
                g1, g2 = glass.lensing.shear_from_convergence(kappa)

                # --- Loop over rotations and shape-noise realizations ---
                for rot_idx, ang in enumerate(rotation_angles):
                    # Rotate mask pixels
                    rot_mask = rotate_mask_array(mask=self.mask, nside=self.nside, rot_deg=ang, flip=False)
                    # rot_mask = self.mask
                    for noise_idx in range(num_shape_noise_realisations):
                        aug_idx = rot_idx * num_shape_noise_realisations + noise_idx

                        # Sample galaxies
                        for lon, lat, count in glass.points.positions_from_delta(
                            ngal, delta, self.galaxy_bias, rot_mask,
                            rng=self.rng
                        ):
                            gal_z = glass.galaxies.redshifts_from_nz(
                                count, z_vals, dndz,
                                rng=self.rng
                            )
                            gal_eps = glass.shapes.ellipticity_intnorm(
                                count, self.sigma_e[tomo], rng=self.rng
                            )
                            shear = glass.galaxies.galaxy_shear(
                                lon, lat, gal_eps,
                                kappa, g1, g2
                            )

                            E1, E2 = self._apply_shear_bias(tomo, shear, lat)

                            # Undo rotation to original frame
                            # inv_rot = hp.Rotator(rot=[-ang, 0, 0], deg=True)
                            # ra_final, dec_final = inv_rot(lon, lat, lonlat=True)

                            # Append rows to preallocated augmentation array
                            rows = np.empty(count, dtype=self.row_dtype)

                            rows['RA'] = lon
                            rows['DEC'] = lat
                            rows['Z_TRUE'] = gal_z
                            rows['ZBIN'] = tomo
                            rows['E1'] = E1
                            rows['E2'] = E2
                            all_catalogues_lists[aug_idx].append(rows)

            if self.debug:
                print(f"Shell {i+1}/{n_shells} processed in {time.time() - start_time:.2f} seconds.")
        # all_catalogues = [np.concatenate(cat_list) for cat_list in all_catalogues_lists]
        print(f"Total augmentations generated: {n_aug}")
        return all_catalogues_lists



    def _apply_shear_bias(self, tomo, shear, lat):
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


class GowerStreetSimulator(BaseSimulator):
    """
    Simulator reading from a data directory of lightcone files.
    """
    def __init__(
        self,
        data_dir,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.steps, self.ws = self._load_shells(data_dir)

    def _load_shells(self, data_dir):
        zvals = np.genfromtxt(
            os.path.join(data_dir, 'z_values.txt'),
            delimiter=',', names=True
        )[::-1]
        steps, shells = [], []
        for row in zvals:
            step, zmin, zmax = int(row['Step']), row['z_near'], row['z_far']
            if zmin > 2: continue
            za = np.linspace(zmin, zmax, 100)
            wa = np.ones_like(za)
            shells.append(glass.shells.RadialWindow(za, wa, 0.5*(zmin+zmax)))
            steps.append(step)
        return steps, shells

    def get_matter_fields(self):
        for i, step in enumerate(self.steps):
            if self.debug:
                print(f"loading step {step} with zeff of {self.ws[i].zeff:.2f}, file run.{step:05d}.lightcone.npy")

            arr = np.load(
                os.path.join(self.data_dir, f'run.{step:05d}.lightcone.npy')
            )
            if hp.get_nside(arr) != self.nside:
                arr = hp.ud_grade(arr.astype(float), self.nside, power=-2)
            delta = arr / np.mean(arr) - 1
            yield i, delta


class GlassLogNormalSimulator(BaseSimulator):
    """
    Simulator using precomputed lognormal fields.
    """
    def __init__(
        self,
        matter_fields,
        ws,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._matter = matter_fields
        self.ws = ws

    def get_matter_fields(self):
        for i, delta in enumerate(self._matter):
            yield i, delta
