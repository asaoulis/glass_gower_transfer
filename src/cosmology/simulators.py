import os
import numpy as np
import healpy as hp
import glass
import time
from abc import ABC, abstractmethod
from .nla import kappa_ia_nla_m
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import camb
from .systematics import NoSystematics
ZERO_TOL = 1e-12   # numerical tolerance for "zero"


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
        sigma_e,
        mask,
        nside,
        nbins,
        rng = None,
        debug=True,
        systematics=None
    ):
        # cosmology & geometry
        self.cosmo = cosmo
        self.los_z_integration = los_z_integration
        self.tomo_nz = tomo_nz

        # galaxy bias (separate)
        self.galaxy_bias = galaxy_bias
        # shear bias: dict with keys m_bias, c1_north, c2_north, c1_south, c2_south
        # NLA parameters: dict with keys f_red, a_ia, b_ia, log10_M_eff

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

        self.systematics = systematics or NoSystematics()
        print(f"Using systematics model: {self.systematics}", flush=True)

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
        print("Original simulations fixed kappa, partition version", flush=True)
        ngals_per_bin = [glass.partition(self.los_z_integration, self.tomo_nz[i], self.ws) for i in range(len(self.tomo_nz))]
        for i, delta in self.get_matter_fields():
            # hp.mollview(np.log10(delta+1), title=f"Delta shell {i}")
            if self.debug:
                print(f"Processing shell {i+1}/{n_shells}...", flush=True)
                start_time = time.time()

            # Lens planes
            self.convergence.add_window(delta, self.ws[i])
            kappa = self.convergence.kappa.copy()
            # hp.mollview(kappa, title=f"Kappa shell {i}")
            for tomo in range(self.nbins):
                z_vals, dndz = glass.shells.restrict(
                    self.los_z_integration,
                    self.tomo_nz[tomo],
                    self.ws[i]
                )
                ngal = np.trapezoid(dndz, z_vals)

                z_eff = np.average(self.los_z_integration, weights=self.tomo_nz[tomo])
                # z_eff = np.average(z_vals, weights=dndz)
                # z_eff = self.ws[i].zeff
                # Intrinsic alignments
                kappa_i = self.systematics.apply_intrinsic_alignments(
                    delta=delta,
                    kappa=kappa,
                    z_eff=z_eff,
                    tomo=tomo,
                )
                gamma_i, = glass.lensing.from_convergence(kappa_i, shear=True)

                # --- Loop over rotations and shape-noise realizations ---
                for rot_idx, ang in enumerate(rotation_angles):
                    # Rotate mask pixels
                    # rot_mask = rotate_mask_array(mask=self.mask, nside=self.nside, rot_deg=ang, flip=False)
                    # self.mask
                    # rot_mask = self.mask
                    for noise_idx in range(num_shape_noise_realisations):
                        aug_idx = rot_idx * num_shape_noise_realisations + noise_idx

                        # Sample galaxies
                        for lon, lat, count in glass.points.positions_from_delta(
                            ngal, delta, self.galaxy_bias, self.mask,
                            # rng=self.rng
                        ):
                            # gal_z = glass.galaxies.redshifts_from_nz(
                            #     count, z_vals, dndz,
                            #     rng=self.rng
                            # )
                            gal_z = glass.galaxies.redshifts(count,self.ws[i], rng=self.rng)
                            gal_eps = glass.shapes.ellipticity_intnorm(
                                count, self.sigma_e[tomo], rng=self.rng
                            )
                            shear = glass.galaxies.galaxy_shear(
                                lon, lat, gal_eps,
                                kappa_i, gamma_i.real, gamma_i.imag
                            )
                            # Apply shear bias
                            # E1, E2 = shear.real, shear.imag
                            E1, E2 = self.systematics.apply_shear_bias(tomo, shear, lat)

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



    # def _apply_shear_bias(self, tomo, shear, lat):
    #     # hemisphere split
    #     north = np.abs(lat) < 15
    #     south = ~north
    #     c_bias = np.zeros_like(shear)
    #     c_bias[north] = self.shear_bias['c1_north'][tomo] + 1j*self.shear_bias['c2_north'][tomo]
    #     c_bias[south] = self.shear_bias['c1_south'][tomo] + 1j*self.shear_bias['c2_south'][tomo]
    #     m = self.shear_bias['m_bias'][tomo]
    #     E1 = (1 + m)*shear.real + c_bias.real
    #     E2 = (1 + m)*shear.imag + c_bias.imag
    #     return E1, E2


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
    # COPIED FROM GLASS CODE EXAMPLES
    # def _load_shells(self, data_dir):
    #     zvals = np.genfromtxt(
    #         os.path.join(data_dir, 'z_values.txt'),
    #         delimiter=',', names=True
    #     )[::-1]
    #     wht = np.ones_like
    #     dz = 0.001
    #     steps, shells = [], []
    #     for row in zvals:
    #         step, zmin, zmax = int(row['Step']), row['z_near'], row['z_far']
    #         if zmin > 2: break
    #         n = max(round((zmax - zmin) / dz), 2)
    #         z = np.linspace(zmin, zmax, n, dtype=np.float64)
    #         # za = np.linspace(zmin, zmax, 100)
    #         w = wht(z)
    #         zeff = np.trapezoid(w * z, z) / np.trapezoid(w, z)
    #         shells.append(glass.shells.RadialWindow(z, w, zeff.item()))
    #         steps.append(step)
    #     return steps, shells

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

from tqdm import tqdm

class FieldSimulator(GlassLogNormalSimulator):

    def run(self):
        n_shells = len(self.ws)
        # mask = np.where(self.mask > 0)[0]
        print("Original simulations fixed kappa, partition version, lmax=1600, glass ngal example", flush=True)
        ngals_per_bin = [glass.partition(self.los_z_integration, self.tomo_nz[i], self.ws) for i in range(len(self.tomo_nz))]
        kappa_tot = np.zeros((len(self.tomo_nz), hp.nside2npix(self.nside)))
        gamma_tot = np.zeros((len(self.tomo_nz), hp.nside2npix(self.nside),2))
        ntot = np.zeros(len(self.tomo_nz))
        for i, delta in self.get_matter_fields():
            self.convergence.add_window(delta, self.ws[i])
            kappa = self.convergence.kappa.copy()
            for tomo in range(self.nbins):
                z_vals, dndz = glass.shells.restrict(
                    self.los_z_integration,
                    self.tomo_nz[tomo],
                    self.ws[i]
                )
                ngal = np.trapezoid(dndz, z_vals)
                # ngal = ngals_per_bin[tomo][i]
                if ngal <= ZERO_TOL:
                    continue
                kappa_i = kappa
                gamma_i, = glass.lensing.from_convergence(kappa_i, lmax=1600, shear=True)
                kappa_tot[tomo] += kappa_i * ngal
                gamma_tot[tomo,:,0] += gamma_i.real * ngal
                gamma_tot[tomo,:,1] += gamma_i.imag * ngal
                ntot[tomo] += ngal
        
        print("Total galaxies per bin:", ntot)
        return kappa_tot, gamma_tot, ntot

import numpy as np
import healpy as hp
import glass
class GowerStreetMatterPower(GlassLogNormalSimulator):
    """
    Empirical 2D matter power spectrum per shell,
    using the GlassLogNormalSimulator API.


    Works for:
      - lognormal fields
      - N-body shells (if provided as delta maps)

    Assumes:
      - full sky
      - thin-shell projection
    """

    def __init__(
        self,
        *args,
        lmax=1600,
        pixwin_correction=True,
        debug=True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lmax = lmax
        self.pixwin_correction = pixwin_correction
        self.debug = debug

    def shell_cls(self, start_index=None, break_index=None):
        """Compute empirical C_ell for each shell.

        Parameters
        ----------
        start_index : int or None
            Index of the first shell to process (inclusive). If None, start at 0.
        break_index : int or None
            Index of the last shell to process (inclusive). If None, go to the
            last available shell.

        Returns
        -------
        cls : ndarray, shape (n_shells, lmax+1)
            cls[i, ell] = C_ell of shell i, where i runs from start_index to
            break_index (or to the last shell if break_index is None).
        """
        n_shells_total = len(self.ws)
        if start_index is None:
            start_index = 0
        if break_index is None or break_index >= n_shells_total:
            break_index = n_shells_total - 1

        if start_index < 0 or start_index > break_index:
            raise ValueError("Invalid start_index / break_index combination")

        n_shells = break_index - start_index + 1
        cls = np.zeros((n_shells, self.lmax + 1))

        for i, delta in self.get_matter_fields():
            # Skip shells before start_index
            if i < start_index:
                continue
            # Stop once we passed break_index
            if i > break_index:
                break

            out_idx = i - start_index

            if self.debug:
                print(
                    f"[shell_cls] shell {i+1}/{n_shells_total} "
                    f"(z_eff={self.ws[i].zeff:.2f})",
                    flush=True,
                )

            # Ensure correct nside
            if hp.get_nside(delta) != self.nside:
                delta = hp.ud_grade(delta.astype(float), self.nside, power=-2)

            cl = hp.anafast(delta, lmax=self.lmax)

            if self.pixwin_correction:
                pixwin = hp.pixwin(self.nside, lmax=self.lmax)
                cl /= pixwin**2

            cls[out_idx] = cl

        return cls

    def camb_cls(self, pars, start_index=None, break_index=None):
        """CAMB prediction for the same shells.

        Parameters
        ----------
        start_index : int or None
            First shell index to include (inclusive).
        break_index : int or None
            Last shell index to include (inclusive).

        Returns
        -------
        cls_camb : ndarray
            CAMB C_ell for shells in the range [start_index, break_index].
        """
        n_shells_total = len(self.ws)
        if start_index is None:
            start_index = 0
        if break_index is None or break_index >= n_shells_total:
            break_index = n_shells_total - 1

        if start_index < 0 or start_index > break_index:
            raise ValueError("Invalid start_index / break_index combination")

        new_pars = self.update_pars_config(pars)
        shells = self.ws[start_index : break_index + 1]
        return glass.ext.camb.matter_cls(new_pars, self.lmax, shells)

    def update_pars_config(self, old_pars):
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

        tr.high_precision = False   # CHANGED
        tr.accurate_massive_neutrinos = False

        tr.kmax = 0.9               # CHANGED
        tr.k_per_logint = 0
        tr.PK_num_redshifts = 1     # CHANGED
        tr.PK_redshifts = [0.0]    # CHANGED
        pars.validate()
        return pars

    def compare_with_camb(self, pars, start_index=None, break_index=None):
        """Return (empirical_cls, camb_cls) aligned as (n_shells, lmax+1).

        Parameters
        ----------
        start_index : int or None
            First shell index to include (inclusive).
        break_index : int or None
            Last shell index to include (inclusive).
        """
        cls_emp = self.shell_cls(start_index=start_index, break_index=break_index)
        cls_camb = self.camb_cls(pars, start_index=start_index, break_index=break_index)
        return cls_emp, cls_camb
