import numpy as np
import healpy as hp

from .simulators import Rotator, generate_rotation_specs


def get_recentred_patch_values(map_data_bins, center_dec_deg, patch_halfwidth_deg, nside, lmax, sample="nearest"):
    """
    Rotate a Dec-band of HEALPix maps so that its declination center lands on
    the celestial equator, then crop the (now centered) full-RA strip into a
    flat Cartesian grid via get_patch_values.

    Unlike KiDS's named_patches (fixed lon/lat crop boxes, never actually
    recentred - see src/KiDS/simulation_config.py), this performs a genuine
    per-patch equatorial recentring rotation before cropping, to minimize
    flat-sky projection error for patches away from Dec=0.

    Parameters
    ----------
    map_data_bins : array-like, shape (nbins, npix)
        HEALPix maps per tomographic bin, at resolution `nside`.
    center_dec_deg : float
        Declination (deg) of the patch's center before recentring.
    patch_halfwidth_deg : float
        Half the patch's total Dec extent (deg); the cropped box spans
        [-patch_halfwidth_deg, +patch_halfwidth_deg] around the equator
        after recentring, and the full 360 deg in RA.
    nside : int
        HEALPix nside of map_data_bins.
    lmax : int
        Bandlimit to use for the spherical-harmonic-based rotation (should
        match the bandlimit the maps were actually synthesized at).
    sample : {"nearest","interp"}
        Passed through to get_patch_values.
    """
    rotator = Rotator(nside, lmax)
    spec = generate_rotation_specs(
        delta_deg=0.0, recenter_deg=-center_dec_deg, flips=(False,), base_angles=(0,), backend="alm",
    )[0]
    rotated = np.array([rotator.rotate_map_alm(m, spec["rot"], spec["flip"]) for m in map_data_bins])

    patch = (180.0, 0.0, 360.0, 2 * patch_halfwidth_deg)
    return get_patch_values(rotated, [patch], nside, rot_angle=0, sample=sample)[0]


def get_patch_values(map_data_bins, patches, nside, rot_angle, sample="nearest"):
    """
    Extract Cartesian patches from HEALPix maps.

    Parameters
    ----------
    map_data_bins : array-like, shape (nbins, npix)
        HEALPix maps per tomographic bin.
    patches : list of tuples
        Each as (lon_center, lat_center, lon_range, lat_range) in deg.
    nside : int
        HEALPix nside of map_data_bins.
    rot_angle : float
        Rotation (deg) applied to patch longitudes.
    sample : {"nearest","interp"}
        - "nearest": nearest-neighbour sampling via ang2pix (no interpolation)
        - "interp" : bilinear interpolation via get_interp_val
    """
    pixelised_tomobin_patches = []
    rotated_patches = [(patch[0] - rot_angle, *patch[1:]) for patch in patches]
    theta_pix_deg = np.degrees(hp.nside2resol(nside))  # native pixel size

    for center_lon, center_lat, lon_range, lat_range in rotated_patches:
        pixelised_tomobins = []
        for map_data in map_data_bins:
            # Define grid with step ~ native HEALPix pixel size
            lon_min, lon_max = center_lon - lon_range/2, center_lon + lon_range/2
            lat_min, lat_max = center_lat - lat_range/2, center_lat + lat_range/2
            dlon = theta_pix_deg
            dlat = theta_pix_deg
            lon = np.arange(lon_min, lon_max, dlon)
            lat = np.arange(lat_min, lat_max, dlat)
            lon_grid, lat_grid = np.meshgrid(lon, lat)

            # Convert to HEALPix spherical coordinates
            theta = np.radians(90.0 - lat_grid)             # colatitude
            phi   = np.radians((lon_grid + 360.0) % 360.0)  # longitude

            if sample == "interp":
                # Bilinear interpolation on the sphere
                values = hp.get_interp_val(map_data, theta.flatten(), phi.flatten())
                values = values.reshape(lon_grid.shape)
            else:
                # Nearest-neighbour sampling: take the pixel at each grid center
                pix = hp.ang2pix(nside, theta.flatten(), phi.flatten())
                values = map_data[pix].reshape(lon_grid.shape)

            pixelised_tomobins.append(values)

        stacked_values = np.stack(pixelised_tomobins)
        pixelised_tomobin_patches.append(stacked_values)

    return pixelised_tomobin_patches