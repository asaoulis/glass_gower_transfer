import numpy as np
import healpy as hp

def get_patch_values(map_data_bins, patches, nside, rot_angle):
    pixelised_tomobin_patches = []
    rotated_patches = [(patch[0] -rot_angle, *patch[1:]) for patch in patches]
    theta_pix_deg = np.degrees(hp.nside2resol(nside))  # ~0.115° (~6.9 arcmin)
    for center_lon, center_lat, lon_range, lat_range in rotated_patches:
        pixelised_tomobins = []
        for map_data in map_data_bins:

            # Define grid with step ~ native HEALPix pixel size
            lon_min, lon_max = center_lon - lon_range/2, center_lon + lon_range/2
            lat_min, lat_max = center_lat - lat_range/2, center_lat + lat_range/2

            # Define grid with step ~ native HEALPix pixel size
            dlon = theta_pix_deg
            dlat = theta_pix_deg
            lon = np.arange(lon_min, lon_max, dlon)
            lat = np.arange(lat_min, lat_max, dlat)

            lon_grid, lat_grid = np.meshgrid(lon, lat)

            # Convert to HEALPix spherical coordinates
            theta = np.radians(90 - lat_grid)  # colatitude
            phi   = np.radians((lon_grid + 360) % 360)  # wrap to [0, 360)

            # Sample the HEALPix map
            values = hp.get_interp_val(map_data, theta.flatten(), phi.flatten())
            values = values.reshape(lon_grid.shape)

            # Save this patch for CNN (values is a 2D numpy array)
            pixelised_tomobins.append(values)
        stacked_values = np.stack(pixelised_tomobins)
        pixelised_tomobin_patches.append(stacked_values)

    return pixelised_tomobin_patches