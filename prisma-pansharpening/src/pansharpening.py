from pathlib import Path


import math
import numpy as np
import rasterio
from rasterio.windows import Window
from skimage.transform import resize
from tqdm import tqdm
import logging
import time

logger = logging.getLogger()

def alpha_estimation(a, imageHR0):
    
    IHc = imageHR0.reshape((imageHR0.shape[0]*imageHR0.shape[1],1), order='F')
    ILRc = a.reshape((a.shape[0]*a.shape[1], a.shape[2]),order='F')
    alpha = np.linalg.lstsq(ILRc,IHc)
    alpha = alpha[0]
    
    return alpha

def orthogonalize_hyperspectral(HS, I):
    H, W, B = HS.shape
    HS_flat = HS.reshape(-1, B)  # (N, B)
    I_flat = I.flatten()         # (N,)

    means = np.mean(HS_flat, axis=0)
    HS_centered = HS_flat - means  # (N, B)

    HS_orth = np.zeros_like(HS_centered)
    basis = []
    proj_coeffs = []

    for b in range(B):
        v = HS_centered[:, b].copy()

        # Transform coeff with Panchromatic
        coeff_I = np.dot(v, I_flat) / np.dot(I_flat, I_flat) 
        v = v - coeff_I * I_flat

        coeffs_prev = []
        for prev in basis:
            coeff = np.dot(v, prev) / np.dot(prev, prev)
            v = v - coeff * prev
            coeffs_prev.append(coeff)

        HS_orth[:, b] = v
        basis.append(v)
        proj_coeffs.append({
            "coeff_I": coeff_I,
            "coeffs_prev": coeffs_prev
        })

    return HS_orth.reshape(H, W, B), means, proj_coeffs


def read_hs_file(hs_file_path):
    """Read hyperspectral raster."""

    with rasterio.open(hs_file_path) as src:
        HS = src.read()

        profile = src.profile.copy()
        descriptions = src.descriptions
        nodata = src.nodata
        crs = src.crs
        transform = src.transform

    # Rasterio: (bands, rows, cols)
    # Convert to: (rows, cols, bands)
    HS = np.moveaxis(HS, 0, -1)

    return {
        "data": HS,
        "profile": profile,
        "descriptions": descriptions,
        "nodata": nodata,
        "crs": crs,
        "transform": transform,
    }


def read_pan_file(pan_file_path):
    """Read panchromatic raster."""

    with rasterio.open(pan_file_path) as src:
        PAN = src.read(1)

        profile = src.profile.copy()
        nodata = src.nodata
        crs = src.crs
        transform = src.transform

    return {
        "data": PAN,
        "profile": profile,
        "nodata": nodata,
        "crs": crs,
        "transform": transform,
    }


def calculate_synthetic_intensity(
    HS,
    PAN,
    alpha_estimation
):
    """Calculate the synthetic PAN/intensity image."""

    r_im, c_im, _ = HS.shape

    # Downsample PAN to HS resolution
    pan_in_down = resize(
        PAN,
        (r_im, c_im),
        order=2,
        preserve_range=True
    )

    # Center HS and PAN
    imageLR0 = HS - np.mean(
        HS,
        axis=(0, 1),
        keepdims=True
    )

    imageHR0 = pan_in_down - np.mean(pan_in_down)

    # Estimate optimal weights
    a = np.dstack(
        (
            imageLR0,
            np.ones((r_im, c_im))
        )
    )

    alpha = alpha_estimation(a, imageHR0)
    alpha = np.reshape(alpha, (1, 1, len(alpha)))

    # Calculate synthetic intensity
    kk2 = np.tile(
        alpha,
        (r_im, c_im, 1)
    )

    kk = np.dstack(
        (
            imageLR0,
            np.ones((r_im, c_im))
        )
    )

    I = np.sum(kk * kk2, axis=2)

    I0 = I - np.mean(I)

    return I, I0


def orthogonalize_hs(
    HS,
    I0,
    orthogonalize_hyperspectral
):
    """Orthogonalize HS bands using Gram-Schmidt."""

    HS_orth, means, coeffs = orthogonalize_hyperspectral(
        HS,
        I0
    )

    return HS_orth, means, coeffs


def adjust_pan(PAN, I):
    """Gain and bias adjust the high-resolution PAN."""

    gain = np.std(I) / np.std(PAN)
    bias = np.mean(I) - np.mean(PAN) * gain

    PAN_adjusted = PAN * gain + bias

    return PAN_adjusted, gain, bias


def save_single_band(
    array,
    output_path,
    profile,
    nodata=None
):
    """Save a single-band raster."""

    profile = profile.copy()

    profile.update(
        dtype=array.dtype,
        count=1,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        nodata=nodata
    )

    with rasterio.open(
        output_path,
        "w",
        **profile
    ) as dst:
        dst.write(array, 1)


def save_multiband(
    array,
    output_path,
    profile,
    descriptions=None,
    nodata=None
):
    """Save a multiband raster."""

    # Convert (rows, cols, bands)
    # -> (bands, rows, cols)
    if array.ndim == 3:
        array_rio = np.transpose(
            array,
            (2, 0, 1)
        )
    else:
        array_rio = array[np.newaxis, :, :]

    profile = profile.copy()

    profile.update(
        dtype=array_rio.dtype,
        count=array_rio.shape[0],
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        nodata=nodata
    )

    with rasterio.open(
        output_path,
        "w",
        **profile
    ) as dst:

        dst.write(array_rio)

        if descriptions is not None:
            for i, desc in enumerate(
                descriptions,
                start=1
            ):
                if desc is not None:
                    dst.set_band_description(i, desc)


def get_output_path(file_path, suffix):
    """Create an output path by adding a suffix before .tiff."""

    file_path = Path(file_path)

    return file_path.with_name(
        file_path.stem + suffix + file_path.suffix
    )

def get_scaled_slice(subset, ratio):
    (r_start, r_end), (c_start, c_end) = subset
    r_start_hr = int(r_start * ratio)
    r_end_hr = int(r_end * ratio)
    c_start_hr = int(c_start * ratio)
    c_end_hr = int(c_end * ratio)
    return ((r_start_hr, r_end_hr), (c_start_hr, c_end_hr))

def invert_orthogonalization(HS_orth, means, proj_coeffs, I_flat):
    H, W, B = HS_orth.shape
    HS_orth_flat = HS_orth.reshape(-1, B)
    reconstructed = np.zeros_like(HS_orth_flat)

    for b in range(B):
        v = HS_orth_flat[:, b].copy()
        v += proj_coeffs[b]["coeff_I"] * I_flat

        for i, coeff in enumerate(proj_coeffs[b]["coeffs_prev"]):
            v += coeff * HS_orth_flat[:, i]

        v += means[b]
        reconstructed[:, b] = v

    return reconstructed.reshape(H, W, B)


def pansharpening_2nd_stage(hs_ortho_file, pan_adj_file, wavelength_sel, fwhm_sel, means, coeffs, ratio):
    """
    Perform inversion of the pansharpening process

    Args:
        images: pairs of enmap and sentinel2 images

    Returns:
        pansharpened: Pansharpened images
    """

    logger.info("Starting pansharpening")

    with rasterio.open(hs_ortho_file) as hs_file:
        HS_crop = hs_file.read()
        band_descriptions = hs_file.descriptions

    with rasterio.open(pan_adj_file) as pan_file:
        arr_pan_in_adj = pan_file.read()
        arr_pan_in_adj = arr_pan_in_adj.squeeze()

        transform = pan_file.transform
        crs = pan_file.crs

    HS_crop = np.transpose(HS_crop, (1, 2, 0))
    r_im, c_im, b_im = HS_crop.shape
    r_pan, c_pan = arr_pan_in_adj.shape

    print('HS Shape:', HS_crop.shape)
    print('Pan Shape:', arr_pan_in_adj.shape)

    hs_pansharp_path = hs_ortho_file.parent / (
        hs_ortho_file.stem.replace('_CROPPED_ortho', '') + '_Pansharpened.tif'
    )

    print(hs_pansharp_path)

    with rasterio.open(
        hs_pansharp_path,
        "w",
        driver="GTiff",
        width=c_pan,
        height=r_pan,
        count=b_im,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=0,
    ) as dst:
        
        metadata = {
            "wavelength": "{" + ",".join(map(str, wavelength_sel)) + "}",
            "wavelength_units": "nanometers",
            "fwhm": "{" + ",".join(map(str, fwhm_sel)) + "}",
        }

        dst.update_tags(**metadata)

        for band_index, description in enumerate(
            band_descriptions,
            start=1,
        ):
            if description is not None:
                description = str(description)

                dst.set_band_description(
                    band_index,
                    description,
                )

                dst.update_tags(
                    band_index,
                    DESCRIPTION=description,
                )

        chunk_size = 128
        pad = 16
        
        if (HS_crop.shape[0] >= chunk_size) and (HS_crop.shape[1] >= chunk_size): 
            
            H, W = HS_crop.shape[:2]
            
            # Generate chunk indices
            range_i = np.arange(0, H // chunk_size) * chunk_size
            if H % chunk_size != 0:
                range_i = np.append(range_i, H - chunk_size)
            
            range_j = np.arange(0, W // chunk_size) * chunk_size
            if W % chunk_size != 0:
                range_j = np.append(range_j, W - chunk_size)
            
            total_chunks = len(range_i) * len(range_j)
            
            with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
                for i in range_i:
                    for j in range_j:
            
                        # --- Core chunk ---
                        r_start, r_end = i, i + chunk_size
                        c_start, c_end = j, j + chunk_size
            
                        # --- Padded chunk ---
                        r0 = max(0, r_start - pad)
                        r1 = min(H, r_end + pad)
                        c0 = max(0, c_start - pad)
                        c1 = min(W, c_end + pad)
            
                        HS_subset = HS_crop[r0:r1, c0:c1, :]
            
                        # --- Scale to PAN ---
                        pan_slice = get_scaled_slice(((r0, r1), (c0, c1)), ratio)
                        (r0_hr, r1_hr), (c0_hr, c1_hr) = pan_slice
            
                        arr_pan_subset = arr_pan_in_adj[r0_hr:r1_hr, c0_hr:c1_hr]
            
                        # Skip empty
                        if np.all(arr_pan_subset == 0):
                            pbar.update(1)
                            continue
            
                        nb = HS_subset.shape[2]
            
                        # --- Resize HS to PAN resolution ---
                        HS_hr = resize(
                            HS_subset,
                            (arr_pan_subset.shape[0], arr_pan_subset.shape[1], nb),
                            order=2,
                            preserve_range=True,
                        )
            
                        # --- Reconstruction ---
                        HS_reconstructed_full = invert_orthogonalization(
                            HS_hr, means, coeffs, arr_pan_subset.flatten()
                        )
            
                        # --- Crop back to core ---
                        row_offset = r_start - r0
                        col_offset = c_start - c0
            
                        row_offset_hr = int(row_offset * ratio)
                        col_offset_hr = int(col_offset * ratio)
                        chunk_size_hr = int(chunk_size * ratio)
            
                        HS_reconstructed = HS_reconstructed_full[
                            row_offset_hr:row_offset_hr + chunk_size_hr,
                            col_offset_hr:col_offset_hr + chunk_size_hr,
                            :
                        ]
            
                        # --- Write output ---
                        r_start_hr = int(r_start * ratio)
                        c_start_hr = int(c_start * ratio)

                        data = np.moveaxis(
                            HS_reconstructed,
                            -1,
                            0
                        ).astype(np.float32)
    
                        window = Window(
                            c_start_hr,
                            r_start_hr,
                            data.shape[2],
                            data.shape[1],
                        )
        
                        
                        dst.write(
                            data,
                            window=window)
            
                        time.sleep(0.01)
                        pbar.update(1)
        
        else:
            nb = HS_crop.shape[2]
        
            # --- Resize HS to PAN resolution ---
            HS_hr = resize(
                HS_crop,
                (arr_pan_in_adj.shape[0], arr_pan_in_adj.shape[1], nb),
                order=2,
                preserve_range=True,
            )
            
            I_flat = arr_pan_in_adj.flatten()
            
            # --- Reconstruction ---
            HS_reconstructed = invert_orthogonalization(
                HS_hr, means, coeffs, I_flat
            )
            
            H, W = arr_pan_in_adj.shape
            
            HS_reconstructed_full = HS_reconstructed.reshape(H, W, nb)
        
            data = np.moveaxis(
                HS_reconstructed_full,
                -1,
                0
            ).astype(np.float32)
    
            data = np.where(
                np.isnan(data),
                0,
                data
            )
    
            # Write all bands
            dst.write(data)


    logger.info("Pansharpening completed")

    return hs_pansharp_path
