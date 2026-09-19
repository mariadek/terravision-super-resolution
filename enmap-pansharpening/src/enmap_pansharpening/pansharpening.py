import numpy as np
import logging
from pathlib import Path
import math
import rasterio
from skimage.transform import resize
from enmap_pansharpening.download.models import PreprocessedPair

from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PansharpeningParameters:
    means: np.ndarray
    coeffs: np.ndarray
    wavelength: list[str]
    fwhm: list[float]

@dataclass
class PansharpeningResult:
    hs_path: Path
    pan_path: Path
    means: np.ndarray
    coeffs: np.ndarray

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


def save_adjusted_pan(
    pan_path,
    pan_adjusted,
    pan,
):
    """Save the adjusted PseudoPAN image."""

    output_path = get_output_path(
        pan_path,
        "_adjusted",
    )

    save_single_band(
        pan_adjusted,
        output_path,
        pan["profile"],
        pan["nodata"],
    )

    return output_path

def save_orthogonalized_hs(
    hs_path,
    hs_orthogonalized,
    hs,
):
    """Save the orthogonalized hyperspectral image."""

    output_path = get_output_path(
        hs_path,
        "_ortho",
    )

    save_multiband(
        hs_orthogonalized,
        output_path,
        hs["profile"],
        hs["descriptions"],
        hs["nodata"],
    )

    return output_path

def process_pansharpening_pair(
    hs_path,
    pan_path,
) -> PansharpeningResult:

    hs = read_hs_file(hs_path)
    pan = read_pan_file(pan_path)

    intensity, intensity_lowres = (
        calculate_synthetic_intensity(
            hs["data"],
            pan["data"],
            alpha_estimation,
        )
    )

    hs_orthogonalized, means, coeffs = (
        orthogonalize_hyperspectral(
            hs["data"],
            intensity_lowres,
        )
    )

    pan_adjusted, _, _ = adjust_pan(
        pan["data"],
        intensity,
    )

    hs_output = save_orthogonalized_hs(
        hs_path,
        hs_orthogonalized,
        hs,
    )

    pan_output = save_adjusted_pan(
        pan_path,
        pan_adjusted,
        pan,
    )

    return PansharpeningResult(
        hs_path=hs_output,
        pan_path=pan_output,
        means=means,
        coeffs=coeffs,
    )

def pansharpen(
    pair: PreprocessedPair,
):
    """Perform first-stage EnMAP pansharpening."""

    logger.info(
        "Starting first-stage pansharpening"
    )

    images = []
    products = []

    result = process_pansharpening_pair(
        pair.enmap_path,
        pair.sentinel2_pan_path,
    )

    images = [
            result.hs_path,
            result.pan_path,
        ]
    

    products = PansharpeningParameters(
            means=result.means,
            coeffs=result.coeffs,
            wavelength=pair.wavelength,
            fwhm=pair.fwhm,
        )

    logger.info(
        "First-stage pansharpening completed"
    )

    return images, products