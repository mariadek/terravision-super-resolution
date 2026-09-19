from pathlib import Path
import logging

import numpy as np
import rasterio
from rasterio.windows import Window
from skimage.transform import resize
from tqdm import tqdm

import enmap_pansharpening.pansharpening as pansharpening

logger = logging.getLogger(__name__)

NODATA = -32768.0
DEFAULT_CHUNK_SIZE = 128
DEFAULT_PADDING = 16
DEFAULT_SCALE_RATIO = 3

def to_int16(data: np.ndarray) -> np.ndarray:
    """
    Convert reconstructed hyperspectral data to int16.

    - Replace NaN and infinite values with NODATA.
    - Round valid floating-point values.
    - Clip values to the int16 range.
    - Preserve NODATA.
    """
    info = np.iinfo(np.int16)

    data = np.asarray(data, dtype=np.float64)

    invalid = ~np.isfinite(data) | (data == NODATA)

    data = np.nan_to_num(
        data,
        nan=NODATA,
        posinf=info.max,
        neginf=NODATA,
    )

    data = np.clip(
        np.rint(data),
        info.min,
        info.max,
    )

    data[invalid] = NODATA

    return data.astype(np.int16)

def read_reconstruction_inputs(
    hs_path: Path,
    pan_path: Path,
):
    """Read orthogonalized hyperspectral data and adjusted PseudoPAN."""

    with rasterio.open(hs_path) as src:
        hs_data = src.read()
        band_descriptions = src.descriptions

    with rasterio.open(pan_path) as src:
        pan_data = src.read(1)
        transform = src.transform
        crs = src.crs

    # Rasterio: (bands, rows, cols)
    # Processing: (rows, cols, bands)
    hs_data = np.moveaxis(
        hs_data,
        0,
        -1,
    )

    return (
        hs_data,
        pan_data,
        band_descriptions,
        transform,
        crs,
    )


def create_output_path(
    hs_path: Path,
    output_dir: Path,
) -> Path:
    """Create output path for the final pansharpened product."""

    hs_path = Path(hs_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        hs_path.stem
        .replace("_ORTHO_CROPPED", "")
        .replace("_COG", "")
    )

    return output_dir / f"PANSHARP_{stem}.TIF"


def write_spectral_metadata(
    dst,
    wavelengths,
    fwhm,
    band_descriptions,
) -> None:
    """Write EnMAP spectral metadata to the output GeoTIFF."""

    metadata = {
        "wavelength": (
            "{" + ",".join(map(str, wavelengths)) + "}"
        ),
        "wavelength_units": "nanometers",
        "fwhm": (
            "{" + ",".join(map(str, fwhm)) + "}"
        ),
    }

    dst.update_tags(**metadata)

    for band_index, description in enumerate(
        band_descriptions,
        start=1,
    ):
        if description is not None:
            dst.set_band_description(
                band_index,
                description,
            )


def get_chunk_positions(
    size: int,
    chunk_size: int,
) -> list[int]:

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be positive"
        )

    return list(
        range(0, size, chunk_size)
    )


def reconstruct_hyperspectral(
    hs_data: np.ndarray,
    pan_data: np.ndarray,
    means,
    coeffs,
) -> np.ndarray:
    """Upsample and reconstruct one hyperspectral region."""

    number_of_bands = hs_data.shape[2]

    hs_high_resolution = resize(
        hs_data,
        (
            pan_data.shape[0],
            pan_data.shape[1],
            number_of_bands,
        ),
        order=2,
        preserve_range=True,
    )

    reconstructed = pansharpening.invert_orthogonalization(
        hs_high_resolution,
        means,
        coeffs,
        pan_data.flatten(),
    )

    return reconstructed


def process_full_image(
    dst,
    hs_data: np.ndarray,
    pan_data: np.ndarray,
    means,
    coeffs,
) -> None:
    """Reconstruct and write an image that does not require chunking."""

    number_of_bands = hs_data.shape[2]

    reconstructed = reconstruct_hyperspectral(
        hs_data,
        pan_data,
        means,
        coeffs,
    )

    reconstructed = reconstructed.reshape(
        pan_data.shape[0],
        pan_data.shape[1],
        number_of_bands,
    )

    data = np.moveaxis(
        reconstructed,
        -1,
        0,
    )

    data = to_int16(data)

    dst.write(data)


def process_chunk(
    dst,
    hs_data,
    pan_data,
    means,
    coeffs,
    row_start,
    col_start,
    chunk_size,
    padding,
    scale_ratio,
) -> None:
    """Reconstruct and write one padded hyperspectral chunk."""

    height, width = hs_data.shape[:2]

    row_end = min(
        row_start + chunk_size,
        height,
    )
    col_end = min(
        col_start + chunk_size,
        width,
    )

    # Padded low-resolution region
    row0 = max(
        0,
        row_start - padding,
    )
    row1 = min(
        height,
        row_end + padding,
    )

    col0 = max(
        0,
        col_start - padding,
    )
    col1 = min(
        width,
        col_end + padding,
    )

    hs_subset = hs_data[
        row0:row1,
        col0:col1,
        :,
    ]

    (
        (row0_hr, row1_hr),
        (col0_hr, col1_hr),
    ) = pansharpening.get_scaled_slice(
        (
            (row0, row1),
            (col0, col1),
        ),
        scale_ratio,
    )

    pan_subset = pan_data[
        row0_hr:row1_hr,
        col0_hr:col1_hr,
    ]

    if pan_subset.size == 0:
        return

    if np.all(pan_subset == 0):
        return

    reconstructed = reconstruct_hyperspectral(
        hs_subset,
        pan_subset,
        means,
        coeffs,
    )

    # Position of the core area inside the padded region
    row_offset = row_start - row0
    col_offset = col_start - col0

    row_offset_hr = row_offset * scale_ratio
    col_offset_hr = col_offset * scale_ratio

    core_height = (
        row_end - row_start
    ) * scale_ratio

    core_width = (
        col_end - col_start
    ) * scale_ratio

    reconstructed_core = reconstructed[
        row_offset_hr:
        row_offset_hr + core_height,
        col_offset_hr:
        col_offset_hr + core_width,
        :,
    ]

    # Convert reconstructed core to Rasterio layout:
    # (bands, rows, cols)

    data = np.moveaxis(
        reconstructed_core,
        -1,
        0,
    )

    # Calculate output position
    out_row = row_start * scale_ratio
    out_col = col_start * scale_ratio

    # Determine available space in the output raster
    available_height = dst.height - out_row
    available_width = dst.width - out_col

    # Prevent writes outside the raster
    write_height = min(
        data.shape[1],
        available_height,
    )

    write_width = min(
        data.shape[2],
        available_width,
    )

    # Skip chunks entirely outside the raster
    if write_height <= 0 or write_width <= 0:
        logger.warning(
            "Skipping chunk outside raster: row=%s, col=%s",
            out_row,
            out_col,
        )
        return

    # Crop data to match the valid output window
    data = data[
        :,
        :write_height,
        :write_width,
    ]

    # Convert to int16
    data = to_int16(data)

    # Create the matching output window
    window = Window(
        col_off=out_col,
        row_off=out_row,
        width=write_width,
        height=write_height,
    )

    # Write safely
    dst.write(
        data,
        window=window,
    )


def process_chunked_image(
    dst,
    hs_data,
    pan_data,
    means,
    coeffs,
    chunk_size=DEFAULT_CHUNK_SIZE,
    padding=DEFAULT_PADDING,
    scale_ratio=DEFAULT_SCALE_RATIO,
) -> None:
    """Process a large hyperspectral image using overlapping chunks."""

    height, width = hs_data.shape[:2]

    row_positions = get_chunk_positions(
        height,
        chunk_size,
    )

    col_positions = get_chunk_positions(
        width,
        chunk_size,
    )

    total_chunks = (
        len(row_positions)
        * len(col_positions)
    )

    with tqdm(
        total=total_chunks,
        desc="Pansharpening",
    ) as progress:

        for row_start in row_positions:
            for col_start in col_positions:

                process_chunk(
                    dst=dst,
                    hs_data=hs_data,
                    pan_data=pan_data,
                    means=means,
                    coeffs=coeffs,
                    row_start=row_start,
                    col_start=col_start,
                    chunk_size=chunk_size,
                    padding=padding,
                    scale_ratio=scale_ratio,
                )

                progress.update(1)


def reconstruct_single_image(
    hs_ortho_file,
    pan_adjusted_file,
    means,
    coeffs,
    wavelengths,
    fwhm,
    output_dir,
    chunk_size=DEFAULT_CHUNK_SIZE,
    padding=DEFAULT_PADDING,
    scale_ratio=DEFAULT_SCALE_RATIO,
) -> Path:
    """Perform second-stage reconstruction for one EnMAP/Sentinel-2 pair."""

    hs_ortho_file = Path(
        hs_ortho_file
    )

    pan_adjusted_file = Path(
        pan_adjusted_file
    )

    (
        hs_data,
        pan_data,
        band_descriptions,
        transform,
        crs,
    ) = read_reconstruction_inputs(
        hs_ortho_file,
        pan_adjusted_file,
    )

    output_path = create_output_path(
        hs_ortho_file, output_dir
    )

    number_of_bands = hs_data.shape[2]

    logger.info(
        "Creating pansharpened product: %s",
        output_path,
    )

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        width=pan_data.shape[1],
        height=pan_data.shape[0],
        count=number_of_bands,
        dtype="int16",
        crs=crs,
        transform=transform,
        nodata=NODATA,
    ) as dst:

        write_spectral_metadata(
            dst,
            wavelengths,
            fwhm,
            band_descriptions,
        )

        if (
            hs_data.shape[0] >= chunk_size
            and hs_data.shape[1] >= chunk_size
        ):
            process_chunked_image(
                dst=dst,
                hs_data=hs_data,
                pan_data=pan_data,
                means=means,
                coeffs=coeffs,
                chunk_size=chunk_size,
                padding=padding,
                scale_ratio=scale_ratio,
            )

        else:
            process_full_image(
                dst=dst,
                hs_data=hs_data,
                pan_data=pan_data,
                means=means,
                coeffs=coeffs,
            )

    return output_path


def reconstruct_pansharpened_images(
    images,
    pansharpening_products,
    output_dir
):
    """Perform second-stage reconstruction for all image pairs."""

    logger.info(
        "Starting final pansharpening reconstruction"
    )

    outputs = []

    for (
        hs_ortho_file,
        pan_adjusted_file,
    ), params in zip(
        images,
        pansharpening_products,
    ):

        output_path = reconstruct_single_image(
            hs_ortho_file=hs_ortho_file,
            pan_adjusted_file=pan_adjusted_file,
            means=params.means,
            coeffs=params.coeffs,
            wavelengths=params.wavelength,
            fwhm=params.fwhm,
            output_dir = output_dir
        )

        outputs.append(
            output_path
        )

    logger.info(
        "Final pansharpening reconstruction completed"
    )

    return outputs