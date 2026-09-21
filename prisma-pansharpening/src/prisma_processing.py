from pathlib import Path

import h5py
import numpy as np
import rasterio
from tqdm import tqdm
from rasterio.transform import from_origin
import logging

from pyproj import Transformer
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


def generate_PAN(file: str | Path, tmp_dir: str | Path) -> Path:
    """
    Generate a georeferenced PRISMA Panchromatic GeoTIFF
    inside a temporary processing directory.

    Args:
        file: Path to the PRISMA .he5 file.
        tmp_dir: Temporary output directory.

    Returns:
        Path to the generated PAN GeoTIFF.
    """

    file = Path(file)
    tmp_dir = Path(tmp_dir)

    if not file.is_file():
        raise FileNotFoundError(
            f"PRISMA file not found: {file}"
        )

    # Create temporary directory if necessary
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Automatically generate output filename
    output_path = tmp_dir / f"{file.stem}_PAN.tif"

    if output_path.exists():
        logger.info(
            "Panchromatic image already exists. Skipping panchromatic image creation: %s",
            output_path,
        )
        return str(output_path)

    cube_path = (
        "HDFEOS/SWATHS/PRS_L2D_PCO/"
        "Data Fields/Cube"
    )

    with h5py.File(file, "r") as h5f:

        if cube_path not in h5f:
            raise KeyError(
                f"Panchromatic dataset not found: {cube_path}"
            )

        # Read DN values
        pan_dn = h5f[cube_path][...]

        # Scale to reflectance
        scale_min = float(h5f.attrs["L2ScalePanMin"])
        scale_max = float(h5f.attrs["L2ScalePanMax"])

        pan_reflectance = (
            scale_min
            + pan_dn.astype(np.float32)
            * (scale_max - scale_min)
            / 65535.0
        )

        # Georeferencing
        epsg = int(h5f.attrs["Epsg_Code"])

        xmin = min(
            h5f.attrs["Product_ULcorner_easting"],
            h5f.attrs["Product_LLcorner_easting"],
        )

        ymax = max(
            h5f.attrs["Product_ULcorner_northing"],
            h5f.attrs["Product_URcorner_northing"],
        )

    # PRISMA PAN resolution
    resolution = 5.0

    transform = from_origin(
        float(xmin),
        float(ymax),
        resolution,
        resolution,
    )

    height, width = pan_reflectance.shape

    # Write temporary PAN GeoTIFF
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=f"EPSG:{epsg}",
        transform=transform,
        nodata=0,
        compress="deflate",
        tiled=True,
    ) as dst:

        dst.write(pan_reflectance, 1)
        dst.set_band_description(1, "Panchromatic")

    logger.info(
        "Temporary Panchromatic image created: %s",
        output_path,
    )

    return output_path

def generate_HS(
    file: str | Path,
    tmp_dir: str | Path,
) -> tuple[Path, np.ndarray, np.ndarray, list[float]]:
    """
    Generate a georeferenced and filtered PRISMA hyperspectral GeoTIFF.

    The VNIR and SWIR cubes are converted to reflectance, sorted by
    wavelength, filtered for invalid / water-vapour / defective bands,
    and written to a temporary GeoTIFF.

    Args:
        file:
            Path to the PRISMA .he5 file.
        tmp_dir:
            Temporary processing directory.

    Returns:
        tuple containing:
            output_path:
                Path to the generated hyperspectral GeoTIFF.
            wavelengths:
                Wavelengths of retained bands in nanometers.
            fwhm:
                FWHM values of retained bands.
            metadata:
                [
                    cloud_percentage,
                    sea_pixels_percentage,
                    sun_azimuth_angle,
                ]
    """

    file = Path(file)
    tmp_dir = Path(tmp_dir)

    # ---------------------------------------------------------
    # Validate paths
    # ---------------------------------------------------------

    if not file.is_file():
        raise FileNotFoundError(
            f"PRISMA file not found: {file}"
        )

    tmp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = tmp_dir / f"{file.stem}_HS.tif"

    if output_path.exists():
        logger.info(
            "Hyperspectral image already exists. "
            "Skipping hyperspectral image creation: %s",
            output_path,
        )

        # Read spectral metadata from existing GeoTIFF
        with rasterio.open(output_path) as src:
            tags = src.tags()

            wavelengths_filtered = np.array(
                [
                    float(value)
                    for value in tags["wavelength"].strip("{}").split(",")
                ],
                dtype=float,
            )

            fwhm_filtered = np.array(
                [
                    float(value)
                    for value in tags["fwhm"].strip("{}").split(",")
                ],
                dtype=float,
            )

        # Product-level metadata still comes from the PRISMA HE5 file
        with h5py.File(file, "r") as h5f:
            cloud_percentage = float(
                h5f.attrs["Cloudy_pixels_percentage"]
            )
            sea_pixels_percentage = float(
                h5f.attrs["Sea_pixels_percentage"]
            )
            sun_azimuth_angle = float(
                h5f.attrs["Sun_azimuth_angle"]
            )

        metadata = [
            cloud_percentage,
            sea_pixels_percentage,
            sun_azimuth_angle,
        ]

        return (
            output_path,
            wavelengths_filtered,
            fwhm_filtered,
            metadata,
        )

    # ---------------------------------------------------------
    # HDF5 dataset paths
    # ---------------------------------------------------------

    base_path = "HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields"

    vnir_cube_path = f"{base_path}/VNIR_Cube"
    swir_cube_path = f"{base_path}/SWIR_Cube"

    vnir_error_path = (
        f"{base_path}/VNIR_PIXEL_L2_ERR_MATRIX"
    )

    swir_error_path = (
        f"{base_path}/SWIR_PIXEL_L2_ERR_MATRIX"
    )

    # ---------------------------------------------------------
    # Read PRISMA data
    # ---------------------------------------------------------

    with h5py.File(file, "r") as h5f:

        required_datasets = [
            vnir_cube_path,
            swir_cube_path,
            vnir_error_path,
            swir_error_path,
        ]

        for dataset in required_datasets:
            if dataset not in h5f:
                raise KeyError(
                    f"Required PRISMA dataset not found: {dataset}"
                )

        # -----------------------------------------------------
        # Georeferencing
        # -----------------------------------------------------

        epsg = int(h5f.attrs["Epsg_Code"])

        xmin = min(
            h5f.attrs["Product_ULcorner_easting"],
            h5f.attrs["Product_LLcorner_easting"],
        )

        ymax = max(
            h5f.attrs["Product_ULcorner_northing"],
            h5f.attrs["Product_URcorner_northing"],
        )

        # -----------------------------------------------------
        # Product metadata
        # -----------------------------------------------------

        cloud_percentage = float(
            h5f.attrs["Cloudy_pixels_percentage"]
        )

        sea_pixels_percentage = float(
            h5f.attrs["Sea_pixels_percentage"]
        )

        sun_azimuth_angle = float(
            h5f.attrs["Sun_azimuth_angle"]
        )

        # -----------------------------------------------------
        # Spectral metadata
        # -----------------------------------------------------

        wavelengths = np.concatenate(
            [
                h5f.attrs["List_Cw_Vnir"][::-1],
                h5f.attrs["List_Cw_Swir"][::-1],
            ]
        ).astype(float)

        fwhm = np.concatenate(
            [
                h5f.attrs["List_Fwhm_Vnir"][::-1],
                h5f.attrs["List_Fwhm_Swir"][::-1],
            ]
        ).astype(float)

        flags = np.concatenate(
            [
                h5f.attrs["CNM_VNIR_SELECT"][::-1],
                h5f.attrs["CNM_SWIR_SELECT"][::-1],
            ]
        ).astype(float)

        # Sort all metadata using wavelength
        sorted_idx = np.argsort(wavelengths)

        wavelengths = wavelengths[sorted_idx]
        fwhm = fwhm[sorted_idx]
        flags = flags[sorted_idx]

        if not np.all(
            wavelengths[:-1] <= wavelengths[1:]
        ):
            raise ValueError(
                "PRISMA wavelengths could not be sorted correctly."
            )

        # -----------------------------------------------------
        # Read VNIR and SWIR cubes
        # -----------------------------------------------------

        vnir_dn = h5f[vnir_cube_path][...]
        swir_dn = h5f[swir_cube_path][...]

        # Reorganize dimensions to:
        #
        # rows x columns x bands
        #
        vnir_dn = np.swapaxes(
            vnir_dn,
            1,
            2,
        )[:, :, ::-1]

        swir_dn = np.swapaxes(
            swir_dn,
            1,
            2,
        )[:, :, ::-1]

        # -----------------------------------------------------
        # Convert DN to reflectance
        # -----------------------------------------------------

        vnir_min = float(
            h5f.attrs["L2ScaleVnirMin"]
        )
        vnir_max = float(
            h5f.attrs["L2ScaleVnirMax"]
        )

        swir_min = float(
            h5f.attrs["L2ScaleSwirMin"]
        )
        swir_max = float(
            h5f.attrs["L2ScaleSwirMax"]
        )

        vnir_reflectance = (
            vnir_min
            + vnir_dn.astype(np.float32)
            * (vnir_max - vnir_min)
            / 65535.0
        )

        swir_reflectance = (
            swir_min
            + swir_dn.astype(np.float32)
            * (swir_max - swir_min)
            / 65535.0
        )

        cube = np.concatenate(
            [
                vnir_reflectance,
                swir_reflectance,
            ],
            axis=2,
        )

        # Apply wavelength ordering
        cube = cube[:, :, sorted_idx]

        # -----------------------------------------------------
        # Initial invalid bands
        # -----------------------------------------------------

        zero_wavelength_indices = np.where(
            wavelengths == 0.0
        )[0]

        zero_flagged_indices = np.where(
            flags == 0
        )[0]

        # -----------------------------------------------------
        # Water-vapour bands
        # -----------------------------------------------------

        vnir_water_vapour = [
            423.78476,
            415.839,
            406.9934,
        ]

        swir_water_vapour = list(
            range(1350, 1480, 10)
        )

        water_vapour_wavelengths = (
            vnir_water_vapour
            + swir_water_vapour
        )

        water_vapour_indices = {
            int(
                np.abs(wavelengths - wavelength).argmin()
            )
            for wavelength in water_vapour_wavelengths
        }

        # -----------------------------------------------------
        # Pixel error masks
        # -----------------------------------------------------

        swir_error = h5f[swir_error_path][...][::-1]
        vnir_error = h5f[vnir_error_path][...][::-1]

        error_matrix = np.concatenate(
            [vnir_error, swir_error],
            axis=1,
        )

        assert error_matrix.shape == (
            cube.shape[0],
            cube.shape[2],
            cube.shape[1],
        )

        error_mask = error_matrix != 0

        # Mark defective pixels in the hyperspectral cube.
        cube[error_mask.transpose(0, 2, 1)] = 0

        # Fraction of defective spatial pixels per spectral band.
        defective_fraction = error_mask.mean(axis=(0, 2))

        invalid_bands = np.flatnonzero(
            defective_fraction >= 0.10
        ).tolist()

    # ---------------------------------------------------------
    # Combine bands to remove
    # ---------------------------------------------------------

    bands_to_remove = np.unique(
        np.concatenate(
            [
                zero_wavelength_indices,
                zero_flagged_indices,
                np.array(
                    list(water_vapour_indices),
                    dtype=int,
                ),
                np.array(
                    invalid_bands,
                    dtype=int,
                ),
            ]
        )
    )

    all_bands = np.arange(
        cube.shape[2]
    )

    bands_to_keep = np.setdiff1d(
        all_bands,
        bands_to_remove,
    )

    if len(bands_to_keep) == 0:
        raise ValueError(
            "No valid hyperspectral bands remain after filtering."
        )

    img_filtered = cube[
        :,
        :,
        bands_to_keep,
    ]

    wavelengths_filtered = wavelengths[
        bands_to_keep
    ]

    fwhm_filtered = fwhm[
        bands_to_keep
    ]

    logger.info(
        "Removed %d hyperspectral bands; %d bands remain.",
        len(bands_to_remove),
        len(bands_to_keep),
    )

    # ---------------------------------------------------------
    # Raster geotransform
    # ---------------------------------------------------------

    resolution = 30.0

    transform = from_origin(
        float(xmin),
        float(ymax),
        resolution,
        resolution,
    )

    # ---------------------------------------------------------
    # Write hyperspectral GeoTIFF
    # ---------------------------------------------------------

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=img_filtered.shape[0],
        width=img_filtered.shape[1],
        count=img_filtered.shape[2],
        dtype="float32",
        crs=f"EPSG:{epsg}",
        transform=transform,
        nodata=0,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        interleave="band",
    ) as dst:

        dst.update_tags(
            wavelength=(
                "{"
                + ",".join(
                    f"{value:.5f}"
                    for value in wavelengths_filtered
                )
                + "}"
            ),
            wavelength_units="nanometers",
            fwhm=(
                "{"
                + ",".join(
                    f"{value:.5f}"
                    for value in fwhm_filtered
                )
                + "}"
            ),
        )

        for band_idx in tqdm(
            range(img_filtered.shape[2]),
            desc="Writing HS bands",
        ):
            dst.write(
                img_filtered[:, :, band_idx],
                band_idx + 1,
            )

            dst.set_band_description(
                band_idx + 1,
                (
                    f"Band {bands_to_keep[band_idx] + 1} "
                    f"({wavelengths_filtered[band_idx]:.5f} nm)"
                ),
            )

    logger.info(
        "Temporary hyperspectral image created: %s",
        output_path,
    )

    metadata = [
        cloud_percentage,
        sea_pixels_percentage,
        sun_azimuth_angle,
    ]

    return (
        output_path,
        wavelengths_filtered,
        fwhm_filtered,
        metadata,
    )


def get_raster_footprint_wgs84(
    raster_path: str | Path,
) -> Polygon:
    """
    Get the footprint of a raster and transform it to WGS84.

    Args:
        raster_path:
            Path to the input raster.

    Returns:
        Shapely Polygon representing the raster footprint
        in EPSG:4326 (longitude, latitude).

    Raises:
        FileNotFoundError:
            If the raster does not exist.
        ValueError:
            If the raster has no CRS.
    """

    raster_path = Path(raster_path)

    if not raster_path.is_file():
        raise FileNotFoundError(
            f"Raster not found: {raster_path}"
        )

    # ---------------------------------------------------------
    # Read raster bounds and CRS
    # ---------------------------------------------------------

    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs

    if crs is None:
        raise ValueError(
            f"Raster has no CRS: {raster_path}"
        )

    # ---------------------------------------------------------
    # Transform coordinates to WGS84
    # ---------------------------------------------------------

    transformer = Transformer.from_crs(
        crs,
        "EPSG:4326",
        always_xy=True,
    )

    ul = transformer.transform(
        bounds.left,
        bounds.top,
    )

    ur = transformer.transform(
        bounds.right,
        bounds.top,
    )

    lr = transformer.transform(
        bounds.right,
        bounds.bottom,
    )

    ll = transformer.transform(
        bounds.left,
        bounds.bottom,
    )

    # ---------------------------------------------------------
    # Create footprint
    # ---------------------------------------------------------

    footprint = Polygon([
        ul,
        ur,
        lr,
        ll,
    ])

    return footprint