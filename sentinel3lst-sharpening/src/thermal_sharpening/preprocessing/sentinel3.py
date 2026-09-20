import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile


import netCDF4
import numpy as np
from osgeo import gdal


NODATA = -32768.0
OUTPUT_RESOLUTION = 1000
CREATION_OPTIONS = [
    "COMPRESS=DEFLATE",
    "INTERLEAVE=BAND",
    "PREDICTOR=2",
]

import logging

logger = logging.getLogger()

def _write_tiff(path, array, driver, nodata=None):
    """Write a 2D NumPy array to a single-band Float32 GeoTIFF."""
    ysize, xsize = array.shape

    dataset = driver.Create(
        str(path),
        xsize,
        ysize,
        1,
        gdal.GDT_Float32,
    )

    if dataset is None:
        raise RuntimeError(f"Could not create GeoTIFF: {path}")

    band = dataset.GetRasterBand(1)

    if nodata is not None:
        band.SetNoDataValue(nodata)

    band.WriteArray(array)
    band.FlushCache()

    dataset.FlushCache()
    dataset = None


def _read_nc_variable(path, variable_name, fill_value=None):
    """Read a NetCDF variable and return it as a NumPy array."""
    with netCDF4.Dataset(path, "r") as dataset:
        data = dataset.variables[variable_name][:]

    if np.ma.isMaskedArray(data):
        if fill_value is None:
            data = data.filled()
        else:
            data = data.filled(fill_value)

    return np.asarray(data)


def _create_geolocation_vrt(
    vrt_path,
    raster_filename,
    lon_path,
    lat_path,
    xsize,
    ysize,
):
    """Create a VRT using longitude/latitude rasters as geolocation arrays."""

    vrt = f"""<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">
  <Metadata domain="GEOLOCATION">
    <MDI key="X_DATASET">{lon_path}</MDI>
    <MDI key="X_BAND">1</MDI>
    <MDI key="Y_DATASET">{lat_path}</MDI>
    <MDI key="Y_BAND">1</MDI>
    <MDI key="PIXEL_OFFSET">0</MDI>
    <MDI key="LINE_OFFSET">0</MDI>
    <MDI key="PIXEL_STEP">1</MDI>
    <MDI key="LINE_STEP">1</MDI>
  </Metadata>

  <VRTRasterBand dataType="Float32" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{raster_filename}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SourceProperties
        RasterXSize="{xsize}"
        RasterYSize="{ysize}"
        DataType="Float32"
        BlockXSize="256"
        BlockYSize="256"
      />
      <SrcRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}" />
      <DstRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}" />
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
"""

    vrt_path.write_text(vrt)


def _warp(
    source,
    destination,
    *,
    dst_srs,
    geoloc=False,
    bounds=None,
    resolution=None,
    src_nodata=None,
    resample_alg=None,
):
    """Wrapper around gdal.Warp with error checking."""

    kwargs = {
        "dstSRS": dst_srs,
        "geoloc": geoloc,
    }

    if bounds is not None:
        kwargs["outputBounds"] = bounds

    if resolution is not None:
        kwargs["xRes"] = resolution
        kwargs["yRes"] = resolution

    if src_nodata is not None:
        kwargs["srcNodata"] = src_nodata

    if resample_alg is not None:
        kwargs["resampleAlg"] = resample_alg

    if not geoloc:
        kwargs["creationOptions"] = CREATION_OPTIONS

    result = gdal.Warp(
        str(destination),
        str(source),
        **kwargs,
    )

    if result is None:
        raise RuntimeError(
            f"GDAL Warp failed: {source} -> {destination}"
        )

    result.FlushCache()
    result = None


def s3_preprocessor(filename, highfile):
    """
    Reproject Sentinel-3 SLSTR LST and mask data to match a reference raster.

    Parameters
    ----------
    filename : str or Path
        Directory containing:
            geodetic_in.nc
            LST_in.nc
            flags_in.nc

    highfile : str or Path
        Reference raster defining the target projection and bounds.

    Returns
    -------
    tuple[Path, Path]
        Paths to the reprojected LST and mask GeoTIFFs.
    """
    source_dir = Path(filename)
    highfile = Path(highfile)

    required_files = [
        source_dir / "geodetic_in.nc",
        source_dir / "LST_in.nc",
        source_dir / "flags_in.nc",
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    if not highfile.exists():
        raise FileNotFoundError(f"Reference raster not found: {highfile}")

    driver = gdal.GetDriverByName("GTiff")
    if driver is None:
        raise RuntimeError("GDAL GeoTIFF driver is unavailable.")

    # ------------------------------------------------------------------
    # Read source data
    # ------------------------------------------------------------------

    longitude = _read_nc_variable(
        source_dir / "geodetic_in.nc",
        "longitude_in",
    )

    latitude = _read_nc_variable(
        source_dir / "geodetic_in.nc",
        "latitude_in",
    )

    lst = _read_nc_variable(
        source_dir / "LST_in.nc",
        "LST",
        fill_value=NODATA,
    )

    mask = _read_nc_variable(
        source_dir / "flags_in.nc",
        "bayes_in",
        fill_value=0,
    )

    # ------------------------------------------------------------------
    # Validate dimensions
    # ------------------------------------------------------------------

    if longitude.shape != latitude.shape:
        raise ValueError(
            "Longitude and latitude arrays have different shapes: "
            f"{longitude.shape} vs {latitude.shape}"
        )

    expected_shape = longitude.shape

    if lst.shape != expected_shape:
        raise ValueError(
            f"LST shape {lst.shape} does not match "
            f"geolocation shape {expected_shape}"
        )

    if mask.shape != expected_shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match "
            f"geolocation shape {expected_shape}"
        )

    ysize, xsize = expected_shape

    # ------------------------------------------------------------------
    # Read target raster geometry
    # ------------------------------------------------------------------

    reference = gdal.Open(str(highfile), gdal.GA_ReadOnly)

    if reference is None:
        raise RuntimeError(f"Could not open reference raster: {highfile}")

    projection = reference.GetProjection()
    transform = reference.GetGeoTransform()

    if not projection:
        raise ValueError(
            f"Reference raster has no projection: {highfile}"
        )

    minx = transform[0]
    maxy = transform[3]
    maxx = minx + transform[1] * reference.RasterXSize
    miny = maxy + transform[5] * reference.RasterYSize

    target_bounds = (minx, miny, maxx, maxy)

    reference = None

    # ------------------------------------------------------------------
    # Output filenames
    # ------------------------------------------------------------------

    scene_name = source_dir.stem

    output_lst = source_dir.parent / f"Subset_{scene_name}.tiff"
    output_mask = source_dir.parent / f"Subset_Flag_{scene_name}.tiff"

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    # TemporaryDirectory automatically removes everything, including when
    # an exception occurs.
    with tempfile.TemporaryDirectory(prefix="s3_preprocessor_") as tmp:
        work_dir = Path(tmp)

        lon_tif = work_dir / "lon.tif"
        lat_tif = work_dir / "lat.tif"
        data_tif = work_dir / "data.tif"
        mask_tif = work_dir / "mask.tif"

        data_vrt = work_dir / "data.vrt"
        mask_vrt = work_dir / "mask.vrt"

        geolocated_data = work_dir / "geolocated_data.tif"
        geolocated_mask = work_dir / "geolocated_mask.tif"

        warped_data = work_dir / "s3_slstr.tif"
        warped_mask = work_dir / "s3_slstr_mask.tif"

        # Write arrays to temporary GeoTIFFs.
        _write_tiff(lon_tif, longitude, driver)
        _write_tiff(lat_tif, latitude, driver)
        _write_tiff(data_tif, lst, driver, nodata=NODATA)

        # The original code created mask.tif but never wrote the mask.
        _write_tiff(mask_tif, mask, driver)

        # Create geolocation VRTs.
        _create_geolocation_vrt(
            data_vrt,
            data_tif.name,
            lon_tif,
            lat_tif,
            xsize,
            ysize,
        )

        _create_geolocation_vrt(
            mask_vrt,
            mask_tif.name,
            lon_tif,
            lat_tif,
            xsize,
            ysize,
        )

        # Convert swath/geolocation coordinates to EPSG:4326.
        _warp(
            data_vrt,
            geolocated_data,
            dst_srs="EPSG:4326",
            geoloc=True,
            src_nodata=NODATA,
        )

        _warp(
            mask_vrt,
            geolocated_mask,
            dst_srs="EPSG:4326",
            geoloc=True,
            resample_alg=gdal.GRA_NearestNeighbour,
        )

        # Reproject/crop to the target raster geometry.
        _warp(
            geolocated_data,
            warped_data,
            dst_srs=projection,
            bounds=target_bounds,
            resolution=OUTPUT_RESOLUTION,
            src_nodata=NODATA,
            resample_alg=gdal.GRA_Bilinear,
        )

        # Masks should normally use nearest-neighbour interpolation.
        _warp(
            geolocated_mask,
            warped_mask,
            dst_srs=projection,
            bounds=target_bounds,
            resolution=OUTPUT_RESOLUTION,
            resample_alg=gdal.GRA_NearestNeighbour,
        )

        # Move final products out of the temporary directory.
        shutil.move(str(warped_data), str(output_lst))
        shutil.move(str(warped_mask), str(output_mask))

    return output_lst, output_mask


def unzip_sentinel3(inputs3, temp_directory):
    """
    Unzip a Sentinel-3 SLSTR product into a tmp folder.

    Parameters
    ----------
    inputs3 : str or Path
        Path to the Sentinel-3 ZIP file.

    Returns
    -------
    Path
        Path to the extracted .SEN3 directory.
    """
    zip_file_path = Path(inputs3)

    # tmp folder next to the ZIP file
    temp_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    # Expected extracted Sentinel-3 directory
    extracted_zip_file_path = temp_directory / f"{zip_file_path.stem}.SEN3"

    if extracted_zip_file_path.exists():
        logger.info("Sentinel-3 SLSTR is already unzipped.")
        return extracted_zip_file_path

    if not zip_file_path.exists():
        raise FileNotFoundError(
            f"Sentinel-3 ZIP file not found: {zip_file_path}"
        )

    logger.info(f"Unzipping {zip_file_path.name} to {temp_directory}...")

    try:
        with ZipFile(zip_file_path, "r") as zip_obj:
            zip_obj.extractall(temp_directory)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to unzip Sentinel-3 product: {zip_file_path}"
        ) from exc

    if not extracted_zip_file_path.exists():
        raise FileNotFoundError(
            f"ZIP was extracted, but expected directory was not found: "
            f"{extracted_zip_file_path}"
        )

    logger.info("Sentinel-3 SLSTR successfully unzipped.")

    return extracted_zip_file_path