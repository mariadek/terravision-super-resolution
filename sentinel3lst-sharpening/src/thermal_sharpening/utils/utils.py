import os
import rasterio
import numpy as np
import scipy.ndimage as ndi
from osgeo import gdal
from numba import njit, stencil
from shapely.geometry import shape
from pathlib import Path

from typing import Tuple
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import subprocess
from rio_cogeo import cog_validate
import logging

logger = logging.getLogger(__name__)

def intersection_percentage(aoi_geojson, multipolygon_geojson):
    # Convert both geometries to shapely
    aoi_geom = shape(aoi_geojson)
    multipoly = shape(multipolygon_geojson)

    # Compute intersection
    intersection = aoi_geom.intersection(multipoly)

    if intersection.is_empty:
        return 0.0

    intersection_area = intersection.area
    aoi_area = aoi_geom.area

    return (intersection_area / aoi_area) * 100

def mask_resampling(mask, highresfile, output_dir):

    ds10 = gdal.Open(highresfile)
    ds20 = gdal.Open(mask)

    output = output_dir / Path(mask).name.replace(
        '_SCL_20m.jp2',
        '_SCL_10m.tiff'
    )

    driver = gdal.GetDriverByName('GTiff')
    # RasterXSize - columns
    # RasterYSize - rows
    outdata = driver.Create(
        str(output),
        ds10.RasterXSize,
        ds10.RasterYSize,
        1,
        gdal.GDT_UInt16,
        options=[
            "COMPRESS=DEFLATE",
            "TILED=YES",
            "PREDICTOR=2",
        ],
    )
    outdata.SetGeoTransform(ds10.GetGeoTransform())
    outdata.SetProjection(ds10.GetProjection())

    data20 = ds20.ReadAsArray(0, 0, ds20.RasterXSize, ds20.RasterYSize, buf_xsize = ds10.RasterXSize, buf_ysize = ds10.RasterYSize)

    outdata.WriteArray(data20, xoff = 0, yoff = 0)
    outdata.FlushCache()

    outdata = None
    data20 = None
    ds20 = None
    ds10 = None

    return output


def binomialSmoother(data):
    def filterFunction(footprint):
        weight = [1, 2, 1, 2, 4, 2, 1, 2, 1]
        # Don't smooth land and invalid pixels
        if np.isnan(footprint[4]):
            return footprint[4]

        footprintSum = 0
        weightSum = 0
        for i in range(len(weight)):
            # Don't use land and invalid pixels in smoothing of other pixels
            if not np.isnan(footprint[i]):
                footprintSum = footprintSum + weight[i] * footprint[i]
                weightSum = weightSum + weight[i]
        try:
            ans = footprintSum/weightSum
        except ZeroDivisionError:
            ans = footprint[4]
        return ans

    smoothedData = ndi.filters.generic_filter(data, filterFunction, 3)

    return smoothedData

@njit
def removeEdgeNaNs(a, i, j):
    values = np.array([a[i-1, j], a[i+1, j], a[i, j-1], a[i, j+1]])
    values = values[~np.isnan(values)]  # Remove NaN values manually
    if values.size == 0:
        return np.nan  # Return NaN if all elements are NaN
    return values.mean()  # Compute mean of non-NaN values

def create_lst_thumbnail(
    filepath: str,
    thumbnail_size: Tuple[int, int],
    p_min: float = 2,
    p_max: float = 98,
):
    """
    Generates LST thumbnail from .tiff file using Copernicus color ramp

    Inputs:
        - filepath (str): Full path to the input .tif file
        - thumbnail_size (Tuple[int, int]): Thumbnail size (Width, Height)
        - p_min (float): Lower percentile for normalization (default=2)
        - p_max (float): Upper percentile for normalization (default=98)
    """

    colors = [
        "#000080",
        "#0000FF",
        "#00FFFF",
        "#00FF00",
        "#FFFF00",
        "#FF8000",
        "#FF0000",
        "#800000",
    ]
    cmap = LinearSegmentedColormap.from_list("copernicus_lst", colors, N=256)

    with rasterio.open(filepath) as src:
        lst_data = src.read(1)
        nodata = src.nodata

        # Exclude NaN and the raster's NoData value
        valid_mask = ~np.isnan(lst_data)

        if nodata is not None:
            valid_mask &= lst_data != nodata

        valid_data = lst_data[valid_mask]

        if len(valid_data) == 0:
            raise ValueError("No valid data found")

        min_temp = np.percentile(valid_data, p_min)
        max_temp = np.percentile(valid_data, p_max)

        normalized = np.full_like(lst_data, np.nan)
        normalized[valid_mask] = np.clip((lst_data[valid_mask] - min_temp) / (max_temp - min_temp), 0, 1)

        rgba_array = cmap(normalized)
        rgb_array = np.zeros((normalized.shape[0], normalized.shape[1], 3), dtype=np.uint8)

        for i in range(3):
            channel = rgba_array[:, :, i] * 255
            channel[np.isnan(normalized)] = 0
            rgb_array[:, :, i] = channel.astype(np.uint8)

        image = Image.fromarray(rgb_array)
        # Resize
        thumbnail = image.copy()
    
        thumbnail.thumbnail(
            thumbnail_size,
            Image.Resampling.LANCZOS
        )
    
        # Generate output path
        base, _ = os.path.splitext(filepath)
        output_path = f"{base}-ql.jpg"
    
        # Save
        thumbnail.save(output_path)
    
        return Path(output_path)

def convert_to_cog(geotiff_path, bigtiff=False):
    """
    Convert a GeoTIFF to a Cloud Optimized GeoTIFF (COG)
    and validate the output.

    Parameters
    ----------
    geotiff_path : str or Path
        Path to the input GeoTIFF.
    bigtiff : bool, optional
        Enable BigTIFF with optimized settings for large files.

    Returns
    -------
    str
        Output COG filename if conversion and validation succeed.

    Raises
    ------
    FileNotFoundError
        If the input GeoTIFF does not exist.
    subprocess.CalledProcessError
        If GDAL conversion fails.
    RuntimeError
        If COG validation fails.
    """

    geotiff_path = Path(geotiff_path)

    if not geotiff_path.is_file():
        raise FileNotFoundError(
            f"Input GeoTIFF not found: {geotiff_path}"
        )

    # Generate output filename
    cog_filename = geotiff_path.with_name(
        f"{geotiff_path.stem}_COG.TIF"
    )

    # Build GDAL command
    cmd = [
        "gdal_translate",
        str(geotiff_path),
        str(cog_filename),
        "-of", "COG",
    ]

    if bigtiff:
        print("BIGTIFF processing")

        cmd.extend([
            "-co", "COMPRESS=NONE",
            "-co", "NUM_THREADS=ALL_CPUS",
            "-co", "BLOCKSIZE=512",
            "-co", "OVERVIEWS=IGNORE_EXISTING",
            "-co", "BIGTIFF=YES",
        ])
    else:
        cmd.extend([
            "-co", "COMPRESS=LZW",
        ])

    # Convert to COG
    subprocess.run(cmd, check=True)

    # Validate COG
    is_valid, errors, warnings = cog_validate(str(cog_filename))

    if not is_valid:
        raise RuntimeError(
            f"COG validation failed: {cog_filename}\n"
            f"Errors: {errors}\n"
            f"Warnings: {warnings}"
        )

    logger.info(f"COG successfully created: {cog_filename}")

    return str(cog_filename)