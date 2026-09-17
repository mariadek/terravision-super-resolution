import os
import re
import numpy as np
from typing import Tuple
from PIL import Image
from pathlib import Path

import hashlib
import rasterio
from shapely.geometry import shape

def intersection_percentage(aoi_geojson, multipolygon_geojson):
    """
    Calculate the percentage of the AOI covered by a multipolygon.

    Args:
        aoi_geojson: AOI geometry in GeoJSON format.
        multipolygon_geojson: Multipolygon geometry in GeoJSON format.

    Returns:
        Intersection percentage between 0 and 100.
    """

    aoi_geom = shape(aoi_geojson)
    multipoly = shape(multipolygon_geojson)

    # Validate AOI
    if aoi_geom.is_empty or aoi_geom.area == 0:
        return 0.0

    # Validate multipolygon
    if multipoly.is_empty:
        return 0.0

    # Compute intersection
    intersection = aoi_geom.intersection(multipoly)

    if intersection.is_empty:
        return 0.0

    return (
        intersection.area / aoi_geom.area
    ) * 100.0

def find_band(HS_path, target_wavelength):

    with rasterio.open(HS_path) as src:
        metadata = src.tags()

        wavelengths = metadata['wavelength']

        wavelengths = wavelengths.strip("{}").split(",")
        wavelengths = [float(w.strip()) for w in wavelengths]

        target = float(target_wavelength)

        # Find index of closest wavelength
        closest_index = min(
            range(len(wavelengths)),
            key=lambda i: abs(wavelengths[i] - target)
        )

        return closest_index + 1

def bbox_hash(bbox, length=10):
    bbox = f"{bbox[0]:.6f},{bbox[1]:.6f},{bbox[2]:.6f},{bbox[3]:.6f}"
    return hashlib.sha256(bbox.encode()).hexdigest()[:length]

def normalize_band(band: np.ndarray, p_min: float = 2, p_max: float = 98, gamma: float = 1.6):
    """
    Applies contrast enhancement by stretching to Min-Max and gamma correction
    to a single band for better visualization

    Inputs:
        - band: 2D array containing pixel values from a single band
        - pmin: Lower percentile for contrast stretching (default=2)
        - pmax: Upper percentile for contrast stretching (default=98)
        - gamma: Gamma correction factor (default=1.6)

    Outputs:
        - 2D uint8 array with values scaled to the (0,255)
    """

    low = np.percentile(band, p_min)
    high = np.percentile(band, p_max)
    norm = np.clip((band - low) / (high - low), 0, 1)
    gamma_corrected = np.power(norm, 1 / gamma)

    return (gamma_corrected * 255).astype(np.uint8)

def create_thumbnail(
    filepath,
    thumbnail_size=(343, 343),
    rgb_wavelengths=(611, 550, 463)
):
    # Find RGB bands
    red_index = find_band(filepath, rgb_wavelengths[0])
    green_index = find_band(filepath, rgb_wavelengths[1])
    blue_index = find_band(filepath, rgb_wavelengths[2])

    # Read and normalize bands
    with rasterio.open(filepath) as src:
        r = normalize_band(src.read(red_index))
        g = normalize_band(src.read(green_index))
        b = normalize_band(src.read(blue_index))

    # Create RGB image
    rgb = np.stack([r, g, b], axis=-1)
    image = Image.fromarray(rgb)

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
