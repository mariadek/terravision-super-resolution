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