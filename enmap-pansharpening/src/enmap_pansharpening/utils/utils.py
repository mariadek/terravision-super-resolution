import hashlib
import rasterio
from shapely.geometry import shape

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