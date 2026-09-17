import re
from datetime import datetime, timezone
import rasterio
from shapely.geometry import Polygon, mapping, box
from rasterio.warp import transform_bounds


# -------------------------------------------------------------------
# Get bbox and footprint in EPSG:4326
# STAC Item geometry/bbox must be longitude/latitude
# -------------------------------------------------------------------

def get_bbox_and_footprint(raster):

    with rasterio.open(raster) as r:

        bounds_4326 = transform_bounds(
            r.crs,
            "EPSG:4326",
            *r.bounds,
            densify_pts=21
        )

        bbox = list(bounds_4326)

        footprint = box(*bounds_4326)

        file_footprint_meta = {
            "bbox": bbox,
            "footprint": mapping(footprint),
            "crs": r.crs
        }

        return file_footprint_meta

def get_acquisition_datetime(filename):
    """
    Extract the acquisition datetime from an EnMAP filename.
    Returns a timezone-aware datetime in UTC.
    """
    match = re.search(r"_(\d{8}T\d{6})Z", str(filename))

    if not match:
        raise ValueError(f"Acquisition datetime not found: {filename}")

    return datetime.strptime(
        match.group(1), "%Y%m%dT%H%M%S"
    ).replace(tzinfo=timezone.utc)