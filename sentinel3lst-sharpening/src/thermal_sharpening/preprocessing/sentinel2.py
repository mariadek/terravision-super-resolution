import rasterio
from rasterio.warp import transform_bounds


def get_bbox_from_raster(raster_file):
    """
    Get raster bounds and transform them to another CRS.

    Returns
    -------
    tuple
        (minx, miny, maxx, maxy)
    """

    with rasterio.open(raster_file) as src:
        bounds = src.bounds

        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_file}")

    return bounds, src.crs