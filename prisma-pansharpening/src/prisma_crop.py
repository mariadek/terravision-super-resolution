import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union

from pyproj import Transformer
from shapely.ops import transform
from shapely.geometry import box


def bbox_intersection(bbox1, bbox2):
    minx = max(bbox1[0], bbox2[0])
    miny = max(bbox1[1], bbox2[1])
    maxx = min(bbox1[2], bbox2[2])
    maxy = min(bbox1[3], bbox2[3])

    # Check if they actually overlap
    if minx >= maxx or miny >= maxy:
        return None  # No overlap
    
    return (minx, miny, maxx, maxy)

def get_raster_footprint(HS_path):
    """
    Extract the footprint of valid (non-nodata) pixels from a raster.

    Parameters
    ----------
    HS_path : str
        Path to the raster file.

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        Unioned polygon representing the raster's valid-data footprint.
    """
    with rasterio.open(HS_path) as src:
        data = src.read(1)
        nodata = src.nodata

        # Build valid-data mask
        if nodata is not None:
            mask = data != nodata
        else:
            mask = ~np.isnan(data)

        # Convert valid mask to polygons
        polygons = []
        for geom, value in shapes(
            mask.astype(np.uint8),
            mask=mask,
            transform=src.transform
        ):
            if value == 1:
                polygons.append(shape(geom))

        # Merge polygons into a single footprint
        enmap_footprint = unary_union(polygons)

    return enmap_footprint


def get_max_rectangle(footprint, iterations=50):
    """
    Find the largest axis-aligned rectangle that fits inside the
    convex hull of a raster footprint.

    Parameters
    ----------
    enmap_footprint : shapely.geometry
        Input raster footprint.
    iterations : int, optional
        Number of binary-search iterations, by default 50.

    Returns
    -------
    max_rect : shapely.geometry.Polygon
        Maximum rectangle found.
    max_rect_coords : list
        Rectangle exterior coordinates as (y, x) tuples.
    best_width : float
        Rectangle width.
    best_height : float
        Rectangle height.
    """

    # Convex hull of the footprint
    poly = footprint.convex_hull

    # Polygon bounds
    minx, miny, maxx, maxy = poly.bounds

    # Polygon centroid
    cx, cy = poly.centroid.x, poly.centroid.y

    def rect_fits(width, height):
        """Check whether a centered rectangle fits inside the polygon."""
        rect = box(
            cx - width / 2,
            cy - height / 2,
            cx + width / 2,
            cy + height / 2
        )
        return poly.contains(rect)

    # Initialize search bounds
    max_width = maxx - minx
    max_height = maxy - miny

    best_width = 0
    best_height = 0

    # Binary search
    for _ in range(iterations):
        w_try = (best_width + max_width) / 2
        h_try = (best_height + max_height) / 2

        if rect_fits(w_try, h_try):
            best_width = w_try
            best_height = h_try
        else:
            max_width = w_try
            max_height = h_try

    # Construct final rectangle
    max_rect = box(
        cx - best_width / 2,
        cy - best_height / 2,
        cx + best_width / 2,
        cy + best_height / 2
    )

    # Convert coordinates from (x, y) to (y, x)
    max_rect_coords = [
        (y, x) for x, y in max_rect.exterior.coords
    ]

    #print("Max rectangle width:", best_width)
    #print("Max rectangle height:", best_height)


    return max_rect, max_rect_coords, best_width, best_height

def crop_geotiff_by_bbox(input_path, output_path, bbox, wavelength_sel = None, fwhm_sel = None):
    """
    Crop a GeoTIFF using a bounding box.

    Parameters
    ----------
    input_path : str
        Path to input GeoTIFF
    output_path : str
        Path to save cropped GeoTIFF
    bbox : tuple
        (xmin, ymin, xmax, ymax) in same CRS as raster
    """
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.open(input_path) as src:
        # Create window from bbox
        window = from_bounds(*bbox, transform=src.transform)

        # Read data
        data = src.read(window=window)

        # Compute new transform
        transform = src.window_transform(window)

        # Update metadata
        profile = src.profile.copy()
        profile.update({
            "height": data.shape[1],
            "width": data.shape[2],
            "transform": transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        })

        # Write output
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

            if wavelength_sel:
                dst.update_tags(
                    wavelength='{' + ', '.join(wavelength_sel) + '}',
                    wavelength_units='nanometers',
                    fwhm='{' + ', '.join(fwhm_sel) + '}'
                )

            # Preserve band descriptions
            if src.descriptions is not None:
                for i, desc in enumerate(src.descriptions, start=1):
                    dst.set_band_description(i, desc)


def polygon_to_bbox(polygon, target_crs):
    """
    Convert a GeoJSON polygon (EPSG:4326) to a bbox in target CRS.

    Parameters
    ----------
    polygon : dict
        GeoJSON geometry, e.g.
        {
            'type': 'Polygon',
            'coordinates': [...]
        }

    target_crs : str or CRS
        Target CRS, e.g. "EPSG:32634"

    Returns
    -------
    tuple
        (minx, miny, maxx, maxy)
    """

    geom = shape(polygon)

    transformer = Transformer.from_crs(
        "EPSG:4326",
        target_crs,
        always_xy=True
    )

    geom_projected = transform(transformer.transform, geom)

    return geom.bounds, geom_projected.bounds