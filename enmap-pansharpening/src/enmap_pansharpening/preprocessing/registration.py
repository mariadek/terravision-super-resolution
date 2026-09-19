import logging
import rasterio
from pyproj import Transformer
from pathlib import Path


from arosics import COREG_LOCAL
from enmap_pansharpening.utils.utils import find_band, bbox_hash
from enmap_pansharpening.download.models import PreprocessedPair
from enmap_pansharpening.preprocessing.crop import get_raster_footprint, get_max_rectangle, crop_geotiff_by_bbox, polygon_to_bbox, bbox_intersection

logger = logging.getLogger(__name__)

def coregistration(
    pair: PreprocessedPair,
) -> PreprocessedPair:
    """Coregister each EnMAP image to the corresponding Sentinel-2 B04."""
    
    band_index = find_band(
        pair.enmap_path,
        664,
    )

    logger.info(
        "Band index for 664 nm in EnMAP data: %s",
        band_index,
    )

    output_path = pair.enmap_path.with_name(
        f"{pair.enmap_path.stem}_COREGISTERED.TIF"
    )

    kwargs = {
        "grid_res": 30,
        "fmt_out": "GTIFF",
        "resamp_alg_calc": "nearest",
        "max_shift": 30,
        "nodata": (0, -32768),
        "q": True
    }

    coreg = COREG_LOCAL(
        str(pair.sentinel2_b04_path),
        str(pair.enmap_path),
        path_out=str(output_path),
        r_b4match=1,
        s_b4match=band_index,
        **kwargs,
    )

    coreg.correct_shifts()

    if not output_path.is_file():
        raise FileNotFoundError(
            f"Coregistered EnMAP file was not created: {output_path}"
        )

    results = PreprocessedPair(
            enmap_path=output_path,
            sentinel2_b04_path=pair.sentinel2_b04_path,
            sentinel2_pan_path=pair.sentinel2_pan_path,
            wavelength=pair.wavelength,
            fwhm=pair.fwhm,
        )

    return results

def crop(
    pair: PreprocessedPair,
) -> tuple[PreprocessedPair, list]:
    """Crop EnMAP and Sentinel-2 PseudoPAN to their common valid extent."""

    overlap_bboxes = []

    sentinel2_footprint = get_raster_footprint(
        pair.sentinel2_pan_path
    )
    enmap_footprint = get_raster_footprint(
        pair.enmap_path
    )

    max_rect_enmap, _, _, _ = get_max_rectangle(
        enmap_footprint
    )
    max_rect_sentinel2, _, _, _ = get_max_rectangle(
        sentinel2_footprint
    )

    overlap = max_rect_sentinel2.intersection(
        max_rect_enmap
    )

    if overlap.is_empty:
        raise ValueError(
            "EnMAP and Sentinel-2 images have no common valid extent."
        )

    overlap_bbox = overlap.bounds

    enmap_cropped = pair.enmap_path.with_name(
        f"{pair.enmap_path.stem}_CROPPED.TIF"
    )
    sentinel2_cropped = pair.sentinel2_pan_path.with_name(
        f"{pair.sentinel2_pan_path.stem}_cropped"
        f"{pair.sentinel2_pan_path.suffix}"
    )

    crop_geotiff_by_bbox(
        pair.enmap_path,
        enmap_cropped,
        overlap_bbox,
    )
    crop_geotiff_by_bbox(
        pair.sentinel2_pan_path,
        sentinel2_cropped,
        overlap_bbox,
    )

    cropped_pair = PreprocessedPair(
            enmap_path=enmap_cropped,
            sentinel2_b04_path=pair.sentinel2_b04_path,
            sentinel2_pan_path=sentinel2_cropped,
            wavelength=pair.wavelength,
            fwhm=pair.fwhm,
        )

    return cropped_pair, overlap_bbox

def crop_aoi(image_pair, aoi_polygon, over_bbox):
    """
    Crop processed EnMAP and Sentinel-2 images to the intersection
    of the AOI and the overlapping image bounding box.

    Args:
        image_pair: List containing processed EnMAP and Sentinel-2 paths.
        aoi_polygon: AOI polygon.
        over_bbox: Overlapping image bbox (minx, miny, maxx, maxy),
                expressed in the Sentinel-2 CRS.

    Returns:
        List containing cropped EnMAP and Sentinel-2 paths,
        or None if there is no intersection.
    """

    enmap_image, sen2_image = map(Path, image_pair)

    # 1. Get target CRS
    with rasterio.open(sen2_image) as src:
        target_crs = src.crs

    if target_crs is None:
        raise ValueError(
            f"Sentinel-2 image has no CRS: {sen2_image}"
        )

    # 2. Project AOI to Sentinel-2 CRS
    aoi, aoi_projected = polygon_to_bbox(
        aoi_polygon,
        target_crs
    )

    # 3. Calculate bounding-box intersection
    aoi_bbox = bbox_intersection(
        over_bbox,
        aoi_projected
    )

    if aoi_bbox is None:
        logger.warning(
            "No intersection between overlapping bbox %s "
            "and projected AOI bbox %s",
            over_bbox,
            aoi_projected
        )
        return None

    if len(aoi_bbox) != 4:
        raise ValueError(
            f"Invalid intersection bbox: {aoi_bbox}"
        )

    minx, miny, maxx, maxy = aoi_bbox

    # 4. Validate intersection
    if minx >= maxx or miny >= maxy:
        logger.warning(
            "Invalid or empty AOI intersection: %s",
            aoi_bbox
        )
        return None

    # 5. Convert intersection bbox to WGS84 for hashing
    transformer = Transformer.from_crs(
        target_crs,
        "EPSG:4326",
        always_xy=True
    )

    lon1, lat1 = transformer.transform(minx, miny)
    lon2, lat2 = transformer.transform(maxx, maxy)

    aoi_bbox_wgs84 = (
        lon1,
        lat1,
        lon2,
        lat2
    )

    h = bbox_hash(aoi_bbox_wgs84)

    # 6. Generate EnMAP output path
    enmap_cropped_path = (
        enmap_image.parent
        / (
            enmap_image.name.replace(
                "COREGISTERED_CROPPED_ortho.TIF",
                f"{h}.TIF"
            )
        )
    )

    # 7. Crop EnMAP
    crop_geotiff_by_bbox(
        enmap_image,
        enmap_cropped_path,
        aoi_bbox
    )

    # 8. Generate Sentinel-2 output path
    sen2_cropped_path = (
        sen2_image.parent
        / sen2_image.name.replace(
            "_mean_cropped_adjusted.tiff",
            f"_mean_cropped_adjusted_cropped_{h}.tiff"
        )
    )

    # 9. Crop Sentinel-2
    crop_geotiff_by_bbox(
        sen2_image,
        sen2_cropped_path,
        aoi_bbox
    )

    return [
        enmap_cropped_path,
        sen2_cropped_path
    ]
    