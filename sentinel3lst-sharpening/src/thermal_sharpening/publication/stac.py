import logging
from pathlib import Path

import re
from datetime import datetime, timezone
import rasterio
from shapely.geometry import mapping, box, Polygon
from rasterio.warp import transform_bounds
from typing import Optional

import pystac
from pystac.extensions.eo import EOExtension, Band
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import RasterExtension, RasterBand

logger = logging.getLogger(__name__)

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

        return bbox, mapping(footprint), r.crs.to_epsg()
    
    
def get_acquisition_datetime(filename):
    """
    Extract acquisition start and end datetimes from a Sentinel-3 filename.
    Returns two timezone-aware datetime objects in UTC.
    """
    match = re.search(
        r"_(\d{8}T\d{6})_(\d{8}T\d{6})_",
        str(filename)
    )

    if not match:
        raise ValueError(f"Acquisition datetimes not found: {filename}")

    start = datetime.strptime(
        match.group(1), "%Y%m%dT%H%M%S"
    ).replace(tzinfo=timezone.utc)

    end = datetime.strptime(
        match.group(2), "%Y%m%dT%H%M%S"
    ).replace(tzinfo=timezone.utc)

    return start, end

def get_raster_info(raster):
    with rasterio.open(raster) as r:
        rows = r.height
        columns = r.width
        nodata = r.nodata
        transform = list(r.transform)[:6]
        gsd = r.res[0]
        dtype = r.dtypes[0]

    return rows, columns, nodata, transform, gsd, dtype


def create_processed_stac_item(
    *,
    item_id: str,
    collection_id: str,
    start_datetime: datetime,
    end_datetime: datetime,
    geometry: dict,
    bbox: list[float],
    asset_href: str,
    quicklook_href: str,
    sources: list[dict], 
    processing_method: str,
    processing_description: str,
    epsg: int,
    shape: list[int],          # [height, width]
    transform: list[float],    # GDAL/geotransform or STAC proj transform
    bands: list[dict],
    gsd: float = 10.0,
    nodata: Optional[float] = None,
) -> pystac.Item:

    item = pystac.Item(
        id=item_id,
        geometry=geometry,
        bbox=bbox,
        datetime=start_datetime,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        properties={
            "gsd": gsd,

            # Processing / provenance
            "processing:level": "L2A",
            "processing:software": {
                "custom-processing-pipeline": "1.0"
            },
            # Source provenance
            "derived_from:products": [
                {
                    "platform": source["platform"],
                    "product_id": source["product_id"],
                    "item_href": source["item_href"]
                }
                for source in sources
            ],

            # Custom properties describing your derived product
            "processing:method": processing_method,
            "processing:description": processing_description,
            "spatial_resolution": gsd,
        },
        collection=collection_id,
    )

    # -------------------------------------------------------------
    # STAC extensions
    # -------------------------------------------------------------

    EOExtension.add_to(item)
    ProjectionExtension.add_to(item)
    RasterExtension.add_to(item)

    # Projection information
    proj = ProjectionExtension.ext(item)
    proj.epsg = epsg
    proj.shape = shape
    proj.transform = transform

    # -------------------------------------------------------------
    # Main multiband image asset
    # -------------------------------------------------------------

    asset = pystac.Asset(
        href=asset_href,
        media_type=pystac.MediaType.COG,
        roles=["data"],
        title="Processed 10 m multispectral/hyperspectral image",
    )

    item.add_asset("image", asset)

    quicklook = pystac.Asset(
            href=quicklook_href,
            media_type=pystac.MediaType.PNG,
            roles=["thumbnail"],
            title="Spectral Image Quicklook"
        )
    
    item.add_asset("thumbnail", quicklook)

    # -------------------------------------------------------------
    # EO band metadata
    # -------------------------------------------------------------

    eo_asset = EOExtension.ext(asset, add_if_missing=True)

    eo_asset.bands = [
        Band.create(
            name=b["name"],
            common_name=b.get("common_name"),
            description=b.get("description"),
            center_wavelength=b.get("center_wavelength"),
            full_width_half_max=b.get("full_width_half_max"),
        )
        for b in bands
    ]

    # -------------------------------------------------------------
    # Raster metadata
    # One RasterBand entry per GeoTIFF band
    # -------------------------------------------------------------

    raster_asset = RasterExtension.ext(asset, add_if_missing=True)

    raster_asset.bands = [
        RasterBand.create(
            nodata=nodata,
            data_type=b.get("data_type", "int16"),
            spatial_resolution=gsd,
            scale=b.get("scale"),
            offset=b.get("offset"),
        )
        for b in bands
    ]

    return item


def get_platform(filename):
    """
    Extract the Sentinel-2 or Sentinel-3 platform from a product filename.

    Returns:
        str: 'sentinel-2a', 'sentinel-2b', 'sentinel-2c',
             'sentinel-3a', 'sentinel-3b', or 'sentinel-3c'
    """
    match = re.search(
        r"(?:^|_)S([23])([ABC])_",
        str(filename),
        re.IGNORECASE
    )

    if not match:
        raise ValueError(f"Sentinel platform not found: {filename}")

    return f"sentinel-{match.group(1)}{match.group(2).lower()}"

def create_item_json(product_collection_id, sentinel3_scene, sentinel2_scene, image_dir, ql_dir):

    # Convert inputs to Path objects
    image_dir = Path(image_dir)
    ql_dir = Path(ql_dir)

    # --------------------------------------------------
    # 1. Extract raster metadata
    # --------------------------------------------------

    bbox, footprint, crs = (
        get_bbox_and_footprint(image_dir)
    )

    #logger.info("BBox: %s, Footprint: %s, CRS: %s", bbox, footprint, crs)

    start_time, end_time = get_acquisition_datetime(image_dir)


    #logger.info("Acquisition datetime: %s", datetime_utc)

    rows, columns, nodata, transform, gsd, dtype =  (
        get_raster_info(image_dir)
    )

    sen3_source_platform = get_platform(image_dir)
    sen2_source_platform = get_platform(sentinel2_scene.id)

    #logger.info("Rows: %s, Columns: %s, NoData: %s, Transform: %s, GSD: %s", rows, columns, nodata, transform, gsd)

    #logger.info("EnMAP bands: %s", enmap_bands)

    # --------------------------------------------------
    # 2. Construct asset URLs
    # --------------------------------------------------

    collection_href = (
        "https://platform-eo-storage.iccs.gr/"
        f"private-terravision/{product_collection_id}"
    )

    asset_href = f"{collection_href}/{image_dir.name}"

    quicklook_href = f"{collection_href}/{ql_dir.name}"

    # --------------------------------------------------
    # 3. Create STAC Item
    # --------------------------------------------------

    thermal_item = create_processed_stac_item(
        item_id=image_dir.stem,
        collection_id=product_collection_id,
        start_datetime=start_time,
        end_datetime=end_time,
        geometry=footprint,
        bbox=bbox,

        asset_href=asset_href,
        quicklook_href=quicklook_href,

        sources=[
            {
                "platform": f"{sen3_source_platform}",
                "product_id": sentinel3_scene.id,
                "item_href": sentinel3_scene.data_href
            },
            {
                "platform": f"{sen2_source_platform}",
                "product_id": sentinel2_scene.id,
                "item_href": sentinel2_scene.data_href,
            },
        ],
        

        processing_method=(
            "Decision tree-based thermal sharpening using DMS"
        ),

        processing_description=(
            "The Data Mining Sharpener (DMS) algorithm was applied to enhance "
            "the spatial resolution of Sentinel-3 thermal imagery using "
            "high-resolution Sentinel-2 Surface Reflectance (SR) data " 
            "as ancillary information. DMS employs a decision tree-based "
            "machine learning approach to establish relationships between " 
            "coarse-resolution thermal observations and fine-resolution spectral "
            "information. These relationships are subsequently applied to the "
            "high-resolution Sentinel-2 data to predict and disaggregate the "
            "Sentinel-3 thermal measurements into a finer spatial resolution. "
        ),

        epsg=crs,

        shape=[rows, columns],

        transform=transform,

        bands=[
            {
                "name": "LST_10m",
                "description": "Land Surface Temperature",
                "data_type": dtype,
                "nodata": nodata,
                "unit": "K",
            }
        ],

        gsd=gsd,

        nodata=nodata,
    )

    # Remove existing collection links to avoid duplicates
    thermal_item.remove_links("collection")
    thermal_item.remove_links("self")

    # Add the required collection link
    thermal_item.add_link(
        pystac.Link(
            rel=pystac.RelType.COLLECTION,
            target=collection_href,
            media_type="application/json",
        )
    )

    # --------------------------------------------------
    # 4. Prepare output JSON path
    # --------------------------------------------------

    json_path = image_dir.parent / f"{thermal_item.id}.json"
    public_url = f"{collection_href}/{ql_dir.name}/{thermal_item.id}.json"

    # Assign the local STAC Item location
    thermal_item.set_self_href(asset_href)

    # --------------------------------------------------
    # 5. Validate STAC Item
    # --------------------------------------------------

    try:
        thermal_item.validate()

    except Exception:
        logger.exception(
            "STAC validation failed for item: %s",
            thermal_item.id
        )
        raise

    logger.info(
        "STAC Item validated successfully: %s",
        thermal_item.id
    )

    # --------------------------------------------------
    # 6. Save STAC Item JSON
    # --------------------------------------------------

    thermal_item.save_object(
        dest_href=str(json_path),
        include_self_link=True
    )

    logger.info(
        "STAC Item saved to: %s",
        json_path
    )

    return json_path