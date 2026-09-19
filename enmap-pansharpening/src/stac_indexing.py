import re
from datetime import datetime, timezone
import rasterio
from shapely.geometry import Polygon, mapping, box
from rasterio.warp import transform_bounds
from typing import Optional

import pystac
from pystac.extensions.eo import EOExtension, Band
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import RasterExtension, RasterBand
from pystac.extensions.item_assets import ItemAssetsExtension

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
    Extract the acquisition datetime from an EnMAP filename.
    Returns a timezone-aware datetime in UTC.
    """
    match = re.search(r"_(\d{8}T\d{6})Z", str(filename))

    if not match:
        raise ValueError(f"Acquisition datetime not found: {filename}")

    return datetime.strptime(
        match.group(1), "%Y%m%dT%H%M%S"
    ).replace(tzinfo=timezone.utc)

def get_raster_info(raster):
    with rasterio.open(raster) as r:
        rows = r.height
        columns = r.width
        nodata = r.nodata
        transform = list(r.transform)[:6]
        gsd = r.res[0]

    return rows, columns, nodata, transform, gsd

def get_enmap_bands(image_path):
    """
    Extract band metadata from a hyperspectral raster.

    Parameters
    ----------
    image_path : str
        Path to the hyperspectral image.

    Returns
    -------
    list[dict]
        Band metadata including name, description,
        center wavelength (µm), data type, scale, and offset.
    """

    enmap_bands = []

    with rasterio.open(image_path) as src:

        tags = src.tags()

        fwhms = [float(x) for x in tags["fwhm"].strip("{}[]").split(",")]
        wavelengths = [float(x) for x in tags["wavelength"].strip("{}[]").split(",")]

        for i in src.indexes:

            description = src.descriptions[i - 1]

            if description:
                match = re.search(
                    r"(\d+(?:\.\d+)?)\s*nm",
                    description,
                    re.IGNORECASE
                )

                if match:
                    wavelength = float(match.group(1)) / 1000


            # Extract band number from description
            match = re.search(r"band\s*(\d+)", description or "", re.IGNORECASE)
        
            band_number = int(match.group(1)) if match else i
        
            enmap_bands.append({
                "name": f"band_{band_number:03d}",
                "description": description,
                "center_wavelength": wavelengths[i-1],
                "full_width_half_max": fwhms[i-1],
                "data_type": src.dtypes[i - 1],
                "scale": src.scales[i - 1],
                "offset": src.offsets[i - 1],
            })

    return enmap_bands

def create_processed_stac_item(
    *,
    item_id: str,
    collection_id: str,
    datetime_utc: datetime,
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
        datetime=datetime_utc,
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