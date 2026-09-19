import time
import logging

from datetime import timedelta
from enmap_pansharpening.utils.utils import intersection_percentage
from enmap_pansharpening.download.models import Scene


logger = logging.getLogger(__name__)



def search_enmap_images(aoi: dict, enmap_downloader: object, datetime: str | None = None, max_cloud_cover: float | None = 10, overlap_percentage: float | None = 70):
    """
    Search for satellite images using the runtime AOI.

    Args:
        aoi: User-provided AOI JSON.

    Returns:
        Images found for the AOI.
    """

    logger.info("Searching images for AOI")


    logger.debug("DLR username configured: %s", bool(enmap_downloader.username))
    logger.debug("DLR password configured: %s", bool(enmap_downloader.password))

    search_results = enmap_downloader.search(aoi, datetime=datetime)

    scenes = []

    for item in search_results:

        overlap = intersection_percentage(aoi, item.geometry)
        overall_quality = item.properties.get("enmap:overallQuality")
        cloud_cover = item.properties.get("eo:cloud_cover")

        if cloud_cover is None:
            logger.debug("Skipping %s: missing cloud-cover metadata", item.id)
            continue

        if (
            str(overall_quality) != "0"
            or float(cloud_cover) > max_cloud_cover
            or overlap < overlap_percentage
        ):
            continue
        
        scene = Scene(
            id=item.id,
            item=item,
            xml_href= item.assets['metadata'].href, 
            data_href= item.assets['image'].href,
            acquisition_datetime=item.datetime,
            crs=item.properties.get("proj:code"),
            footprint=item.geometry,
            cloud_cover=item.properties.get("eo:cloud_cover"),
            overall_quality=overall_quality
        )
        scenes.append(scene)


    logger.info(
        "Image search completed: %d images found with overall quality == 0 and at least %.1f%% overlap",
        len(scenes),
        overlap_percentage,
    )

    return scenes

def filter_out_existing_scenes(enmap_scenes, enmap_processed):
    """
    Return scene objects that have not already been processed.
    """

    pattern = r"^PANSHARP_|_[a-f0-9]+\.TIF$"

    # Normalize IDs of already processed scenes
    existing_ids = {
        re.sub(pattern, "", item.id)
        for item in enmap_processed
    }

    # Filter out already processed scenes
    final_scenes_to_process = [
        item
        for item in enmap_scenes
        if re.sub(pattern, "", item.id) not in existing_ids
    ]

    return final_scenes_to_process

def search_sentinel2_images(enmap_scenes: list,  aoi: dict, sentinel2_downloader: object, max_cloud_cover: float | None = 10, overlap_percentage: float | None = 70, max_time_diff = 1):
    """
    Search for satellite images using the runtime AOI.

    Args:
        enmap_scanes: A list with the metadata of the retrieved EnMAP scenes

    Returns:
        Images found for each EnMAP scene.
    """

    logger.info("Searching Sentinel-2 images for each EnMAP scene")

    sen2_scenes = {}

    for enmap_scene in enmap_scenes:
        dt = enmap_scene.acquisition_datetime

        start = dt - timedelta(hours=max_time_diff)
        end = dt + timedelta(hours=max_time_diff)

        s2_datetime = (
            f"{start.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4]}/"
            f"{end.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4]}"
        )

        sen2_search_results = sentinel2_downloader.search(
            enmap_scene.footprint,
            datetime=s2_datetime,
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        )

        for item in sen2_search_results:
            if intersection_percentage(
                aoi,
                item.geometry
            ) >= overlap_percentage and intersection_percentage(enmap_scene.footprint, item.geometry) >= overlap_percentage:
    
                scene = Scene(
                    id=item.id,
                    item=item,
                    data_href=item.assets.get("Product").href,
                    acquisition_datetime=item.datetime,
                    crs=item.properties.get("proj:code"),
                    footprint=item.geometry,
                    cloud_cover=item.properties.get("eo:cloud_cover"),
                )
    
                # EnMAP ID as the dictionary key
                sen2_scenes.setdefault(enmap_scene.id, []).append(scene)

        time.sleep(2)  # avoid API rate limiting

    return sen2_scenes
