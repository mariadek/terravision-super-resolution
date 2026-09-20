import logging
import time

from datetime import timedelta

from thermal_sharpening.utils.utils import intersection_percentage
from thermal_sharpening.download.models import Scene


logger = logging.getLogger(__name__)


# ============================================================
# 1. SEARCH SENTINEL-2 SR
# ============================================================

def search_iccssen2sr_images(
    aoi: dict,
    datetime: str,
    iccs_s2_downloader,
    max_cloud_cover: float,
    overlap_percentage: float,
) -> list:
    """
    Search ICCS Sentinel-2 SR scenes within the requested AOI
    and datetime range.

    Only scenes satisfying the minimum AOI overlap and containing
    the required SR asset are returned.
    """

    logger.info(
        "Searching Sentinel-2 SR scenes "
        "(max cloud cover: %s%%, min overlap: %s%%)",
        max_cloud_cover,
        overlap_percentage,
    )

    search_results = iccs_s2_downloader.search(
        aoi,
        datetime,
        query={
            "eo:cloud_cover": {
                "lt": max_cloud_cover
            }
        },
    )

    scenes = []

    for item in search_results:

        overlap = intersection_percentage(
            aoi,
            item.geometry,
        )

        if overlap < overlap_percentage:
            logger.debug(
                "Skipping Sentinel-2 scene %s: overlap %.2f%%",
                item.id,
                overlap,
            )
            continue

        asset = item.assets.get("sr-10-image")

        if asset is None or not asset.href:
            logger.warning(
                "Skipping Sentinel-2 scene %s: "
                "missing sr-10-image asset",
                item.id,
            )
            continue

        scene = Scene(
            id=item.id,
            item=item,
            data_href=asset.href,
            acquisition_datetime=item.datetime,
            crs=item.properties.get("proj:code"),
            footprint=item.geometry,
            cloud_cover=item.properties.get("eo:cloud_cover"),
        )

        scenes.append(scene)

    logger.info(
        "Sentinel-2 SR search completed: %d scenes found",
        len(scenes),
    )

    return scenes


# ============================================================
# 2. SEARCH AND SELECT SENTINEL-3
# ============================================================

def search_sentinel3_images(
    sen2sr_scenes: list,
    aoi: dict,
    s3_downloader,
    max_cloud_cover: float,
    overlap_percentage: float,
    max_time_difference: float,
    request_delay: float = 2.0,
) -> dict:
    """
    Find the best Sentinel-3 scene for each Sentinel-2 SR scene.

    Selection priority:
        1. Smallest acquisition time difference.
        2. Highest AOI overlap (tie-breaker).

    Returns:
        {
            sentinel2_id: best_sentinel3_scene
        }

    Sentinel-2 scenes without a valid Sentinel-3 match
    are excluded.
    """

    if not sen2sr_scenes:
        logger.warning(
            "No Sentinel-2 SR scenes provided."
        )
        return {}

    logger.info(
        "Searching Sentinel-3 scenes for %d Sentinel-2 scenes",
        len(sen2sr_scenes),
    )

    best_matches = {}

    for index, sen2_scene in enumerate(sen2sr_scenes, start=1):

        logger.info(
            "Searching Sentinel-3 for Sentinel-2 %s (%d/%d)",
            sen2_scene.id,
            index,
            len(sen2sr_scenes),
        )

        dt = sen2_scene.acquisition_datetime

        start = dt - timedelta(hours=max_time_difference)
        end = dt + timedelta(hours=max_time_difference)

        datetime_range = (
            f"{start.isoformat(timespec='milliseconds')}/"
            f"{end.isoformat(timespec='milliseconds')}"
        )

        search_results = s3_downloader.search(
            sen2_scene.footprint,
            datetime=datetime_range,
            query={
                "eo:cloud_cover": {
                    "lt": max_cloud_cover
                }
            },
        )

        best_scene = None
        best_rank = None

        for item in search_results:

            overlap = intersection_percentage(
                aoi,
                item.geometry,
            )

            if overlap < overlap_percentage:
                continue

            asset = item.assets.get("product")

            if asset is None or not asset.href:
                logger.warning(
                    "Skipping Sentinel-3 scene %s: "
                    "missing product asset",
                    item.id,
                )
                continue

            if item.datetime is None:
                logger.warning(
                    "Skipping Sentinel-3 scene %s: "
                    "missing acquisition datetime",
                    item.id,
                )
                continue

            time_difference = abs(
                (item.datetime - dt).total_seconds()
            ) / 60.0

            # Explicitly enforce the time constraint.
            if time_difference > max_time_difference * 60:
                continue

            # Smaller rank is better.
            # Scene ID ensures deterministic selection
            # when time difference and overlap are identical.
            rank = (
                time_difference,
                -overlap,
                item.id,
            )

            if best_rank is not None and rank >= best_rank:
                continue

            best_scene = Scene(
                id=item.id,
                item=item,
                data_href=asset.href,
                acquisition_datetime=item.datetime,
                crs=item.properties.get("proj:code"),
                footprint=item.geometry,
                cloud_cover=item.properties.get("eo:cloud_cover"),
                intersection_percentage=overlap,
                time_difference_minutes=time_difference,
            )

            best_rank = rank

        if best_scene is not None:

            best_matches[sen2_scene.id] = best_scene

            logger.info(
                "Selected Sentinel-3 %s for Sentinel-2 %s "
                "(time difference: %.2f min, overlap: %.2f%%)",
                best_scene.id,
                sen2_scene.id,
                best_scene.time_difference_minutes,
                best_scene.intersection_percentage,
            )

        else:
            logger.warning(
                "No suitable Sentinel-3 scene found for %s",
                sen2_scene.id,
            )

        # Delay only between requests, not after the last one.
        if request_delay > 0 and index < len(sen2sr_scenes):
            time.sleep(request_delay)

    logger.info(
        "Sentinel-3 search completed: %d/%d Sentinel-2 scenes matched",
        len(best_matches),
        len(sen2sr_scenes),
    )

    return best_matches


# ============================================================
# 3. SELECT UNIQUE SENTINEL-3 SCENES
# ============================================================

def get_best_by_scene(
    sen2sr_scenes: list,
    sen3_scenes: dict,
) -> list:
    """
    Select one Sentinel-2 pairing per unique Sentinel-3 scene.

    When multiple Sentinel-2 scenes select the same Sentinel-3
    scene, retain the pairing with:

        1. Smallest acquisition time difference.
        2. Highest AOI overlap (tie-breaker).

    Returns:
        [
            {
                "key": sentinel2_id,
                "sen2_scene": Scene,
                "sen3_scene": Scene,
            },
            ...
        ]
    """

    if not sen2sr_scenes or not sen3_scenes:
        logger.info("No scene pairs available for selection.")
        return []

    sen2_lookup = {
        scene.id: scene
        for scene in sen2sr_scenes
    }

    best_by_scene = {}

    for sen2_id, sen3_scene in sen3_scenes.items():

        sen2_scene = sen2_lookup.get(sen2_id)

        if sen2_scene is None:
            logger.warning(
                "Skipping Sentinel-3 %s: "
                "Sentinel-2 scene %s not found",
                sen3_scene.id,
                sen2_id,
            )
            continue

        scene_id = sen3_scene.id

        rank = (
            sen3_scene.time_difference_minutes,
            -sen3_scene.intersection_percentage,
            sen2_id,
        )

        existing = best_by_scene.get(scene_id)

        if existing is not None and rank >= existing["rank"]:
            continue

        best_by_scene[scene_id] = {
            "key": sen2_id,
            "sen2_scene": sen2_scene,
            "sen3_scene": sen3_scene,
            "rank": rank,
        }

    # Remove the internal ranking field before returning.
    results = []

    for record in best_by_scene.values():
        results.append({
            "key": record["key"],
            "sen2_scene": record["sen2_scene"],
            "sen3_scene": record["sen3_scene"],
        })

    logger.info(
        "Selected %d unique Sentinel-3 scenes from %d matches",
        len(results),
        len(sen3_scenes),
    )

    return results

def filter_out_existing_scenes(best_sen3_scenes, sen3_processed):
    processed_ids = {
        item.id for item in sen3_processed
    }

    return [
        record
        for record in best_sen3_scenes
        if record["sen3_scene"].id not in processed_ids
    ]

def search_sentinel2_cloud(best_by_scene: list, downloader):
    """
    Search for Sentinel-2 cloud masks corresponding to each selected
    Sentinel-2 SR scene and add them directly to best_by_scene.

    Args:
        best_by_scene: List of dictionaries containing:
            - key
            - sen2_scene
            - sen3_scene

    Returns:
        Updated best_by_scene containing an additional "s2_cloud" field.
    """

    logger.info(
        "Searching Sentinel-2 cloud mask for each selected Sentinel-2 SR scene"
    )


    for record in best_by_scene:

        sen2_scene = record["sen2_scene"]

        # Default if no cloud product is found
        record["s2_cloud"] = None

        if sen2_scene is None:
            logger.warning(
                f"Sentinel-2 SR scene not found for {record['key']}"
            )
            continue

        sen2_search_results = downloader.search(
            sen2_scene.footprint,
            datetime=sen2_scene.acquisition_datetime,
            query={
                "id": {
                    "eq": sen2_scene.id.removeprefix("SR_")
                }
            },
        )

        for item in sen2_search_results:

            record["s2_cloud"] = Scene(
                id=sen2_scene.id,
                item=item,
                data_href=item.assets.get("SCL_20m").href,
                acquisition_datetime=item.datetime,
                crs=item.properties.get("proj:code"),
                footprint=item.geometry,
                cloud_cover=item.properties.get("eo:cloud_cover"),
            )

            # Exact product ID, so one result is sufficient
            break

        if record["s2_cloud"] is None:
            logger.warning(
                f"No Sentinel-2 cloud mask found for "
                f"{sen2_scene.id}"
            )

        time.sleep(2)

    return best_by_scene
