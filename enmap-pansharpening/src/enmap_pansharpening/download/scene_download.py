import os
import logging

logger = logging.getLogger(__name__)

def download_images(
        enmap_scene,
        sen2_scenes,
        enmap_downloader,
        sentinel2_downloader,
        enmap_data_root,
        sen2_data_root,
        download_full_sen2_item
    ):
        """
        Download an EnMAP scene and its matching Sentinel-2 scenes.

        Args:
            enmap_scene: EnMAP scene for the given AOI.
            sen2_scenes: Dictionary mapping EnMAP scene IDs to
                        matching Sentinel-2 scenes.
            enmap_downloader: Initialized EnMAP downloader.
            sentinel2_downloader: Initialized Sentinel-2 downloader.

        Returns:
            List of [EnMAP path, Sentinel-2 path] pairs.
        """

        logger.info(
            "Start downloading EnMAP and matching Sentinel-2 scenes..."
        )

        scenes = sen2_scenes.get(enmap_scene.id, [])

        if not scenes:
            sen2_scenes.pop(enmap_scene.id, None)

            logger.warning(
                "No matching Sentinel-2 scenes found for EnMAP scene %s",
                enmap_scene.id
            )

            return None

        logger.info(
            "EnMAP scene %s: %d Sentinel-2 scenes",
            enmap_scene.id,
            len(scenes)
        )

        # Select the closest acquisition in time, breaking ties by cloud cover.
        scenes = sorted(
            scenes,
            key=lambda scene: (
                abs(
                    (
                        scene.acquisition_datetime
                        - enmap_scene.acquisition_datetime
                    ).total_seconds()
                ),
                (
                    float(scene.cloud_cover)
                    if scene.cloud_cover is not None
                    else float("inf")
                ),
            ),
        )

        sen2_scene = scenes[0]
        logger.info("Selected Sentinel-2 scene %s for EnMAP %s", sen2_scene.id, enmap_scene.id)
        enmap_download_path = enmap_downloader.download_item(
            enmap_scene.item, download_root=enmap_data_root
        )

        if download_full_sen2_item:

            # Download the whole Sentinel-2 item
            sen2_download_path = sentinel2_downloader.download_item(
                sen2_scene.item,
                download_root=sen2_data_root
            )

        else:

            # Download only the required Sentinel-2 10 m bands
            sentinel2_downloader.download_s3_assets(
                sen2_scene.item,
                ("B04_10m", "B03_10m", "B02_10m", "B08_10m"),
                download_root=sen2_data_root
            )

            # Sentinel-2 assets are stored under the scene directory
            sen2_download_path = os.path.join(
                sen2_data_root,
                sen2_scene.item.id
            )

        # Store the EnMAP / Sentinel-2 pair
        images = [
            str(enmap_download_path),
            str(sen2_download_path)
        ]

        return images, sen2_scene
        