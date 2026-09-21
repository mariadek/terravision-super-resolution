import os
import json
import httpx
import logging
import shutil
from pathlib import Path

import boto3

from dotenv import load_dotenv

import enmap_pansharpening.download.iccs as iccs_stac
import enmap_pansharpening.download.scene_search as scene_search
import enmap_pansharpening.download.scene_download as scene_download
import enmap_pansharpening.preprocessing.preprocess as preprocess
import enmap_pansharpening.preprocessing.registration as registration
import enmap_pansharpening.publication.stac as stac
from enmap_pansharpening.download.enmap import EnMAPDownloader
from enmap_pansharpening.download.sentinel2 import Sentinel2Downloader

from enmap_pansharpening.utils.utils import create_thumbnail, convert_to_cog

import enmap_pansharpening.pansharpening as pansharpening
from enmap_pansharpening.reconstruction import (
    reconstruct_single_image,
)

import s3_upload

logger = logging.getLogger(__name__)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)


class PipelineConfig:

    """
    Configuration and execution class for the Terravision
    EnMAP Pansharpening pipeline.
    """

    def __init__(self, config):

        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        load_dotenv(PROJECT_ROOT / ".env")

        self.ICCS_STAC_URL = "https://platform-eo.iccs.gr/stac"
        session = boto3.Session(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                )
        self.s3_client = session.client(
            "s3",
            endpoint_url="https://platform-eo-storage.iccs.gr",
        )

        # ---------------------------------------------------------
        # Data repository configuration
        # ---------------------------------------------------------
        self.data_directory = Path(config['input']['directory'])
        self.enmap_data_root = Path(config['input']['enmap_data_root'])
        self.sen2_data_root =  Path(config['input']['sen2_data_root'])

        # ---------------------------------------------------------
        # Image search configuration
        # ---------------------------------------------------------
        self.overlap_percentage = config['search']['min_overlap']
        self.max_cloud_cover = config['sentinel2_search']['max_cloud_cover']
        self.max_time_diff = config['sentinel2_search']['max_time_difference_hours']

        # ---------------------------------------------------------
        # CDSE Sentinel-2 Download Configuration
        # ---------------------------------------------------------
        self.download_full_sen2_item = config['sentinel2_download']['download_full_sen2_item']

        # ---------------------------------------------------------
        # Processing configuration
        # ---------------------------------------------------------
        self.crop_to_aoi = config['processing']['crop_to_aoi']

        self.chunk_size = config['pansharpening']['chunk_size']
        self.padding_size = config['pansharpening']['padding_size']

        # ---------------------------------------------------------
        # Output configuration
        # ---------------------------------------------------------
        self.output_directory = Path(config['output_directory'])
        self.output_cog = config['output_COG']

        # Temporary processing files
        self.temp_directory = self.data_directory / "tmp"
        self.cleanup_data_tmp = config['cleanup_data_tmp']

        # Thumbnail settings
        self.thumbnail_size = tuple(config["thumbnail_size"])

        # S3 upload settings
        self.s3_upload = config['storage']['s3']['enabled']
        self.s3_bucket = config['storage']['s3']['bucket']
        self.s3_collection_dir = config['storage']['s3']['prefix']

        # STAC settings
        self.stac_indexing_enabled = config['stac']['enabled']
        self.product_collection_id = config['stac']['collection_id']

    # =============================================================
    # PIPELINE FUNCTIONS
    # =============================================================

    def housekeeping(self, directory: str | Path) -> None:
        """Safely remove only the temporary processing directory."""

        directory = Path(directory).resolve()
        temp_directory = self.temp_directory.resolve()

        if directory != temp_directory:
            raise ValueError(
                f"Refusing to delete non-temporary directory: {directory}"
            )

        if directory == Path(directory.anchor):
            raise ValueError(
                "Refusing to delete a filesystem root."
            )

        if not directory.exists():
            logger.info(
                "Temporary directory does not exist: %s",
                directory,
            )
            return

        logger.info(
            "Cleaning temporary directory: %s",
            directory,
        )

        try:
            shutil.rmtree(directory)

        except OSError:
            logger.exception(
                "Failed to clean temporary directory: %s",
                directory,
            )
            raise

        logger.info(
            "Temporary directory removed successfully."
        )

    # =============================================================
    # MAIN PIPELINE
    # =============================================================

    def run(self, aoi: dict, datetime: str | None = None):
        """
        Execute the complete TERRAVISION EnMAP pansharpening pipeline.

        Args:
            aoi: User-provided GeoJSON Polygon/MultiPolygon.
            datetime: Optional STAC datetime or datetime interval.

        Returns:
            List of final pansharpened product paths, or None when no suitable
            source imagery is found.
        """

        logger.info("Starting pipeline - TERRAVISION EnMAP Pansharpening")
        outputs = []
        failures = []

        # 1. Search ICCS STAC - Check if there is the relevant collection and if not stop the workflow
        iccs_username = os.environ["ICCS_USERNAME"]
        iccs_password = os.environ["ICCS_PASSWORD"]

        logger.info("ICCS Username: %s", iccs_username)
        logger.info("ICCS Password loaded: %s", bool(iccs_password))

        iccs_auth = iccs_stac.KeycloakAuth(iccs_username, iccs_password)
        logger.info("Checking ICCS STAC for collection '%s' at %s", self.product_collection_id, self.ICCS_STAC_URL,)
        collection_exists = iccs_stac.collection_exists(self.ICCS_STAC_URL, self.product_collection_id, iccs_auth,)
        logger.info("ICCS STAC collection '%s' exists: %s", self.product_collection_id, collection_exists,)

        if not collection_exists:
            logger.info("Need to create the STAC collection '%s' at %s", self.product_collection_id,  self.ICCS_STAC_URL,)
            return
       
        # 2. Search in ICCS STAC to find the enmap scenes that have been already processed
        enmap_processed = iccs_stac.search_ICCS_collection(aoi, datetime, self.product_collection_id, iccs_auth)
        if enmap_processed is not None:
            logger.info(f"{len(enmap_processed)} EnMAP products have already been processed.")

        # 3. Search EnMAP scenes using the runtime AOI and optional datetime.
        enmap_downloader = EnMAPDownloader()
        sentinel2_downloader = Sentinel2Downloader()

        logger.info("DLR Username: %s", enmap_downloader.username)
        logger.info("DLR Password loaded: %s", bool(enmap_downloader.password))

        logger.info("CDSE Username: %s", sentinel2_downloader.username)
        logger.info("CDSE Password loaded: %s", bool(sentinel2_downloader.password))

        enmap_scenes = scene_search.search_enmap_images(aoi, enmap_downloader, datetime, self.max_cloud_cover, self.overlap_percentage)
        if len(enmap_scenes) == 0:
            logger.info(f"No EnMAP scenes found. Pipeline stopped.")
            return
       
        # 4. Remove from enmap_scenes the scenes already processed
        enmap_scenes_to_process = (
            scene_search.filter_out_existing_scenes(enmap_scenes, enmap_processed)
            if enmap_processed
            else enmap_scenes
        )

        logger.info(f"{len(enmap_scenes_to_process)} EnMAP scenes will be processed")


        # 5. Search temporally matching Sentinel-2 scenes.
        sentinel2_scenes = scene_search.search_sentinel2_images(enmap_scenes_to_process, aoi, sentinel2_downloader, self.max_cloud_cover, self.overlap_percentage, self.max_time_diff)

        # Loop over each EnMAP scene to download and process the data, deleting temporary files after each iteration.
        for enmap_scene in enmap_scenes_to_process:
            try:
                # 6. Download the EnMAP and Sentinel-2 pair
                download_path = scene_download.download_images(
                    enmap_scene,
                    sentinel2_scenes,
                    enmap_downloader,
                    sentinel2_downloader,
                    self.enmap_data_root,
                    self.sen2_data_root,
                    self.download_full_sen2_item
                )

                if not download_path:
                    continue

                # 7. Preprocess source imagery
                download_path, selected_sentinel2 = download_path
                enmap_path, sen2_path = download_path
                processed = preprocess.preprocess_pairs(enmap_path, sen2_path, self.download_full_sen2_item, self.temp_directory)

                # 8. Coregister EnMAP to Sentinel-2
                coregistered = registration.coregistration(processed)

                # 9. Crop both datasets to their common valid extent
                cropped, bbox = registration.crop(coregistered)

                # 10. First pansharpening stage
                intermediate, parameters = pansharpening.pansharpen(cropped)

                # 11. Crop to AOI
                if self.crop_to_aoi:

                    cropped = registration.crop_aoi(
                        intermediate,
                        aoi,
                        bbox,
                    )

                    if cropped is None:
                        logger.warning(
                            "Skipping image pair: AOI does not intersect "
                            "the overlapping EnMAP–Sentinel-2 footprint."
                        )
                        continue

                    intermediate = cropped

                hs_ortho_file, pan_adjusted_file = intermediate

                # 12. Final pansharpening/reconstruction stage
                output = reconstruct_single_image(
                    hs_ortho_file,
                    pan_adjusted_file,
                    parameters.means,
                    parameters.coeffs,
                    parameters.wavelength,
                    parameters.fwhm,
                    self.output_directory,
                    self.chunk_size,
                    self.padding_size,
                )

                if output is None:
                    raise RuntimeError("Reconstruction returned None.")

                output_path = Path(output)

                if not output_path.is_file():
                    raise RuntimeError(
                        f"Reconstructed product does not exist: {output_path}"
                    )

                if output_path.stat().st_size == 0:
                    raise RuntimeError(
                        f"Reconstructed product is empty: {output_path}"
                    )

                # 13. Create thumbnails
                output_ql = create_thumbnail(output_path, self.thumbnail_size)

                # 14. Create COG
                if self.output_cog:
                    logger.info("Converting to COG ...")
                    output_path = convert_to_cog(output_path)

                outputs.append(str(output_path))

                # 15. Upload final products and thumbnails to ICCS S3.
                if self.s3_upload:
                    s3_upload.put_output_to_s3(output_path, self.s3_bucket, self.s3_collection_dir, self.s3_client)
                    s3_upload.put_output_to_s3(output_ql, self.s3_bucket, self.s3_collection_dir, self.s3_client)

                # 16. Collect metadata from final product. Output a json and use it to post the item
                item_json = stac.create_item_json(self.product_collection_id, enmap_scene, selected_sentinel2, output_path, output_ql)

                # 17. Post item at collection
                if self.stac_indexing_enabled:
                    with httpx.Client(headers=iccs_auth.headers) as client:
                        response = client.put(
                            url=(
                                f"{self.ICCS_STAC_URL}/collections/"
                                f"{self.product_collection_id}/items/"
                                f"{Path(output_path).stem}"
                            ),
                            json=json.loads(
                                item_json.read_text(encoding="utf-8")
                            ),
                            timeout=60.0,
                        )
                        response.raise_for_status()

                logger.info(
                    "Successfully processed EnMAP scene: %s",
                    enmap_scene.id
                )

            except Exception:
                failures.append(enmap_scene.id)
                logger.exception(
                    "Error processing EnMAP scene: %s",
                    enmap_scene.id)
                
            finally:
                # 12. Clean temporary files regardless of success or failure
                if self.cleanup_data_tmp:

                    try:
                        logger.info(
                            "Cleaning temporary processing files..."
                        )

                        self.housekeeping(self.data_directory)

                    except Exception:
                        logger.exception(
                            "Error during temporary file cleanup."
                        )

        logger.info("Pipeline finished: %d products, %d failed scenes", len(outputs), len(failures))
        if failures:
            logger.warning("Failed EnMAP scene IDs: %s", ", ".join(failures))

        return outputs