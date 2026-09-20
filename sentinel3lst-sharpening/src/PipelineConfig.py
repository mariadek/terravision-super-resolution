from pathlib import Path
from datetime import datetime, timedelta

import os
import sys
import time
import httpx
import logging
import hashlib
import shutil

import boto3

import numpy as np
from tqdm import tqdm

from dotenv import load_dotenv

import thermal_sharpening.download.iccs as iccs_stac
import thermal_sharpening.download.scene_search as scene_search

from thermal_sharpening.download.sentinel2 import Sentinel2Downloader
from thermal_sharpening.download.sentinel3 import Sentinel3Downloader
from thermal_sharpening.download.sentinel2SR_ICCS import ICCSSentinel2Downloader
from thermal_sharpening.utils.utils import mask_resampling, create_lst_thumbnail, convert_to_cog
from thermal_sharpening.download.models import Scene
import thermal_sharpening.preprocessing.sentinel3 as sentinel3_processor
import thermal_sharpening.tsharpening as thermal_sharpening
import thermal_sharpening.publication.stac as stac

import s3_upload

logger = logging.getLogger()
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)


def bbox_hash(bbox, length=10):
    bbox = f"{bbox[0]:.6f},{bbox[1]:.6f},{bbox[2]:.6f},{bbox[3]:.6f}"
    return hashlib.sha256(bbox.encode()).hexdigest()[:length]


class PipelineConfig:
    """
    Configuration and execution class for the Terravision
    Sentinel-3 Thermal Sharpening pipeline.
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

        self.data_directory = Path("data")
        self.sen3_data_root = self.data_directory / "sentinel3" / "sentinel-3-sl-2-lst-ntc"
        self.sen2_data_root = self.data_directory / "sentinel2" / "sentinel-2-l2a"
        self.sen2sr_data_root = self.data_directory / "sentinel2" / "sentinel-2-l2a-sr-10m"
        self.temp_directory = self.data_directory / "tmp"

        # ---------------------------------------------------------
        # Image search configuration
        # ---------------------------------------------------------

        self.max_cloud_cover = config['search']['max_cloud_cover']
        self.overlap_percentage = config['search']['min_overlap']
        self.max_time_difference = config['search']['max_time_difference_hours']

        # ---------------------------------------------------------
        # Output configuration
        # ---------------------------------------------------------
        self.output_directory = config['output']['directory']
        self.output_cog = config['output_COG']
        self.cleanup_data_tmp = config['cleanup_data_tmp']

        # Thumbnail settings
        self.thumbnail_size = tuple(config["thumbnail_size"])

        #----------------------------------------------------------
        # S3 storage configuration
        self.s3_upload = config['storage']['s3']['enabled']
        self.s3_bucket = config['storage']['s3']['bucket']
        self.s3_collection_dir = config['storage']['s3']['prefix']
        self.product_collection_id = self.s3_collection_dir

        # STAC configuration
        self.stac_indexing_enabled = config['stac']['enabled']
        self.product_collection_id = config['stac']['collection_id']

    # =============================================================
    # PIPELINE FUNCTIONS
    # =============================================================
    
    def download_images(self, scene_pair: dict, iccs_downloader, sentinel2_downloader, sentinel3_downloader):
        """
        Download the selected Sentinel-2 SR, Sentinel-3,
        and Sentinel-2 cloud-mask scenes.

        Args:
            scene_pair: Dictionary containing:
                - sen2_scene: Sentinel-2 SR Scene
                - sen3_scene: matched Sentinel-3 Scene
                - s2_cloud: Sentinel-2 cloud-mask Scene

        Returns:
            List containing paths to the downloaded image triplets:
            
                [sen2_sr_path, sen3_path, sen2_cloud_path],
                ...
            ]
        """

        logger.info(
            "Start Downloading Sentinel-3 and Matching Sentinel-2 scenes..."
        )

        image_pair_paths = []

        sen2_scene = scene_pair.get("sen2_scene")
        sen3_scene = scene_pair.get("sen3_scene")
        sen2_cloud = scene_pair.get("s2_cloud")

        # Check Sentinel-2 SR
        if sen2_scene is None:
            logger.warning(
                "No Sentinel-2 SR scene for %s",
                scene_pair.get("key"),
            )
            return image_pair_paths

        # Check Sentinel-3
        if sen3_scene is None:
            logger.warning(
                "No Sentinel-3 scene for %s",
                sen2_scene.id,
            )
            return image_pair_paths

        # Check Sentinel-2 cloud mask
        if sen2_cloud is None:
            logger.warning(
                "No Sentinel-2 cloud mask for %s",
                sen2_scene.id,
            )
            return image_pair_paths

        logger.info(
            "Sentinel-2 SR scene %s -> "
            "Sentinel-3 scene %s "
            "(time difference: %.2f minutes)",
            sen2_scene.id,
            sen3_scene.id,
            sen3_scene.time_difference_minutes,
        )

        # -----------------------------------------
        # Sentinel-2 SR Download - ICCS
        # -----------------------------------------
        sen2sr_download_path = iccs_downloader.download_item(
            sen2_scene,
            download_root=self.sen2sr_data_root,
        )

        # -----------------------------------------
        # Sentinel-3 Download - CDSE
        # -----------------------------------------
        sen3_download_path = sentinel3_downloader.download_item(
            sen3_scene.item,
            download_root=self.sen3_data_root,
        )

        # -----------------------------------------
        # Sentinel-2 Cloud Mask - CDSE
        # -----------------------------------------
        sen2cloud_download_path = (
            sentinel2_downloader.download_s3_asset(
                sen2_cloud.item,
                "SCL_20m",
                download_root=self.sen2_data_root,
            )
        )

        image_pair_paths = [
            sen2sr_download_path,
            sen3_download_path,
            sen2cloud_download_path,
            ]
        
        return image_pair_paths
    

    def thermal_sharpening(self, sen2sr_scene, lowResFilename_reprojected, s2_mask, s3_mask):

        commonOpts = {"highResFile":                    sen2sr_scene,
                      "lowResFile":                     lowResFilename_reprojected,
                      "highResQualityFile":             s2_mask,
                      "lowResQualityFile":              s3_mask,
                      "highResGoodQualityFlags":        [4, 5, 7],      # Sentinel 2
                      "lowResGoodQualityFlags":         [0],            # Sentinel 3
                      "cvHomogeneityThreshold":         0.2,
                      "movingWindowSize":               15 * 100,
                      "disaggregatingTemperature":      True}

        dtOpts =     {"perLeafLinearRegression":    True,
                      "linearRegressionExtrapolationRatio": 0.25}

        opts = commonOpts.copy()
        opts.update(dtOpts)

        disaggregator = thermal_sharpening.DecisionTreeSharpener(**opts)

        logger.info("Training Thermal Sharpening Regressor...")
        disaggregator.trainSharpener()

        logger.info("Appy Thermal Sharpening Regressor...")
        local_result, global_result = disaggregator.applySharpener()

        logger.info("Combining Global and Local Predictions ...")
        combined_result = disaggregator.combination(local_result, global_result)

        logger.info("Residual analysis...")
        corrected_result = disaggregator.residualAnalysis(
            combined_result, output_dir=self.output_directory
        )

        logger.info("Sharpening completed: %s", corrected_result)

        return corrected_result

    def housekeeping(self) -> None:
        """Remove the entire data and tmp directory."""
        if not self.data_directory.exists():
            logger.info("No data directory to clean up.")
            return

        logger.info(
            "Cleaning data directory: %s",
            self.data_directory,
        )

        try:
            shutil.rmtree(self.data_directory)
        except OSError as exc:
            logger.warning(
                "Could not completely remove data directory %s: %s",
                self.data_directory,
                exc,
            )
        else:
            logger.info("Data directory removed successfully.")


    # =============================================================
    # MAIN PIPELINE
    # =============================================================

    def run(self, aoi: dict, datetime: str):
        """
        Execute the complete pipeline.

        The AOI is supplied at runtime and is therefore not part
        of the fixed YAML configuration.

        Args:
            aoi: User-provided AOI from JSON.

            datetime: User-provided datetime range from JSON.

        Returns:
            Pipeline results.
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
        sen3_processed = iccs_stac.search_ICCS_collection(aoi, datetime, self.product_collection_id, iccs_auth)
        if sen3_processed is not None:
            logger.info(f"{len(sen3_processed)} Sentinel-3 LST products have already been processed.")

        # 3. Search Sentinel-2 SR images from ICCS STAC using the user defined aoi and datetime 
        iccs_downloader = ICCSSentinel2Downloader()

        sen2sr_scenes = scene_search.search_iccssen2sr_images(aoi, datetime, iccs_downloader, self.max_cloud_cover, self.overlap_percentage)
        if not sen2sr_scenes:
            logger.warning("No SR Sentinel-2 images found for this aoi and datetime. You have to execute the Sentinel-2 SR workflow first.")
            return None

        # 3. Search Sentinel-3 SLSTR images using retrieved Sentinel-2 SR image footprints and acquisition datetime
        sentinel3_downloader = Sentinel3Downloader()
        logger.info("CDSE Username: %s", sentinel3_downloader.username)
        logger.info("CDSE Password loaded: %s", bool(sentinel3_downloader.password),)

        sen3_scenes = scene_search.search_sentinel3_images(sen2sr_scenes, aoi, sentinel3_downloader, self.max_cloud_cover, self.overlap_percentage, self.max_time_difference)
        if not sen3_scenes:
            logger.warning("No Sentinel-3 images found for the retrieved EnMAP images")
            return None

        best_sen3_scenes = scene_search.get_best_by_scene(sen2sr_scenes, sen3_scenes)

        # 4. Remove from best_sen3_scenes the scenes already processed
        sen3_scenes_to_process = (
            scene_search.filter_out_existing_scenes(best_sen3_scenes, sen3_processed)
            if sen3_processed
            else best_sen3_scenes
        )

        logger.info(f"{len(sen3_scenes_to_process)} Sentinel-3 LST scenes will be processed")

        # 5. Search Sentinel-2 Cloud Mask from CDSE
        sentinel2_downloader = Sentinel2Downloader()
        sen3_scenes_to_process_clouds = scene_search.search_sentinel2_cloud(sen3_scenes_to_process, sentinel2_downloader)

        # 6. Loop over each Sentinel-3 scene to download and process the data, deleting temporary files after each iteration.

        for scene_pair in sen3_scenes_to_process_clouds:
            sen2_scene = scene_pair.get("sen2_scene")
            sen3_scene = scene_pair.get("sen3_scene")
            sen2_cloud = scene_pair.get("s2_cloud")

            try:

                # 7. Download Sentinel-2 SR from ICCS S3 Storage
                image_paths = self.download_images(scene_pair, 
                                                iccs_downloader, 
                                                sentinel2_downloader, 
                                                sentinel3_downloader)

                if not image_paths:
                    continue

                # 8. Data Preprocessing
                sen2sr_scene_path, sen3_scene_path, s2_mask20m_path = image_paths
                lowResFilename = sentinel3_processor.unzip_sentinel3(sen3_scene_path, self.temp_directory)

                # 9. Extract and Resample s2_mask at 10 meters
                s2_mask10m = mask_resampling(s2_mask20m_path, sen2sr_scene_path, self.temp_directory)

                # 10. Get SR Sentinel-2 bounding box
                # Reproject Sentinel-3 SLSTR and crop to SR Sentinel-2 bounding box
                lowResFilename_reprojected, s3_mask = sentinel3_processor.s3_preprocessor(lowResFilename, sen2sr_scene_path)

                # 11. Thermal Sharpening
                logger.info('Starting Thermal Sharperning ...')
                output = self.thermal_sharpening(sen2sr_scene_path, lowResFilename_reprojected, s2_mask10m, s3_mask)

                if output is None:
                    raise RuntimeError("Thermal Sharpening returned None.")

                output_path = Path(output)

                if not output_path.is_file():
                    raise RuntimeError(
                        f"Thermal sharpened product does not exist: {output_path}"
                    )

                if output_path.stat().st_size == 0:
                    raise RuntimeError(
                        f"Thermal sharpened product is empty: {output_path}"
                    )

                outputs.append(str(output_path))

                # 12. Create thumbnails
                output_ql = create_lst_thumbnail(output_path, self.thumbnail_size)

                # 14. Create COG
                if self.output_cog:
                    logger.info("Converting to COG ...")
                    output_path = convert_to_cog(output_path)

                # 13. Upload final products and thumbnails to ICCS S3.
                if self.s3_upload:
                    s3_upload.put_output_to_s3(output_ql, self.s3_bucket, self.s3_collection_dir, self.s3_client)
                    s3_upload.put_output_to_s3(output_path, self.s3_bucket, self.s3_collection_dir, self.s3_client)

                # 14. TODO: Create STAC metadata for final products.
                item_json = stac.create_item_json(self.product_collection_id, sen3_scene, sen2_scene, output_path, output_ql)
            

                # 15. TODO: Post item at collection
                if self.stac_indexing_enabled:
                    with httpx.Client(headers=iccs_auth.headers) as client:
                        response = client.put(
                            url=(
                                f"{self.ICCS_STAC_URL}/collections/"
                                f"{self.product_collection_id}/items/"
                                f"{output_path.stem}"
                            ),
                            json=json.loads(
                                item_json.read_text(encoding="utf-8")
                            ),
                            timeout=60.0,
                        )
                        response.raise_for_status()

                logger.info(
                    "Successfully processed Sentinel3 scene: %s",
                    sen3_scene.id,
                )

            except Exception:
                failures.append(sen3_scene.id)
                logger.exception(
                    "Error processing Sentinel-3 scene: %s",
                    sen3_scene.id,
                )

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
            logger.warning("Failed Sentinel-3 scene IDs: %s", ", ".join(failures))

        return outputs
        

        