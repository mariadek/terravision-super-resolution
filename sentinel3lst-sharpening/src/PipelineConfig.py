from pathlib import Path
from datetime import datetime, timedelta
from zipfile import ZipFile

import os
import sys
import time
import logging
import hashlib
import shutil

import numpy as np
from tqdm import tqdm

from dotenv import load_dotenv

from thermal_sharpening.download.iccs import find_ICCS_S3TS
from thermal_sharpening.download.sentinel2 import Sentinel2Downloader
from thermal_sharpening.download.sentinel3 import Sentinel3Downloader
from thermal_sharpening.download.sentinel2SR_ICCS import ICCSSentinel2Downloader
from thermal_sharpening.utils.utils import intersection_percentage, mask_resampling
from thermal_sharpening.download.models import Scene
import thermal_sharpening.preprocessing.sentinel2 as sentinel2_processor
import thermal_sharpening.preprocessing.sentinel3 as sentinel3_processor
import thermal_sharpening.tsharpening as thermal_sharpening
import s3_upload

logger = logging.getLogger()

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

        self.max_cloud_cover = 10
        self.overlap_percentage = 80

        # ---------------------------------------------------------
        # Output configuration
        # ---------------------------------------------------------
        self.output_directory = "outputs"
        self.cleanup_data_tmp = False

        #----------------------------------------------------------
        # S3 storage configuration
        self.save_to_s3 = True
        self.s3_visibility = "public"
        self.s3_data_directory = "sentinel-3-lst-ntc-ts-10m" 

        # S3 credentials
        #CLIENT_ID = os.environ["ICCS_ACCESS_KEY_ID"]
        #CLIENT_SECRET = os.environ["ICCS_SECRET_ACCESS_KEY"]

    # =============================================================
    # PIPELINE FUNCTIONS
    # =============================================================

    def search_iccssen2sr_images(self, aoi: dict, datetime: str):
        """
        Search for satellite images using the runtime AOI.

        Args:
            aoi: User-provided AOI JSON.

        Returns:
            Images found for the AOI.
        """

        logger.info("Searching Sentinel-2 SR images for user-defined AOI and datetime")

        iccss2downloader = ICCSSentinel2Downloader()
        
        search_results = iccss2downloader.search(aoi, datetime, query={"eo:cloud_cover": {"lt": self.max_cloud_cover}})

        scenes = []

        for item in search_results:

            overlap = intersection_percentage(aoi, item.geometry)

            if overlap < self.overlap_percentage:
                    continue
            
            scene = Scene(
                scene_id=item.id,
                item=item,
                data_href= item.assets['sr-10-image'].href,
                acquisition_datetime=item.datetime,
                crs=item.properties.get("proj:code"),
                footprint=item.geometry,
                cloud_cover=item.properties.get("eo:cloud_cover"),
            )
            scenes.append(scene)


        logger.info(
            f"Image search completed: %d images found with greater than {self.overlap_percentage} percent overlap with aoi",
            len(scenes)
        )

        return scenes

    def search_sentinel3_images(self, sen2sr_scenes: list, aoi: dict):
        """
        Search for Sentinel-3 images corresponding to each Sentinel-2 SR scene.

        Sentinel-3 scenes are sorted by:
            1. Intersection percentage with the AOI (descending)
            2. Time difference from the Sentinel-2 acquisition time (ascending)

        Args:
            sen2sr_scenes: List of Sentinel-2 SR scenes retrieved from ICCS.
            aoi: Runtime AOI.

        Returns:
            Dictionary mapping Sentinel-2 scene IDs to sorted Sentinel-3 scenes.
        """

        logger.info("Searching Sentinel-3 images for each Sentinel-2 SR scene")

        sen3_scenes = {}
        downloader = Sentinel3Downloader()

        for sen2sr_scene in sen2sr_scenes:
            dt = sen2sr_scene.acquisition_datetime

            start = dt - timedelta(hours=1)
            end = dt + timedelta(hours=1)

            s2_datetime = (
                f"{start.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4]}/"
                f"{end.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4]}"
            )

            sen3_search_results = downloader.search(
                sen2sr_scene.footprint,
                datetime=s2_datetime,
                query={"eo:cloud_cover": {"lt": self.max_cloud_cover}},
            )

            for item in sen3_search_results:

                # Calculate AOI intersection
                intersection = intersection_percentage(
                    aoi,
                    item.geometry
                )

                if intersection <= self.overlap_percentage:
                    continue

                # Calculate temporal difference in minutes
                time_difference = abs(
                    (item.datetime - dt).total_seconds()
                ) / 60

                scene = Scene(
                    scene_id=item.id,
                    item=item,
                    data_href=item.assets.get("product").href,
                    acquisition_datetime=item.datetime,
                    crs=item.properties.get("proj:code"),
                    footprint=item.geometry,
                    cloud_cover=item.properties.get("eo:cloud_cover"),
                    intersection_percentage = intersection,
                    time_difference_minutes = time_difference
                )

                # Sentinel-2 scene ID as dictionary key
                sen3_scenes.setdefault(
                    sen2sr_scene.scene_id,
                    []
                ).append(scene)

            time.sleep(2)  # Avoid rate limiting

        # Sort Sentinel-3 scenes for each Sentinel-2 scene
        for scene_id, scenes in sen3_scenes.items():
            scenes.sort(
                key=lambda scene: (
                    -scene.intersection_percentage,
                    scene.time_difference_minutes,
                )
            )
            sen3_scenes[scene_id] = scenes[0] # Pick only the best match

        return sen3_scenes

    def search_sentinel2_cloud(self, best_by_scene: list):
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

        downloader = Sentinel2Downloader()

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
                        "eq": sen2_scene.scene_id.removeprefix("SR_")
                    }
                },
            )

            for item in sen2_search_results:

                record["s2_cloud"] = Scene(
                    scene_id=sen2_scene.scene_id,
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
                    f"{sen2_scene.scene_id}"
                )

            time.sleep(2)

        return best_by_scene

    
    def download_images(self, best_by_scene: list):
        """
        Download the selected Sentinel-2 SR, Sentinel-3,
        and Sentinel-2 cloud-mask scenes.

        Args:
            best_by_scene: List of dictionaries containing:
                - key: Sentinel-2 SR scene ID
                - sen2_scene: Sentinel-2 SR Scene
                - sen3_scene: matched Sentinel-3 Scene
                - s2_cloud: Sentinel-2 cloud-mask Scene

        Returns:
            List containing paths to the downloaded image triplets:
            [
                [sen2_sr_path, sen3_path, sen2_cloud_path],
                ...
            ]
        """

        logger.info(
            "Start Downloading Sentinel-3 and Matching Sentinel-2 scenes..."
        )

        iccs_downloader = ICCSSentinel2Downloader()
        sentinel2_downloader = Sentinel2Downloader()
        sentinel3_downloader = Sentinel3Downloader()

        logger.info("ICCS Username: %s", iccs_downloader.username)
        logger.info(
            "ICCS Password loaded: %s",
            bool(iccs_downloader.password),
        )

        logger.info("CDSE Username: %s", sentinel3_downloader.username)
        logger.info(
            "CDSE Password loaded: %s",
            bool(sentinel3_downloader.password),
        )

        image_pair_paths = []

        for record in best_by_scene:

            sen2_scene = record.get("sen2_scene")
            sen3_scene = record.get("sen3_scene")
            sen2_cloud = record.get("s2_cloud")

            # Check Sentinel-2 SR
            if sen2_scene is None:
                logger.warning(
                    "No Sentinel-2 SR scene for %s",
                    record.get("key"),
                )
                continue

            # Check Sentinel-3
            if sen3_scene is None:
                logger.warning(
                    "No Sentinel-3 scene for %s",
                    sen2_scene.scene_id,
                )
                continue

            # Check Sentinel-2 cloud mask
            if sen2_cloud is None:
                logger.warning(
                    "No Sentinel-2 cloud mask for %s",
                    sen2_scene.scene_id,
                )
                continue

            logger.info(
                "Sentinel-2 SR scene %s -> "
                "Sentinel-3 scene %s "
                "(time difference: %.2f minutes)",
                sen2_scene.scene_id,
                sen3_scene.scene_id,
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

            image_pair_paths.append(
                [
                    sen2sr_download_path,
                    sen3_download_path,
                    sen2cloud_download_path,
                ]
            )

        return image_pair_paths
    
    def unzip_sentinel3(self, inputs3):
        """
        Unzip a Sentinel-3 SLSTR product into a tmp folder.

        Parameters
        ----------
        inputs3 : str or Path
            Path to the Sentinel-3 ZIP file.

        Returns
        -------
        Path
            Path to the extracted .SEN3 directory.
        """
        zip_file_path = Path(inputs3)

        # tmp folder next to the ZIP file
        self.temp_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        
        # Expected extracted Sentinel-3 directory
        extracted_zip_file_path = self.temp_directory / f"{zip_file_path.stem}.SEN3"

        if extracted_zip_file_path.exists():
            logger.info("Sentinel-3 SLSTR is already unzipped.")
            return extracted_zip_file_path

        if not zip_file_path.exists():
            raise FileNotFoundError(
                f"Sentinel-3 ZIP file not found: {zip_file_path}"
            )

        logger.info(f"Unzipping {zip_file_path.name} to {self.temp_directory}...")

        try:
            with ZipFile(zip_file_path, "r") as zip_obj:
                zip_obj.extractall(self.temp_directory)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to unzip Sentinel-3 product: {zip_file_path}"
            ) from exc

        if not extracted_zip_file_path.exists():
            raise FileNotFoundError(
                f"ZIP was extracted, but expected directory was not found: "
                f"{extracted_zip_file_path}"
            )

        logger.info("Sentinel-3 SLSTR successfully unzipped.")

        return extracted_zip_file_path

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

        logger.info("Starting pipeline - Terravision Sentinel-3 SLSTR Thermal Sharpening")

        # 1. Check if Sentinel-3 SLSTR scenes have been already sharpened - ICCS STAC
        sentinel3_ts_items = find_ICCS_S3TS(aoi, datetime)

        '''
        if sentinel3_ts_items:
            logger.info(
                "TS product already exists. Stopping execution."
            )
            return
        '''


        # 2. Search Sentinel-2 SR images from ICCS STAC using the user defined aoi and datetime
        sen2sr_scenes = self.search_iccssen2sr_images(aoi, datetime)
        
        if not sen2sr_scenes:
            logger.warning("No SR Sentinel-2 images found for this aoi and datetime. You have to execute the Sentinel-2 SR workflow first.")
            return None

        
        # 3. Search Sentinel-3 SLSTR images using retrieved Sentinel-2 SR image footprints and acquisition datetime
        sen3_scenes = self.search_sentinel3_images(sen2sr_scenes, aoi)
        
        if not sen3_scenes:
            logger.warning("No Sentinel-3 images found for the retrieved EnMAP images")
            return None

        best_by_scene_dict = {}

        # Lookup original Sentinel-2 scenes by scene_id
        sen2_lookup = {
            scene.scene_id: scene
            for scene in sen2sr_scenes
        }

        for key, sen3_scene in sen3_scenes.items():
            scene_id = sen3_scene.scene_id

            if (
                scene_id not in best_by_scene_dict
                or sen3_scene.time_difference_minutes
                < best_by_scene_dict[scene_id]["sen3_scene"].time_difference_minutes
            ):
                best_by_scene_dict[scene_id] = {
                    "key": key,
                    "sen2_scene": sen2_lookup.get(key),
                    "sen3_scene": sen3_scene,
                }

        best_by_scene = list(best_by_scene_dict.values())


        # 4. Search Sentinel-2 Cloud Mask from CDSE
        sen2_clouds = self.search_sentinel2_cloud(best_by_scene)

        # 5. Download Sentinel-2 SR from ICCS S3 Storage
        image_paths = self.download_images(best_by_scene)

        outputs = []
        # 6. Data Preprocessing
        for i, (sen2sr_scene, sen3_scene, s2_mask20m) in enumerate(image_paths):
            lowResFilename = self.unzip_sentinel3(sen3_scene)

            # Extract and Resample s2_mask at 10 meters
            s2_mask10m = mask_resampling(s2_mask20m, sen2sr_scene, self.temp_directory)

            # Get SR Sentinel-2 bounding box
            # Reproject Sentinel-3 SLSTR and crop to SR Sentinel-2 bounding box
            lowResFilename_reprojected, s3_mask = sentinel3_processor.s3_preprocessor(lowResFilename, sen2sr_scene)

            logger.info('Starting Thermal Sharperning ...')
            output = self.thermal_sharpening(sen2sr_scene, lowResFilename_reprojected, s2_mask10m, s3_mask)

            outputs.append(output)

        if self.cleanup_data_tmp:
            self.housekeeping()


        # 7. Create Output Thumbnails
        '''
        def create_thumbnail(feature_id: str):
            from eo_library.thumbnails import create_thumbnail

            tiff_path = f"/mnt/workdir/outputs/SR_{feature_id}.tiff"
            output_path = f"/mnt/workdir/outputs/SR_{feature_id}-ql.jpg"

            thumbnail_size = (343, 343)
            create_thumbnail(tiff_path, output_path, thumbnail_size)

        '''

        # 8. Save output and thumbnails to ICCS S3 - keep s3 links (output and thumbnails) for stac indexing
        if self.save_to_s3:

            s3_client = s3_upload.create_s3_client(
                 os.environ['S3_CLIENT_ID'],
                 os.environ['S3_CLIENT_SECRET'],
            )
            logger.info(f'Saving Thermal Sharperning outputs to S3 bucket s3://{self.s3_visibility}/{self.s3_data_directory}')

            for output_filepath in outputs:
                output_filename = Path(output_filepath).name

                '''
                s3_upload.upload_if_not_exists_safe(
                    s3_client,
                    output_filepath,
                    self.s3_visibility,
                    os.path.join(self.s3_data_directory, output_filename)
                )
                '''

        # 9. Check if STAC Collection exists, else create it.


        # 10. 
        '''
        def update_catalogue(feature_id: str):
            from eo_library.stac_models import create_s2l2sr_stac_item
            from eo_library.auth_utils import get_auth_header_client_credentials
            from pystac import Item
            import json
            import httpx

            with open(f"/mnt/workdir/features/{feature_id}.json", "r") as f:
                item: Item = Item.from_dict(json.loads(f.read()))

            created = create_s2l2sr_stac_item(item).to_dict()
            auth_header = get_auth_header_client_credentials(
                client_id=os.environ["CLIENT_ID"],
                client_secret=os.environ["CLIENT_SECRET"],
                token_endpoint="https://auth-eo.iccs.gr/realms/eo-platform/protocol/openid-connect/token",
            )

            with httpx.Client(headers=auth_header) as client:
                response = client.put(
                    url=f"https://platform-eo.iccs.gr/stac/collections/sentinel-2-l2a-sr-10m/items/{created['id']}",
                    json=created,
                )
                response.raise_for_status()
        '''


        logger.info("Pipeline finished successfully")

        return outputs
        
            

            

        
