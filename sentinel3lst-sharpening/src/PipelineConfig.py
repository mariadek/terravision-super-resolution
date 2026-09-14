from pathlib import Path
from datetime import datetime, timedelta
from zipfile import ZipFile

import os
import sys
import time
import logging
import hashlib

import rasterio
import numpy as np
from tqdm import tqdm

from thermal_sharpening.download.iccs import find_ICCS_S3TS
from thermal_sharpening.download.sentinel2 import Sentinel2Downloader
from thermal_sharpening.download.sentinel3 import Sentinel3Downloader
from thermal_sharpening.download.sentinel2SR_ICCS import ICCSSentinel2Downloader
from thermal_sharpening.utils.utils import intersection_percentage, mask_extractor
from thermal_sharpening.download.models import Scene
import thermal_sharpening.preprocessing.sentinel2 as sentinel2_processor
import thermal_sharpening.preprocessing.sentinel3 as sentinel3_processor
import thermal_sharpening.tsharpening as thermal_sharpening

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

        # ---------------------------------------------------------
        # Data repository configuration
        # ---------------------------------------------------------

        self.sen3_data_root = Path("data/sentinel3/sentinel-3-sl-2-lst-ntc")
        self.sen2_data_root =  Path("data/sentinel2/sentinel-2-l2a")
        self.sen2sr_data_root =  Path("data/sentinel2/sentinel-2-l2a-sr-10m")

        # ---------------------------------------------------------
        # Image search configuration
        # ---------------------------------------------------------
        self.image_search_provider = "sentinel"

        self.max_cloud_cover = 10

        self.max_results = 3

        self.overlap_percentage = 80

        # ---------------------------------------------------------
        # Processing configuration
        # ---------------------------------------------------------
        self.crop_to_aoi = True

        # ---------------------------------------------------------
        # Output configuration
        # ---------------------------------------------------------
        self.output_directory = "outputs"

        self.output_format = "geotiff"

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

            print(item.assets['sr-10-image'])
            
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
                query={"eo:cloud_cover": {"lt": 10}},
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
        for sen2sr_scene in sen2sr_scenes:

            scene_id = sen2sr_scene.scene_id

            if scene_id not in sen3_scenes:
                continue

            sen3_scenes[scene_id].sort(
                key=lambda scene: (
                    -scene.intersection_percentage,  # Higher is better
                    scene.time_difference_minutes,   # Lower is better
                )
            )
            sen3_scenes[scene_id] = sen3_scenes[scene_id][0]

        return sen3_scenes

    def search_sentinel2_cloud(self, sen2sr_scenes: list):
            """
            Search for Sentinel-2 cloud masks corresponding to each Sentinel-2 SR scene.
    
            Args:
                sen2sr_scenes: List of Sentinel-2 SR scenes retrieved from ICCS.
    
            Returns:
                Dictionary mapping Sentinel-2 scene cloud IDs to Sentinel-2 scenes.
            """
    
            logger.info("Searching Sentinel-2 cloud mask for each Sentinel-2 SR scene")
    
            s2cloud_scenes = {}
            downloader = Sentinel2Downloader()
    
            for sen2sr_scene in sen2sr_scenes:
    
                sen2_search_results = downloader.search(
                    sen2sr_scene.footprint,
                    datetime=sen2sr_scene.acquisition_datetime,
                    query={"id": {"eq": sen2sr_scene.scene_id.removeprefix("SR_")}},
                )

                for item in sen2_search_results:
                    scene = Scene(
                        scene_id=sen2sr_scene.scene_id,
                        item= item,
                        data_href=item.assets.get("SCL_20m").href,
                        acquisition_datetime=item.datetime,
                        crs=item.properties.get("proj:code"),
                        footprint=item.geometry,
                        cloud_cover=item.properties.get("eo:cloud_cover"),
                    )
    
                    # Sentinel-2 scene ID as dictionary key
                    s2cloud_scenes[sen2sr_scene.scene_id] = scene
    
                time.sleep(2)  # Avoid rate limiting

    
            return s2cloud_scenes

    
    def download_images(self, sen2sr_scenes, sen2_clouds, sen3_scenes):
        """
        Download the images returned by the search.

        Args:
            

        Returns:
            List of downloaded image paths.
        """
        images = []        
        logger.info("Start Downloading Sentinel-3 and Matching Sentinel-2 scenes...")

        iccs_downloader = ICCSSentinel2Downloader()

        logger.info("ICCS Username: %s", iccs_downloader.username)
        logger.info("ICCS Password loaded: %s", bool(iccs_downloader.password))

        sentinel2_downloader = Sentinel2Downloader()
        sentinel3_downloader = Sentinel3Downloader()

        logger.info("CDSE Username: %s", sentinel3_downloader.username)
        logger.info("CDSE Password loaded: %s", bool(sentinel3_downloader.password))

        image_pair_paths = []
        for sen2_scene in sen2sr_scenes:
            sen3_scene = sen3_scenes.get(sen2_scene.scene_id)
            sen2_cloud = sen2_clouds.get(sen2_scene.scene_id)

            if sen3_scene:
                logger.info(
                    f"Sentinel2 SR scene {sen2_scene.scene_id}: "
                    f"Sentinel-3 scenes {sen3_scene.scene_id}"
                )

                # Sentinel-2 SR Download - ICCS
                sen2sr_download_path = iccs_downloader.download_item(sen2_scene, download_root=self.sen2sr_data_root)


                # Sentinel-3 Download - CDSE
                sen3_download_path = sentinel3_downloader.download_item(
                    sen3_scene.item,
                    download_root=self.sen3_data_root
                )   

                # Sentinel-2 Cloud - CDSE - S3
                print(sen2_cloud.item)
                sen2cloud_download_path = sentinel2_downloader.download_s3_asset(sen2_cloud.item, 'SCL_20m', download_root=self.sen2_data_root)

                image_pair_paths.append([sen2sr_download_path, sen3_download_path, sen2cloud_download_path])

        return image_pair_paths

    def unzip_sentinel3(self, inputs3):
        """
        Unzip a Sentinel-3 SLSTR product if it has not already been extracted.

        Parameters
        ----------
        inputs3 : str or Path
            Path to the Sentinel-3 ZIP file.
        lowResFilename : str or Path
            Path to a file expected to exist after extraction.

        Returns
        -------
        Path
            Path to the expected extracted file.
        """
        zip_file = Path(inputs3)
        low_res_file = Path(inputs3.replace(".zip", '.SEN3'))

        if low_res_file.exists():
            logger.info("Sentinel-3 SLSTR is already unzipped.")
            return low_res_file

        if not zip_file.exists():
            raise FileNotFoundError(f"Sentinel-3 ZIP file not found: {zip_file}")

        logger.info(f"Unzipping {zip_file.name}...")

        try:
            with ZipFile(zip_file, "r") as zip_obj:
                zip_obj.extractall(zip_file.parent)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to unzip Sentinel-3 product: {zip_file}"
            ) from exc

        if not low_res_file.exists():
            raise FileNotFoundError(
                f"ZIP was extracted, but expected file was not found: "
                f"{low_res_file}"
            )

        logger.info("Sentinel-3 SLSTR successfully unzipped.")

        return low_res_file

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
            combined_result
        )

        logger.info("Sharpening completed: %s", corrected_result)


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

        for sen2sr_scene in sen2sr_scenes:
            s3_scene = sen3_scenes.get(sen2sr_scene.scene_id)

            if s3_scene is None:
                logger.info(
                    f"Sen2SR scene {sen2sr_scene.scene_id}: no matching Sentinel-3 scene"
                )
                continue

            logger.info(
                f"Sen2SR scene {sen2sr_scene.scene_id}: "
                f"intersection: {s3_scene.intersection_percentage:.1f}% | "
                f"time difference: {s3_scene.time_difference_minutes:.1f} min"
            )


        # 4. Search Sentinel-2 Cloud Mask from CDSE
        sen2_clouds = self.search_sentinel2_cloud(sen2sr_scenes)

        for sen2_name, sen2_cloud in sen2_clouds.items():
            logger.info(
                f"Sentinel-2 name {sen2_name} and "
                f"Sentinel-2 cloud link: {sen2_cloud.data_href}"
            )
            logger.info(f"Sentinel-3 path {sen3_scenes.get(sen2_name)}")

        # 5. Download Sentinel-2 SR from ICCS S3 Storage
        image_paths = self.download_images(sen2sr_scenes, sen2_clouds, sen3_scenes)

        # 6. Data Preprocessing
        for i, (sen2sr_scene, sen3_scene, s2_mask20m) in enumerate(image_paths):
            lowResFilename = self.unzip_sentinel3(sen3_scene)

            # Extract and Resample s2_mask at 10 meters
            s2_mask10m = mask_extractor(s2_mask20m, sen2sr_scene)

            # Get SR Sentinel-2 bounding box
            # Reproject Sentinel-3 SLSTR and crop to SR Sentinel-2 bounding box
            lowResFilename_reprojected, s3_mask = sentinel3_processor.s3_preprocessor(lowResFilename, sen2sr_scene)

            logger.info('Starting Thermal Sharperning ...')
            self.thermal_sharpening(sen2sr_scene, lowResFilename_reprojected, s2_mask10m, s3_mask)


            

        

        '''

        # 3. Download images
        images = self.download_images(enmap_scenes, sen2_scenes)

        # 4. Preprocess
        # 4.1 Enmap remove bad bands
        enmap_processed, enmap_metadata = self.enmap_bband_removal(images)

        # Sentinel-2 extract 10 m and create panchromatic
        sen2_processed = self.preprocess_sen2_images(images)

        # 5. Coregistration

        coregistered_pairs = self.coregistration(images)

        # 6. Crop images
        cropped_pairs, over_bboxes = self.crop(coregistered_pairs)

        print('Cropped pairs', cropped_pairs)

        # 7. Pansharpen - 1st Stage
        images, pansharpening_products = self.pansharpen(cropped_pairs)

        # 8. Crop to aoi 
        if self.crop_to_aoi: 
            print('Cropping')
            images = self.crop_aoi(images, aoi, over_bboxes)

        # 9. Pansharpening - Final Stage
        pansharpened_images = self.pansharpening_2nd_stage(images, enmap_metadata, pansharpening_products)

        # 10. Create json file for STAC 
        '''


        '''

        # 11. Upload on ICCS S3
        output = self.save_results(results)

        logger.info("Pipeline finished successfully")

        return output
        '''