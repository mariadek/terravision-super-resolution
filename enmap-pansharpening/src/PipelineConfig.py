import os
import re
import json
import time
import logging
import shutil
from pathlib import Path
from datetime import timedelta
from dataclasses import dataclass

import boto3

import rasterio
import numpy as np
from rasterio.windows import Window
from pyproj import Transformer
from arosics import COREG_LOCAL
from skimage.transform import resize

from dotenv import load_dotenv

from enmap_pansharpening.download.enmap import EnMAPDownloader
from enmap_pansharpening.download.sentinel2 import Sentinel2Downloader
from enmap_pansharpening.download.models import Scene
import enmap_pansharpening.download.iccs as iccs_stac
from enmap_pansharpening.utils.utils import intersection_percentage, find_band, bbox_hash, create_thumbnail, convert_to_cog
from enmap_pansharpening.preprocessing.enmap_band_removal import EnMAP
from enmap_pansharpening.preprocessing.sentinel2_panchromatic import Sentinel2
from enmap_pansharpening.preprocessing.crop import get_raster_footprint, get_max_rectangle, crop_geotiff_by_bbox, polygon_to_bbox, bbox_intersection
import enmap_pansharpening.pansharpening as pansharpening
from enmap_pansharpening.reconstruction import (
    reconstruct_single_image,
)
import s3_upload
import stac_indexing

logger = logging.getLogger(__name__)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

@dataclass
class PreprocessedPair:
    """Files and metadata produced for one EnMAP/Sentinel-2 pair."""

    enmap_path: Path
    sentinel2_b04_path: Path
    sentinel2_pan_path: Path
    wavelength: list[str]
    fwhm: list[float]


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

    def search_enmap_images(self, aoi: dict, datetime: str | None = None):
        """
        Search for satellite images using the runtime AOI.

        Args:
            aoi: User-provided AOI JSON.

        Returns:
            Images found for the AOI.
        """

        logger.info("Searching images for AOI")

        downloader = EnMAPDownloader()

        logger.debug("DLR username configured: %s", bool(downloader.username))
        logger.debug("DLR password configured: %s", bool(downloader.password))

        search_results = downloader.search(aoi, datetime=datetime)

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
                or float(cloud_cover) > self.max_cloud_cover
                or overlap < self.overlap_percentage
            ):
                continue
            
            scene = Scene(
                scene_id=item.id,
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
            self.overlap_percentage,
        )

        return scenes

    def filter_out_existing_scenes(self, enmap_scenes, enmap_processed):
        """
        Return scene objects that have not already been processed.
        """

        pattern = r"^PANSHARP_|_[a-f0-9]+\.TIF$"

        # Normalize IDs of already processed scenes
        existing_ids = {
            re.sub(pattern, "", item.scene_id)
            for item in enmap_processed
        }

        # Filter out already processed scenes
        final_scenes_to_process = [
            item
            for item in enmap_scenes
            if re.sub(pattern, "", item.scene_id) not in existing_ids
        ]

        return final_scenes_to_process

    def search_sentinel2_images(self, enmap_scenes: list,  aoi: dict):
        """
        Search for satellite images using the runtime AOI.

        Args:
            enmap_scanes: A list with the metadata of the retrieved EnMAP scenes

        Returns:
            Images found for each EnMAP scene.
        """

        logger.info("Searching Sentinel-2 images for each EnMAP scene")

        sen2_scenes = {}

        downloader = Sentinel2Downloader()

        for enmap_scene in enmap_scenes:
            dt = enmap_scene.acquisition_datetime

            start = dt - timedelta(hours=self.max_time_diff)
            end = dt + timedelta(hours=self.max_time_diff)

            s2_datetime = (
                f"{start.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4]}/"
                f"{end.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4]}"
            )

            sen2_search_results = downloader.search(
                enmap_scene.footprint,
                datetime=s2_datetime,
                query={"eo:cloud_cover": {"lt": self.max_cloud_cover}},
            )

            for item in sen2_search_results:
                if intersection_percentage(
                    aoi,
                    item.geometry
                ) >= self.overlap_percentage and intersection_percentage(enmap_scene.footprint, item.geometry) >= self.overlap_percentage:
        
                    scene = Scene(
                        scene_id=item.id,
                        item=item,
                        data_href=item.assets.get("Product").href,
                        acquisition_datetime=item.datetime,
                        crs=item.properties.get("proj:code"),
                        footprint=item.geometry,
                        cloud_cover=item.properties.get("eo:cloud_cover"),
                    )
        
                    # EnMAP ID as the dictionary key
                    sen2_scenes.setdefault(enmap_scene.scene_id, []).append(scene)

            time.sleep(2)  # avoid API rate limiting

        return sen2_scenes

    def download_images(
        self,
        enmap_scene,
        sen2_scenes,
        enmap_downloader,
        sentinel2_downloader
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

        images = []

        logger.info(
            "Start downloading EnMAP and matching Sentinel-2 scenes..."
        )

        scenes = sen2_scenes.get(enmap_scene.scene_id, [])

        if not scenes:
            sen2_scenes.pop(enmap_scene.scene_id, None)

            logger.warning(
                "No matching Sentinel-2 scenes found for EnMAP scene %s",
                enmap_scene.scene_id
            )

            return images

        logger.info(
            "EnMAP scene %s: %d Sentinel-2 scenes",
            enmap_scene.scene_id,
            len(scenes)
        )

        # Download EnMAP scene once
        enmap_download_path = enmap_downloader.download_item(
            enmap_scene.item,
            download_root=self.enmap_data_root
        )

        # Download matching Sentinel-2 scenes
        for sen2_scene in scenes:

            if self.download_full_sen2_item:

                # Download the whole Sentinel-2 item
                sen2_download_path = sentinel2_downloader.download_item(
                    sen2_scene.item,
                    download_root=self.sen2_data_root
                )

            else:

                # Download only the required Sentinel-2 10 m bands
                sentinel2_downloader.download_s3_assets(
                    sen2_scene.item,
                    ("B04_10m", "B03_10m", "B02_10m", "B08_10m"),
                    download_root=self.sen2_data_root
                )

                # Sentinel-2 assets are stored under the scene directory
                sen2_download_path = os.path.join(
                    self.sen2_data_root,
                    sen2_scene.item.id
                )

            # Store the EnMAP / Sentinel-2 pair
            images = [
                str(enmap_download_path),
                str(sen2_download_path)
            ]

        return images
    

    def preprocess_enmap(
        self,
        enmap_path: str | Path,
    ) -> tuple[Path, list[str], list[float]]:
        """Preprocess one EnMAP product and return its explicit output path."""

        enmap_path = Path(enmap_path)

        metadata_file = next(enmap_path.rglob("*-METADATA.XML"), None)
        data_file = next(enmap_path.rglob("*-SPECTRAL_IMAGE_COG.TIF"), None)

        if metadata_file is None:
            raise FileNotFoundError(
                f"No *-METADATA.XML file found under {enmap_path}"
            )

        if data_file is None:
            raise FileNotFoundError(
                f"No *-SPECTRAL_IMAGE_COG.TIF file found under {enmap_path}"
            )

        enmap = EnMAP(metadata_file)

        if enmap.root is None:
            raise ValueError(
                f"Could not parse EnMAP metadata file: {metadata_file}"
            )

        logger.info("Read EnMAP metadata: %s", metadata_file)

        band_list = enmap.get_band_list(
            "OrderByWavelengthOverlapOption"
        )

        filtered_bands = enmap.remove_water_absorption_bands(
            band_list
        )

        logger.info(
            "%d bands remain after water-absorption removal.",
            len(filtered_bands),
        )

        good_bands, removed_bands = enmap.remove_bad_bands(
            filtered_bands
        )

        logger.info("Removed bad EnMAP bands: %s", removed_bands)
        logger.info("%d EnMAP bands remain.", len(good_bands))

        data, profile = enmap.read_enmap(
            data_file,
            good_bands,
        )

        self.temp_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        processed_path = self.temp_directory / data_file.name

        enmap.write_enmap(
            processed_path,
            data,
            profile,
            good_bands,
        )

        wavelength = enmap.metadata["wavelength"]
        fwhm = enmap.metadata["fwhm"]

        wavelength_sel = [
            str(wavelength[index - 1])
            for index in good_bands
        ]
        fwhm_sel = [
            fwhm[index - 1]
            for index in good_bands
        ]

        return processed_path, wavelength_sel, fwhm_sel

    def preprocess_sentinel2_zip(
        self,
        sentinel2_path: str | Path,
    ) -> tuple[Path, Path]:
        """
        Extract Sentinel-2 bands from zip and create the mean/PseudoPAN image.

        Returns:
            Sentinel-2 B04 path and mean/PseudoPAN path.
        """

        sentinel2_path = Path(sentinel2_path)

        if sentinel2_path is None:
            raise FileNotFoundError(
               f"Expected file not found: {sentinel2_path}"
            )

        sentinel2 = Sentinel2(s2_zip = str(sentinel2_path))

        self.temp_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        sentinel2.extract_bands(
            output_directory=self.temp_directory,
        )

        
        band_names = ["B02", "B03", "B04", "B08"]

        band_files = [
            os.path.join(
                self.temp_directory,
                f"{sentinel2_path.stem}_{band_name}.tiff",
            )
            for band_name in band_names
        ]

        sentinel2.selectedbands = band_files

        pan_path = Path(
            sentinel2.create_mean_image(
                output_directory=self.temp_directory,
            )
        )

        b04_path = (
            self.temp_directory
            / f"{sentinel2_path.stem}_B04.tiff"
        )

        if not b04_path.is_file():
            raise FileNotFoundError(
                f"Expected extracted B04 file not found: {b04_path}"
            )

        if not pan_path.is_file():
            raise FileNotFoundError(
                f"Expected mean/PseudoPAN file not found: {pan_path}"
            )

        return b04_path, pan_path

    def preprocess_sentinel2_assets(
        self,
        sentinel2_path: str | Path,
    ) -> tuple[Path, Path]:
        """
        Create the mean/PseudoPAN image from Sentinel-3 assets.

        Returns:
            Sentinel-2 B04 path and mean/PseudoPAN path.
        """

        sentinel2_path = Path(sentinel2_path)

        band_names = [
            "B02_10m.jp2",
            "B03_10m.jp2",
            "B04_10m.jp2",
            "B08_10m.jp2",
        ]

        band_files = []

        for band_name in band_names:
            files = list(sentinel2_path.glob(f"*_{band_name}"))

            if not files:
                raise FileNotFoundError(
                    f"No {band_name} file found in {sentinel2_path}"
                )

            if len(files) > 1:
                raise RuntimeError(
                    f"Multiple {band_name} files found in {sentinel2_path}: "
                    f"{[f.name for f in files]}"
                )

            band_files.append(files[0])

            logger.info(
                "Found Sentinel-2 band %s: %s",
                band_name,
                files[0],
            )

        sentinel2 = Sentinel2(subdatasets = band_files)

        self.temp_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        pan_path = Path(
            sentinel2.create_mean_image(
                output_directory=self.temp_directory,
            )
        )

        b04_path = next(
            path for path in band_files
            if path.name.endswith("B04_10m.jp2")
        )


        if not b04_path.is_file():
            raise FileNotFoundError(
                f"Expected extracted B04 file not found: {b04_path}"
            )

        if not pan_path.is_file():
            raise FileNotFoundError(
                f"Expected mean/PseudoPAN file not found: {pan_path}"
            )

        return b04_path, pan_path


    def preprocess_pairs(
        self,
        enmap_path, 
        sentinel2_path
    ) -> PreprocessedPair:
        """Preprocess downloaded pairs and keep paths/metadata together."""

        (
            processed_enmap,
            wavelength,
            fwhm,
        ) = self.preprocess_enmap(enmap_path)        

        if self.download_full_sen2_item:
            (
                sentinel2_b04,
                sentinel2_pan,
            ) = self.preprocess_sentinel2_zip(sentinel2_path)
        else:
                (
                sentinel2_b04,
                sentinel2_pan,
            ) = self.preprocess_sentinel2_assets(sentinel2_path)


        processed_pair = PreprocessedPair(
                enmap_path=processed_enmap,
                sentinel2_b04_path=sentinel2_b04,
                sentinel2_pan_path=sentinel2_pan,
                wavelength=wavelength,
                fwhm=fwhm,
            )

        return processed_pair

    def coregistration(
        self,
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
        self,
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
    
    def pansharpen(
        self,
        pair: PreprocessedPair,
    ):
        """Perform first-stage EnMAP pansharpening."""

        logger.info(
            "Starting first-stage pansharpening"
        )

        images = []
        products = []

        result = pansharpening.process_pansharpening_pair(
            pair.enmap_path,
            pair.sentinel2_pan_path,
        )

        images = [
                result.hs_path,
                result.pan_path,
            ]
        

        products = pansharpening.PansharpeningParameters(
                means=result.means,
                coeffs=result.coeffs,
                wavelength=pair.wavelength,
                fwhm=pair.fwhm,
            )

        logger.info(
            "First-stage pansharpening completed"
        )

        return images, products

    def crop_aoi(self, image_pair, aoi_polygon, over_bbox):
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


    def housekeeping(self, directory) -> None:
        """Remove the entire data and tmp directory."""
        if not directory.exists():
            logger.info("No data directory to clean up.")
            return

        logger.info(
            "Cleaning tmp directory: %s",
            directory,
        )

        try:
            shutil.rmtree(directory)
        except OSError as exc:
            logger.warning(
                "Could not completely remove tmp directory %s: %s",
                directory,
                exc,
            )
        else:
            logger.info("Data directory removed successfully.")

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

        # 1. Search ICCS STAC - Check if there is the relevant collection and if not stop the workflow
        username = os.environ["ICCS_USERNAME"]
        password = os.environ["ICCS_PASSWORD"]
        iccs_auth = iccs_stac.KeycloakAuth(username, password)
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
        enmap_scenes = self.search_enmap_images(aoi, datetime=datetime)
        if len(enmap_scenes) == 0:
            logger.info(f"No EnMAP scenes found. Pipeline stopped.")
            return
       
        # 4. Remove from enmap_scenes the scenes already processed
        enmap_scenes_to_process = (
            self.filter_out_existing_scenes(enmap_scenes, enmap_processed)
            if enmap_processed
            else enmap_scenes
        )

        logger.info(f"{len(enmap_scenes_to_process)} EnMAP scenes will be processed")


        # 5. Search temporally matching Sentinel-2 scenes.
        sentinel2_scenes = self.search_sentinel2_images(enmap_scenes_to_process, aoi)

        # Loop over each EnMAP scene to download and process the data, deleting temporary files after each iteration.
        enmap_downloader = EnMAPDownloader()
        sentinel2_downloader = Sentinel2Downloader()

        logger.info("DLR Username: %s", enmap_downloader.username)
        logger.info("DLR Password loaded: %s", bool(enmap_downloader.password))

        logger.info("CDSE Username: %s", sentinel2_downloader.username)
        logger.info("CDSE Password loaded: %s", bool(sentinel2_downloader.password))

        for enmap_scene in enmap_scenes_to_process:

            try:
                # 6. Download the EnMAP and Sentinel-2 pair
                download_path = self.download_images(
                    enmap_scene,
                    sentinel2_scenes,
                    enmap_downloader,
                    sentinel2_downloader
                )

                if not download_path:
                    continue

                # 7. Preprocess source imagery
                enmap_path, sen2_path = download_path
                processed = self.preprocess_pairs(enmap_path, sen2_path)

                # 8. Coregister EnMAP to Sentinel-2
                coregistered = self.coregistration(processed)

                # 9. Crop both datasets to their common valid extent
                cropped, bbox = self.crop(coregistered)

                # 10. First pansharpening stage
                intermediate, parameters = self.pansharpen(cropped)

                # 11. Crop to AOI
                if self.crop_to_aoi:

                    cropped = self.crop_aoi(
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
                    output = convert_to_cog(output_path)

                outputs.append(str(output_path))

                # 15. Upload final products and thumbnails to ICCS S3.
                if self.s3_upload:
                    s3_upload.put_output_to_s3(output_path, self.s3_bucket, self.s3_collection_dir, self.s3_client)
                    s3_upload.put_output_to_s3(output_ql, self.s3_bucket, self.s3_collection_dir, self.s3_client)

                # 16. TODO: Create STAC metadata for final products.
                #file_footprint = stac_indexing.get_bbox_and_footprint(output_path)
            

                # 17. TODO: Post item at collection


                logger.info(
                    "Successfully processed EnMAP scene: %s",
                    enmap_scene.scene_id
                )

            except Exception:
                logger.exception(
                    "Error processing EnMAP scene: %s",
                    enmap_scene.scene_id)
                

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

        logger.info("Pipeline finished")

        return outputs