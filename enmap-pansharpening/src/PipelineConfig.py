import os
import time
import logging
import shutil
from pathlib import Path
from datetime import timedelta
from dataclasses import dataclass


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
from enmap_pansharpening.utils.utils import intersection_percentage, find_band, bbox_hash
from enmap_pansharpening.preprocessing.enmap_band_removal import EnMAP
from enmap_pansharpening.preprocessing.sentinel2_panchromatic import Sentinel2
from enmap_pansharpening.preprocessing.crop import get_raster_footprint, get_max_rectangle, crop_geotiff_by_bbox, polygon_to_bbox, bbox_intersection
import enmap_pansharpening.pansharpening as pansharpening
from enmap_pansharpening.reconstruction import (
    reconstruct_pansharpened_images,
)

logger = logging.getLogger(__name__)

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

        # ---------------------------------------------------------
        # Output configuration
        # ---------------------------------------------------------
        self.output_directory = Path(config['output_directory'])


        # Temporary processing files
        self.temp_directory = Path("data/tmp")
        self.cleanup_data_tmp = True
        self.request_delay_seconds = 2

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
                ) >= self.overlap_percentage:
        
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

            time.sleep(self.request_delay_seconds)  # avoid API rate limiting

        return sen2_scenes

    def download_images(self, enmap_scenes, sen2_scenes):
        """
        Download the images returned by the search.

        Args:
            enmap_scenes: List with EnMAP images for the given aoi
            sen2_scenes: Dictionary of EnMAP image metadata (key) and their matching Sentinel-2 image metadata.

        Returns:
            List of downloaded image paths.
        """
        images = []        
        logger.info("Start Downloading EnMAP and Matching Sentinel-2 scenes...")

        enmap_downloader = EnMAPDownloader()

        logger.info("DLR Username: %s", enmap_downloader.username)
        logger.info("DLR Password loaded: %s", bool(enmap_downloader.password))

        sentinel2_downloader = Sentinel2Downloader()

        logger.info("CDSE Username: %s", sentinel2_downloader.username)
        logger.info("CDSE Password loaded: %s", bool(sentinel2_downloader.password))

        for enmap_scene in enmap_scenes:
            scenes = sen2_scenes.get(enmap_scene.scene_id, [])

            if scenes:
                logger.info(
                    f"EnMAP scene {enmap_scene.scene_id}: "
                    f"{len(scenes)} Sentinel-2 scenes"
                )

                # EnMAP download
                enmap_download_path = enmap_downloader.download_item(    
                    enmap_scene.item,
                    download_root=self.enmap_data_root
                )   

                # Sentinel-2 download 
                for sen2_scene in scenes:
                    if self.download_full_sen2_item:
                        # Download the whole Sentinel-2 item
                        sen2_download_path = sentinel2_downloader.download_item(
                            sen2_scene.item,
                            download_root=self.sen2_data_root
                        )

                    else:
                        # Download only the required Sentinel-2 10 m bands
                        sen2_download_paths = sentinel2_downloader.download_s3_assets(
                            sen2_scene.item,
                            ("B04_10m", "B03_10m", "B02_10m", "B08_10m"),
                            download_root=self.sen2_data_root
                        )

                        # In this case the Sentinel-2 data are stored under the scene directory
                        sen2_download_path = os.path.join(
                            self.sen2_data_root,
                            sen2_scene.item.id
                        )

                    images.append([
                        os.path.join(
                            self.enmap_data_root,
                            enmap_scene.item.id
                        ),
                        sen2_download_path
                    ])
                
            else:
                # Remove EnMAP scene entry from sen2_scenes
                sen2_scenes.pop(enmap_scene.scene_id, None)

        if not sen2_scenes:
            logger.warning(
                "No matching Sentinel-2 scenes found for any EnMAP scene. "
                "Pansharpening cannot be performed."
            )
            return 

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
        image_pairs: list,
    ) -> list[PreprocessedPair]:
        """Preprocess downloaded pairs and keep paths/metadata together."""

        logger.info(
            "Preprocessing %d EnMAP/Sentinel-2 pair(s)",
            len(image_pairs),
        )

        processed_pairs: list[PreprocessedPair] = []

        for enmap_path, sentinel2_path in image_pairs:
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


            processed_pairs.append(
                PreprocessedPair(
                    enmap_path=processed_enmap,
                    sentinel2_b04_path=sentinel2_b04,
                    sentinel2_pan_path=sentinel2_pan,
                    wavelength=wavelength,
                    fwhm=fwhm,
                )
            )

        return processed_pairs

    def coregistration(
        self,
        pairs: list[PreprocessedPair],
    ) -> list[PreprocessedPair]:
        """Coregister each EnMAP image to the corresponding Sentinel-2 B04."""

        results: list[PreprocessedPair] = []

        for pair in pairs:
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

            results.append(
                PreprocessedPair(
                    enmap_path=output_path,
                    sentinel2_b04_path=pair.sentinel2_b04_path,
                    sentinel2_pan_path=pair.sentinel2_pan_path,
                    wavelength=pair.wavelength,
                    fwhm=pair.fwhm,
                )
            )

        return results

    def crop(
        self,
        pairs: list[PreprocessedPair],
    ) -> tuple[list[PreprocessedPair], list]:
        """Crop EnMAP and Sentinel-2 PseudoPAN to their common valid extent."""

        cropped_pairs: list[PreprocessedPair] = []
        overlap_bboxes = []

        for pair in pairs:
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

            overlap_bbox = list(overlap.bounds)

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

            cropped_pairs.append(
                PreprocessedPair(
                    enmap_path=enmap_cropped,
                    sentinel2_b04_path=pair.sentinel2_b04_path,
                    sentinel2_pan_path=sentinel2_cropped,
                    wavelength=pair.wavelength,
                    fwhm=pair.fwhm,
                )
            )
            overlap_bboxes.append(overlap_bbox)

        return cropped_pairs, overlap_bboxes
    
    def pansharpen(
        self,
        pairs: list[PreprocessedPair],
    ):
        """Perform first-stage EnMAP pansharpening."""

        logger.info(
            "Starting first-stage pansharpening"
        )

        images = []
        products = []

        for pair in pairs:

            result = pansharpening.process_pansharpening_pair(
                pair.enmap_path,
                pair.sentinel2_pan_path,
            )

            images.append(
                [
                    result.hs_path,
                    result.pan_path,
                ]
            )

            products.append(
                pansharpening.PansharpeningParameters(
                    means=result.means,
                    coeffs=result.coeffs,
                    wavelength=pair.wavelength,
                    fwhm=pair.fwhm,
                )
            )

        logger.info(
            "First-stage pansharpening completed"
        )

        return images, products

    def crop_aoi(self, images, aoi_polygon, over_bboxes):
        """
        Crop processed intermediate EnMAP and Sentinel-2 images to aoi bbox.

        Args:
            images: List of processed enmap (orthogonalized) and sentinel-2 images (mean adjusted).

        Returns:
            Cropped images to aoi.
        """

        cropped_images = []

        for [enmap_image, sen2_image], over_bbox in zip(images, over_bboxes):

            with rasterio.open(sen2_image) as src:
                target_crs = src.crs

            aoi, aoi_projected = polygon_to_bbox(aoi_polygon, target_crs)

            aoi_bbox = bbox_intersection(over_bbox, aoi_projected)

            minx, miny, maxx, maxy = aoi_bbox

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

            enmap_cropped_path = enmap_image.parent / enmap_image.name.replace(
                'COREGISTERED_CROPPED_ortho.TIF',
                f'COREGISTERED_CROPPED_ORTHO_CROPPED_{h}.TIF'
            )
            
            crop_geotiff_by_bbox(enmap_image, enmap_cropped_path, aoi_bbox) # add to the final workflow, global wavelength_sel, fwhm_sel)
            
            sen2_cropped_path = sen2_image.parent / sen2_image.name.replace(
                '_mean_cropped_adjusted.tiff',
                f'_mean_cropped_adjusted_cropped_{h}.tiff'
            )
            
            crop_geotiff_by_bbox(sen2_image, sen2_cropped_path, aoi_bbox) 

            cropped_images.append([enmap_cropped_path, sen2_cropped_path])

        return cropped_images


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

        # 1. Search EnMAP scenes using the runtime AOI and optional datetime.
        enmap_scenes = self.search_enmap_images(aoi, datetime=datetime)
        if len(enmap_scenes) == 0:
            logger.info("Pipeline stopped.")
            return

        # 2. Search temporally matching Sentinel-2 scenes.
        sentinel2_scenes = self.search_sentinel2_images(enmap_scenes, aoi)

        # 3. Download source imagery.
        downloads = self.download_images(enmap_scenes, sentinel2_scenes)
        if downloads is None:
            logger.info("Pipeline stopped.")
            return

        # 4. Preprocess source imagery and retain explicit paths/metadata.
        processed = self.preprocess_pairs(downloads)

        # 5. Coregister EnMAP to Sentinel-2 using returned paths.
        coregistered = self.coregistration(processed)

        # 6. Crop both datasets to their common valid extent.
        cropped, bboxes = self.crop(coregistered)

        # 7. First pansharpening stage.
        intermediate, parameters = self.pansharpen(cropped)

        if self.crop_to_aoi:
            intermediate = self.crop_aoi(
                intermediate,
                aoi,
                bboxes,
            )

        # 9. Final pansharpening/reconstruction stage.
        outputs = reconstruct_pansharpened_images(
            intermediate,
            parameters,
            self.output_directory
        )

        # 10. TODO: Create STAC metadata for final products.
        # 11. TODO: Upload final products to ICCS S3.

        # 12. Remove temporary processing files only after successful completion.
        if self.cleanup_data_tmp:
            self.housekeeping()

        logger.info("Pipeline finished successfully")

        return outputs
