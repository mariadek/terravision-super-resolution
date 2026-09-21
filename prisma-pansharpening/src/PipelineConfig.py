import os
import time
import logging
import rasterio
import hashlib
import numpy as np

from pathlib import Path
from shapely.geometry import mapping
from datetime import datetime, timedelta
from arosics import COREG_LOCAL
from pyproj import Transformer

from prisma_processing import generate_PAN, generate_HS, get_raster_footprint_wgs84
from sentinel2 import Sentinel2Downloader

from utils import unzip_prisma_file, intersection_percentage
from iccs import ICCSSTAC
import prisma_crop
import pansharpening

logger = logging.getLogger()


def bbox_hash(bbox, length=10):
    bbox = f"{bbox[0]:.6f},{bbox[1]:.6f},{bbox[2]:.6f},{bbox[3]:.6f}"
    return hashlib.sha256(bbox.encode()).hexdigest()[:length]

class PipelineConfig:
    """
    Configuration and execution class for the Terravision
    PRISMA Pansharpening pipeline.
    """

    def __init__(self, config):

        # ---------------------------------------------------------
        # Data repository configuration
        # ---------------------------------------------------------

        self.prisma_data_root = Path(config['input']['prisma_directory'])
        #self.sen2_data_root =  Path(config['input']['sen2_directory'])
        
        
        # ---------------------------------------------------------
        # ICCS STAC and S3 Configuration
        # ---------------------------------------------------------
        self.iccs_stac_collection = config['stac']['collection_id']

        # ---------------------------------------------------------
        # Image search configuration
        # ---------------------------------------------------------

        self.max_cloud_cover = config['search']['max_cloud_cover']
        self.overlap_percentage = config['search']['min_overlap']
        self.max_time_diff = config['search']['max_time_difference_hours']

        # ---------------------------------------------------------
        # Pansharpening configuration
        # ---------------------------------------------------------
        self.crop_to_aoi = config['processing']['crop_to_aoi']
        self.max_workers = config['processing']['pansharpening']['max_workers']

        # ---------------------------------------------------------
        # Output configuration
        # ---------------------------------------------------------
        self.output_directory = "outputs"

        self.output_format = "geotiff"

    # =============================================================
    # PIPELINE FUNCTIONS
    # =============================================================

    def check_processed_prisma_scenes(self, prisma_scenes, aoi):
        """
        Check which PRISMA scenes have already been processed
        and are available in the ICCS STAC collection.

        Args:
            prisma_scenes: List of PRISMA ZIP filenames.
            aoi: Area of interest used for the STAC search.

        Returns:
            tuple:
                matching_items: Already processed STAC items.
                missing_prisma_files: PRISMA scenes requiring processing.
        """

        # ---------------------------------------------------------
        # Initialize ICCS STAC
        # ---------------------------------------------------------

        iccs = ICCSSTAC()

        logger.info("ICCS STAC Username: %s", iccs.username)
        logger.info(
            "ICCS STAC Password loaded: %s",
            bool(iccs.password),
        )

        # ---------------------------------------------------------
        # Build expected STAC ID prefixes
        # ---------------------------------------------------------

        prisma_prefixes = {
            prisma_file.removesuffix(".zip")
            + "_HS_coreg_pansharp": prisma_file
            for prisma_file in prisma_scenes
        }

        # ---------------------------------------------------------
        # Search STAC
        # ---------------------------------------------------------

        items = iccs.search(
            self.iccs_stac_collection,
            aoi,
        )

        # ---------------------------------------------------------
        # Find already processed scenes
        # ---------------------------------------------------------

        found_prisma_files = set()
        matching_items = []

        for item in items:
            for prefix, prisma_file in prisma_prefixes.items():

                if item.id.startswith(prefix):
                    found_prisma_files.add(prisma_file)
                    matching_items.append(item)
                    break

        # ---------------------------------------------------------
        # Find scenes requiring processing
        # ---------------------------------------------------------

        missing_prisma_files = [
            prisma_file
            for prisma_file in prisma_scenes
            if prisma_file not in found_prisma_files
        ]

        logger.info(
            "%d/%d PRISMA scenes already processed.",
            len(found_prisma_files),
            len(prisma_scenes),
        )

        return matching_items, missing_prisma_files

    def get_prisma_date(self, prisma_path: str | Path) -> datetime:
        """
        Extract the acquisition datetime from a PRISMA filename.

        Example:
            PRS_L2D_STD_20230822110951_20230822110955_0001_PAN.tif

        Returns:
            datetime object corresponding to:
            2023-08-22
        """

        prisma_path = Path(prisma_path)

        filename = prisma_path.stem
        parts = filename.split("_")

        if len(parts) < 4:
            raise ValueError(
                f"Invalid PRISMA filename: {filename}"
            )

        timestamp_str = parts[3]

        try:
            return datetime.strptime(
                timestamp_str,
                "%Y%m%d%H%M%S",
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid PRISMA timestamp '{timestamp_str}' "
                f"in filename: {filename}"
            ) from exc

    def search_sentinel2(self, aoi, datetime, download_path):
        """
        Search for Sentinel-2 cloud masks corresponding to each Sentinel-2 SR scene.

        Args:
            

        Returns:
            
        """

        logger.info("Searching Sentinel-2 ... ")

        downloader = Sentinel2Downloader()

        logger.info("CDSE Username: %s", downloader.username)
        logger.info(
            "CDSE Password loaded: %s",
            bool(downloader.password),
        )

        start = datetime - timedelta(hours=self.max_time_diff)
        end = datetime + timedelta(hours=self.max_time_diff)

        s2_datetime = (
            f"{start.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4]}/"
            f"{end.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4]}"
        )
        

        sen2_search_results = downloader.search(
            aoi,
            datetime=s2_datetime,
            query={"eo:cloud_cover": {"lt": self.max_cloud_cover}}
        )

        candidates = []
        for item in sen2_search_results:

            overlap = intersection_percentage(
                aoi,
                item.geometry,
            )

            if overlap < self.overlap_percentage:
                continue

            cloud_cover = item.properties.get(
                "eo:cloud_cover",
                float("inf"),
            )

            candidates.append({
                "item": item,
                "item_id": item.id,
                "overlap": overlap,
                "cloud_cover": cloud_cover,
            })

        if not candidates:
            return None

        # Highest overlap first, lowest cloud cover second
        candidates.sort(
            key=lambda x: (
                -x["overlap"],
                x["cloud_cover"],
            )
        )

        best_item = candidates[0]["item"]

        sen2_download_path = downloader.download_s3_assets(best_item, ['B04_10m'], download_root=download_path)

        logger.info(
                "Temporary Sentinel-2 Red Band Downloaded: %s",
                sen2_download_path.get("B04_10m"),
            )

        return sen2_download_path.get("B04_10m")

    def find_band(self, hs_path, target_wavelength=664):
        """Return the 1-based band number closest to the target wavelength."""

        with rasterio.open(hs_path) as src:
            metadata = src.tags()

            wavelength_str = metadata.get("wavelength")

            if wavelength_str is None:
                raise ValueError("No 'wavelength' metadata found in the raster.")

            wavelengths = [
                float(w.strip())
                for w in wavelength_str.strip("{}").split(",")
            ]

            closest_index = min(
                range(len(wavelengths)),
                key=lambda i: abs(wavelengths[i] - target_wavelength)
            )

            band_number = closest_index + 1

            print(
                f"Target: {target_wavelength} nm | "
                f"Closest: {wavelengths[closest_index]} nm | "
                f"Band: {band_number}"
            )

            return band_number

    def coregistration(self, prisma_file, sen2_file, prisma_red_index=1):
    
        """
        Perform coregistration of EnMAP to Sentinel 2 (reference).

        Args:
            images: Preprocessed images.

        Returns:
            Coregistered image pairs.
        """ 

        coregistered_path = prisma_file.replace('.tif', '_COREG.tif')

        if Path(coregistered_path).exists():
            logger.info(
                "Coregistered image already exists. Skipping coregistration: %s",
                coregistered_path,
            )
            return str(coregistered_path)

        kwargs = {
            'grid_res': 30,
            'fmt_out': 'GTIFF',
            'resamp_alg_calc': 'nearest',
            'max_shift': 30,
            'nodata': (0, 0),
            'q': True} # quiet mode

        CRL = COREG_LOCAL(sen2_file, prisma_file,
                            path_out = coregistered_path, r_b4match = 1, s_b4match = prisma_red_index, **kwargs)
        CRL.correct_shifts()

        logger.info(
                "Temporary coregistered image created: %s",
                coregistered_path,
            )


        return coregistered_path


    def crop(self, pan_path, hs_path):
        """
        Crop PRISMA PAN and hyperspectral images to their overlapping bbox.

        If cropped files already exist, skip their creation.

        Args:
            pan_path: Path to coregistered panchromatic image.
            hs_path: Path to coregistered hyperspectral image.

        Returns:
            pan_cropped_path, hs_cropped_path, over_bbox
        """

        pan_path = Path(pan_path)
        hs_path = Path(hs_path)

        # ---------------------------------------------------------
        # Calculate image footprints
        # ---------------------------------------------------------

        pan_footprint = prisma_crop.get_raster_footprint(
            str(pan_path)
        )

        hs_footprint = prisma_crop.get_raster_footprint(
            str(hs_path)
        )

        # ---------------------------------------------------------
        # Find maximum valid rectangles
        # ---------------------------------------------------------

        max_rect_hs, _, _, _ = prisma_crop.get_max_rectangle(
            hs_footprint
        )

        max_rect_pan, _, _, _ = prisma_crop.get_max_rectangle(
            pan_footprint
        )

        # ---------------------------------------------------------
        # Calculate overlapping bounding box
        # ---------------------------------------------------------

        overlap_bbox = max_rect_pan.intersection(
            max_rect_hs
        )

        if overlap_bbox.is_empty:
            raise ValueError(
                "PAN and HS images do not have an overlapping area."
            )

        over_bbox = list(overlap_bbox.bounds)

        # ---------------------------------------------------------
        # Output paths
        # ---------------------------------------------------------

        pan_cropped_path = pan_path.with_name(
            f"{pan_path.stem}_CROPPED.tif"
        )

        hs_cropped_path = hs_path.with_name(
            f"{hs_path.stem}_CROPPED.tif"
        )

        # ---------------------------------------------------------
        # Crop PAN
        # ---------------------------------------------------------

        if pan_cropped_path.exists():
            logger.info(
                "Cropped PAN image already exists. "
                "Skipping crop: %s",
                pan_cropped_path,
            )
        else:
            prisma_crop.crop_geotiff_by_bbox(
                str(pan_path),
                str(pan_cropped_path),
                over_bbox,
            )

            logger.info(
                "Cropped PAN image created: %s",
                pan_cropped_path,
            )

        # ---------------------------------------------------------
        # Crop hyperspectral image
        # ---------------------------------------------------------

        if hs_cropped_path.exists():
            logger.info(
                "Cropped hyperspectral image already exists. "
                "Skipping crop: %s",
                hs_cropped_path,
            )
        else:
            prisma_crop.crop_geotiff_by_bbox(
                str(hs_path),
                str(hs_cropped_path),
                over_bbox,
            )

            logger.info(
                "Cropped hyperspectral image created: %s",
                hs_cropped_path,
            )

        return (
            str(pan_cropped_path),
            str(hs_cropped_path),
            over_bbox,
        )

    def pansharpen_1st_stage(self, pan_path, hs_path):
        """
        Perform EnMAP pansharpening.

        Args:
            images: Preprocessed images.

        Returns:
            hs_ortho_path, pan_adjusted_path, means, coeffs
        """

        logger.info("Starting pansharpening")

        hs = pansharpening.read_hs_file(hs_path)
        pan = pansharpening.read_pan_file(pan_path)

        # --------------------------------------------------
        # Step 1: Synthetic intensity
        # --------------------------------------------------
        print("Step 1: Calculating synthetic intensity")

        I, I0 = pansharpening.calculate_synthetic_intensity(
            hs["data"],
            pan["data"],
            pansharpening.alpha_estimation
        )

        # --------------------------------------------------
        # Step 2: Gram-Schmidt orthogonalization
        # --------------------------------------------------
        print("Step 2: Gram-Schmidt orthogonalization")

        HS_orth, means, coeffs = pansharpening.orthogonalize_hyperspectral(
            hs["data"],
            I0
        )

        # --------------------------------------------------
        # Step 3: Adjust PAN
        # --------------------------------------------------
        print("Step 3: Adjusting PAN")

        PAN_adjusted, gain, bias = pansharpening.adjust_pan(
            pan["data"],
            I
        )

        # --------------------------------------------------
        # Save adjusted PAN
        # --------------------------------------------------
        pan_adjusted_path = pansharpening.get_output_path(
            pan_path,
            "_adjusted"
        )

        pansharpening.save_single_band(
            PAN_adjusted,
            pan_adjusted_path,
            pan["profile"],
            pan["nodata"]

        )

        # --------------------------------------------------
        # Save orthogonalized HS
        # --------------------------------------------------
        hs_ortho_path = pansharpening.get_output_path(
            hs_path,
            "_ortho"
        )

        pansharpening.save_multiband(
            HS_orth,
            hs_ortho_path,
            hs["profile"],
            hs["descriptions"],
            hs["nodata"]
        )

        logger.info("Adjusted PAN: %s", pan_adjusted_path)
        logger.info("Orthogonalized HS: %s", hs_ortho_path)

        logger.info("1st Stage of Pansharpening completed")

        return hs_ortho_path, pan_adjusted_path, means, coeffs

    def crop_aoi(self, hs_image, pan_image, aoi_polygon, over_bbox):
        """
        Crop processed PRISMA hyperspectral and panchromatic images
        to the AOI bounding box.

        The AOI is projected to the raster CRS and intersected with the
        valid overlapping extent of the PAN and HS images.

        If an output file already exists, cropping is skipped.

        Args:
            hs_image:
                Path to the processed hyperspectral image.

            pan_image:
                Path to the processed panchromatic image.

            aoi_polygon:
                AOI polygon in WGS84 coordinates.

            over_bbox:
                Valid overlapping bounding box of the PAN and HS images,
                expressed in the raster CRS.

        Returns:
            tuple:
                hs_cropped_path:
                    Path to the AOI-cropped hyperspectral image.

                pan_cropped_path:
                    Path to the AOI-cropped panchromatic image.

                aoi_bbox:
                    Final AOI bounding box in the raster CRS.
        """

        hs_image = Path(hs_image)
        pan_image = Path(pan_image)

        # ---------------------------------------------------------
        # Get raster CRS
        # ---------------------------------------------------------

        with rasterio.open(pan_image) as src:
            target_crs = src.crs

        if target_crs is None:
            raise ValueError(
                f"PAN image has no CRS: {pan_image}"
            )

        # ---------------------------------------------------------
        # Project AOI to raster CRS
        # ---------------------------------------------------------

        _, aoi_projected = prisma_crop.polygon_to_bbox(
            aoi_polygon,
            target_crs,
        )

        # ---------------------------------------------------------
        # Intersect AOI with valid PAN/HS overlap
        # ---------------------------------------------------------

        aoi_bbox = prisma_crop.bbox_intersection(
            over_bbox,
            aoi_projected,
        )

        if aoi_bbox is None:
            raise ValueError(
                "AOI does not intersect the valid PAN/HS overlap."
            )

        minx, miny, maxx, maxy = aoi_bbox

        if minx >= maxx or miny >= maxy:
            raise ValueError(
                f"Invalid AOI bounding box after intersection: {aoi_bbox}"
            )

        # ---------------------------------------------------------
        # Convert AOI bbox to WGS84 for deterministic hash
        # ---------------------------------------------------------

        transformer = Transformer.from_crs(
            target_crs,
            "EPSG:4326",
            always_xy=True,
        )

        lon1, lat1 = transformer.transform(
            minx,
            miny,
        )

        lon2, lat2 = transformer.transform(
            maxx,
            maxy,
        )

        aoi_bbox_wgs84 = (
            min(lon1, lon2),
            min(lat1, lat2),
            max(lon1, lon2),
            max(lat1, lat2),
        )

        h = bbox_hash(aoi_bbox_wgs84)

        # ---------------------------------------------------------
        # Create output filenames
        # ---------------------------------------------------------

        hs_cropped_path = hs_image.with_name(
            f"{hs_image.stem}_AOI_{h}{hs_image.suffix}"
        )

        pan_cropped_path = pan_image.with_name(
            f"{pan_image.stem}_AOI_{h}{pan_image.suffix}"
        )

        # ---------------------------------------------------------
        # Crop hyperspectral image
        # ---------------------------------------------------------

        if hs_cropped_path.exists():
            logger.info(
                "AOI-cropped hyperspectral image already exists. "
                "Skipping crop: %s",
                hs_cropped_path,
            )
        else:
            prisma_crop.crop_geotiff_by_bbox(
                hs_image,
                hs_cropped_path,
                aoi_bbox,
            )

            logger.info(
                "AOI-cropped hyperspectral image created: %s",
                hs_cropped_path,
            )

        # ---------------------------------------------------------
        # Crop panchromatic image
        # ---------------------------------------------------------

        if pan_cropped_path.exists():
            logger.info(
                "AOI-cropped panchromatic image already exists. "
                "Skipping crop: %s",
                pan_cropped_path,
            )
        else:
            prisma_crop.crop_geotiff_by_bbox(
                pan_image,
                pan_cropped_path,
                aoi_bbox,
            )

            logger.info(
                "AOI-cropped panchromatic image created: %s",
                pan_cropped_path,
            )

        return hs_cropped_path, pan_cropped_path, aoi_bbox
        

    # =============================================================
    # MAIN PIPELINE
    # =============================================================

    def run(self, aoi: dict, prisma_scenes: list, config: dict):
        """
        Execute the complete pipeline.

        The AOI is supplied at runtime and is therefore not part
        of the fixed YAML configuration.

        Args:
            aoi: User-provided AOI JSON.
            prisma_input: User-provided list of PRISMA images for the defined aoi
            config: config.yaml from configs directory

        Returns:
            Pipeline results.
        """

        logger.info("Starting pipeline - Terravision PRISMA Pansharpening")

        if not prisma_scenes:
            logger.warning("No PRISMA scenes provided by the user.")
            return 0

        # Step 0. Check if PRISMA images are already pansharpened and available on ICCS STAC - Search by name and aoi
        '''
        matching_items, missing_prisma_files = (
            self.check_processed_prisma_scenes(
                prisma_scenes,
                aoi,
            )
        )

        # All PRISMA scenes are already processed
        if not missing_prisma_files:
            logger.info(
                "All PRISMA scenes have already been processed. Exiting pipeline."
            )
            return matching_items

        # Otherwise continue only with missing scenes
        logger.info(
            "%d PRISMA scene(s) require processing.",
            len(missing_prisma_files),
        )
        '''

        missing_prisma_files = prisma_scenes # Before deployment delete this line - Only for testing

        # Permanent output/data directory
        tmp_root = Path("tmp")
        tmp_root.mkdir(parents=True, exist_ok=True)

        # Step 1. Check if PRISMA files exist in the prisma data directory
        prisma_paths = []
        for prisma in missing_prisma_files:
            prisma_path = self.prisma_data_root / prisma
            if prisma_path.exists():
                prisma_paths.append(prisma_path)
            else:
                logger.warning("PRISMA file does not exist: %s", prisma_path)

        # Step 2. Unzip PRISMA Files
        prisma_filepaths = []
        for prisma_path in prisma_paths:
            prisma_filepaths.append(unzip_prisma_file(prisma_path, tmp_root))

        # Step 3. Read PRISMA Files
        for prisma_file in prisma_filepaths:

            pan_path = generate_PAN(
                prisma_file,
                tmp_root,
            )

            (
                hs_path,
                wavelengths,
                fwhm,
                hs_metadata,
            ) = generate_HS(
                prisma_file,
                tmp_root,
            )
            
            # Step 4. Search and Download Sentinel-2 - Red band from CDSE STAC Catalog
            footprint = get_raster_footprint_wgs84(pan_path)
            prisma_acquisition_datetime = self.get_prisma_date(pan_path)

            sen2_path = self.search_sentinel2(mapping(footprint), prisma_acquisition_datetime, tmp_root)
            if sen2_path is None:
                logger.warning(
                    "No suitable Sentinel-2 scene found. "
                    "Pipeline terminated."
                )
                return

            # Step 5. Coregistration
            coregistered_pan_path = self.coregistration(str(pan_path), str(sen2_path))

            red_index = self.find_band(hs_path, target_wavelength=664)
            coregistered_hs_path = self.coregistration(str(hs_path), str(sen2_path), prisma_red_index = red_index)

            # Step 6. Crop Images
            pan_cropped_path, hs_cropped_path, over_bbox = self.crop(coregistered_pan_path, coregistered_hs_path)

            # Step 7. 1st Stage Pansharpening
            hs_ortho_path, pan_adjusted_path, means, coeffs = self.pansharpen_1st_stage(pan_cropped_path, hs_cropped_path)

            # Step 8. Crop to AOI
            if self.crop_to_aoi: 
                logger.info('Crop to aoi')
                hs_cropped_path, pan_cropped_path, aoi_bbox = self.crop_aoi(hs_ortho_path, pan_adjusted_path, aoi, over_bbox)


            # Step 9. 2st Stage Pansharpening - Needs correct and parallelization
            pansharpened_image_path = pansharpening.pansharpening_2nd_stage(hs_cropped_path, pan_cropped_path, wavelengths, fwhm, means, coeffs, ratio = 6)



        



        '''

        # 11. Upload on ICCS S3
        output = self.save_results(results)

        logger.info("Pipeline finished successfully")

        return output
        '''