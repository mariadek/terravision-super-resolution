

import os
import re
from osgeo import gdal
from pathlib import Path

class Sentinel2:
    """Utilities for reading and extracting Sentinel-2 bands from a ZIP file."""

    def __init__(self, s2_zip=None, subdatasets=None, selectedbands=None):
        if s2_zip is None and subdatasets is None:
            raise ValueError(
                "At least one of s2_zip or subdatasets must be provided."
            )

        self.s2_zip = s2_zip
        self.subdatasets = subdatasets
        self.selectedbands = selectedbands

        if self.s2_zip is None:
            self.selectedbands = self.subdatasets

    def read_sentinel2_zip(self):
        """Open the Sentinel-2 ZIP and get its subdatasets."""
        sen2_file = gdal.Open(self.s2_zip)

        assert sen2_file is not None, (
            f"Could not open Sentinel-2 file: {sen2_file}"
        )

        self.subdatasets = sen2_file.GetMetadata("SUBDATASETS")

        return self.subdatasets

    def write_band(self, ds, band_number, output_filepath):
        """Read one band from a Sentinel-2 subdataset and write it to GeoTIFF."""
        band = ds.GetRasterBand(band_number)
        data = band.ReadAsArray()

        driver = gdal.GetDriverByName("GTiff")

        out = driver.Create(
            output_filepath,
            ds.RasterXSize,
            ds.RasterYSize,
            1,
            gdal.GDT_Int16,
        )

        out.SetGeoTransform(ds.GetGeoTransform())
        out.SetProjection(ds.GetProjection())
        out.GetRasterBand(1).WriteArray(data)
        out.FlushCache()

        data = None
        out = None

    def extract_bands(self, output_directory):
        """Extract Sentinel-2 B02, B03, B04 and B08 bands."""

        self.read_sentinel2_zip()

        for key in self.subdatasets.keys():

            if not key.endswith("_NAME"):
                continue

            ds = gdal.Open(self.subdatasets[key])

            if key != "SUBDATASET_1_NAME":
                continue

            bands = {
                1: ("B04", "B4, central wavelength 665 nm"),
                2: ("B03", "B3, central wavelength 560 nm"),
                3: ("B02", "B2, central wavelength 490 nm"),
                4: ("B08", "B8, central wavelength 842 nm"),
            }

            for band_number, (band_name, expected_description) in bands.items():

                description = ds.GetRasterBand(
                    band_number
                ).GetDescription()

                if description == expected_description:

                    new_filepath = os.path.join(
                        output_directory,
                        f"{Path(self.s2_zip).stem}_{band_name}.tiff",
                    )

                    self.write_band(
                        ds,
                        band_number,
                        new_filepath,
                    )

            ds = None

    def create_mean_image(self, output_directory):
        """Create an image from the pixel-wise mean of B02, B03, B04 and B08."""


        # Open the four bands
        datasets = [gdal.Open(filepath) for filepath in self.selectedbands]

        for filepath, ds in zip(self.selectedbands, datasets):
            assert ds is not None, (
                f"Could not open Sentinel-2 band: {filepath}"
            )

        # Read bands as Float32
        data = [
            ds.GetRasterBand(1).ReadAsArray().astype("float32")
            for ds in datasets
        ]

        # Calculate pixel-wise mean
        mean_image = sum(data) / len(data)

        # Use the first band as reference for spatial information
        reference_ds = datasets[0]

        path = Path(self.selectedbands[0])

        if path.suffix.lower() == ".jp2":
            # Extracted Sentinel-2 band:
            # parent directory contains the product name
            product_name = path.parent.name

        elif path.suffix.lower() in (".tif", ".tiff"):
            # Generated asset:
            # remove _B02, _B03, etc.
            product_name = re.sub(
                r"_B\d{2}$",
                "",
                path.stem,
            )

        else:
            raise ValueError(f"Unsupported band file: {path}")

        output_filepath = Path(output_directory) / f"{product_name}_mean.tiff"

        driver = gdal.GetDriverByName("GTiff")

        out = driver.Create(
            output_filepath,
            reference_ds.RasterXSize,
            reference_ds.RasterYSize,
            1,
            gdal.GDT_Float32,
            options=[
                "COMPRESS=DEFLATE",
                "TILED=YES",
                "PREDICTOR=3",
                "BIGTIFF=IF_SAFER",
            ],
        )

        out.SetGeoTransform(reference_ds.GetGeoTransform())
        out.SetProjection(reference_ds.GetProjection())

        out.GetRasterBand(1).WriteArray(mean_image)
        out.GetRasterBand(1).SetDescription(
            "Mean of B02, B03, B04 and B08"
        )

        out.FlushCache()

        # Close datasets
        for ds in datasets:
            ds = None

        out = None
        data = None
        mean_image = None

        return output_filepath


    