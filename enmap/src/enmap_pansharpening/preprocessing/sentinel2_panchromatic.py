

import os
from osgeo import gdal
from pathlib import Path

class Sentinel2:
    """Utilities for reading and extracting Sentinel-2 bands from a ZIP file."""

    def __init__(self, s2_zip):
        self.s2_zip = s2_zip
        self.subdatasets = None

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

        zip_name = Path(self.s2_zip).stem

        band_names = ["B02", "B03", "B04", "B08"]

        band_files = [
            os.path.join(
                output_directory,
                f"{zip_name}_{band_name}.tiff",
            )
            for band_name in band_names
        ]

        # Open the four bands
        datasets = [gdal.Open(filepath) for filepath in band_files]

        for filepath, ds in zip(band_files, datasets):
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

        output_filepath = os.path.join(
            output_directory,
            f"{zip_name}_mean.tiff",
        )

        driver = gdal.GetDriverByName("GTiff")

        out = driver.Create(
            output_filepath,
            reference_ds.RasterXSize,
            reference_ds.RasterYSize,
            1,
            gdal.GDT_Float32,
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