import rasterio
import numpy as np
import xml.etree.ElementTree as ET


class EnMAP:
    """Utilities for reading, processing, and writing EnMAP L2A data."""

    def __init__(self, xml_file=None):
        self.xml_file = xml_file
        self.root = None
        self.metadata = {}

        if xml_file is not None:
            self.read_enmap_xml_file(xml_file)
            self.metadata = self.get_metadata_properties()

    def read_enmap_xml_file(self, file_path):
        """Read and parse an EnMAP XML metadata file."""

        try:
            tree = ET.parse(file_path)
            self.root = tree.getroot()
            return self.root

        except ET.ParseError as e:
            raise ValueError(
                f"Error parsing XML file {file_path}: {e}"
            ) from e

    def get_metadata_properties(self):
        """Extract properties from the EnMAP metadata XML."""

        if self.root is None:
            raise RuntimeError("XML file has not been loaded.")

        wavelength = [
            float(item.text)
            for item in self.root.findall(
                "specific/bandCharacterisation/bandID/"
                "wavelengthCenterOfBand"
            )
        ]

        fwhm = [
            item.text
            for item in self.root.findall(
                "specific/bandCharacterisation/bandID/FWHMOfBand"
            )
        ]

        gains = [
            item.text
            for item in self.root.findall(
                "specific/bandCharacterisation/bandID/GainOfBand"
            )
        ]

        offsets = [
            item.text
            for item in self.root.findall(
                "specific/bandCharacterisation/bandID/OffsetOfBand"
            )
        ]

        band_statistics_std_dev = [
            item.text
            for item in self.root.findall(
                "product/bandStatistics/bandID/stdDeviation"
            )
        ]

        band_statistics_mean = [
            item.text
            for item in self.root.findall(
                "product/bandStatistics/bandID/mean"
            )
        ]

        cloud_cover = float(
            self.root.find(
                "specific/qualityFlag/cloudCover"
            ).text
        )

        snow_cover = float(
            self.root.find(
                "specific/qualityFlag/snowCover"
            ).text
        )

        return {
            "wavelength": wavelength,
            "fwhm": fwhm,
            "gains": gains,
            "offsets": offsets,
            "bandStatisticsStdDev": band_statistics_std_dev,
            "bandStatisticsMean": band_statistics_mean,
            "cloud_cover": cloud_cover,
            "snow_cover": snow_cover,
        }

    
    def get_band_list(self, detector_overlap):
        """Determine the band list according to detector overlap."""

        wavelength = self.metadata["wavelength"]

        values = np.array(wavelength, float)

        assert np.all(
            values[:-1] <= values[1:]
        ), "wavelengths are assumed to be sorted"

        vnir_band_numbers = [
            int(text)
            for text in self.root.find(
                "specific/vnirProductQuality/"
                "expectedChannelsList"
            ).text.split(",")
        ]

        swir_band_numbers = [
            int(text)
            for text in self.root.find(
                "specific/swirProductQuality/"
                "expectedChannelsList"
            ).text.split(",")
        ]

        overlap_start = wavelength[swir_band_numbers[0] - 1]
        overlap_end = wavelength[vnir_band_numbers[-1] - 1]

        if detector_overlap == "OrderByDetectorOverlapOption":
            band_list = vnir_band_numbers + swir_band_numbers

        elif detector_overlap == "OrderByWavelengthOverlapOption":
            band_list = list(range(1, len(wavelength) + 1))

        elif detector_overlap == "VnirOnlyOverlapOption":
            band_list = vnir_band_numbers.copy()

            band_list.extend(
                band_no
                for band_no in swir_band_numbers
                if wavelength[band_no - 1] > overlap_end
            )

        elif detector_overlap == "SwirOnlyOverlapOption":
            band_list = [
                band_no
                for band_no in vnir_band_numbers
                if wavelength[band_no - 1] < overlap_start
            ]

            band_list.extend(swir_band_numbers)

        else:
            raise ValueError(
                f"Unknown detector overlap option: {detector_overlap}"
            )

        return band_list

    def remove_bad_bands(self, band_list):
        """Remove bands with zero standard deviation or invalid mean."""

        std_dev = self.metadata["bandStatisticsStdDev"]
        mean = self.metadata["bandStatisticsMean"]

        good_band_list = []
        removed_band_list = []

        for band_no in band_list:
            if (
                std_dev[band_no - 1] != "0"
                and mean[band_no - 1] != "-1000000"
            ):
                good_band_list.append(band_no)
            else:
                removed_band_list.append(band_no)

        return good_band_list, removed_band_list

    def remove_water_absorption_bands(self, band_list):
        """Remove bands affected by water vapor absorption."""

        wavelength = self.metadata["wavelength"]

        water_ranges = [
            (1331.0, 1448.0),
            (1796.0, 1938.0),
        ]

        absorption_centers = [
            940,
            1130,
            725,
            760,
            820,
        ]

        absorption_ranges = [
            (center - 1, center + 1)
            for center in absorption_centers
        ]

        ranges = water_ranges + absorption_ranges

        def in_any_range(value):
            return any(
                start <= value <= end
                for start, end in ranges
            )

        final_band_list = [
            band_no
            for band_no in band_list
            if not in_any_range(
                wavelength[band_no - 1]
            )
        ]

        return final_band_list

    def read_enmap(self, enmap_filename, band_list):
        """Read selected bands from an EnMAP GeoTIFF."""

        with rasterio.open(enmap_filename) as src:
            data = src.read(
                indexes=band_list
            ).astype("float32")

            profile = src.profile.copy()

        return data, profile

    def write_enmap(
        self,
        enmap_filename,
        data,
        profile,
        final_band_list,
    ):
        """Write selected EnMAP bands to a GeoTIFF."""

        wavelength = self.metadata["wavelength"]
        fwhm = self.metadata["fwhm"]
        std_dev = self.metadata["bandStatisticsStdDev"]
        mean = self.metadata["bandStatisticsMean"]

        profile.update({
            "driver": "GTiff",
            "dtype": "float32",
            "count": len(final_band_list),
            "compress": "DEFLATE",
            "predictor": 2,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "interleave": "band",
        })

        wavelength_sel = [
            str(wavelength[i - 1])
            for i in final_band_list
        ]

        fwhm_sel = [
            fwhm[i - 1]
            for i in final_band_list
        ]

        with rasterio.open(
            enmap_filename,
            "w",
            **profile,
        ) as dst:

            dst.write(data)

            # Global metadata
            dst.update_tags(
                wavelength="{" + ", ".join(wavelength_sel) + "}",
                wavelength_units="nanometers",
                fwhm="{" + ", ".join(fwhm_sel) + "}",
            )

            # Per-band metadata
            for i, band_index in enumerate(
                final_band_list,
                start=1,
            ):
                band_std = std_dev[band_index - 1]
                band_mean = mean[band_index - 1]

                bbl = (
                    "0"
                    if (
                        band_std == "0"
                        and band_mean == "-1000000"
                    )
                    else "1"
                )

                dst.set_band_description(
                    i,
                    f"band {band_index} "
                    f"({wavelength_sel[i - 1]} Nanometers)",
                )

                dst.update_tags(
                    i,
                    bbl=bbl,
                )