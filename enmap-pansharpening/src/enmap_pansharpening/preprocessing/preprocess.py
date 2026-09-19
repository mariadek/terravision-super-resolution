import logging
from pathlib import Path

from enmap_pansharpening.preprocessing.enmap_band_removal import EnMAP
from enmap_pansharpening.preprocessing.sentinel2_panchromatic import Sentinel2
from enmap_pansharpening.download.models import PreprocessedPair


logger = logging.getLogger(__name__)


def preprocess_enmap(
        enmap_path: str | Path,
        temp_directory: str | Path,
    ) -> tuple[Path, list[str], list[float]]:
        """Preprocess one EnMAP product and return its explicit output path."""

        enmap_path = Path(enmap_path)
        temp_directory = Path(temp_directory)


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

        temp_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        processed_path = temp_directory / data_file.name

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
        sentinel2_path: str | Path,
        temp_directory: str | Path,
    ) -> tuple[Path, Path]:
        """
        Extract Sentinel-2 bands from zip and create the mean/PseudoPAN image.

        Returns:
            Sentinel-2 B04 path and mean/PseudoPAN path.
        """

        sentinel2_path = Path(sentinel2_path)
        temp_directory = Path(temp_directory)

        if sentinel2_path is None:
            raise FileNotFoundError(
               f"Expected file not found: {sentinel2_path}"
            )

        sentinel2 = Sentinel2(s2_zip = str(sentinel2_path))

        temp_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        sentinel2.extract_bands(
            output_directory=temp_directory,
        )

        
        band_names = ["B02", "B03", "B04", "B08"]

        band_files = [
            os.path.join(
                temp_directory,
                f"{sentinel2_path.stem}_{band_name}.tiff",
            )
            for band_name in band_names
        ]

        sentinel2.selectedbands = band_files

        pan_path = Path(
            sentinel2.create_mean_image(
                output_directory=temp_directory,
            )
        )

        b04_path = (
            temp_directory
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
        sentinel2_path: str | Path,
        temp_directory: str | Path,
    ) -> tuple[Path, Path]:
        """
        Create the mean/PseudoPAN image from Sentinel-3 assets.

        Returns:
            Sentinel-2 B04 path and mean/PseudoPAN path.
        """

        sentinel2_path = Path(sentinel2_path)
        temp_directory = Path(temp_directory)

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

        temp_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        pan_path = Path(
            sentinel2.create_mean_image(
                output_directory=temp_directory,
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
        enmap_path: str | Path,
        sentinel2_path: str | Path,
        download_full_sen2_item: bool | None = False,
        temp_directory: str | Path | None = "tmp"
    ) -> PreprocessedPair:
        """Preprocess downloaded pairs and keep paths/metadata together."""

        (
            processed_enmap,
            wavelength,
            fwhm,
        ) = preprocess_enmap(enmap_path, temp_directory)        

        if download_full_sen2_item:
            (
                sentinel2_b04,
                sentinel2_pan,
            ) = preprocess_sentinel2_zip(sentinel2_path, temp_directory)
        else:
                (
                sentinel2_b04,
                sentinel2_pan,
            ) = preprocess_sentinel2_assets(sentinel2_path, temp_directory)


        processed_pair = PreprocessedPair(
                enmap_path=processed_enmap,
                sentinel2_b04_path=sentinel2_b04,
                sentinel2_pan_path=sentinel2_pan,
                wavelength=wavelength,
                fwhm=fwhm,
            )

        return processed_pair
