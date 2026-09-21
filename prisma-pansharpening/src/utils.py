import logging
from pathlib import Path
from zipfile import ZipFile
from shapely.geometry import shape
import rasterio

logger = logging.getLogger(__name__)


def unzip_prisma_file(prisma_zip_path: str | Path, output_path: str | Path) -> Path:
    """
    Extract all contents of a PRISMA ZIP archive and return the path
    to the contained .he5 file.

    The archive is extracted into a directory next to the ZIP file,
    named after the ZIP archive.

    Before extracting, all archive members are checked. If every file
    already exists, extraction is skipped. If one or more files are
    missing, the archive is extracted again.

    Args:
        prisma_zip_path: Path to the PRISMA ZIP archive.
        output_path: Directory of the extracted .he5 file

    Returns:
        Path to the extracted .he5 file.

    Raises:
        FileNotFoundError: If the ZIP archive does not exist.
        ValueError: If no .he5 file exists in the archive.
    """
    prisma_zip_path = Path(prisma_zip_path)

    # ---------------------------------------------------------
    # Validate ZIP
    # ---------------------------------------------------------

    if not prisma_zip_path.is_file():
        raise FileNotFoundError(
            f"PRISMA ZIP file not found: {prisma_zip_path}"
        )

    # image.zip -> image/
    extract_dir = output_path / prisma_zip_path.stem

    # ---------------------------------------------------------
    # Read archive
    # ---------------------------------------------------------

    with ZipFile(prisma_zip_path, "r") as zip_file:

        members = zip_file.infolist()

        # Find HE5 files
        he5_files = [
            Path(member.filename)
            for member in members
            if not member.is_dir()
            and member.filename.lower().endswith(".he5")
        ]

        if not he5_files:
            raise ValueError(
                f"No .he5 file found in archive: {output_path}"
            )

        if len(he5_files) > 1:
            logger.warning(
                "Multiple .he5 files found in %s. Using %s",
                output_path,
                he5_files[0],
            )

        # ---------------------------------------------------------
        # Check extracted contents
        # ---------------------------------------------------------

        missing_files = [
            member.filename
            for member in members
            if not member.is_dir()
            and not (extract_dir / member.filename).is_file()
        ]

        # ---------------------------------------------------------
        # Extract if necessary
        # ---------------------------------------------------------

        if missing_files:

            logger.info(
                "%d file(s) missing from %s. Extracting archive...",
                len(missing_files),
                extract_dir,
            )

            extract_dir.mkdir(parents=True, exist_ok=True)

            zip_file.extractall(extract_dir)

            logger.info(
                "PRISMA archive extracted successfully: %s",
                extract_dir,
            )

        else:

            logger.info(
                "All PRISMA archive contents already extracted: %s",
                extract_dir,
            )

        # ---------------------------------------------------------
        # Return HE5
        # ---------------------------------------------------------

        prisma_file = extract_dir / he5_files[0]

        if not prisma_file.is_file():
            raise FileNotFoundError(
                f"PRISMA .he5 file was not extracted correctly: {prisma_file}"
            )

    return prisma_file


def intersection_percentage(aoi_geojson, multipolygon_geojson):
    # Convert both geometries to shapely
    aoi_geom = shape(aoi_geojson)
    multipoly = shape(multipolygon_geojson)

    # Compute intersection
    intersection = aoi_geom.intersection(multipoly)

    if intersection.is_empty:
        return 0.0

    intersection_area = intersection.area
    aoi_area = aoi_geom.area

    return (intersection_area / aoi_area) * 100