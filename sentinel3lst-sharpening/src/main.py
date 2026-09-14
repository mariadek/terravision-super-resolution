import argparse
import json
import logging
from pathlib import Path
from typing import Any

from PipelineConfig import PipelineConfig
from config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def validate_coordinates(coordinates: Any) -> None:
    """Recursively validate GeoJSON coordinates."""

    if not isinstance(coordinates, list) or not coordinates:
        raise ValueError("GeoJSON 'coordinates' must be a non-empty list.")

    # Coordinate pair: [longitude, latitude]
    if all(isinstance(value, (int, float)) for value in coordinates):
        if len(coordinates) < 2:
            raise ValueError(
                "Each coordinate must contain at least "
                "[longitude, latitude]."
            )

        lon, lat = coordinates[:2]

        if not -180 <= lon <= 180:
            raise ValueError(
                f"Invalid longitude {lon}. "
                "Longitude must be between -180 and 180."
            )

        if not -90 <= lat <= 90:
            raise ValueError(
                f"Invalid latitude {lat}. "
                "Latitude must be between -90 and 90."
            )

        return

    for item in coordinates:
        validate_coordinates(item)


def validate_aoi(aoi: Any) -> Any:
    """Validate either a bounding box or GeoJSON geometry."""

    # --------------------------------------------------
    # Option 1: Bounding box
    # [min_lon, min_lat, max_lon, max_lat]
    # --------------------------------------------------
    if isinstance(aoi, list):
        if len(aoi) != 4:
            raise ValueError(
                "Bounding-box AOI must contain four values: "
                "[min_lon, min_lat, max_lon, max_lat]."
            )

        if not all(isinstance(value, (int, float)) for value in aoi):
            raise ValueError("AOI bounding-box coordinates must be numeric.")

        min_lon, min_lat, max_lon, max_lat = aoi

        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            raise ValueError(
                "AOI longitude values must be between -180 and 180."
            )

        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            raise ValueError(
                "AOI latitude values must be between -90 and 90."
            )

        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError(
                "Invalid AOI bounding box: minimum coordinates must "
                "be smaller than maximum coordinates."
            )

        return [float(value) for value in aoi]

    # --------------------------------------------------
    # Option 2: GeoJSON geometry
    # --------------------------------------------------
    if isinstance(aoi, dict):
        geometry_type = aoi.get("type")
        coordinates = aoi.get("coordinates")

        if geometry_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError(
                "GeoJSON AOI type must be 'Polygon' or 'MultiPolygon'."
            )

        if coordinates is None:
            raise ValueError(
                "GeoJSON AOI is missing the 'coordinates' field."
            )

        validate_coordinates(coordinates)

        return {
            "type": geometry_type,
            "coordinates": coordinates,
        }

    raise ValueError(
        "'aoi' must be either a bounding box or a GeoJSON "
        "Polygon/MultiPolygon."
    )


def load_user_input(
    user_input_path: str,
) -> tuple[Any, str | None]:
    """Load and validate runtime inputs from a JSON file."""

    path = Path(user_input_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"User input file not found: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as f:
            user_input: dict[str, Any] = json.load(f)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in user input file '{path}': {exc}"
        ) from exc

    # AOI is required
    if "aoi" not in user_input:
        raise ValueError(
            "Missing required field: 'aoi'"
        )

    aoi = validate_aoi(user_input["aoi"])

    # Datetime is optional
    acquisition_datetime = user_input.get("datetime")

    if acquisition_datetime in ("", None):
        acquisition_datetime = None

    elif not isinstance(acquisition_datetime, str):
        raise ValueError(
            "'datetime' must be a string, null, or empty."
        )

    return aoi, acquisition_datetime


def main(user_input_path: str, config_path: str) -> None:
    """Run the TERRAVISION EnMAP pansharpening pipeline."""

    logger.info(
        "Starting TERRAVISION EnMAP pansharpening pipeline"
    )

    logger.info(
        "Loading user input from: %s",
        user_input_path,
    )

    aoi, acquisition_datetime = load_user_input(
        user_input_path
    )

    logger.info(
        "AOI type: %s",
        aoi.get("type")
        if isinstance(aoi, dict)
        else "BoundingBox",
    )

    if acquisition_datetime:
        logger.info(
            "Acquisition datetime: %s",
            acquisition_datetime,
        )
    else:
        logger.info(
            "No acquisition datetime provided"
        )

    config = load_config(config_path)

    pipeline = PipelineConfig(config)

    pipeline.run(
        aoi=aoi,
        datetime=acquisition_datetime
    )

    logger.info(
        "Pipeline completed successfully"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "TERRAVISION Sentinel-3 LST Thermal Sharpening Pipeline"
        )
    )

    parser.add_argument(
        "--user_input",
        required=True,
        help=(
            "Path to a JSON file containing a GeoJSON AOI "
            "or bounding box and an optional acquisition datetime."
        ),
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help=(
            "Path to the pipeline configuration file "
            "(default: config/config.yaml)."
        ),
    )

    args = parser.parse_args()

    try:
        main(
            user_input_path=args.user_input,
            config_path=args.config,
            )

    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc