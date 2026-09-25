import json
import argparse
import logging
from typing import Any
from pathlib import Path

from config import load_config
from PipelineConfig import PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

def validate_aoi(aoi: Any) -> Any:
    """Validate GeoJSON geometry."""

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
    """Run the TERRAVISION thermal sharpening pipeline."""

    logger.info(
        "Starting TERRAVISION thermal sharpening pipeline"
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
        datetime=acquisition_datetime,
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
            "with a bounding box and an optional acquisition datetime."
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
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