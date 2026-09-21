import argparse
import json
import logging

from config import load_config
from PipelineConfig import PipelineConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main(user_input_path: str, config_path: str) -> None:
    """Run the TERRAVISION EnMAP Pansharpening Pipeline."""

    logger.info("Starting TERRAVISION EnMAP Pansharpening Pipeline")

    # Load user-provided input
    logger.info("Loading user input from: %s", user_input_path)

    with open(user_input_path, "r", encoding="utf-8") as f:
        user_json = json.load(f)

    # Validate required input fields
    if not isinstance(user_json, dict):
        raise ValueError("User input must be a JSON object.")

    required_fields = {"aoi", "prisma_scenes"}
    missing_fields = required_fields - user_json.keys()

    if missing_fields:
        raise ValueError(
            f"Missing required fields: {', '.join(sorted(missing_fields))}"
        )

    aoi = user_json["aoi"]
    prisma_scenes = user_json["prisma_scenes"]

    # Load pipeline configuration
    logger.info("Loading configuration from: %s", config_path)

    config = load_config(config_path)

    # Initialize and execute pipeline
    logger.info("Initializing pipeline")

    pipeline = PipelineConfig(config)

    logger.info("Executing pipeline")

    pipeline.run(
        aoi=aoi,
        prisma_scenes=prisma_scenes,
        config=config,
    )

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TERRAVISION EnMAP Pansharpening Pipeline"
    )

    parser.add_argument(
        "--user_input",
        required=True,
        help=(
            "Path to a JSON file containing the AOI "
            "and PRISMA scenes."
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help=(
            "Path to the pipeline configuration file "
            "(default: configs/config.yaml)."
        ),
    )

    args = parser.parse_args()

    try:
        main(
            user_input_path=args.user_input,
            config_path=args.config,
        )

    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    except Exception:
        logger.exception("Pipeline execution failed")
        raise SystemExit(1)