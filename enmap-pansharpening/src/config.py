import logging
from pathlib import Path
from typing import Any

import yaml


logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def load_config(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load and validate the pipeline configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.
              Defaults to 'configs/config.yaml'.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML is invalid, empty, or not a mapping.
        OSError: If the configuration file cannot be read.
    """

    # Use default configuration if no path is provided
    config_path = (
        Path(path).expanduser()
        if path is not None
        else DEFAULT_CONFIG_PATH
    )

    # Check file existence
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    logger.info(
        "Loading configuration from: %s",
        config_path,
    )

    # Load YAML configuration
    try:
        with config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            config = yaml.safe_load(file)

    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML configuration in '{config_path}': {exc}"
        ) from exc

    except OSError as exc:
        raise OSError(
            f"Unable to read configuration file "
            f"'{config_path}': {exc}"
        ) from exc

    # Validate configuration structure
    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration file '{config_path}' must contain "
            "a non-empty YAML mapping."
        )

    if not config:
        raise ValueError(
            f"Configuration file '{config_path}' is empty."
        )

    logger.info(
        "Configuration loaded successfully"
    )

    return config