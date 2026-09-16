from pathlib import Path
import yaml


def load_config(path=None):

    # Use default configuration if no path is provided
    if not path:
        path = "configs/config.yaml"

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config