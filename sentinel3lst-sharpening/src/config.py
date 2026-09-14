from pathlib import Path
import yaml

def load_config(path="configs/example.yaml"):
    path = Path(path)

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config