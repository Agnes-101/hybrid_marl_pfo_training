import yaml
from pathlib import Path

def load_config(config_name="default_configs"):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / f"{config_name}.yaml"
    )

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
        