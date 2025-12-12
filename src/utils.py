import os
import yaml


def read_config_yaml(config_path=None):
    if config_path is None:
        config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../src/config.yaml")
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
