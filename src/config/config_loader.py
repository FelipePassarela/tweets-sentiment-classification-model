import yaml


def get_config(config_path="src/config/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
