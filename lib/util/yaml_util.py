"""Yaml utility methods."""
import yaml


def load_yaml(path):
    """Load yaml file into dict.

    Args:
        path (str): Path to load yaml from.

    Returns:

    """
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config
