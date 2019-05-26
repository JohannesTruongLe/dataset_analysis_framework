"""Yaml utility methods."""

import yaml


def load_yaml(path):
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.BaseLoader)
    return config
