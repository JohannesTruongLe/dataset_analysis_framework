"""ConfigBase"""
from abc import ABC, abstractmethod
import attrdict

import jsonschema
import logging
from pathlib import Path

from .constants import PATH
from ..util import error_util, yaml_util

LOGGER = logging.getLogger(__name__)


class ConfigBase(ABC):
    """Config Base class."""

    @property
    @abstractmethod
    # Please define a json schema
    def json_schema(self):
        pass

    def __init__(self, config):
        """Init.

        Args:
            config (dict): Configuration stored in a dict.

        """
        jsonschema.validate(config, schema=self.json_schema)
        config = _convert_paths(config)
        self._config = attrdict.AttrDict(config)

    @classmethod
    def build_from_yaml(cls, config_path):
        """Build config class from yaml file.

        Args:
            config_path (str): See config.common.ConfigBase.build_from_yaml() for further details.

        Returns:
            config.config.KITTIConfig: Configuration class.

        """
        LOGGER.debug("Load yaml file from %s", config_path)
        config_dict = yaml_util.load_yaml(config_path)
        return cls(config_dict)

    @property
    def config(self):
        """Return configuration as Attribute Dict.

        Returns:
            attrdict.Attrdict: Configuration.

        """
        return self._config


def _convert_paths(config):
    """"""
    error_util.check_instance_type(config, dict)
    for key, item in config.items():
        # If 'path' in key and instance --> turn str to Path object
        if PATH in key and isinstance(item, str):
            config[key] = Path(item)

        # Go through all branches
        elif isinstance(item, dict):
            config[key] = _convert_paths(item)

    return config
