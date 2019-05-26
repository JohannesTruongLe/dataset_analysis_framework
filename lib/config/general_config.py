"""General Config class."""

from .base_class import ConfigBase


class Config(ConfigBase):

    json_schema = {
        "type": "object",
        "required": []
    }

    def __init__(self, config):
        """Init.

        Args:
            config (AttrDict): AttrDict holding the configs in the structure shown in json_schema.

        """
        super().__init__(config)
