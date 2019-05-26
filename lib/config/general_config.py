"""General Config class."""

from .base_class import ConfigBase


class Config(ConfigBase):

    # TODO Json schema
    json_schema = {
        "type": "object",
        "required": []
    }

    def __init__(self, config):
        super().__init__(config)
