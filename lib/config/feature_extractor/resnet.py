"""Config class for ResNet Feature extractor."""

from lib.config.base_class import ConfigBase


class ResNetConfig(ConfigBase):

    # TODO Json schema
    json_schema = {
        "type": "object",
        "properties": {
            "frozen_graph_path": {
                "type": "string"
            },
            "input_layer": {
                "type": "string"
            },
            "output_layer": {
                "type": "string"
            }
        },
        "required": ["frozen_graph_path", "input_layer", "output_layer"]
    }

    def __init__(self, config):
        super().__init__(config)
