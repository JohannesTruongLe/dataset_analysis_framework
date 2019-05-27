"""Config class for ResNet Feature extractor."""

from lib.config.base_class import ConfigBase


class ResNetConfig(ConfigBase):
    """ResNetConfig Class."""
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
