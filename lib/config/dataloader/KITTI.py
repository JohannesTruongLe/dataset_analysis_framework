"""KITTI Config class."""
from lib.config.base_class import ConfigBase


class KITTIConfig(ConfigBase):
    """KITTIConfig Class implementation."""

    json_schema = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string"
            },
            "label_path": {
                "type": "string"
            },
            "save_labels_as_dataframe_path": {
                "type": "string"
            }
        },
        "required": ["image_path", "label_path"]
    }
