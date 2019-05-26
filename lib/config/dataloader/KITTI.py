"""KITTI Config class."""

from lib.config.base_class import ConfigBase


class KITTIConfig(ConfigBase):

    # TODO Json schema
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

    def __init__(self, config):
        super().__init__(config)
