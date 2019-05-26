"""Mix in implementations."""

from ..config import Config


class YAMLBuildMixIn:

    @classmethod
    def build_from_yaml(cls, config_path):
        """Build class instance from YAML.

        Args:
            config_path (str): Path to config file.

        Returns:
            Class instance

        """
        config = Config.build_from_yaml(config_path=config_path)
        return cls(**config.config)
