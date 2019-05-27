"""Mix in implementations."""
from lib.config.general_config import Config


class YAMLBuildMixIn:
    """YAML Build MixIn class.

    Convenient way of mixing in the build_from_yaml() method into classes.
    """
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
