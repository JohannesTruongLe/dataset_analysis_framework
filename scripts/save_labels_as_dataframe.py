"""Save all labels to pickled pandas Data Frame.""" # TODO Better doc string hier oben
import logging

from lib.config import Config
from lib.dataloader.KITTI import KITTIDataLoader
from lib.util import configure_logging_verbosity, default_config_parse

LOGGER = logging.getLogger(__name__)


def save_labels_as_dataframe(image_path, label_path, save_labels_as_dataframe_path):
    """Save labels as dataframe.

    This function just wraps the dataloader.base_class.store_labels() method. Specify the place to store the labels in
    the dataloader config.

    Args:
        image_path (str): Refer to KITTIDataLoader.init().
        label_path (str): Refer to KITTIDataLoader.init().
        save_labels_as_dataframe_path (str): Refer to KITTIDataLoader.init().

    """
    LOGGER.info("Save labels ...")
    dataloader = KITTIDataLoader(image_path=image_path,
                                 label_path=label_path,
                                 save_labels_as_dataframe_path=save_labels_as_dataframe_path)
    dataloader.store_labels()


def _main():
    """Main script."""
    args = default_config_parse(default_config_path='settings/dataloader/KITTI.yaml')
    config = Config.build_from_yaml(args.config)
    configure_logging_verbosity(verbose=args.verbose)
    save_labels_as_dataframe(**config.config)


if __name__ == '__main__':
    _main()
