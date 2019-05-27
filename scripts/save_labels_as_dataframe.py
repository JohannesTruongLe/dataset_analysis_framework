"""Save all labels to pickled pandas Data Frame.

This feature will output a .pickle file to the predefined location in the config
(default: settings/datalaoder/KITTI.yaml). The pickle file will hold a pandas.DataFrame with size
[n_bounding_boxes, 15], where n_samples is the total amount of bounding boxes in the KITTI dataset
and 15 refers to the annotated attributes of each bounding box. The index of each bounding box is
FileName_Idx where FileName is the base name of the file of the label and Idx is the position in the
file.

"""
import logging

from lib.config.general_config import Config
from lib.dataloader.KITTI import KITTIDataLoader
from lib.util.logging_util import configure_logging_verbosity
from lib.util.argparse_util import default_config_parse

LOGGER = logging.getLogger(__name__)


def save_labels_as_dataframe(image_path, label_path, save_labels_as_dataframe_path):
    """Save labels as dataframe.

    This function just wraps the dataloader.base_class.store_labels() method. Specify the place to
    store the labels in the dataloader config.

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
