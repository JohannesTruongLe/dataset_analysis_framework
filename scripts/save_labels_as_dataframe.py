# TODO DOC STRING ME
from lib.dataloader.KITTI import KITTIDataLoader
from lib.util import configure_logging_verbosity, default_config_parse


def save_labels_as_dataframe(dataloader_config):
    """Save labels as dataframe.

    This function just wraps the dataloader.base_class.store_labels() method. Specify the place to store the labels in
    the dataloader config.

    Args:
        dataloader_config (str): Path to dataloader config.

    """
    dataloader = KITTIDataLoader.build_from_yaml(config_path=dataloader_config)  # TODO factory?
    dataloader.store_labels()


def _main():
    """Main script."""
    args = default_config_parse(default_config_path='settings/dataloader/KITTI.yaml')
    configure_logging_verbosity(verbose=args.verbose)
    save_labels_as_dataframe(dataloader_config=args.config)


if __name__ == '__main__':
    _main()
