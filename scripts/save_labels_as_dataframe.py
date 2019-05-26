# TODO DOC STRING ME
import argparse

from lib.dataloader.KITTI import KITTIDataLoader
from lib.util import configure_logging_verbosity, string_to_bool


def _parse_args():
    """Parse inline commands.

    Returns:
        argparse.Namespace: For details type save_labels_as_dataframe.py --help into terminal.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Path to config file.", required=True, type=str)
    parser.add_argument('--verbose',
                        help="Increase output verbosity.",
                        required=False,
                        default='False',
                        type=str)

    args = parser.parse_args()
    args.verbose = string_to_bool(args.verbose)

    return args


def save_labels_as_dataframe(dataloader_config, verbose=False):
    """Save labels as dataframe.

    This function just wraps the dataloader.base_class.store_labels() method. Specify the place to store the labels in
    the dataloader config.

    Args:
        dataloader_config (str): Path to dataloader config.
        verbose (bool): Whether to turn of verbose or not.

    """
    configure_logging_verbosity(verbose=verbose)
    dataloader = KITTIDataLoader.build_from_yaml(config_path=dataloader_config)  # TODO factory?
    dataloader.store_labels()


def _main():
    """Main script."""
    args = _parse_args()
    save_labels_as_dataframe(dataloader_config=args.config,
                             verbose=args.verbose)


if __name__ == '__main__':
    _main()
