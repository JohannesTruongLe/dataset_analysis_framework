# TODO DOC STRING ME
# TODO Add logging
import argparse
from pathlib import Path

from lib.dataloader.KITTI import KITTIDataLoader
from lib.util import configure_logging_verbosity
# TODO Argparse
OUTPUT_PATH = Path("C:/workspace/data/meta/data.pickle")
INPUT_PATH = Path('C:/workspace/data/KITTI/label_2/training/label_2')
N_SAMPLES = None


def _parse_args():
    """Parse inline commands.

    Returns:
        argparse.Namespace: Arguments TODO Add stuff here.

    """

    parser =argparse.ArgumentParser()
    parser.add_argument('--config', help="Path to config file.", required=True, type=str)
    parser.add_argument('--verbose', help="Increase output verbosity.", required=False, default=False, type=bool)
    args = parser.parse_args()

    return args


def _main():
    """Main script."""
    args = _parse_args()
    configure_logging_verbosity(verbose=args.verbose)
    dataloader = KITTIDataLoader.build_from_yaml(config_path=args.config) # TODO factory?
    dataloader.store_labels()


if __name__ == '__main__':
    _main()
