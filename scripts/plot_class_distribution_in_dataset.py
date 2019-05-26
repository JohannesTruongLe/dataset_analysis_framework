"""Script to analyze the class distribution in a dataset."""
import argparse
import logging

import numpy as np
import pandas as pd

from lib.config.general_config import Config
from lib.dataloader.constants import TYPE, CLASS_LIST
from lib.util import string_to_bool, configure_logging_verbosity, save_bar_chart

LOGGER = logging.getLogger(__name__)


def _parse_args():
    """Parse inline commands.

    Returns:
        argparse.Namespace: For details type plot_class_distribution_in_dataset.py --help into terminal.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help="Path to config file.",
                        type=str,
                        default='settings/scripts/plot_class_distribution_in_dataset.yaml')
    parser.add_argument('--verbose',
                        help="Increase output verbosity.",
                        required=False,
                        default='False',
                        type=str)

    args = parser.parse_args()
    args.verbose = string_to_bool(args.verbose)

    return args


def plot_class_distribution(label_path, output_path, verbose=False):
    """Save bar chart of class distribution to disk.

    Args:
        label_path (pathlib.Path or str): Path to dataframe holding all labels.
        output_path (pathlib.Path): Path to save file to.
        verbose (bool): Set verbosity.

    """
    configure_logging_verbosity(verbose=verbose)
    LOGGER.info("Save Class distribution plot ...")
    labels = pd.read_pickle(str(label_path))

    n_samples = []
    for class_type in CLASS_LIST:
        n_samples.append(np.sum(labels[TYPE] == class_type))

    save_bar_chart(data=n_samples,
                   output_path=output_path,
                   y_label="Number of samples",
                   x_tick_labels=CLASS_LIST,
                   title="Class distribution.")
    LOGGER.debug("Finished saving.")


def _main():
    """Main script."""
    args = _parse_args()
    config = Config.build_from_yaml(args.config)

    plot_class_distribution(**config.config,
                            verbose=args.verbose)


if __name__ == '__main__':
    _main()
