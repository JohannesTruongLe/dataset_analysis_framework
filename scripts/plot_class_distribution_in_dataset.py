"""Script to plot the class distribution in a dataset."""
import logging

import numpy as np
import pandas as pd

from lib.config.general_config import Config
from lib.dataloader.constants import TYPE, CLASS_LIST
from lib.util import configure_logging_verbosity, save_bar_chart, default_config_parse

LOGGER = logging.getLogger(__name__)


def plot_class_distribution(label_path, output_path):
    """Save bar chart of class distribution to disk.

    Args:
        label_path (pathlib.Path or str): Path to dataframe holding all labels.
        output_path (pathlib.Path): Path to save file to.

    """
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
    args = default_config_parse(default_config_path='settings/scripts/plot_class_distribution_in_dataset.yaml')
    configure_logging_verbosity(verbose=args.verbose)
    config = Config.build_from_yaml(args.config)

    plot_class_distribution(**config.config)


if __name__ == '__main__':
    _main()
