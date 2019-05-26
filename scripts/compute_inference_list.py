"""Script to compute list to perform inference on.""" # TODO explain better pls
import argparse

import numpy as np
import pandas as pd

from lib.config.general_config import Config
from lib.dataloader.constants import TYPE, CLASS_LIST #TODO Factory for CLass list?
from lib.util import string_to_bool, configure_logging_verbosity


def _parse_args():
    """Parse inline commands.

    Returns:
        argparse.Namespace: For details type plot_class_distribution_in_dataset.py --help into terminal.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help="Path to config file.",
                        type=str,
                        default='settings/scripts/compute_inference_list.yaml')
    parser.add_argument('--verbose',
                        help="Increase output verbosity.",
                        required=False,
                        default='False',
                        type=str)

    args = parser.parse_args()
    args.verbose = string_to_bool(args.verbose)

    return args


def compute_inference_list(label_path, output_path, seed=42, verbose=False):
    """Compute inference list and save to disk.

    This method will save a .txt file with each column holding an unique identifier for a label. For each class
    n amount of samples are written to the file. n is equal to the minimum amount of samples for a class. For KITTI,
    Pedestrian_sitting is the class with the fewest occurrences (222), so for every class 222 samples would be chosen.

    Args:
        label_path (str or pathlib.Path): Path to labels as pickled pandas Data Frame file.
        output_path (str or pathlib.Path): Path to save the inference list to.
        seed (int): Random seed to enable reproducibility.
        verbose (True): Set verbosity.

    """
    configure_logging_verbosity(verbose=verbose)
    random_state = np.random.RandomState(seed)
    labels = pd.read_pickle(str(label_path))
    n_samples_dict = dict()

    # Count samples per class
    for class_types in CLASS_LIST:
        n_samples_dict[class_types] = np.sum(labels[TYPE] == class_types)

    # Get from each class the same amount of samples like the class with the smallest number of samples
    min_n = n_samples_dict[min(n_samples_dict, key=n_samples_dict.get)]
    inference_list = []
    for class_types in CLASS_LIST:
        labels_one_class = labels[labels[TYPE] == class_types]
        identifier = random_state.choice(labels_one_class.index.values, size=min_n, replace=False)
        inference_list.append(identifier)

    inference_list = [item for sublist in inference_list for item in sublist]
    np.savetxt(str(output_path), inference_list, fmt='%s')


def _main():
    """Main script."""
    args = _parse_args()
    config = Config.build_from_yaml(args.config)

    compute_inference_list(**config.config,
                           verbose=args.verbose)


if __name__ == '__main__':
    _main()