# TODO DOCSTRING ME
import argparse

import tqdm
import numpy as np
import pandas as pd

from lib.clustering.tsne import TSNE
from lib.config import Config
from lib.util import string_to_bool, save_scatter_plot_with_classes


def _parse_args():
    """Parse inline commands.

    Returns:
        argparse.Namespace: For details type compute_embedded_space.py --help into terminal.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help="Path to config file.",
                        type=str,
                        default='settings/scripts/compute_embedded_space.yaml')
    parser.add_argument('--verbose',
                        help="Increase output verbosity.",
                        required=False,
                        default='False',
                        type=str)

    args = parser.parse_args()
    args.verbose = string_to_bool(args.verbose)

    return args


def load_data_from_directory(feature_path, type_container=None):
    """Load data from directory and prepare.

    Args:
        feature_path (Str): Path to bounding box features.
        type_container (pandas.DataFrame or None): pandas Dataframe holding the types of whole dataset. Index must
            match the file names in feature_path.

    Returns:
        numpy.ndarray(numpy.float): Features of bounding boxes in size [n_samples, n_features]
        types (str): Class type refering to each sample. Only returned if type_container is not None.
        identifier (str): Uniqie label identifier. Only returned if type_container is not None.

    """
    data = []
    file_list = list(feature_path.glob('*'))
    for path_to_data in tqdm.tqdm(file_list, desc='Load features'):
        data.append(np.load(path_to_data))

    data = np.array(data)

    output = data

    if type_container is not None:
        identifier = np.array([file.stem for file in file_list])
        types = np.array([type_container.loc[file.stem] for file in file_list])
        output = (data, types, identifier)

    return output


def _save_embedded_features(feature_path, label_path, output_path, output_plot_path):
    """Perform TSNE and save features.

    Args:
        feature_path (str or pathlib.Path): Path to box features.
        label_path (str or pathlib.Path): Path to pickled pandas Data Frame of labels.
        output_path (str or pathlib.Path or None): Path to save embedded features to. Does not save if None.
        output_plot_path (str or pathlib.Path or None): Path to save plot to. Does not save if None.

    """
    type_container = pd.read_pickle(str(label_path))['type']
    data, types, _ = load_data_from_directory(feature_path, type_container)

    # Normalize data
    data -= np.mean(data)
    data /= np.std(data)

    embedded_space = TSNE().fit(data)
    if output_path:
        np.save(output_path, embedded_space)

    if output_plot_path:
        CLASS_COLOR_DICT = {
            'Car': 'b',
            'Van': 'g',
            'Truck': 'r',
            'Pedestrian': 'c',
            'Person_sitting': 'm',
            'Cyclist': 'm',
            'Tram': 'y'
        }

        save_scatter_plot_with_classes(output_path=output_plot_path,
                                       types=types,
                                       data=embedded_space,
                                       class_color_dict=CLASS_COLOR_DICT)


def _main():
    """Main script."""
    args = _parse_args()
    config = Config.build_from_yaml(args.config)
    _save_embedded_features(**config.config)


if __name__ == '__main__':
    _main()
