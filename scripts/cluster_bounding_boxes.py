# TODO DOCSTRING ME

from pathlib import Path

import tqdm
import numpy as np
import matplotlib.pyplot as plt
from lib.clustering.tsne import TSNE
import pandas as pd

# TODO Config
# TODO Argparse

N_SAMPLES = 21
PATH_TO_FEATURES = Path('C:/workspace/data/meta/box_features_trial')
PATH_TO_LABELS = Path('C:/workspace/data/meta/data.pickle')


CLASS_COLOR_DICT = {
    'Car': 'b',
    'Van': 'g',
    'Truck': 'r',
    'Pedestrian': 'c',
    'Person_sitting': 'm',
    'Cyclist': 'm',
    'Tram': 'y'
}


def _load_data_from_directory(path, n_samples=None):

    if not n_samples:
        n_samples = -1

    data = []
    for path_to_data in tqdm.tqdm(list(path.glob('*'))[:n_samples]):
        data.append(np.load(path_to_data))

    return np.array(data)


def _plot(type_container, embedded_space, class_color_dict):

    for type, color in class_color_dict.items():
        subset = embedded_space[type_container == type, :]
        plt.scatter(subset[:, 0], subset[:, 1], c=color, label=type)
        plt.legend()
    plt.show()


def _main():

    type_container = pd.read_pickle(str(PATH_TO_LABELS))['type'][:N_SAMPLES]
    data = _load_data_from_directory(PATH_TO_FEATURES, n_samples=N_SAMPLES)

    # Normalize data
    data -= np.mean(data)
    data /= np.std(data)

    embedded_space = TSNE(n_iter=1000).fit(data)
    _plot(type_container=type_container,
          embedded_space=embedded_space,
          class_color_dict=CLASS_COLOR_DICT)


if __name__ == '__main__':
    _main()
