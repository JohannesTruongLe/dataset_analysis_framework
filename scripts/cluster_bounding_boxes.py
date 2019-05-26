# TODO DOCSTRING ME

from pathlib import Path

import tqdm
import numpy as np
import matplotlib.pyplot as plt
from lib.clustering.tsne import TSNE
import pandas as pd

# TODO Config
# TODO Argparse

N_SAMPLES = 1364
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
        n_samples = len(list(path.glob('*')))

    data = []
    file_list = list(path.glob('*'))[:n_samples]
    for path_to_data in tqdm.tqdm(file_list):

        data.append(np.load(path_to_data))

    return np.array(data), file_list


def _plot(types, embedded_space, class_color_dict):

    for type, color in class_color_dict.items():
        subset = embedded_space[types == type, :]
        plt.scatter(subset[:, 0], subset[:, 1], c=color, label=type)
        plt.legend()
    plt.show()


def _main():

    type_container = pd.read_pickle(str(PATH_TO_LABELS))['type']
    data, file_list = _load_data_from_directory(PATH_TO_FEATURES, n_samples=None)
    types = np.array([type_container.loc[file.stem] for file in file_list])

    # Normalize data
    data -= np.mean(data)
    data /= np.std(data)

    embedded_space = TSNE().fit(data)
    np.save('C:/workspace/data/meta/embedded_data', data)
    _plot(types=types,
          embedded_space=embedded_space,
          class_color_dict=CLASS_COLOR_DICT)


if __name__ == '__main__':
    _main()
