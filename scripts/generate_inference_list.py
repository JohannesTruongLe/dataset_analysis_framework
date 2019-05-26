# TODO DOCSTRING ME
import numpy as np
import pandas as pd
from pathlib import Path
from lib.dataloader.constants import (TYPE, X_MIN, X_MAX, Y_MIN, Y_MAX,)

# TODO Config
# TODO Argparse

PATH_TO_LABELS = Path('C:/workspace/data/meta/data.pickle')
OUTPUT_PATH = Path('C:/workspace/data/meta/inference_list.txt')

CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram']


def compute_inference_list(path_to_labels, seed=42):
    random_state = np.random.RandomState(seed)
    labels = pd.read_pickle(str(path_to_labels))
    n_samples_dict = dict()

    # Count samples per class
    for type in CLASSES:
        n_samples_dict[type] = np.sum(labels[TYPE] == type)

    # Get from each class the same amount of samples like the class with the smallest number of samples
    min_n = n_samples_dict[min(n_samples_dict, key=n_samples_dict.get)]
    inference_list = []
    for type in CLASSES:
        labels_one_class = labels[labels[TYPE] == type]
        identifier = random_state.choice(labels_one_class.index.values, size=min_n, replace=False)
        inference_list.append(identifier)

    inference_list = [item for sublist in inference_list for item in sublist]
    np.savetxt(str(OUTPUT_PATH), inference_list, fmt='%s')


def _main():
    compute_inference_list(path_to_labels=PATH_TO_LABELS)


if __name__ == '__main__':
    _main()
