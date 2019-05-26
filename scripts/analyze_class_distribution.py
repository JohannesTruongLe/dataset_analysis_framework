# TODO DOCSTRING ME
import numpy as np
import pandas as pd
from pathlib import Path
from lib.dataloader.constants import (TYPE, X_MIN, X_MAX, Y_MIN, Y_MAX,)
import matplotlib.pyplot as plt
import tqdm

# TODO Config
# TODO Argparse

PATH_TO_LABELS = Path('C:/workspace/data/meta/data.pickle')
CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram']


def analyze(path_to_labels):
    labels = pd.read_pickle(str(path_to_labels))

    n_samples = []
    for type in CLASSES:
        n_samples.append(np.sum(labels[TYPE] == type))

    ind = np.arange(len(CLASSES))  # the x locations for the groups
    width = 0.75       # the width of the bars

    fig, ax = plt.subplots()
    rects = ax.bar(ind, n_samples, width, color='b')

    ax.set_ylabel('Numer of Samples')
    ax.set_title('Samples per Class')
    ax.set_xticks(ind)
    ax.set_xticklabels(CLASSES)


    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.0*height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    autolabel(rects)

    plt.show()


def _main():
    analyze(path_to_labels=PATH_TO_LABELS)


if __name__ == '__main__':
    _main()
