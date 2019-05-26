# TODO DOCSTRING ME

from pathlib import Path

import tqdm
import numpy as np
import matplotlib.pyplot as plt
from lib.clustering.tsne import TSNE
import pandas as pd
from PIL import Image
from lib.dataloader.constants import (TYPE, X_MIN, X_MAX, Y_MIN, Y_MAX,)

# TODO Config
# TODO Argparse

N_SAMPLES = 10
PATH_TO_FEATURES = Path('C:/workspace/data/meta/box_features_trial')
PATH_TO_LABELS = Path('C:/workspace/data/meta/data.pickle')
OUTPUT_PATH = Path('C:/workspace/data/meta/hard_samples')
IMAGE_PATH = Path('C:/workspace/data/KITTI/image_2/training/image_2')
EMBEDDED_PATH = Path('C:/workspace/data/meta/embedded_data.npy')


CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram']


def _load_data_from_directory(path, type_container=None, n_samples=None):

    if not n_samples:
        n_samples = len(list(path.glob('*')))

    data = []
    file_list = list(path.glob('*'))[:n_samples]
    for path_to_data in tqdm.tqdm(file_list):

        data.append(np.load(path_to_data))

    data = np.array(data)

    output = data

    if type_container is not None:

        identifier = np.array([file.stem for file in file_list])
        types = np.array([type_container.loc[file.stem] for file in file_list])
        output = (data, types, identifier)

    return output


def _main():

    label_dataframe = pd.read_pickle(str(PATH_TO_LABELS))
    _, types, identifier = _load_data_from_directory(PATH_TO_FEATURES,
                                                     label_dataframe['type'],
                                                     n_samples=None)

    data = np.load(EMBEDDED_PATH)
    median_dict = dict()
    for type in CLASSES:
        median_dict[type] = np.median(data[types == type], axis=0)

    for source in CLASSES:
        for target in CLASSES:
            if source is not target:
                type_mask = types == target

                target_data = data[type_mask]
                source_vector = np.expand_dims(median_dict[source], axis=0).repeat(target_data.shape[0], axis=0)
                distance = np.linalg.norm((source_vector-target_data), axis=1)
                idx = np.argsort(distance)[:N_SAMPLES]
                df_idx_list = identifier[type_mask][idx]

                for sample_idx in df_idx_list:
                    Path(OUTPUT_PATH/source).mkdir(exist_ok=True)
                    Path(OUTPUT_PATH/source/target).mkdir(exist_ok=True)
                    x_min, x_max, y_min, y_max = label_dataframe.loc[sample_idx][[X_MIN, X_MAX, Y_MIN, Y_MAX]]
                    save_cropped_image(source_path=IMAGE_PATH / (sample_idx.split('_')[0] + '.png'),
                                       output_path=OUTPUT_PATH/source/target/(sample_idx + '.png'),
                                       x_min=int(x_min),
                                       x_max=int(x_max),
                                       y_min=int(y_min),
                                       y_max=int(y_max))


def save_cropped_image(source_path, output_path, x_min, x_max, y_min, y_max, zoom_factor=5):
    #
    image = np.array(Image.open(source_path))

    d_x = abs(x_min-x_max)/2 * zoom_factor
    d_y = abs(y_min-y_max)/2 * zoom_factor

    center_y = (y_max+y_min)/2
    center_x = (x_max+x_min)/2

    y_min = int(max(0, center_y - d_y))
    x_min = int(max(0, center_x - d_x))
    y_max = int(min(image.shape[0], center_y + d_y))
    x_max = int(min(image.shape[1], center_x + d_x))

    cropped_image = image[y_min:y_max, x_min:x_max]
    Image.fromarray(cropped_image).save(output_path)


if __name__ == '__main__':
    _main()
