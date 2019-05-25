# TODO DOCSTRING ME
import numpy as np
import pandas as pd
from pathlib import Path
from lib.dataloader.constants import (TYPE, X_MIN, X_MAX, Y_MIN, Y_MAX,)
from PIL import Image

import tqdm

# TODO Config
# TODO Argparse

PATH_TO_FEATURES = Path('C:/workspace/data/meta/feature_maps_complete')
PATH_TO_LABELS = Path('C:/workspace/data/meta/data.pickle')
PATH_TO_IMAGES = Path("C:/workspace/data/KITTI/image_2/training/image_2")
OUTPUT_PATH = Path('C:/workspace/data/meta/box_features')


def save_bounding_box_features(path_to_features, path_to_images, output_path, path_to_labels):
    labels = pd.read_pickle(str(path_to_labels))

    # Loop through files
    for file in tqdm.tqdm(list(path_to_features.glob('*'))):
        base_name = file.stem

        feature_map = np.load(file)

        # Get needed labels
        labels_per_image = labels.loc[[base_name], [TYPE, Y_MIN, Y_MAX, X_MIN, X_MAX]]
        image_size = Image.open(path_to_images / (base_name + '.png')).size

        for idx, label in enumerate(labels_per_image.iterrows()):
            label = label[1]  # The first entry is the index of the dataframe, which we do not need
            bbox_feature = _get_bbox_feature(image_size=image_size,
                                             feature_map=feature_map,
                                             x_min=label[X_MIN],
                                             x_max=label[X_MAX],
                                             y_min=label[Y_MIN],
                                             y_max=label[Y_MAX])

            file_name = output_path / (base_name + '_' + str(idx))
            np.save(file_name, bbox_feature)


def _get_bbox_feature(image_size, feature_map, x_min, x_max, y_min, y_max):
    feature_map_size = feature_map.shape[:2]

    bbox_center_orig_images = ((x_min + x_max) / 2,
                               (y_min + y_max) / 2)
    bbox_center_relative = [center / size for center, size in zip(bbox_center_orig_images, image_size)]

    bbox_center_feature_map = [int(relative_position * size)
                               for size, relative_position
                               in zip(feature_map_size, bbox_center_relative)]
    bbox_feature = feature_map[bbox_center_feature_map[0], bbox_center_feature_map[1], :]
    return bbox_feature


def _main():
    save_bounding_box_features(path_to_features=PATH_TO_FEATURES,
                               path_to_images=PATH_TO_IMAGES,  # Images are needed due to the size TODO Get rid of this need
                               output_path=OUTPUT_PATH,
                               path_to_labels=PATH_TO_LABELS)


if __name__ == '__main__':
    _main()
