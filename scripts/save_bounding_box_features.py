# TODO DOCSTRING ME
import numpy as np
import pandas as pd
from pathlib import Path
from lib.dataloader.constants import (TYPE, X_MIN, X_MAX, Y_MIN, Y_MAX,)
from PIL import Image

import tqdm

# TODO Config
# TODO Argparse

PATH_TO_FEATURES = Path('C:/workspace/data/meta/feature_maps')
PATH_TO_LABELS = Path('C:/workspace/data/meta/data.pickle')
PATH_TO_IMAGES = Path("C:/workspace/data/KITTI/image_2/training/image_2")
OUTPUT_PATH = Path('C:/workspace/data/meta/box_features_trial')
INFERENCE_LIST_PATH = Path('C:/workspace/data/meta/inference_list.txt')


def save_bounding_box_features(path_to_features, path_to_images, output_path, path_to_labels, path_to_inference_list):
    labels = pd.read_pickle(str(path_to_labels))
    inference_list = np.loadtxt(str(path_to_inference_list), dtype=np.str)

    # Loop through files
    for label_id in tqdm.tqdm(inference_list):
        base_name = label_id.split('_')[0]
        feature_map_path = path_to_features/(base_name + '.npy')
        feature_map = np.load(feature_map_path)

        # Get needed labels
        image_size = Image.open(path_to_images / (base_name + '.png')).size

        label = labels.loc[label_id]
        bbox_feature = _get_bbox_feature(image_size=image_size,
                                         feature_map=feature_map,
                                         x_min=label[X_MIN],
                                         x_max=label[X_MAX],
                                         y_min=label[Y_MIN],
                                         y_max=label[Y_MAX])

        file_name = output_path / label_id
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
                               path_to_labels=PATH_TO_LABELS,
                               path_to_inference_list=INFERENCE_LIST_PATH)


if __name__ == '__main__':
    _main()
