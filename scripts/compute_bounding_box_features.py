"""Save bounding box features to disk."""  # TODO describe better what it does
import argparse

import numpy as np
import pandas as pd
from PIL import Image
import tqdm

from lib.config import Config
from lib.dataloader.constants import X_MIN, X_MAX, Y_MIN, Y_MAX
from lib.util import string_to_bool


def _parse_args():
    """Parse inline commands.

    Returns:
        argparse.Namespace: For details type compute_bounding_box_features.py --help into terminal.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help="Path to config file.",
                        type=str,
                        default='settings/scripts/compute_bounding_box_features.yaml')
    parser.add_argument('--verbose',
                        help="Increase output verbosity.",
                        required=False,
                        default='False',
                        type=str)

    args = parser.parse_args()
    args.verbose = string_to_bool(args.verbose)

    return args


def save_bounding_box_features(feature_path, image_path, output_path, label_path, inference_list_path):  # TODO Use generators instead of pasing paths
    """Save features of each bounding box given in inference list to disk.

    Each feature saved to disk will have the name of the unique label.

    Args:
        feature_path (path or pathlib.Path): Path to feature maps matching inference list.
        image_path (path or pathlib.Path): Path to images which hold images in inference list.
        output_path (path or pathlib.Path): Path to save features to.
        label_path (path or pathlib.Path): Path to pickled pandas Data Frame file.
        inference_list_path (path or pathlib.Path): Path to inference list.

    """
    labels = pd.read_pickle(str(label_path))
    inference_list = np.loadtxt(str(inference_list_path), dtype=np.str)

    # Loop through files
    for label_id in tqdm.tqdm(inference_list, desc='Save bounding boxes'):
        base_name = label_id.split('_')[0]
        feature_map_path = feature_path / (base_name + '.npy')
        feature_map = np.load(feature_map_path)

        # Get needed labels
        image_size = Image.open(image_path / (base_name + '.png')).size

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
    """Get bounding box features.

    The feature is the center pixel of a bounding box.

    Args:
        image_size (tuple): Size of an image in [x, y].
        feature_map (numpy.ndarray(numpy.float)): Feature map of size [x, y, n_features] of whole image where the
            bounding box lies in.
        x_min (int): Minimum X coordinate of bounding box.
        x_max (int): Maximum X coordinate of bounding box.
        y_min (int): Minimum Y coordinate of bounding box.
        y_max (int): Maximum Y coordinate of bounding box.

    Returns:
        numpy.ndarray(numpy.float): Feature of size [n_features.]

    """
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
    """Main script."""
    args = _parse_args()
    config = Config.build_from_yaml(args.config)
    save_bounding_box_features(**config.config)


if __name__ == '__main__':
    _main()
