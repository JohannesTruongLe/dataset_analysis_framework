"""Save bounding box features to disk.

This script calculates for each entry in inference_list.txt the feature of a bounding box, given the
feature maps are already calculated. The features will be stores as .npy files to the disk. The
location and both latter values are by default defined in
settings/scripts/compute_bounding_box_features.yaml.

For a more detailed description about the feature maps refer to scripts/build_feature_maps.py
docstring. For more details for the inference_list.txt, please refer to
scripts/compute_inference_list.py docstring.

"""
import logging

import numpy as np
import pandas as pd
from PIL import Image
import tqdm

from lib.config.general_config import Config
from lib.dataloader.constants.KITTI import X_MIN, X_MAX, Y_MIN, Y_MAX
from lib.util.argparse_util import default_config_parse
from lib.util.logging_util import configure_logging_verbosity

LOGGER = logging.getLogger(__name__)


def save_bounding_box_features(feature_path,
                               image_path,
                               output_path,
                               label_path,
                               inference_list_path):
    """Save features of each bounding box given in inference list to disk.

    Each feature saved to disk will have the name of the unique label.

    Args:
        feature_path (path or pathlib.Path): Path to feature maps matching inference list.
        image_path (path or pathlib.Path): Path to images which hold images in inference list.
        output_path (path or pathlib.Path): Path to save features to.
        label_path (path or pathlib.Path): Path to pickled pandas Data Frame file.
        inference_list_path (path or pathlib.Path): Path to inference list.

    """
    LOGGER.info("Save boundinx box features ...")
    labels = pd.read_pickle(str(label_path))
    inference_list = np.loadtxt(str(inference_list_path), dtype=np.str)
    output_path.mkdir(parents=True, exist_ok=True
                      )
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
        feature_map (numpy.ndarray(numpy.float)): Feature map of size [x, y, n_features] of whole
            image where the bounding box lies in.
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
    bbox_center_relative = [center / size
                            for center, size in zip(bbox_center_orig_images, image_size)]

    bbox_center_feature_map = [int(relative_position * size)
                               for size, relative_position
                               in zip(feature_map_size, bbox_center_relative)]
    bbox_feature = feature_map[bbox_center_feature_map[0], bbox_center_feature_map[1], :]
    return bbox_feature


def _main():
    """Main script."""
    args = default_config_parse(
        default_config_path='settings/scripts/compute_bounding_box_features.yaml')
    configure_logging_verbosity(verbose=args.verbose)
    config = Config.build_from_yaml(args.config)
    save_bounding_box_features(**config.config)


if __name__ == '__main__':
    _main()
