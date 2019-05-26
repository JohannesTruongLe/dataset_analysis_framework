""""This scripts uses an DNN to build feature maps and save them to the disk.

The script will store for each image mentioned in inference_list.txt a feature map to a predefined location. The
features will be stored as .npy files to the disk. All three values are specified by default in
settings/scripts/compute_feature_maps.yaml. The DNN used is a ResNet from the
Tensorflow Object Detection API. To download its weights as a frozen graph, follow this link:
http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_kitti_2018_01_28.tar.gz.

"""
import logging
import numpy as np
import tqdm
from PIL import Image

from lib.config import Config
import lib.feature_extractor.resnet as resnet
from lib.util import configure_logging_verbosity, default_config_parse

LOGGER = logging.getLogger(__name__)


def save_features(file_list, output_path, model):
    """Perform inference and save feature to disk.

    Args:
        file_list (list(str or pathlib.Path)): List holding absolute paths to images to peform inference on.
        output_path (str or pathlib.Path): Output path to store feature maps in.
        model (feature_extractor.FeatureExtractorBase): Feature extractor model.

    """
    output_path.mkdir(parents=True, exist_ok=True)
    for file in tqdm.tqdm(file_list, desc="Compute Features"):
        img = Image.open(file)
        image_np = np.array(img)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        out = model.inference(input_data=image_np_expanded)
        np.save(output_path/file.stem, out)


def _get_label_names(inference_list_path):
    """Get label name out of inference list.

    This method returns the label base name (so the file base name) not the label identifier.

    Args:
        inference_list_path (str or pathlib.Path): Refer to compute_feature_maps for details.

    Returns:
        numpy.ndarray: List of unique label base names.
    """
    inference_list = np.loadtxt(str(inference_list_path), dtype=np.str)
    label_names = [entry.split('_')[0] for entry in inference_list]
    label_names = np.unique(label_names)

    return label_names


def compute_feature_maps(output_path,
                         input_path,
                         inference_list_path,
                         resnet_config_path,
                         verbose=False):
    """Compute feature maps and save to disk.

    The feature maps will be stored to output_path.

    Args:
        output_path (str or pathlib.Path): Path to save feature maps to.
        input_path (str or pathlib.Path): Path to image folder.
        inference_list_path (str or pathlib.Path): Path to inference list
            (example creation and details at scripts/computer_inference_list.py).
        resnet_config_path (str or pathlib.Path): Path to ResNet config file.
        verbose (bool): Set verbosity.

    """
    LOGGER.info("Compute feature maps ... ")
    configure_logging_verbosity(verbose=verbose)
    label_names = _get_label_names(inference_list_path)
    image_paths = [input_path/(label_name + '.png') for label_name in label_names]
    model = resnet.ResNet.build_from_yaml(resnet_config_path)
    save_features(file_list=image_paths,
                  output_path=output_path,
                  model=model)


def _main():
    """Main script."""
    args = default_config_parse(default_config_path='settings/scripts/compute_feature_maps.yaml')
    configure_logging_verbosity(verbose=args.verbose)
    config = Config.build_from_yaml(args.config)
    compute_feature_maps(**config.config)


if __name__ == '__main__':
    _main()
