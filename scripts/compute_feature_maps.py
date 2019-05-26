""""This scripts uses an DNN to build feature maps and save them to the disk.""" # TODO refactor parse_args? + BEtter script descrption + Logging
import argparse

import numpy as np
import tqdm
from PIL import Image

from lib.config import Config
import lib.feature_extractor.resnet as resnet
from lib.util import string_to_bool, configure_logging_verbosity


def _parse_args():
    """Parse inline commands.

    Returns:
        argparse.Namespace: For details type compute_feature_maps.py --help into terminal.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help="Path to config file.",
                        type=str,
                        default='settings/scripts/compute_feature_maps.yaml')
    parser.add_argument('--verbose',
                        help="Increase output verbosity.",
                        required=False,
                        default='False',
                        type=str)

    args = parser.parse_args()
    args.verbose = string_to_bool(args.verbose)

    return args


def save_features(file_list, output_path, model):
    """Perform inference and save feature to disk.

    Args:
        file_list (list(str or pathlib.Path)): List holding absolute paths to images to peform inference on.
        output_path (str or pathlib.Path): Output path to store feature maps in.
        model (TODO YO WRITE SOMTH HERE): Feature extractor model.

    """
    for file in tqdm.tqdm(file_list):
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
    configure_logging_verbosity(verbose=verbose)
    label_names = _get_label_names(inference_list_path)
    image_paths = [input_path/(label_name + '.png') for label_name in label_names]
    model = resnet.ResNet.build_from_yaml(resnet_config_path)  # TODO Factory?
    save_features(file_list=image_paths,
                  output_path=output_path,
                  model=model)


def _main():
    """Main script."""
    args = _parse_args()
    config = Config.build_from_yaml(args.config)
    compute_feature_maps(**config.config)


if __name__ == '__main__':
    _main()
