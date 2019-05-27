"""Run whole toolchain.

The toolchain performs TSNE on the dataset and find samples which are hard to classify. This is done
by comparing thefeatures produced by a ResNet architecture. This is done by extracting objects with
features similar to other classes.

The script does executes following steps:

1. Turn all KITTI labels to pandas DataFrame
2. Plot Class Distribution across dataset,
3. Compute list of images and bounding boxes to perform inference on (class balancement is enforced)
4. Compute feature maps for every image chosen in step 3.
5. Get features for every single bounding box in step 3.
6. Compute TSNE on all bounding boxes.
7. Grab the hard samples.

For details for every single step, pleae refer to the scripts.

"""

from lib.config.general_config import Config
from lib.util.argparse_util import default_config_parse
from lib.util.logging_util import configure_logging_verbosity
from scripts.save_labels_as_dataframe import save_labels_as_dataframe
from scripts.plot_class_distribution_in_dataset import plot_class_distribution
from scripts.compute_inference_list import compute_inference_list
from scripts.compute_feature_maps import compute_feature_maps
from scripts.compute_bounding_box_features import save_bounding_box_features
from scripts.compute_embedded_space import save_embedded_features
from scripts.compute_hard_samples import save_hard_samples


def _run_toolchain(config):
    """Run whole toolchain.

    Args:
        config (attrdict.AttrDict): Dictionary holding the configuration for the whole toolchain.

    """
    save_labels_as_dataframe(**config.save_labels_as_dataframe)
    plot_class_distribution(**config.plot_class_distribution)
    compute_inference_list(**config.compute_inference_list)
    compute_feature_maps(**config.compute_feature_maps)
    save_bounding_box_features(**config.compute_bounding_box_features)
    save_embedded_features(**config.compute_embedded_space)
    save_hard_samples(**config.compute_hard_samples)


def _main():
    """Main script."""
    args = default_config_parse(default_config_path='settings/run.yaml')
    configure_logging_verbosity(verbose=args.verbose)
    config = Config.build_from_yaml(args.config)
    _run_toolchain(config=config.config)


if __name__ == '__main__':
    _main()
