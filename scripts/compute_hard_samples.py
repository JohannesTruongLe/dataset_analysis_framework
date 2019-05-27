"""Script to compute hard samples.

This scripts computes difficult samples and saves their crops to the disk. Difficult samples are
found as followed:

* The median is calculated for each class.
* For each median, find the closest samples of other classes. These samples are considered hard.

Input and output locations are defined in settings/scripts/compute_hard_samples.yaml by default.

"""
from pathlib import Path

import numpy as np
import tqdm
import pandas as pd

from lib.config.general_config import Config
from lib.dataloader.constants.KITTI import X_MIN, X_MAX, Y_MIN, Y_MAX, CLASS_LIST
from lib.util.image_util import save_cropped_image
from lib.util.logging_util import configure_logging_verbosity
from lib.util.argparse_util import default_config_parse


def save_hard_samples(feature_path,
                      label_path,
                      output_path,
                      image_path,
                      embedded_space_path,
                      n_samples):
    """Save hard samples to disk.

    The closest n objects of one class to another classes median are considered as hard samples.

    Args:
        feature_path (str or pathlib.Path): Path to features, which where used to generate the
            embedded_space .npy file, which is referenced in embedded_space_path.
        label_path (str or pathlib.Path): Path to pickeled pandas Data Frame label of the data set.
        output_path (str or pathlib.Path): Path where to store the output to.
        image_path (str or pathlib.Path): Path to image folder.
        embedded_space_path (str or pathlib.Path): Path to embedded space .npy file. Example
            generation in scripts/compute_embedded_space.py
        n_samples (int): Amount of samples to store per class comparison.

    """
    label_dataframe = pd.read_pickle(str(label_path))

    # Get identifier and types for each sample
    file_list = list(feature_path.glob('*'))
    identifier = np.array([file.stem for file in file_list])
    types = np.array([label_dataframe['type'].loc[file.stem] for file in file_list])

    data = np.load(embedded_space_path)
    median_dict = dict()
    for type_class in CLASS_LIST:
        median_dict[type_class] = np.median(data[types == type_class], axis=0)

    # For each class, find the closest samples from other classes
    for source in CLASS_LIST:
        for target in CLASS_LIST:
            if source is not target:
                type_mask = types == target

                # Find the closest 10 samples from other classes
                target_data = data[type_mask]
                source_vector = np.expand_dims(median_dict[source],
                                               axis=0).repeat(target_data.shape[0], axis=0)
                distance = np.linalg.norm((source_vector-target_data), axis=1)
                idx = np.argsort(distance)[:n_samples]
                df_idx_list = identifier[type_mask][idx]

                # Save each sample to the disk
                for diff_id, sample_idx in tqdm.tqdm(
                        enumerate(df_idx_list),
                        desc="Comparing %s against %s" % (source, target),
                        total=n_samples):
                    Path(output_path/source/target).mkdir(exist_ok=True, parents=True)
                    x_min, x_max, y_min, y_max = label_dataframe.loc[sample_idx][[X_MIN,
                                                                                  X_MAX,
                                                                                  Y_MIN,
                                                                                  Y_MAX]]
                    source_path = image_path / (sample_idx.split('_')[0] + '.png')
                    output_path = output_path/source/target / \
                                  (sample_idx + '_' + str(diff_id) + '.png')
                    save_cropped_image(source_path=source_path,
                                       output_path=output_path,
                                       x_min=int(x_min),
                                       x_max=int(x_max),
                                       y_min=int(y_min),
                                       y_max=int(y_max))


def _main():
    """Main script."""
    args = default_config_parse(default_config_path='settings/scripts/compute_hard_samples.yaml')
    configure_logging_verbosity(verbose=args.verbose)
    config = Config.build_from_yaml(args.config)
    save_hard_samples(**config.config)


if __name__ == '__main__':
    _main()
