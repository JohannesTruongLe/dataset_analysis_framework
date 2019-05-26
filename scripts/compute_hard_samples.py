"""Script to compute hard samples."""  # TODO Better notes
import argparse
from pathlib import Path

import numpy as np
import tqdm
import pandas as pd

from lib.config import Config
from lib.dataloader.constants import X_MIN, X_MAX, Y_MIN, Y_MAX, CLASS_LIST
from lib.util import save_cropped_image, string_to_bool


def _parse_args():
    """Parse inline commands.

    Returns:
        argparse.Namespace: For details type compute_embedded_space.py --help into terminal.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help="Path to config file.",
                        type=str,
                        default='settings/scripts/compute_hard_samples.yaml')
    parser.add_argument('--verbose',
                        help="Increase output verbosity.",
                        required=False,
                        default='False',
                        type=str)

    args = parser.parse_args()
    args.verbose = string_to_bool(args.verbose)

    return args


def save_hard_samples(feature_path, label_path, output_path, image_path, embedded_space_path, n_samples):
    """Save hard samples to disk.

    The closest n objects of one class to another classes median are considered as hard samples.

    Args:
        feature_path (str or pathlib.Path): Path to features, which where used to generate the embedded_space .npy file,
            which is referenced in embedded_space_path.
        label_path (str or pathlib.Path): Path to pickeled pandas Data Frame label of the data set.
        output_path (str or pathlib.Path): Path where to store the output to.
        image_path (str or pathlib.Path): Path to image folder.
        embedded_space_path (str or pathlib.Path): Path to embedded space .npy file. Example generation in
            scripts/compute_embedded_space.py
        n_samples (int): Amount of samples to store per class comparison.

    """
    label_dataframe = pd.read_pickle(str(label_path))

    # Get identifier and types for each sample
    file_list = list(feature_path.glob('*'))
    identifier = np.array([file.stem for file in file_list])
    types = np.array([label_dataframe['type'].loc[file.stem] for file in file_list])

    data = np.load(embedded_space_path)
    median_dict = dict()
    for type in CLASS_LIST:
        median_dict[type] = np.median(data[types == type], axis=0)

    # For each class, find the closest samples from other classes
    for source in CLASS_LIST:
        for target in CLASS_LIST:
            if source is not target:
                type_mask = types == target

                # Find the closest 10 samples from other classes
                target_data = data[type_mask]
                source_vector = np.expand_dims(median_dict[source], axis=0).repeat(target_data.shape[0], axis=0)
                distance = np.linalg.norm((source_vector-target_data), axis=1)
                idx = np.argsort(distance)[:n_samples]
                df_idx_list = identifier[type_mask][idx]

                # Save each sample to the disk
                for diff_id, sample_idx in tqdm.tqdm(enumerate(df_idx_list),
                                                     desc="Comparing %s against %s" % (source, target),
                                                     total=n_samples):
                    Path(output_path/source/target).mkdir(exist_ok=True, parents=True)
                    x_min, x_max, y_min, y_max = label_dataframe.loc[sample_idx][[X_MIN, X_MAX, Y_MIN, Y_MAX]]
                    save_cropped_image(source_path=image_path / (sample_idx.split('_')[0] + '.png'),
                                       output_path=output_path/source/target/(sample_idx + '_' + str(diff_id) + '.png'),
                                       x_min=int(x_min),
                                       x_max=int(x_max),
                                       y_min=int(y_min),
                                       y_max=int(y_max))


def _main():
    """Main script."""
    args = _parse_args()
    config = Config.build_from_yaml(args.config)
    save_hard_samples(**config.config)


if __name__ == '__main__':
    _main()
