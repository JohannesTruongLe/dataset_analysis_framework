# TODO DOC STRING ME
# TODO Add logging
import argparse
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image

from lib.dataloader.KITTI import KITTIDataLoader
# TODO Argparse
OUTPUT_PATH = Path("C:/workspace/data/meta/data.pickle")
INPUT_PATH = Path('C:/workspace/data/KITTI/label_2/training/label_2')
N_SAMPLES = None


def save_labels_as_dataframe(output_path, dataloader, n_samples=None):
    dataloader.store_labels(output_path=output_path,
                            n_samples=n_samples)


def _main():
    dataloader = KITTIDataLoader(label_path=INPUT_PATH) # TODO Make it flexible which datalaoder is used, factory?
    save_labels_as_dataframe(output_path=OUTPUT_PATH,
                             dataloader=dataloader,
                             n_samples=N_SAMPLES)


if __name__ == '__main__':
    _main()