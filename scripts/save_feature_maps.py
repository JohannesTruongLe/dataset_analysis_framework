# TODO DOC STRING ME
# TODO Add logging
import argparse
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image

import lib.feature_extractor.resnet as resnet

# TODO Argparse
OUTPUT_PATH = Path("C:/workspace/data/meta/feature_maps_complete")
INPUT_PATH = Path('C:/workspace/data/KITTI/image_2/training/image_2')


def save_features(input_path, output_path, model):
    for file in tqdm.tqdm(list(input_path.glob('*'))):
        img = Image.open(file)
        image_np = np.array(img)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        out = model.retrieve_tensor(input_data=image_np_expanded)
        np.save(output_path/file.stem, out)


def _main():
    model = resnet.ResNet()
    save_features(input_path=INPUT_PATH,
                  output_path=OUTPUT_PATH,
                  model=model)

if __name__ == '__main__':
    _main()