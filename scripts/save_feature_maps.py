# TODO DOC STRING ME
# TODO Add logging
import argparse
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image

import lib.feature_extractor.resnet as resnet

# TODO Argparse
OUTPUT_PATH = Path("C:/workspace/data/meta/feature_maps")
INPUT_PATH = Path('C:/workspace/data/KITTI/image_2/training/image_2')
INFERENCE_LIST_PATH = Path('C:/workspace/data/meta/inference_list.txt')


def save_features(file_list, output_path, model):
    for file in tqdm.tqdm(file_list):
        img = Image.open(file)
        image_np = np.array(img)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        out = model.retrieve_tensor(input_data=image_np_expanded)
        np.save(output_path/file.stem, out)


def get_label_names(inference_list_path):
    inference_list = np.loadtxt(str(inference_list_path), dtype=np.str)
    label_names = [entry.split('_')[0] for entry in inference_list]
    label_names = np.unique(label_names)

    return label_names


def _main():
    label_names = get_label_names(INFERENCE_LIST_PATH)
    label_paths = [INPUT_PATH/(label_name + '.png') for label_name in label_names]
    model = resnet.ResNet()
    save_features(file_list=label_paths,
                  output_path=OUTPUT_PATH,
                  model=model)


if __name__ == '__main__':
    _main()
