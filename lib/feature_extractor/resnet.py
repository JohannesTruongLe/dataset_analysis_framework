# TODO DOCSTRING ME
# TODO BASECLASS
import urllib.request
import tarfile
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import tqdm

from tensorflow.keras.models import Model

PATH_TO_FROZEN_GRAPH = 'C:/workspace/data/weights/faster_rcnn_resnet101_kitti_2018_01_28/frozen_inference_graph.pb' #TODO Config
INPUT_LAYER = 'import/image_tensor:0'
OUTPUT_LAYER = 'import/FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block3/unit_23/bottleneck_v1/Relu:0'

class ResNet:

    def __init__(self,
                 path_to_frozen_graph=PATH_TO_FROZEN_GRAPH,
                 input_layer_name=INPUT_LAYER,
                 output_layer_name=OUTPUT_LAYER,
                 model_name=None):  # TODO make this download if None

        self._input_layer_name = input_layer_name
        self._output_layer_name = output_layer_name
        self._path_to_frozen_graph = path_to_frozen_graph
        self._graph = self.load_model(name=model_name)

        self._input_tensor = self._graph.get_tensor_by_name(self._input_layer_name) # TODO make this more flexible, dynamic loading needed --> method to write to this variable
        self._output_tensor = self._graph.get_tensor_by_name(self._output_layer_name)

    def load_model(self, path_to_frozen_graph=None, name=None):  #  TODO split content from function
        if not path_to_frozen_graph:
            path_to_frozen_graph = self._path_to_frozen_graph

        graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_frozen_graph, 'rb') as file:
            serialized_graph = file.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name=name)

        graph = tf.get_default_graph()
        return graph

    def retrieve_tensor(self, input_data, output_tensor_name=None, input_tensor_name=None):

        # Handle default arguments or choosing tensors by hand
        if not output_tensor_name and not input_tensor_name:
            output_tensor = self._output_tensor
            input_tensor = self._input_tensor
        else:
            if output_tensor_name:
                output_tensor = self._graph.get_tensor_by_name(output_tensor_name)
            if input_tensor_name:
                input_tensor = self._graph.get_tensor_by_name(input_tensor_name)

        with tf.Session() as sess:
            feature_map = sess.run(output_tensor, feed_dict={input_tensor: input_data})

        return feature_map


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            last_feature_layer = tf.get_default_graph().get_tensor_by_name(
                    'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block3/unit_21/bottleneck_v1/add:0')

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(last_feature_layer, feed_dict={image_tensor: image})

    return output_dict


# TODO Delete this
if __name__ == '__main__':
    # Check if this is working
    model = ResNet()
    for file in tqdm.tqdm(list(Path('C:/workspace/data/KITTI/image_2/training/image_2').glob('*'))):
        img = Image.open(file)
        image_np = np.array(img)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        out = model.retrieve_tensor(input_data=image_np_expanded)
        print(out.shape)

        # np.save(Path("C:/workspace/KITTI_analysis/data/meta/feature_maps_complete")/file.stem, out)
