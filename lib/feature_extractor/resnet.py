# TODO DOCSTRING ME
# TODO BASECLASS

import numpy as np
import tensorflow as tf


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

        # This is needed to get rid of the batch and match the image format
        feature_map = np.swapaxes(feature_map, 1, 2).squeeze(axis=0)

        return feature_map
