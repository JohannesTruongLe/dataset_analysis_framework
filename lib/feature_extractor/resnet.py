"""ResNet implementation of FeatureExtractorBase class."""
import numpy as np
import tensorflow as tf

from lib.feature_extractor.base_class import FeatureExtractorBase


class ResNet(FeatureExtractorBase):
    """ResNet Feature Extractor."""

    def __init__(self,
                 frozen_graph_path=None,  # TODO download pls
                 input_layer_name='import/image_tensor:0',
                 output_layer_name='import/FirstStageFeatureExtractor/resnet_v1_101/'
                                   'resnet_v1_101/block3/unit_21/bottleneck_v1/add:0'):
        """Init.

        Args:
            frozen_graph_path (str or pathlib.Path or None): Will download graph if None. Elsewise, Path to .pb file.
            input_layer_name (str): Name of input tensor.
            output_layer_name (str): Name of output tensor.

        """
        self._input_layer_name = input_layer_name
        self._output_layer_name = output_layer_name
        self._frozen_graph_path = str(frozen_graph_path)
        self._graph = self._load_model()
        self._input_tensor = self._graph.get_tensor_by_name(self._input_layer_name)
        self._output_tensor = self._graph.get_tensor_by_name(self._output_layer_name)

    def _load_model(self):
        """Load the model.

        Returns:
            tensorflow.python.framework.ops.Graph: Tensorflow Graph.

        """
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(self._frozen_graph_path, 'rb') as file:
            serialized_graph = file.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def)

        graph = tf.get_default_graph()
        return graph

    def inference(self, input_data):
        """Perform inference.

        Args:
            input_data (numpy.ndarray(np.float)): Input data.

        Returns:
            numpy.ndarray(np.float): Feature map.

        """
        with tf.Session() as sess:
            feature_map = sess.run(self._output_tensor, feed_dict={self._input_tensor: input_data})
        # This is needed to get rid of the batch and match the image format
        feature_map = np.swapaxes(feature_map, 1, 2).squeeze(axis=0)

        return feature_map
