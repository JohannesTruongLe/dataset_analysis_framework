"""Base Class for Feature Extractors"""
from abc import ABC, abstractmethod

from lib.common.mix_ins import YAMLBuildMixIn


class FeatureExtractorBase(ABC, YAMLBuildMixIn):
    """Feature extractor base class."""

    @abstractmethod
    def inference(self, input_data):
        """Perform inference.

        Args:
            input_data (numpy.ndarray(np.float)): Input data to perform inference on.

        Returns:
            numpy.ndarray(np.float): Output features

        """
