"""Clustering base class."""

from abc import ABC, abstractmethod

from lib.common.mix_ins import YAMLBuildMixIn


class ManifoldBase(ABC, YAMLBuildMixIn):
    """Clustering Base Class."""

    @abstractmethod
    def fit(self, data):
        """Cluster data.

        Args:
            data (numpy.ndarray(numpy.float)): Data to fit with shape [n_samples, n_features]

        Returns:
            numpy.ndarray(numpy.float)): Data in embedded space with shape
                [n_samples, reduced_feature_space]

        """
