"""DataLoader Base class definition."""

from abc import ABC, abstractmethod
import logging

import pandas as pd
import tqdm

from ..common import YAMLBuildMixIn

LOGGER = logging.getLogger(__name__)


class DataLoaderBase(ABC, YAMLBuildMixIn):
    @abstractmethod
    def generate_sample(self): #TODO flexible here?
        """Yield sample from dataset.


        Yields:
            Sample from dataset.

        """

    def build_label_dataframe(self):
        """Build pandas Data Frame holding all labels.

        Data Frame structure will be defined in generate_sample() implementation.

        Returns:
            pandas.Dataframe: Data Frame holding labels.

        """
        dataframe_list = []

        for frame in tqdm.tqdm(self.generate_sample(), desc='Build Data Frame'):
            dataframe_list.append(frame)

        dataframe = pd.concat(dataframe_list)

        return dataframe

    def store_labels(self):
        """Store labels as Data Frame pickle file.

        This is method is not abstract since it is only needed if one wants to store labels.
        """
        raise NotImplementedError()