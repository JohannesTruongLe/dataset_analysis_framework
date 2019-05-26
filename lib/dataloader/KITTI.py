"""KITTI Data Loader implementation of DataLoaderBase."""
import logging
import pandas as pd

from ..config import KITTIConfig
from ..dataloader.base_class import DataLoaderBase
from ..dataloader import KITTI_COLS

LOGGER = logging.getLogger(__name__)


class KITTIDataLoader(DataLoaderBase):
    # TODO Docstring that velo not needed
    def __init__(self,
                 image_path,
                 label_path,
                 save_labels_as_dataframe_path=None):
        """Init.

        Args:

        """
        super().__init__()
        LOGGER.debug("Build KITTI dataloader ...")
        self._image_path = image_path
        self._label_path = label_path
        self._save_labels_as_dataframe_path = save_labels_as_dataframe_path

    @classmethod
    def build_from_yaml(cls, config_path):
        """Build KITTI dataloader from yaml file.

        Args:
            config_path (str): See config.KITTIConfig.build_from_yaml() for further details.

        Returns:
            ComponentHandler: ComponentHandler object.

        """
        kitti_config = KITTIConfig.build_from_yaml(config_path=config_path)
        return cls(**kitti_config.config)

    def generate_sample(self):
        # TODO make this flexible, whether it returns labels or images --> Build into save feature maps script
        # TODO make it raise error if someone wants sth but did not provide path

        for file in list(self._label_path.iterdir()):
            LOGGER.debug("Reading from %s" % file)
            labels_from_one_file = pd.read_csv(file, index_col=None, header=None, names=KITTI_COLS, delimiter=' ')
            # Set file name + label position in file as index
            base_name = file.stem
            sample_index = [ base_name + '_' +str(label_idx) for label_idx in range(labels_from_one_file.shape[0])]
            labels_from_one_file.insert(loc=0, column='name', value=sample_index)
            labels_from_one_file.set_index('name', inplace=True)

            yield labels_from_one_file

    def store_labels(self, output_path = None):
        """Store labels as Data Frame pickle file.

        Args:
            output_path (str or None): Path to store file. If None, take path given during init.

        """
        if not output_path:
            output_path = self._save_labels_as_dataframe_path

        assert self._save_labels_as_dataframe_path, "No path to store to given."
        LOGGER.debug("Store data ...")
        labels = self.build_label_dataframe()
        labels.to_pickle(output_path)
