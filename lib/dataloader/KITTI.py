"""KITTI Data Loader implementation of DataLoaderBase."""
import logging
import pandas as pd

from lib.dataloader.base_class import DataLoaderBase
from lib.dataloader.constants.KITTI import KITTI_COLS

LOGGER = logging.getLogger(__name__)


class KITTIDataLoader(DataLoaderBase):
    """KITTI DataLoader class."""
    def __init__(self,
                 image_path,
                 label_path,
                 save_labels_as_dataframe_path=None):
        """Init.

        Args:
            image_path (str or pathlib.Path): Path to images of KITTI Dataset.
            label_path (str or pathlib.Path): Path of labels of KITTI Dataset.
            save_labels_as_dataframe_path (str or pathlib.Path): Path to save the pickled pandas
            DataFrame to. Can be None if saving is not needed.

        """
        super().__init__()
        LOGGER.debug("Build KITTI dataloader ...")

        self._image_path = image_path
        self._label_path = label_path
        self._save_labels_as_dataframe_path = save_labels_as_dataframe_path

    def generate_sample(self):
        """Generate sample.

        Yields:
            pandas.DataFrame: Pandas Dataframe holding labels from one file. Column names can be
            seen in dataloader.constants.KITTI_COLS. Index is the FILENAME_LABELPOSITION in file.

        """
        for file in list(self._label_path.iterdir()):
            LOGGER.debug("Reading from %s", file)
            labels_from_one_file = pd.read_csv(file,
                                               index_col=None,
                                               header=None,
                                               names=KITTI_COLS,
                                               delimiter=' ')

            # Set file name + label position in file as index
            base_name = file.stem
            sample_index = [base_name + '_' + str(label_idx)
                            for label_idx in range(labels_from_one_file.shape[0])]
            labels_from_one_file.insert(loc=0, column='name', value=sample_index)
            labels_from_one_file.set_index('name', inplace=True)

            yield labels_from_one_file

    def store_labels(self, output_path=None):
        """Store labels as Data Frame pickle file.

        Args:
            output_path (str or pathlib.Path or None): Path to store file. If None, take path given
                during init.

        """
        if not output_path:
            output_path = self._save_labels_as_dataframe_path

        assert self._save_labels_as_dataframe_path, "No path to store to given."
        LOGGER.debug("Store data ...")
        labels = self.build_label_dataframe()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        labels.to_pickle(output_path)
