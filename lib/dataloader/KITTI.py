# TODO Doc string me

import logging
import tqdm
import pandas as pd


from lib.dataloader.constants import (TYPE, TRUNCATED, OCCLUDED, ALPHA, X_MIN, X_MAX,
                                      Y_MIN, Y_MAX, SIZE_X, SIZE_Y, SIZE_Z, LOCATION_X,
                                      LOCATION_Y, LOCATION_Z, ROTATION_Y)


LOGGER = logging.getLogger(__name__)
KITTI_COLS = [TYPE,  # TODO Move me somewhere, class attribute?
              TRUNCATED,
              OCCLUDED,
              ALPHA,
              X_MIN,
              Y_MIN,
              X_MAX,
              Y_MAX,
              SIZE_X,
              SIZE_Y,
              SIZE_Z,
              LOCATION_X,
              LOCATION_Y,
              LOCATION_Z,
              ROTATION_Y]


class KITTIDataLoader: # TODO BaseClass me

    def __init__(self,
                 image_path=None,
                 label_path=None):
        """Init.

        Args:
            kitti_config (lib.config.config.KITTIConfig): KITTIConfig class.

        """
        LOGGER.debug("Build KITTI dataloader ...")
        # err_helper.check_instance_type(kitti_config, config.KITTIConfig) TODO Make this useable again
        self._image_path = image_path
        self._label_path = label_path

    @classmethod
    def build_from_yaml(cls, config_path):
        """Build KITTI dataloader from yaml file.

        Args:
            config_path (str): See config.KITTIConfig.build_from_yaml() for further details.

        Returns:
            ComponentHandler: ComponentHandler object.

        """
        """
        LOGGER.debug("Load yaml file from %s", config_path)
        kitti_config = config.KITTIConfig.build_from_yaml(config_path=config_path)
        return cls(kitti_config=kitti_config)
        """
        pass  # TODO make this useable again

    def generate_sample(self, n_samples=None):
        # TODO make this flexible, whether it returns labels or images --> Build into save feature maps script
        # TODO make it raise error if someone wants sth but did not provide path
        break_flag = False
        if n_samples:
            break_flag = True

        for idx, file in tqdm.tqdm(enumerate(list(self._label_path.iterdir())),
                                   desc="Load from directory",
                                   total=n_samples):
            labels_from_one_file = pd.read_csv(file, index_col=None, header=None, names=KITTI_COLS, delimiter=' ')

            # Set file name + label position in file as index
            base_name = file.stem
            sample_index = [ base_name + '_' +str(label_idx) for label_idx in range(labels_from_one_file.shape[0])]
            labels_from_one_file.insert(loc=0, column='name', value=sample_index)
            labels_from_one_file.set_index('name', inplace=True)

            # Break if number of needed samples are reached (if n_samples is set)
            if break_flag and idx == (n_samples + 1):
                break
            yield labels_from_one_file

    def build_dataframe(self, n_samples=None):
        dataframe_list = []

        # Handle if we want to have a specific amount of samples
        for idx, frame in enumerate(self.generate_sample(n_samples=n_samples)):
            dataframe_list.append(frame)

        dataframe = pd.concat(dataframe_list)

        return dataframe

    def store_labels(self, output_path, n_samples=None):
        """"""
        LOGGER.debug("Store data ...")
        labels = self.build_dataframe(n_samples=n_samples)
        labels.to_pickle(output_path)
