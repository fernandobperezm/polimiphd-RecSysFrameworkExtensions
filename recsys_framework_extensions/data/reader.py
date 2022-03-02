import os
import traceback
from abc import ABC, abstractmethod

from recsys_framework.Data_manager.DataReader import DataReader

from recsys_framework_extensions.dataset import BaseDataset


class BaseDataReader(DataReader, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    @abstractmethod
    def dataset(self) -> BaseDataset:
        pass

    def load_data(self, save_folder_path=None):
        """
        :param save_folder_path:    path in which to save the loaded dataset
                                    None    use default "dataset_name/original/"
                                    False   do not save
        :return:
        """

        # Use default e.g., "dataset_name/original/"
        if save_folder_path is None:
            save_folder_path = self.DATASET_SPLIT_ROOT_FOLDER + self._get_dataset_name_root() + self._get_dataset_name_data_subfolder()

        # If save_folder_path contains any path try to load a previously built split from it
        if save_folder_path is not False and not self.reload_from_original_data:

            try:
                loaded_dataset = BaseDataset.empty_dataset()
                loaded_dataset.load_data(save_folder_path)

                self._print("Verifying data consistency...")
                loaded_dataset.verify_data_consistency()
                self._print("Verifying data consistency... Passed!")

                loaded_dataset.print_statistics()
                return loaded_dataset

            except FileNotFoundError:

                self._print("Preloaded data not found, reading from original files...")

            except Exception:

                self._print("Reading split from {} caused the following exception...".format(save_folder_path))
                traceback.print_exc()
                raise Exception("{}: Exception while reading split".format(self._get_dataset_name()))

        self._print("Loading original data")
        loaded_dataset = self._load_from_original_file()

        self._print("Verifying data consistency...")
        loaded_dataset.verify_data_consistency()
        self._print("Verifying data consistency... Passed!")

        if save_folder_path not in [False]:

            # If directory does not exist, create
            if not os.path.exists(save_folder_path):
                self._print("Creating folder '{}'".format(save_folder_path))
                os.makedirs(save_folder_path)

            else:
                self._print("Found already existing folder '{}'".format(save_folder_path))

            loaded_dataset.save_data(save_folder_path)

            self._print("Saving complete!")

        loaded_dataset.print_statistics()
        return loaded_dataset
