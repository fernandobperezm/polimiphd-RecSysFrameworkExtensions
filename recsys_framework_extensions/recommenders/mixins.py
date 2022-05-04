from typing import Optional

from recsys_framework_extensions.data.io import DataIO

from recsys_framework_extensions.logging import get_logger

logger = get_logger(
    logger_name=__file__,
)


class MixinEmptySaveModel:
    RECOMMENDER_NAME = ""

    def save_model(self, folder_path: str, file_name: str = None):
        logger.debug(
            f"CALLED {self.save_model.__name__} in {MixinEmptySaveModel.__name__}"
        )

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        DataIO.s_save_data(
            folder_path=folder_path,
            file_name=file_name,
            data_dict_to_save=dict(),
        )


class MixinLoadModel:
    RECOMMENDER_NAME = ""

    def load_model(self, folder_path: str, file_name: str = None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        logger.info(
            f"Loading model from file '{folder_path + file_name}'"
        )

        data_dict = DataIO.s_load_data(
            folder_path=folder_path,
            file_name=file_name,
        )

        for attrib_name in data_dict.keys():
            self.__setattr__(attrib_name, data_dict[attrib_name])

        logger.info(
            "Loading complete"
        )
