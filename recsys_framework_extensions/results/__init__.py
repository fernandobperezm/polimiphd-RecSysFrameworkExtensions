import logging
from typing import Any

import pandas as pd

from recsys_framework_extensions.data.io import DataIO
from Utils.ResultFolderLoader import ResultFolderLoader, _get_algorithm_file_name_list


logger = logging.getLogger(__file__)


def _get_algorithm_metadata_to_print_list(
    result_folder_path,
    algorithm_list,
    KNN_similarity_list=None,
    ICM_names_list=None,
    UCM_names_list=None,
):
    dataIO = DataIO(folder_path=result_folder_path)

    algorithm_file_name_list = _get_algorithm_file_name_list(
        algorithm_list=algorithm_list,
        KNN_similarity_list=KNN_similarity_list,
        ICM_names_list=ICM_names_list,
        UCM_names_list=UCM_names_list,
    )

    algorithm_metadata_to_print_list = []

    for algorithm_file_dict in algorithm_file_name_list:
        if algorithm_file_dict is None:
            algorithm_metadata_to_print_list.append(None)
            continue

        algorithm_file_name = algorithm_file_dict["algorithm_file_name"]

        search_metadata = None

        if algorithm_file_name is not None:
            try:
                search_metadata = dataIO.load_data(algorithm_file_name + "_metadata")
            except FileNotFoundError:
                pass
            except Exception as e:
                print(
                    "Exception: '{}' while reading file: '{}'".format(
                        str(e), algorithm_file_name + "_metadata"
                    )
                )
                pass

        algorithm_file_dict["search_metadata"] = search_metadata
        algorithm_metadata_to_print_list.append(algorithm_file_dict)

    return algorithm_metadata_to_print_list


class ExtendedResultFolderLoader(ResultFolderLoader):
    def __init__(
        self,
        folder_path,
        base_algorithm_list=None,
        other_algorithm_list=None,
        KNN_similarity_list=None,
        ICM_names_list=None,
        UCM_names_list=None,
    ):
        super().__init__(
            folder_path=folder_path,
            base_algorithm_list=base_algorithm_list,
            other_algorithm_list=other_algorithm_list,
            KNN_similarity_list=KNN_similarity_list,
            ICM_names_list=ICM_names_list,
            UCM_names_list=UCM_names_list,
        )

        # Re-read metadata as it can fail silently on the base class (inner functions throw exceptions, but they are caught).
        self._metadata_list = _get_algorithm_metadata_to_print_list(
            self._folder_path,
            algorithm_list=self._algorithm_list,
            KNN_similarity_list=self._KNN_similarity_list,
            ICM_names_list=self._ICM_names_list,
            UCM_names_list=self._UCM_names_list,
        )

    def get_hyperparameters_dataframe(self):
        column_labels = [
            "algorithm_row_label",
            "hyperparameter_name",
            "hyperparameter_value",
        ]

        records: list[dict[str, Any]] = []

        for row_index, row_dict in enumerate(self._metadata_list):
            if row_dict is None:
                # Add None row to preserve the separation between different groups of algorithms
                # I don't like this but is simple enough and it works
                # result_dataframe = result_dataframe.append(
                #     {
                #         "algorithm_row_label": "algorithm_group_{}".format(row_index),
                #         "hyperparameter_name": None,
                #         "hyperparameter_value": None,
                #     },
                #     ignore_index=True,
                # )
                records.append(
                    {
                        "algorithm_row_label": "algorithm_group_{}".format(row_index),
                        "hyperparameter_name": None,
                        "hyperparameter_value": None,
                    },
                )

                continue

            algorithm_row_label = row_dict["algorithm_row_label"]
            search_metadata = row_dict["search_metadata"]

            # If search failed or was not done, add placeholder
            if (
                search_metadata is None
                or search_metadata["hyperparameters_best"] is None
            ):
                hyperparameters_best = {None: None}

            else:
                hyperparameters_best = search_metadata["hyperparameters_best"]

                # If it doesn't have hyperparameters don't add in dataframe
                if len(hyperparameters_best) == 0:
                    continue

            for (
                hyperparameter_name,
                hyperparameter_value,
            ) in hyperparameters_best.items():
                # result_dataframe = result_dataframe.append(
                #     {
                #         "algorithm_row_label": algorithm_row_label,
                #         "hyperparameter_name": hyperparameter_name,
                #         "hyperparameter_value": hyperparameter_value,
                #     },
                #     ignore_index=True,
                # )
                records.append(
                    {
                        "algorithm_row_label": algorithm_row_label,
                        "hyperparameter_name": hyperparameter_name,
                        "hyperparameter_value": hyperparameter_value,
                    },
                )


        result_dataframe = pd.DataFrame.from_records(
            data=records,
            columns=column_labels,
        ).set_index(
            ["algorithm_row_label", "hyperparameter_name"],
            inplace=False,
        )

        return result_dataframe
