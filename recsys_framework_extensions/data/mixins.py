import os
from functools import cached_property
from typing import Any, Callable, NamedTuple, Protocol

import attrs
import dask.dataframe as dd
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as st
from Data_manager.DataReader_utils import compute_density
from Data_manager.Dataset import gini_index
from scipy.stats.stats import DescribeResult
from tqdm import tqdm

from recsys_framework_extensions.data.io import DataIO
from recsys_framework_extensions.evaluation import EvaluationStrategy
from recsys_framework_extensions.logging import get_logger

logger = get_logger(
    logger_name=__file__,
)


class InteractionsDataSplits(NamedTuple):
    sp_urm_train: sp.csr_matrix
    sp_urm_validation: sp.csr_matrix
    sp_urm_train_validation: sp.csr_matrix
    sp_urm_test: sp.csr_matrix


class ImpressionsDataSplits(NamedTuple):
    sp_uim_train: sp.csr_matrix
    sp_uim_validation: sp.csr_matrix
    sp_uim_train_validation: sp.csr_matrix
    sp_uim_test: sp.csr_matrix


class DatasetConfigBackupMixin:
    @staticmethod
    def save_config(
        config: object,
        folder_path: str,
    ) -> None:
        file_name = "dataset_config.zip"
        if DataIO.s_file_exists(folder_path=folder_path, file_name=file_name):
            return

        DataIO.s_save_data(
            folder_path=folder_path,
            file_name=file_name,
            data_dict_to_save={
                "config": attrs.asdict(config),
                "sha256_hash": config.sha256_hash,  # type: ignore
            }
        )


class DataIOMixin:
    def load_from_data_io(
        self,
        file_name: str,
        folder_path: str,
        to_dict_func: Callable[[], dict],
    ) -> dict:
        dict_data: dict

        file_exists = DataIO.s_file_exists(
            folder_path=folder_path,
            file_name=file_name,
        )

        if not file_exists:
            dict_data = to_dict_func()

            DataIO.s_save_data(
                folder_path=folder_path,
                file_name=file_name,
                data_dict_to_save=dict_data
            )

        else:
            dict_data = DataIO.s_load_data(
                folder_path=folder_path,
                file_name=file_name,
            )

        return dict_data


class NumpyDataMixin:
    encoding = "ASCII"
    allow_pickle = True  # Loading impressions (arrays of arrays) require allow_pickle=True

    def _to_numpy(
        self,
        np_arr: np.ndarray,
        file_path: str,
    ) -> None:
        np.savez(
            file=file_path,
            np_arr=np_arr,
        )

    def load_from_numpy(
        self,
        file_path: str,
        to_numpy_func: Callable[[], np.ndarray],
    ) -> np.ndarray:

        if not os.path.exists(file_path):
            np_arr = to_numpy_func()
            self._to_numpy(
                np_arr=np_arr,
                file_path=file_path,
            )

        else:
            npz_data: dict[str, np.ndarray]
            with np.load(
                file_path,
                encoding=self.encoding,
                allow_pickle=self.allow_pickle
            ) as npz_data:
                np_arr = npz_data["np_arr"]

        return np_arr


class ParquetDataMixin:
    engine = "pyarrow"
    use_nullable_dtypes = True

    def _to_parquet(
        self,
        df: pd.DataFrame,
        file_path: str,
    ) -> None:
        logger.debug(
            f"Saving dataframe as parquet in {file_path}."
        )
        df.to_parquet(
            path=file_path,
            engine=self.engine,
        )

    def load_parquet(
        self,
        file_path: str,
        to_pandas_func: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        if not os.path.exists(file_path):
            df_data = to_pandas_func()
            self._to_parquet(
                df=df_data,
                file_path=file_path,
            )

        else:
            df_data = pd.read_parquet(
                path=file_path,
                engine=self.engine,
                use_nullable_dtypes=self.use_nullable_dtypes,
            )

        return df_data

    def load_parquets(
        self,
        file_paths: list[str],
        to_pandas_func: Callable[[], list[pd.DataFrame]],
    ) -> list[pd.DataFrame]:
        any_file_not_created = any(
            not os.path.exists(file_path)
            for file_path in file_paths
        )

        if any_file_not_created:
            dataframes = to_pandas_func()
            for file_path, df in zip(file_paths, dataframes):
                self._to_parquet(
                    df=df,
                    file_path=file_path,
                )

        else:
            dataframes = [
                pd.read_parquet(
                    path=file_path,
                    engine=self.engine,
                    use_nullable_dtypes=self.use_nullable_dtypes,
                )
                for file_path in file_paths
            ]

        return dataframes


class DaskParquetDataMixin:
    engine = "pyarrow"

    def _to_parquet(
        self,
        df: dd.DataFrame,
        folder_path: str,
    ) -> None:
        df.to_parquet(
            path=folder_path,
            engine=self.engine,
        )

    def load_parquet(
        self,
        folder_path: str,
        to_dask_func: Callable[[], dd.DataFrame],
    ) -> pd.DataFrame:
        folder_exists = os.path.exists(folder_path) and os.path.isdir(folder_path)

        if not folder_exists:
            ddf = to_dask_func()

            self._to_parquet(
                df=ddf,
                folder_path=folder_path,
            )
        else:
            ddf = dd.read_parquet(
                path=folder_path,
                engine=self.engine,
            )

        return ddf


class SparseDataMixin:
    compressed = True

    def _sparse_matrix_to_disk(
        self,
        sparse_matrix: sp.spmatrix,
        file_path: str,
    ) -> None:
        logger.debug(
            f"Saving sparse matrix in {file_path}."
        )
        sp.save_npz(
            file_path,
            matrix=sparse_matrix,
            compressed=self.compressed,
        )

    def load_sparse_matrix(
        self,
        file_path: str,
        to_sparse_matrix_func: Callable[[], sp.spmatrix],
    ) -> sp.spmatrix:
        if not os.path.exists(file_path):
            sparse_matrix = to_sparse_matrix_func()
            self._sparse_matrix_to_disk(
                sparse_matrix=sparse_matrix,
                file_path=file_path,
            )

        else:
            sparse_matrix = sp.load_npz(file_path)

        return sparse_matrix

    def load_sparse_matrices(
        self,
        file_paths: list[str],
        to_sparse_matrices_func: Callable[[], list[sp.spmatrix]],
    ) -> list[sp.spmatrix]:
        any_file_not_created = any(
            not os.path.exists(file_path)
            for file_path in file_paths
        )

        if any_file_not_created:
            sparse_matrices = to_sparse_matrices_func()
            for file_path, sparse_matrix in zip(file_paths, sparse_matrices):
                self._sparse_matrix_to_disk(
                    sparse_matrix=sparse_matrix,
                    file_path=file_path,
                )

        else:
            sparse_matrices = [
                sp.load_npz(file_path)
                for file_path in file_paths
            ]

        return sparse_matrices


class BaseDataMixin:
    _save_folder_path: str = ""
    __save_file_name: str = "dataset_global_attributes"

    dataset_name: str
    dataset_config: dict[str, Any]
    dataset_sha256_hash: str
    mapper_item_original_id_to_index: dict[int, int]
    mapper_user_original_id_to_index: dict[int, int]

    def verify_data_consistency(self) -> None:
        pass

    def print_statistics(self) -> None:
        pass

    def _assert_is_initialized(self) -> None:
        pass

    def get_dataset_name(self):
        return self.dataset_name

    def get_mapper_item_original_id_to_index(self):
        return self.mapper_item_original_id_to_index

    def get_mapper_user_original_id_to_index(self):
        return self.mapper_user_original_id_to_index

    def get_global_mapper_dict(self):
        return {
            "user_original_ID_to_index": self.mapper_user_original_id_to_index,
            "item_original_ID_to_index": self.mapper_item_original_id_to_index,
        }

    def save_data(self, save_folder_path) -> None:
        logger.debug(
            f"{self.__class__.__name__}|{self.save_data.__name__}|{self.__save_file_name=}"
        )

        self._save_folder_path = save_folder_path

        DataIO.s_save_data(
            folder_path=self._save_folder_path,
            file_name=self.__save_file_name,
            data_dict_to_save={
                "mapper_item_original_id_to_index": self.mapper_item_original_id_to_index,
                "mapper_user_original_id_to_index": self.mapper_user_original_id_to_index,
                "dataset_name": self.dataset_name,
                "dataset_config": self.dataset_config,
                "dataset_sha256_hash": self.dataset_sha256_hash,
            },
        )

    def load_data(self, save_folder_path) -> None:
        logger.debug(
            f"{self.__class__.__name__}|{self.load_data.__name__}|{self.__save_file_name=}"
        )

        self._save_folder_path = save_folder_path

        global_attributes_dict = DataIO.s_load_data(
            folder_path=self._save_folder_path,
            file_name=self.__save_file_name,
        )

        for attrib_name, attrib_object in global_attributes_dict.items():
            self.__setattr__(attrib_name, attrib_object)


class LazyBaseDataMixin:
    _dataset_name: str = ""
    _dataset_sha256_hash: str = ""
    _dataset_config: dict[str, Any] = dict()
    _mapper_user_original_id_to_index: dict[Any, int] = dict()
    _mapper_item_original_id_to_index: dict[Any, int] = dict()

    _key_dataset_name: str = "dataset_name"
    _key_dataset_config: str = "dataset_config"
    _key_dataset_sha256_hash: str = "dataset_sha256_hash"
    _key_mapper_user_original_id_to_index: str = "mapper_user_original_id_to_index"
    _key_mapper_item_original_id_to_index: str = "mapper_item_original_id_to_index"

    _save_folder_path: str = ""
    __save_file_name: str = "dataset_global_attributes"

    @cached_property
    def dataset_name(self) -> str:
        global_attributes_dict = DataIO.s_load_data(
            folder_path=self._save_folder_path,
            file_name=self.__save_file_name,
        )

        return global_attributes_dict[
            self._key_dataset_name
        ]

    @cached_property
    def dataset_config(self) -> dict[str, Any]:
        global_attributes_dict = DataIO.s_load_data(
            folder_path=self._save_folder_path,
            file_name=self.__save_file_name,
        )

        return global_attributes_dict[
            self._key_dataset_config
        ]

    @cached_property
    def dataset_sha256_hash(self) -> str:
        global_attributes_dict = DataIO.s_load_data(
            folder_path=self._save_folder_path,
            file_name=self.__save_file_name,
        )

        return global_attributes_dict[
            self._key_dataset_sha256_hash
        ]

    @cached_property
    def mapper_user_original_id_to_index(self) -> dict[Any, int]:
        global_attributes_dict = DataIO.s_load_data(
            folder_path=self._save_folder_path,
            file_name=self.__save_file_name,
        )

        return global_attributes_dict[
            self._key_mapper_user_original_id_to_index
        ]

    @cached_property
    def mapper_item_original_id_to_index(self) -> dict[Any, int]:
        global_attributes_dict = DataIO.s_load_data(
            folder_path=self._save_folder_path,
            file_name=self.__save_file_name,
        )

        return global_attributes_dict[
            self._key_mapper_item_original_id_to_index
        ]

    def verify_data_consistency(self) -> None:
        pass

    def print_statistics(self) -> None:
        pass

    def _assert_is_initialized(self) -> None:
        pass

    def get_dataset_name(self):
        return self.dataset_name

    def get_mapper_item_original_id_to_index(self):
        return self.mapper_item_original_id_to_index

    def get_mapper_user_original_id_to_index(self):
        return self.mapper_user_original_id_to_index

    def save_data(self, save_folder_path) -> None:
        logger.debug(
            f"{self.__class__.__name__}|{self.save_data.__name__}|{self.__save_file_name=}"
        )

        self._save_folder_path = save_folder_path

        DataIO.s_save_data(
            folder_path=self._save_folder_path,
            file_name=self.__save_file_name,
            data_dict_to_save={
                self._key_dataset_name: self._dataset_name,
                self._key_dataset_config: self._dataset_config,
                self._key_dataset_sha256_hash: self._dataset_sha256_hash,
                self._key_mapper_user_original_id_to_index: self._mapper_user_original_id_to_index,
                self._key_mapper_item_original_id_to_index: self._mapper_item_original_id_to_index,
            },
        )

    def load_data(self, save_folder_path) -> None:
        logger.debug(
            f"{self.__class__.__name__}|{self.load_data.__name__}|{self.__save_file_name=}"
        )

        self._save_folder_path = save_folder_path

        logger.debug(
            f"Checking that dataset exists, if not, raise exception."
        )

        data_exists = DataIO.s_file_exists(
            folder_path=self._save_folder_path,
            file_name=self.__save_file_name,
        )

        if not data_exists:
            raise FileNotFoundError(
                f"No dataset exists on {self._save_folder_path} with the name of {self.__save_file_name}."
            )


class CSRMatrixStatisticsMixin:
    dataset_name: str
    statistics_matrix: sp.csr_matrix
    statistics_matrix_name: str

    def print_statistics_matrix(
        self,
    ) -> None:
        n_interactions = self.statistics_matrix.nnz
        n_users, n_items = self.statistics_matrix.shape

        uim_all = sp.csr_matrix(self.statistics_matrix)
        user_profile_length = np.ediff1d(uim_all.indptr)

        max_interactions_per_user = user_profile_length.max(initial=np.NINF)
        mean_interactions_per_user = user_profile_length.mean()
        std_interactions_per_user = user_profile_length.std()
        min_interactions_per_user = user_profile_length.min(initial=np.PINF)

        uim_all = sp.csc_matrix(uim_all)
        item_profile_length = np.ediff1d(uim_all.indptr)

        max_interactions_per_item = item_profile_length.max(initial=np.NINF)
        mean_interactions_per_item = item_profile_length.mean()
        std_interactions_per_item = item_profile_length.std()
        min_interactions_per_item = item_profile_length.min(initial=np.PINF)

        logger.info(
            f"DataReader: current dataset is: {self.dataset_name} - {self.statistics_matrix_name}\n"
            f"\tNumber of items: {n_items}\n"
            f"\tNumber of users: {n_users}\n"
            f"\tNumber of interactions: {n_interactions}\n"
            f"\tValue range: {np.min(uim_all.data):.2f}-{np.max(uim_all.data):.2f}\n"
            f"\tInteraction density: {compute_density(uim_all):.2E}\n"
            f"\tInteractions per user:\n"
            f"\t\t Min: {min_interactions_per_user:.2E}\n"
            f"\t\t Mean \u00B1 std: {mean_interactions_per_user:.2E} \u00B1 {std_interactions_per_user:.2E} \n"
            f"\t\t Max: {max_interactions_per_user:.2E}\n"
            f"\tInteractions per item:\n"
            f"\t\t Min: {min_interactions_per_item:.2E}\n"
            f"\t\t Mean \u00B1 std: {mean_interactions_per_item:.2E} \u00B1 {std_interactions_per_item:.2E} \n"
            f"\t\t Max: {max_interactions_per_item:.2E}\n"
            f"\tGini Index: {gini_index(user_profile_length):.2f}\n"
        )


class InteractionsMixin(CSRMatrixStatisticsMixin, BaseDataMixin):
    NAME_URM_ALL = "URM_all"

    NAME_URM_TIMESTAMP_TRAIN = "URM_timestamp_train"
    NAME_URM_TIMESTAMP_VALIDATION = "URM_timestamp_validation"
    NAME_URM_TIMESTAMP_TEST = "URM_timestamp_test"

    NAME_URM_LEAVE_LAST_K_OUT_TRAIN = "URM_leave_last_k_out_train"
    NAME_URM_LEAVE_LAST_K_OUT_VALIDATION = "URM_leave_last_k_out_validation"
    NAME_URM_LEAVE_LAST_K_OUT_TEST = "URM_leave_last_k_out_test"

    is_interactions_implicit: bool
    interactions: dict[str, sp.csr_matrix]

    def verify_data_consistency(self) -> None:
        super().verify_data_consistency()

        print_preamble = f"{self.dataset_name} consistency check:"

        if len(self.interactions.values()) == 0:
            raise ValueError(
                f"{print_preamble} No interactions exist"
            )

        urm_all = self.get_URM_all()
        num_users, num_items = urm_all.shape
        num_interactions = urm_all.nnz

        if num_interactions <= 0:
            raise ValueError(
                f"{print_preamble} Number of interactions in URM is 0."
            )

        if self.is_interactions_implicit and np.any(urm_all.data != 1.0):
            raise ValueError(
                f"{print_preamble} The DataReader is stated to be implicit but the main URM is not"
            )

        if urm_all.shape <= (0, 0):
            raise ValueError(
                f"{print_preamble} No users or items in the URM_all matrix. Shape is {urm_all.shape}"
            )

        for URM_name, URM_object in self.interactions.items():
            if urm_all.shape != URM_object.shape:
                raise ValueError(
                    f"{print_preamble} Number of users or items are different between URM_all and {URM_name}. Shapes "
                    f"are {urm_all.shape} and {URM_object.shape}, respectively."
                )

        # Check if item index-id and user index-id are consistent
        if len(set(self.mapper_user_original_id_to_index.values())) != len(self.mapper_user_original_id_to_index):
            raise ValueError(
                f"{print_preamble} user it-to-index mapper values do not have a 1-to-1 correspondence with the key"
            )

        if len(set(self.mapper_item_original_id_to_index.values())) != len(self.mapper_item_original_id_to_index):
            raise ValueError(
                f"{print_preamble} item it-to-index mapper values do not have a 1-to-1 correspondence with the key"
            )

        if num_users != len(self.mapper_user_original_id_to_index):
            raise ValueError(
                f"{print_preamble} user ID-to-index mapper contains a number of keys different then the number of users"
            )

        if num_items != len(self.mapper_item_original_id_to_index):
            raise ValueError(
                f"{print_preamble} ({num_items=}/{len(self.mapper_item_original_id_to_index)=}"
                f"mapper contains a number of keys different then the number of items"
            )

        if num_users < max(self.mapper_user_original_id_to_index.values()):
            raise ValueError(
                f"{print_preamble} user ID-to-index mapper contains indices greater than number of users."
            )

        if num_items < max(self.mapper_item_original_id_to_index.values()):
            raise ValueError(
                f"{print_preamble} item ID-to-index mapper contains indices greater than number of item."
            )

        # Check if every non-empty user and item has a mapper value
        URM_all = sp.csc_matrix(urm_all)
        nonzero_items_mask = np.ediff1d(URM_all.indptr) > 0
        nonzero_items = np.arange(0, num_items, dtype=np.int32)[nonzero_items_mask]

        if not np.isin(
            nonzero_items,
            np.array(list(self.mapper_item_original_id_to_index.values()))
        ).all():
            raise ValueError(
                f"{print_preamble} there exist items with interactions that do not have a mapper entry"
            )

        URM_all = sp.csr_matrix(urm_all)
        nonzero_users_mask = np.ediff1d(URM_all.indptr) > 0
        nonzero_users = np.arange(0, num_users, dtype=np.int32)[nonzero_users_mask]
        if not np.isin(
            nonzero_users,
            np.array(list(self.mapper_user_original_id_to_index.values()))
        ).all():
            raise ValueError(
                f"{print_preamble} there exist users with interactions that do not have a mapper entry"
            )

    def print_statistics(self) -> None:
        super().print_statistics()

        for matrix_name, matrix in self.interactions.items():
            self.statistics_matrix = matrix
            self.statistics_matrix_name = matrix_name
            self.print_statistics_matrix()

    def _assert_is_initialized(self):
        super()._assert_is_initialized()

        if self.interactions is None:
            raise ValueError(
                f"DataReader {self.dataset_name}: Unable to load data split. The split has not been generated"
                f" yet, call the load_data function to do so."
            )

    def get_URM_all(self) -> sp.csr_matrix:
        return self.interactions[self.NAME_URM_ALL].copy()

    def get_loaded_URM_names(self):
        return list(self.interactions.keys())

    def get_loaded_URM_items(self):
        return self.interactions.items()

    def get_urm_by_name(self, name: str) -> sp.csr_matrix:
        return self.interactions[name].copy()

    def get_urm_splits(self, evaluation_strategy: EvaluationStrategy):
        if evaluation_strategy == EvaluationStrategy.LEAVE_LAST_K_OUT:
            return self._get_urm_leave_last_k_out_splits()
        elif evaluation_strategy == EvaluationStrategy.TIMESTAMP:
            return self._get_urm_timestamp_splits()
        else:
            raise ValueError(
                f"Requested split ({evaluation_strategy}) does not exist."
            )

    def _get_urm_leave_last_k_out_splits(self) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
        return (
            self.interactions[self.NAME_URM_LEAVE_LAST_K_OUT_TRAIN],
            self.interactions[self.NAME_URM_LEAVE_LAST_K_OUT_VALIDATION],
            self.interactions[self.NAME_URM_LEAVE_LAST_K_OUT_TEST],
        )

    def _get_urm_timestamp_splits(self) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
        return (
            self.interactions[self.NAME_URM_TIMESTAMP_TRAIN],
            self.interactions[self.NAME_URM_TIMESTAMP_VALIDATION],
            self.interactions[self.NAME_URM_TIMESTAMP_TEST],
        )

    def save_data(self, save_folder_path):
        super().save_data(
            save_folder_path=save_folder_path,
        )

        data_io = DataIO(folder_path=save_folder_path)
        data_io.save_data(
            data_dict_to_save={
                "interactions": self.interactions,
                "is_interactions_implicit": self.is_interactions_implicit,
            },
            file_name="dataset_URM"
        )

    def load_data(self, save_folder_path):
        super().load_data(
            save_folder_path=save_folder_path,
        )

        data_io = DataIO(folder_path=save_folder_path)
        impressions_attributes_dict = data_io.load_data(
            file_name="dataset_URM"
        )

        for attrib_name, attrib_object in impressions_attributes_dict.items():
            self.__setattr__(attrib_name, attrib_object)


class LazyInteractionsMixin(CSRMatrixStatisticsMixin, LazyBaseDataMixin):
    NAME_URM_ALL = "URM_all"

    NAME_URM_TIMESTAMP_TRAIN = "URM_timestamp_train"
    NAME_URM_TIMESTAMP_VALIDATION = "URM_timestamp_validation"
    NAME_URM_TIMESTAMP_TRAIN_VALIDATION = "URM_timestamp_train_validation"
    NAME_URM_TIMESTAMP_TEST = "URM_timestamp_test"

    NAME_URM_LEAVE_LAST_K_OUT_TRAIN = "URM_leave_last_k_out_train"
    NAME_URM_LEAVE_LAST_K_OUT_VALIDATION = "URM_leave_last_k_out_validation"
    NAME_URM_LEAVE_LAST_K_OUT_TRAIN_VALIDATION = "URM_leave_last_k_out_train_validation"
    NAME_URM_LEAVE_LAST_K_OUT_TEST = "URM_leave_last_k_out_test"

    _key_interactions: str = "interactions"
    _key_is_interactions_implicit: str = "is_interactions_implicit"
    __save_file_name: str = "dataset_URM"

    @cached_property
    def interactions(self) -> dict[str, sp.csr_matrix]:
        data_dict = DataIO.s_load_data(
            folder_path=self._save_folder_path,
            file_name=self.__save_file_name,
        )

        return data_dict[
            self._key_interactions
        ]

    @cached_property
    def is_interactions_implicit(self) -> bool:
        data_dict = DataIO.s_load_data(
            folder_path=self._save_folder_path,
            file_name=self.__save_file_name,
        )

        return data_dict[
            self._key_is_interactions_implicit
        ]

    def save_data(self, save_folder_path):
        logger.debug(
            f"{self.__class__.__name__}|{self.save_data.__name__}|{self.__save_file_name=}"
        )

        super().save_data(
            save_folder_path=save_folder_path,
        )
        self._save_folder_path = save_folder_path

        DataIO.s_save_data(
            file_name=self.__save_file_name,
            folder_path=save_folder_path,
            data_dict_to_save={
                self._key_interactions: self.interactions,
                self._key_is_interactions_implicit: self.is_interactions_implicit,
            },
        )

    def get_urm_splits(self, evaluation_strategy: EvaluationStrategy) -> InteractionsDataSplits:
        if evaluation_strategy == EvaluationStrategy.LEAVE_LAST_K_OUT:
            return self._get_urm_leave_last_k_out_splits()
        elif evaluation_strategy == EvaluationStrategy.TIMESTAMP:
            return self._get_urm_timestamp_splits()
        else:
            raise ValueError(
                f"Requested split ({evaluation_strategy}) does not exist."
            )

    def _get_urm_leave_last_k_out_splits(self) -> InteractionsDataSplits:
        return InteractionsDataSplits(
            sp_urm_train=self.interactions[self.NAME_URM_LEAVE_LAST_K_OUT_TRAIN],
            sp_urm_validation=self.interactions[self.NAME_URM_LEAVE_LAST_K_OUT_VALIDATION],
            sp_urm_train_validation=self.interactions[self.NAME_URM_LEAVE_LAST_K_OUT_TRAIN_VALIDATION],
            sp_urm_test=self.interactions[self.NAME_URM_LEAVE_LAST_K_OUT_TEST],
        )

    def _get_urm_timestamp_splits(self) -> InteractionsDataSplits:
        return InteractionsDataSplits(
            sp_urm_train=self.interactions[self.NAME_URM_TIMESTAMP_TRAIN],
            sp_urm_validation=self.interactions[self.NAME_URM_TIMESTAMP_VALIDATION],
            sp_urm_train_validation=self.interactions[self.NAME_URM_TIMESTAMP_TRAIN_VALIDATION],
            sp_urm_test=self.interactions[self.NAME_URM_TIMESTAMP_TEST],
        )


class ImpressionsMixin(CSRMatrixStatisticsMixin, BaseDataMixin):
    NAME_UIM_ALL = "UIM_all"

    NAME_UIM_TIMESTAMP_TRAIN = "UIM_timestamp_train"
    NAME_UIM_TIMESTAMP_VALIDATION = "UIM_timestamp_validation"
    NAME_UIM_TIMESTAMP_TEST = "UIM_timestamp_test"

    NAME_UIM_LEAVE_LAST_K_OUT_TRAIN = "UIM_leave_last_k_out_train"
    NAME_UIM_LEAVE_LAST_K_OUT_VALIDATION = "UIM_leave_last_k_out_validation"
    NAME_UIM_LEAVE_LAST_K_OUT_TEST = "UIM_leave_last_k_out_test"

    is_impressions_implicit: bool
    impressions: dict[str, sp.csr_matrix]

    def print_statistics(self) -> None:
        super().print_statistics()

        for matrix_name, matrix in self.impressions.items():
            self.statistics_matrix = matrix
            self.statistics_matrix_name = matrix_name
            self.print_statistics_matrix()

    def _assert_is_initialized(self):
        super()._assert_is_initialized()

        if self.impressions is None:
            raise ValueError(
                f"DataReader {self.dataset_name}: Unable to load data split. The split has not been generated"
                f" yet, call the load_data function to do so."
            )

    def get_uim_by_name(self, name: str) -> sp.csr_matrix:
        return self.impressions[name].copy()

    def get_uim_all(self) -> sp.csr_matrix:
        return self.impressions[self.NAME_UIM_ALL].copy()

    def get_uim_splits(self, evaluation_strategy: EvaluationStrategy):
        if evaluation_strategy == EvaluationStrategy.LEAVE_LAST_K_OUT:
            return self._get_uim_leave_last_k_out_splits()
        elif evaluation_strategy == EvaluationStrategy.TIMESTAMP:
            return self._get_uim_timestamp_splits()
        else:
            raise ValueError(
                f"Requested split ({evaluation_strategy}) does not exist."
            )

    def _get_uim_leave_last_k_out_splits(self) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
        return (
            self.impressions[self.NAME_UIM_LEAVE_LAST_K_OUT_TRAIN],
            self.impressions[self.NAME_UIM_LEAVE_LAST_K_OUT_VALIDATION],
            self.impressions[self.NAME_UIM_LEAVE_LAST_K_OUT_TEST],
        )

    def _get_uim_timestamp_splits(self) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
        return (
            self.impressions[self.NAME_UIM_TIMESTAMP_TRAIN],
            self.impressions[self.NAME_UIM_TIMESTAMP_VALIDATION],
            self.impressions[self.NAME_UIM_TIMESTAMP_TEST],
        )

    def get_loaded_UIM_names(self):
        return list(self.impressions.keys())

    def get_loaded_UIM_items(self):
        return self.impressions.items()

    def save_data(self, save_folder_path):
        super().save_data(
            save_folder_path=save_folder_path,
        )

        data_io = DataIO(folder_path=save_folder_path)
        data_io.save_data(
            data_dict_to_save={
                "impressions": self.impressions,
                "is_impressions_implicit": self.is_impressions_implicit,
            },
            file_name="dataset_UIM"
        )

    def load_data(self, save_folder_path):
        super().load_data(
            save_folder_path=save_folder_path,
        )

        data_io = DataIO(folder_path=save_folder_path)
        impressions_attributes_dict = data_io.load_data(
            file_name="dataset_UIM"
        )

        for attrib_name, attrib_object in impressions_attributes_dict.items():
            self.__setattr__(attrib_name, attrib_object)


class LazyImpressionsMixin(CSRMatrixStatisticsMixin, LazyBaseDataMixin):
    NAME_UIM_ALL = "UIM_all"

    NAME_UIM_TIMESTAMP_TRAIN = "UIM_timestamp_train"
    NAME_UIM_TIMESTAMP_VALIDATION = "UIM_timestamp_validation"
    NAME_UIM_TIMESTAMP_TRAIN_VALIDATION = "UIM_timestamp_train_validation"
    NAME_UIM_TIMESTAMP_TEST = "UIM_timestamp_test"

    NAME_UIM_LEAVE_LAST_K_OUT_TRAIN = "UIM_leave_last_k_out_train"
    NAME_UIM_LEAVE_LAST_K_OUT_VALIDATION = "UIM_leave_last_k_out_validation"
    NAME_UIM_LEAVE_LAST_K_OUT_TRAIN_VALIDATION = "UIM_leave_last_k_out_train_validation"
    NAME_UIM_LEAVE_LAST_K_OUT_TEST = "UIM_leave_last_k_out_test"

    _key_impressions: str = "impressions"
    _key_is_impressions_implicit: str = "is_impressions_implicit"
    __save_file_name: str = "dataset_UIM"

    @cached_property
    def impressions(self) -> dict[str, sp.csr_matrix]:
        data_dict = DataIO.s_load_data(
            folder_path=self._save_folder_path,
            file_name=self.__save_file_name,
        )

        return data_dict[
            self._key_impressions
        ]

    @cached_property
    def is_impressions_implicit(self) -> bool:
        data_dict = DataIO.s_load_data(
            folder_path=self._save_folder_path,
            file_name=self.__save_file_name,
        )

        return data_dict[
            self._key_is_impressions_implicit
        ]

    def save_data(self, save_folder_path):
        logger.debug(
            f"{self.__class__.__name__}|{self.save_data.__name__}|{self.__save_file_name=}"
        )

        super().save_data(
            save_folder_path=save_folder_path,
        )
        self._save_folder_path = save_folder_path

        DataIO.s_save_data(
            file_name=self.__save_file_name,
            folder_path=save_folder_path,
            data_dict_to_save={
                self._key_impressions: self.impressions,
                self._key_is_impressions_implicit: self.is_impressions_implicit,
            },
        )

    def get_uim_splits(self, evaluation_strategy: EvaluationStrategy) -> ImpressionsDataSplits:
        if evaluation_strategy == EvaluationStrategy.LEAVE_LAST_K_OUT:
            return self._get_uim_leave_last_k_out_splits()
        elif evaluation_strategy == EvaluationStrategy.TIMESTAMP:
            return self._get_uim_timestamp_splits()
        else:
            raise ValueError(
                f"Requested split ({evaluation_strategy}) does not exist."
            )

    def _get_uim_leave_last_k_out_splits(self) -> ImpressionsDataSplits:
        return ImpressionsDataSplits(
            sp_uim_train=self.impressions[self.NAME_UIM_LEAVE_LAST_K_OUT_TRAIN],
            sp_uim_validation=self.impressions[self.NAME_UIM_LEAVE_LAST_K_OUT_VALIDATION],
            sp_uim_train_validation=self.impressions[self.NAME_UIM_LEAVE_LAST_K_OUT_TRAIN_VALIDATION],
            sp_uim_test=self.impressions[self.NAME_UIM_LEAVE_LAST_K_OUT_TEST],
        )

    def _get_uim_timestamp_splits(self) -> ImpressionsDataSplits:
        return ImpressionsDataSplits(
            sp_uim_train=self.impressions[self.NAME_UIM_TIMESTAMP_TRAIN],
            sp_uim_validation=self.impressions[self.NAME_UIM_TIMESTAMP_VALIDATION],
            sp_uim_train_validation=self.impressions[self.NAME_UIM_TIMESTAMP_TRAIN_VALIDATION],
            sp_uim_test=self.impressions[self.NAME_UIM_TIMESTAMP_TEST],
        )


class PandasDataFramesMixin(BaseDataMixin):
    NAME_DF_FILTERED = "DF_filtered"

    NAME_DF_TIMESTAMP_TRAIN = "DF_timestamp_train"
    NAME_DF_TIMESTAMP_VALIDATION = "DF_timestamp_validation"
    NAME_DF_TIMESTAMP_TEST = "DF_timestamp_test"

    NAME_DF_LEAVE_LAST_K_OUT_TRAIN = "DF_leave_last_k_out_train"
    NAME_DF_LEAVE_LAST_K_OUT_VALIDATION = "DF_leave_last_k_out_validation"
    NAME_DF_LEAVE_LAST_K_OUT_TEST = "DF_leave_last_k_out_test"

    dataframes: dict[str, pd.DataFrame]

    def print_statistics(self) -> None:
        super().print_statistics()

        for df_name, df in self.dataframes.items():
            df_describe = df.describe(
                exclude=[object],
                percentiles=[0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99],
                datetime_is_numeric=True,
            )
            logger.info(
                f"DataReader: current dataframe is: {self.dataset_name} - {df_name}\n"
                f"\n\t* {df_describe}"

            )

    def _assert_is_initialized(self):
        super()._assert_is_initialized()

        if self.dataframes is None:
            raise ValueError(
                f"DataReader {self.dataset_name}: Unable to load data split. The split has not been generated"
                f" yet, call the load_data function to do so."
            )

    def get_df_by_name(self, name: str) -> pd.DataFrame:
        return self.dataframes[name].copy()

    def get_df_all(self) -> pd.DataFrame:
        return self.dataframes[self.NAME_DF_FILTERED].copy()

    def get_df_splits(self, evaluation_strategy: EvaluationStrategy):
        if evaluation_strategy == EvaluationStrategy.LEAVE_LAST_K_OUT:
            return self._get_df_leave_last_k_out_splits()
        elif evaluation_strategy == EvaluationStrategy.TIMESTAMP:
            return self._get_df_timestamp_splits()
        else:
            raise ValueError(
                f"Requested split ({evaluation_strategy}) does not exist."
            )

    def _get_df_leave_last_k_out_splits(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (
            self.dataframes[self.NAME_DF_LEAVE_LAST_K_OUT_TRAIN],
            self.dataframes[self.NAME_DF_LEAVE_LAST_K_OUT_VALIDATION],
            self.dataframes[self.NAME_DF_LEAVE_LAST_K_OUT_TEST],
        )

    def _get_df_timestamp_splits(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (
            self.dataframes[self.NAME_DF_TIMESTAMP_TRAIN],
            self.dataframes[self.NAME_DF_TIMESTAMP_VALIDATION],
            self.dataframes[self.NAME_DF_TIMESTAMP_TEST],
        )

    def get_loaded_df_names(self):
        return list(self.dataframes.keys())

    def get_loaded_df_items(self):
        return self.dataframes.items()

    def save_data(self, save_folder_path):
        super().save_data(
            save_folder_path=save_folder_path,
        )

        data_io = DataIO(folder_path=save_folder_path)
        data_io.save_data(
            data_dict_to_save={
                "dataframes": self.dataframes,
            },
            file_name="dataset_DFs"
        )

    def load_data(self, save_folder_path):
        super().load_data(
            save_folder_path=save_folder_path,
        )

        data_io = DataIO(folder_path=save_folder_path)
        dataframes_attributes_dict = data_io.load_data(
            file_name="dataset_DFs"
        )

        for attrib_name, attrib_object in dataframes_attributes_dict.items():
            self.__setattr__(attrib_name, attrib_object)


class LazyPandasDataFramesMixin(LazyBaseDataMixin):
    NAME_DF_FILTERED = "DF_filtered"

    NAME_DF_TIMESTAMP_TRAIN = "DF_timestamp_train"
    NAME_DF_TIMESTAMP_VALIDATION = "DF_timestamp_validation"
    NAME_DF_TIMESTAMP_TRAIN_VALIDATION = "DF_timestamp_train_validation"
    NAME_DF_TIMESTAMP_TEST = "DF_timestamp_test"

    NAME_DF_LEAVE_LAST_K_OUT_TRAIN = "DF_leave_last_k_out_train"
    NAME_DF_LEAVE_LAST_K_OUT_VALIDATION = "DF_leave_last_k_out_validation"
    NAME_DF_LEAVE_LAST_K_OUT_TRAIN_VALIDATION = "DF_leave_last_k_out_train_validation"
    NAME_DF_LEAVE_LAST_K_OUT_TEST = "DF_leave_last_k_out_test"

    _key_dataframes: str = "dataframes"
    __save_file_name: str = "dataset_DFs"

    @cached_property
    def dataframes(self) -> dict[str, pd.DataFrame]:
        data_dict = DataIO.s_load_data(
            folder_path=self._save_folder_path,
            file_name=self.__save_file_name,
        )

        return data_dict[
            self._key_dataframes
        ]

    def save_data(self, save_folder_path):
        logger.debug(
            f"{self.__class__.__name__}|{self.save_data.__name__}|{self.__save_file_name=}"
        )

        super().save_data(
            save_folder_path=save_folder_path,
        )

        DataIO.s_save_data(
            folder_path=save_folder_path,
            file_name=self.__save_file_name,
            data_dict_to_save={
                self._key_dataframes: self.dataframes,
            },
        )


class LazyImpressionsFeaturesMixin(LazyBaseDataMixin):
    """
    The main structure of this mixin are two dictionaries: `impressions_features_dataframes` and
    `impressions_features_sparse_matrices`, both hold the same data but are contained in different structures. Both
    dictionaries are lazy, i.e., if the feature requested is not already in the dictionary, then it searches for the
    feature on disk.

    Laziness avoids to load probably expensive data structures on memory if they're not used by the agent using the
    dataset, e.g., the impressions features exist but the recommender being trained does not use impressions features.
    """
    _impressions_features_dataframes: dict[str, pd.DataFrame] = dict()
    _impressions_features_sparse_matrices: dict[str, sp.csr_matrix] = dict()

    _folder_name_impressions_features = "dataset_impressions_features"
    __save_file_name: str = "dataset_impressions_features"

    def dataframe_available_features(self) -> list[str]:
        available_features: dict[str, list[str]] = DataIO.s_load_data(
            folder_path=os.path.join(
                self._save_folder_path, self._folder_name_impressions_features, "",
            ),
            file_name=f"available_features",
        )

        return available_features["dataframes"]

    def sparse_matrices_available_features(self) -> list[str]:
        available_features: dict[str, list[str]] = DataIO.s_load_data(
            folder_path=os.path.join(
                self._save_folder_path, self._folder_name_impressions_features, "",
            ),
            file_name=f"available_features",
        )

        return available_features["sparse_matrices"]

    def dataframe_impression_feature(self, feature: str) -> pd.DataFrame:
        data_dict: dict[str, pd.DataFrame] = DataIO.s_load_data(
            folder_path=os.path.join(
                self._save_folder_path, self._folder_name_impressions_features, "",
            ),
            file_name=f"dataframe_{feature}",
        )

        return data_dict[feature]

    def sparse_matrix_impression_feature(self, feature: str) -> sp.csr_matrix:
        data_dict: dict[str, sp.csr_matrix] = DataIO.s_load_data(
            folder_path=os.path.join(
                self._save_folder_path, self._folder_name_impressions_features, "",
            ),
            file_name=f"sparse_matrix_{feature}",
        )

        return data_dict[feature]

    def save_data(self, save_folder_path: str):
        logger.debug(
            f"{self.__class__.__name__}|{self.save_data.__name__}|{self.__save_file_name=}"
        )

        super().save_data(
            save_folder_path=save_folder_path,
        )

        save_folder_path = os.path.join(
            save_folder_path, self._folder_name_impressions_features, "",
        )

        # First, save the available features
        DataIO.s_save_data(
            folder_path=save_folder_path,
            file_name="available_features",
            data_dict_to_save={
                "dataframes": list(self._impressions_features_dataframes.keys()),
                "sparse_matrices": list(self._impressions_features_sparse_matrices.keys()),
            }
        )

        # Second, save each feature dataframe independently.
        for key_feature, df_feature in self._impressions_features_dataframes.items():
            DataIO.s_save_data(
                folder_path=save_folder_path,
                file_name=f"dataframe_{key_feature}",
                data_dict_to_save={
                    key_feature: df_feature,
                }
            )

        # Third, save each feature sparse matrix independently.
        for key_feature, sp_feature in self._impressions_features_sparse_matrices.items():
            DataIO.s_save_data(
                folder_path=save_folder_path,
                file_name=f"sparse_matrix_{key_feature}",
                data_dict_to_save={
                    key_feature: sp_feature,
                }
            )


class DatasetStatisticsMixin:
    statistics: dict[str, Any]
    statistics_folder: str
    statistics_file_name: str

    def compute_statistics_df_on_csr_matrix(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        user_column: str,
        item_column: str
    ) -> None:
        user_codes, user_uniques = pd.factorize(
            df[user_column]
        )

        item_codes, item_uniques = pd.factorize(
            df[item_column]
        )

        num_users = user_uniques.shape[0]
        num_items = item_uniques.shape[0]

        row_indices = user_codes
        col_indices = item_codes
        data = np.ones_like(
            a=row_indices,
            dtype=np.int32,
        )

        assert df.shape[0] == row_indices.shape[0]
        assert row_indices.shape == col_indices.shape
        assert row_indices.shape == data.shape

        matrix: sp.csr_matrix = sp.csr_matrix(
            (
                data,
                (row_indices, col_indices)
            ),
            shape=(num_users, num_items),
            dtype=np.int32,
        )

        self.compute_statistics_csr_matrix(matrix=matrix, dataset_name=dataset_name)

    def compute_statistics_csr_matrix(
        self,
        matrix: sp.csr_matrix,
        dataset_name: str,
    ) -> None:
        category_name = "sparse_matrix"

        matrix_csc = matrix.tocsc(copy=True)

        if dataset_name not in self.statistics:
            self.statistics[dataset_name] = dict()

        self.statistics[dataset_name][category_name] = dict()
        self.statistics[dataset_name][category_name]["csr_matrix"] = matrix
        self.statistics[dataset_name][category_name]["csc_matrix"] = matrix_csc

        user_profile_length: np.ndarray = np.ediff1d(matrix.indptr)
        user_profile_stats: DescribeResult = st.describe(
            a=user_profile_length,
            axis=0,
            nan_policy="raise",
        )
        self.statistics[dataset_name][category_name]["interactions_by_users"] = {
            "num_observations": user_profile_stats.nobs,
            "min": user_profile_stats.minmax[0],
            "max": user_profile_stats.minmax[1],
            "mean": user_profile_stats.mean,
            "variance": user_profile_stats.variance,
            "skewness": user_profile_stats.skewness,
            "kurtosis": user_profile_stats.kurtosis,
            "gini_index": gini_index(
                array=user_profile_length,
            ),
        }

        item_profile_length: np.ndarray = np.ediff1d(matrix_csc.indptr)
        item_profile_stats: DescribeResult = st.describe(
            a=item_profile_length,
            axis=0,
            nan_policy="raise",
        )
        self.statistics[dataset_name][category_name]["interactions_by_items"] = {
            "num_observations": item_profile_stats.nobs,
            "min": item_profile_stats.minmax[0],
            "max": item_profile_stats.minmax[1],
            "mean": item_profile_stats.mean,
            "variance": item_profile_stats.variance,
            "skewness": item_profile_stats.skewness,
            "kurtosis": item_profile_stats.kurtosis,
            "gini_index": gini_index(
                array=item_profile_length
            ),
        }

    def compare_two_dataframes(
        self,
        df: pd.DataFrame,
        df_name: str,
        df_other: pd.DataFrame,
        df_other_name: str,
        columns_to_compare: list[str],
    ) -> None:
        dataset_name = f"{df_name}-{df_other_name}"
        if dataset_name not in self.statistics:
            self.statistics[dataset_name] = dict()

        if df_name not in self.statistics[dataset_name]:
            self.statistics[dataset_name][df_name] = dict()

        if df_other_name not in self.statistics[dataset_name]:
            self.statistics[dataset_name][df_other_name] = dict()

        df_num_records = df.shape[0]
        df_other_num_records = df_other.shape[0]

        relative_difference_num_records = df_num_records - df_other_num_records
        relative_percentage_num_records = (df_num_records - df_other_num_records) * 100 / df_num_records

        self.statistics[dataset_name][df_name]["num_records"] = df_num_records
        self.statistics[dataset_name][df_other_name]["num_records"] = df_other_num_records

        self.statistics[dataset_name]["relative_difference_num_records"] = relative_difference_num_records
        self.statistics[dataset_name]["relative_percentage_num_records"] = relative_percentage_num_records

        for column in columns_to_compare:
            df_num_uniques_column = df[column].nunique(dropna=True)
            df_other_num_uniques_column = df_other[column].nunique(dropna=True)

            relative_difference_num_uniques = df_num_uniques_column - df_other_num_uniques_column
            relative_percentage_num_uniques = (
                                                  df_num_uniques_column - df_other_num_uniques_column) * 100 / df_num_uniques_column

            self.statistics[dataset_name][df_name][f"num_uniques_{column}"] = df_num_records
            self.statistics[dataset_name][df_other_name][f"num_uniques_{column}"] = df_other_num_records

            self.statistics[dataset_name]["relative_difference_num_uniques"] = relative_difference_num_uniques
            self.statistics[dataset_name]["relative_percentage_num_uniques"] = relative_percentage_num_uniques

    def compare_two_sparse_matrices(
        self,
        csr_matrix: sp.csr_matrix,
        csr_matrix_name: str,
        other_csr_matrix: sp.csr_matrix,
        other_csr_matrix_name: str,
    ) -> None:

        dataset_name = f"{csr_matrix_name}-{other_csr_matrix_name}"
        if dataset_name not in self.statistics:
            self.statistics[dataset_name] = dict()

        if csr_matrix_name not in self.statistics[dataset_name]:
            self.statistics[dataset_name][csr_matrix_name] = dict()

        if other_csr_matrix_name not in self.statistics[dataset_name]:
            self.statistics[dataset_name][other_csr_matrix_name] = dict()

        csr_matrix_num_records = csr_matrix.nnz
        csr_matrix_other_num_records = other_csr_matrix.nnz

        relative_difference_num_records = csr_matrix_num_records - csr_matrix_other_num_records
        relative_percentage_num_records = (relative_difference_num_records) * 100 / csr_matrix_num_records

        self.statistics[dataset_name][csr_matrix_name]["num_records"] = csr_matrix_num_records
        self.statistics[dataset_name][other_csr_matrix_name]["num_records"] = csr_matrix_other_num_records

        self.statistics[dataset_name]["relative_difference_num_records"] = relative_difference_num_records
        self.statistics[dataset_name]["relative_percentage_num_records"] = relative_percentage_num_records

        csr_matrix_profile_lengths = [
            np.ediff1d(csr_matrix.indptr),
            np.ediff1d(csr_matrix.tocsc(copy=True).indptr)
        ]
        other_csr_matrix_profile_lengths = [
            np.ediff1d(other_csr_matrix.indptr),
            np.ediff1d(other_csr_matrix.tocsc(copy=True).indptr)
        ]
        profile_length_types = [
            "user",
            "items",
        ]

        for pl, other_pl, pl_type in zip(
            csr_matrix_profile_lengths,
            other_csr_matrix_profile_lengths,
            profile_length_types,
        ):
            pl_stats: DescribeResult = st.describe(
                a=pl,
                axis=0,
                nan_policy="raise",
            )
            other_pl_stats: DescribeResult = st.describe(
                a=other_pl,
                axis=0,
                nan_policy="raise",
            )

            self.statistics[dataset_name][csr_matrix_name][f"interactions_by_{pl_type}"] = {
                "num_observations": pl_stats.nobs,
                "min": pl_stats.minmax[0],
                "max": pl_stats.minmax[1],
                "mean": pl_stats.mean,
                "variance": pl_stats.variance,
                "std": np.sqrt(pl_stats.variance),
                "skewness": pl_stats.skewness,
                "kurtosis": pl_stats.kurtosis,
                "gini_index": gini_index(array=pl),
            }

            self.statistics[dataset_name][other_csr_matrix_name][f"interactions_by_{pl_type}"] = {
                "num_observations": other_pl_stats.nobs,
                "min": other_pl_stats.minmax[0],
                "max": other_pl_stats.minmax[1],
                "mean": other_pl_stats.mean,
                "variance": other_pl_stats.variance,
                "std": np.sqrt(other_pl_stats.variance),
                "skewness": other_pl_stats.skewness,
                "kurtosis": other_pl_stats.kurtosis,
                "gini_index": gini_index(array=other_pl),
            }

            self.statistics[dataset_name][f"relative_difference_interactions_by_{pl_type}"] = {
                "num_observations": (
                    self.statistics[dataset_name][csr_matrix_name][f"interactions_by_{pl_type}"]["num_observations"]
                    - self.statistics[dataset_name][other_csr_matrix_name][f"interactions_by_{pl_type}"][
                        "num_observations"]
                ),
                "min": (
                    self.statistics[dataset_name][csr_matrix_name][f"interactions_by_{pl_type}"]["min"]
                    - self.statistics[dataset_name][other_csr_matrix_name][f"interactions_by_{pl_type}"]["min"]
                ),
                "max": (
                    self.statistics[dataset_name][csr_matrix_name][f"interactions_by_{pl_type}"]["max"]
                    - self.statistics[dataset_name][other_csr_matrix_name][f"interactions_by_{pl_type}"]["max"]
                ),
                "mean": (
                    self.statistics[dataset_name][csr_matrix_name][f"interactions_by_{pl_type}"]["mean"]
                    - self.statistics[dataset_name][other_csr_matrix_name][f"interactions_by_{pl_type}"]["mean"]
                ),
                "variance": (
                    self.statistics[dataset_name][csr_matrix_name][f"interactions_by_{pl_type}"]["variance"]
                    - self.statistics[dataset_name][other_csr_matrix_name][f"interactions_by_{pl_type}"]["variance"]
                ),
                "std": (
                    self.statistics[dataset_name][csr_matrix_name][f"interactions_by_{pl_type}"]["std"]
                    - self.statistics[dataset_name][other_csr_matrix_name][f"interactions_by_{pl_type}"]["std"]
                ),
                "skewness": (
                    self.statistics[dataset_name][csr_matrix_name][f"interactions_by_{pl_type}"]["skewness"]
                    - self.statistics[dataset_name][other_csr_matrix_name][f"interactions_by_{pl_type}"]["skewness"]
                ),
                "kurtosis": (
                    self.statistics[dataset_name][csr_matrix_name][f"interactions_by_{pl_type}"]["kurtosis"]
                    - self.statistics[dataset_name][other_csr_matrix_name][f"interactions_by_{pl_type}"]["kurtosis"]
                ),
            }

            self.statistics[dataset_name][f"relative_percentage_interactions_by_{pl_type}"] = {
                "num_observations": (
                    self.statistics[dataset_name][f"relative_difference_interactions_by_{pl_type}"][
                        "num_observations"] * 100
                    / self.statistics[dataset_name][csr_matrix_name][f"interactions_by_{pl_type}"]["num_observations"]
                ),
                "min": (
                    self.statistics[dataset_name][f"relative_difference_interactions_by_{pl_type}"]["min"] * 100
                    / self.statistics[dataset_name][csr_matrix_name][f"interactions_by_{pl_type}"]["min"]
                ),
                "max": (
                    self.statistics[dataset_name][f"relative_difference_interactions_by_{pl_type}"]["max"] * 100
                    / self.statistics[dataset_name][csr_matrix_name][f"interactions_by_{pl_type}"]["max"]
                ),
                "mean": (
                    self.statistics[dataset_name][f"relative_difference_interactions_by_{pl_type}"]["mean"] * 100
                    / self.statistics[dataset_name][csr_matrix_name][f"interactions_by_{pl_type}"]["mean"]
                ),
                "variance": (
                    self.statistics[dataset_name][f"relative_difference_interactions_by_{pl_type}"]["variance"] * 100
                    / self.statistics[dataset_name][csr_matrix_name][f"interactions_by_{pl_type}"]["variance"]
                ),
                "std": (
                    self.statistics[dataset_name][f"relative_difference_interactions_by_{pl_type}"]["std"] * 100
                    / self.statistics[dataset_name][csr_matrix_name][f"interactions_by_{pl_type}"]["std"]
                ),
                "skewness": (
                    self.statistics[dataset_name][f"relative_difference_interactions_by_{pl_type}"]["skewness"] * 100
                    / self.statistics[dataset_name][csr_matrix_name][f"interactions_by_{pl_type}"]["skewness"]
                ),
                "kurtosis": (
                    self.statistics[dataset_name][f"relative_difference_interactions_by_{pl_type}"]["kurtosis"] * 100
                    / self.statistics[dataset_name][csr_matrix_name][f"interactions_by_{pl_type}"]["kurtosis"]
                ),
            }

        # csr_matrix_user_profile_length: np.ndarray = np.ediff1d(csr_matrix.indptr)
        # csr_matrix_user_profile_stats: DescribeResult = st.describe(
        #     a=csr_matrix_user_profile_length,
        #     axis=0,
        #     nan_policy="raise",
        # )
        #
        # other_csr_matrix_user_profile_length: np.ndarray = np.ediff1d(other_csr_matrix.indptr)
        # other_csr_matrix_user_profile_stats: DescribeResult = st.describe(
        #     a=other_csr_matrix_user_profile_length,
        #     axis=0,
        #     nan_policy="raise",
        # )
        #
        # self.statistics[dataset_name][csr_matrix_name]["interactions_by_users"] = {
        #     "num_observations": csr_matrix_user_profile_stats.nobs,
        #     "min": csr_matrix_user_profile_stats.minmax[0],
        #     "max": csr_matrix_user_profile_stats.minmax[1],
        #     "mean": csr_matrix_user_profile_stats.mean,
        #     "variance": csr_matrix_user_profile_stats.variance,
        #     "std": np.sqrt(csr_matrix_user_profile_stats.variance),
        #     "skewness": csr_matrix_user_profile_stats.skewness,
        #     "kurtosis": csr_matrix_user_profile_stats.kurtosis,
        #     "gini_index": gini_index(array=csr_matrix_user_profile_length,),
        # }
        #
        # self.statistics[dataset_name][other_csr_matrix_name]["interactions_by_users"] = {
        #     "num_observations": other_csr_matrix_user_profile_stats.nobs,
        #     "min": other_csr_matrix_user_profile_stats.minmax[0],
        #     "max": other_csr_matrix_user_profile_stats.minmax[1],
        #     "mean": other_csr_matrix_user_profile_stats.mean,
        #     "variance": other_csr_matrix_user_profile_stats.variance,
        #     "std": np.sqrt(other_csr_matrix_user_profile_stats.variance),
        #     "skewness": other_csr_matrix_user_profile_stats.skewness,
        #     "kurtosis": other_csr_matrix_user_profile_stats.kurtosis,
        #     "gini_index": gini_index(array=other_csr_matrix_user_profile_length),
        # }
        #
        # self.statistics[dataset_name][f"relative_difference_interactions_by_{pl_type}"] = {
        #     "num_observations": (
        #         self.statistics[dataset_name][csr_matrix_name]["interactions_by_users"]["num_observations"]
        #         - self.statistics[dataset_name][other_csr_matrix_name]["interactions_by_users"]["num_observations"]
        #     ),
        #     "min": (
        #         self.statistics[dataset_name][csr_matrix_name]["interactions_by_users"]["min"]
        #         - self.statistics[dataset_name][other_csr_matrix_name]["interactions_by_users"]["min"]
        #     ),
        #     "max": (
        #         self.statistics[dataset_name][csr_matrix_name]["interactions_by_users"]["max"]
        #         - self.statistics[dataset_name][other_csr_matrix_name]["interactions_by_users"]["max"]
        #     ),
        #     "mean": (
        #         self.statistics[dataset_name][csr_matrix_name]["interactions_by_users"]["mean"]
        #         - self.statistics[dataset_name][other_csr_matrix_name]["interactions_by_users"]["mean"]
        #     ),
        #     "variance": (
        #         self.statistics[dataset_name][csr_matrix_name]["interactions_by_users"]["variance"]
        #         - self.statistics[dataset_name][other_csr_matrix_name]["interactions_by_users"]["variance"]
        #     ),
        #     "std": (
        #         self.statistics[dataset_name][csr_matrix_name]["interactions_by_users"]["std"]
        #         - self.statistics[dataset_name][other_csr_matrix_name]["interactions_by_users"]["std"]
        #     ),
        #     "skewness": (
        #         self.statistics[dataset_name][csr_matrix_name]["interactions_by_users"]["skewness"]
        #         - self.statistics[dataset_name][other_csr_matrix_name]["interactions_by_users"]["skewness"]
        #     ),
        #     "kurtosis": (
        #         self.statistics[dataset_name][csr_matrix_name]["interactions_by_users"]["kurtosis"]
        #         - self.statistics[dataset_name][other_csr_matrix_name]["interactions_by_users"]["kurtosis"]
        #     ),
        #     "gini_index": gini_index(array=other_csr_matrix_user_profile_length),
        # }
        #
        # self.statistics[dataset_name]["relative_percentage_interactions_by_users"] = {
        #     "num_observations": (
        #         self.statistics[dataset_name][f"relative_difference_interactions_by_{pl_type}"]["num_observations"] * 100
        #         / self.statistics[dataset_name][csr_matrix_name]["interactions_by_users"]["num_observations"]
        #     ),
        #     "min": (
        #         self.statistics[dataset_name][f"relative_difference_interactions_by_{pl_type}"]["min"] * 100
        #         / self.statistics[dataset_name][csr_matrix_name]["interactions_by_users"]["min"]
        #     ),
        #     "max": (
        #         self.statistics[dataset_name][f"relative_difference_interactions_by_{pl_type}"]["max"] * 100
        #         / self.statistics[dataset_name][csr_matrix_name]["interactions_by_users"]["max"]
        #     ),
        #     "mean": (
        #         self.statistics[dataset_name][f"relative_difference_interactions_by_{pl_type}"]["mean"] * 100
        #         / self.statistics[dataset_name][csr_matrix_name]["interactions_by_users"]["mean"]
        #     ),
        #     "variance": (
        #         self.statistics[dataset_name][f"relative_difference_interactions_by_{pl_type}"]["variance"] * 100
        #         / self.statistics[dataset_name][csr_matrix_name]["interactions_by_users"]["variance"]
        #     ),
        #     "std": (
        #         self.statistics[dataset_name][f"relative_difference_interactions_by_{pl_type}"]["std"] * 100
        #         / self.statistics[dataset_name][csr_matrix_name]["interactions_by_users"]["std"]
        #     ),
        #     "skewness": (
        #         self.statistics[dataset_name][f"relative_difference_interactions_by_{pl_type}"]["skewness"] * 100
        #         / self.statistics[dataset_name][csr_matrix_name]["interactions_by_users"]["skewness"]
        #     ),
        #     "kurtosis": (
        #         self.statistics[dataset_name][f"relative_difference_interactions_by_{pl_type}"]["kurtosis"] * 100
        #         / self.statistics[dataset_name][csr_matrix_name]["interactions_by_users"]["kurtosis"]
        #     ),
        #     "gini_index": gini_index(array=other_csr_matrix_user_profile_length),
        # }

    def compute_statistics(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        columns_for_unique: list[str],
        columns_for_profile_length: list[str],
        columns_for_gini: list[str],
        columns_to_group_by: list[str],
    ) -> None:
        if dataset_name not in self.statistics:
            self.statistics[dataset_name] = dict()

        self.statistics[dataset_name]["num_records"] = df.shape[0]
        # self.statistics[dataset_name]["describe"] = df.describe(
        #     include=[np.number, "category"],
        #     exclude=[object],
        #     datetime_is_numeric=True,
        #     percentiles=[0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99]
        # )

        for column in tqdm(columns_for_unique):
            if column not in self.statistics[dataset_name]:
                self.statistics[dataset_name][column] = dict()

            self.statistics[dataset_name][column]["num_unique"] = df[column].nunique()
            self.statistics[dataset_name][column]["unique"] = df[column].unique()

        for column in tqdm(columns_for_profile_length):
            if column not in self.statistics[dataset_name]:
                self.statistics[dataset_name][column] = dict()

            self.statistics[dataset_name][column][f"profile_length"] = (
                df[column].value_counts(
                    ascending=False,
                    sort=False,
                    normalize=False,
                    dropna=False,
                ).rename(
                    "profile_length"
                ).to_frame()
            )

            self.statistics[dataset_name][column][f"profile_length_normalized"] = (
                df[column].value_counts(
                    ascending=False,
                    sort=False,
                    normalize=True,
                    dropna=False,
                ).rename(
                    "profile_length_normalized"
                ).to_frame()
            )

        for column in tqdm(columns_for_gini):
            if column not in self.statistics[dataset_name]:
                self.statistics[dataset_name][column] = dict()

            # notna is there because columns_to_keep might be NA.
            self.statistics[dataset_name][column]["gini_index_values_labels"] = gini_index(
                array=df[column]
            )
            self.statistics[dataset_name][column]["gini_index_values_counts"] = gini_index(
                array=df[column].value_counts(
                    dropna=True,
                    normalize=False,
                ),
            )

        # for column in tqdm(columns_to_group_by):
        #     if column not in self.statistics[dataset_name]:
        #         self.statistics[dataset_name][column] = dict()
        #
        #     # If the column is categorical, then a groupby will return for all categorical values,
        #     df_group_by = df.groupby(
        #         by=[column],
        #         as_index=False,
        #         observed=True,
        #     )
        #
        #     self.statistics[dataset_name][column][f"group_by_profile_length"] = df_group_by[column].count()
        #     self.statistics[dataset_name][column][f"group_by_describe"] = df_group_by[column].describe()
        #     self.statistics[dataset_name][column][f"group_by_agg"] = df_group_by[column].agg([
        #         "min",
        #         "max",
        #         "count",
        #         "size",
        #         "first",
        #         "last",
        #         "var",
        #         "std",
        #         "mean",
        #     ])

    def save_statistics(self):
        data_io = DataIO(
            folder_path=self.statistics_folder,
        )
        data_io.save_data(
            file_name=self.statistics_file_name,
            data_dict_to_save=self.statistics,
        )

    def load_statistics(self):
        data_io = DataIO(
            folder_path=self.statistics_folder,
        )
        self.statistics = data_io.load_data(
            file_name=self.statistics_file_name,
        )
