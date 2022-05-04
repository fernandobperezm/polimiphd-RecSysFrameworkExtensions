from typing import Any, cast, Optional

import tqdm
import numpy as np
import pandas as pd
import scipy.sparse as sp

from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

from recsys_framework_extensions.decorators import log_calling_args
from recsys_framework_extensions.logging import get_logger

logger = get_logger(
    logger_name=__name__,
)


def _compute_statistics(
    df: pd.DataFrame,
    df_keep: pd.DataFrame,
    sparse_matrix: sp.csr_matrix,
) -> None:

    nonzero_rows, nonzero_cols = sparse_matrix.nonzero()

    num_unique_users = np.unique(nonzero_rows).shape[0]
    num_unique_items = np.unique(nonzero_cols).shape[0]
    num_unique_data = np.unique(sparse_matrix.data).shape[0]

    logger.info(
        f"Sparse Matrix Creation Statistics:"
        f"\nDF:"
        f"\n\t shape={df.shape}"
        f"\nDF - NAs Dropped:"
        f"\n\t shape={df_keep.shape}"
        f"\nCSR Sparse Matrix:"
        f"\n\t nnz={sparse_matrix.nnz}"
        f"\n\t shape={sparse_matrix.shape}"
        f"\nData Points:"
        f"\n\t num_unique_users={num_unique_users}"
        f"\n\t num_unique_items={num_unique_items}"
        f"\n\t num_unique_data={num_unique_data}"
    )


def assert_mappers_are_equal(
    mapper_id_to_index: dict[Any, Any],
    other_mapper_id_to_index: dict[Any, Any],
) -> None:
    # The == operator in python compares two dictionaries by their key and values automatically, i.e., it compares
    # that both dictionaries share the same (key, value) pairs.
    if mapper_id_to_index == other_mapper_id_to_index:
        return

    differences_not_in_other = set(
        mapper_id_to_index.items()
    ).difference(
        other_mapper_id_to_index.items()
    )

    differences_not_in_original = set(
        other_mapper_id_to_index.items()
    ).difference(
        mapper_id_to_index.items()
    )

    message = (
        f"Expected mappers to be equal but found differences in them. "
        f"\nValues not in the calculated mapper: \n* {differences_not_in_other}."
        f"\nValues not in the original mapper: \n* {differences_not_in_original}."
    )

    logger.error(message)
    raise AssertionError(message)


def create_sparse_matrix_from_dataframe(
    df: pd.DataFrame,
    users_column: str,
    items_column: str,
    binarize_interactions: bool,
    mapper_user_id_to_index: dict[Any, Any],
    mapper_item_id_to_index: dict[Any, Any],
    data_column: Optional[str] = None,
) -> sp.csr_matrix:
    # Just keep a copy of the columns_to_keep we're interested in.
    if data_column is None:
        df = df[[users_column, items_column]].copy()
    else:
        df = df[[users_column, items_column, data_column]].copy()

    # Explode item lists so rows can be inserted faster. This may blow up memory.
    if "object" == str(df[items_column].dtype):
        # Warning, in pandas versions <1.3 we cannot use lists of columns_to_keep.
        df = df.explode(
            column=items_column,
            ignore_index=False,
        )

    df_keep = df.dropna(
        axis="index",
        inplace=False,
        how="any",
    )

    # With this info we can see if we are having null values in the rows we're loading to the URM.
    df.info(
        show_counts=True,
        memory_usage="deep",
    )
    df_keep.info(
        show_counts=True,
        memory_usage="deep",
    )

    users = df_keep[users_column].to_numpy()
    items = df_keep[items_column].to_numpy()
    if data_column is not None:
        data = df_keep[data_column].to_numpy()
    else:
        data = np.ones_like(users, dtype=np.int32)

    builder_sparse_matrix = IncrementalSparseMatrix_FilterIDs(
        preinitialized_col_mapper=mapper_item_id_to_index,
        on_new_col="add",
        preinitialized_row_mapper=mapper_user_id_to_index,
        on_new_row="add"
    )

    builder_sparse_matrix.add_data_lists(
        row_list_to_add=users,
        col_list_to_add=items,
        data_list_to_add=data,
    )

    sparse_matrix = builder_sparse_matrix.get_SparseMatrix()
    if binarize_interactions:
        sparse_matrix.data = np.ones_like(sparse_matrix.data, dtype=np.int32)

    assert_mappers_are_equal(
        mapper_id_to_index=mapper_user_id_to_index,
        other_mapper_id_to_index=builder_sparse_matrix.get_row_token_to_id_mapper()
    )

    assert_mappers_are_equal(
        mapper_id_to_index=mapper_item_id_to_index,
        other_mapper_id_to_index=builder_sparse_matrix.get_column_token_to_id_mapper(),
    )

    _compute_statistics(
        df=df,
        df_keep=df_keep,
        sparse_matrix=sparse_matrix,
    )

    return sparse_matrix


def create_sparse_matrix_from_iter_dataframe(
    df: pd.DataFrame,
    users_column: str,
    items_column: str,
    binarize_interactions: bool,
    mapper_user_id_to_index: dict[Any, Any],
    mapper_item_id_to_index: dict[Any, Any],
) -> sp.csr_matrix:
    builder_sparse_matrix = IncrementalSparseMatrix_FilterIDs(
        preinitialized_col_mapper=mapper_item_id_to_index,
        on_new_col="add",
        preinitialized_row_mapper=mapper_user_id_to_index,
        on_new_row="add"
    )

    # Explode item lists so rows can be inserted faster. This may blow up memory.
    if "object" != str(df[items_column].dtype):
        raise ValueError(
            f"This method should be used only when the items column is of 'object' dtype and is an array."
        )

    num_null_rows = 0
    row: pd.DataFrame
    for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        user = cast(
            int,
            row[users_column]
        )
        items = cast(
            np.ndarray,
            row[items_column],
        )
        data = 1

        if items.size == 0:
            num_null_rows += 1
            continue

        builder_sparse_matrix.add_single_row(
            row_index=user,
            col_list=items,
            data=data,
        )

    df_keep = df[
        df[items_column].notna()
    ]

    df_keep_drop_na = df.dropna(
        axis="index",
        inplace=False,
        how="any",
    )

    # With this info we can see if we are having null values in the rows we're loading to the URM.
    df.info(
        show_counts=True,
        memory_usage="deep",
    )
    df_keep.info(
        show_counts=True,
        memory_usage="deep",
    )
    df_keep_drop_na.info(
        show_counts=True,
        memory_usage="deep",
    )
    print(f"{num_null_rows=}")

    sparse_matrix = builder_sparse_matrix.get_SparseMatrix()
    if binarize_interactions:
        sparse_matrix.data = np.ones_like(sparse_matrix.data, dtype=np.int32)

    assert_mappers_are_equal(
        mapper_id_to_index=mapper_user_id_to_index,
        other_mapper_id_to_index=builder_sparse_matrix.get_row_token_to_id_mapper()
    )

    assert_mappers_are_equal(
        mapper_id_to_index=mapper_item_id_to_index,
        other_mapper_id_to_index=builder_sparse_matrix.get_column_token_to_id_mapper(),
    )

    _compute_statistics(
        df=df,
        df_keep=df_keep,
        sparse_matrix=sparse_matrix,
    )

    return sparse_matrix
