import enum
from typing import Any, cast, Literal, Optional, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from recsys_framework_extensions.logging import get_logger

tqdm.pandas()


logger = get_logger(
    logger_name=__file__,
)


T_KEEP = Literal["first", "last", False]
T_AXIS = Literal["index", "columns"]
T_MERGE = Literal["left", "inner", "right"]
T_HOW = Literal["neq", "eq", "geq", "ge", "leq", "le", "isin", "not_isin"]


class E_KEEP(enum.Enum):
    FIRST = "first"
    LAST = "last"
    FALSE = False


def _compute_statistics(
    func_name: str,
    message: str,
    df_orig: pd.DataFrame,
    df_keep: pd.DataFrame,
    df_removed: pd.DataFrame,
    **func_kwargs
) -> None:
    columns = ["user_id", "item_id", "impression_id"]

    statistics: dict[str, dict[str, Any]] = {
        **func_kwargs,
        "original_dataset": {
            column: {}
            for column in columns
        },
        "new_dataset": {
            column: {}
            for column in columns
        },
        "affected_dataset": {
            column: {}
            for column in columns
        },
        "removed": {
            column: {}
            for column in columns
        },
    }

    for column in columns:
        if column not in df_orig.columns:
            statistics["original_dataset"][column]["str"] = ""
            statistics["new_dataset"][column]["str"] = ""
            statistics["affected_dataset"][column]["str"] = ""
            statistics["removed"][column]["str"] = ""
            continue

        num_total_column = df_orig[column].nunique()

        num_keep_column = df_keep[column].nunique()
        num_affected_column = df_removed[column].nunique()
        num_removed_column = num_total_column - num_keep_column

        percentage_keep_column = (num_keep_column / num_total_column) * 100
        percentage_affected_column = (num_affected_column / num_total_column) * 100
        percentage_removed_column = (num_removed_column / num_total_column) * 100

        statistics["original_dataset"][column]["str"] = (
            f"\n\t* # {column}: {num_total_column} (100%)"
        )
        statistics["original_dataset"][column]["num"] = num_total_column

        statistics["new_dataset"][column]["str"] = (
            f"\n\t* # {column}: {num_keep_column}/{num_total_column} ({percentage_keep_column:.2f}%)"
        )
        statistics["new_dataset"][column]["num"] = num_keep_column
        statistics["new_dataset"][column]["percentage"] = percentage_keep_column

        statistics["affected_dataset"][column]["str"] = (
            f"\n\t* # {column}: {num_affected_column}/{num_total_column} ({percentage_affected_column:.2f}%)"
        )
        statistics["affected_dataset"][column]["num"] = num_affected_column
        statistics["affected_dataset"][column]["percentage"] = percentage_affected_column

        statistics["removed"][column]["str"] = (
            f"\n\t* # {column}: {num_removed_column}/{num_total_column} ({percentage_removed_column:.2f}%)"
        )
        statistics["removed"][column]["num"] = num_removed_column
        statistics["removed"][column]["percentage"] = percentage_removed_column

    num_total_records = df_orig.shape[0]

    num_keep_records = df_keep.shape[0]
    num_affected_records = df_removed.shape[0]
    num_removed_records = num_total_records - num_keep_records

    percentage_keep_records = (num_keep_records / num_total_records) * 100
    percentage_affected_records = (num_affected_records / num_total_records) * 100
    percentage_removed_records = (num_removed_records / num_total_records) * 100

    statistics["original_dataset"]["records"] = dict()
    statistics["new_dataset"]["records"] = dict()
    statistics["affected_dataset"]["records"] = dict()
    statistics["removed"]["records"] = dict()

    statistics["original_dataset"]["records"]["str"] = (
        f"\n\t* # Records: {num_total_records} (100%)"
    )
    statistics["original_dataset"]["records"]["num"] = num_total_records
    statistics["original_dataset"]["records"]["percentage"] = 100.0

    statistics["new_dataset"]["records"]["str"] = (
        f"\n\t* # Records: {num_keep_records}/{num_total_records} ({percentage_keep_records:.2f}%)"
    )
    statistics["new_dataset"]["records"]["num"] = num_keep_records
    statistics["new_dataset"]["records"]["percentage"] = percentage_keep_records

    statistics["affected_dataset"]["records"]["str"] = (
        f"\n\t* # Records: {num_affected_records}/{num_total_records} ({percentage_affected_records:.2f}%)"
    )
    statistics["affected_dataset"]["records"]["num"] = num_affected_records
    statistics["affected_dataset"]["records"]["percentage"] = percentage_affected_records

    statistics["removed"]["records"]["str"] = (
        f"\n\t* # Records: {num_removed_records}/{num_total_records} ({percentage_removed_records:.2f}%)"
    )
    statistics["removed"]["records"]["num"] = num_removed_records
    statistics["removed"]["records"]["percentage"] = percentage_removed_records

    logger.warning(
        f"Function {func_name}. "
        f"Found {num_affected_records}/{num_total_records} ({percentage_affected_records:.2f}%) {message}. "
        f"Function kwargs:"
        f"\n\t* {func_kwargs}"
        f"\nStatistics of the original dataset:"
        f"{statistics['original_dataset']['records']['str']}"
        f"{''.join([statistics['original_dataset'][column]['str'] for column in columns])}"
        
        f"\nStatistics of the new dataset:"
        f"{statistics['new_dataset']['records']['str']}"
        f"{''.join([statistics['new_dataset'][column]['str'] for column in columns])}"
        
        f"\nStatistics of the affected dataset:"
        f"{statistics['affected_dataset']['records']['str']}"
        f"{''.join([statistics['affected_dataset'][column]['str'] for column in columns])}"
        
        f"\nStatistics of the removed information:"
        f"{statistics['removed']['records']['str']}"
        f"{''.join([statistics['removed'][column]['str'] for column in columns])}"
        
        f"\nStatistics dict: "
        f"\n\t* {statistics=}"
    )


def _compute_statistics_impressions(
    func_name: str,
    message: str,
    df_orig: pd.DataFrame,
    df_keep: pd.DataFrame,
    df_removed: pd.DataFrame,
    **func_kwargs
) -> None:
    num_total_records = df_orig.shape[0]

    num_keep_records = df_keep.shape[0]
    percentage_keep_records = (num_keep_records / num_total_records) * 100

    num_records_with_repeated_interactions = df_removed.shape[0]
    percentage_records_with_repeated_interactions = (num_records_with_repeated_interactions / num_total_records) * 100

    num_removed_records = num_total_records - num_keep_records
    percentage_removed_records = (num_removed_records / num_total_records) * 100

    statistics_dict = dict(
        **func_kwargs,
        original_dataset=dict(
            num_records=num_total_records,
        ),
        new_dataset=dict(
            num_records=num_keep_records,
            percentage_records=percentage_keep_records,
        ),
        affected_dataset=dict(
            num_records=num_records_with_repeated_interactions,
            percentage_records=percentage_records_with_repeated_interactions,
        ),
        removed=dict(
            num_records=num_removed_records,
            percentage_records=percentage_removed_records,
        ),
    )

    logger.warning(
        f"Function {func_name}. "
        f"Found {num_removed_records}/{num_total_records} ({percentage_removed_records:.2f}%) {message}. "
        f"Function kwargs:"
        f"\n\t* {func_kwargs}"
        f"\nStatistics of the new dataset:"
        f"\n\t* # Records: {num_keep_records}/{num_total_records} ({percentage_keep_records:.2f}%)"
        f"\nStatistics of the other dataset:"
        f"\n\t* # Records: {num_records_with_repeated_interactions}/{num_total_records} ({percentage_records_with_repeated_interactions:.2f}%)"
        f"\nStatistics of the removed information:"
        f"\n\t* # Records: {num_removed_records}/{num_total_records} ({percentage_removed_records:.2f}%)"
        f"\nStatistics dict: "
        f"\n\t* {statistics_dict}"
    )


def remove_duplicates_in_interactions(
    df: pd.DataFrame,
    columns_to_compare: list[str],
    keep: Optional[E_KEEP],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Removes duplicates in a dataframe using the `drop_duplicates` method of Pandas.

    See Also
    --------
    pandas.DataFrame.drop_duplicates : drops duplicates in Pandas DataFrame.

    Returns
    -------
    A tuple. The first position is the dataframe without the duplicated records. The second position is the dataframe of
    the removed records.
    """
    assert (
        keep is None
        or keep in E_KEEP
        # or not keep
        # or keep in ["first", "last"]
    )

    num_unique_indices = df.index.nunique()
    num_indices = df.index.shape[0]

    if num_unique_indices != num_indices:
        raise ValueError(
            f"The function {remove_duplicates_in_interactions.__name__} needs the dataframe's index to not contain "
            f"duplicates. Number of unique values in the index: {num_unique_indices}. Number of indices: {num_indices}"
        )

    keep_val = (
        None
        if keep is None
        else keep.value
    )

    df_without_duplicates = df.drop_duplicates(
        subset=columns_to_compare,
        keep=keep_val,
        inplace=False,
        ignore_index=False,
    )

    df_duplicates = df.drop(
        df_without_duplicates.index,
        inplace=False,
    )

    _compute_statistics(
        func_name="remove_duplicates_in_interactions",
        message="repeated user-item interactions",
        df_orig=df,
        df_keep=df_without_duplicates,
        df_removed=df_duplicates,
        keep=keep,
    )

    return df_without_duplicates, df_duplicates


def remove_users_without_min_number_of_interactions(
    df: pd.DataFrame,
    users_column: str,
    min_number_of_interactions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Removes from a dataframe those users without a certain number of interactions_exploded.

    Returns
    -------
    A tuple. The first element is the dataframe with the records of selected users removed. The second element is a
    dataframe of the removed records.
    """
    if min_number_of_interactions < 0:
        raise ValueError(
            f"min_number_of_interactions must be greater than zero. Value passed: {min_number_of_interactions}"
        )

    num_unique_indices = df.index.nunique()
    num_indices = df.index.shape[0]

    if num_unique_indices != num_indices:
        raise ValueError(
            f"The function {remove_users_without_min_number_of_interactions.__name__} needs the dataframe's index to "
            f"not contain duplicates. Number of unique values in the index: {num_unique_indices}. "
            f"Number of indices: {num_indices}"
        )

    # There are two ways to do this calculation.
    # 1) df.groupby(by=user_column, as_index=False)[user_column].size()
    # 2) df.value_counts(subset=["user_id"], normalize=False, sort=False, ascending=False, dropna=True).to_frame("size").reset_index(drop=False)
    # In both cases we will have a DataFrame with a 0...N index, a column `user_column` with the unique user ids, and
    # a column `size` containing the frequency of each user id in the dataset.
    # 1) is slightly faster than 2), a jupyter-lab %%timeit yields (5.58 ms ± 144 µs per loop (mean ± std. dev.
    # of 7 runs, 100 loops each) vs 6.55 ms ± 139 µs per loop (mean ± std. dev. of 7 runs, 100 loops each).
    grouped_df = df.groupby(
        by=users_column,
        as_index=False,
    )[users_column].size()

    users_to_keep = grouped_df[
        grouped_df["size"] >= min_number_of_interactions
        ][users_column]

    df_users_to_keep = df[
        df[users_column].isin(users_to_keep)
    ].copy()

    df_removed_users = df.drop(
        df_users_to_keep.index
    )

    _compute_statistics(
        func_name="remove_users_without_min_number_of_interactions",
        message="interactions of users without the minimum # of interactions",
        df_orig=df,
        df_keep=df_users_to_keep,
        df_removed=df_removed_users,
        min_number_of_interactions=min_number_of_interactions,
    )

    return df_users_to_keep, df_removed_users


def filter_impressions_by_interactions_index(
    df_impressions: pd.DataFrame,
    df_interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indices_in_interactions_outside_impressions = set(df_interactions.index).difference(df_impressions.index)
    if len(indices_in_interactions_outside_impressions) > 0:
        raise AssertionError(
            f"This function needs the interaction indices to be a subset of the impression indices, this is because "
            f"the interactions set has been filtered. Found the following indices in the interactions that are not in "
            f"the impressions ones: {indices_in_interactions_outside_impressions}."
        )

    # To create the UIM we need to have the (user_id, impressions) pair.
    # Also, we need to filter the impressions by the same filters of the interactions.
    # We do this by selecting from the impression set the remaining indices in the interactions after we've
    # applied all filters.
    # NOTE: be careful, this can be done with a join too
    # (e.g., df_impressions_filtered.join(other=df_interactions_filtered)), but given that the interactions have
    # several repeated indices (given that we exploded it before), then a join would create repeated impressions
    # records.
    df_impressions_keep = df_impressions[
        df_impressions.index.isin(df_interactions.index)
    ].copy()

    df_impressions_removed = df_impressions.drop(
        df_impressions_keep.index,
        inplace=False,
    )

    _compute_statistics(
        func_name="filter_impressions_by_interactions_index",
        message="impressions inside the filtered interactions.",
        df_orig=df_impressions,
        df_keep=df_impressions_keep,
        df_removed=df_impressions_removed,
    )

    return df_impressions_keep, df_impressions_removed


def filter_dataframe_by_column(
    df_to_filter: pd.DataFrame,
    df_filterer: pd.DataFrame,
    column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values_outside_filter = set(df_filterer[column]).difference(df_to_filter[column])
    if len(values_outside_filter) > 0:
        raise AssertionError(
            f"This function needs the `df_filterer[column]` values to be a subset of `df_to_filter[column]`, "
            f"due to the assumption that `df_filterer` has been filtered already and we want to apply the same filter "
            f"to `df_to_filter`. Found the following values in the `df_filterer` that are not in `df_to_filter`: "
            f"{values_outside_filter}."
        )

    # To create the UIM we need to have the (user_id, impressions) pair.
    # Also, we need to filter the impressions by the same filters of the interactions.
    # We do this by selecting from the impression set the remaining indices in the interactions after we've
    # applied all filters.
    # NOTE: be careful, this can be done with a join too
    # (e.g., df_impressions_filtered.join(other=df_interactions_filtered)), but given that the interactions have
    # several repeated indices (given that we exploded it before), then a join would create repeated impressions
    # records.
    df_keep = df_to_filter[
        df_to_filter[column].isin(df_filterer[column])
    ].copy()

    df_removed = df_to_filter.drop(
        df_keep.index,
        inplace=False,
    )

    _compute_statistics_impressions(
        func_name="filter_dataframe_by_column",
        message="values inside the filter.",
        df_orig=df_to_filter,
        df_keep=df_keep,
        df_removed=df_removed,
    )

    return df_keep, df_removed


def remove_records_by_threshold(
    df: pd.DataFrame,
    column: str,
    threshold: Any,
    how: T_HOW = "geq",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filters the dataset on the column `column` using a threshold `threshold`.

    In particular, this method keeps those records that are greater or equal than `threshold`. Therefore, discarding
    those that are strictly less than `threshold`.

    Notes
    -----
    This method preserves the original indices.
    """
    if how == "neq":
        df_filter_to_keep = df[column] != threshold
    elif how == "eq":
        df_filter_to_keep = df[column] == threshold
    elif how == "geq":
        df_filter_to_keep = df[column] >= threshold
    elif how == "ge":
        df_filter_to_keep = df[column] > threshold
    elif how == "leq":
        df_filter_to_keep = df[column] <= threshold
    elif how == "le":
        df_filter_to_keep = df[column] < threshold
    elif how == "isin":
        df_filter_to_keep = df[column].isin(threshold)
    elif how == "not_isin":
        df_filter_to_keep = ~df[column].isin(threshold)
    else:
        raise ValueError(
            f"Received invalid value for ``how`` ({how}). Valid values are: {T_HOW}"
        )

    df_filter_to_keep = cast(pd.Series, df_filter_to_keep)

    df_keep = df[df_filter_to_keep].copy()
    df_removed = df[~df_filter_to_keep].copy()

    _compute_statistics(
        func_name="remove_records_by_threshold",
        message="Filter by threshold value",
        df_orig=df,
        df_keep=df_keep,
        df_removed=df_removed,
        column=column,
        threshold=threshold,
        how=how,
    )

    return df_keep, df_removed


def remove_records_na(
    df: pd.DataFrame,
    column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filters the dataset on the column `column` using a threshold `threshold`.

    In particular, this method keeps those records that are greater or equal than `threshold`. Therefore, discarding
    those that are strictly less than `threshold`.

    Notes
    -----
    This method preserves the original indices.
    """
    df_filter_to_keep = cast(
        pd.Series,
        df[column].notna()
    )

    df_keep = df[df_filter_to_keep].copy()
    df_removed = df[~df_filter_to_keep].copy()

    _compute_statistics(
        func_name="remove_records_na",
        message="Filter by NA",
        df_orig=df,
        df_keep=df_keep,
        df_removed=df_removed,
        column=column,
    )

    return df_keep, df_removed


def apply_custom_function(
    df: pd.DataFrame,
    column: str,
    func: Callable[[], Any],
    func_name: str,
    axis: T_AXIS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_keep = df.copy()
    df_keep[column] = df[column].progress_apply(
        func=func,
    )

    df_removed = df.drop(
        df_keep.index,
        inplace=False,
    )

    _compute_statistics(
        func_name=func_name,
        message=f"Applied {func_name}",
        df_orig=df,
        df_keep=df_keep,
        df_removed=df_removed,
        column=column,
        custom_func_name=func_name,
        axis=axis,
    )

    return df_keep, df_removed


def merge_two_dataframes(
    df: pd.DataFrame,
    other: pd.DataFrame,
    how: T_MERGE,
    left_on: Optional[str],
    right_on: Optional[str],
    left_index: bool,
    right_index: bool,
    suffixes: tuple[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if left_on is None and right_on is None and not left_index and not right_index:
        raise ValueError(
            "At least two of 'left_on', 'right_on', 'left_index', or 'right_index' must be specified. Column names "
            "for 'left_on' and 'right_on' or True for 'left_index' and 'right_index'."
        )

    df_keep = df.merge(
        right=other,
        how=how,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
        suffixes=suffixes,
        sort=False,
    )

    df_removed = df.drop(
        df_keep.index,
    )

    return df_keep, df_removed


def split_sequential_train_test_by_column_threshold(
    df: pd.DataFrame,
    column: str,
    threshold: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partitions the dataset into train and test after a certain threshold.

    This method is specially useful when partitioning a dataset on timestamps is required. In particular, this method
    assigns to the *train set* those records that are less or equal than `threshold`. Therefore, the *test set*
    contains those values that are greater than `threshold`.

    Notes
    -----
    This method preserves the original indices.
    """
    df_filter = cast(
        pd.Series,
        df[column] <= threshold
    )

    df_train = df[df_filter].copy()
    df_test = df[~df_filter].copy()

    _compute_statistics(
        func_name="split_sequential_train_test_by_column_threshold",
        message="Split by threshold value",
        df_orig=df,
        df_keep=df_train,
        df_removed=df_test,
        column=column,
        threshold=threshold,
    )

    return df_train, df_test


def split_sequential_train_test_by_num_records_on_test(
    df: pd.DataFrame,
    group_by_column: str,
    num_records_in_test: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partitions the dataset sequentially by a group key maintaining a certain number of records in the test set.

    Notes
    -----
    This method creates *first* the test set and then the train set, due to the pandas API. Therefore, if any group has
    less than (`num_records_in_test` + 1) interactions_exploded, these record will be sent to the test set instead of
    the train set. Moreover, if this happens, the method will raise a ValueError instance.

    This method preserves the original indices.

    Raises
    ------
    ValueError
        If any group has 0 or 1 records.

    """
    assert num_records_in_test > 0

    num_unique_indices = df.index.nunique()
    num_indices = df.index.shape[0]

    if num_unique_indices != num_indices:
        raise ValueError(
            f"The function {split_sequential_train_test_by_num_records_on_test.__name__} needs the dataframe's index to "
            f"not contain duplicates. Number of unique values in the index: {num_unique_indices}. "
            f"Number of indices: {num_indices}"
        )

    # There are two ways to generate a leave-last-out strategy.
    # The first uses df.groupby(...).nth[-1].
    # The second uses df.reset_index(drop=False).groupby(...).last().set_index("index").
    # Both approaches take the last record of each group with their respective indices.
    # The training set is the resulting dataframe (named df_test)
    # The training set is then df.drop(test_set.index) (named df_train)
    # NOTE: a groupby on a categorical column will have rows for categories not inside the dataframe, e.g.,
    # if some users were removed from the interactions, the groupby will show their sizes as 0 because they're part
    # of the category. We avoid this by setting `observed=True`.
    grouped_df = df.groupby(
        by=group_by_column,
        as_index=False,
        observed=True,
    )

    # This variable tells the minimum size of each group.
    min_num_records_by_group = num_records_in_test + 1
    grouped_size_df = grouped_df[group_by_column].size()

    # NOTE: Not needed anymore given the `observed=True` in the GroupBy.
    # Given that the groupby counts for all user, even if they are not in the interactions, then we must filter the
    # grouped dataframe by the actual users in the interactions dataframe.
    # users_in_df = grouped_size_df[
    #     grouped_size_df[group_by_column].isin(df[group_by_column].unique())
    # ]
    non_valid_groups = grouped_size_df["size"] < min_num_records_by_group
    if np.any(non_valid_groups):
        message = (
            f"Cannot partition the dataset given that the following groups do not have at least "
            f"{min_num_records_by_group} interaction records:"
            f"\n{grouped_size_df[non_valid_groups]}"
        )
        logger.error(message)
        raise ValueError(message)

    nths_to_take = np.arange(
        start=-1,
        step=-1,
        stop=-(num_records_in_test + 1),
        dtype=np.int32,
    )
    # We need to convert nths_to_take as a list because pandas <1.4 does not support np.ndarrays.
    df_test = grouped_df.nth(
        n=list(nths_to_take),
    ).copy()

    df_train = df.drop(
        df_test.index,
        inplace=False
    ).copy()

    _compute_statistics(
        func_name="split_sequential_train_test_by_num_records_on_test",
        message="Split by leave-last-k-out",
        df_orig=df,
        df_keep=df_train,
        df_removed=df_test,
        group_by_column=group_by_column,
        num_records_in_test=num_records_in_test,
    )

    return df_train, df_test


def randomly_sample_by_column_values(
    df: pd.DataFrame,
    frac_to_keep: float,
    column: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partitions the dataset sequentially by a group key maintaining a certain number of records in the test set.

    Notes
    -----
    This method creates *first* the test set and then the train set, due to the pandas API. Therefore, if any group has
    less than (`num_records_in_test` + 1) interactions_exploded, these record will be sent to the test set instead of
    the train set. Moreover, if this happens, the method will raise a ValueError instance.

    This method preserves the original indices.

    Raises
    ------
    ValueError
        If any group has 0 or 1 records.

    """
    assert 0.0 <= frac_to_keep <= 1.0

    if frac_to_keep == 1.0:
        df_keep = df
        df_removed = df.DataFrame(data=None, columns=df.columns, index=df.index)

    elif frac_to_keep == 0.0:
        df_keep = df.DataFrame(data=None, columns=df.columns, index=df.index)
        df_removed = df

    else:
        # We want to have the numpy array because the columns_to_keep can be categoricals.
        if "categorical" in str(df[column].dtype):
            arr_unique_values: np.ndarray = df[column].unique().to_numpy()
        else:
            arr_unique_values: np.ndarray = df[column].unique()

        num_unique_values = arr_unique_values.shape[0]

        unique_values_to_keep: np.ndarray = np.random.default_rng(seed=seed).choice(
            a=arr_unique_values,
            size=int(num_unique_values * frac_to_keep),
            replace=False,
            shuffle=True,
            p=None,  # This ensures uniformly-sampled values.
        )

        df_keep = df[
            df[column].isin(unique_values_to_keep)
        ].copy()

        df_removed = df.drop(
            df_keep.index,
            inplace=False,
        ).copy()

    _compute_statistics(
        func_name="randomly_sample_by_column_values",
        message=f"Sampled dataframe by column {column} using a fraction of {frac_to_keep} with the seed {seed}",
        df_orig=df,
        df_keep=df_keep,
        df_removed=df_removed,
        frac_to_keep=frac_to_keep,
        column=column,
        seed=seed,
    )

    return df_keep, df_removed


# if __name__ == "__main__":
#     test_df = pd.DataFrame(
#         data=dict(
#             a=[0, 1, 2, 3, 4, 5],
#             b=[3, 4, 5, 6, 7, 8],
#             group_id=[99, 100, 101, 101, 101, 100],
#         ),
#         index=[9, 10, 11, 12, 13, 14],
#     )
#
#     expected_df_train = pd.DataFrame(
#         data=dict(
#             a=[0, 1, 2, 3],
#             b=[3, 4, 5, 6],
#             group_id=[99, 100, 101, 101],
#         ),
#         index=[9, 10, 11, 12],
#     )
#     expected_df_validation = pd.DataFrame(
#         data=dict(
#             a=[4],
#             b=[7],
#             group_id=[101],
#         ),
#         index=[13],
#     )
#     expected_df_test = pd.DataFrame(
#         data=dict(
#             a=[5],
#             b=[8],
#             group_id=[100],
#         ),
#         index=[14],
#     )
#
#     df_keep, df_removed = remove_users_without_min_number_of_interactions(
#         df=test_df,
#         users_column="group_id",
#         min_number_of_interactions=3,
#     )
#
#     df_train, df_test = split_sequential_train_test_by_num_records_on_test(
#         df=df_keep,
#         group_by_column="group_id",
#         num_records_in_test=1,
#     )
#
#     df_train, df_validation = split_sequential_train_test_by_num_records_on_test(
#         df=df_train,
#         group_by_column="group_id",
#         num_records_in_test=1,
#     )
#
#     df_train, df_test = split_sequential_train_test_by_column_threshold(
#         df=test_df,
#         column="b",
#         threshold=7
#     )
#
#     df_train, df_validation = split_sequential_train_test_by_column_threshold(
#         df=df_train,
#         column="b",
#         threshold=6
#     )
#
#     assert expected_df_train.equals(
#         other=df_train,
#     )
#     assert expected_df_validation.equals(
#         other=df_validation,
#     )
#     assert expected_df_test.equals(
#         other=df_test,
#     )
