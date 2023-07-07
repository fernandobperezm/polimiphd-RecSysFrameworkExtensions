import datetime
import logging
from itertools import tee, chain
from typing import Literal, Iterable, TypeVar, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from recsys_framework_extensions.decorators import log_calling_args


logger = logging.getLogger(__name__)

tqdm.pandas()


T_LAST_SEEN_GRANULARITY = Literal["euclidean", "days", "seconds"]
T_KEEP = Literal["first", "last", False]


T_GEN = TypeVar('T_GEN')

_dict_functions = {
    "euclidean": lambda x: x,
    "days": lambda x: x.days,
    "seconds": lambda x: x.seconds,
}


def iterator_previous_and_current(iterable: Iterable[T_GEN]) -> Iterable[tuple[Optional[T_GEN], T_GEN]]:
    prevs: Iterable[Optional[T_GEN]]
    items: Iterable[T_GEN]

    prevs, items = tee(iterable, 2)
    prevs = chain([None], prevs)

    return zip(prevs, items)


def _compute_diff_timestamps_per_user_item_group(
    series_group: pd.Series,
) -> pd.DataFrame:
    """

    Parameters
    ----------
    series_group
        Pandas Series that contains datetime objects.

    Returns
    -------
    Pandas Series
        A pandas series containing the difference between the current and the previous timestamp for each user-item
        group.
    """
    assert _is_column_a_datetime(df=series_group, column="")

    timestamps = series_group.to_list()

    iter_timestamps = iterator_previous_and_current(timestamps)

    diff_per_group = []

    tup: tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]
    for prev_timestamp, current_timestamp in iter_timestamps:
        if prev_timestamp is None:
            tup = (None, None, None, None, None)

        else:
            diff = (current_timestamp - prev_timestamp)

            assert isinstance(diff, datetime.timedelta)

            total_seconds = diff.total_seconds()

            if total_seconds == 0:
                tup = (None, None, None, None, None)

            else:
                calculator_minutes = 60
                total_minutes = total_seconds / calculator_minutes

                calculator_hours = 60 * calculator_minutes
                total_hours = total_seconds / calculator_hours

                calculator_days = 24 * calculator_hours
                total_days = total_seconds / calculator_days

                calculator_weeks = 7 * calculator_days
                total_weeks = total_seconds / calculator_weeks

                tup = (total_seconds, total_minutes, total_hours, total_days, total_weeks)

        diff_per_group.append(tup)

    assert len(diff_per_group) == len(timestamps)

    columns = [
        "feature_last_seen_total_seconds",
        "feature_last_seen_total_minutes",
        "feature_last_seen_total_hours",
        "feature_last_seen_total_days",
        "feature_last_seen_total_weeks",
    ]
    index = series_group.index

    if len(diff_per_group) == 0:
        return pd.DataFrame(
            data=None,
            columns=columns,
            index=index,
        )

    return pd.DataFrame.from_records(
        data=diff_per_group,
        columns=columns,
        index=index,
    )


def _compute_diff_euclidean_timestamps_per_user_item_group(
    series_group: pd.Series,
) -> pd.DataFrame:
    """

    Parameters
    ----------
    series_group
        Pandas Series that contains datetime objects.

    Returns
    -------
    Pandas Series
        A pandas series containing the difference between the current and the previous timestamp for each user-item
        group.
    """
    timestamps = series_group.to_list()
    iter_timestamps = iterator_previous_and_current(timestamps)

    diff_per_group = []

    diff: tuple[Optional[int]]
    for prev_timestamp, current_timestamp in iter_timestamps:
        if prev_timestamp is None:
            diff = (None,)

        else:
            diff = (current_timestamp - prev_timestamp,)

        diff_per_group.append(diff)

    assert len(diff_per_group) == len(timestamps)

    return pd.DataFrame.from_records(
        data=diff_per_group,
        columns=["feature_last_seen_euclidean"],
        index=series_group.index,
    )


@log_calling_args
def extract_frequency_user_item(
    df: pd.DataFrame,
    users_column: str,
    items_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_column = f"feature-{users_column}-{items_column}-frequency"

    df = _preprocess_df_for_feature_extraction(
        df=df,
        columns_to_keep=[users_column, items_column],
        items_column=items_column,
    )

    # This creates a dataframe with columns_to_keep: users_column|items_column|frequency
    df_user_item_frequency = df.value_counts(
        subset=[users_column, items_column],
        normalize=False,
        sort=False,
        ascending=False,
    ).to_frame(
        name=feature_column,
    ).reset_index(
        drop=False,
    )

    df_user_item_frequency = _check_empty_dataframe(
        df=df_user_item_frequency,
        columns=[users_column, items_column, feature_column]
    )

    return df_user_item_frequency, pd.DataFrame([])


@log_calling_args
def extract_last_seen_user_item(
    df: pd.DataFrame,
    users_column: str,
    items_column: str,
    timestamp_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _preprocess_df_for_feature_extraction(
        df=df,
        columns_to_keep=[users_column, items_column, timestamp_column],
        items_column=items_column,
    )

    df = df.sort_values(
        by=[users_column, items_column, timestamp_column],
        ascending=True,
        axis="index",
        inplace=False,
        ignore_index=True,
    )

    # Keep a dataframe of user-item pairs that have been impressed more than once.
    df = df.groupby(
        by=[users_column, items_column],
        as_index=False,
        sort=False,
        observed=True,  # Avoid computing this on categorical.
        dropna=True,
    ).filter(
        lambda df_group: df_group.shape[0] >= 2
    )

    if _is_column_a_datetime(df, column=timestamp_column):
        feature_columns = [
            "feature_last_seen_total_seconds",
            "feature_last_seen_total_minutes",
            "feature_last_seen_total_hours",
            "feature_last_seen_total_days",
            "feature_last_seen_total_weeks",
        ]
        func_df_apply = _compute_diff_timestamps_per_user_item_group
    else:
        feature_columns = ["feature_last_seen_euclidean"]
        func_df_apply = _compute_diff_euclidean_timestamps_per_user_item_group

    if df.empty:
        df_keep = pd.DataFrame(data=None, columns=feature_columns)
        df_removed = pd.DataFrame(data=None, columns=feature_columns)
    else:
        df_grouped = df.groupby(
            by=[users_column, items_column],
            as_index=False,
            sort=False,
            observed=True,  # Avoid computing this on categorical.
            dropna=True,
            group_keys=False  # In Pandas 2.0 this defaults to True, however, this causes the resulting dataframe into having a multi-index composed of the values in <users_column> and <items_column>. We don't want a multi-index because it causes the `df.merge` call to fail with an exception.
        )
        df_last_seen_components: pd.DataFrame = df_grouped[timestamp_column].progress_apply(
            func=func_df_apply,
        )

        assert df.shape[0] == df_last_seen_components.shape[0]

        df = df.merge(
            right=df_last_seen_components,
            how="inner",
            left_index=True,
            right_index=True,
            left_on=None,
            right_on=None,
            suffixes=("", ""),
            sort=False,
        )

        df_last_user_item_record = df_grouped.nth([-1])

        df_keep: pd.DataFrame = df[
            df.index.isin(
                df_last_user_item_record.index
            )
        ]

        df_removed: pd.DataFrame = df.drop(
            df_last_user_item_record.index
        )

    df_keep = _check_empty_dataframe(
        df=df_keep,
        columns=[users_column, items_column, timestamp_column, *feature_columns],
    )

    df_removed = _check_empty_dataframe(
        df=df_removed,
        columns=[users_column, items_column, timestamp_column, *feature_columns],
    )

    return df_keep, df_removed


def _extract_group_last_seen_user_item(
    gdf: pd.DataFrame,
    timestamp_column: str,
) -> Optional[pd.DataFrame]:
    if gdf.shape[0] <= 1:
        return None

    # Sort each group by its timestamp
    gdf = gdf.sort_values(
        by=timestamp_column, ascending=True, axis="index", inplace=False, ignore_index=False
    )

    if _is_column_a_datetime(df=gdf, column=timestamp_column):
        df_features = _compute_diff_timestamps_per_user_item_group(
            series_group=gdf[timestamp_column]
        )
    else:
        df_features = _compute_diff_euclidean_timestamps_per_user_item_group(
            series_group=gdf[timestamp_column]
        )

    # Add the new columns to the group. They share the index so they will align automatically.
    gdf = pd.concat(
        [gdf, df_features],
        join="inner",
        ignore_index=False,
        verify_integrity=False,
        axis="columns",
        sort=False
    ).dropna(
        axis="index",
        how="any",
        inplace=False,
    ).tail(
        # Take the last record of the group, feature last_seen assumes it is the last_seen in the dataset.
        n=1
    )

    return gdf


@log_calling_args
def extract_last_seen_user_item_2(
    df: pd.DataFrame,
    users_column: str,
    items_column: str,
    timestamp_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if _is_column_a_datetime(df, column=timestamp_column):
        feature_columns = [
            "feature_last_seen_total_seconds",
            "feature_last_seen_total_minutes",
            "feature_last_seen_total_hours",
            "feature_last_seen_total_days",
            "feature_last_seen_total_weeks",
        ]
    else:
        feature_columns = ["feature_last_seen_euclidean"]

    df = _preprocess_df_for_feature_extraction(
        df=df,
        columns_to_keep=[users_column, items_column, timestamp_column],
        items_column=items_column,
    )

    df_keep = df.groupby(
        by=[users_column, items_column],
        as_index=False,
        sort=False,
        observed=True,  # Avoid computing this on categorical.
        dropna=True,
        group_keys=False,
    ).progress_apply(
        # Do on each group.
        lambda gdf: _extract_group_last_seen_user_item(gdf=gdf, timestamp_column=timestamp_column)
    )

    df_removed = df.drop(
        df_keep.index
    )

    df_keep = _check_empty_dataframe(
        df=df_keep,
        columns=[users_column, items_column, timestamp_column, *feature_columns]
    )

    df_removed = _check_empty_dataframe(
        df=df_removed,
        columns=[users_column, items_column, timestamp_column, *feature_columns]
    )

    return df_keep, df_removed


@log_calling_args
def extract_position_user_item(
    df: pd.DataFrame,
    users_column: str,
    items_column: str,
    positions_column: Optional[str],
    to_keep: T_KEEP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert _is_column_an_object(df=df, column=items_column)

    positions_in_df = positions_column is not None and positions_column in df.columns
    feature_column = f"feature-{users_column}-{items_column}-position"

    # To extract positions, we first need a list of positions for each
    if positions_in_df:
        df = df[[users_column, items_column, positions_column]]
    else:
        df = df[[users_column, items_column]]
    df = df.dropna(
        axis="index",
        inplace=False,
        how="any",
    )

    # To extract the position, we need first to create an array of positions in the dataframe for each impression
    # record, i.e., the columns should be user_id|impression|position and a record looking like 5|[6,1,3]|[0,1,2].
    # This line assumes that the dataframe does not contain NAs.
    assert not all(df[items_column].isna())
    if positions_in_df:
        df[feature_column] = df[positions_column]
    else:
        df[feature_column] = df[items_column].str.len().progress_map(np.arange)

    # Pandas version <1.3 there is no multi-column explode. Therefore, we must apply the `explode` on the two columns
    # we're interested: `items_column` and `position`, as both are lists and need to be exploded together.
    df = df.set_index(
        [users_column]
    ).apply(
        pd.Series.explode,
    ).reset_index(
        # Returns the `user_id` column and creates a new integer-based index with unique values for each row.
        drop=False,
    ).astype({
        feature_column: np.int32,
    })

    df_keep: pd.DataFrame = df.drop_duplicates(
        subset=[users_column, items_column],
        keep=to_keep,
        inplace=False,
        ignore_index=False,
    )

    df_removed: pd.DataFrame = df.drop(df_keep.index)

    df_keep = _check_empty_dataframe(
        df=df_keep,
        columns=[users_column, items_column, feature_column]
    )

    df_removed = _check_empty_dataframe(
        df=df_removed,
        columns=[users_column, items_column, feature_column]
    )

    return df_keep, df_removed


@log_calling_args
def extract_timestamp_user_item(
    df: pd.DataFrame,
    users_column: str,
    items_column: str,
    timestamp_column: str,
    to_keep: T_KEEP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_column = f"feature-{users_column}-{items_column}-timestamp"

    df = _preprocess_df_for_feature_extraction(
        df=df,
        columns_to_keep=[users_column, items_column, timestamp_column],
        items_column=items_column,
    )

    # Get the timestamp as the UNIX epoch measured in seconds (// 10**9 part) or if its already an integer, then use it.
    if _is_column_a_datetime(df=df, column=timestamp_column):
        df[feature_column] = df[timestamp_column].astype(np.int64) / 10**9
    else:
        df[feature_column] = df[timestamp_column].copy()

    df_keep: pd.DataFrame = df.drop_duplicates(
        subset=[users_column, items_column],
        keep=to_keep,
        inplace=False,
        ignore_index=False,
    )

    df_removed: pd.DataFrame = df.drop(
        df_keep.index,
    )

    df_keep = _check_empty_dataframe(
        df=df_keep,
        columns=[users_column, items_column, feature_column]
    )

    df_removed = _check_empty_dataframe(
        df=df_removed,
        columns=[users_column, items_column, feature_column]
    )

    return df_keep, df_removed


def _preprocess_df_for_feature_extraction(
    df: pd.DataFrame,
    columns_to_keep: list[str],
    items_column: str,
) -> pd.DataFrame:
    # Just keep a copy of the columns_to_keep we're interested in.
    assert len(columns_to_keep) > 0

    df = df[columns_to_keep]

    # Explode item lists so rows can be inserted faster. This may blow up memory.
    is_items_column_object = _is_column_an_object(df=df, column=items_column)
    if is_items_column_object:
        # Warning, in pandas versions <1.3 we cannot use lists of columns_to_keep.
        df = df.explode(
            column=items_column,
            ignore_index=False,
        )

    df = df.dropna(
        axis="index",
        inplace=False,
        how="any",
    ).reset_index(
        drop=True,
    )

    return df


def _is_column_an_object(
    df: pd.DataFrame,
    column: str,
) -> bool:
    return "object" == str(df[column].dtype)


def _is_column_a_datetime(
    df: Union[pd.DataFrame, pd.Series],
    column: str,
) -> bool:
    if isinstance(df, pd.Series):
        dtype = df.dtype
    else:
        dtype = df[column].dtype

    return "datetime64" in str(dtype)


def _check_empty_dataframe(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            data=None,
            columns=columns,
        )
    else:
        return df

