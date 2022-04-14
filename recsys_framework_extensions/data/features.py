import datetime
import numpy as np
import pandas as pd

from tqdm import tqdm

from typing import cast, Literal, Any, Iterable, TypeVar, Optional, Union
from itertools import tee, chain


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
    timestamps = series_group.to_list()
    iter_timestamps = iterator_previous_and_current(timestamps)

    diff_per_group = []

    tup: tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]
    for prev_timestamp, current_timestamp in iter_timestamps:
        if prev_timestamp is None:
            tup = (None, None, None, None, None, None)

        else:
            diff = (current_timestamp - prev_timestamp)

            if isinstance(diff, datetime.timedelta):
                total_seconds = int(diff.total_seconds())

                calculator_minutes = 60
                total_minutes = int(total_seconds // calculator_minutes)

                calculator_hours = 60 * calculator_minutes
                total_hours = int(total_seconds // calculator_hours)

                calculator_days = 24 * calculator_hours
                total_days = int(total_seconds // calculator_days)

                calculator_weeks = 7 * calculator_days
                total_weeks = int(diff.total_seconds() // calculator_weeks)

                tup = (None, total_seconds, total_minutes, total_hours, total_days, total_weeks)
            else:
                tup = (None, None, None, None, None)

        diff_per_group.append(tup)

    assert len(diff_per_group) == len(timestamps)

    return pd.DataFrame.from_records(
        data=diff_per_group,
        columns=["diff_euclidean", "total_seconds", "total_minutes", "total_hours", "total_days", "total_weeks"],
        index=series_group.index,
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
    assert "datetime" not in str()

    timestamps = series_group.to_list()
    iter_timestamps = iterator_previous_and_current(timestamps)

    diff_per_group = []

    tup: tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]
    for prev_timestamp, current_timestamp in iter_timestamps:
        if prev_timestamp is None:
            diff = None

        else:
            diff = (current_timestamp - prev_timestamp)

            if isinstance(diff, datetime.timedelta):
                total_seconds = int(diff.total_seconds())

                calculator_minutes = 60
                total_minutes = int(total_seconds // calculator_minutes)

                calculator_hours = 60 * calculator_minutes
                total_hours = int(total_seconds // calculator_hours)

                calculator_days = 24 * calculator_hours
                total_days = int(total_seconds // calculator_days)

                calculator_weeks = 7 * calculator_days
                total_weeks = int(diff.total_seconds() // calculator_weeks)

                tup = (None, total_seconds, total_minutes, total_hours, total_days, total_weeks)
            else:
                tup = (None, None, None, None, None)

        diff_per_group.append(tup)

    assert len(diff_per_group) == len(timestamps)

    return pd.DataFrame.from_records(
        data=diff_per_group,
        columns=[
            "diff_euclidean",
            "total_seconds",
            "total_minutes",
            "total_hours",
            "total_days",
            "total_weeks"
        ],
        index=series_group.index,
    )


def _compute_last_seen_per_user_item_group(
    series_group: pd.Series,
    granularity: T_LAST_SEEN_GRANULARITY,
) -> list:
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
    func_granularity = _dict_functions[granularity]

    timestamps = series_group.to_list()
    iter_timestamps = iterator_previous_and_current(timestamps)

    last_seen_per_group = []

    for prev_timestamp, current_timestamp in iter_timestamps:
        if prev_timestamp is None:
            last_seen = None

        else:
            diff = (current_timestamp - prev_timestamp)
            last_seen = func_granularity(diff)

        last_seen_per_group.append(last_seen)

    assert len(last_seen_per_group) == len(timestamps)

    return last_seen_per_group


def extract_frequency_user_item(
    df: pd.DataFrame,
    users_column: str,
    items_column: str,
) -> pd.DataFrame:
    # Just keep a copy of the columns_to_keep we're interested in.
    df = df[[users_column, items_column]]

    # Explode item lists so rows can be inserted faster. This may blow up memory.
    if "object" == str(df[items_column].dtype):
        # Warning, in pandas versions <1.3 we cannot use lists of columns_to_keep.
        df = df.explode(
            column=items_column,
            ignore_index=False,
        )

    df = df.dropna(
        axis="index",
        inplace=False,
        how="any",
    )

    df = cast(pd.DataFrame, df)

    # This creates a dataframe with columns_to_keep: users_column|items_column|frequency
    df_user_item_frequency = df.value_counts(
        subset=[users_column, items_column],
        normalize=False,
        sort=False,
        ascending=False,
    ).to_frame(
        name="frequency",
    ).reset_index(
        drop=False,
    )

    print(df)
    df.info(memory_usage="deep", show_counts=True)

    print(df_user_item_frequency)
    df_user_item_frequency.info(memory_usage="deep", show_counts=True)

    return df_user_item_frequency


def extract_last_seen_user_item(
    df: pd.DataFrame,
    users_column: str,
    items_column: str,
    timestamp_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # TODO: fernando-debugger. user_id==422 and impressions==1443 have 8 interactions.
    #  p df[(df["user_id"] == 422) & (df["impressions"] == 1443)]

    # Just keep a copy of the columns_to_keep we're interested in.
    df = df[[users_column, items_column, timestamp_column]].head(10000)
    df = cast(pd.DataFrame, df)

    # Explode item lists so rows can be inserted faster. This may blow up memory.
    if "object" == str(df[items_column].dtype):
        # Warning, in pandas versions <1.3 we cannot use lists of columns_to_keep.
        df = df.explode(
            column=items_column,
            ignore_index=False,
        )

    df = df.reset_index(
        drop=False,
    )

    df = df.dropna(
        axis="index",
        inplace=False,
        how="any",
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
        lambda df_group: len(df_group["timestamp"]) >= 2
    )

    df_grouped = df.groupby(
        by=[users_column, items_column],
        as_index=False,
        sort=False,
        observed=True,  # Avoid computing this on categorical.
        dropna=True,
    )

    df_last_seen_components: pd.DataFrame = df_grouped["timestamp"].progress_apply(
        func=_compute_diff_timestamps_per_user_item_group,
    )

    # df_last_seen_components: pd.DataFrame = df_grouped["timestamp"].progress_transform(
    #     func=_compute_diff_timestamps_per_user_item_group,
    # ).rename(
    #     columns_to_keep={
    #         "timestamp": "diff_timestamps"
    #     },
    # )

    # # Explode item lists so rows can be inserted faster. This may blow up memory.
    # if str(df[items_column].dtype).startswith("datetime64"):
    #     # Warning, in pandas versions <1.3 we cannot use lists of columns_to_keep.
    #     df = df.explode(
    #         column=items_column,
    #         ignore_index=False,
    #     )
    #
    #     ["diff_timestamps"].dt.components

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

    df_keep = df_keep.set_index("index")
    df_removed = df_removed.set_index("index")

    print(df)
    df.info(memory_usage="deep", show_counts=True)

    print(df_last_user_item_record)
    df_last_user_item_record.info(memory_usage="deep", show_counts=True)

    print(df_keep)
    df_keep.info(memory_usage="deep", show_counts=True)

    print(df_removed)
    df_removed.info(memory_usage="deep", show_counts=True)

    return df_keep, df_removed

    # df_last_seen_last_two = df.groupby(
    #     by=[users_column, items_column],
    #     as_index=False,
    #     sort=False,
    #     observed=True,  # Avoid computing this on categorical.
    #     dropna=True,
    # ).nth(
    #     n=[-2, -1],
    # )

    # transform_df_grouped = df_grouped.progress_transform(
    #     func=_compute_last_seen_per_user_item_group
    # )
    #
    # apply_series_grouped_computed: pd.TimedeltaIndex = df_grouped.progress_apply(
    #     func=_compute_last_seen_per_user_item_group
    # )
    #
    # aggregate_df_grouped = df_grouped.progress_aggregate(
    #     agg_last_seen=pd.NamedAgg(
    #         column="timestamp",
    #         aggfunc=_compute_last_seen_per_user_item_group
    #     ),
    # )


def extract_position_user_item(
    df: pd.DataFrame,
    users_column: str,
    items_column: str,
    to_keep: T_KEEP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert _is_column_an_object(df=df, column=items_column)

    # To extract positions, we first need a list of positions for each
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
    df["feature_position"] = df[items_column].str.len().progress_map(np.arange)

    # Pandas version <1.3 there is no multi-column explode. Therefore, we must apply the `explode` on the two columns
    # we're interested: `items_column` and `position`, as both are lists and need to be exploded together.
    df = df.set_index(
        [users_column]
    ).apply(
        pd.Series.explode,
    ).reset_index(
        # Returns the `user_id` column and creates a new integer-based index with unique values for each row.
        drop=False,
    )

    df_keep: pd.DataFrame = df.drop_duplicates(
        subset=[users_column, items_column],
        keep=to_keep,
        inplace=False,
        ignore_index=False,
    )

    df_removed: pd.DataFrame = df.drop(df_keep.index)

    return df_keep, df_removed


def extract_timestamp_user_item(
    df: pd.DataFrame,
    users_column: str,
    items_column: str,
    timestamp_column: str,
    to_keep: T_KEEP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import pdb
    pdb.set_trace()

    df = _preprocess_df_for_feature_extraction(
        df=df,
        columns_to_keep=[users_column, items_column, timestamp_column],
        items_column=items_column,
    )

    # Get the timestamp as the UNIX epoch measured in seconds (// 10**9 part) or if its already an integer, then use it.
    if _is_column_a_datetime(df=df, column=timestamp_column):
        df["feature_timestamp"] = df[timestamp_column].astype(np.int64) // 10**9
    else:
        df["feature_timestamp"] = df[timestamp_column].copy()

    df_keep: pd.DataFrame = df.drop_duplicates(
        subset=[users_column, items_column],
        keep=to_keep,
        inplace=False,
        ignore_index=False,
    )

    df_removed: pd.DataFrame = df.drop(
        df_keep.index,
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
    df: pd.DataFrame,
    column: str,
) -> bool:
    return "datetime64" in str(df[column].dtype)
