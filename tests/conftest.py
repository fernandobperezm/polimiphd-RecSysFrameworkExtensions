import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sp
from Recommenders.BaseRecommender import BaseRecommender

from sklearn.model_selection import train_test_split
from pytest import fixture
from faker import Faker

# ORIGINAL SEED IS 1234567890.
# Do not change seed unless you want to generate different fake data points. This is not
# recommended as all tests cases are done expecting the data created with this random seed.
seed = 1234567890

Faker.seed(seed)
fake = Faker()

rng = np.random.default_rng(seed=seed)

NUM_USERS = 1000
NUM_ITEMS = 700
NUM_INTERACTIONS = 1000

ALL_USER_IDS = np.arange(start=0, stop=NUM_USERS, step=1, dtype=np.int32)
ALL_ITEM_IDS = np.arange(start=0, stop=NUM_ITEMS, step=1, dtype=np.int32)

MIN_TIMESTAMP = datetime.datetime(year=2022, month=1, day=1, hour=0, minute=0, second=1)
MAX_TIMESTAMP = datetime.datetime(year=2022, month=1, day=2, hour=23, minute=59, second=59)

USER_IDS = rng.choice(
    a=ALL_USER_IDS, size=NUM_INTERACTIONS, replace=True, shuffle=True,
)
ITEM_IDS = rng.choice(
    a=ALL_ITEM_IDS, size=NUM_INTERACTIONS, replace=True, shuffle=True,
)
TIMESTAMPS = np.array(
    [
        fake.date_time_between(start_date=MIN_TIMESTAMP, end_date=MAX_TIMESTAMP)
        for _ in range(NUM_INTERACTIONS)
    ],
    dtype=object
)


@fixture
def df() -> pd.DataFrame:
    dataframe = pd.DataFrame(
        data={
            "user_id": USER_IDS,
            "item_id": ITEM_IDS,
            "timestamp": TIMESTAMPS,
        }
    ).sort_values(
        by="timestamp",
        ascending=True,
        inplace=False,
        ignore_index=True,
    )
    return dataframe


@fixture
def df_debug(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(
        subset="user_id",
        inplace=False,
        keep="last",
        ignore_index=False,
    ).sort_values(
        by="user_id",
        ascending=True,
        inplace=False,
        ignore_index=False,
    )
    return df


@fixture
def urm(df: pd.DataFrame) -> sp.csr_matrix:
    df_data = df.drop_duplicates(
        subset=["user_id", "item_id"],
        keep="first",
        inplace=False,
    )

    arr_user_id = df_data["user_id"].to_numpy()
    arr_item_id = df_data["item_id"].to_numpy()
    arr_data = np.ones_like(arr_item_id, dtype=np.int32)

    return sp.csr_matrix(
        (
            arr_data,
            (arr_user_id, arr_item_id),
        ),
        shape=(NUM_USERS, NUM_ITEMS),
    )


@fixture
def splits(urm: sp.csr_matrix) -> tuple[sp.csr_matrix, ...]:
    train: sp.csr_matrix
    validation: sp.csr_matrix
    train_validation: sp.csr_matrix
    test: sp.csr_matrix

    train_validation, test = train_test_split(
        urm,
        test_size=0.2,
        random_state=1234,
        shuffle=True,
    )

    train, validation = train_test_split(
        train_validation,
        test_size=0.2,
        random_state=1234,
        shuffle=True,
    )

    return train, validation, train_validation, test


@fixture
def recommender(splits: list[sp.csr_matrix]) -> BaseRecommender:
    from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

    recommender = ItemKNNCFRecommender(
        URM_train=splits[2],
    )
    recommender.fit(
        topK=3,
        shrink=2,
        similarity="cosine",
        normalize=True,
        feature_weighting="none",
        URM_bias=None,
    )

    return recommender
