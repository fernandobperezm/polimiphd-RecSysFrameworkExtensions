from typing import Any, Optional

import scipy.sparse as sp
import pandas as pd

from recsys_framework_extensions.data.mixins import (
    ImpressionsMixin,
    InteractionsMixin,
    PandasDataFramesMixin,
    LazyInteractionsMixin,
    LazyImpressionsMixin,
    LazyPandasDataFramesMixin,
)


class BaseDataset(LazyImpressionsMixin, LazyInteractionsMixin, LazyPandasDataFramesMixin):
    def __init__(
        self,
        dataset_name: str = None,
        mapper_item_original_id_to_index: dict[Any, int] = None,
        mapper_user_original_id_to_index: dict[Any, int] = None,
        impressions: dict[str, sp.csr_matrix] = None,
        interactions: dict[str, sp.csr_matrix] = None,
        dataframes: dict[str, pd.DataFrame] = None,
        is_impressions_implicit: bool = None,
        is_interactions_implicit: bool = None,
    ):
        if dataset_name is not None:
            self.dataset_name = dataset_name

        if mapper_item_original_id_to_index is not None:
            self.mapper_item_original_id_to_index = mapper_item_original_id_to_index

        if mapper_user_original_id_to_index is not None:
            self.mapper_user_original_id_to_index = mapper_user_original_id_to_index

        if impressions is not None:
            self.impressions = impressions

        if interactions is not None:
            self.interactions = interactions

        if dataframes is not None:
            self.dataframes = dataframes

        if is_impressions_implicit is not None:
            self.is_impressions_implicit = is_impressions_implicit

        if is_interactions_implicit is not None:
            self.is_interactions_implicit = is_interactions_implicit
