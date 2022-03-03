from typing import Any

import scipy.sparse as sp

from recsys_framework_extensions.data.mixins import ImpressionsMixin, InteractionsMixin


class BaseDataset(ImpressionsMixin, InteractionsMixin):
    def __init__(
        self,
        dataset_name: str,
        impressions: dict[str, sp.csr_matrix],
        interactions: dict[str, sp.csr_matrix],
        mapper_item_original_id_to_index: dict[Any, int],
        mapper_user_original_id_to_index: dict[Any, int],
        is_impressions_implicit: bool,
        is_interactions_implicit: bool,
    ):
        self.dataset_name = dataset_name
        self.impressions = impressions
        self.interactions = interactions
        self.mapper_item_original_id_to_index = mapper_item_original_id_to_index
        self.mapper_user_original_id_to_index = mapper_user_original_id_to_index
        self.is_impressions_implicit = is_impressions_implicit
        self.is_interactions_implicit = is_interactions_implicit

    @staticmethod
    def empty_dataset() -> "BaseDataset":
        return BaseDataset(
            dataset_name="",
            impressions=dict(),
            interactions=dict(),
            mapper_item_original_id_to_index=dict(),
            mapper_user_original_id_to_index=dict(),
            is_impressions_implicit=False,
            is_interactions_implicit=False,
        )


class BinaryImplicitDataset(BaseDataset):
    def __init__(
        self,
        dataset_name: str,
        impressions: dict[str, sp.csr_matrix],
        interactions: dict[str, sp.csr_matrix],
        mapper_item_original_id_to_index: dict[Any, int],
        mapper_user_original_id_to_index: dict[Any, int],
    ):
        super().__init__(
            dataset_name=dataset_name,
            impressions=impressions,
            interactions=interactions,
            mapper_item_original_id_to_index=mapper_item_original_id_to_index,
            mapper_user_original_id_to_index=mapper_user_original_id_to_index,
            is_impressions_implicit=True,
            is_interactions_implicit=True,
        )
