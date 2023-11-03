import numpy as np
import scipy.sparse as sp
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender

from recsys_framework_extensions.recommenders.graph_based.p3_alpha import (
    ExtendedP3AlphaRecommender,
)


class TestExtendedP3AlphaRecommender:
    def test_same_results_on_old_and_extended_version(
        self,
        urm: sp.csr_matrix,
        num_users: int,
        num_items: int,
    ):
        # Arrange
        test_top_k = 50
        test_alpha = 1.5
        test_normalize_similarity = True

        test_users_array = np.arange(
            start=0,
            stop=num_users,
            dtype=np.int32,
        )
        test_items_array = np.arange(
            start=0,
            stop=num_items,
            dtype=np.int32,
        )

        # Act
        recommender_framework = P3alphaRecommender(
            URM_train=urm,
        )
        recommender_extended = ExtendedP3AlphaRecommender(
            urm_train=urm,
        )

        recommender_framework.fit(
            alpha=test_alpha,
            topK=test_top_k,
            normalize_similarity=test_normalize_similarity,
        )
        recommender_extended.fit(
            alpha=test_alpha,
            top_k=test_top_k,
            normalize_similarity=test_normalize_similarity,
        )

        w_sparse_framework: sp.csr_matrix = (
            recommender_framework.W_sparse.copy().astype(np.float32)
        )
        w_sparse_extended: sp.csr_matrix = recommender_extended.W_sparse.copy().astype(
            np.float32
        )

        score_framework = recommender_framework._compute_item_score(
            user_id_array=test_users_array,
            items_to_compute=test_items_array,
        )
        score_extended = recommender_extended._compute_item_score(
            user_id_array=test_users_array,
            items_to_compute=test_items_array,
        )

        score_framework = score_framework.astype(np.float64)
        score_extended = score_extended.astype(np.float64)

        # Assert
        assert (w_sparse_framework != w_sparse_extended).nnz == 0
        assert np.array_equal(
            score_framework,
            score_extended,
            equal_nan=True,
        )
