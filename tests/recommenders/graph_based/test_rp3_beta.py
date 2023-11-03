import numpy as np
import scipy.sparse as sp
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender

from recsys_framework_extensions.recommenders.graph_based.rp3_beta import (
    ExtendedRP3BetaRecommender,
)


class TestExtendedRP3BetaRecommender:
    def test_same_results_on_old_and_extended_version(
        self,
        urm: sp.csr_matrix,
        num_users: int,
        num_items: int,
    ):
        # Arrange
        test_top_k = 50
        test_alpha = 1.5
        test_beta = 0.5
        test_normalize_similarity = False

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
        recommender_framework = RP3betaRecommender(
            URM_train=urm,
        )
        recommender_extended = ExtendedRP3BetaRecommender(
            urm_train=urm,
        )

        recommender_framework.fit(
            alpha=test_alpha,
            beta=test_beta,
            topK=test_top_k,
            normalize_similarity=test_normalize_similarity,
        )
        recommender_extended.fit(
            alpha=test_alpha,
            beta=test_beta,
            top_k=test_top_k,
            normalize_similarity=test_normalize_similarity,
        )

        w_sparse_framework = recommender_framework.W_sparse.copy()
        w_sparse_extended = recommender_extended.W_sparse.copy()

        score_framework: np.ndarray = recommender_framework._compute_item_score(
            user_id_array=test_users_array,
            items_to_compute=test_items_array,
        )
        score_extended: np.ndarray = recommender_extended._compute_item_score(
            user_id_array=test_users_array,
            items_to_compute=test_items_array,
        )

        score_framework = score_framework.astype(np.float32)
        score_extended = score_extended.astype(np.float32)

        # Assert
        assert (w_sparse_framework != w_sparse_extended).nnz == 0
        assert np.array_equal(
            score_framework,
            score_extended,
            equal_nan=True,
        )
