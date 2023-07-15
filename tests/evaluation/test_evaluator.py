import numpy as np
import pytest
from Recommenders.BaseRecommender import BaseRecommender
from scipy import sparse

from Evaluation.Evaluator import EvaluatorHoldout as RecSysFrameworkEvaluatorHoldout
from recsys_framework_extensions.evaluation.Evaluator import (
    ExtendedEvaluatorHoldout as ExtendedEvaluatorHoldout,
)
import logging

logger = logging.getLogger(__name__)


class TestEvaluator:
    def test_evaluator_old_and_new_are_equivalent(
        self,
        splits: list[sparse.csr_matrix],
        recommender: BaseRecommender,
    ):
        urm_train_validation = splits[2]
        urm_test = splits[-1]

        test_cutoffs = [1, 2, 3, 4, 5, 10, 20, 30, 50, 100]
        test_min_ratings_per_user = 1
        test_exclude_seen = True

        old_evaluator = RecSysFrameworkEvaluatorHoldout(
            URM_test_list=urm_test.copy(),
            cutoff_list=test_cutoffs,
            min_ratings_per_user=test_min_ratings_per_user,
            exclude_seen=test_exclude_seen,
        )

        new_evaluator = ExtendedEvaluatorHoldout(
            urm_test=urm_test.copy(),
            urm_train=urm_train_validation.copy(),
            cutoff_list=test_cutoffs,
            min_ratings_per_user=test_min_ratings_per_user,
            exclude_seen=test_exclude_seen,
        )

        # Act
        df_results_old, str_results_old = old_evaluator.evaluateRecommender(
            recommender_object=recommender,
        )
        df_results_new, str_results_new = new_evaluator.evaluateRecommender(
            recommender_object=recommender,
        )

        # Assert
        for col in df_results_new.columns:
            assert np.allclose(
                df_results_old[col].to_numpy(dtype=np.float32),
                df_results_new[col].to_numpy(dtype=np.float32),
                equal_nan=True,
            )
            assert np.allclose(
                df_results_new[col].to_numpy(dtype=np.float32),
                df_results_old[col].to_numpy(dtype=np.float32),
                equal_nan=True,
            )
            logger.debug(f"Successfully validated the metric '{col}'")
