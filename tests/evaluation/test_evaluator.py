from Recommenders.BaseRecommender import BaseRecommender
from scipy import sparse
import pandas as pd

from Evaluation.Evaluator import EvaluatorHoldout as RecSysFrameworkEvaluatorHoldout
from recsys_framework_extensions.evaluation.Evaluator import ExtendedEvaluatorHoldout as ExtendedEvaluatorHoldout
from recsys_framework_extensions.logging import get_logger

logger = get_logger(
    logger_name=__file__,
)


class TestEvaluator:
    def test_evaluator_old_and_new_are_equivalent(
        self,
        splits: list[sparse.csr_matrix],
        recommender: BaseRecommender,
    ):
        urm_train_validation = splits[2]
        urm_test = splits[-1]

        test_cutoffs = [1, 2, 3, 4, 5]
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
            print(col, df_results_old[col], df_results_new[col])
            pd.testing.assert_series_equal(
                df_results_old[col],
                df_results_new[col],
                check_exact=False,
                check_dtype=False,
            )
            logger.debug(
                f"Successfully validated the metric '{col}'"
            )

    import pytest
    @pytest.mark.skip
    def test_evaluator_aggregate_on_users_is_correct(
        self,
        splits: list[sparse.csr_matrix],
        recommender: BaseRecommender,
    ):
        urm_train_validation = splits[2]
        urm_test = splits[-1]

        test_cutoffs = [1, 2, 3, 4, 5]
        test_min_ratings_per_user = 1
        test_exclude_seen = True

        new_evaluator = ExtendedEvaluatorHoldout(
            urm_test=urm_test.copy(),
            urm_train=urm_train_validation.copy(),
            cutoff_list=test_cutoffs,
            min_ratings_per_user=test_min_ratings_per_user,
            exclude_seen=test_exclude_seen,
        )

        # Act
        df_results_new = new_evaluator.compute_mean_score_on_evaluated_users(
            recommender_object=recommender,
        )

        # Assert
        assert False

