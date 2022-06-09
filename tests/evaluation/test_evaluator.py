from Recommenders.BaseRecommender import BaseRecommender
from scipy import sparse
import pandas as pd

from Evaluation.Evaluator import EvaluatorHoldout as RecSysFrameworkEvaluatorHoldout
from recsys_framework_extensions.evaluation.Evaluator import EvaluatorHoldoutToDisk as ExtendedEvaluatorHoldout


class TestEvaluator:
    def test_evaluator_old_and_new_are_equivalent(
        self,
        splits: list[sparse.csr_matrix],
        recommender: BaseRecommender,
    ):
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
            print(col)
            pd.testing.assert_series_equal(
                df_results_old[col],
                df_results_new[col],
                check_exact=False,
            )
