import itertools
import os
from functools import partial

import numba as nb
import numpy as np
import pandas as pd
import recsys_framework_extensions.evaluation.statistics_tests as st_tests
import scipy.sparse as sp
from Evaluation.Evaluator import EvaluatorHoldout, EvaluatorMetrics
from Recommenders.BaseRecommender import BaseRecommender
from recsys_framework_extensions.data.mixins import ParquetDataMixin
from recsys_framework_extensions.evaluation.metrics import nb_precision, nb_ndcg, nb_recall
from recsys_framework_extensions.logging import get_logger

logger = get_logger(
    logger_name=__name__,
)


def evaluate_users(
    arr_recommended_items: np.ndarray,
    arr_user_indptr: np.ndarray,
    arr_relevant_items: np.ndarray,
    arr_relevance_scores: np.ndarray,
    arr_user_ids: np.ndarray,
    max_cutoff: int,
    cutoff: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_users = arr_user_ids.shape[0]

    assert (num_users, ) == arr_user_ids.shape
    assert (num_users, max_cutoff) == arr_recommended_items.shape
    assert arr_relevance_scores.shape == arr_relevant_items.shape

    arr_cutoff_precision = np.zeros_like(arr_user_ids, dtype=np.float64)
    arr_cutoff_recall = np.zeros_like(arr_user_ids, dtype=np.float64)
    arr_cutoff_ndcg = np.zeros_like(arr_user_ids, dtype=np.float64)

    for idx_user_id, test_user_id in enumerate(arr_user_ids):
        # Compute predictions for a batch of users using vectorization,
        # much more efficient than computing it one at a time
        user_recommended_items: np.ndarray = arr_recommended_items[test_user_id]

        user_profile_start = arr_user_indptr[test_user_id]
        user_profile_end = arr_user_indptr[test_user_id + 1]

        user_relevant_items = arr_relevant_items[user_profile_start:user_profile_end]
        user_relevance_scores = arr_relevance_scores[user_profile_start:user_profile_end]

        # Being the URM CSR, the indices are the non-zero column indexes
        user_is_recommended_item_relevant = np.asarray(
            [
                rec_item in user_relevant_items
                for rec_item in user_recommended_items
            ]
        )

        is_relevant_current_cutoff = user_is_recommended_item_relevant[:cutoff]
        recommended_items_current_cutoff = user_recommended_items[:cutoff]

        arr_cutoff_precision[idx_user_id] = nb_precision(
            is_relevant=is_relevant_current_cutoff
        )
        arr_cutoff_recall[idx_user_id] = nb_recall(
            is_relevant=is_relevant_current_cutoff,
            pos_items=user_relevant_items
        )
        arr_cutoff_ndcg[idx_user_id] = nb_ndcg(
            ranked_list=recommended_items_current_cutoff,
            pos_items=user_relevant_items,
            relevance=user_relevance_scores,
            at=cutoff
        )

    return (
        arr_cutoff_precision,
        arr_cutoff_recall,
        arr_cutoff_ndcg,
    )


nb_evaluate_users = nb.njit(evaluate_users)


nb_evaluate_users(
    arr_recommended_items=np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=np.int32),
    arr_user_indptr=np.array([0, 1], dtype=np.int32),
    arr_relevant_items=np.array([2, 2], dtype=np.int32),
    arr_relevance_scores=np.array([1, 1], dtype=np.int32),
    arr_user_ids=np.array([0, 1], dtype=np.int32),
    max_cutoff=5,
    cutoff=2,
)


class EvaluatorHoldoutToDisk(ParquetDataMixin, EvaluatorHoldout):
    """EvaluatorHoldoutToDisk

    """

    EVALUATOR_NAME = "EvaluatorHoldoutToDisk"

    def __init__(
        self,
        urm_test: sp.csr_matrix,
        cutoff_list: list[int],
        min_ratings_per_user: int = 1,
        exclude_seen: bool = True,
        diversity_object=None,
        ignore_items=None,
        ignore_users=None,
        verbose: bool = True,
    ):

        super().__init__(
            urm_test,
            cutoff_list=cutoff_list,
            diversity_object=diversity_object,
            min_ratings_per_user=min_ratings_per_user,
            exclude_seen=exclude_seen,
            ignore_items=ignore_items,
            ignore_users=ignore_users,
            verbose=verbose
        )

        self._str_cutoffs = [
            str(cutoff)
            for cutoff in self.cutoff_list
        ]

        self._cutoffs: list[int] = self.cutoff_list

        self._str_metrics: list[str] = [
            EvaluatorMetrics.PRECISION.value,
            EvaluatorMetrics.RECALL.value,
            EvaluatorMetrics.NDCG.value,
        ]
        self._metrics = [
            EvaluatorMetrics.PRECISION,
            EvaluatorMetrics.RECALL,
            EvaluatorMetrics.NDCG,
        ]

    def evaluate_recommender(
        self,
        recommender: BaseRecommender,
        recommender_name: str,
        folder_export_results: str,
    ) -> pd.DataFrame:

        file_path = os.path.join(
            folder_export_results, f"{recommender_name}_accuracy.parquet"
        )
        partial_evaluate_recommender = partial(
            self._evaluate_recommender,
            recommender=recommender
        )

        df_results = self.load_parquet(
            file_path=file_path,
            to_pandas_func=partial_evaluate_recommender,
        )

        return df_results

    def compute_recommender_confidence_intervals(
        self,
        recommender: BaseRecommender,
        recommender_name: str,
        folder_export_results: str,
    ) -> pd.DataFrame:
        file_path = os.path.join(
            folder_export_results, f"{recommender_name}_confidence_intervals.parquet"
        )
        partial_evaluate_recommender = partial(
            self._compute_recommender_confidence_intervals,
            recommender=recommender,
            recommender_name=recommender_name,
            folder_export_results=folder_export_results,
        )

        df_results = self.load_parquet(
            file_path=file_path,
            to_pandas_func=partial_evaluate_recommender,
        )

        return df_results

    def _evaluate_recommender(
        self,
        recommender: BaseRecommender,
    ) -> pd.DataFrame:
        if self.ignore_items_flag:
            recommender.set_items_to_ignore(
                items_to_ignore=self.ignore_items_ID
            )

        arr_user_ids = np.asarray(
            self.users_to_evaluate,
            dtype=np.int32,
        )

        recommended_items: list[list[int]] = recommender.recommend(
            user_id_array=arr_user_ids,
            remove_seen_flag=self.exclude_seen,
            cutoff=self.max_cutoff,
            remove_top_pop_flag=False,
            remove_custom_items_flag=self.ignore_items_flag,
            return_scores=False,
        )

        arr_recommended_items = np.asarray(
            recommended_items,
            dtype=np.int32,
        )

        index = pd.MultiIndex.from_product(
            iterables=[arr_user_ids, [recommender.RECOMMENDER_NAME]],
            names=["user_id", "recommender"],
        )

        columns = pd.MultiIndex.from_product(
            iterables=[self._str_cutoffs, self._str_metrics],
            names=["cutoff", "metric"],
        )

        df_results = pd.DataFrame(
            data=None,
            index=index,
            columns=columns,
        )
        for cutoff in self.cutoff_list:
            (
                arr_cutoff_precision,
                arr_cutoff_recall,
                arr_cutoff_ndcg,
            ) = nb_evaluate_users(
                arr_recommended_items=arr_recommended_items,
                arr_user_indptr=self.URM_test.indptr,
                arr_relevant_items=self.URM_test.indices,
                arr_relevance_scores=self.URM_test.data,
                arr_user_ids=arr_user_ids,
                max_cutoff=self.max_cutoff,
                cutoff=cutoff
            )

            df_results[(str(cutoff), EvaluatorMetrics.PRECISION.value)] = arr_cutoff_precision
            df_results[(str(cutoff), EvaluatorMetrics.RECALL.value)] = arr_cutoff_recall
            df_results[(str(cutoff), EvaluatorMetrics.NDCG.value)] = arr_cutoff_ndcg

        if self.ignore_items_flag:
            recommender.reset_items_to_ignore()

        return df_results

    def _compute_recommender_confidence_intervals(
        self,
        recommender: BaseRecommender,
        recommender_name: str,
        folder_export_results: str,
    ) -> pd.DataFrame:
        df_scores = self.evaluate_recommender(
            recommender=recommender,
            recommender_name=recommender_name,
            folder_export_results=folder_export_results,
        )

        cutoffs = self._str_cutoffs
        metrics = self._str_metrics
        stats = ["mean", "std", "var"]
        algorithms = ["t-test", "normal"]
        ci_values = ["lower", "upper"]
        p_values = [0.1, 0.05, 0.01, 0.001]
        str_p_values = [str(p_value) for p_value in p_values]

        columns = [("recommender", "", "", "", "")]
        for cutoff, metric in itertools.product(cutoffs, metrics):
            columns += itertools.product([cutoff], [metric], stats, [""], [""])
            columns += itertools.product([cutoff], [metric], algorithms, str_p_values, ci_values)

        data: dict[tuple, list] = {
            col: []
            for col in columns
        }
        data[("recommender", "", "", "", "")].append(recommender_name)
        for cutoff in self._str_cutoffs:
            for metric in self._str_metrics:
                scores = df_scores[(cutoff, metric)].to_numpy(dtype=np.float64)

                data[(cutoff, metric, "mean", "", "")].append(scores.mean(dtype=np.float64))
                data[(cutoff, metric, "std", "", "")].append(scores.std(dtype=np.float64))
                data[(cutoff, metric, "var", "", "")].append(scores.var(dtype=np.float64))

                for p_value in p_values:
                    recommender_confidence_intervals = st_tests.calculate_confidence_intervals_on_scores_mean(
                        scores=scores,
                        p_value=p_value,
                    )

                    for computed_ci in recommender_confidence_intervals.confidence_intervals:
                        data[(cutoff, metric, computed_ci.algorithm, str(p_value), "lower")].append(computed_ci.lower)
                        data[(cutoff, metric, computed_ci.algorithm, str(p_value), "upper")].append(computed_ci.upper)

        mi_columns = pd.MultiIndex.from_tuples(columns)

        df_results = pd.DataFrame(
            data=data,
            columns=mi_columns,
        )
        return df_results
