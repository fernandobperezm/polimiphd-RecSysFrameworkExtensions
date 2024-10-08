import itertools
import os

from enum import Enum
from functools import partial
from typing import Sequence, Union

import numpy as np
import pandas as pd
import recsys_framework_extensions.evaluation.statistics_tests as st_tests
import scipy.sparse as sp
from Evaluation.Evaluator import (
    EvaluatorHoldout,
    EvaluatorMetrics,
    get_result_string_df,
)
from Recommenders.BaseRecommender import BaseRecommender
from recsys_framework_extensions.data.mixins import ParquetDataMixin, NumpyDictDataMixin
from recsys_framework_extensions.decorators import timeit
from recsys_framework_extensions.evaluation.loops import (
    evaluate_loop,
    count_recommended_items_loop,
)
import recsys_framework_extensions.evaluation.metric.nb_impl as metrics
import logging
from tqdm import tqdm


logger = logging.getLogger(__name__)


class ExtendedEvaluatorMetrics(Enum):
    POSITION_FIRST_RELEVANT = "POSITION_FIRST_RELEVANT"


class ExtendedEvaluatorHoldout(EvaluatorHoldout, ParquetDataMixin, NumpyDictDataMixin):
    """ExtendedEvaluatorHoldout"""

    EVALUATOR_NAME = "ExtendedEvaluatorHoldout"

    def __init__(
        self,
        urm_test: sp.csr_matrix,
        urm_train: sp.csr_matrix,
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
            min_ratings_per_user=min_ratings_per_user,
            diversity_object=diversity_object,
            exclude_seen=exclude_seen,
            ignore_items=ignore_items,
            ignore_users=ignore_users,
            verbose=verbose,
        )
        self.urm_train = urm_train

        self._cutoffs: list[int] = self.cutoff_list
        self._str_cutoffs = [str(cutoff) for cutoff in self._cutoffs]

        self._metrics = [
            EvaluatorMetrics.PRECISION,
            EvaluatorMetrics.RECALL,
            EvaluatorMetrics.MAP,
            EvaluatorMetrics.NDCG,
            EvaluatorMetrics.MRR,
            EvaluatorMetrics.HIT_RATE,
            EvaluatorMetrics.ARHR,
            EvaluatorMetrics.F1,
            EvaluatorMetrics.COVERAGE_USER,
            EvaluatorMetrics.COVERAGE_USER_HIT,
            EvaluatorMetrics.COVERAGE_ITEM,
            EvaluatorMetrics.COVERAGE_ITEM_HIT,
            # EvaluatorMetrics.USERS_IN_GT,
            # EvaluatorMetrics.ITEMS_IN_GT,
            EvaluatorMetrics.NOVELTY,
            EvaluatorMetrics.RATIO_NOVELTY,
            EvaluatorMetrics.DIVERSITY_GINI,
            EvaluatorMetrics.RATIO_DIVERSITY_GINI,
            EvaluatorMetrics.DIVERSITY_HERFINDAHL,
            EvaluatorMetrics.RATIO_DIVERSITY_HERFINDAHL,
            EvaluatorMetrics.SHANNON_ENTROPY,
            EvaluatorMetrics.RATIO_SHANNON_ENTROPY,
            # ExtendedEvaluatorMetrics.POSITION_FIRST_RELEVANT,
        ]
        # Only mean-based metrics can be statistically compared.
        self._metrics_statistical_tests = [
            EvaluatorMetrics.PRECISION,
            EvaluatorMetrics.RECALL,
            EvaluatorMetrics.MAP,
            EvaluatorMetrics.NDCG,
            EvaluatorMetrics.MRR,
            EvaluatorMetrics.HIT_RATE,
            EvaluatorMetrics.ARHR,
            EvaluatorMetrics.F1,
        ]
        self._str_metrics = [str(metric.value) for metric in self._metrics]
        self._str_metrics_statistical_tests = [
            str(metric.value) for metric in self._metrics_statistical_tests
        ]

    def compute_mean_score_on_evaluated_users(
        self,
        recommender_object: BaseRecommender,
    ) -> pd.DataFrame:
        (
            df_scores,
            dict_recommended_item_distribution,
            dict_relevant_recommended_item_distribution,
        ) = self._evaluate_recommender(
            recommender=recommender_object,
        )

        num_users_evaluated = df_scores.shape[0]
        if num_users_evaluated <= 0:
            raise ValueError("TODO: fernando-debugger complete")

        novelty_scores_train = metrics.nb_novelty_train(
            urm_train=self.urm_train,
        )

        df_mean_scores = df_scores.describe()

        for cutoff in self._str_cutoffs:
            (
                coverage_item,
                coverage_item_hit,
                diversity_gini,
                ratio_diversity_gini,
                diversity_herfindahl,
                ratio_diversity_herfindahl,
                shannon_entropy,
                ratio_shannon_entropy,
            ) = count_recommended_items_loop(
                arr_count_recommended_items=dict_recommended_item_distribution[cutoff],
                arr_count_relevant_recommended_items=dict_relevant_recommended_item_distribution[
                    cutoff
                ],
                arr_item_ids_to_ignore=self.ignore_items_ID,
                urm_train=self.urm_train,
            )

            df_mean_scores[(cutoff, EvaluatorMetrics.COVERAGE_ITEM.value)][
                "mean"
            ] = coverage_item
            df_mean_scores[(cutoff, EvaluatorMetrics.COVERAGE_ITEM_HIT.value)][
                "mean"
            ] = coverage_item_hit

            df_mean_scores[(cutoff, EvaluatorMetrics.DIVERSITY_GINI.value)][
                "mean"
            ] = diversity_gini
            df_mean_scores[(cutoff, EvaluatorMetrics.RATIO_DIVERSITY_GINI.value)][
                "mean"
            ] = ratio_diversity_gini

            df_mean_scores[(cutoff, EvaluatorMetrics.DIVERSITY_HERFINDAHL.value)][
                "mean"
            ] = diversity_herfindahl
            df_mean_scores[(cutoff, EvaluatorMetrics.RATIO_DIVERSITY_HERFINDAHL.value)][
                "mean"
            ] = ratio_diversity_herfindahl

            df_mean_scores[(cutoff, EvaluatorMetrics.SHANNON_ENTROPY.value)][
                "mean"
            ] = shannon_entropy
            df_mean_scores[(cutoff, EvaluatorMetrics.RATIO_SHANNON_ENTROPY.value)][
                "mean"
            ] = ratio_shannon_entropy

            # The original framework computes the F1 only on the mean precision and recall.
            df_mean_scores[(cutoff, EvaluatorMetrics.F1.value)][
                "mean"
            ] = metrics.nb_f1_score(
                score_precision=df_mean_scores[
                    (cutoff, EvaluatorMetrics.PRECISION.value)
                ]["mean"],
                score_recall=df_mean_scores[(cutoff, EvaluatorMetrics.RECALL.value)][
                    "mean"
                ],
            )

            df_mean_scores[(cutoff, EvaluatorMetrics.RATIO_NOVELTY.value)][
                "mean"
            ] = metrics.nb_ratio_recommendation_vs_train(
                metric_train=novelty_scores_train,
                metric_recommendations=df_mean_scores[
                    (cutoff, EvaluatorMetrics.NOVELTY.value)
                ]["mean"],
            )

            df_mean_scores[(cutoff, EvaluatorMetrics.COVERAGE_USER.value)][
                "mean"
            ] = metrics.nb_coverage_user_mean(
                arr_user_mask=df_scores[
                    (cutoff, EvaluatorMetrics.COVERAGE_USER.value)
                ].to_numpy(),
                arr_users_to_ignore=self.ignore_users_ID,
                num_total_users=self.n_users,
            )

            df_mean_scores[(cutoff, EvaluatorMetrics.COVERAGE_USER_HIT.value)][
                "mean"
            ] = metrics.nb_coverage_user_mean(
                arr_user_mask=df_scores[
                    (cutoff, EvaluatorMetrics.COVERAGE_USER_HIT.value)
                ].to_numpy(),
                arr_users_to_ignore=self.ignore_users_ID,
                num_total_users=self.n_users,
            )

        return df_mean_scores

    @timeit
    def evaluateRecommender(
        self,
        recommender_object: BaseRecommender,
    ):
        dict_results = self._create_empty_results_dict()

        df_mean_scores = self.compute_mean_score_on_evaluated_users(
            recommender_object=recommender_object,
        )

        for cutoff, metric in itertools.product(self._str_cutoffs, self._str_metrics):
            try:
                dict_results[int(cutoff)][metric] = df_mean_scores[(cutoff, metric)][
                    "mean"
                ]
            except Exception as e:
                print(cutoff, metric, df_mean_scores, dict_results)
                raise e

        df_results = pd.DataFrame.from_dict(
            data=dict_results,
            orient="index",
        )
        df_results.index.rename(name="cutoff", inplace=True)

        results_run_string = get_result_string_df(df_results)

        return df_results, results_run_string

    def evaluate_recommender(
        self,
        recommender: BaseRecommender,
        recommender_name: str,
        folder_export_results: str,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
        file_path_results_dataframe = os.path.join(
            folder_export_results, f"{recommender_name}_accuracy.parquet"
        )
        file_path_results_recommended_item_distribution = os.path.join(
            folder_export_results,
            f"{recommender_name}_recommended_item_distribution.npz",
        )
        file_path_results_relevant_recommended_item_distribution = os.path.join(
            folder_export_results,
            f"{recommender_name}_relevant_recommended_item_distribution.npz",
        )

        if all(
            os.path.exists(f)
            for f in [
                file_path_results_dataframe,
                file_path_results_recommended_item_distribution,
                file_path_results_relevant_recommended_item_distribution,
            ]
        ):
            # If files exists, just load them from disk.
            df_results = self.load_parquet(
                file_path=file_path_results_dataframe,
                to_pandas_func=lambda: pd.DataFrame(),
            )

            dict_recommended_item_distribution = self.load_dict_from_numpy(
                file_path=file_path_results_recommended_item_distribution,
                to_dict_func=lambda: {"a": np.array([])},
            )

            dict_relevant_recommended_item_distribution = self.load_dict_from_numpy(
                file_path=file_path_results_relevant_recommended_item_distribution,
                to_dict_func=lambda: {"a": np.array([])},
            )

        else:
            # Compute them and save them to disk.
            (
                df_results,
                dict_recommended_item_distribution,
                dict_relevant_recommended_item_distribution,
            ) = self._evaluate_recommender(
                recommender=recommender,
            )
            _ = self.load_parquet(
                file_path=file_path_results_dataframe,
                to_pandas_func=lambda: df_results,
            )
            _ = self.load_dict_from_numpy(
                file_path=file_path_results_recommended_item_distribution,
                to_dict_func=lambda: dict_recommended_item_distribution,
            )
            _ = self.load_dict_from_numpy(
                file_path=file_path_results_relevant_recommended_item_distribution,
                to_dict_func=lambda: dict_relevant_recommended_item_distribution,
            )

        return (
            df_results,
            dict_recommended_item_distribution,
            dict_relevant_recommended_item_distribution,
        )

    def _evaluate_recommender(
        self,
        recommender: BaseRecommender,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
        assert self.urm_train.shape == self.URM_test.shape

        if self.ignore_items_flag:
            recommender.set_items_to_ignore(items_to_ignore=self.ignore_items_ID)

        arr_user_ids = np.asarray(
            self.users_to_evaluate,
            dtype=np.int32,
        )

        # We will have in memory a matrix of shape=(num_users_to_evaluate, num_items_in_recommendations), but internally
        # we create a matrix with shape=(num_users_to_evaluate, num_items).
        # We want to keep the inner matrix using bytes=desired_batch_size_in_memory.
        # We will use more as we keep other data structures on memory.
        num_users, num_items = self.URM_test.shape
        desired_batch_size_in_memory = 1e9
        expected_batch_size_in_memory = num_users * num_items * 8

        if expected_batch_size_in_memory <= desired_batch_size_in_memory:
            num_users_in_batch = 1
        else:
            num_users_in_batch = int(desired_batch_size_in_memory // (num_items * 8))

        num_batches = int(num_users // num_users_in_batch)

        logger.debug(
            f"DEBUG-Evaluation of recommenders-Calculations batch size:"
            f"\n\t*{num_users=}"
            f"\n\t*{num_items=}"
            f"\n\t*{desired_batch_size_in_memory=}"
            f"\n\t*{expected_batch_size_in_memory=}"
            f"\n\t*{num_users_in_batch=}"
            f"\n\t*{num_batches=}"
        )

        arr_user_ids_batches = np.array_split(
            arr_user_ids,
            indices_or_sections=num_batches,
        )

        dict_cutoff_recommended_item_counters: dict[str, np.ndarray] = {
            cutoff: np.zeros(shape=(num_items,), dtype=np.int32)
            for cutoff in self._str_cutoffs
        }
        dict_cutoff_relevant_recommended_item_counters: dict[str, np.ndarray] = {
            cutoff: np.zeros(shape=(num_items,), dtype=np.int32)
            for cutoff in self._str_cutoffs
        }

        list_df_results_per_user_batch: list[pd.DataFrame] = []

        logger.info(f"Evaluating recommender.")
        for arr_batch_user_id in tqdm(arr_user_ids_batches):
            list_batch_recommended_items: list[list[int]] = recommender.recommend(
                arr_batch_user_id,
                remove_seen_flag=self.exclude_seen,
                cutoff=self.max_cutoff,
                remove_top_pop_flag=False,
                remove_custom_items_flag=self.ignore_items_flag,
                return_scores=False,
            )

            index = pd.MultiIndex.from_product(
                iterables=[arr_batch_user_id, [recommender.RECOMMENDER_NAME]],
                names=["user_id", "recommender"],
            )

            dict_results_in_batch: dict[tuple[str, str], Union[np.ndarray, float]] = {}

            for cutoff in self.cutoff_list:
                (
                    arr_cutoff_average_precision,
                    arr_cutoff_precision,
                    arr_cutoff_recall,
                    arr_cutoff_ndcg,
                    arr_cutoff_rr,
                    arr_cutoff_hit_rate,
                    arr_cutoff_arhr_all_hits,
                    arr_cutoff_f1_score,
                    arr_cutoff_novelty_score,
                    arr_cutoff_coverage_users,
                    arr_cutoff_coverage_users_hit,
                    arr_cutoff_position_first_relevant_item,
                    arr_count_recommended_items,
                    arr_count_relevant_recommended_items,
                ) = evaluate_loop(
                    urm_test=self.URM_test,
                    urm_train=self.urm_train,
                    list_batch_recommended_items=list_batch_recommended_items,
                    arr_batch_user_ids=arr_batch_user_id,
                    num_users=num_users,
                    num_items=num_items,
                    cutoff=cutoff,
                    max_cutoff=self.max_cutoff,
                )

                dict_results_in_batch.update(
                    {
                        (
                            str(cutoff),
                            EvaluatorMetrics.MAP.value,
                        ): arr_cutoff_average_precision,
                        (
                            str(cutoff),
                            EvaluatorMetrics.PRECISION.value,
                        ): arr_cutoff_precision,
                        (
                            str(cutoff),
                            EvaluatorMetrics.RECALL.value,
                        ): arr_cutoff_recall,
                        (
                            str(cutoff),
                            EvaluatorMetrics.NDCG.value,
                        ): arr_cutoff_ndcg,
                        (
                            str(cutoff),
                            EvaluatorMetrics.MRR.value,
                        ): arr_cutoff_rr,
                        (
                            str(cutoff),
                            EvaluatorMetrics.HIT_RATE.value,
                        ): arr_cutoff_hit_rate,
                        (
                            str(cutoff),
                            EvaluatorMetrics.ARHR.value,
                        ): arr_cutoff_arhr_all_hits,
                        (
                            str(cutoff),
                            EvaluatorMetrics.F1.value,
                        ): arr_cutoff_f1_score,
                        (
                            str(cutoff),
                            EvaluatorMetrics.NOVELTY.value,
                        ): arr_cutoff_novelty_score,
                        (
                            str(cutoff),
                            EvaluatorMetrics.COVERAGE_USER.value,
                        ): arr_cutoff_coverage_users,
                        (
                            str(cutoff),
                            EvaluatorMetrics.COVERAGE_USER_HIT.value,
                        ): arr_cutoff_coverage_users_hit,
                        # The following diversity metrics only make sense when computed on all users, here we're only using a placeholder.
                        (
                            str(cutoff),
                            EvaluatorMetrics.COVERAGE_ITEM.value,
                        ): 0.0,
                        (
                            str(cutoff),
                            EvaluatorMetrics.COVERAGE_ITEM_HIT.value,
                        ): 0.0,
                        (
                            str(cutoff),
                            EvaluatorMetrics.RATIO_NOVELTY.value,
                        ): 0.0,
                        (
                            str(cutoff),
                            EvaluatorMetrics.DIVERSITY_GINI.value,
                        ): 0.0,
                        (
                            str(cutoff),
                            EvaluatorMetrics.RATIO_DIVERSITY_GINI.value,
                        ): 0.0,
                        (
                            str(cutoff),
                            EvaluatorMetrics.DIVERSITY_HERFINDAHL.value,
                        ): 0.0,
                        (
                            str(cutoff),
                            EvaluatorMetrics.RATIO_DIVERSITY_HERFINDAHL.value,
                        ): 0.0,
                        (
                            str(cutoff),
                            EvaluatorMetrics.SHANNON_ENTROPY.value,
                        ): 0.0,
                        (
                            str(cutoff),
                            EvaluatorMetrics.RATIO_SHANNON_ENTROPY.value,
                        ): 0.0,
                        (
                            str(cutoff),
                            ExtendedEvaluatorMetrics.POSITION_FIRST_RELEVANT.value,
                        ): arr_cutoff_position_first_relevant_item,
                    }
                )

                dict_cutoff_recommended_item_counters[
                    str(cutoff)
                ] += arr_count_recommended_items
                dict_cutoff_relevant_recommended_item_counters[
                    str(cutoff)
                ] += arr_count_relevant_recommended_items

            if self.ignore_items_flag:
                recommender.reset_items_to_ignore()

            list_df_results_per_user_batch.append(
                pd.DataFrame(
                    data=dict_results_in_batch,
                    index=index,
                    dtype=np.float64,
                )
            )

        df_results = pd.concat(
            list_df_results_per_user_batch,
            axis="index",
        )

        return (
            df_results,
            dict_cutoff_recommended_item_counters,
            dict_cutoff_relevant_recommended_item_counters,
        )

    def compute_recommenders_statistical_tests(
        self,
        *,
        dataset: str,
        recommender_baseline: BaseRecommender,
        recommender_baseline_name: str,
        recommender_baseline_folder: str,
        recommender_others: Sequence[BaseRecommender],
        recommender_others_names: Sequence[str],
        recommender_others_folders: Sequence[str],
        folder_export_results: str,
    ) -> Sequence[pd.DataFrame]:
        file_paths = [
            os.path.join(
                folder_export_results,
                f"{recommender_baseline_name}_groupwise_statistical_tests.parquet",
            ),
            os.path.join(
                folder_export_results,
                f"{recommender_baseline_name}_pairwise_statistical_tests.parquet",
            ),
        ]

        partial_compute_recommenders_statistical_tests = partial(
            self._compute_recommenders_statistical_tests,
            dataset=dataset,
            recommender_baseline=recommender_baseline,
            recommender_baseline_name=recommender_baseline_name,
            recommender_baseline_folder=recommender_baseline_folder,
            recommender_others=recommender_others,
            recommender_others_names=recommender_others_names,
            recommender_others_folders=recommender_others_folders,
        )

        df_results_groupwise, df_results_pairwise = self.load_parquets(
            file_paths=file_paths,
            to_pandas_func=partial_compute_recommenders_statistical_tests,
        )

        return df_results_groupwise, df_results_pairwise

    def _compute_recommenders_statistical_tests(
        self,
        *,
        dataset: str,
        recommender_baseline: BaseRecommender,
        recommender_baseline_name: str,
        recommender_baseline_folder: str,
        recommender_others: Sequence[BaseRecommender],
        recommender_others_names: Sequence[str],
        recommender_others_folders: Sequence[str],
    ) -> list[pd.DataFrame]:
        (
            df_scores_baseline,
            _,
            _,
        ) = self.evaluate_recommender(
            recommender=recommender_baseline,
            recommender_name=recommender_baseline_name,
            folder_export_results=recommender_baseline_folder,
        )

        list_df_scores_others = []
        for recommender, recommender_name, recommender_folder in zip(
            recommender_others,
            recommender_others_names,
            recommender_others_folders,
        ):
            logger.warning(
                "Evaluating recommender %(recommender_name)s",
                {"recommender_name": recommender_name},
            )
            results_recommender = self.evaluate_recommender(
                recommender=recommender,
                recommender_name=recommender_name,
                folder_export_results=recommender_folder,
            )[0]

            list_df_scores_others.append(results_recommender)

        # TODO: FERNANDO-DEBUGGER. Uncomment or remove.
        # list_df_scores_others = [
        #     self.evaluate_recommender(
        #         recommender=recommender,
        #         recommender_name=recommender_name,
        #         folder_export_results=recommender_folder,
        #     )[0]
        #     for recommender, recommender_name, recommender_folder in zip(
        #         recommender_others,
        #         recommender_others_names,
        #         recommender_others_folders,
        #     )
        # ]

        num_other_recommenders = len(list_df_scores_others)
        assert num_other_recommenders >= 1

        stats_groupwise = ["friedman"]
        stats_pairwise = [
            "paired_t_test",
            "wilcoxon",
            "wilcoxon_zsplit",
            "sign",
            "bonferroni-paired_t_test",
            "bonferroni-wilcoxon",
            "bonferroni-wilcoxon_zsplit",
            "bonferroni-sign",
        ]
        inner_stats_groupwise = ["num_measurements", "alpha", "p_value", "hypothesis"]
        inner_stats_pairwise = ["alpha", "p_value", "hypothesis"]

        alternatives = [
            st_tests.StatisticalTestAlternative.LESS,
            st_tests.StatisticalTestAlternative.TWO_SIDED,
            st_tests.StatisticalTestAlternative.GREATER,
        ]
        str_alternatives = [alternative.value for alternative in alternatives]
        alphas = [0.05]
        str_alphas = [str(alpha) for alpha in alphas]

        columns_groupwise = [
            ("dataset", "", "", "", "", ""),
            ("recommender", "", "", "", "", ""),
        ]
        columns_groupwise += itertools.product(
            self._str_cutoffs,
            self._str_metrics_statistical_tests,
            stats_groupwise,
            str_alternatives,
            str_alphas,
            inner_stats_groupwise,
        )

        columns_pairwise = [
            ("dataset", "", "", "", "", ""),
            ("recommender_base", "", "", "", "", ""),
            ("recommender_other", "", "", "", "", ""),
        ]
        columns_pairwise += itertools.product(
            self._str_cutoffs,
            self._str_metrics_statistical_tests,
            stats_pairwise,
            str_alternatives,
            str_alphas,
            inner_stats_pairwise,
        )

        data_groupwise: dict[tuple, list] = {col: [] for col in columns_groupwise}
        data_groupwise[("dataset", "", "", "", "", "")].append(dataset)
        data_groupwise[("recommender", "", "", "", "", "")].append(
            recommender_baseline_name
        )

        data_pairwise: dict[tuple, list] = {col: [] for col in columns_pairwise}
        data_pairwise[("dataset", "", "", "", "", "")] += [
            dataset
        ] * num_other_recommenders
        data_pairwise[("recommender_base", "", "", "", "", "")] += [
            recommender_baseline_name
        ] * num_other_recommenders
        data_pairwise[("recommender_other", "", "", "", "", "")] += [
            rec_name for rec_name in recommender_others_names
        ]

        for cutoff, metric in itertools.product(
            self._str_cutoffs, self._str_metrics_statistical_tests
        ):
            # np.asarray converts the array to shape (# recommenders, # users), the functions need
            # arrays of shape (# users, # recommenders). Hence, we transpose the scores.
            arr_scores_baseline = df_scores_baseline[(cutoff, metric)].to_numpy(
                dtype=np.float64
            )
            arr_scores_others = np.vstack(
                [
                    df[(cutoff, metric)].to_numpy(dtype=np.float64)
                    for df in list_df_scores_others
                ],
            )

            for alternative in alternatives:
                results_statistical_tests = (
                    st_tests.compute_statistical_tests_of_base_vs_others(
                        scores_base=arr_scores_baseline,
                        scores_others=arr_scores_others,
                        alphas=alphas,
                        alternative=alternative,
                    )
                )

                for idx, alpha in enumerate(alphas):
                    data_groupwise[
                        (
                            cutoff,
                            metric,
                            "friedman",
                            alternative.value,
                            str(alpha),
                            "alpha",
                        )
                    ].append(alpha)
                    data_groupwise[
                        (
                            cutoff,
                            metric,
                            "friedman",
                            alternative.value,
                            str(alpha),
                            "p_value",
                        )
                    ].append(results_statistical_tests.friedman.p_value)
                    data_groupwise[
                        (
                            cutoff,
                            metric,
                            "friedman",
                            alternative.value,
                            str(alpha),
                            "hypothesis",
                        )
                    ].append(results_statistical_tests.friedman.hypothesis[idx].value)
                    data_groupwise[
                        (
                            cutoff,
                            metric,
                            "friedman",
                            alternative.value,
                            str(alpha),
                            "num_measurements",
                        )
                    ].append(results_statistical_tests.friedman.num_measurements)

                    # PAIRED T-TEST
                    data_pairwise[
                        (
                            cutoff,
                            metric,
                            "paired_t_test",
                            alternative.value,
                            str(alpha),
                            "alpha",
                        )
                    ] += [alpha] * num_other_recommenders
                    data_pairwise[
                        (
                            cutoff,
                            metric,
                            "paired_t_test",
                            alternative.value,
                            str(alpha),
                            "p_value",
                        )
                    ] += [
                        res.p_value for res in results_statistical_tests.paired_t_test
                    ]
                    data_pairwise[
                        (
                            cutoff,
                            metric,
                            "paired_t_test",
                            alternative.value,
                            str(alpha),
                            "hypothesis",
                        )
                    ] += [
                        res.hypothesis[idx].value
                        for res in results_statistical_tests.paired_t_test
                    ]

                    # WILCOXON SIGNED-RANKS TEST - TIES HANDLED BY WILCOX
                    data_pairwise[
                        (
                            cutoff,
                            metric,
                            "wilcoxon",
                            alternative.value,
                            str(alpha),
                            "alpha",
                        )
                    ] += [alpha] * num_other_recommenders
                    data_pairwise[
                        (
                            cutoff,
                            metric,
                            "wilcoxon",
                            alternative.value,
                            str(alpha),
                            "p_value",
                        )
                    ] += [res.p_value for res in results_statistical_tests.wilcoxon]
                    data_pairwise[
                        (
                            cutoff,
                            metric,
                            "wilcoxon",
                            alternative.value,
                            str(alpha),
                            "hypothesis",
                        )
                    ] += [
                        res.hypothesis[idx].value
                        for res in results_statistical_tests.wilcoxon
                    ]

                    # WILCOXON SIGNED-RANKS TEST - TIES HANDLED BY ZSPLIT
                    data_pairwise[
                        (
                            cutoff,
                            metric,
                            "wilcoxon_zsplit",
                            alternative.value,
                            str(alpha),
                            "alpha",
                        )
                    ] += [alpha] * num_other_recommenders
                    data_pairwise[
                        (
                            cutoff,
                            metric,
                            "wilcoxon_zsplit",
                            alternative.value,
                            str(alpha),
                            "p_value",
                        )
                    ] += [
                        res.p_value for res in results_statistical_tests.wilcoxon_zsplit
                    ]
                    data_pairwise[
                        (
                            cutoff,
                            metric,
                            "wilcoxon_zsplit",
                            alternative.value,
                            str(alpha),
                            "hypothesis",
                        )
                    ] += [
                        res.hypothesis[idx].value
                        for res in results_statistical_tests.wilcoxon_zsplit
                    ]

                    # SIGN TEST
                    data_pairwise[
                        (cutoff, metric, "sign", alternative.value, str(alpha), "alpha")
                    ] += [alpha] * num_other_recommenders
                    data_pairwise[
                        (
                            cutoff,
                            metric,
                            "sign",
                            alternative.value,
                            str(alpha),
                            "p_value",
                        )
                    ] += [res.p_value for res in results_statistical_tests.sign]
                    data_pairwise[
                        (
                            cutoff,
                            metric,
                            "sign",
                            alternative.value,
                            str(alpha),
                            "hypothesis",
                        )
                    ] += [
                        res.hypothesis[idx].value
                        for res in results_statistical_tests.sign
                    ]

                    # BONFERRONI CORRECTION TO PAIRED T-TEST
                    data_pairwise[
                        (
                            cutoff,
                            metric,
                            "bonferroni-paired_t_test",
                            alternative.value,
                            str(alpha),
                            "alpha",
                        )
                    ] += [
                        results_statistical_tests.bonferroni_paired_t_test.corrected_alphas[
                            idx
                        ]
                    ] * num_other_recommenders
                    data_pairwise[
                        (
                            cutoff,
                            metric,
                            "bonferroni-paired_t_test",
                            alternative.value,
                            str(alpha),
                            "p_value",
                        )
                    ] += [
                        p_val
                        for p_val in results_statistical_tests.bonferroni_paired_t_test.corrected_p_values
                    ]
                    data_pairwise[
                        (
                            cutoff,
                            metric,
                            "bonferroni-paired_t_test",
                            alternative.value,
                            str(alpha),
                            "hypothesis",
                        )
                    ] += [
                        h[idx].value
                        for h in results_statistical_tests.bonferroni_paired_t_test.hypothesis
                    ]

                    # BONFERRONI CORRECTION TO WILCOXON SIGNED-RANKS TEST - TIES HANDLED BY WILCOX
                    data_pairwise[
                        (
                            cutoff,
                            metric,
                            "bonferroni-wilcoxon",
                            alternative.value,
                            str(alpha),
                            "alpha",
                        )
                    ] += [
                        results_statistical_tests.bonferroni_wilcoxon.corrected_alphas[
                            idx
                        ]
                    ] * num_other_recommenders
                    data_pairwise[
                        (
                            cutoff,
                            metric,
                            "bonferroni-wilcoxon",
                            alternative.value,
                            str(alpha),
                            "p_value",
                        )
                    ] += [
                        p_val
                        for p_val in results_statistical_tests.bonferroni_wilcoxon.corrected_p_values
                    ]
                    data_pairwise[
                        (
                            cutoff,
                            metric,
                            "bonferroni-wilcoxon",
                            alternative.value,
                            str(alpha),
                            "hypothesis",
                        )
                    ] += [
                        h[idx].value
                        for h in results_statistical_tests.bonferroni_wilcoxon.hypothesis
                    ]

                    # BONFERRONI CORRECTION TO WILCOXON SIGNED-RANKS TEST - TIES HANDLED BY ZSPLIT
                    data_pairwise[
                        (
                            cutoff,
                            metric,
                            "bonferroni-wilcoxon_zsplit",
                            alternative.value,
                            str(alpha),
                            "alpha",
                        )
                    ] += [
                        results_statistical_tests.bonferroni_wilcoxon_zsplit.corrected_alphas[
                            idx
                        ]
                    ] * num_other_recommenders
                    data_pairwise[
                        (
                            cutoff,
                            metric,
                            "bonferroni-wilcoxon_zsplit",
                            alternative.value,
                            str(alpha),
                            "p_value",
                        )
                    ] += [
                        p_val
                        for p_val in results_statistical_tests.bonferroni_wilcoxon_zsplit.corrected_p_values
                    ]
                    data_pairwise[
                        (
                            cutoff,
                            metric,
                            "bonferroni-wilcoxon_zsplit",
                            alternative.value,
                            str(alpha),
                            "hypothesis",
                        )
                    ] += [
                        h[idx].value
                        for h in results_statistical_tests.bonferroni_wilcoxon_zsplit.hypothesis
                    ]

                    # BONFERRONI CORRECTION TO SIGN TEST
                    data_pairwise[
                        (
                            cutoff,
                            metric,
                            "bonferroni-sign",
                            alternative.value,
                            str(alpha),
                            "alpha",
                        )
                    ] += [
                        results_statistical_tests.bonferroni_sign.corrected_alphas[idx]
                    ] * num_other_recommenders
                    data_pairwise[
                        (
                            cutoff,
                            metric,
                            "bonferroni-sign",
                            alternative.value,
                            str(alpha),
                            "p_value",
                        )
                    ] += [
                        p_val
                        for p_val in results_statistical_tests.bonferroni_sign.corrected_p_values
                    ]
                    data_pairwise[
                        (
                            cutoff,
                            metric,
                            "bonferroni-sign",
                            alternative.value,
                            str(alpha),
                            "hypothesis",
                        )
                    ] += [
                        h[idx].value
                        for h in results_statistical_tests.bonferroni_sign.hypothesis
                    ]

        mi_columns_groupwise = pd.MultiIndex.from_tuples(columns_groupwise)
        mi_columns_pairwise = pd.MultiIndex.from_tuples(columns_pairwise)

        df_results_groupwise = pd.DataFrame(
            data=data_groupwise,
            columns=mi_columns_groupwise,
        )

        df_results_pairwise = pd.DataFrame(
            data=data_pairwise,
            columns=mi_columns_pairwise,
        )

        return [
            df_results_groupwise,
            df_results_pairwise,
        ]

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

    def _compute_recommender_confidence_intervals(
        self,
        recommender: BaseRecommender,
        recommender_name: str,
        folder_export_results: str,
    ) -> pd.DataFrame:
        (
            df_scores,
            _,
            _,
        ) = self.evaluate_recommender(
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
            columns += itertools.product(
                [cutoff], [metric], algorithms, str_p_values, ci_values
            )

        data: dict[tuple, list] = {col: [] for col in columns}
        data[("recommender", "", "", "", "")].append(recommender_name)
        for cutoff, metric in itertools.product(self._str_cutoffs, self._str_metrics):
            scores = df_scores[(cutoff, metric)].to_numpy(dtype=np.float64)

            data[(cutoff, metric, "mean", "", "")].append(scores.mean(dtype=np.float64))
            data[(cutoff, metric, "std", "", "")].append(scores.std(dtype=np.float64))
            data[(cutoff, metric, "var", "", "")].append(scores.var(dtype=np.float64))

            for p_value in p_values:
                recommender_confidence_intervals = (
                    st_tests.calculate_confidence_intervals_on_scores_mean(
                        scores=scores,
                        alpha=p_value,
                    )
                )

                for (
                    computed_ci
                ) in recommender_confidence_intervals.confidence_intervals:
                    data[
                        (cutoff, metric, computed_ci.algorithm, str(p_value), "lower")
                    ].append(computed_ci.lower)
                    data[
                        (cutoff, metric, computed_ci.algorithm, str(p_value), "upper")
                    ].append(computed_ci.upper)

        mi_columns = pd.MultiIndex.from_tuples(columns)

        df_results = pd.DataFrame(
            data=data,
            columns=mi_columns,
        )
        return df_results

    def _create_empty_results_dict(self) -> dict[int, dict[str, float]]:
        return {
            cutoff: {metric: 0.0 for metric in self._str_metrics}
            for cutoff in self.cutoff_list
        }
