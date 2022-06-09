import itertools
import os
from functools import partial
from typing import Sequence

import numpy as np
import pandas as pd
import recsys_framework_extensions.evaluation.statistics_tests as st_tests
import scipy.sparse as sp
from Evaluation.Evaluator import EvaluatorHoldout, EvaluatorMetrics, get_result_string_df
from Recommenders.BaseRecommender import BaseRecommender
from recsys_framework_extensions.data.mixins import ParquetDataMixin
from recsys_framework_extensions.evaluation.loops import evaluate_loop
from recsys_framework_extensions.logging import get_logger
from tqdm import tqdm


logger = get_logger(
    logger_name=__name__,
)


class EvaluatorHoldoutToDisk(EvaluatorHoldout, ParquetDataMixin):
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
            min_ratings_per_user=min_ratings_per_user,
            diversity_object=diversity_object,
            exclude_seen=exclude_seen,
            ignore_items=ignore_items,
            ignore_users=ignore_users,
            verbose=verbose
        )

        self._cutoffs: list[int] = self.cutoff_list
        self._str_cutoffs = [
            str(cutoff)
            for cutoff in self._cutoffs
        ]
        self._str_cutoffs = ["10", "50"]

        self._metrics = [
            EvaluatorMetrics.MAP,
            EvaluatorMetrics.PRECISION,
            EvaluatorMetrics.RECALL,
            EvaluatorMetrics.NDCG,
            EvaluatorMetrics.MRR,
            EvaluatorMetrics.HIT_RATE,
            EvaluatorMetrics.ARHR,
            EvaluatorMetrics.F1,
        ]
        self._str_metrics: list[str] = [
            metric.value
            for metric in self._metrics
        ]

    def evaluateRecommender(
        self,
        recommender_object: BaseRecommender,
    ):
        df_scores = self._evaluate_recommender(
            recommender=recommender_object
        )
        num_users_evaluated = df_scores.shape[0]

        dict_results = self._create_empty_results_dict()

        if num_users_evaluated > 0:
            for cutoff in self._str_cutoffs:
                for metric in self._str_metrics:
                    # The original framework computes the F1 only on the mean precision and recall.
                    # Computing the f1 and then taking the mean is equivalent.
                    mean_cutoff_metric_score = df_scores[(cutoff, metric)].mean()

                    if EvaluatorMetrics.F1.value == metric:
                        precision = df_scores[(cutoff, EvaluatorMetrics.PRECISION.value)].mean()
                        recall = df_scores[(cutoff, EvaluatorMetrics.RECALL.value)].mean()

                        if np.isclose(precision + recall, 0):
                            mean_cutoff_metric_score = 0.
                        else:
                            mean_cutoff_metric_score = (
                                (2 * precision * recall)
                                / (precision + recall)
                            )

                    dict_results[int(cutoff)][metric] = mean_cutoff_metric_score
        else:
            logger.warning(
                "No users had a sufficient number of relevant items"
            )

        df_results = pd.DataFrame(
            columns=dict_results[self.cutoff_list[0]].keys(),
            index=self.cutoff_list,
        )
        df_results.index.rename(
            "cutoff",
            inplace=True,
        )

        for int_cutoff in dict_results.keys():
            df_results.loc[str(int_cutoff)] = dict_results[int_cutoff]

        results_run_string = get_result_string_df(df_results)

        return df_results, results_run_string

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

    def compute_recommenders_statistical_tests(
        self,
        recommender_baseline: BaseRecommender,
        recommender_baseline_name: str,
        recommender_baseline_folder: str,
        recommender_others: Sequence[BaseRecommender],
        recommender_others_names: Sequence[str],
        recommender_others_folders: Sequence[str],
        folder_export_results: str,
    ) -> Sequence[pd.DataFrame]:
        file_paths = [
            os.path.join(folder_export_results, f"{recommender_baseline_name}_groupwise_statistical_tests.parquet"),
            os.path.join(folder_export_results, f"{recommender_baseline_name}_pairwise_statistical_tests.parquet"),
        ]

        partial_compute_recommenders_statistical_tests = partial(
            self._compute_recommenders_statistical_tests,
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

    def _create_empty_results_dict(
        self
    ) -> dict[int, dict[str, float]]:
        return {
            cutoff: {
                metric: 0.
                for metric in self._str_metrics
            }
            for cutoff in self.cutoff_list
        }

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

        arr_user_ids_batches = np.array_split(
            arr_user_ids,
            indices_or_sections=100,
        )

        num_users = arr_user_ids.shape[0]

        list_df_results = []

        logger.info(
            f"Evaluating recommender."
        )
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
                    arr_cutoff_average_precision,
                    arr_cutoff_precision,
                    arr_cutoff_recall,
                    arr_cutoff_ndcg,
                    arr_cutoff_rr,
                    arr_cutoff_hit_rate,
                    arr_cutoff_arhr_all_hits,
                    arr_cutoff_f1_score,
                ) = evaluate_loop(
                    urm_test=self.URM_test,
                    list_batch_recommended_items=list_batch_recommended_items,
                    arr_batch_user_ids=arr_batch_user_id,
                    num_users=num_users,
                    max_cutoff=self.max_cutoff,
                    cutoff=cutoff,
                )

                df_results[(str(cutoff), EvaluatorMetrics.MAP.value)] = arr_cutoff_average_precision
                df_results[(str(cutoff), EvaluatorMetrics.PRECISION.value)] = arr_cutoff_precision
                df_results[(str(cutoff), EvaluatorMetrics.RECALL.value)] = arr_cutoff_recall
                df_results[(str(cutoff), EvaluatorMetrics.NDCG.value)] = arr_cutoff_ndcg
                df_results[(str(cutoff), EvaluatorMetrics.MRR.value)] = arr_cutoff_rr
                df_results[(str(cutoff), EvaluatorMetrics.HIT_RATE.value)] = arr_cutoff_hit_rate
                df_results[(str(cutoff), EvaluatorMetrics.ARHR.value)] = arr_cutoff_arhr_all_hits
                df_results[(str(cutoff), EvaluatorMetrics.F1.value)] = arr_cutoff_f1_score

            if self.ignore_items_flag:
                recommender.reset_items_to_ignore()

            list_df_results.append(df_results)

        df_results = pd.concat(
            list_df_results,
            axis="index",
        )

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
                        alpha=p_value,
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

    def _compute_recommenders_statistical_tests(
        self,
        recommender_baseline: BaseRecommender,
        recommender_baseline_name: str,
        recommender_baseline_folder: str,
        recommender_others: Sequence[BaseRecommender],
        recommender_others_names: Sequence[str],
        recommender_others_folders: Sequence[str],
    ) -> list[pd.DataFrame]:
        df_scores_baseline = self.evaluate_recommender(
            recommender=recommender_baseline,
            recommender_name=recommender_baseline_name,
            folder_export_results=recommender_baseline_folder,
        )

        list_df_scores_others = [
            self.evaluate_recommender(
                recommender=recommender,
                recommender_name=recommender_name,
                folder_export_results=recommender_folder,
            )
            for recommender, recommender_name, recommender_folder in zip(
                recommender_others, recommender_others_names, recommender_others_folders,
            )
        ]

        num_other_recommenders = len(list_df_scores_others)
        assert num_other_recommenders >= 1

        stats_groupwise = ["friedman"]
        stats_pairwise = ["wilcoxon", "bonferroni-wilcoxon", "sign", "bonferroni-sign"]
        inner_stats_groupwise = ["num_measurements", "alpha", "p_value", "hypothesis"]
        inner_stats_pairwise = ["alpha", "p_value", "hypothesis"]

        alternatives = [
            st_tests.StatisticalTestAlternative.TWO_SIDED,
            st_tests.StatisticalTestAlternative.LESS,
            st_tests.StatisticalTestAlternative.GREATER,
        ]
        str_alternatives = [
            alternative.value
            for alternative in alternatives
        ]
        alphas = [
            0.1,
            0.05,
            0.01,
        ]
        str_alphas = [str(alpha) for alpha in alphas]

        columns_groupwise = [("recommender", "", "", "", "", "")]
        columns_groupwise += itertools.product(
            self._str_cutoffs, self._str_metrics, stats_groupwise, str_alternatives, str_alphas, inner_stats_groupwise,
        )

        columns_pairwise = [("recommender_base", "", "", "", "", ""), ("recommender_other", "", "", "", "", "")]
        columns_pairwise += itertools.product(
            self._str_cutoffs, self._str_metrics, stats_pairwise, str_alternatives, str_alphas, inner_stats_pairwise
        )

        data_groupwise: dict[tuple, list] = {
            col: []
            for col in columns_groupwise
        }
        data_groupwise[("recommender", "", "", "", "", "")].append(
            recommender_baseline_name
        )

        data_pairwise: dict[tuple, list] = {
            col: []
            for col in columns_pairwise
        }
        data_pairwise[("recommender_base", "", "", "", "", "")] += [
            recommender_baseline_name
        ] * num_other_recommenders
        data_pairwise[("recommender_other", "", "", "", "", "")] += [
            rec_name
            for rec_name in recommender_others_names
        ]

        for cutoff in self._str_cutoffs:
            for metric in self._str_metrics:
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
                    results_statistical_tests = st_tests.compute_statistical_tests_of_base_vs_others(
                        scores_base=arr_scores_baseline,
                        scores_others=arr_scores_others,
                        alphas=alphas,
                        alternative=alternative,
                    )

                    for idx, alpha in enumerate(alphas):
                        data_groupwise[(cutoff, metric, "friedman", alternative.value, str(alpha), "alpha")].append(
                            alpha
                        )
                        data_groupwise[(cutoff, metric, "friedman", alternative.value, str(alpha), "p_value")].append(
                            results_statistical_tests.friedman.p_value
                        )
                        data_groupwise[(cutoff, metric, "friedman", alternative.value, str(alpha), "hypothesis")].append(
                            results_statistical_tests.friedman.hypothesis[idx].value
                        )
                        data_groupwise[(cutoff, metric, "friedman", alternative.value, str(alpha), "num_measurements")].append(
                            results_statistical_tests.friedman.num_measurements
                        )

                        data_pairwise[(cutoff, metric, "wilcoxon", alternative.value, str(alpha), "alpha")] += [
                            alpha
                        ] * num_other_recommenders
                        data_pairwise[(cutoff, metric, "wilcoxon", alternative.value, str(alpha), "p_value")] += [
                            res.p_value
                            for res in results_statistical_tests.wilcoxon
                        ]
                        data_pairwise[(cutoff, metric, "wilcoxon", alternative.value, str(alpha), "hypothesis")] += [
                            res.hypothesis[idx].value
                            for res in results_statistical_tests.wilcoxon
                        ]

                        data_pairwise[(cutoff, metric, "sign", alternative.value, str(alpha), "alpha")] += [
                            alpha
                        ] * num_other_recommenders
                        data_pairwise[(cutoff, metric, "sign", alternative.value, str(alpha), "p_value")] += [
                            res.p_value
                            for res in results_statistical_tests.sign
                        ]
                        data_pairwise[(cutoff, metric, "sign", alternative.value, str(alpha), "hypothesis")] += [
                            res.hypothesis[idx].value
                            for res in results_statistical_tests.sign
                        ]

                        data_pairwise[(cutoff, metric, "bonferroni-wilcoxon", alternative.value, str(alpha), "alpha")] += [
                            results_statistical_tests.bonferroni_wilcoxon.corrected_alphas[idx]
                        ] * num_other_recommenders
                        data_pairwise[(cutoff, metric, "bonferroni-wilcoxon", alternative.value, str(alpha), "p_value")] += [
                            p_val
                            for p_val in results_statistical_tests.bonferroni_wilcoxon.corrected_p_values
                        ]
                        data_pairwise[(cutoff, metric, "bonferroni-wilcoxon", alternative.value, str(alpha), "hypothesis")] += [
                            h[idx].value
                            for h in results_statistical_tests.bonferroni_wilcoxon.hypothesis
                        ]

                        data_pairwise[(cutoff, metric, "bonferroni-sign", alternative.value, str(alpha), "alpha")] += [
                            results_statistical_tests.bonferroni_sign.corrected_alphas[idx]
                        ] * num_other_recommenders
                        data_pairwise[(cutoff, metric, "bonferroni-sign", alternative.value, str(alpha), "p_value")] += [
                            p_val
                            for p_val in results_statistical_tests.bonferroni_sign.corrected_p_values
                        ]
                        data_pairwise[(cutoff, metric, "bonferroni-sign", alternative.value, str(alpha), "hypothesis")] += [
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
            df_results_groupwise, df_results_pairwise,
        ]
