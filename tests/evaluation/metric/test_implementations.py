import numpy as np

import recsys_framework_extensions.evaluation.metric.nb_impl as nb_metric
import recsys_framework_extensions.evaluation.metric.py_impl as py_metric
import Evaluation.metrics as framework_metric

import logging

logger = logging.getLogger(__name__)


class TestMetrics:
    ranked_list = np.asarray([3, 2, 1], dtype=np.int32)
    is_relevant = np.asarray([False, False, True], dtype=np.bool_)
    cutoff = 3
    pos_items = np.asarray([1], dtype=np.int32)
    relevance = np.asarray([1.0], dtype=np.float32)
    scores = np.asarray([1.0], dtype=np.float32)
    counter_recommended_items = np.array([1, 1, 1, 0, 0], dtype=np.int32)
    item_popularity = np.asarray([10, 5, 2, 3, 8], dtype=np.int32)
    num_items: int = item_popularity.shape[0]
    num_interactions: int = np.sum(item_popularity)
    num_users: int = 1

    def test_numba_and_python_implementations_are_equal(
        self,
    ):
        # Arrange

        # Act

        # Assert
        assert np.array_equal(
            nb_metric.nb_precision(is_relevant=self.is_relevant),
            py_metric.py_precision(is_relevant=self.is_relevant),
        )
        assert np.array_equal(
            nb_metric.nb_recall(is_relevant=self.is_relevant, pos_items=self.pos_items),
            py_metric.py_recall(is_relevant=self.is_relevant, pos_items=self.pos_items),
        )
        assert np.array_equal(
            nb_metric.nb_average_precision(is_relevant=self.is_relevant),
            py_metric.py_average_precision(is_relevant=self.is_relevant),
        )
        assert np.array_equal(
            nb_metric.nb_ndcg(
                ranked_list=self.ranked_list,
                pos_items=self.pos_items,
                relevance=self.relevance,
                at=2,
            ),
            py_metric.py_ndcg(
                ranked_list=self.ranked_list,
                pos_items=self.pos_items,
                relevance=self.relevance,
                at=2,
            ),
        )
        assert np.array_equal(
            nb_metric.nb_rr(is_relevant=self.is_relevant),
            py_metric.py_rr(is_relevant=self.is_relevant),
        )
        assert np.array_equal(
            nb_metric.nb_hit_rate(is_relevant=self.is_relevant),
            py_metric.py_hit_rate(is_relevant=self.is_relevant),
        )
        assert np.array_equal(
            nb_metric.nb_arhr_all_hits(is_relevant=self.is_relevant),
            py_metric.py_arhr_all_hits(is_relevant=self.is_relevant),
        )
        assert np.array_equal(
            nb_metric.nb_f1_score(
                score_precision=nb_metric.nb_precision(is_relevant=self.is_relevant),
                score_recall=nb_metric.nb_recall(
                    is_relevant=self.is_relevant, pos_items=self.pos_items
                ),
            ),
            py_metric.py_f1_score_micro_averaged(
                score_precision=py_metric.py_precision(is_relevant=self.is_relevant),
                score_recall=py_metric.py_recall(
                    is_relevant=self.is_relevant, pos_items=self.pos_items
                ),
            ),
        )
        assert np.array_equal(
            nb_metric.nb_coverage_user(is_relevant=self.is_relevant),
            py_metric.py_coverage_user(is_relevant=self.is_relevant),
        )
        assert np.array_equal(
            nb_metric.nb_coverage_user_hit(is_relevant=self.is_relevant),
            py_metric.py_coverage_user_hit(is_relevant=self.is_relevant),
        )
        assert np.array_equal(
            nb_metric.nb_coverage_item(
                recommended_counter=self.counter_recommended_items
            ),
            py_metric.py_coverage_item(
                recommended_counter=self.counter_recommended_items
            ),
        )
        assert np.array_equal(
            nb_metric.nb_novelty(
                recommended_items_ids=self.ranked_list,
                item_popularity=self.item_popularity,
                num_items=self.num_items,
                num_interactions=self.num_interactions,
            ),
            py_metric.py_novelty(
                recommended_items_ids=self.ranked_list,
                item_popularity=self.item_popularity,
                num_items=self.num_items,
                num_interactions=self.num_interactions,
            ),
        )
        assert np.array_equal(
            nb_metric.nb_diversity_gini(
                recommended_counter=self.counter_recommended_items,
            ),
            py_metric.py_diversity_gini(
                recommended_counter=self.counter_recommended_items,
            ),
        )
        assert np.array_equal(
            nb_metric.nb_diversity_herfindahl(
                recommended_counter=self.counter_recommended_items,
            ),
            py_metric.py_diversity_herfindahl(
                recommended_counter=self.counter_recommended_items,
            ),
        )
        assert np.array_equal(
            nb_metric.nb_shannon_entropy(
                recommended_counter=self.counter_recommended_items,
            ),
            py_metric.py_shannon_entropy(
                recommended_counter=self.counter_recommended_items,
            ),
        )
        assert np.array_equal(
            nb_metric.nb_ratio_recommendation_vs_train(
                metric_train=0.5,
                metric_recommendations=0.8,
            ),
            py_metric.py_ratio_recommendation_vs_train(
                metric_train=0.5,
                metric_recommendations=0.8,
            ),
        )
        assert np.array_equal(
            nb_metric.nb_position_relevant_items(
                is_relevant=self.is_relevant,
                cutoff=self.cutoff,
            ),
            py_metric.py_position_relevant_items(
                is_relevant=self.is_relevant,
                cutoff=self.cutoff,
            ),
        )

    def test_framework_and_extended_python_implementations_are_equal(
        self,
    ):
        # Arrange
        empty_array = np.asarray([], dtype=np.int32)
        framework_metric_hit_rate = framework_metric.HIT_RATE()
        framework_metric_f1_score = (
            lambda precision, recall: 2 * (precision * recall) / (precision + recall)
        )
        framework_metric_coverage_user = framework_metric.Coverage_User(
            n_users=self.num_users, ignore_users=empty_array
        )
        framework_metric_coverage_user_hit = framework_metric.Coverage_User_HIT(
            n_users=self.num_users, ignore_users=empty_array
        )
        framework_metric_coverage_item = framework_metric.Coverage_Item(
            n_items=self.num_items, ignore_items=empty_array
        )
        framework_metric_diversity_herfindahl = framework_metric.Diversity_Herfindahl(
            n_items=self.num_items, ignore_items=empty_array
        )
        framework_metric_diversity_gini = framework_metric.Gini_Diversity(
            n_items=self.num_items, ignore_items=empty_array
        )
        framework_metric_diversity_shannon = framework_metric.Shannon_Entropy(
            n_items=self.num_items, ignore_items=empty_array
        )

        # Act
        framework_metric_hit_rate.add_recommendations(
            is_relevant=self.is_relevant,
        )
        framework_metric_coverage_user.add_recommendations(
            recommended_items_ids=self.ranked_list,
            user_id=0,
        )
        framework_metric_coverage_user_hit.add_recommendations(
            is_relevant=self.is_relevant,
            user_id=0,
        )
        framework_metric_coverage_item.add_recommendations(
            recommended_items_ids=self.ranked_list,
        )
        framework_metric_diversity_herfindahl.add_recommendations(
            recommended_items_ids=self.ranked_list,
        )
        framework_metric_diversity_gini.add_recommendations(
            recommended_items_ids=self.ranked_list,
        )
        framework_metric_diversity_shannon.add_recommendations(
            recommended_items_ids=self.ranked_list,
        )

        # Assert
        assert np.array_equal(
            framework_metric.precision(is_relevant=self.is_relevant),
            py_metric.py_precision(is_relevant=self.is_relevant),
        )
        assert np.array_equal(
            framework_metric.recall(
                is_relevant=self.is_relevant, pos_items=self.pos_items
            ),
            py_metric.py_recall(is_relevant=self.is_relevant, pos_items=self.pos_items),
        )
        assert np.array_equal(
            framework_metric.average_precision(is_relevant=self.is_relevant),
            py_metric.py_average_precision(is_relevant=self.is_relevant),
        )
        assert np.array_equal(
            framework_metric.ndcg(
                ranked_list=self.ranked_list,
                pos_items=self.pos_items,
                relevance=self.relevance,
                at=2,
            ),
            py_metric.py_ndcg(
                ranked_list=self.ranked_list,
                pos_items=self.pos_items,
                relevance=self.relevance,
                at=2,
            ),
        )
        assert np.array_equal(
            framework_metric.rr(is_relevant=self.is_relevant),
            py_metric.py_rr(is_relevant=self.is_relevant),
        )
        assert np.array_equal(
            framework_metric_hit_rate.get_metric_value(),
            py_metric.py_hit_rate(is_relevant=self.is_relevant),
        )
        assert np.array_equal(
            framework_metric.arhr_all_hits(is_relevant=self.is_relevant),
            py_metric.py_arhr_all_hits(is_relevant=self.is_relevant),
        )
        assert np.array_equal(
            framework_metric_f1_score(
                precision=framework_metric.precision(is_relevant=self.is_relevant),
                recall=framework_metric.recall(
                    is_relevant=self.is_relevant, pos_items=self.pos_items
                ),
            ),
            py_metric.py_f1_score_micro_averaged(
                score_precision=py_metric.py_precision(is_relevant=self.is_relevant),
                score_recall=py_metric.py_recall(
                    is_relevant=self.is_relevant, pos_items=self.pos_items
                ),
            ),
        )
        assert np.array_equal(
            framework_metric_coverage_user.get_metric_value(),
            py_metric.py_coverage_user(is_relevant=self.is_relevant),
        )
        assert np.array_equal(
            framework_metric_coverage_user_hit.get_metric_value(),
            py_metric.py_coverage_user_hit(is_relevant=self.is_relevant),
        )
        assert np.array_equal(
            framework_metric_coverage_item.get_metric_value(),
            py_metric.py_coverage_item(
                recommended_counter=self.counter_recommended_items
            ),
        )
        assert np.array_equal(
            framework_metric_diversity_gini.get_metric_value(),
            py_metric.py_diversity_gini(
                recommended_counter=self.counter_recommended_items,
            ),
        )
        assert np.allclose(
            framework_metric_diversity_herfindahl.get_metric_value(),
            py_metric.py_diversity_herfindahl(
                recommended_counter=self.counter_recommended_items,
            ),
            equal_nan=True,
        )
        assert np.allclose(
            framework_metric_diversity_shannon.get_metric_value(),
            py_metric.py_shannon_entropy(
                recommended_counter=self.counter_recommended_items,
            ),
            equal_nan=True,
        )

    def test_position_relevant_items_yield_correct_results(self):
        # Arrange
        cases_to_try = [
            np.array([True, True, True], dtype=np.bool8),
            np.array([False, True, True], dtype=np.bool8),
            np.array([False, False, True], dtype=np.bool8),
            np.array([False, False, False], dtype=np.bool8),
        ]
        expected_results = [
            3,
            2,
            1,
            0,
        ]

        # Act
        obtained_results = [
            nb_metric.nb_position_relevant_items(
                is_relevant=case_is_relevant,
                cutoff=3,
            )
            for case_is_relevant in cases_to_try
        ]

        # Assert
        assert all(
            np.array_equal(
                expected,
                obtained,
                equal_nan=True,
            )
            for expected, obtained in zip(expected_results, obtained_results)
        )
