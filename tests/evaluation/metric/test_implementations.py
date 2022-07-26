import numpy as np

import recsys_framework_extensions.evaluation.metric.nb_impl as nb_metric
import recsys_framework_extensions.evaluation.metric.py_impl as py_metric

from recsys_framework_extensions.logging import get_logger

logger = get_logger(
    logger_name=__file__,
)


class TestMetrics:
    ranked_list = np.asarray([3, 2, 1], dtype=np.int32)
    is_relevant = np.asarray([False, False, True], dtype=np.bool_)
    pos_items = np.asarray([1], dtype=np.int32)
    relevance = np.asarray([1.], dtype=np.float32)
    scores = np.asarray([1.], dtype=np.float32)
    counter_recommended_items = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    item_popularity = np.asarray([10, 5, 2, 3, 8], dtype=np.int32)
    num_items = item_popularity.shape[0]
    num_interactions = np.sum(item_popularity)

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
            nb_metric.nb_ndcg(ranked_list=self.ranked_list, pos_items=self.pos_items, relevance=self.relevance, at=2),
            py_metric.py_ndcg(ranked_list=self.ranked_list, pos_items=self.pos_items, relevance=self.relevance, at=2),
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
                score_recall=nb_metric.nb_recall(is_relevant=self.is_relevant, pos_items=self.pos_items),
            ),
            py_metric.py_f1_score_micro_averaged(
                score_precision=py_metric.py_precision(is_relevant=self.is_relevant),
                score_recall=py_metric.py_recall(is_relevant=self.is_relevant, pos_items=self.pos_items)
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
            nb_metric.nb_coverage_item(recommended_counter=self.counter_recommended_items),
            py_metric.py_coverage_item(recommended_counter=self.counter_recommended_items),
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
            )
        )
        
