import numba as nb
import numpy as np
import time
from scipy import sparse

import Evaluation.metrics as framework_metric
import recsys_framework_extensions.evaluation.metric.py_impl as py_metric

from recsys_framework_extensions.sparse.utils import compute_item_popularity_from_urm
from recsys_framework_extensions.logging import get_logger


logger = get_logger(
    logger_name=__file__,
)

nb_average_precision = nb.njit(
    py_metric.py_average_precision
)
nb_precision = nb.njit(
    py_metric.py_precision
)
nb_recall = nb.njit(
    py_metric.py_recall
)
nb_rr = nb.njit(
    py_metric.py_rr
)
nb_hit_rate = nb.njit(
    py_metric.py_hit_rate
)
nb_arhr_all_hits = nb.njit(
    py_metric.py_arhr_all_hits
)
nb_f1_score = nb.njit(
    py_metric.py_f1_score_micro_averaged
)
nb_coverage_item = nb.njit(
    py_metric.py_coverage_item
)
nb_coverage_user = nb.njit(
    py_metric.py_coverage_user
)
nb_coverage_user_hit = nb.njit(
    py_metric.py_coverage_user_hit
)
nb_coverage_user_mean = nb.njit(
    py_metric.py_coverage_user_mean
)
nb_novelty = nb.njit(
    py_metric.py_novelty
)
nb_diversity_gini = nb.njit(
    py_metric.py_diversity_gini
)
nb_diversity_herfindahl = nb.njit(
    py_metric.py_diversity_herfindahl
)
nb_shannon_entropy = nb.njit(
    py_metric.py_shannon_entropy
)
nb_ratio_recommendation_vs_train = nb.njit(
    py_metric.py_ratio_recommendation_vs_train
)
nb_dcg = nb.njit(
    framework_metric.dcg
)
nb_prepare_ndcg = nb.njit(
    py_metric.py_prepare_ndcg
)


@nb.njit
def nb_ndcg(
    ranked_list: np.ndarray,
    pos_items: np.ndarray,
    relevance: np.ndarray,
    at: int
) -> float:
    rank_scores = nb_prepare_ndcg(
        ranked_list=ranked_list,
        pos_items=pos_items,
        relevance=relevance,
        at=at,
    )

    # DCG uses the relevance of the recommended items
    rank_dcg = nb_dcg(rank_scores)

    if rank_dcg == 0.0:
        return 0.0

    # IDCG has all relevances to 1 (or the values provided), up to the number of items in the test set that can fit in the list length
    ideal_dcg = nb_dcg(np.sort(relevance)[::-1][:at])

    if ideal_dcg == 0.0:
        return 0.0

    ndcg_ = rank_dcg / ideal_dcg

    return ndcg_


@nb.njit
def _nb_novelty_train(
    num_users: int,
    num_items: int,
    num_interactions: int,
    arr_train_item_popularity: np.ndarray,
    arr_urm_indices: np.ndarray,
    arr_urm_indptr: np.ndarray,
) -> float:
    novelty_score = 0.

    for user_id in range(num_users):
        user_profile_start = arr_urm_indptr[user_id]
        user_profile_end = arr_urm_indptr[user_id + 1]

        arr_user_relevant_items = np.asarray(
            arr_urm_indices[user_profile_start:user_profile_end],
            dtype=np.int32,
        )

        novelty_score += nb_novelty(
            recommended_items_ids=arr_user_relevant_items,
            item_popularity=arr_train_item_popularity,
            num_items=num_items,
            num_interactions=num_interactions,
        )

    return novelty_score / num_users


def nb_novelty_train(
    urm_train: sparse.csr_matrix,
) -> float:
    num_users, num_items = urm_train.shape
    num_interactions = urm_train.nnz

    arr_train_item_popularity = compute_item_popularity_from_urm(
        urm=urm_train,
    )

    return _nb_novelty_train(
        num_users=num_users,
        num_items=num_items,
        num_interactions=num_interactions,
        arr_train_item_popularity=arr_train_item_popularity,
        arr_urm_indices=urm_train.indices,
        arr_urm_indptr=urm_train.indptr,
    )


_ranked_list = np.asarray([3, 2, 1], dtype=np.int32)
_is_relevant = np.asarray([False, False, True], dtype=np.bool_)
_pos_items = np.asarray([1], dtype=np.int32)
_relevance = np.asarray([1.], dtype=np.float32)
_scores = np.asarray([1.], dtype=np.float32)
_counter_recommended_items = np.array([1, 2, 3, 4, 5], dtype=np.int32)
_item_popularity = np.asarray([10, 5, 2, 3, 8], dtype=np.int32)
_num_items = _item_popularity.shape[0]
_num_interactions = np.sum(_item_popularity)

start = time.time()
logger.info(
    "Compiling numba jit-optimized metrics."
)

nb_precision(
    is_relevant=_is_relevant
)
nb_recall(
    is_relevant=_is_relevant, pos_items=_pos_items
)
nb_average_precision(
    is_relevant=_is_relevant
)
nb_ndcg(
    ranked_list=_ranked_list, pos_items=_pos_items, relevance=_relevance, at=2
)
nb_dcg(
    scores=_scores
)
nb_rr(
    is_relevant=_is_relevant
)
nb_hit_rate(
    is_relevant=_is_relevant
)
nb_arhr_all_hits(
    is_relevant=_is_relevant,
)
nb_f1_score(
    score_precision=nb_precision(is_relevant=_is_relevant),
    score_recall=nb_recall(is_relevant=_is_relevant, pos_items=_pos_items),
)
nb_coverage_user(
    is_relevant=_is_relevant,
)
nb_coverage_user_hit(
    is_relevant=_is_relevant,
)
nb_coverage_item(
    recommended_counter=_counter_recommended_items,
)
nb_novelty(
    recommended_items_ids=_ranked_list,
    item_popularity=_item_popularity,
    num_items=_num_items,
    num_interactions=_num_interactions,
)
nb_diversity_gini(
    recommended_counter=_counter_recommended_items,
)
nb_diversity_herfindahl(
    recommended_counter=_counter_recommended_items,
)
nb_shannon_entropy(
    recommended_counter=_counter_recommended_items,
)
nb_ratio_recommendation_vs_train(
    metric_train=0.5,
    metric_recommendations=0.8,
)

end = time.time()
logger.info(
    f"Finished compiling numba jit-optimized metrics. Took {end - start:.2f} seconds."
)


