import numpy as np
import scipy.sparse as sp

from Evaluation.metrics import (
    dcg,
    _compute_diversity_gini,
    _compute_diversity_herfindahl,
)
from recsys_framework_extensions.sparse.utils import compute_item_popularity_from_urm

py_dcg = dcg
py_diversity_gini = _compute_diversity_gini
py_diversity_herfindahl = _compute_diversity_herfindahl


def py_average_precision(is_relevant: np.ndarray) -> float:
    if len(is_relevant) == 0:
        a_p = 0.0
    else:
        p_at_k = (
            is_relevant * np.cumsum(is_relevant) / (1 + np.arange(is_relevant.shape[0]))
        )
        a_p = np.sum(p_at_k) / is_relevant.shape[0]

    assert 0 <= a_p <= 1
    return a_p


def py_precision(is_relevant: np.ndarray) -> float:
    if len(is_relevant) == 0:
        precision_score = 0.0
    else:
        precision_score = np.sum(is_relevant, dtype=np.float64) / len(is_relevant)

    assert 0 <= precision_score <= 1
    return precision_score


def py_recall(is_relevant: np.ndarray, pos_items: np.ndarray) -> float:
    recall_score: float = np.sum(is_relevant, dtype=np.float64) / pos_items.shape[0]

    assert 0 <= recall_score <= 1
    return recall_score


def py_prepare_ndcg(
    ranked_list: np.ndarray, pos_items: np.ndarray, relevance: np.ndarray, at: int
):
    # Create a dictionary associating item_id to its relevance
    # it2rel[item] -> relevance[item]
    it2rel = dict()
    for pos_i, rel_i in zip(pos_items, relevance):
        it2rel[pos_i] = rel_i

    # Creates array of length "at" with the relevance associated to the item in that position
    ranked_relevance = []
    for rec_i in ranked_list:
        rel = 0.0
        if rec_i in it2rel:
            rel = it2rel[rec_i]

        ranked_relevance.append(rel)

    rank_scores = np.array(ranked_relevance, dtype=np.float64)[:at]

    return rank_scores


def py_ndcg(
    ranked_list: np.ndarray, pos_items: np.ndarray, relevance: np.ndarray, at: int
) -> float:
    rank_scores = py_prepare_ndcg(
        ranked_list=ranked_list,
        pos_items=pos_items,
        relevance=relevance,
        at=at,
    )

    # DCG uses the relevance of the recommended items
    rank_dcg = py_dcg(rank_scores)

    if rank_dcg == 0.0:
        return 0.0

    # IDCG has all relevances to 1 (or the values provided), up to the number of items in the test set that can fit in the list length
    ideal_dcg = py_dcg(np.sort(relevance)[::-1][:at])

    if ideal_dcg == 0.0:
        return 0.0

    ndcg_ = rank_dcg / ideal_dcg

    return ndcg_


def py_rr(is_relevant: np.ndarray) -> float:
    """
    Reciprocal rank of the FIRST relevant item in the ranked list (0 if none)
    :param is_relevant: boolean array
    :return:
    """
    if 0 == is_relevant.shape[0]:
        return 0.0

    ranks = np.arange(1, len(is_relevant) + 1)[is_relevant]

    if len(ranks) > 0:
        return 1.0 / ranks[0]
    else:
        return 0.0


def py_hit_rate(is_relevant: np.ndarray) -> bool:
    return np.any(is_relevant)


def py_coverage_item(recommended_counter: np.ndarray) -> float:
    recommended_mask = recommended_counter > 0
    return recommended_mask.sum() / recommended_counter.shape[0]


def py_coverage_user(is_relevant: np.ndarray) -> bool:
    return is_relevant.size > 0


def py_coverage_user_hit(is_relevant: np.ndarray) -> bool:
    return np.any(is_relevant)


def py_coverage_user_mean(
    arr_user_mask: np.ndarray,
    arr_users_to_ignore: np.ndarray,
    num_total_users: int,
) -> float:
    num_ignored_users = arr_users_to_ignore.shape[0]

    return arr_user_mask.sum() / (num_total_users - num_ignored_users)


def py_arhr_all_hits(is_relevant: np.ndarray) -> float:
    # average reciprocal hit-rank (ARHR) of all relevant items
    # As opposed to MRR, ARHR takes into account all relevant items and not just the first
    # pag 17
    # http://glaros.dtc.umn.edu/gkhome/fetch/papers/itemrsTOIS04.pdf
    # https://emunix.emich.edu/~sverdlik/COSC562/ItemBasedTopTen.pdf

    p_reciprocal = (
        1
        / np.arange(
            start=1.0,
            stop=len(is_relevant) + 1.0,
            step=1.0,
        )
    ).astype(np.float64)
    arhr_score = is_relevant.astype(np.float64).dot(p_reciprocal)

    assert not np.isnan(arhr_score)
    return arhr_score


def py_f1_score_micro_averaged(
    score_precision: float,
    score_recall: float,
) -> float:
    if 0.0 == score_precision + score_recall:
        return 0.0

    return (2 * score_precision * score_recall) / (score_precision + score_recall)


def py_novelty(
    recommended_items_ids: np.ndarray,
    item_popularity: np.ndarray,
    num_items: int,
    num_interactions: int,
) -> float:
    if recommended_items_ids.size <= 0:
        return 0.0

    recommended_items_popularity = item_popularity[recommended_items_ids]

    probability = recommended_items_popularity / num_interactions
    probability = probability[probability != 0]

    return np.sum(-np.log2(probability) / num_items)


def py_novelty_train(
    urm_train: sp.csr_matrix,
) -> float:
    num_users, num_items = urm_train.shape
    num_interactions = urm_train.nnz

    arr_train_item_popularity = compute_item_popularity_from_urm(
        urm=urm_train,
    )

    novelty_score = 0.0

    for user_id in range(num_users):
        user_profile_start = urm_train.indptr[user_id]
        user_profile_end = urm_train.indptr[user_id + 1]

        arr_user_relevant_items = np.asarray(
            urm_train.indices[user_profile_start:user_profile_end],
            dtype=np.int32,
        )

        novelty_score += py_novelty(
            recommended_items_ids=arr_user_relevant_items,
            item_popularity=arr_train_item_popularity,
            num_items=num_items,
            num_interactions=num_interactions,
        )

    return novelty_score / num_users


def py_shannon_entropy(
    recommended_counter: np.ndarray,
) -> float:
    # Ignore from the computation both ignored items and items with zero occurrence.
    # Zero occurrence items will have zero probability and will not change the result, butt will generate nans if
    # used in the log
    recommended_counter_mask = np.ones_like(recommended_counter, dtype=np.bool_)
    recommended_counter_mask[recommended_counter == 0] = False

    recommended_counter = recommended_counter[recommended_counter_mask]

    n_recommendations = recommended_counter.sum()

    recommended_probability = recommended_counter / n_recommendations

    shannon_entropy = -np.sum(
        recommended_probability * np.log2(recommended_probability)
    )

    return shannon_entropy


def py_ratio_recommendation_vs_train(
    metric_train: float,
    metric_recommendations: float,
) -> float:
    return metric_recommendations / metric_train


def py_position_relevant_items(
    is_relevant: np.ndarray,
    cutoff: int,
) -> int:
    """

    Parameters
    ----------
    is_relevant : a boolean array that indicates whether any recommended item `i` is relevant to the user or not. Is relevant may be empty indicating no recommendations for a user.

    Returns
    -------

    """
    assert is_relevant.ndim == 1
    # generates an assert error which I think may be related to some recommendation lists having less items than the
    # cutoff.
    # assert is_relevant.size == cutoff
    if is_relevant.size > cutoff:
        raise ValueError(
            f"The function `py_position_relevant_items` needs the `is_relevant` parameter to have the same number of elements as the cutoff. Num received values: is_relevant.size={is_relevant.size} - cutoff={cutoff}"
        )

    arr_bool_is_relevant = np.asarray(is_relevant, dtype=np.bool8)

    for position, item_relevant in enumerate(arr_bool_is_relevant):
        if item_relevant:
            return cutoff - position

    return 0
