import numba as nb
import numpy as np
import scipy.sparse as sp

from Evaluation.metrics import dcg, _compute_diversity_gini, _compute_diversity_herfindahl, _compute_shannon_entropy

from recsys_framework_extensions.sparse.utils import compute_item_popularity_from_urm


def average_precision(is_relevant: np.ndarray) -> float:

    if len(is_relevant) == 0:
        a_p = 0.0
    else:
        p_at_k = (
            is_relevant * np.cumsum(is_relevant)
            / (1 + np.arange(is_relevant.shape[0]))
        )
        a_p = np.sum(p_at_k) / is_relevant.shape[0]

    assert 0 <= a_p <= 1
    return a_p


def precision(is_relevant: np.ndarray) -> float:
    if len(is_relevant) == 0:
        precision_score = 0.0
    else:
        precision_score = np.sum(is_relevant, dtype=np.float64) / len(is_relevant)

    assert 0 <= precision_score <= 1
    return precision_score


def recall(is_relevant: np.ndarray, pos_items: np.ndarray) -> float:

    recall_score: float = np.sum(is_relevant, dtype=np.float64) / pos_items.shape[0]

    assert 0 <= recall_score <= 1
    return recall_score


def ndcg(ranked_list: np.ndarray, pos_items: np.ndarray, relevance: np.ndarray, at: int) -> float:
    # Create a dictionary associating item_id to its relevance
    # it2rel[item] -> relevance[item]
    it2rel = dict()
    for pos_i, rel_i in zip(pos_items, relevance):
        it2rel[pos_i] = rel_i

    # Creates array of length "at" with the relevance associated to the item in that position
    ranked_relevance = []
    for rec_i in ranked_list:
        rel = 0.
        if rec_i in it2rel:
            rel = it2rel[rec_i]

        ranked_relevance.append(rel)

    rank_scores = np.array(ranked_relevance, dtype=np.float64)[:at]

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


def rr(is_relevant: np.ndarray) -> float:
    """
    Reciprocal rank of the FIRST relevant item in the ranked list (0 if none)
    :param is_relevant: boolean array
    :return:
    """
    if 0 == is_relevant.shape[0]:
        return 0.0

    ranks = np.arange(1, len(is_relevant) + 1)[is_relevant]

    if len(ranks) > 0:
        return 1. / ranks[0]
    else:
        return 0.0


def hit_rate(is_relevant: np.ndarray) -> bool:
    return np.any(is_relevant)


def arhr_all_hits(is_relevant: np.ndarray) -> float:
    # average reciprocal hit-rank (ARHR) of all relevant items
    # As opposed to MRR, ARHR takes into account all relevant items and not just the first
    # pag 17
    # http://glaros.dtc.umn.edu/gkhome/fetch/papers/itemrsTOIS04.pdf
    # https://emunix.emich.edu/~sverdlik/COSC562/ItemBasedTopTen.pdf

    p_reciprocal = (
        1 / np.arange(
            start=1.,
            stop=len(is_relevant) + 1.,
            step=1.0,
        )
    ).astype(np.float64)
    arhr_score = is_relevant.astype(np.float64).dot(p_reciprocal)

    assert not np.isnan(arhr_score)
    return arhr_score


def f1_score_micro_averaged(
    score_precision: float,
    score_recall: float,
) -> float:
    if 0. == score_precision + score_recall:
        return 0.

    return (2 * score_precision * score_recall) / (score_precision + score_recall)


def novelty(
    recommended_items_ids: np.ndarray,
    item_popularity: np.ndarray,
    num_items: int,
    num_interactions: int,
) -> float:
    if recommended_items_ids.size <= 0:
        return 0.

    recommended_items_popularity = item_popularity[recommended_items_ids]

    probability = recommended_items_popularity / num_interactions
    probability = probability[probability != 0]

    return np.sum(
        -np.log2(probability) / num_items
    )


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


def novelty_train(
    urm_train: sp.csr_matrix,
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


def shannon_entropy(
    recommended_counter: np.ndarray,
) -> float:
    # Ignore from the computation both ignored items and items with zero occurrence.
    # Zero occurrence items will have zero probability and will not change the result, butt will generate nans if
    # used in the log
    recommended_counter_mask = np.ones_like(recommended_counter, dtype=np.bool_)
    recommended_counter_mask[recommended_counter == 0] = False

    recommended_counter = recommended_counter[recommended_counter_mask]

    n_recommendations = recommended_counter.sum()

    recommended_probability = recommended_counter/n_recommendations

    shannon_entropy = -np.sum(recommended_probability * np.log2(recommended_probability))

    return shannon_entropy


def ratio_recommendation_vs_train(
    metric_train: float,
    metric_recommendations: float,
) -> float:
    return metric_recommendations / metric_train


nb_average_precision = nb.njit(average_precision)
nb_precision = nb.njit(precision)
nb_recall = nb.njit(recall)
nb_ndcg = nb.njit(ndcg)
nb_dcg = nb.njit(dcg)
nb_rr = nb.njit(rr)
nb_hit_rate = nb.njit(hit_rate)
nb_arhr_all_hits = nb.njit(arhr_all_hits)
nb_f1_score = nb.njit(f1_score_micro_averaged)
nb_novelty = nb.njit(novelty)
nb_diversity_gini = nb.njit(_compute_diversity_gini)
nb_diversity_herfindahl = nb.njit(_compute_diversity_herfindahl)
nb_shannon_entropy = nb.njit(shannon_entropy)
nb_ratio_recommendation_vs_train = nb.njit(ratio_recommendation_vs_train)

_ranked_list = np.asarray([3, 2, 1], dtype=np.int32)
_is_relevant = np.asarray([False, False, True], dtype=np.bool_)
_pos_items = np.asarray([1], dtype=np.int32)
_relevance = np.asarray([1.], dtype=np.float32)
_scores = np.asarray([1.], dtype=np.float32)
_counter_recommended_items = np.array([1, 2, 3, 4, 5], dtype=np.int32)


nb_average_precision(is_relevant=_is_relevant)
nb_precision(is_relevant=_is_relevant)
nb_recall(is_relevant=_is_relevant, pos_items=_pos_items)
nb_ndcg(ranked_list=_ranked_list, pos_items=_pos_items, relevance=_relevance, at=2)
nb_dcg(scores=_scores)
nb_rr(is_relevant=_is_relevant)
nb_f1_score(
    score_precision=nb_precision(is_relevant=_is_relevant),
    score_recall=nb_recall(is_relevant=_is_relevant, pos_items=_pos_items),
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

assert np.allclose(
    nb_average_precision(is_relevant=_is_relevant),
    average_precision(is_relevant=_is_relevant),
)
assert np.allclose(
    nb_precision(is_relevant=_is_relevant),
    precision(is_relevant=_is_relevant),
)
assert np.allclose(
    nb_recall(is_relevant=_is_relevant, pos_items=_pos_items),
    recall(is_relevant=_is_relevant, pos_items=_pos_items),
)
assert np.allclose(
    nb_ndcg(ranked_list=_ranked_list, pos_items=_pos_items, relevance=_relevance, at=2),
    ndcg(ranked_list=_ranked_list, pos_items=_pos_items, relevance=_relevance, at=2),
)
assert np.allclose(
    nb_rr(is_relevant=_is_relevant),
    rr(is_relevant=_is_relevant),
)
assert np.allclose(
    nb_hit_rate(is_relevant=_is_relevant),
    hit_rate(is_relevant=_is_relevant),
)
assert np.allclose(
    nb_arhr_all_hits(is_relevant=_is_relevant),
    arhr_all_hits(is_relevant=_is_relevant),
)
assert np.allclose(
    nb_f1_score(
        nb_precision(is_relevant=_is_relevant),
        nb_recall(is_relevant=_is_relevant, pos_items=_pos_items),
    ),
    f1_score_micro_averaged(
        score_precision=nb_precision(is_relevant=_is_relevant),
        score_recall=nb_recall(is_relevant=_is_relevant, pos_items=_pos_items),
    ),
)
assert np.allclose(
    nb_diversity_gini(
        recommended_counter=_counter_recommended_items,
    ),
    _compute_diversity_gini(
        recommended_counter=_counter_recommended_items,
    ),
)
assert np.allclose(
    nb_diversity_herfindahl(
        recommended_counter=_counter_recommended_items,
    ),
    _compute_diversity_herfindahl(
        recommended_counter=_counter_recommended_items,
    ),
)
assert np.allclose(
    nb_shannon_entropy(
        recommended_counter=_counter_recommended_items,
    ),
    _compute_shannon_entropy(
        recommended_counter=_counter_recommended_items,
    ),
)


