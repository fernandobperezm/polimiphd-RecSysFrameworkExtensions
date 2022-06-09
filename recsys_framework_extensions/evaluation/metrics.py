import numba as nb
import numpy as np

from Evaluation.metrics import dcg


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
    if score_precision + score_recall == 0:
        return 0.

    return (2 * score_precision * score_recall) / (score_precision + score_recall)


nb_average_precision = nb.njit(average_precision)
nb_precision = nb.njit(precision)
nb_recall = nb.njit(recall)
nb_ndcg = nb.njit(ndcg)
nb_dcg = nb.njit(dcg)
nb_rr = nb.njit(rr)
nb_hit_rate = nb.njit(hit_rate)
nb_arhr_all_hits = nb.njit(arhr_all_hits)
nb_f1_score = nb.njit(f1_score_micro_averaged)

_ranked_list = np.asarray([3, 2, 1], dtype=np.int32)
_is_relevant = np.asarray([False, False, True], dtype=np.bool_)
_pos_items = np.asarray([1], dtype=np.int32)
_relevance = np.asarray([1.], dtype=np.float32)
_scores = np.asarray([1.], dtype=np.float32)


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





