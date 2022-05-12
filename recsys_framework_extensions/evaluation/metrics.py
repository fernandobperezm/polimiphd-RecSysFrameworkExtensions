import numba as nb
import numpy as np

from Evaluation.metrics import dcg


def precision(is_relevant):
    if len(is_relevant) == 0:
        precision_score = 0.0
    else:
        precision_score = np.sum(is_relevant, dtype=np.float64) / len(is_relevant)

    assert 0 <= precision_score <= 1
    return precision_score


def recall(is_relevant, pos_items):

    recall_score = np.sum(is_relevant, dtype=np.float64) / pos_items.shape[0]

    assert 0 <= recall_score <= 1
    return recall_score


def ndcg(ranked_list, pos_items, relevance, at: int):
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


nb_precision = nb.njit(precision)
nb_recall = nb.njit(recall)
nb_ndcg = nb.njit(ndcg)
nb_dcg = nb.njit(dcg)

_ranked_list = np.asarray([3, 2, 1], dtype=np.int32)
_is_relevant = np.asarray([False, False, True], dtype=np.bool_)
_pos_items = np.asarray([1], dtype=np.int32)
_relevance = np.asarray([1.], dtype=np.float32)
_scores = np.asarray([1.], dtype=np.float32)


nb_precision(is_relevant=_is_relevant)
nb_recall(is_relevant=_is_relevant, pos_items=_pos_items)
nb_ndcg(ranked_list=_ranked_list, pos_items=_pos_items, relevance=_relevance, at=2)
nb_dcg(scores=_scores)


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





