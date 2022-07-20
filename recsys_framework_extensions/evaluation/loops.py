from typing import Union, cast

import numpy as np
import numba as nb
import pandas as pd
import scipy.sparse as sp

import recsys_framework_extensions.evaluation.metrics as metrics
from recsys_framework_extensions.sparse.utils import compute_item_popularity_from_urm


def _assert_recommendations_array(
    arr_urm_indptr: np.ndarray,
    arr_urm_indices: np.ndarray,
    arr_urm_data: np.ndarray,
    arr_batch_recommended_items: Union[np.ndarray, list[np.ndarray]],
    arr_batch_user_ids: np.ndarray,
    num_users: int,
    max_cutoff: int,
) -> None:
    assert num_users <= arr_urm_indptr.shape[0]
    assert arr_urm_data.shape == arr_urm_indices.shape

    num_batch_users = arr_batch_user_ids.shape[0]
    if isinstance(arr_batch_recommended_items, np.ndarray):
        assert (num_batch_users, max_cutoff) == arr_batch_recommended_items.shape
        assert arr_batch_user_ids.shape[0] == arr_batch_recommended_items.shape[0]
    else:
        assert num_batch_users == len(arr_batch_recommended_items)
        assert arr_batch_user_ids.shape[0] == len(arr_batch_recommended_items)


def _convert_to_nb_list_of_arrays(
    list_batch_recommended_items: list[list[int]],
) -> list[np.ndarray]:
    typed_arr_batch_recommended_items = nb.typed.List()
    for rec in list_batch_recommended_items:
        typed_arr_batch_recommended_items.append(
            np.asarray(rec, dtype=np.int32)
        )

    return typed_arr_batch_recommended_items


def _get_arr_batch_recommendations(
    list_batch_recommended_items: list[list[int]],
) -> Union[np.ndarray, list[np.ndarray]]:
    try:
        return np.asarray(
            list_batch_recommended_items,
            dtype=np.int32,
        )
    except:
        return _convert_to_nb_list_of_arrays(
            list_batch_recommended_items=list_batch_recommended_items,
        )


@nb.njit
def _nb_loop_evaluate_users(
    arr_urm_indptr: np.ndarray,
    arr_urm_indices: np.ndarray,
    arr_urm_data: np.ndarray,
    arr_batch_user_ids: np.ndarray,
    arr_batch_recommended_items: Union[np.ndarray, list[np.ndarray]],
    arr_train_item_popularity: np.ndarray,
    cutoff: int,
    num_items: int,
    num_interactions: int,
) -> tuple[np.ndarray, ...]:

    arr_cutoff_average_precision = np.zeros_like(arr_batch_user_ids, dtype=np.float64)
    arr_cutoff_precision = np.zeros_like(arr_batch_user_ids, dtype=np.float64)
    arr_cutoff_recall = np.zeros_like(arr_batch_user_ids, dtype=np.float64)
    arr_cutoff_ndcg = np.zeros_like(arr_batch_user_ids, dtype=np.float64)
    arr_cutoff_rr = np.zeros_like(arr_batch_user_ids, dtype=np.float64)
    arr_cutoff_hit_rate = np.zeros_like(arr_batch_user_ids, dtype=np.float64)
    arr_cutoff_arhr_all_hits = np.zeros_like(arr_batch_user_ids, dtype=np.float64)
    arr_cutoff_f1_score = np.zeros_like(arr_batch_user_ids, dtype=np.float64)
    arr_cutoff_novelty_score = np.zeros_like(arr_batch_user_ids, dtype=np.float64)
    arr_count_recommended_items = np.zeros(shape=(num_items, ), dtype=np.int32)

    # for idx_batch_user_id, test_user_id in enumerate(arr_batch_user_ids):
    for idx_batch_user_id, tuple_user_recommendations in enumerate(
        zip(arr_batch_user_ids, arr_batch_recommended_items)
    ):
        test_user_id, user_recommended_items = tuple_user_recommendations
        user_recommended_items = np.asarray(user_recommended_items, dtype=np.int32)

        user_profile_start = arr_urm_indptr[test_user_id]
        user_profile_end = arr_urm_indptr[test_user_id + 1]

        user_relevant_items = arr_urm_indices[user_profile_start:user_profile_end]
        user_relevance_scores = arr_urm_data[user_profile_start:user_profile_end]

        # Being the URM CSR, the indices are the non-zero column indexes
        user_is_recommended_item_relevant = np.asarray(
            [
                rec_item in user_relevant_items
                for rec_item in user_recommended_items
            ]
        )

        is_relevant_current_cutoff = user_is_recommended_item_relevant[:cutoff]
        recommended_items_current_cutoff = user_recommended_items[:cutoff]

        arr_cutoff_average_precision[idx_batch_user_id] = metrics.nb_average_precision(
            is_relevant=is_relevant_current_cutoff,
        )

        arr_cutoff_precision[idx_batch_user_id] = metrics.nb_precision(
            is_relevant=is_relevant_current_cutoff,
        )

        arr_cutoff_recall[idx_batch_user_id] = metrics.nb_recall(
            is_relevant=is_relevant_current_cutoff,
            pos_items=user_relevant_items,
        )

        arr_cutoff_ndcg[idx_batch_user_id] = metrics.nb_ndcg(
            ranked_list=recommended_items_current_cutoff,
            pos_items=user_relevant_items,
            relevance=user_relevance_scores,
            at=cutoff,
        )

        arr_cutoff_rr[idx_batch_user_id] = metrics.nb_rr(
            is_relevant=is_relevant_current_cutoff,
        )

        arr_cutoff_hit_rate[idx_batch_user_id] = metrics.nb_hit_rate(
            is_relevant=is_relevant_current_cutoff,
        )

        arr_cutoff_arhr_all_hits[idx_batch_user_id] = metrics.nb_arhr_all_hits(
            is_relevant=is_relevant_current_cutoff,
        )

        arr_cutoff_f1_score[idx_batch_user_id] = metrics.nb_f1_score(
            score_precision=arr_cutoff_precision[idx_batch_user_id],
            score_recall=arr_cutoff_recall[idx_batch_user_id],
        )

        arr_cutoff_novelty_score[idx_batch_user_id] = metrics.nb_novelty(
            recommended_items_ids=recommended_items_current_cutoff,
            item_popularity=arr_train_item_popularity,
            num_items=num_items,
            num_interactions=num_interactions,
        )

        for item_id in recommended_items_current_cutoff:
            arr_count_recommended_items[item_id] += 1

    return (
        arr_cutoff_average_precision,
        arr_cutoff_precision,
        arr_cutoff_recall,
        arr_cutoff_ndcg,
        arr_cutoff_rr,
        arr_cutoff_hit_rate,
        arr_cutoff_arhr_all_hits,
        arr_cutoff_f1_score,
        arr_cutoff_novelty_score,
        arr_count_recommended_items,
    )


_nb_loop_evaluate_users(
    arr_urm_indptr=np.array([0, 1, 2], dtype=np.int32),
    arr_urm_indices=np.array([1, 1], dtype=np.int32),
    arr_urm_data=np.array([1, 1], dtype=np.int32),
    arr_batch_recommended_items=np.array([[2, 1, 3, 4, 5], [2, 5, 8, 9, 1]], dtype=np.int32),
    arr_batch_user_ids=np.array([0, 1], dtype=np.int32),
    arr_train_item_popularity=np.array([0, 2, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
    cutoff=2,
    num_items=10,
    num_interactions=2,
)


def evaluate_loop(
    urm_test: sp.csr_matrix,
    urm_train: sp.csr_matrix,
    list_batch_recommended_items: list[list[int]],
    arr_batch_user_ids: np.ndarray,
    num_users: int,
    num_items: int,
    max_cutoff: int,
    cutoff: int,
) -> tuple[np.ndarray, ...]:
    arr_urm_indptr = cast(
        np.ndarray,
        urm_test.indptr,
    )
    arr_urm_indices: np.ndarray = cast(
        np.ndarray,
        urm_test.indices,
    )
    arr_urm_data: np.ndarray = cast(
        np.ndarray,
        urm_test.data,
    )

    arr_train_item_popularity = compute_item_popularity_from_urm(
        urm=urm_train,
    )

    arr_batch_recommended_items = _get_arr_batch_recommendations(
        list_batch_recommended_items=list_batch_recommended_items,
    )

    _assert_recommendations_array(
        arr_urm_indptr=arr_urm_indptr,
        arr_urm_indices=arr_urm_indices,
        arr_urm_data=arr_urm_data,
        arr_batch_recommended_items=arr_batch_recommended_items,
        arr_batch_user_ids=arr_batch_user_ids,
        num_users=num_users,
        max_cutoff=max_cutoff,
    )

    return _nb_loop_evaluate_users(
        arr_urm_indptr=arr_urm_indptr,
        arr_urm_indices=arr_urm_indices,
        arr_urm_data=arr_urm_data,
        arr_batch_recommended_items=arr_batch_recommended_items,
        arr_batch_user_ids=arr_batch_user_ids,
        arr_train_item_popularity=arr_train_item_popularity,
        cutoff=cutoff,
        num_items=num_items,
        num_interactions=urm_train.nnz,
    )


@nb.njit
def _nb_loop_count_recommended_items(
    arr_count_recommended_items: np.ndarray,
    arr_count_train_items: np.ndarray,
    arr_item_ids_to_ignore: np.ndarray,
) -> tuple[float, ...]:

    arr_mask_valid_recommended_items = np.ones_like(arr_count_recommended_items, dtype=np.bool_)
    arr_mask_valid_recommended_items[arr_item_ids_to_ignore] = False

    arr_mask_valid_train_items = np.ones_like(arr_count_train_items, dtype=np.bool_)
    arr_mask_valid_train_items[arr_item_ids_to_ignore] = False

    arr_count_valid_recommended_items = arr_count_recommended_items[arr_mask_valid_recommended_items]
    arr_count_valid_train_items = arr_count_train_items[arr_mask_valid_train_items]

    # First compute on train data (for ratio)
    diversity_gini_train = metrics.nb_diversity_gini(
        recommended_counter=arr_count_valid_train_items,
    )
    diversity_herfindahl_train = metrics.nb_diversity_herfindahl(
        recommended_counter=arr_count_valid_train_items,
    )
    shannon_entropy_train = metrics.nb_shannon_entropy(
        recommended_counter=arr_count_valid_train_items,
    )

    # Then compute on recommendations (for normal and ratio metrics)
    diversity_gini_recommendations = metrics.nb_diversity_gini(
        recommended_counter=arr_count_valid_recommended_items,
    )
    diversity_herfindahl_recommendations = metrics.nb_diversity_herfindahl(
        recommended_counter=arr_count_valid_recommended_items,
    )
    shannon_entropy_recommendations = metrics.nb_shannon_entropy(
        recommended_counter=arr_count_valid_recommended_items,
    )

    ratio_diversity_gini = metrics.nb_ratio_recommendation_vs_train(
        metric_train=diversity_gini_train,
        metric_recommendations=diversity_gini_recommendations,
    )
    ratio_diversity_herfindahl = metrics.nb_ratio_recommendation_vs_train(
        metric_train=diversity_herfindahl_train,
        metric_recommendations=diversity_herfindahl_recommendations,
    )
    ratio_shannon_entropy = metrics.nb_ratio_recommendation_vs_train(
        metric_train=shannon_entropy_train,
        metric_recommendations=shannon_entropy_recommendations,
    )

    return (
        diversity_gini_recommendations,
        ratio_diversity_gini,

        diversity_herfindahl_recommendations,
        ratio_diversity_herfindahl,

        shannon_entropy_recommendations,
        ratio_shannon_entropy,
    )


def count_recommended_items_loop(
    arr_count_recommended_items: np.ndarray,
    arr_item_ids_to_ignore: np.ndarray,
    urm_train: sp.csr_matrix,
) -> tuple[float, ...]:
    arr_count_recommended_items = np.asarray(
        arr_count_recommended_items, dtype=np.int32,
    )

    arr_item_ids_to_ignore = np.asarray(
        arr_item_ids_to_ignore, dtype=np.int32,
    )

    arr_count_train_items = np.asarray(
        np.ediff1d(
            sp.csc_matrix(urm_train).indptr
        ),
        dtype=np.int32,
    )

    return _nb_loop_count_recommended_items(
        arr_item_ids_to_ignore=arr_item_ids_to_ignore,
        arr_count_train_items=arr_count_train_items,
        arr_count_recommended_items=arr_count_recommended_items,
    )



