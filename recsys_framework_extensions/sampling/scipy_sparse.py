import numba as nb
import numpy as np


@nb.njit
def nb_sample_user(
    num_users: int,
    num_items: int,
    urm_csr_indptr: np.ndarray,
) -> tuple[int, int, int, int]:
    sample_user = -1
    user_profile_start = -1
    user_profile_end = -1
    num_seen_items = 0
    while num_seen_items == 0 or num_seen_items == num_items:
        sample_user = np.random.randint(
            low=0, high=num_users,
        )

        user_profile_start = urm_csr_indptr[sample_user]
        user_profile_end = urm_csr_indptr[sample_user + 1]
        num_seen_items = user_profile_end - user_profile_start

    return sample_user, user_profile_start, user_profile_end, num_seen_items,


@nb.njit
def nb_sample_positive_item(
    num_seen_items: int,
    user_profile_start: int,
    urm_csr_indices: np.ndarray,
    urm_csr_data: np.ndarray,
) -> tuple[int, float]:
    idx_positive = np.random.randint(
        low=0, high=num_seen_items,
    )

    sample_item = urm_csr_indices[user_profile_start + idx_positive]
    sample_rating = urm_csr_data[user_profile_start + idx_positive]

    return sample_item, sample_rating


@nb.njit
def nb_sample_negative_item(
    num_items: int,
    num_seen_items: int,
    urm_csr_indices: np.ndarray,
    user_profile_start: int,
) -> tuple[int, float]:
    sample_item = -1
    sample_rating = 0.

    neg_item_selected = False
    while not neg_item_selected:
        sample_item = np.random.randint(
            low=0, high=num_items,
        )

        index = 0
        # Indices data is sorted, so I don't need to go to the end of the current row
        while index < num_seen_items and urm_csr_indices[user_profile_start + index] < sample_item:
            index += 1

        # If the positive item in position 'index' is == sample.item, negative not selected
        # If the positive item in position 'index' is > sample.item or index == n_seen_items, negative selected
        if index == num_seen_items or urm_csr_indices[user_profile_start + index] > sample_item:
            neg_item_selected = True

    return sample_item, sample_rating
