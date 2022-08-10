import numba as nb
import numpy as np
import sparse


@nb.njit(parallel=False)
def nb_sample_only_positives_from_sparse_coo_matrix(
    num_users: int,
    num_items: int,
    num_samples: int,
    urm: sparse.COO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    urm_rows, urm_cols = urm.coords[0], urm.coords[1]
    urm_data = urm.data
    urm_shape = urm.shape

    assert (num_users, num_items) == urm_shape
    assert urm_rows.shape == urm_cols.shape
    assert urm_rows.shape == urm_data.shape

    num_elements_in_urm = urm_data.shape[0]

    arr_indices = np.arange(
        start=0, stop=num_elements_in_urm, step=1,
    )

    arr_sampled_indices = np.random.choice(
        arr_indices, size=num_samples, replace=True,
    )

    arr_sampled_user_ids = urm_rows[arr_sampled_indices].copy().astype(np.int32)
    arr_sampled_item_ids = urm_cols[arr_sampled_indices].copy().astype(np.int32)
    arr_sampled_rating = urm_data[arr_sampled_indices].copy().astype(np.float64)

    return arr_sampled_user_ids, arr_sampled_item_ids, arr_sampled_rating


@nb.njit(parallel=False)
def nb_sample_only_negatives_from_sparse_csr_matrix(
    num_users: int,
    num_items: int,
    num_samples: int,
    urm_csr_indptr: np.ndarray,
    urm_csr_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    user_profiles_length = np.ediff1d(urm_csr_indptr)
    mask_non_valid_users_to_sample = np.asarray(
        (user_profiles_length == 0.) | (user_profiles_length == num_items),
        dtype=np.bool8,
    )

    arr_sampled_user_ids = np.random.randint(low=0, high=num_users - 1, size=num_samples)
    arr_sampled_item_ids = np.random.randint(low=0, high=num_items - 1, size=num_samples)

    for idx_sample, sample in enumerate(
        zip(
            arr_sampled_user_ids,
            arr_sampled_item_ids
        )
    ):
        sample_user_id, sample_item_id = sample

        # Ensure user can be sampled.
        user_is_not_valid = mask_non_valid_users_to_sample[sample_user_id]
        while user_is_not_valid:
            sample_user_id = np.random.randint(low=0, high=num_users - 1)
            user_is_not_valid = mask_non_valid_users_to_sample[sample_user_id]

        # Ensure item can be sampled:
        idx_user_profile_start = urm_csr_indptr[sample_user_id]
        idx_user_profile_end = urm_csr_indptr[sample_user_id + 1]
        user_profile = urm_csr_indices[idx_user_profile_start:idx_user_profile_end]

        item_is_not_valid = sample_item_id in user_profile
        while item_is_not_valid:
            sample_item_id = np.random.randint(low=0, high=num_items - 1)
            item_is_not_valid = sample_item_id in user_profile

        arr_sampled_user_ids[idx_sample] = sample_user_id
        arr_sampled_item_ids[idx_sample] = sample_item_id

    arr_sampled_user_ids = arr_sampled_user_ids.astype(np.int32)
    arr_sampled_item_ids = arr_sampled_item_ids.astype(np.int32)
    arr_sampled_ratings = np.zeros(shape=(num_samples,), dtype=np.float64)

    return arr_sampled_user_ids, arr_sampled_item_ids, arr_sampled_ratings


@nb.njit(parallel=False)
def nb_sample_positives_and_negatives_from_sparse_matrix(
    num_users: int,
    num_items: int,
    num_samples: int,
    urm_coo: sparse.COO,
    urm_csr_indptr: np.ndarray,
    urm_csr_indices: np.ndarray,
    frac_negative_sampling: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_negative_samples = int(num_samples * frac_negative_sampling)
    num_positive_samples = num_samples - num_negative_samples

    (
        arr_positive_sampled_user_ids,
        arr_positive_sampled_item_ids,
        arr_positive_sampled_ratings,
    ) = nb_sample_only_positives_from_sparse_coo_matrix(
        num_users=num_users,
        num_items=num_items,
        num_samples=num_positive_samples,
        urm=urm_coo,
    )

    (
        arr_negative_sampled_user_ids,
        arr_negative_sampled_item_ids,
        arr_negative_sampled_ratings,
    ) = nb_sample_only_negatives_from_sparse_csr_matrix(num_users=num_users, num_items=num_items,
                                                        num_samples=num_negative_samples, urm_csr_indptr=urm_csr_indptr,
                                                        urm_csr_indices=urm_csr_indices)

    arr_sampled_user_ids = np.concatenate(
        (arr_positive_sampled_user_ids, arr_negative_sampled_user_ids),
    ).astype(
        np.int32,
    )
    arr_sampled_item_ids = np.concatenate(
        (arr_positive_sampled_item_ids, arr_negative_sampled_item_ids),
    ).astype(
        np.int32,
    )
    arr_sampled_ratings = np.concatenate(
        (arr_positive_sampled_ratings, arr_negative_sampled_ratings),
    ).astype(
        np.float64,
    )

    arr_indices = np.arange(start=0, stop=num_samples, step=1)
    np.random.shuffle(arr_indices)

    arr_sampled_user_ids = arr_sampled_user_ids[arr_indices]
    arr_sampled_item_ids = arr_sampled_item_ids[arr_indices]
    arr_sampled_ratings = arr_sampled_ratings[arr_indices]

    return (
        arr_sampled_user_ids,
        arr_sampled_item_ids,
        arr_sampled_ratings,
    )
