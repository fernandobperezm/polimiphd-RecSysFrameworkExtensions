import numba as nb
import numpy as np
import scipy.sparse as scipy_sparse
import sparse

from recsys_framework_extensions.sampling.sparse import (
    nb_sample_only_positives_from_sparse_coo_matrix,
    nb_sample_only_negatives_from_sparse_csr_matrix,
    nb_sample_positives_and_negatives_from_sparse_matrix,
)


@nb.njit
def set_seed(seed) -> None:
    np.random.seed(seed)


class TestSampleFromURM:
    test_num_samples = 100_000
    test_seed = 1234
    test_frac_negative_sampling = 0.3

    def test_function_sample_only_positives_returns_samples_in_urm(
        self, urm: scipy_sparse.csr_matrix, num_users: int, num_items: int,
    ):
        # Arrange
        urm_coo = sparse.COO.from_scipy_sparse(urm)
        urm_dok = sparse.DOK.from_scipy_sparse(urm)

        # Act
        set_seed(seed=self.test_seed)
        (
            arr_user_ids,
            arr_item_ids,
            arr_ratings,
        ) = nb_sample_only_positives_from_sparse_coo_matrix(
            num_users=num_users,
            num_items=num_items,
            num_samples=self.test_num_samples,
            urm=urm_coo,
        )

        # Assert
        assert np.all(
            urm_dok[user_id, item_id] == rating
            for user_id, item_id, rating in zip(arr_user_ids, arr_item_ids, arr_ratings)
        ) and np.all(arr_ratings == 1)

    def test_function_sample_only_negatives_returns_samples_not_in_urm(
        self, urm: scipy_sparse.csr_matrix, num_users: int, num_items: int,
    ):
        # Arrange
        urm_dok = sparse.DOK.from_scipy_sparse(urm)

        # Act
        set_seed(seed=self.test_seed)
        (
            arr_user_ids,
            arr_item_ids,
            arr_ratings,
        ) = nb_sample_only_negatives_from_sparse_csr_matrix(
            num_users=num_users,
            num_items=num_items,
            num_samples=self.test_num_samples,
            urm_csr_indptr=urm.indptr,
            urm_csr_indices=urm.indices
        )

        # Assert
        assert np.all(
            urm_dok[user_id, item_id] == rating
            for user_id, item_id, rating in zip(arr_user_ids, arr_item_ids, arr_ratings)
        ) and np.all(arr_ratings == 0)

    def test_function_sample_positive_and_negatives_return_correct_samples(
        self, urm: scipy_sparse.csr_matrix, num_users: int, num_items: int,
    ):
        # Arrange
        urm_coo = sparse.COO.from_scipy_sparse(urm)
        urm_dok = sparse.DOK.from_scipy_sparse(urm)

        # Act
        set_seed(seed=self.test_seed)
        (
            arr_user_ids,
            arr_item_ids,
            arr_ratings,
        ) = nb_sample_positives_and_negatives_from_sparse_matrix(
            num_users=num_users,
            num_items=num_items,
            num_samples=self.test_num_samples,
            urm_coo=urm_coo,
            urm_csr_indptr=urm.indptr,
            urm_csr_indices=urm.indices,
            frac_negative_sampling=self.test_frac_negative_sampling,
        )

        # Assert
        assert np.all(
            urm_dok[user_id, item_id] == rating
            for user_id, item_id, rating in zip(arr_user_ids, arr_item_ids, arr_ratings)
        )
        assert np.isclose(
            self.test_frac_negative_sampling,
            arr_ratings[arr_ratings == 0.].size / self.test_num_samples,
        )
