import sys
import time

import attrs
import numpy as np
import scipy.sparse as sp
import sklearn
from Recommenders.BaseSimilarityMatrixRecommender import (
    BaseItemSimilarityMatrixRecommender,
)
from Recommenders.Recommender_utils import similarityMatrixTopK, check_matrix
from Recommenders.Similarity.Compute_Similarity_Python import (
    Incremental_Similarity_Builder,
)
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
from sklearn.preprocessing import normalize
from skopt.space import Integer, Real, Categorical

from recsys_framework_extensions.recommenders.base import (
    SearchHyperParametersBaseRecommender,
)


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersRP3BetaRecommender(SearchHyperParametersBaseRecommender):
    top_k: Integer = attrs.field(
        default=Integer(
            low=5,
            high=1000,
            prior="uniform",
            base=10,
        )
    )
    alpha: Real = attrs.field(
        default=Real(
            low=0,
            high=2,
            prior="uniform",
            base=10,
        )
    )
    beta: Real = attrs.field(
        default=Real(
            low=0,
            high=2,
            prior="uniform",
            base=10,
        )
    )
    normalize_similarity: Real = attrs.field(
        default=Categorical(
            [True, False],
        ),
    )


class ExtendedRP3BetaRecommender(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "ExtendedRP3BetaRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            URM_train=urm_train,
            verbose=verbose,
        )
        self.p_ui = sp.csr_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items),
            dtype=np.int32,
        )
        self.p_iu = sp.csr_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items),
            dtype=np.int32,
        )
        self.arr_degree = np.zeros(
            shape=self.n_items,
            dtype=np.float32,
        )

        self.W_sparse: sp.csr_matrix = sp.csr_matrix(
            (self.n_items, self.n_items),
            dtype=np.float32,
        )

        self.alpha: float = 1.0
        self.beta: float = 0.6
        self.top_k: int = 100
        self.normalize_similarity: bool = False

    def __str__(
        self,
    ) -> str:
        return (
            f"ExtendedRP3Beta("
            f"alpha={self.alpha}, "
            f"beta={self.beta}, "
            f"top_k={self.top_k}, "
            f"normalize_similarity={self.normalize_similarity}"
            f")"
        )

    def create_degree_array(
        self,
    ):
        _, num_items = self.URM_train.shape

        X_bool = self.URM_train.transpose(
            copy=True,
        )
        X_bool.data = np.ones(
            shape=X_bool.data.size,
            dtype=np.float32,
        )
        # Taking the degree of each item to penalize top popular
        # Some rows might be zero, make sure their degree remains zero
        X_bool_sum = np.array(X_bool.sum(axis=1)).ravel()
        non_zero_mask = X_bool_sum != 0.0
        arr_degree = np.zeros(
            shape=num_items,
            dtype=np.float32,
        )
        arr_degree[non_zero_mask] = np.power(
            X_bool_sum[non_zero_mask],
            -self.beta,
        )

        self.arr_degree = arr_degree

    def create_adjacency_matrix(
        self,
    ):
        # Pui is the row-normalized urm
        Pui = normalize(
            self.URM_train,
            norm="l1",
            axis=1,
        )

        # Piu is the column-normalized, "boolean" urm transposed
        X_bool = self.URM_train.transpose(
            copy=True,
        )
        X_bool.data = np.ones(
            X_bool.data.size,
            dtype=np.float32,
        )
        # ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(
            X_bool,
            norm="l1",
            axis=1,
        )

        # Alfa power
        if self.alpha != 1.0:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        self.p_ui = Pui
        self.p_iu = Piu

    def create_similarity_matrix(
        self,
    ):
        # Final matrix is computed as self.p_ui * Piu * self.p_ui
        # Multiplication unpacked for memory usage reasons
        block_dim = 200
        d_t = self.p_iu

        similarity_builder = Incremental_Similarity_Builder(
            self.p_ui.shape[1],
            initial_data_block=self.p_ui.shape[1] * self.top_k,
            dtype=np.float32,
        )

        start_time = time.time()
        start_time_print_batch = start_time

        for current_block_start_row in range(0, self.p_ui.shape[1], block_dim):
            if current_block_start_row + block_dim > self.p_ui.shape[1]:
                block_dim = self.p_ui.shape[1] - current_block_start_row

            similarity_block = (
                d_t[current_block_start_row : current_block_start_row + block_dim, :]
                * self.p_ui
            )
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = np.multiply(
                    similarity_block[row_in_block, :], self.arr_degree
                )
                row_data[current_block_start_row + row_in_block] = 0

                relevant_items_partition = np.argpartition(
                    -row_data,
                    self.top_k - 1,
                    axis=0,
                )[: self.top_k]
                row_data = row_data[relevant_items_partition]

                # Incrementally build sparse matrix, do not add zeros
                if np.any(row_data == 0.0):
                    non_zero_mask = row_data != 0.0
                    relevant_items_partition = relevant_items_partition[non_zero_mask]
                    row_data = row_data[non_zero_mask]

                similarity_builder.add_data_lists(
                    row_list_to_add=np.ones(len(row_data), dtype=np.int32)
                    * (current_block_start_row + row_in_block),
                    col_list_to_add=relevant_items_partition,
                    data_list_to_add=row_data,
                )

            if (
                time.time() - start_time_print_batch > 300
                or current_block_start_row + block_dim == self.p_ui.shape[1]
            ):
                new_time_value, new_time_unit = seconds_to_biggest_unit(
                    time.time() - start_time
                )

                self._print(
                    f"Similarity column {current_block_start_row + block_dim} "
                    f"({100.0 * float(current_block_start_row + block_dim) / self.p_ui.shape[1]:4.1f}%), "
                    f"{float(current_block_start_row + block_dim) / (time.time() - start_time):.2f} "
                    f"column/sec. "
                    f"Elapsed time {new_time_value:.2f} {new_time_unit}"
                )

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_print_batch = time.time()

        w_sparse = similarity_builder.get_SparseMatrix()

        if self.normalize_similarity:
            w_sparse = sklearn.preprocessing.normalize(
                w_sparse,
                norm="l1",
                axis=1,
            )

        if self.top_k:
            w_sparse = similarityMatrixTopK(
                w_sparse,
                k=self.top_k,
            )

        self.W_sparse = check_matrix(
            w_sparse,
            format="csr",
        )

    def fit(
        self,
        *,
        top_k: int = 100,
        alpha: float = 1.0,
        beta: float = 0.6,
        normalize_similarity: bool = False,
        **kwargs,
    ) -> None:
        self.top_k = int(top_k)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.normalize_similarity = bool(normalize_similarity)

        self.create_adjacency_matrix()
        self.create_degree_array()
        self.create_similarity_matrix()
