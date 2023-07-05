import scipy.sparse
import sparse
import time

from recsys_framework_extensions.recommenders.mf.funksvd.recommender_julia import (
    MatrixFactorizationFunkSVD,
)


class TestTrainMFFunkSVD:
    def test_runs(
        self, urm: scipy.sparse.csr_matrix, num_users: int, num_items: int,
    ):
        MatrixFactorizationFunkSVD
