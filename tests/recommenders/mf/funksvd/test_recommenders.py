import scipy.sparse
import sparse
import time

from recsys_framework_extensions.recommenders.mf.funksvd.recommender import (
    MatrixFactorizationFunkSVD,
)
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_FunkSVD_Cython


class TestTrainMFFunkSVD:
    num_factors = 150
    num_epochs = 1000
    batch_size = 1_000
    learning_rate = 1e-1
    test_frac_negative_sampling = 0.5
    seed = 1234567890
    reg_user = 1e-2
    reg_item = 1e-2
    reg_bias = 1e-1
    sgd_mode = "sgd"
    sgd_gamma = 0.995
    sgd_beta_1 = 0.9
    sgd_beta_2 = 0.999
    use_bias = True
    init_mean = 0.
    init_std_dev = 1.

    def test_extended_is_faster(
        self, urm: scipy.sparse.csr_matrix, num_users: int, num_items: int,
    ):
        # Arrange
        recsys_funk_svd = MatrixFactorization_FunkSVD_Cython(
            urm,
        )
        extended_funk_svd = MatrixFactorizationFunkSVD(
            urm_train=urm,
        )

        #jit-compile
        extended_funk_svd.fit(
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            init_mean=self.init_mean,
            init_std_dev=self.init_std_dev,
            learning_rate=self.learning_rate,
            num_factors=self.num_factors,
            random_seed=self.seed,

            reg_user=self.reg_user,
            reg_item=self.reg_item,
            reg_bias=self.reg_bias,

            quota_negative_interactions=self.test_frac_negative_sampling,
            quota_dropout=None,

            sgd_mode=self.sgd_mode,
            sgd_gamma=self.sgd_gamma,
            sgd_beta_1=self.sgd_beta_1,
            sgd_beta_2=self.sgd_beta_2,

            use_bias=self.use_bias,
            use_embeddings=True,
        )
        extended_funk_svd = MatrixFactorizationFunkSVD(
            urm_train=urm,
        )

        # Act
        print("Start RecSys")
        recsys_start = time.time()
        recsys_funk_svd.fit(
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            num_factors=self.num_factors,
            positive_threshold_BPR=None,
            learning_rate=self.learning_rate,
            use_bias=self.use_bias,
            use_embeddings=True,
            sgd_mode=self.sgd_mode,
            negative_interactions_quota=self.test_frac_negative_sampling,
            dropout_quota=None,
            init_mean=self.init_mean,
            init_std_dev=self.init_std_dev,
            user_reg=self.reg_user,
            item_reg=self.reg_item,
            bias_reg=self.reg_bias,
            positive_reg=0.0,
            negative_reg=0.0,
            random_seed=self.seed,
        )
        recsys_end = time.time()

        print("Start Extended")
        extended_start = time.time()
        extended_funk_svd.fit(
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            init_mean=self.init_mean,
            init_std_dev=self.init_std_dev,
            learning_rate=self.learning_rate,
            num_factors=self.num_factors,
            random_seed=self.seed,

            reg_user=self.reg_user,
            reg_item=self.reg_item,
            reg_bias=self.reg_bias,

            quota_negative_interactions=self.test_frac_negative_sampling,
            quota_dropout=None,

            sgd_mode=self.sgd_mode,
            sgd_gamma=self.sgd_gamma,
            sgd_beta_1=self.sgd_beta_1,
            sgd_beta_2=self.sgd_beta_2,

            use_bias=self.use_bias,
            use_embeddings=True,
        )
        extended_end = time.time()

        # Assert
        time_recsys = recsys_end - recsys_start
        time_extended = extended_end - extended_start
        print(f"{time_extended=:.2f} - {time_recsys=:.2f}")
        assert time_extended < time_recsys
