import scipy.sparse
import numpy as np

from recsys_framework_extensions.recommenders.mf.funksvd.training import (
    train_mf_funk_svd,
    NumbaFunkSVDOptimizer,
)


class TestTrainMFFunkSVD:
    num_factors = 50
    num_epochs = 10
    batch_size = 1_000
    learning_rate = 1e-1
    test_frac_negative_sampling = 1
    seed = 1234567890
    reg_user = 1e-2
    reg_item = 1e-2
    reg_bias = 1e-1
    sgd_mode = "sgd"
    sgd_gamma = 0.9
    sgd_beta_1 = 0.9
    sgd_beta_2 = 0.999
    use_bias = True

    def test_consecutive_calls_are_equal(
        self, urm: scipy.sparse.csr_matrix, num_users: int, num_items: int,
    ):
        # Arrange
        optimizer = NumbaFunkSVDOptimizer(
            num_users=num_users,
            num_items=num_items,
            num_factors=self.num_factors,
            sgd_mode=self.sgd_mode,
            gamma=self.sgd_gamma,
            beta_1=self.sgd_beta_1,
            beta_2=self.sgd_beta_2,

        )
        optimizer_2 = NumbaFunkSVDOptimizer(
            num_users=num_users,
            num_items=num_items,
            num_factors=self.num_factors,
            sgd_mode=self.sgd_mode,
            gamma=self.sgd_gamma,
            beta_1=self.sgd_beta_1,
            beta_2=self.sgd_beta_2,
        )

        # Act
        (
            result_user_embeddings,
            result_item_embeddings,

            result_bias_global,
            result_bias_users,
            result_bias_items,

            result_loss,
        ) = train_mf_funk_svd(
            urm_train=urm,
            num_users=num_users,
            num_items=num_items,

            num_factors=self.num_factors,
            num_epochs=self.num_epochs,

            batch_size=self.batch_size,
            frac_negative_sampling=self.test_frac_negative_sampling,
            learning_rate=self.learning_rate,
            optimizer=optimizer,

            reg_user=self.reg_user,
            reg_item=self.reg_item,
            reg_bias=self.reg_bias,

            use_bias=self.use_bias,

            seed=self.seed,
        )

        (
            result_user_embeddings_2,
            result_item_embeddings_2,
            result_bias_global_2,
            result_bias_users_2,
            result_bias_items_2,
            result_loss_2,
        ) = train_mf_funk_svd(
            urm_train=urm,
            num_users=num_users,
            num_items=num_items,

            num_factors=self.num_factors,
            num_epochs=self.num_epochs,

            batch_size=self.batch_size,
            frac_negative_sampling=self.test_frac_negative_sampling,
            learning_rate=self.learning_rate,
            optimizer=optimizer_2,

            reg_user=self.reg_user,
            reg_item=self.reg_item,
            reg_bias=self.reg_bias,

            use_bias=self.use_bias,


            seed=self.seed,
        )

        # Assert
        assert np.array_equal(result_user_embeddings, result_user_embeddings_2)
        assert np.array_equal(result_item_embeddings, result_item_embeddings_2)
        assert np.array_equal(result_bias_global, result_bias_global_2)
        assert np.array_equal(result_bias_users, result_bias_users_2)
        assert np.array_equal(result_bias_items, result_bias_items_2)
        assert np.array_equal(result_loss, result_loss_2)
