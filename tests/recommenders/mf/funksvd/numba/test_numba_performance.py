import time
import numpy as np
import scipy
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_FunkSVD_Cython

from recsys_framework_extensions.recommenders.mf.funksvd.numba.mf_funk_svd import train_mf_funk_svd
from recsys_framework_extensions.recommenders.mf.funksvd.numba.mf_funk_svd_class import NumbaFunkSVDTrainer, train_instance
from recsys_framework_extensions.recommenders.mf.funksvd.numba.mf_funk_svd_v2 import train_mf_funk_svd as \
    train_mf_funk_svd_v2
from recsys_framework_extensions.recommenders.mf.funksvd.numba.mf_funk_svd_namedtuples import \
    train_mf_funk_svd as train_mf_funk_svd_namedtuples, init_optimizers


class TestNumbaImplPerformance:
    num_factors = 150
    num_epochs = 100
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
        urm = urm.astype(np.float32)
        urm.sort_indices()

        opt_compile = init_optimizers(
            sgd_mode=self.sgd_mode,
            gamma=self.sgd_gamma,
            beta_1=self.sgd_beta_1,
            beta_2=self.sgd_beta_2,
            num_users=num_users,
            num_items=num_items,
            num_factors=self.num_factors,
        )
        opt_train = init_optimizers(
            sgd_mode=self.sgd_mode,
            gamma=self.sgd_gamma,
            beta_1=self.sgd_beta_1,
            beta_2=self.sgd_beta_2,
            num_users=num_users,
            num_items=num_items,
            num_factors=self.num_factors,
        )
        recsys_funk_svd = MatrixFactorization_FunkSVD_Cython(
            urm,
        )
        numba_trainer = NumbaFunkSVDTrainer(
            embeddings_mean=self.init_mean,
            embeddings_std_dev=self.init_std_dev,
            num_users=num_users,
            num_items=num_items,
            reg_user=self.reg_user,
            reg_item=self.reg_item,
            reg_bias=self.reg_bias,
            urm_csr_indices=urm.indices,
            urm_csr_indptr=urm.indptr,
            urm_csr_data=urm.data,
            batch_size=self.batch_size,
            frac_negative_sampling=self.test_frac_negative_sampling,
            learning_rate=self.learning_rate,
            num_epochs=self.num_epochs,
            num_factors=self.num_factors,
            num_samples=urm.nnz,
            use_bias=self.use_bias,
            seed=self.seed,
        )
        train_instance(
            instance=numba_trainer,
        )

        # jit-compile the main numba functions
        train_mf_funk_svd(
            urm_train=urm,
            num_users=num_users,
            num_items=num_items,
            num_factors=self.num_factors,
            num_epochs=self.num_epochs,

            reg_user=self.reg_user,
            reg_item=self.reg_item,
            reg_bias=self.reg_bias,

            batch_size=self.batch_size,
            frac_negative_sampling=self.test_frac_negative_sampling,
            learning_rate=self.learning_rate,
            use_bias=self.use_bias,
            seed=self.seed,
        )
        train_mf_funk_svd_v2(
            urm_train=urm,
            num_users=num_users,
            num_items=num_items,
            num_factors=self.num_factors,
            num_epochs=self.num_epochs,

            reg_user=self.reg_user,
            reg_item=self.reg_item,
            reg_bias=self.reg_bias,

            batch_size=self.batch_size,
            frac_negative_sampling=self.test_frac_negative_sampling,
            learning_rate=self.learning_rate,
            use_bias=self.use_bias,
            seed=self.seed,
        )
        train_mf_funk_svd_namedtuples(
            urm_train=urm,
            num_users=num_users,
            num_items=num_items,
            num_factors=self.num_factors,
            num_epochs=self.num_epochs,
            embeddings_mean=self.init_mean,
            embeddings_std_dev=self.init_std_dev,

            reg_user=self.reg_user,
            reg_item=self.reg_item,
            reg_bias=self.reg_bias,

            batch_size=self.batch_size,
            frac_negative_sampling=self.test_frac_negative_sampling,
            learning_rate=self.learning_rate,
            use_bias=self.use_bias,
            seed=self.seed,
            optimizers=opt_compile,
        )

        # Act
        print("Start V1")
        start_v1 = time.time()
        results_v1 = train_mf_funk_svd(
            urm_train=urm,
            num_users=num_users,
            num_items=num_items,
            num_factors=self.num_factors,
            num_epochs=self.num_epochs,

            reg_user=self.reg_user,
            reg_item=self.reg_item,
            reg_bias=self.reg_bias,

            batch_size=self.batch_size,
            frac_negative_sampling=self.test_frac_negative_sampling,
            learning_rate=self.learning_rate,
            use_bias=self.use_bias,
            seed=self.seed,
        )
        end_v1 = time.time()

        print("Start V2")
        start_v2 = time.time()
        results_v2 = train_mf_funk_svd_v2(
            urm_train=urm,
            num_users=num_users,
            num_items=num_items,
            num_factors=self.num_factors,
            num_epochs=self.num_epochs,

            reg_user=self.reg_user,
            reg_item=self.reg_item,
            reg_bias=self.reg_bias,

            batch_size=self.batch_size,
            frac_negative_sampling=self.test_frac_negative_sampling,
            learning_rate=self.learning_rate,
            use_bias=self.use_bias,
            seed=self.seed,
        )
        end_v2 = time.time()

        print("Start Numba FunkSVD Class")
        start_numba_class = time.time()
        train_instance(
            instance=numba_trainer,
        )
        end_numba_class = time.time()
        for epoch, loss in enumerate(numba_trainer.arr_epoch_losses):
            print(f"Epoch: {epoch} - loss: {loss}")

        print("Start Numba FunkSVD NamedTuples")
        start_namedtuples = time.time()
        results_namedtuples = train_mf_funk_svd_namedtuples(
            urm_train=urm,
            num_users=num_users,
            num_items=num_items,
            num_factors=self.num_factors,
            num_epochs=self.num_epochs,
            embeddings_mean=self.init_mean,
            embeddings_std_dev=self.init_std_dev,

            reg_user=self.reg_user,
            reg_item=self.reg_item,
            reg_bias=self.reg_bias,

            optimizers=opt_train,

            batch_size=self.batch_size,
            frac_negative_sampling=self.test_frac_negative_sampling,
            learning_rate=self.learning_rate,
            use_bias=self.use_bias,
            seed=self.seed,
        )
        end_namedtuples = time.time()
        for epoch, loss in enumerate(results_namedtuples[2]):
            print(f"Epoch: {epoch} - loss: {loss}")

        print("Start RecSys")
        start_recsys = time.time()
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
        end_recsys = time.time()

        # Assert
        time_namedtuples = end_namedtuples - start_namedtuples
        time_recsys = end_recsys - start_recsys
        assert time_namedtuples <= time_recsys
