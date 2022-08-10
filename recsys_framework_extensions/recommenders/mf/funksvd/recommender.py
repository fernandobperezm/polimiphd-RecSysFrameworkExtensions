from typing import Optional

import sparse
from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping

from recsys_framework_extensions.recommenders.mf.funksvd.training import (
    init_mf_funk_svd,
    run_epoch_funk_svd,
    NumbaFunkSVDOptimizer,
)

import scipy.sparse
import numpy as np


class MatrixFactorizationFunkSVD(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):
    RECOMMENDER_NAME = "MatrixFactorization_Numba_Recommender"

    def __init__(
        self,
        urm_train: scipy.sparse.csr_matrix,
        verbose: bool = True,
    ):
        super().__init__(
            urm_train,
            verbose=verbose
        )

        self.urm_coo = sparse.COO.from_scipy_sparse(self.URM_train)

        self.n_users, self.n_items = self.URM_train.shape
        self.num_users, self.num_items = self.URM_train.shape
        self.num_samples: int = urm_train.nnz

        self.normalize = False
        self.algorithm_name = "MF FunkSVD"

        self.epochs: int = 300
        self.batch_size: int = 1000
        self.learning_rate: float = 0.001

        self.init_mean: float = 0.
        self.init_std_dev: float = 1.
        self.num_factors: int = 10

        self.quota_negative_interactions: float = 0.
        self.quota_dropout: Optional[float] = None

        self.reg_user: float = 0.
        self.reg_item: float = 0.
        self.reg_bias: float = 0.

        self.random_seed: int = 1234

        self.sgd_mode = "sgd"
        self.sgd_gamma = 0.9
        self.sgd_beta_1 = 0.9
        self.sgd_beta_2 = 0.999

        self.use_bias: bool = True
        self.use_embeddings: bool = True

        self.optimizer: NumbaFunkSVDOptimizer = NumbaFunkSVDOptimizer(
            sgd_mode=self.sgd_mode,
            num_users=self.num_users,
            num_items=self.num_items,
            num_factors=self.num_factors,
            gamma=self.sgd_gamma,
            beta_1=self.sgd_beta_1,
            beta_2=self.sgd_beta_2,
        )

        self.USER_factors = np.empty(shape=(self.num_users, self.num_factors), dtype=np.float64)
        self.ITEM_factors = np.empty(shape=(self.num_items, self.num_factors), dtype=np.float64)
        self.USER_bias = np.zeros(shape=(self.num_items,), dtype=np.float64)
        self.ITEM_bias = np.zeros(shape=(self.num_items,), dtype=np.float64)
        self.GLOBAL_bias = np.zeros(shape=(1,), dtype=np.float64)

    def fit(
        self,
        *,
        epochs=300,
        batch_size=1000,
        learning_rate=0.001,

        num_factors=10,
        init_mean=0.0,
        init_std_dev=0.1,

        quota_negative_interactions=0.0,
        quota_dropout=None,

        reg_user=0.0,
        reg_item=0.0,
        reg_bias=0.0,

        sgd_mode='sgd',
        sgd_gamma=0.9,
        sgd_beta_1=0.9,
        sgd_beta_2=0.999,

        use_bias=True,
        use_embeddings=True,

        random_seed=None,
        **early_stopping_kwargs,
    ):
        assert 0 <= quota_negative_interactions < 1.0, \
            "{}: negative_interactions_quota must be a float value >=0 and < 1.0, provided was '{}'".format(
                self.RECOMMENDER_NAME,
                quota_negative_interactions,
            )

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.num_factors = num_factors

        self.reg_user = reg_user
        self.reg_item = reg_item
        self.reg_bias = reg_bias

        self.sgd_mode = sgd_mode
        self.sgd_gamma = sgd_gamma
        self.sgd_beta_1 = sgd_beta_1
        self.sgd_beta_2 = sgd_beta_2

        self.optimizer: NumbaFunkSVDOptimizer = NumbaFunkSVDOptimizer(
            sgd_mode=self.sgd_mode,
            num_users=self.num_users,
            num_items=self.num_items,
            num_factors=self.num_factors,
            gamma=self.sgd_gamma,
            beta_1=self.sgd_beta_1,
            beta_2=self.sgd_beta_2,
        )

        self.quota_dropout = quota_dropout
        self.quota_negative_interactions = quota_negative_interactions

        self.use_bias = use_bias
        self.use_embeddings = use_embeddings

        self.random_seed = random_seed

        (
            self.USER_factors,
            self.ITEM_factors,
            self.GLOBAL_bias,
            self.USER_bias,
            self.ITEM_bias,
        ) = init_mf_funk_svd(
            embeddings_mean=self.init_mean,
            embeddings_std_dev=self.init_std_dev,
            num_users=self.num_users,
            num_items=self.num_items,
            num_factors=self.num_factors,
            seed=self.random_seed,
        )

        self._prepare_model_for_validation()
        self._update_best_model()
        self._train_with_early_stopping(
            epochs,
            algorithm_name=self.algorithm_name,
            **early_stopping_kwargs,
        )

        self.USER_factors = self.USER_factors_best
        self.ITEM_factors = self.ITEM_factors_best

        if self.use_bias:
            self.USER_bias = self.USER_bias_best
            self.ITEM_bias = self.ITEM_bias_best
            self.GLOBAL_bias = self.GLOBAL_bias_best

    def _prepare_model_for_validation(self):
        pass

    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()

        if self.use_bias:
            self.USER_bias_best = self.USER_bias.copy()
            self.ITEM_bias_best = self.ITEM_bias.copy()
            self.GLOBAL_bias_best = self.GLOBAL_bias

    def _run_epoch(
        self,
        num_epoch: int,
    ):
        (
            self.USER_factors,
            self.ITEM_factors,
            self.GLOBAL_bias,
            self.USER_bias,
            self.ITEM_bias,
        ) = run_epoch_funk_svd(
            num_users=self.num_users,
            num_items=self.num_items,
            num_samples=self.num_samples,

            user_embeddings=self.USER_factors,
            item_embeddings=self.ITEM_factors,

            bias_global=self.GLOBAL_bias,
            bias_users=self.USER_bias,
            bias_items=self.ITEM_bias,

            urm_coo=self.urm_coo,
            urm_csr_indptr=self.URM_train.indptr,
            urm_csr_indices=self.URM_train.indices,

            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            num_factors=self.num_factors,

            reg_user=self.reg_user,
            reg_item=self.reg_item,
            reg_bias=self.reg_bias,

            # optimizer=self.optimizer,
            quota_negative_interactions=self.quota_negative_interactions,
            use_bias=self.use_bias,
        )
