# import logging
# import numpy as np
# import scipy.sparse
# import time
#
# import sparse
#
# from recsys_framework_extensions.decorators import timeit
# from recsys_framework_extensions.recommenders.mf.funksvd.recommender import MatrixFactorizationFunkSVD
# from recsys_framework_extensions.recommenders.mf.funksvd.training import (
#     NumbaFunkSVDOptimizer,
#     run_epoch_funk_svd,
#     init_mf_funk_svd,
#     train_mf_funk_svd, nb_sample_funk_svd_from_sparse_matrix,
# )
#
#
# @timeit
# def compile_numba_sample_funk_svd():
#     num_users, num_items, num_interactions = 1000, 10000, 100000
#     batch_size = 1000
#     num_factors = 100
#     num_epochs = 10
#     init_mean = 0.
#     init_std_dev = 1.
#     learning_rate = 0.001
#     quota_negative_interactions = 0.6
#     reg_user = 0.01
#     reg_item = 0.01
#     reg_bias = 0.01
#     sgd_mode = "sgd"
#     sgd_gamma = 0.99
#     sgd_beta_1 = 0.99
#     sgd_beta_2 = 0.99
#     seed = 1234
#     use_bias = True
#
#     np.random.seed(seed)
#     arr_users = np.random.random_integers(low=0, high=num_users - 1, size=num_interactions)
#     arr_items = np.random.random_integers(low=0, high=num_items - 1, size=num_interactions)
#     arr_data = np.ones_like(arr_users)
#     urm = scipy.sparse.coo_matrix(
#         (arr_data, (arr_users, arr_items)),
#         dtype=np.float32,
#         shape=(num_users, num_items),
#     ).tocsr()
#     urm.sum_duplicates()
#     urm_coo = sparse.COO.from_scipy_sparse(urm)
#
#     (
#         arr_user_ids,
#         arr_item_ids,
#         arr_ratings,
#     ) = nb_sample_funk_svd_from_sparse_matrix(
#         num_users=num_users,
#         num_items=num_items,
#         num_samples=urm.nnz,
#         urm_coo=urm_coo,
#         urm_csr_indptr=urm.indptr,
#         urm_csr_indices=urm.indices,
#         frac_negative_sampling=quota_negative_interactions
#     )
#
#
# @timeit
# def compile_numba_recommender_funk_svd():
#     num_users, num_items, num_interactions = 1000, 10000, 100000
#     batch_size = 1000
#     num_factors = 100
#     num_epochs = 10
#     init_mean = 0.
#     init_std_dev = 1.
#     learning_rate = 0.001
#     quota_negative_interactions = 0.6
#     reg_user = 0.01
#     reg_item = 0.01
#     reg_bias = 0.01
#     sgd_mode = "sgd"
#     sgd_gamma = 0.99
#     sgd_beta_1 = 0.99
#     sgd_beta_2 = 0.99
#     seed = 1234
#     use_bias = True
#
#     np.random.seed(seed)
#     arr_users = np.random.random_integers(low=0, high=num_users - 1, size=num_interactions)
#     arr_items = np.random.random_integers(low=0, high=num_items - 1, size=num_interactions)
#     arr_data = np.ones_like(arr_users)
#     urm = scipy.sparse.coo_matrix(
#         (arr_data, (arr_users, arr_items)),
#         dtype=np.float32,
#         shape=(num_users, num_items),
#     ).tocsr()
#     urm.sum_duplicates()
#
#     MatrixFactorizationFunkSVD(
#         urm_train=urm,
#     ).fit(
#         batch_size=batch_size,
#         epochs=num_epochs,
#         init_mean=init_mean,
#         init_std_dev=init_std_dev,
#         learning_rate=learning_rate,
#         num_factors=num_factors,
#         random_seed=seed,
#
#         reg_user=reg_user,
#         reg_item=reg_item,
#         reg_bias=reg_bias,
#
#         quota_negative_interactions=quota_negative_interactions,
#         quota_dropout=None,
#
#         sgd_mode=sgd_mode,
#         sgd_gamma=sgd_gamma,
#         sgd_beta_1=sgd_beta_1,
#         sgd_beta_2=sgd_beta_2,
#
#         use_bias=use_bias,
#         use_embeddings=True,
#     )
#
#
# @timeit
# def compile_numba_functions_funk_svd():
#     num_users, num_items, num_interactions = 1000, 10000, 100000
#     batch_size = 1000
#     num_factors = 100
#     num_epochs = 10
#     init_mean = 0.
#     init_std_dev = 1.
#     learning_rate = 0.001
#     quota_negative_interactions = 0.6
#     reg_user = 0.01
#     reg_item = 0.01
#     reg_bias = 0.01
#     sgd_mode = "sgd"
#     sgd_gamma = 0.99
#     sgd_beta_1 = 0.99
#     sgd_beta_2 = 0.99
#     seed = 1234
#     use_bias = True
#
#     np.random.seed(seed)
#     arr_users = np.random.random_integers(low=0, high=num_users - 1, size=num_interactions)
#     arr_items = np.random.random_integers(low=0, high=num_items - 1, size=num_interactions)
#     arr_data = np.ones_like(arr_users)
#     urm = scipy.sparse.coo_matrix(
#         (arr_data, (arr_users, arr_items)),
#         dtype=np.float32,
#         shape=(num_users, num_items),
#     ).tocsr()
#     urm.sum_duplicates()
#
#     train_mf_funk_svd(
#         urm_train=urm,
#
#         num_users=num_users,
#         num_items=num_items,
#         num_factors=num_factors,
#         num_epochs=num_epochs,
#
#         batch_size=batch_size,
#         learning_rate=learning_rate,
#
#         seed=seed,
#
#         reg_user=reg_user,
#         reg_item=reg_item,
#         reg_bias=reg_bias,
#
#         frac_negative_sampling=quota_negative_interactions,
#
#         # sgd_mode=sgd_mode,
#         # sgd_gamma=sgd_gamma,
#         # sgd_beta_1=sgd_beta_1,
#         # sgd_beta_2=sgd_beta_2,
#
#         use_bias=use_bias,
#     )
#
#
# logging.info(
#     f"Jit-compiling numba functions for MF FunkSVD"
# )
# compile_numba_functions_funk_svd()
# compile_numba_functions_funk_svd()
#
# compile_numba_recommender_funk_svd()
# compile_numba_recommender_funk_svd()
#
# compile_numba_sample_funk_svd()
# compile_numba_sample_funk_svd()
