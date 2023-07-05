from typing import Union

import numpy as np
import numba as nb
import scipy.sparse
import sparse

from recsys_framework_extensions.optimizer import (
    NumbaOptimizer,
)


@nb.experimental.jitclass
class NumbaFunkSVDOptimizer:
    optimizer_embeddings_users: NumbaOptimizer
    optimizer_embeddings_items: NumbaOptimizer
    optimizer_bias_global: NumbaOptimizer
    optimizer_bias_users: NumbaOptimizer
    optimizer_bias_items: NumbaOptimizer

    def __init__(
        self,
        sgd_mode: str,
        num_users: int,
        num_items: int,
        num_factors: int,
        gamma: float,
        beta_1: float,
        beta_2: float,
    ):
        self.optimizer_embeddings_users = NumbaOptimizer(
            sgd_mode,
            (num_users, num_factors),
            gamma,
            beta_1,
            beta_2,
        )
        self.optimizer_embeddings_items = NumbaOptimizer(
            sgd_mode,
            (num_items, num_factors),
            gamma,
            beta_1,
            beta_2,
        )
        self.optimizer_bias_global = NumbaOptimizer(
            sgd_mode,
            (1, 1),
            gamma,
            beta_1,
            beta_2,
        )
        self.optimizer_bias_users = NumbaOptimizer(
            sgd_mode,
            (num_users, 1),
            gamma,
            beta_1,
            beta_2,
        )
        self.optimizer_bias_items = NumbaOptimizer(
            sgd_mode,
            (num_items, 1),
            gamma,
            beta_1,
            beta_2,
        )

    def after_batch(self):
        self.optimizer_bias_global.after_batch()
        self.optimizer_bias_users.after_batch()
        self.optimizer_bias_items.after_batch()
        self.optimizer_embeddings_items.after_batch()
        self.optimizer_embeddings_users.after_batch()


@nb.njit
def nb_numpy_seed(
    seed: int,
):
    np.random.seed(seed)


@nb.njit
def create_embeddings_from_normal_distribution(
    mean: float,
    std_dev: float,
    shape: Union[tuple[int], tuple[int, int]],
) -> np.ndarray:
    return np.random.normal(
        loc=mean,
        scale=std_dev,
        size=shape,
    )


@nb.njit
def nb_sample_funk_svd_user(
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
def nb_sample_funk_svd_positive_item(
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
def nb_sample_funk_svd_negative_item(
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


@nb.njit
def nb_sample_funk_svd_from_sparse_matrix(
    num_users: int,
    num_items: int,
    urm_csr_indptr: np.ndarray,
    urm_csr_indices: np.ndarray,
    urm_csr_data: np.ndarray,
    frac_negative_sampling: float,
) -> tuple[int, int, float]:
    sample_user, user_profile_start, user_profile_end, num_seen_items = nb_sample_funk_svd_user(
        num_users=num_users,
        num_items=num_items,
        urm_csr_indptr=urm_csr_indptr,
    )

    sample_positive = np.random.random() <= frac_negative_sampling

    if sample_positive:
        sample_item, sample_rating = nb_sample_funk_svd_positive_item(
            num_seen_items=num_seen_items,
            user_profile_start=user_profile_start,
            urm_csr_indices=urm_csr_indices,
            urm_csr_data=urm_csr_data,
        )
    else:
        sample_item, sample_rating = nb_sample_funk_svd_negative_item(
            num_items=num_items,
            num_seen_items=num_seen_items,
            user_profile_start=user_profile_start,
            urm_csr_indices=urm_csr_indices,
        )

    return sample_user, sample_item, sample_rating


@nb.njit
def nb_epoch_mf_funk_svd(
    embeddings_users: np.ndarray,
    embeddings_items: np.ndarray,

    bias_global: np.ndarray,
    bias_users: np.ndarray,
    bias_items: np.ndarray,

    num_users: int,
    num_items: int,
    num_factors: int,

    reg_user: float,
    reg_item: float,
    reg_bias: float,

    urm_csr_indptr: np.ndarray,
    urm_csr_indices: np.ndarray,
    urm_csr_data: np.ndarray,

    batch_size: int,
    frac_negative_sampling: float,
    learning_rate: float,
    num_samples: int,
    # optimizers: NumbaFunkSVDOptimizer,
    use_bias: bool,

) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:

    epoch_loss: float = 0.
    num_batches = int(num_samples / batch_size) + 1

    arr_batch_sampled_user_ids = np.empty(shape=(batch_size,), dtype=np.int32)
    arr_batch_sampled_item_ids = np.empty(shape=(batch_size,), dtype=np.int32)
    arr_batch_sampled_ratings = np.empty(shape=(batch_size,), dtype=np.float32)

    arr_accumulator_user_embeddings = np.zeros_like(embeddings_users, dtype=np.float32)
    arr_accumulator_item_embeddings = np.zeros_like(embeddings_items, dtype=np.float32)
    arr_accumulator_bias_global = np.zeros_like(bias_global, dtype=np.float32)
    arr_accumulator_bias_users = np.zeros_like(bias_users, dtype=np.float32)
    arr_accumulator_bias_items = np.zeros_like(bias_items, dtype=np.float32)

    for _ in range(num_batches):
        arr_accumulator_user_embeddings[:] = 0.
        arr_accumulator_item_embeddings[:] = 0.
        arr_accumulator_bias_global[:] = 0.
        arr_accumulator_bias_users[:] = 0.
        arr_accumulator_bias_items[:] = 0.

        for idx_sample in range(batch_size):
            (
                sampled_user_id,
                sampled_item_id,
                sampled_rating,
            ) = nb_sample_funk_svd_from_sparse_matrix(
                num_users=num_users,
                num_items=num_items,
                urm_csr_indptr=urm_csr_indptr,
                urm_csr_indices=urm_csr_indices,
                urm_csr_data=urm_csr_data,
                frac_negative_sampling=frac_negative_sampling,
            )

            arr_batch_sampled_user_ids[idx_sample] = sampled_user_id
            arr_batch_sampled_item_ids[idx_sample] = sampled_item_id
            arr_batch_sampled_ratings[idx_sample] = sampled_rating

            prediction_error = nb_compute_prediction_mf_funk_svd(
                bias_global=bias_global,
                bias_users=bias_users,
                bias_items=bias_items,

                embeddings_users=embeddings_users,
                embeddings_items=embeddings_items,

                num_factors=num_factors,

                sample_user=sampled_user_id,
                sample_item=sampled_item_id,
                sample_rating=sampled_rating,

                use_bias=use_bias,
            )

            nb_compute_gradients_mf_funk_svd(
                arr_gradients_embeddings_users=arr_accumulator_user_embeddings,
                arr_gradients_embeddings_items=arr_accumulator_item_embeddings,
                arr_gradients_bias_global=arr_accumulator_bias_global,
                arr_gradients_bias_users=arr_accumulator_bias_users,
                arr_gradients_bias_items=arr_accumulator_bias_items,

                bias_global=bias_global,
                bias_users=bias_users,
                bias_items=bias_items,

                embeddings_users=embeddings_users,
                embeddings_items=embeddings_items,

                num_factors=num_factors,
                prediction_error=prediction_error,

                reg_bias=reg_bias,
                reg_item=reg_item,
                reg_user=reg_user,

                sample_user=sampled_user_id,
                sample_item=sampled_item_id,

                use_bias=use_bias,
            )

            epoch_loss += prediction_error ** 2

        (
            embeddings_users,
            embeddings_items,

            bias_global,
            bias_users,
            bias_items,
        ) = apply_batch_updates_to_latent_factors(
            embeddings_users=embeddings_users,
            embeddings_items=embeddings_items,

            bias_global=bias_global,
            bias_users=bias_users,
            bias_items=bias_items,

            arr_accumulator_user_embeddings=arr_accumulator_user_embeddings,
            arr_accumulator_item_embeddings=arr_accumulator_item_embeddings,
            arr_accumulator_bias_global=arr_accumulator_bias_global,
            arr_accumulator_bias_users=arr_accumulator_bias_users,
            arr_accumulator_bias_items=arr_accumulator_bias_items,

            batch_users=arr_batch_sampled_user_ids,
            batch_items=arr_batch_sampled_item_ids,
            batch_size=batch_size,

            learning_rate=learning_rate,
            num_factors=num_factors,
            # optimizers=optimizers,
            use_bias=use_bias,
        )

        # optimizers.after_batch()

    return (
        embeddings_users,
        embeddings_items,
        bias_global,
        bias_users,
        bias_items,
        epoch_loss,
    )


nb_epoch_mf_funk_svd.parallel_diagnostics(level=4)


@nb.njit
def nb_train_mf_funk_svd(
    urm_csr_indptr: np.ndarray,
    urm_csr_indices: np.ndarray,
    urm_csr_data: np.ndarray,

    num_users: int,
    num_items: int,

    num_samples: int,
    num_factors: int,
    num_epochs: int,

    reg_user: float,
    reg_item: float,
    reg_bias: float,

    batch_size: int,
    frac_negative_sampling: float,
    learning_rate: float,
    # optimizers: NumbaFunkSVDOptimizer,
    use_bias: bool,

    seed: int = 1234,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (
        user_embeddings,
        item_embeddings,
        bias_global,
        bias_users,
        bias_items,
    ) = init_mf_funk_svd(
        embeddings_mean=0.,
        embeddings_std_dev=1.,
        num_users=num_users,
        num_items=num_items,
        num_factors=num_factors,
        seed=seed,
    )
    arr_epoch_losses = np.empty(shape=(num_epochs, ), dtype=np.float64)

    for epoch in range(num_epochs):
        (
            user_embeddings,
            item_embeddings,

            bias_global,
            bias_users,
            bias_items,

            epoch_loss,
        ) = nb_epoch_mf_funk_svd(
            embeddings_users=user_embeddings,
            embeddings_items=item_embeddings,

            bias_global=bias_global,
            bias_users=bias_users,
            bias_items=bias_items,

            num_users=num_users,
            num_items=num_items,
            num_factors=num_factors,

            reg_user=reg_user,
            reg_item=reg_item,
            reg_bias=reg_bias,

            urm_csr_indptr=urm_csr_indptr,
            urm_csr_indices=urm_csr_indices,
            urm_csr_data=urm_csr_data,

            batch_size=batch_size,
            frac_negative_sampling=frac_negative_sampling,
            learning_rate=learning_rate,
            num_samples=num_samples,
            # optimizers=optimizers,
            use_bias=use_bias,
        )

        arr_epoch_losses[epoch] = epoch_loss

    return (
        user_embeddings,
        item_embeddings,

        bias_global,
        bias_users,
        bias_items,

        arr_epoch_losses,
    )


@nb.njit
def nb_compute_prediction_mf_funk_svd(
    bias_global: np.ndarray,
    bias_users: np.ndarray,
    bias_items: np.ndarray,

    embeddings_users: np.ndarray,
    embeddings_items: np.ndarray,

    num_factors: int,

    sample_user: int,
    sample_item: int,
    sample_rating: float,

    use_bias: bool,
) -> float:
    prediction = 0.
    if use_bias:
        prediction += bias_global[0] + bias_users[sample_user] + bias_items[sample_item]

    for idx_factor in nb.prange(num_factors):
        prediction += (
            embeddings_users[sample_user, idx_factor]
            * embeddings_items[sample_item, idx_factor]
        )

    prediction_error = sample_rating - prediction

    return prediction_error


@nb.njit
def nb_compute_gradients_mf_funk_svd(
    arr_gradients_embeddings_users: np.ndarray,
    arr_gradients_embeddings_items: np.ndarray,
    arr_gradients_bias_global: np.ndarray,
    arr_gradients_bias_users: np.ndarray,
    arr_gradients_bias_items: np.ndarray,

    bias_global: np.ndarray,
    bias_users: np.ndarray,
    bias_items: np.ndarray,

    embeddings_users: np.ndarray,
    embeddings_items: np.ndarray,

    num_factors: int,
    prediction_error: float,

    reg_bias: float,
    reg_user: float,
    reg_item: float,

    sample_user: int,
    sample_item: int,

    use_bias: bool,
) -> None:
    if use_bias:
        local_gradient_global_bias = prediction_error - reg_bias * bias_global[0]
        local_gradient_user_bias = prediction_error - reg_bias * bias_users[sample_user]
        local_gradient_item_bias = prediction_error - reg_bias * bias_items[sample_item]

        arr_gradients_bias_global[0] = local_gradient_global_bias
        arr_gradients_bias_users[sample_user] = local_gradient_user_bias
        arr_gradients_bias_items[sample_item] = local_gradient_item_bias

    for idx_factor in nb.prange(num_factors):
        # Copy original value to avoid messing up the updates
        W_u = embeddings_users[sample_user, idx_factor]
        H_i = embeddings_items[sample_item, idx_factor]

        # Compute gradients
        local_gradient_user = prediction_error * H_i - (reg_user * W_u)
        local_gradient_item = prediction_error * W_u - (reg_item * H_i)

        arr_gradients_embeddings_users[sample_user, idx_factor] += local_gradient_user
        arr_gradients_embeddings_items[sample_item, idx_factor] += local_gradient_item


@nb.njit
def nb_sample_mf_funk_svd(
    embeddings_users: np.ndarray,
    embeddings_items: np.ndarray,

    bias_global: np.ndarray,
    bias_users: np.ndarray,
    bias_items: np.ndarray,

    arr_accumulator_embeddings_users: np.ndarray,
    arr_accumulator_embeddings_items: np.ndarray,

    arr_accumulator_bias_global: np.ndarray,
    arr_accumulator_bias_users: np.ndarray,
    arr_accumulator_bias_items: np.ndarray,

    reg_user: float,
    reg_item: float,
    reg_bias: float,

    sample_user: int,
    sample_item: int,
    sample_rating: float,

    use_bias: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    num_factors = embeddings_users.shape[1]

    prediction = 0.
    if use_bias:
        prediction += bias_global[0] + bias_users[sample_user] + bias_items[sample_item]

    for idx_factor in range(num_factors):
        prediction += (
            embeddings_users[sample_user, idx_factor]
            * embeddings_items[sample_item, idx_factor]
        )

    prediction_error = sample_rating - prediction

    if use_bias:
        local_gradient_global_bias = prediction_error - reg_bias * bias_global[0]
        local_gradient_user_bias = prediction_error - reg_bias * bias_users[sample_user]
        local_gradient_item_bias = prediction_error - reg_bias * bias_items[sample_item]

        arr_accumulator_bias_global[0] = local_gradient_global_bias
        arr_accumulator_bias_users[sample_user] = local_gradient_user_bias
        arr_accumulator_bias_items[sample_item] = local_gradient_item_bias

    for idx_factor in range(num_factors):
        # Copy original value to avoid messing up the updates
        W_u = embeddings_users[sample_user, idx_factor]
        H_i = embeddings_items[sample_item, idx_factor]

        # Compute gradients
        local_gradient_user = prediction_error * H_i - (reg_user * W_u)
        local_gradient_item = prediction_error * W_u - (reg_item * H_i)

        arr_accumulator_embeddings_users[sample_user, idx_factor] += local_gradient_user
        arr_accumulator_embeddings_items[sample_item, idx_factor] += local_gradient_item

    return (
        arr_accumulator_embeddings_users,
        arr_accumulator_embeddings_items,

        arr_accumulator_bias_global,
        arr_accumulator_bias_users,
        arr_accumulator_bias_items,

        prediction_error,
    )


@nb.njit
def apply_batch_updates_to_latent_factors(
    embeddings_users: np.ndarray,
    embeddings_items: np.ndarray,

    bias_global: np.ndarray,
    bias_users: np.ndarray,
    bias_items: np.ndarray,

    arr_accumulator_user_embeddings: np.ndarray,
    arr_accumulator_item_embeddings: np.ndarray,
    arr_accumulator_bias_global: np.ndarray,
    arr_accumulator_bias_users: np.ndarray,
    arr_accumulator_bias_items: np.ndarray,

    batch_users: np.ndarray,
    batch_items: np.ndarray,
    batch_size: int,

    learning_rate: float,
    num_factors: int,
    # optimizers: NumbaFunkSVDOptimizer,
    use_bias: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if use_bias:
        local_gradient_bias_global = arr_accumulator_bias_global[0] / batch_size
        # local_gradient_bias_global = optimizers.optimizer_bias_global.adaptive_gradient(
        #     local_gradient_bias_global, 0, 0,
        # )

        # apply updates
        bias_global[0] += learning_rate * local_gradient_bias_global
        arr_accumulator_bias_global[0] = 0.

    for sample_item in batch_items:
        if use_bias:
            local_gradient_bias_item = arr_accumulator_bias_items[sample_item] / batch_size
            # local_gradient_bias_item = optimizers.optimizer_bias_items.adaptive_gradient(
            #     local_gradient_bias_item, sample_item, 0,
            # )

            bias_items[sample_item] += learning_rate * local_gradient_bias_item
            arr_accumulator_bias_items[sample_item] = 0.

        for idx_factor in range(num_factors):
            local_gradient_item = arr_accumulator_item_embeddings[sample_item, idx_factor] / batch_size
            # local_gradient_item = optimizers.optimizer_embeddings_items.adaptive_gradient(
            #     local_gradient_item, sample_item, idx_factor,
            # )

            embeddings_items[sample_item, idx_factor] += learning_rate * local_gradient_item
            arr_accumulator_item_embeddings[sample_item] = 0.

    for sample_user in batch_users:
        if use_bias:
            local_gradient_bias_user = bias_users[sample_user] / batch_size
            # local_gradient_bias_user = optimizers.optimizer_bias_users.adaptive_gradient(
            #     local_gradient_bias_user, sample_user, 0,
            # )

            bias_users[sample_user] += learning_rate * local_gradient_bias_user
            arr_accumulator_bias_users[sample_user] = 0.

        for idx_factor in range(num_factors):
            local_gradient_user = arr_accumulator_user_embeddings[sample_user, idx_factor] / batch_size
            # local_gradient_user = optimizers.optimizer_embeddings_users.adaptive_gradient(
            #     local_gradient_user, sample_user, idx_factor,
            # )

            embeddings_users[sample_user, idx_factor] += learning_rate * local_gradient_user
            arr_accumulator_user_embeddings[sample_user] = 0.

    return (
        embeddings_users,
        embeddings_items,

        bias_global,
        bias_users,
        bias_items,
    )


@nb.njit
def init_mf_funk_svd(
    num_users: int,
    num_items: int,
    num_factors: int,

    embeddings_mean: float,
    embeddings_std_dev: float,

    seed: int = 1234,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nb_numpy_seed(seed=seed)

    arr_user_embeddings = create_embeddings_from_normal_distribution(
        mean=embeddings_mean,
        std_dev=embeddings_std_dev,
        shape=(num_users, num_factors),
    )
    arr_item_embeddings = create_embeddings_from_normal_distribution(
        mean=embeddings_mean,
        std_dev=embeddings_std_dev,
        shape=(num_items, num_factors),
    )
    bias_global = np.zeros(
        shape=(1,),
    )
    bias_users = np.zeros(
        shape=(num_users,),
    )
    bias_items = np.zeros(
        shape=(num_items,),
    )

    return (
        arr_user_embeddings,
        arr_item_embeddings,

        bias_global,
        bias_users,
        bias_items,
    )


run_epoch_funk_svd = nb_epoch_mf_funk_svd


def train_mf_funk_svd(
    urm_train: scipy.sparse.csr_matrix,
    num_users: int,
    num_items: int,

    num_factors: int,
    num_epochs: int,

    reg_user: float,
    reg_item: float,
    reg_bias: float,

    batch_size: int,
    frac_negative_sampling: float,
    learning_rate: float,
    # optimizers: NumbaFunkSVDOptimizer,
    use_bias: bool,

    seed: int = 1234,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert urm_train.shape == (num_users, num_items)
    assert 0. <= frac_negative_sampling <= 1.

    num_samples = urm_train.nnz
    urm_coo = sparse.COO.from_scipy_sparse(urm_train)

    (
        arr_user_embeddings,
        arr_item_embeddings,

        arr_bias_global,
        arr_bias_users,
        arr_bias_items,

        arr_epoch_losses,
    ) = nb_train_mf_funk_svd(
        urm_csr_indptr=urm_train.indptr,
        urm_csr_indices=urm_train.indices,
        urm_csr_data=urm_train.data,

        num_users=num_users,
        num_items=num_items,

        num_samples=num_samples,
        num_factors=num_factors,
        num_epochs=num_epochs,

        reg_user=reg_user,
        reg_item=reg_item,
        reg_bias=reg_bias,

        batch_size=batch_size,
        frac_negative_sampling=frac_negative_sampling,
        learning_rate=learning_rate,
        # optimizers=optimizers,

        seed=seed,
        use_bias=use_bias,
    )

    for epoch, epoch_loss in enumerate(arr_epoch_losses):
        print(
            f"Epoch: {epoch} - loss: {epoch_loss}"
        )

    return (
        arr_user_embeddings,
        arr_item_embeddings,
        arr_bias_global,
        arr_bias_users,
        arr_bias_items,
        arr_epoch_losses,
    )
