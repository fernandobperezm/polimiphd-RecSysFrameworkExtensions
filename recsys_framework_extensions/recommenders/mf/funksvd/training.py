from typing import Union

import numpy as np
import numba as nb
import scipy.sparse
import sparse
from tqdm import tqdm

from recsys_framework_extensions.optimizer import (
    NumbaOptimizer,
)
from recsys_framework_extensions.sampling.sparse import (
    nb_sample_only_negatives_from_sparse_csr_matrix,
    nb_sample_only_positives_from_sparse_coo_matrix,
    nb_sample_positives_and_negatives_from_sparse_matrix,
)


tqdm.pandas()


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
def nb_sample_funk_svd_from_sparse_matrix(
    num_users: int,
    num_items: int,
    num_samples: int,
    urm_coo: sparse.COO,
    urm_csr_indptr: np.ndarray,
    urm_csr_indices: np.ndarray,
    frac_negative_sampling: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if frac_negative_sampling == 1.:
        return nb_sample_only_negatives_from_sparse_csr_matrix(
            num_users=num_users,
            num_items=num_items,
            num_samples=num_samples,
            urm_csr_indptr=urm_csr_indptr,
            urm_csr_indices=urm_csr_indices
        )

    elif frac_negative_sampling == 0.:
        return nb_sample_only_positives_from_sparse_coo_matrix(
            num_users=num_users,
            num_items=num_items,
            num_samples=num_samples,
            urm=urm_coo,
        )

    else:
        return nb_sample_positives_and_negatives_from_sparse_matrix(
            num_users=num_users,
            num_items=num_items,
            num_samples=num_samples,
            urm_coo=urm_coo,
            urm_csr_indptr=urm_csr_indptr,
            urm_csr_indices=urm_csr_indices,
            frac_negative_sampling=frac_negative_sampling,
        )


@nb.njit
def nb_epoch_mf_funk_svd(
    arr_user_ids: np.ndarray,
    arr_item_ids: np.ndarray,
    arr_ratings: np.ndarray,

    embeddings_users: np.ndarray,
    embeddings_items: np.ndarray,

    bias_global: np.ndarray,
    bias_users: np.ndarray,
    bias_items: np.ndarray,

    num_factors: int,

    reg_user: float,
    reg_item: float,
    reg_bias: float,

    batch_size: int,
    learning_rate: float,
    # optimizer: NumbaFunkSVDOptimizer,
    use_bias: bool,

) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    epoch_loss: float = 0.

    list_batch_users = np.array_split(arr_user_ids, batch_size)
    list_batch_items = np.array_split(arr_item_ids, batch_size)
    list_batch_ratings = np.array_split(arr_ratings, batch_size)

    assert len(list_batch_users) == len(list_batch_items)
    assert len(list_batch_users) == len(list_batch_ratings)

    num_batches = len(list_batch_users)

    for idx_batch in range(num_batches):
        batch_users = list_batch_users[idx_batch]
        batch_items = list_batch_items[idx_batch]
        batch_ratings = list_batch_ratings[idx_batch]

        arr_accumulator_user_embeddings = np.zeros_like(embeddings_users, dtype=np.float64)
        arr_accumulator_item_embeddings = np.zeros_like(embeddings_items, dtype=np.float64)
        arr_accumulator_bias_global = np.zeros_like(bias_global, dtype=np.float64)
        arr_accumulator_bias_users = np.zeros_like(bias_users, dtype=np.float64)
        arr_accumulator_bias_items = np.zeros_like(bias_items, dtype=np.float64)

        for sample_user, sample_item, sample_rating in zip(
            batch_users, batch_items, batch_ratings
        ):
            (
                arr_accumulator_user_embeddings,
                arr_accumulator_item_embeddings,

                arr_accumulator_bias_global,
                arr_accumulator_bias_users,
                arr_accumulator_bias_items,

                prediction_error,
            ) = nb_sample_mf_funk_svd(
                embeddings_users=embeddings_users,
                embeddings_items=embeddings_items,

                bias_global=bias_global,
                bias_users=bias_users,
                bias_items=bias_items,

                arr_accumulator_embeddings_users=arr_accumulator_user_embeddings,
                arr_accumulator_embeddings_items=arr_accumulator_item_embeddings,
                arr_accumulator_bias_global=arr_accumulator_bias_global,
                arr_accumulator_bias_users=arr_accumulator_bias_users,
                arr_accumulator_bias_items=arr_accumulator_bias_items,

                reg_user=reg_user,
                reg_item=reg_item,
                reg_bias=reg_bias,

                sample_user=sample_user,
                sample_item=sample_item,
                sample_rating=sample_rating,

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

            batch_users=batch_users,
            batch_items=batch_items,
            batch_size=batch_size,

            learning_rate=learning_rate,
            num_factors=num_factors,
            # optimizer=optimizer,
            use_bias=use_bias,
        )

        # optimizer.after_batch()

    return (
        embeddings_users,
        embeddings_items,
        bias_global,
        bias_users,
        bias_items,
        epoch_loss,
    )


@nb.njit
def nb_train_mf_funk_svd(
    urm_coo: sparse.COO,
    urm_csr_indptr: np.ndarray,
    urm_csr_indices: np.ndarray,

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
    # optimizer: NumbaFunkSVDOptimizer,
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
            arr_user_ids,
            arr_item_ids,
            arr_ratings,
        ) = nb_sample_funk_svd_from_sparse_matrix(
            num_users=num_users,
            num_items=num_items,
            num_samples=num_samples,
            urm_coo=urm_coo,
            urm_csr_indptr=urm_csr_indptr,
            urm_csr_indices=urm_csr_indices,
            frac_negative_sampling=frac_negative_sampling
        )

        (
            user_embeddings,
            item_embeddings,

            bias_global,
            bias_users,
            bias_items,

            epoch_loss,
        ) = nb_epoch_mf_funk_svd(
            arr_user_ids=arr_user_ids,
            arr_item_ids=arr_item_ids,
            arr_ratings=arr_ratings,

            embeddings_users=user_embeddings,
            embeddings_items=item_embeddings,

            bias_global=bias_global,
            bias_users=bias_users,
            bias_items=bias_items,

            reg_user=reg_user,
            reg_item=reg_item,
            reg_bias=reg_bias,

            batch_size=batch_size,
            num_factors=num_factors,
            learning_rate=learning_rate,
            # optimizer=optimizer,
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
    # optimizer: NumbaFunkSVDOptimizer,
    use_bias: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if use_bias:
        local_gradient_bias_global = arr_accumulator_bias_global[0] / batch_size
        # local_gradient_bias_global = optimizer.optimizer_bias_global.adaptive_gradient(
        #     local_gradient_bias_global, 0, 0,
        # )

        # apply updates
        bias_global[0] += learning_rate * local_gradient_bias_global
        arr_accumulator_bias_global[0] = 0.

    for sample_item in batch_items:
        if use_bias:
            local_gradient_bias_item = arr_accumulator_bias_items[sample_item] / batch_size
            # local_gradient_bias_item = optimizer.optimizer_bias_items.adaptive_gradient(
            #     local_gradient_bias_item, sample_item, 0,
            # )

            bias_items[sample_item] += learning_rate * local_gradient_bias_item
            arr_accumulator_bias_items[sample_item] = 0.

        for idx_factor in range(num_factors):
            local_gradient_item = arr_accumulator_item_embeddings[sample_item, idx_factor] / batch_size
            # local_gradient_item = optimizer.optimizer_embeddings_items.adaptive_gradient(
            #     local_gradient_item, sample_item, idx_factor,
            # )

            embeddings_items[sample_item, idx_factor] += learning_rate * local_gradient_item
            arr_accumulator_item_embeddings[sample_item] = 0.

    for sample_user in batch_users:
        if use_bias:
            local_gradient_bias_user = bias_users[sample_user] / batch_size
            # local_gradient_bias_user = optimizer.optimizer_bias_users.adaptive_gradient(
            #     local_gradient_bias_user, sample_user, 0,
            # )

            bias_users[sample_user] += learning_rate * local_gradient_bias_user
            arr_accumulator_bias_users[sample_user] = 0.

        for idx_factor in range(num_factors):
            local_gradient_user = arr_accumulator_user_embeddings[sample_user, idx_factor] / batch_size
            # local_gradient_user = optimizer.optimizer_embeddings_users.adaptive_gradient(
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


@nb.njit
def run_epoch_funk_svd(
    num_users: int,
    num_items: int,
    num_samples: int,

    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,

    bias_global: np.ndarray,
    bias_users: np.ndarray,
    bias_items: np.ndarray,

    urm_coo: sparse.COO,
    urm_csr_indptr: np.ndarray,
    urm_csr_indices: np.ndarray,

    batch_size: int,
    learning_rate: float,
    num_factors: int,
    # optimizer: NumbaFunkSVDOptimizer,
    quota_negative_interactions: float,

    reg_user: float,
    reg_item: float,
    reg_bias: float,

    use_bias: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (
        arr_user_ids,
        arr_item_ids,
        arr_ratings,
    ) = nb_sample_funk_svd_from_sparse_matrix(
        num_users=num_users,
        num_items=num_items,
        num_samples=num_samples,
        urm_coo=urm_coo,
        urm_csr_indptr=urm_csr_indptr,
        urm_csr_indices=urm_csr_indices,
        frac_negative_sampling=quota_negative_interactions,
    )

    (
        user_embeddings,
        item_embeddings,

        bias_global,
        bias_users,
        bias_items,

        epoch_loss,
    ) = nb_epoch_mf_funk_svd(
        arr_user_ids=arr_user_ids,
        arr_item_ids=arr_item_ids,
        arr_ratings=arr_ratings,

        embeddings_users=user_embeddings,
        embeddings_items=item_embeddings,

        bias_global=bias_global,
        bias_users=bias_users,
        bias_items=bias_items,

        reg_user=reg_user,
        reg_item=reg_item,
        reg_bias=reg_bias,

        batch_size=batch_size,
        learning_rate=learning_rate,
        num_factors=num_factors,
        # optimizer=optimizer,
        use_bias=use_bias,
    )

    return (
        user_embeddings,
        item_embeddings,
        bias_global,
        bias_users,
        bias_items,
    )


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
    # optimizer: NumbaFunkSVDOptimizer,
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
        urm_coo=urm_coo,
        urm_csr_indptr=urm_train.indptr,
        urm_csr_indices=urm_train.indices,

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
        # optimizer=optimizer,

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
