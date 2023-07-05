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


@nb.experimental.jitclass(
    [
        ("arr_epoch_losses", nb.float32[:]),
        ("arr_embeddings_user", nb.float32[:, :]),
        ("arr_embeddings_item", nb.float32[:, :]),
        ("arr_bias_global", nb.float32[:]),
        ("arr_bias_users", nb.float32[:]),
        ("arr_bias_items", nb.float32[:]),

        ("arr_accumulator_embeddings_users", nb.float32[:, :]),
        ("arr_accumulator_embeddings_items", nb.float32[:, :]),
        ("arr_accumulator_bias_global", nb.float32[:]),
        ("arr_accumulator_bias_users", nb.float32[:]),
        ("arr_accumulator_bias_items", nb.float32[:]),

        ("arr_batch_sampled_user_ids", nb.int32[:]),
        ("arr_batch_sampled_item_ids", nb.int32[:]),
        ("arr_batch_sampled_ratings", nb.float32[:]),

        ("urm_csr_indices", nb.int32[:]),
        ("urm_csr_indptr", nb.int32[:]),
        ("urm_csr_data", nb.float32[:]),
    ]
)
class NumbaFunkSVDTrainer:
    batch_size: int
    embeddings_mean: float
    embeddings_std_dev: float
    frac_negative_sampling: float
    learning_rate: float
    num_batches: int
    num_epochs: int
    num_factors: int
    num_samples: int
    seed: int
    use_bias: bool

    num_users: int
    num_items: int

    arr_epoch_losses: np.ndarray
    arr_embeddings_user: np.ndarray
    arr_embeddings_item: np.ndarray
    arr_bias_global: np.ndarray
    arr_bias_users: np.ndarray
    arr_bias_items: np.ndarray

    arr_batch_sampled_user_ids: np.ndarray
    arr_batch_sampled_item_ids: np.ndarray
    arr_batch_sampled_ratings: np.ndarray

    arr_accumulator_embeddings_users: np.ndarray
    arr_accumulator_embeddings_items: np.ndarray
    arr_accumulator_bias_global: np.ndarray
    arr_accumulator_bias_users: np.ndarray
    arr_accumulator_bias_items: np.ndarray

    reg_user: float
    reg_item: float
    reg_bias: float

    urm_csr_indices: np.ndarray
    urm_csr_indptr: np.ndarray
    urm_csr_data: np.ndarray

    def __init__(
        self,
        embeddings_mean: float,
        embeddings_std_dev: float,

        num_users: int,
        num_items: int,

        reg_user: float,
        reg_item: float,
        reg_bias: float,

        urm_csr_indices: np.ndarray,
        urm_csr_indptr: np.ndarray,
        urm_csr_data: np.ndarray,

        batch_size: int,
        frac_negative_sampling: float,
        learning_rate: float,
        num_epochs: int,
        num_factors: int,
        num_samples: int,
        # optimizers: NumbaFunkSVDOptimizer,
        use_bias: bool,

        seed: int = 1234,
    ):

        self.num_users = num_users
        self.num_items = num_items

        self.reg_user = reg_user
        self.reg_item = reg_item
        self.reg_bias = reg_bias

        self.urm_csr_indices = urm_csr_indices
        self.urm_csr_indptr = urm_csr_indptr
        self.urm_csr_data = urm_csr_data

        self.batch_size = batch_size
        self.embeddings_mean = embeddings_mean
        self.embeddings_std_dev = embeddings_std_dev
        self.frac_negative_sampling = frac_negative_sampling
        self.learning_rate = learning_rate
        self.num_batches = int(num_samples / batch_size) + 1
        self.num_epochs = num_epochs
        self.num_factors = num_factors
        self.num_samples = num_samples
        self.seed = seed
        self.use_bias = use_bias

        nb_numpy_seed(
            seed=self.seed,
        )

        self.arr_embeddings_user = np.random.normal(
            loc=self.embeddings_mean,
            scale=self.embeddings_std_dev,
            size=(self.num_users, self.num_factors),
        ).astype(
            np.float32,
        )

        self.arr_embeddings_item = np.random.normal(
            loc=self.embeddings_mean,
            scale=self.embeddings_std_dev,
            size=(self.num_items, self.num_factors),
        ).astype(
            np.float32,
        )

        self.arr_bias_global = np.zeros(
            shape=(1,),
        ).astype(
            np.float32
        )
        self.arr_bias_users = np.zeros(
            shape=(self.num_users,),
        ).astype(
            np.float32
        )
        self.arr_bias_items = np.zeros(
            shape=(self.num_items,),
        ).astype(
            np.float32
        )

        self.arr_epoch_losses = np.empty(
            shape=(self.num_epochs,),
        ).astype(
            np.float32
        )

        self.arr_batch_sampled_user_ids = np.empty(
            shape=(self.batch_size,),
        ).astype(
            np.int32
        )
        self.arr_batch_sampled_item_ids = np.empty_like(
            self.arr_batch_sampled_user_ids, dtype=np.int32,
        )
        self.arr_batch_sampled_ratings = np.empty_like(
            self.arr_batch_sampled_user_ids, dtype=np.float32,
        )

        self.arr_accumulator_embeddings_users = np.zeros_like(
            self.arr_embeddings_user,
        )
        self.arr_accumulator_embeddings_items = np.zeros_like(
            self.arr_embeddings_item,
        )
        self.arr_accumulator_bias_global = np.zeros_like(
            self.arr_bias_global,
        )
        self.arr_accumulator_bias_users = np.zeros_like(
            self.arr_bias_users,
        )
        self.arr_accumulator_bias_items = np.zeros_like(
            self.arr_bias_items,
        )

    def train_all_epochs(self):
        for epoch in range(self.num_epochs):
            self.run_epoch(epoch=epoch)

    def run_epoch(self, epoch: int):
        epoch_loss = 0.

        for _ in range(self.num_batches):
            self.arr_accumulator_embeddings_users[:] = 0.
            self.arr_accumulator_embeddings_items[:] = 0.
            self.arr_accumulator_bias_global[:] = 0.
            self.arr_accumulator_bias_users[:] = 0.
            self.arr_accumulator_bias_items[:] = 0.

            for idx_sample in range(self.batch_size):
                (
                    sampled_user_id,
                    sampled_item_id,
                    sampled_rating,
                ) = self.get_sample()

                self.arr_batch_sampled_user_ids[idx_sample] = sampled_user_id
                self.arr_batch_sampled_item_ids[idx_sample] = sampled_item_id
                self.arr_batch_sampled_ratings[idx_sample] = sampled_rating

                prediction_error = self.compute_prediction_error(
                    sample_user=sampled_user_id,
                    sample_item=sampled_item_id,
                    sample_rating=sampled_rating,
                )

                self.compute_gradients(
                    prediction_error=prediction_error,
                    sample_user=sampled_user_id,
                    sample_item=sampled_item_id,
                )

                epoch_loss += prediction_error ** 2

            self.apply_gradients()
            self.after_batch()

        self.arr_epoch_losses[epoch] = epoch_loss

    def get_sample(self):
        sample_user, user_profile_start, user_profile_end, num_seen_items = nb_sample_funk_svd_user(
            num_users=self.num_users,
            num_items=self.num_items,
            urm_csr_indptr=self.urm_csr_indptr,
        )

        sample_positive = np.random.random() <= self.frac_negative_sampling

        if sample_positive:
            sample_item, sample_rating = nb_sample_funk_svd_positive_item(
                num_seen_items=num_seen_items,
                user_profile_start=user_profile_start,
                urm_csr_indices=self.urm_csr_indices,
                urm_csr_data=self.urm_csr_data,
            )
        else:
            sample_item, sample_rating = nb_sample_funk_svd_negative_item(
                num_items=self.num_items,
                num_seen_items=num_seen_items,
                user_profile_start=user_profile_start,
                urm_csr_indices=self.urm_csr_indices,
            )

        return sample_user, sample_item, sample_rating

    def compute_prediction_error(self, sample_user: int, sample_item: int, sample_rating: float):
        prediction = 0.
        if self.use_bias:
            prediction += (
                self.arr_bias_global[0]
                + self.arr_bias_users[sample_user]
                + self.arr_bias_items[sample_item]
            )

        for idx_factor in nb.prange(self.num_factors):
            prediction += (
                self.arr_embeddings_user[sample_user, idx_factor]
                * self.arr_embeddings_item[sample_item, idx_factor]
            )

        prediction_error = sample_rating - prediction

        return prediction_error

    def compute_gradients(self, sample_user: int, sample_item: int, prediction_error: float):
        if self.use_bias:
            local_gradient_global_bias = prediction_error - self.reg_bias * self.arr_bias_global[0]
            local_gradient_user_bias = prediction_error - self.reg_bias * self.arr_bias_users[sample_user]
            local_gradient_item_bias = prediction_error - self.reg_bias * self.arr_bias_items[sample_item]

            self.arr_accumulator_bias_global[0] = local_gradient_global_bias
            self.arr_accumulator_bias_users[sample_user] = local_gradient_user_bias
            self.arr_accumulator_bias_items[sample_item] = local_gradient_item_bias

        for idx_factor in nb.prange(self.num_factors):
            # Copy original value to avoid messing up the updates
            W_u = self.arr_embeddings_user[sample_user, idx_factor]
            H_i = self.arr_embeddings_item[sample_item, idx_factor]

            # Compute gradients
            local_gradient_user = prediction_error * H_i - (self.reg_user * W_u)
            local_gradient_item = prediction_error * W_u - (self.reg_item * H_i)

            self.arr_accumulator_embeddings_users[sample_user, idx_factor] += local_gradient_user
            self.arr_accumulator_embeddings_items[sample_item, idx_factor] += local_gradient_item

    def apply_gradients(self):
        if self.use_bias:
            local_gradient_bias_global = self.arr_accumulator_bias_global[0] / self.batch_size
            # local_gradient_bias_global = optimizers.optimizer_bias_global.adaptive_gradient(
            #     local_gradient_bias_global, 0, 0,
            # )

            # apply updates
            self.arr_bias_global[0] += self.learning_rate * local_gradient_bias_global
            self.arr_accumulator_bias_global[0] = 0.

        for sample_item in self.arr_batch_sampled_item_ids:
            if self.use_bias:
                local_gradient_bias_item = self.arr_accumulator_bias_items[sample_item] / self.batch_size
                # local_gradient_bias_item = optimizers.optimizer_bias_items.adaptive_gradient(
                #     local_gradient_bias_item, sample_item, 0,
                # )

                self.arr_bias_items[sample_item] += self.learning_rate * local_gradient_bias_item
                self.arr_accumulator_bias_items[sample_item] = 0.

            for idx_factor in range(self.num_factors):
                local_gradient_item = self.arr_accumulator_embeddings_items[sample_item, idx_factor] / self.batch_size
                # local_gradient_item = optimizers.optimizer_embeddings_items.adaptive_gradient(
                #     local_gradient_item, sample_item, idx_factor,
                # )

                self.arr_embeddings_item[sample_item, idx_factor] += self.learning_rate * local_gradient_item
                self.arr_accumulator_embeddings_items[sample_item] = 0.

        for sample_user in self.arr_batch_sampled_user_ids:
            if self.use_bias:
                local_gradient_bias_user = self.arr_bias_users[sample_user] / self.batch_size
                # local_gradient_bias_user = optimizers.optimizer_bias_users.adaptive_gradient(
                #     local_gradient_bias_user, sample_user, 0,
                # )

                self.arr_bias_users[sample_user] += self.learning_rate * local_gradient_bias_user
                self.arr_accumulator_bias_users[sample_user] = 0.

            for idx_factor in range(self.num_factors):
                local_gradient_user = self.arr_accumulator_embeddings_users[sample_user, idx_factor] / self.batch_size
                # local_gradient_user = optimizers.optimizer_embeddings_users.adaptive_gradient(
                #     local_gradient_user, sample_user, idx_factor,
                # )

                self.arr_embeddings_user[sample_user, idx_factor] += self.learning_rate * local_gradient_user
                self.arr_accumulator_embeddings_users[sample_user] = 0.

    def after_batch(self):
        pass

    def reset_internal_state(self):
        nb_numpy_seed(
            seed=self.seed
        )

        self.arr_embeddings_user = np.random.normal(
            loc=self.embeddings_mean,
            scale=self.embeddings_std_dev,
            size=(self.num_users, self.num_factors),
        ).astype(
            np.float32,
        )

        self.arr_embeddings_item = np.random.normal(
            loc=self.embeddings_mean,
            scale=self.embeddings_std_dev,
            size=(self.num_items, self.num_factors),
        ).astype(
            np.float32,
        )

        self.arr_bias_global = np.zeros(
            shape=(1,),
        ).astype(
            np.float32
        )
        self.arr_bias_users = np.zeros(
            shape=(self.num_users,),
        ).astype(
            np.float32
        )
        self.arr_bias_items = np.zeros(
            shape=(self.num_items,),
        ).astype(
            np.float32
        )

        self.arr_epoch_losses = np.empty(
            shape=(self.num_epochs,),
        ).astype(
            np.float32
        )

        self.arr_batch_sampled_user_ids = np.empty(
            shape=(self.batch_size,),
        ).astype(
            np.int32
        )
        self.arr_batch_sampled_item_ids = np.empty_like(
            self.arr_batch_sampled_user_ids, dtype=np.int32,
        )
        self.arr_batch_sampled_ratings = np.empty_like(
            self.arr_batch_sampled_user_ids, dtype=np.float32,
        )

        self.arr_accumulator_embeddings_users = np.zeros_like(
            self.arr_embeddings_user,
        )
        self.arr_accumulator_embeddings_items = np.zeros_like(
            self.arr_embeddings_item,
        )
        self.arr_accumulator_bias_global = np.zeros_like(
            self.arr_bias_global,
        )
        self.arr_accumulator_bias_users = np.zeros_like(
            self.arr_bias_users,
        )
        self.arr_accumulator_bias_items = np.zeros_like(
            self.arr_bias_items,
        )


@nb.njit
def train_instance(
    instance: NumbaFunkSVDTrainer
) -> None:
    instance.reset_internal_state()
    instance.train_all_epochs()


@nb.njit
def nb_numpy_seed(
    seed: int,
):
    np.random.seed(seed)


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


