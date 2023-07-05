import collections

import numba as nb
import numpy as np
import scipy.sparse

from recsys_framework_extensions.optimizer import init_optimizer_state, OptimizerParameters, \
    get_after_gradient_function, get_adaptive_gradient_function
from recsys_framework_extensions.sampling.scipy_sparse import (
    nb_sample_negative_item,
    nb_sample_positive_item,
    nb_sample_user,
)

Model = collections.namedtuple(
    "Model", [
        "arr_embeddings_users", "arr_embeddings_items",
        "arr_biases_global", "arr_biases_users", "arr_biases_items"
    ],
)
CSRMatrix = collections.namedtuple(
    "CSRMatrix", ["arr_indices", "arr_indptr", "arr_data"],
)
Losses = collections.namedtuple(
    "Losses", ["arr_losses_epochs"],
)
Samples = collections.namedtuple(
    "Samples", ["arr_user_ids", "arr_item_ids", "arr_ratings"],
)
Sample = collections.namedtuple(
    "Sample", ["user_id", "item_id", "rating"]
)
Gradients = collections.namedtuple(
    "Gradients", [
        "arr_embeddings_users", "arr_embeddings_items",
        "arr_biases_global", "arr_biases_users", "arr_biases_items",
    ]
)
Parameters = collections.namedtuple(
    "Parameters", [
        "batch_size", "embeddings_mean", "embeddings_std_dev", "frac_negative_sampling", "learning_rate",
        "num_batches", "num_epochs", "num_factors", "num_samples", "num_users", "num_items",
        "reg_user", "reg_item", "reg_bias",
        "seed", "use_bias",
    ]
)
Optimizers = collections.namedtuple(
    "Optimizers", [
        "func_adaptive_gradient",
        "func_after_gradient",
        "embeddings_users",
        "embeddings_items",
        "biases_global",
        "biases_users",
        "biases_items",
    ]
)


@nb.njit
def nb_numpy_seed(
    seed: int,
):
    np.random.seed(seed)


@nb.njit
def nb_init_csr_matrix(
    arr_indices: np.ndarray,
    arr_indptr: np.ndarray,
    arr_data: np.ndarray,
) -> CSRMatrix:
    return CSRMatrix(
        arr_indices=arr_indices,
        arr_indptr=arr_indptr,
        arr_data=arr_data,
    )


@nb.njit
def nb_init_parameters(
    embeddings_mean: float,
    embeddings_std_dev: float,

    num_users: int,
    num_items: int,

    reg_user: float,
    reg_item: float,
    reg_bias: float,

    batch_size: int,
    frac_negative_sampling: float,
    learning_rate: float,
    num_epochs: int,
    num_factors: int,
    num_samples: int,
    seed: int,
    use_bias: bool,
) -> Parameters:
    num_batches = int(num_samples / batch_size) + 1

    return Parameters(
        batch_size=batch_size,
        embeddings_mean=embeddings_mean,
        embeddings_std_dev=embeddings_std_dev,
        frac_negative_sampling=frac_negative_sampling,
        learning_rate=learning_rate,
        num_batches=num_batches,
        num_epochs=num_epochs,
        num_factors=num_factors,
        num_samples=num_samples,
        num_users=num_users,
        num_items=num_items,
        reg_user=reg_user,
        reg_item=reg_item,
        reg_bias=reg_bias,
        seed=seed,
        use_bias=use_bias,
    )


@nb.njit
def nb_init_losses(
    parameters: Parameters,
) -> Losses:
    arr_losses_epochs = np.empty(
        shape=(parameters.num_epochs,),
    ).astype(
        np.float32
    )
    return Losses(
        arr_losses_epochs=arr_losses_epochs,
    )


@nb.njit
def nb_init_model(
    parameters: Parameters,
) -> Model:
    arr_embeddings_users = np.random.normal(
        loc=parameters.embeddings_mean,
        scale=parameters.embeddings_std_dev,
        size=(parameters.num_users, parameters.num_factors),
    ).astype(
        np.float32,
    )

    arr_embeddings_items = np.random.normal(
        loc=parameters.embeddings_mean,
        scale=parameters.embeddings_std_dev,
        size=(parameters.num_items, parameters.num_factors),
    ).astype(
        np.float32,
    )

    arr_biases_global = np.zeros(
        shape=(1,),
    ).astype(
        np.float32
    )

    arr_biases_users = np.zeros(
        shape=(parameters.num_users,),
    ).astype(
        np.float32
    )

    arr_biases_items = np.zeros(
        shape=(parameters.num_items,),
    ).astype(
        np.float32
    )

    return Model(
        arr_embeddings_users=arr_embeddings_users,
        arr_embeddings_items=arr_embeddings_items,
        arr_biases_global=arr_biases_global,
        arr_biases_users=arr_biases_users,
        arr_biases_items=arr_biases_items,
    )


def init_optimizers(
    sgd_mode: str,
    gamma: float,
    beta_1: float,
    beta_2: float,
    num_users: int,
    num_items: int,
    num_factors: int,
) -> Optimizers:
    optimizer_parameters = OptimizerParameters(
        sgd_mode=sgd_mode,
        gamma=gamma,
        beta_1=beta_1,
        beta_2=beta_2,
    )
    opt_embeddings_users = init_optimizer_state(
        parameters=optimizer_parameters,
        shape=(num_users, num_factors),
    )
    opt_embeddings_items = init_optimizer_state(
        parameters=optimizer_parameters,
        shape=(num_items, num_factors),
    )
    opt_biases_global = init_optimizer_state(
        parameters=optimizer_parameters,
        shape=(1, 1),
    )
    opt_biases_users = init_optimizer_state(
        parameters=optimizer_parameters,
        shape=(num_users, 1),
    )
    opt_biases_items = init_optimizer_state(
        parameters=optimizer_parameters,
        shape=(num_items, 1),
    )
    func_adaptive_gradient = get_adaptive_gradient_function(
        sgd_mode=sgd_mode,
    )
    func_after_gradient = get_after_gradient_function(
        sgd_mode=sgd_mode,
    )

    return Optimizers(
        func_adaptive_gradient=func_adaptive_gradient,
        func_after_gradient=func_after_gradient,
        embeddings_users=opt_embeddings_users,
        embeddings_items=opt_embeddings_items,
        biases_global=opt_biases_global,
        biases_users=opt_biases_users,
        biases_items=opt_biases_items,
    )


@nb.njit
def nb_init_samples(
    parameters: Parameters,
) -> Samples:
    arr_user_ids = np.empty(
        shape=(parameters.batch_size,),
    ).astype(
        np.int64
    )
    arr_item_ids = np.empty(
        shape=(parameters.batch_size,),
    ).astype(
        np.int64
    )
    arr_ratings = np.empty(
        shape=(parameters.batch_size,),
    ).astype(
        np.float32,
    )

    return Samples(
        arr_user_ids=arr_user_ids,
        arr_item_ids=arr_item_ids,
        arr_ratings=arr_ratings,
    )


@nb.njit
def nb_init_gradients(
    model: Model,
) -> Gradients:
    arr_embeddings_users = np.zeros_like(
        model.arr_embeddings_users,
    )
    arr_embeddings_items = np.zeros_like(
        model.arr_embeddings_items,
    )
    arr_biases_global = np.zeros_like(
        model.arr_biases_global,
    )
    arr_biases_users = np.zeros_like(
        model.arr_biases_users,
    )
    arr_biases_items = np.zeros_like(
        model.arr_biases_items,
    )
    return Gradients(
        arr_embeddings_users=arr_embeddings_users,
        arr_embeddings_items=arr_embeddings_items,
        arr_biases_global=arr_biases_global,
        arr_biases_users=arr_biases_users,
        arr_biases_items=arr_biases_items,
    )


@nb.njit
def init_mf_funk_svd(
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
    use_bias: bool,

    seed: int = 1234,
) -> tuple[CSRMatrix, Gradients, Losses, Model, Parameters, Samples]:
    nb_numpy_seed(seed=seed)

    csr_matrix = nb_init_csr_matrix(
        arr_indices=urm_csr_indices,
        arr_indptr=urm_csr_indptr,
        arr_data=urm_csr_data,
    )
    parameters = nb_init_parameters(
        embeddings_mean=embeddings_mean,
        embeddings_std_dev=embeddings_std_dev,
        num_users=num_users,
        num_items=num_items,
        reg_user=reg_user,
        reg_item=reg_item,
        reg_bias=reg_bias,
        batch_size=batch_size,
        frac_negative_sampling=frac_negative_sampling,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        num_factors=num_factors,
        num_samples=num_samples,
        seed=seed,
        use_bias=use_bias,
    )
    losses = nb_init_losses(
        parameters=parameters,
    )
    model = nb_init_model(
        parameters=parameters,
    )
    gradients = nb_init_gradients(
        model=model,
    )
    samples = nb_init_samples(
        parameters=parameters,
    )

    return (
        csr_matrix,
        gradients,
        losses,
        model,
        parameters,
        samples,
    )


@nb.njit
def nb_sample_funk_svd(
    csr_matrix: CSRMatrix,
    parameters: Parameters,
) -> Sample:
    sample_user, user_profile_start, user_profile_end, num_seen_items = nb_sample_user(
        num_users=parameters.num_users,
        num_items=parameters.num_items,
        urm_csr_indptr=csr_matrix.arr_indptr,
    )

    sample_positive = np.random.random() <= parameters.frac_negative_sampling

    if sample_positive:
        sample_item, sample_rating = nb_sample_positive_item(
            num_seen_items=num_seen_items,
            user_profile_start=user_profile_start,
            urm_csr_indices=csr_matrix.arr_indices,
            urm_csr_data=csr_matrix.arr_data,
        )
    else:
        sample_item, sample_rating = nb_sample_negative_item(
            num_items=parameters.num_items,
            num_seen_items=num_seen_items,
            user_profile_start=user_profile_start,
            urm_csr_indices=csr_matrix.arr_indices,
        )

    return Sample(
        user_id=sample_user,
        item_id=sample_item,
        rating=sample_rating,
    )


@nb.njit
def nb_compute_prediction_error(
    model: Model,
    parameters: Parameters,
    sample: Sample,
) -> float:
    prediction = 0.
    if parameters.use_bias:
        prediction += (
            model.arr_biases_global[0]
            + model.arr_biases_users[sample.user_id]
            + model.arr_biases_items[sample.item_id]
        )

    for idx_factor in nb.prange(parameters.num_factors):
        prediction += (
            model.arr_embeddings_users[sample.user_id, idx_factor]
            * model.arr_embeddings_items[sample.item_id, idx_factor]
        )

    prediction_error = sample.rating - prediction

    return prediction_error


@nb.njit
def nb_compute_gradients(
    model: Model,
    parameters: Parameters,
    gradients: Gradients,
    sample: Sample,
    prediction_error: float,
) -> None:
    if parameters.use_bias:
        local_gradient_global_bias = prediction_error - parameters.reg_bias * model.arr_biases_global[0]
        local_gradient_user_bias = prediction_error - parameters.reg_bias * model.arr_biases_users[sample.user_id]
        local_gradient_item_bias = prediction_error - parameters.reg_bias * model.arr_biases_items[sample.item_id]

        gradients.arr_biases_global[0] += local_gradient_global_bias
        gradients.arr_biases_users[sample.user_id] += local_gradient_user_bias
        gradients.arr_biases_items[sample.item_id] += local_gradient_item_bias

    for idx_factor in nb.prange(parameters.num_factors):
        # Copy original value to avoid messing up the updates
        W_u = model.arr_embeddings_users[sample.user_id, idx_factor]
        H_i = model.arr_embeddings_items[sample.item_id, idx_factor]

        # Compute gradients
        local_gradient_user = prediction_error * H_i - (parameters.reg_user * W_u)
        local_gradient_item = prediction_error * W_u - (parameters.reg_item * H_i)

        gradients.arr_embeddings_users[sample.user_id, idx_factor] += local_gradient_user
        gradients.arr_embeddings_items[sample.item_id, idx_factor] += local_gradient_item


@nb.njit
def nb_apply_gradients(
    model: Model,
    gradients: Gradients,
    samples: Samples,
    optimizers: Optimizers,
    parameters: Parameters,
) -> None:
    if parameters.use_bias:
        local_gradient_bias_global = gradients.arr_biases_global[0] / parameters.batch_size
        local_gradient_bias_global = optimizers.func_adaptive_gradient(
            optimizers.biases_global,
            local_gradient_bias_global,
            0,
            0,
        )

        # apply updates
        model.arr_biases_global[0] += parameters.learning_rate * local_gradient_bias_global

    for sample_item in samples.arr_item_ids:
        if parameters.use_bias:
            local_gradient_bias_item = gradients.arr_biases_items[sample_item] / parameters.batch_size
            local_gradient_bias_item = optimizers.func_adaptive_gradient(
                optimizers.biases_items,
                local_gradient_bias_item,
                sample_item,
                0,
            )

            # apply updates
            model.arr_biases_items[sample_item] += parameters.learning_rate * local_gradient_bias_item
            # arr_accumulator_bias_items[sample_item] = 0.

        for idx_factor in range(parameters.num_factors):
            local_gradient_item = gradients.arr_embeddings_items[sample_item, idx_factor] / parameters.batch_size
            local_gradient_item = optimizers.func_adaptive_gradient(
                optimizers.embeddings_items,
                local_gradient_item,
                sample_item,
                idx_factor,
            )

            model.arr_embeddings_items[sample_item, idx_factor] += parameters.learning_rate * local_gradient_item
            # arr_accumulator_item_embeddings[sample_item] = 0.

    for sample_user in samples.arr_user_ids:
        if parameters.use_bias:
            local_gradient_bias_user = gradients.arr_biases_users[sample_user] / parameters.batch_size
            local_gradient_bias_user = optimizers.func_adaptive_gradient(
                optimizers.biases_users,
                local_gradient_bias_user,
                sample_user,
                0,
            )

            model.arr_biases_users[sample_user] += parameters.learning_rate * local_gradient_bias_user
            # arr_accumulator_bias_users[sample_user] = 0.

        for idx_factor in range(parameters.num_factors):
            local_gradient_user = gradients.arr_embeddings_users[sample_user, idx_factor] / parameters.batch_size
            local_gradient_user = optimizers.func_adaptive_gradient(
                optimizers.embeddings_users,
                local_gradient_user,
                sample_user,
                idx_factor,
            )

            model.arr_embeddings_users[sample_user, idx_factor] += parameters.learning_rate * local_gradient_user
            # arr_accumulator_user_embeddings[sample_user] = 0.


@nb.njit
def nb_after_gradient(
    optimizers: Optimizers,
) -> None:
    optimizers.func_after_gradient(optimizers.biases_global)
    optimizers.func_after_gradient(optimizers.biases_users)
    optimizers.func_after_gradient(optimizers.biases_items)
    optimizers.func_after_gradient(optimizers.embeddings_users)
    optimizers.func_after_gradient(optimizers.embeddings_items)


@nb.njit
def nb_epoch_mf_funk_svd(
    epoch: int,
    csr_matrix: CSRMatrix,
    gradients: Gradients,
    losses: Losses,
    model: Model,
    optimizers: Optimizers,
    parameters: Parameters,
    samples: Samples,
) -> tuple[CSRMatrix, Gradients, Losses, Model, Optimizers, Parameters, Samples]:
    epoch_loss = 0.

    for _ in range(parameters.num_batches):
        gradients.arr_embeddings_users[:] = 0.
        gradients.arr_embeddings_items[:] = 0.
        gradients.arr_biases_global[:] = 0.
        gradients.arr_biases_users[:] = 0.
        gradients.arr_biases_items[:] = 0.

        for idx_sample in range(parameters.batch_size):
            sample = nb_sample_funk_svd(
                parameters=parameters,
                csr_matrix=csr_matrix,
            )

            samples.arr_user_ids[idx_sample] = sample.user_id
            samples.arr_item_ids[idx_sample] = sample.item_id
            samples.arr_ratings[idx_sample] = sample.rating

            prediction_error = nb_compute_prediction_error(
                model=model,
                parameters=parameters,
                sample=sample,
            )

            nb_compute_gradients(
                model=model,
                gradients=gradients,
                parameters=parameters,
                sample=sample,
                prediction_error=prediction_error,
            )

            epoch_loss += prediction_error ** 2

        nb_apply_gradients(
            model=model,
            gradients=gradients,
            samples=samples,
            optimizers=optimizers,
            parameters=parameters,
        )

        nb_after_gradient(
            optimizers=optimizers,
        )

    losses.arr_losses_epochs[epoch] = epoch_loss

    return (
        csr_matrix,
        gradients,
        losses,
        model,
        optimizers,
        parameters,
        samples,
    )


@nb.njit
def nb_train_all_epochs(
    urm_csr_indptr: np.ndarray,
    urm_csr_indices: np.ndarray,
    urm_csr_data: np.ndarray,

    optimizers: Optimizers,

    reg_user: float,
    reg_item: float,
    reg_bias: float,

    batch_size: int,
    embeddings_mean: float,
    embeddings_std_dev: float,
    frac_negative_sampling: float,
    learning_rate: float,
    num_samples: int,
    num_factors: int,
    num_epochs: int,
    num_users: int,
    num_items: int,

    seed: int,
    use_bias: bool,
) -> tuple[CSRMatrix, Gradients, Losses, Model, Optimizers, Parameters, Samples]:
    (
        csr_matrix,
        gradients,
        losses,
        model,
        parameters,
        samples,
    ) = init_mf_funk_svd(
        num_users=num_users,
        num_items=num_items,
        reg_user=reg_user,
        reg_item=reg_item,
        reg_bias=reg_bias,
        batch_size=batch_size,
        urm_csr_indices=urm_csr_indices,
        urm_csr_indptr=urm_csr_indptr,
        urm_csr_data=urm_csr_data,
        embeddings_mean=embeddings_mean,
        embeddings_std_dev=embeddings_std_dev,
        frac_negative_sampling=frac_negative_sampling,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        num_factors=num_factors,
        num_samples=num_samples,
        seed=seed,
        use_bias=use_bias,
    )

    for epoch in range(num_epochs):
        (
            csr_matrix,
            gradients,
            losses,
            model,
            optimizers,
            parameters,
            samples,
        ) = nb_epoch_mf_funk_svd(
            epoch=epoch,
            csr_matrix=csr_matrix,
            losses=losses,
            model=model,
            gradients=gradients,
            optimizers=optimizers,
            parameters=parameters,
            samples=samples,
        )

    return (
        csr_matrix,
        gradients,
        losses,
        model,
        optimizers,
        parameters,
        samples,
    )


run_epoch_funk_svd = nb_epoch_mf_funk_svd


def train_mf_funk_svd(
    urm_train: scipy.sparse.csr_matrix,
    optimizers: Optimizers,

    reg_user: float,
    reg_item: float,
    reg_bias: float,

    batch_size: int,
    embeddings_mean: float,
    embeddings_std_dev: float,
    frac_negative_sampling: float,
    learning_rate: float,
    num_factors: int,
    num_epochs: int,
    num_users: int,
    num_items: int,
    use_bias: bool,
    seed: int = 1234,
) -> tuple[CSRMatrix, Gradients, Losses, Model, Optimizers, Parameters, Samples]:
    assert urm_train.shape == (num_users, num_items)
    assert 0. <= frac_negative_sampling <= 1.

    urm_train = urm_train.astype(np.float32)
    urm_train.sort_indices()
    num_samples = urm_train.nnz

    (
        csr_matrix,
        gradients,
        losses,
        model,
        optimizers,
        parameters,
        samples,
    ) = nb_train_all_epochs(
        urm_csr_indptr=urm_train.indptr,
        urm_csr_indices=urm_train.indices,
        urm_csr_data=urm_train.data,
        num_users=num_users,
        num_items=num_items,
        reg_user=reg_user,
        reg_item=reg_item,
        reg_bias=reg_bias,
        batch_size=batch_size,
        embeddings_mean=embeddings_mean,
        embeddings_std_dev=embeddings_std_dev,
        frac_negative_sampling=frac_negative_sampling,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        num_factors=num_factors,
        num_samples=num_samples,
        optimizers=optimizers,
        seed=seed,
        use_bias=use_bias,
    )

    return (
        csr_matrix,
        gradients,
        losses,
        model,
        optimizers,
        parameters,
        samples,
    )
