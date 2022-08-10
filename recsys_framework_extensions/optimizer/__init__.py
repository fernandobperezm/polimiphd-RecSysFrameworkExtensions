import numba as nb
import numpy as np


@nb.experimental.jitclass([
    ("sgd_cache", nb.float64[:, :]),
    ("sgd_momentum_1", nb.float64[:, :]),
    ("sgd_momentum_2", nb.float64[:, :]),
    ("gamma", nb.float64),
    ("beta_1", nb.float64),
    ("beta_2", nb.float64),
    ("beta_1_power_t", nb.float64),
    ("beta_2_power_t", nb.float64),
])
class NumbaOptimizer:
    sgd_cache: np.ndarray
    sgd_momentum_1: np.ndarray
    sgd_momentum_2: np.ndarray
    sgd_mode: str
    gamma: float
    beta_1: float
    beta_2: float

    def __init__(
        self,
        sgd_mode: str,
        shape: tuple[int, int],
        gamma: float,
        beta_1: float,
        beta_2: float,
    ):
        self.sgd_cache = np.zeros(shape=shape, dtype=np.float64)
        self.sgd_momentum_1 = np.zeros(shape=shape, dtype=np.float64)
        self.sgd_momentum_2 = np.zeros(shape=shape, dtype=np.float64)
        self.sgd_mode = sgd_mode

        self.gamma = gamma

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_1_power_t = beta_1
        self.beta_2_power_t = beta_2

    def _gradient_adagrad(
        self,
        gradient: float,
        idx_row: int,
        idx_col: int,
    ) -> float:
        self.sgd_cache[idx_row, idx_col] += gradient ** 2

        gradient_update = (
            gradient /
            (
                np.sqrt(
                    self.sgd_cache[idx_row, idx_col]
                )
                + 1e-8
            )
        )

        return gradient_update

    def _gradient_rmsprop(
        self,
        gradient: float,
        idx_row: int,
        idx_col: int,
    ) -> float:
        self.sgd_cache[idx_row, idx_col] = (
            self.sgd_cache[idx_row, idx_col]
            * self.gamma
            + (1 - self.gamma) * gradient ** 2
        )

        gradient_update = (
            gradient /
            (
                np.sqrt(
                    self.sgd_cache[idx_row, idx_col]
                )
                + 1e-8
            )
        )

        return gradient_update

    def _gradient_adam(
        self,
        gradient: float,
        idx_row: int,
        idx_col: int,
    ) -> float:
        self.sgd_momentum_1[idx_row, idx_col] = (
            self.sgd_momentum_1[idx_row, idx_col] * self.beta_1 + (1 - self.beta_1) * gradient
        )

        self.sgd_momentum_2[idx_row, idx_col] = (
            self.sgd_momentum_2[idx_row, idx_col] * self.beta_2 + (1 - self.beta_2) * gradient
        )

        momentum_1 = self.sgd_momentum_1[idx_row, idx_col] / (1 - self.beta_1_power_t)
        momentum_2 = self.sgd_momentum_2[idx_row, idx_col] / (1 - self.beta_2_power_t)

        gradient_update = (
            momentum_1
            / (np.sqrt(momentum_2) + 1e-8)
        )

        return gradient_update

    def _gradient_sgd(
        self,
        gradient: float,
    ) -> float:
        return gradient

    def adaptive_gradient(
        self,
        gradient: float,
        idx_row: int,
        idx_col: int,
    ):
        if "adagrad" == self.sgd_mode:
            return self._gradient_adagrad(
                gradient=gradient,
                idx_col=idx_col,
                idx_row=idx_row,
            )

        elif "rmsprop" == self.sgd_mode:
            return self._gradient_rmsprop(
                gradient=gradient,
                idx_col=idx_col,
                idx_row=idx_row,
            )

        elif "adam" == self.sgd_mode:
            self._gradient_adam(
                gradient=gradient,
                idx_col=idx_col,
                idx_row=idx_row,
            )

        else:
            return self._gradient_sgd(gradient=gradient)

    def after_batch(self):
        if "adam" == self.sgd_mode:
            self.beta_1_power_t *= self.beta_1
            self.beta_2_power_t *= self.beta_2


@nb.experimental.jitclass([("sgd_cache", nb.float64[:, :])])
class NumbaAdaGradOptimizer:
    sgd_cache: np.ndarray

    def __init__(self, shape: tuple[int, int]):
        self.sgd_cache = np.zeros(shape=shape, dtype=np.float64)

    def adaptive_gradient(
        self,
        gradient: float,
        idx_row: int,
        idx_col: int,
        *args,
        **kwargs
    ):
        self.sgd_cache[idx_row, idx_col] += gradient ** 2

        gradient_update = (
            gradient /
            (
                np.sqrt(
                    self.sgd_cache[idx_row, idx_col]
                )
                + 1e-8
            )
        )

        return gradient_update

    def after_batch(self, *args, **kwargs):
        pass


@nb.experimental.jitclass([("sgd_cache", nb.float64[:, :]), ("gamma", nb.float64)])
class NumbaRMSPropOptimizer:
    sgd_cache: np.ndarray
    gamma: float

    def __init__(self, shape: tuple[int, int], gamma: float = 0.9):
        self.sgd_cache = np.zeros(shape=shape, dtype=np.float64)
        self.gamma = gamma

    def adaptive_gradient(
        self,
        gradient: float,
        idx_row: int,
        idx_col: int,
        *args,
        **kwargs
    ):
        self.sgd_cache[idx_row, idx_col] = (
            self.sgd_cache[idx_row, idx_col]
            * self.gamma
            + (1 - self.gamma) * gradient ** 2
        )

        gradient_update = (
            gradient /
            (
                np.sqrt(
                    self.sgd_cache[idx_row, idx_col]
                )
                + 1e-8
            )
        )

        return gradient_update

    def after_batch(self, *args, **kwargs):
        pass


@nb.experimental.jitclass([
    ("sgd_momentum_1", nb.float64[:, :]),
    ("sgd_momentum_2", nb.float64[:, :]),
    ("beta_1", nb.float64),
    ("beta_2", nb.float64),
    ("beta_1_power_t", nb.float64),
    ("beta_2_power_t", nb.float64),
])
class NumbaAdamOptimizer:
    sgd_momentum_1: np.ndarray
    sgd_momentum_2: np.ndarray
    beta_1: float
    beta_2: float
    beta_1_power_t: float
    beta_2_power_t: float

    def __init__(
        self,
        shape: tuple[int, int],
        beta_1: float = 0.9,
        beta_2: float = 0.999,
    ):
        self.sgd_momentum_1 = np.zeros(shape=shape, dtype=np.float64)
        self.sgd_momentum_2 = np.zeros(shape=shape, dtype=np.float64)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_1_power_t = beta_1
        self.beta_2_power_t = beta_2

    def adaptive_gradient(
        self,
        gradient: float,
        idx_row: int,
        idx_col: int,
        *args,
        **kwargs
    ):
        self.sgd_momentum_1[idx_row, idx_col] = (
            self.sgd_momentum_1[idx_row, idx_col] * self.beta_1 + (1 - self.beta_1) * gradient
        )

        self.sgd_momentum_2[idx_row, idx_col] = (
            self.sgd_momentum_2[idx_row, idx_col] * self.beta_2 + (1 - self.beta_2) * gradient
        )

        momentum_1 = self.sgd_momentum_1[idx_row, idx_col] / (1 - self.beta_1_power_t)
        momentum_2 = self.sgd_momentum_2[idx_row, idx_col] / (1 - self.beta_2_power_t)

        gradient_update = (
            momentum_1
            / (np.sqrt(momentum_2) + 1e-8)
        )

        return gradient_update

    def after_batch(self, *args, **kwargs):
        self.beta_1_power_t *= self.beta_1
        self.beta_2_power_t *= self.beta_2


@nb.experimental.jitclass
class NumbaSGDOptimizer:
    def __init__(self):
        pass

    def adaptive_gradient(
        self,
        gradient: float,
        idx_row: int,
        idx_col: int,
        *args,
        **kwargs,
    ):
        return gradient

    def after_batch(self, *args, **kwargs):
        pass
