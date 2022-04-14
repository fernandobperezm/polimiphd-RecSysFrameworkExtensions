from enum import Enum
from typing import Literal

import numpy as np
import scipy.sparse as sp


class EvaluationStrategy(Enum):
    TIMESTAMP = "TIMESTAMP"
    LEAVE_LAST_K_OUT = "LEAVE_LAST_K_OUT"


T_EXCLUDE = Literal["users", "items"]


def exclude_from_evaluation(
    urm_test: sp.csr_matrix,
    frac_to_exclude: float,
    type_to_exclude: T_EXCLUDE,
    seed: int,
) -> np.ndarray:
    """

    Parameters
    ----------
    seed
    type_to_exclude
    frac_to_exclude
    urm_test

    Returns
    -------

    """
    assert 0.0 <= frac_to_exclude <= 1.0

    if "users" == type_to_exclude:
        num_values = urm_test.shape[0]
    elif "items" == type_to_exclude:
        num_values = urm_test.shape[1]
    else:
        raise ValueError("")

    arr_values_all = np.arange(
        start=0,
        step=1,
        stop=num_values,
        dtype=np.int32,
    )

    arr_values_size = int(num_values * frac_to_exclude)

    if 0 == arr_values_size:
        return np.array([], dtype=np.int32)
    elif 1 == arr_values_size:
        return arr_values_all
    else:
        return np.random.default_rng(seed=seed).choice(
            a=arr_values_all,
            size=arr_values_size,
            replace=False,
            shuffle=True,
            p=None,  # This ensures uniformly-sampled values.
        )
