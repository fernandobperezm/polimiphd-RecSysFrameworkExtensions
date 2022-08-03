from typing import Any, Sequence

import numpy as np
from scipy import sparse

from Utils.assertions_on_data_for_experiments import assert_disjoint_matrices


def ensure_csr_matrix(
    urm: Any,
) -> None:
    validation = sparse.isspmatrix_csr(urm)
    if not validation:
        raise ValueError(
            f"Expected an sparse csr matrix from class {sparse.csr_matrix}. Obtained: {type(urm)}"
        )


def ensure_implicit_dataset(
    urm: sparse.csr_matrix,
) -> None:
    validation = np.all(urm.data == 1.)
    if not validation:
        raise ValueError(
            f"Expected an implicit (only 1s, no explicit zeros) urm. Number of non-one elements: "
            f"{urm.data[urm.data != 1].size}"
        )


def ensure_leave_k_out_dataset(
    urm: sparse.csr_matrix,
    k: int = 1,
) -> None:
    validation = np.all(np.ediff1d(urm.indptr) == k)
    if not validation:
        raise ValueError(
            f"Expected a leave-{k}-out dataset, i.e., with all users having {k} interactions. Number of users with "
            f"different than {k} interactions: {urm.indices[np.ediff1d(urm.indices) == k].size}"
        )


def ensure_leave_one_out_dataset(
    urm: sparse.csr_matrix,
) -> None:
    return ensure_leave_k_out_dataset(
        urm=urm,
        k=1,
    )


def ensure_disjoint_sparse_matrices(
    urm_list: Sequence[sparse.csr_matrix],
) -> None:
    return assert_disjoint_matrices(
        URM_list=urm_list,
    )
