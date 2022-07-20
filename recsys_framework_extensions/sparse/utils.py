import numpy as np
from scipy import sparse


def compute_item_popularity_from_urm(
    urm: sparse.csr_matrix
) -> np.ndarray:
    urm_csc = sparse.csc_matrix(urm)
    urm_csc.eliminate_zeros()

    arr_item_popularity = np.ediff1d(
        urm_csc.indptr,
    )

    return arr_item_popularity
