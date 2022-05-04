from typing import Sequence

import numpy as np
import numba


@numba.jit(nopython=True, parallel=False)
def _compute_rank_by_indices(
    arr_sorted_indices_by_row: np.ndarray,
) -> np.ndarray:
    assert len(arr_sorted_indices_by_row.shape) == 2
    num_rows, num_cols = arr_sorted_indices_by_row.shape

    arr_scores = np.empty_like(arr_sorted_indices_by_row, dtype=np.float32)

    for idx_row in range(num_rows):
        rank_value = 1
        for idx_col in range(num_cols):
            sorted_idx_col = arr_sorted_indices_by_row[idx_row, idx_col]
            arr_scores[idx_row, sorted_idx_col] = rank_value
            rank_value += 1

    return arr_scores


def rank_data_by_row(keys: Sequence[np.ndarray]) -> np.ndarray:
    """

    Parameters
    ----------
    keys: Sequence of numpy arrays
        A sequence of 2D arrays to rank data row-wise. The rank is created by ordering items using as keys the arrays
        from right-to-left (first ranks considering values in keys[-1], then keys[-2], keys[-3]...). This odd
        behavior is because internally this function uses `numpy.lexsort` and this function requires keys to be in
        this way.

    Notes
    -----
    This method is equivalent to `scipy.stats.rankdata(a, method="ordinal", axis=1)` but faster because
    `numpy.lexsort` is faster than `numpy.argsort` (this last used internally by scipy.stats.rankdata).

    Returns
    -------
    numpy.ndarray
        A numpy array with `ordinal` values indicating the rank of each item , i.e., the lowest score is 1 and the
        highest N, being N the number of columns.

    Examples
    --------
    If there are no ties, then the ranking is given by the right-most array (only considers `[1, 7, 2]` in the
    following example)

    >>> from numpy import array, float32
    >>> rank_data_by_row(keys=(
    ...     array([[9, 8, 7]]),
    ...     array([[1, 7, 2]]),
    ... ))
    array([[1., 3., 2.]], dtype=float32)

    Ties in an array are resolved by immediate left array to the current key (considers `[1, 1, 1]` then
    `[9, 8, 7]` in the following example)

    >>> from numpy import array, float32
    >>> rank_data_by_row(keys=(
    ...     array([[9, 8, 7]]),
    ...     array([[1, 1, 1]]),
    ... ))
    array([[3., 2., 1.]], dtype=float32)

    Each row is ranked independently of each other and the resulting array will have the same dimension as the
    input arrays (ranks

    >>> from numpy import array, float32
    >>> rank_data_by_row(keys=(
    ...     array([[9, 8, 7], [1, 2, 3]]),
    ...     array([[1, 1, 1], [1, 1, 1]]),
    ... ))
    array([[3., 2., 1.],
           [1., 2., 3.]], dtype=float32)
    """
    arr_sorted_indices_by_row: np.ndarray = np.lexsort(
        keys=keys,
        axis=1,
    )

    return _compute_rank_by_indices(arr_sorted_indices_by_row=arr_sorted_indices_by_row)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

else:
    # jit-compile this function with expected values.
    rank_data_by_row(keys=(
        np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32),
        np.array([[4., 5., 6.], [1., 2., 3.]], dtype=np.float32),
    ))

    rank_data_by_row(keys=(
        np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32),
        np.array([[4., 5., 6.], [1., 2., 3.]], dtype=np.float64),
    ))

    rank_data_by_row(keys=(
        np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float64),
        np.array([[4., 5., 6.], [1., 2., 3.]], dtype=np.float32),
    ))

    rank_data_by_row(keys=(
        np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float64),
        np.array([[4., 5., 6.], [1., 2., 3.]], dtype=np.float64),
    ))
