from typing import Optional

import attrs
import scipy.sparse as sp
from skopt.space import Integer, Real, Categorical


from recsys_framework_extensions.recommenders.base import (
    SearchHyperParametersBaseRecommender,
    FitParametersBaseRecommender,
)
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import (
    MatrixFactorization_BPR_Cython,
)


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersMFBPRRecommender(SearchHyperParametersBaseRecommender):
    sgd_mode: Categorical = attrs.field(
        default=Categorical(
            ["sgd", "adagrad", "adam"],
        )
    )
    epochs: Categorical = attrs.field(
        default=Categorical(
            [1500],
        )
    )
    num_factors: Integer = attrs.field(
        default=Integer(
            low=1,
            high=200,
            prior="uniform",
            base=10,
        )
    )
    batch_size: Categorical = attrs.field(
        default=Categorical(
            [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        ),
    )
    positive_reg: Real = attrs.field(
        default=Real(
            low=1e-5,
            high=1e-2,
            prior="log-uniform",
            base=10,
        ),
    )
    negative_reg: Real = attrs.field(
        default=Real(
            low=1e-5,
            high=1e-2,
            prior="log-uniform",
            base=10,
        ),
    )
    learning_rate: Real = attrs.field(
        default=Real(
            low=1e-4,
            high=1e-1,
            prior="log-uniform",
            base=10,
        ),
    )


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersMFBPRRecommenderWithDropout(
    SearchHyperParametersMFBPRRecommender
):
    dropout_quota: Real = Real(
        low=0.01,
        high=0.7,
        prior="uniform",
        base=10,
    )


@attrs.define(kw_only=True, frozen=True, slots=False)
class FitParametersMFBPRRecommender(FitParametersBaseRecommender):
    positive_threshold_BPR: Optional[int] = attrs.field(default=None)


class ExtendedMFBPRRecommender(MatrixFactorization_BPR_Cython):
    def __init__(
        self,
        *,
        urm_train: sp.csr_matrix,
        **kwargs,
    ):
        super().__init__(
            URM_train=urm_train,
        )


__all__ = [
    "SearchHyperParametersMFBPRRecommender",
    "SearchHyperParametersMFBPRRecommenderWithDropout",
    "FitParametersMFBPRRecommender",
    "ExtendedMFBPRRecommender",
]
