from typing import Optional

import attrs
import scipy.sparse as sp
from skopt.space import Integer, Real, Categorical


from recsys_framework_extensions.recommenders.base import (
    SearchHyperParametersBaseRecommender,
    FitParametersBaseRecommender,
)
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersSLIMBPRRecommender(SearchHyperParametersBaseRecommender):
    topK: Integer = attrs.field(
        default=Integer(
            low=5,
            high=1000,
            prior="uniform",
            base=10,
        )
    )
    epochs: Categorical = attrs.field(
        default=Categorical(
            [1500],
        )
    )
    symmetric: Categorical = attrs.field(
        default=Categorical(
            [True, False],
        ),
    )
    sgd_mode: Categorical = attrs.field(
        default=Categorical(
            ["sgd", "adagrad", "adam"],
        ),
    )
    lambda_i: Real = attrs.field(
        default=Real(
            low=1e-5,
            high=1e-2,
            prior="log-uniform",
            base=10,
        ),
    )
    lambda_j: Real = attrs.field(
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
class FitParametersSLIMBPRRecommender(FitParametersBaseRecommender):
    positive_threshold_BPR: Optional[int] = attrs.field(default=None)
    train_with_sparse_weights: bool = attrs.field(default=False)
    allow_train_with_sparse_weights: bool = attrs.field(default=False)


@attrs.define(kw_only=True, frozen=True, slots=False)
class FitParametersSLIMBPRRecommenderAllowSparseTraining(
    FitParametersSLIMBPRRecommender
):
    train_with_sparse_weights: bool = attrs.field(default=True)
    allow_train_with_sparse_weights: bool = attrs.field(default=True)


class ExtendedSLIMBPRRecommender(SLIM_BPR_Cython):
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
    "SearchHyperParametersSLIMBPRRecommender",
    "FitParametersSLIMBPRRecommender",
    "FitParametersSLIMBPRRecommenderAllowSparseTraining",
    "ExtendedSLIMBPRRecommender",
]
