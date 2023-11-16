from typing import Optional

import attrs
import scipy.sparse as sp
from skopt.space import Integer, Real


from recsys_framework_extensions.recommenders.base import (
    SearchHyperParametersBaseRecommender,
)
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender


hyperparameters_range_dictionary = {
    "topK": Integer(5, 1000),
    "l1_ratio": Real(low=1e-5, high=1.0, prior="log-uniform"),
    "alpha": Real(low=1e-3, high=1.0, prior="uniform"),
}


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersSLIMElasticNetRecommender(
    SearchHyperParametersBaseRecommender
):
    topK: Integer = attrs.field(
        default=Integer(
            low=5,
            high=1000,
            prior="uniform",
            base=10,
        )
    )
    l1_ratio: Real = attrs.field(
        default=Real(
            low=1e-5,
            high=1.0,
            prior="log-uniform",
            base=10,
        ),
    )
    alpha: Real = attrs.field(
        default=Real(
            low=1e-3,
            high=1.0,
            prior="uniform",
            base=10,
        ),
    )


class ExtendedSLIMElasticNetRecommender(SLIMElasticNetRecommender):
    def __init__(
        self,
        *,
        urm_train: sp.csr_matrix,
        **kwargs,
    ):
        super().__init__(
            URM_train=urm_train,
        )
