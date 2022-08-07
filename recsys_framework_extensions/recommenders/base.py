from __future__ import annotations

from typing import Type, TypeVar, Callable, Optional
from typing_extensions import ParamSpec

import abc
import attrs
import scipy.sparse as sp
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender

import logging
from recsys_framework_extensions.recommenders.mixins import MixinLoadModel


logger = logging.getLogger(__name__)


_RecommenderParams = ParamSpec("_RecommenderParams")


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersBaseRecommender(abc.ABC):
    pass


class AbstractExtendedBaseRecommender(MixinLoadModel, BaseRecommender, abc.ABC):
    @abc.abstractmethod
    def __init__(self, urm_train: sp.csr_matrix, **kwargs):
        super().__init__(URM_train=urm_train)
        ...

    @abc.abstractmethod
    def fit(self, *args, **kwargs) -> None:
        # This method exists solely to avoid warnings that the implementation of `fit` does not match the
        # BaseRecommender definition. This warning is issued because the `fit` in BaseRecommender does not has
        # *args and **kwargs.
        ...

    @abc.abstractmethod
    def validate_load_trained_recommender(self, *args, **kwargs) -> None:
       ...

    # @classmethod
    # def load_trained_recommender(
    #     cls: Callable[_RecommenderParams, _RecommenderExtendedInstance],
    #     urm_train: sp.csr_matrix,
    #     folder_path: str,
    #     file_name_postfix: str,
    #     *args: _RecommenderParams.args,
    #     **kwargs: _RecommenderParams.kwargs,
    # ) -> Optional[_RecommenderExtendedInstance]:
    #     """
    #     Convenience classmethod that lets classes inheriting from this abstract class to define their own arguments
    #     required to load a trained recommender from disk.
    #
    #     """
    #     validated = cls._validate_load_trained_recommender(**kwargs)
    #     if not validated:
    #         return None
    #
    #     return load_extended_recommender(
    #         recommender_class=cls,
    #         urm_train=urm_train,
    #         folder_path=folder_path,
    #         file_name_postfix=file_name_postfix,
    #         **kwargs,
    #     )


_RecommenderExtendedInstance = TypeVar("_RecommenderExtendedInstance", bound=AbstractExtendedBaseRecommender)
_RecommenderBaseInstance = TypeVar("_RecommenderBaseInstance", bound=BaseRecommender)


# @catch_exception_return_none
def load_extended_recommender(
    recommender_class: Callable[_RecommenderParams, _RecommenderExtendedInstance],
    folder_path: str,
    file_name_postfix: str,
    *args: _RecommenderParams.args,
    **kwargs: _RecommenderParams.kwargs,
) -> Optional[_RecommenderExtendedInstance]:
    try:
        recommender_instance = recommender_class(
            *args,
            **kwargs,
        )
        recommender_instance.load_model(
            folder_path=folder_path,
            file_name=f"{recommender_instance.RECOMMENDER_NAME}_{file_name_postfix}",
        )
        recommender_instance.validate_load_trained_recommender()
    except Exception as e:
        logger.exception(f"Could not load recommender {recommender_class} with \n\t* {args=} \n\t* {kwargs=}")
        return None

    return recommender_instance


# @catch_exception_return_none
def load_recsys_framework_recommender(
    *,
    recommender_class: Type[_RecommenderBaseInstance],
    folder_path: str,
    file_name_postfix: str,
    urm_train: sp.csr_matrix,
    similarity: Optional[str] = None,
) -> Optional[_RecommenderBaseInstance]:
    try:
        recommender_name = recommender_class.RECOMMENDER_NAME
        if issubclass(recommender_class, (ItemKNNCFRecommender, UserKNNCFRecommender)):
            if similarity is None or similarity == "":
                raise ValueError(f"Received an empty similarity when expecting a similarity value. Accepted values are ["
                                 f"{['cosine', 'tversky', 'asymmetric', 'jaccard', 'dice']}")

            recommender_name = f"{recommender_class.RECOMMENDER_NAME}_{similarity}"

        recommender_instance = recommender_class(
            URM_train=urm_train,
        )
        recommender_instance.load_model(
            folder_path=folder_path,
            file_name=f"{recommender_name}_{file_name_postfix}",
        )
        recommender_instance.RECOMMENDER_NAME = recommender_name

    except:
        return None

    return recommender_instance
