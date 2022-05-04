import abc

import attrs


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersBaseRecommender(abc.ABC):
    pass
