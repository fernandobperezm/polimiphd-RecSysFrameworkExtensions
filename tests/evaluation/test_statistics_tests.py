import numpy as np
import pytest

from recsys_framework_extensions.evaluation.statistics_tests import (
    compute_statistical_tests_of_base_vs_others,
    StatisticalTestAlternative,
    StatisticalTestHypothesis,
)


_NUM_USERS_TO_EVALUATE = 100
_NUM_ITEMS_TO_EVALUATE = 1000


@pytest.fixture
def recommender_scores_up_to_point2() -> np.ndarray:
    rng = np.random.default_rng(seed=1234)

    return rng.integers(
        low=0,
        high=2000,
        size=_NUM_USERS_TO_EVALUATE,
        dtype=np.int64
    ) / 10000.


@pytest.fixture
def recommender_scores_from_point1_to_point8() -> np.ndarray:
    rng = np.random.default_rng(seed=1234)

    return rng.integers(
        low=1000,
        high=8000,
        size=_NUM_USERS_TO_EVALUATE,
        dtype=np.int64
    ) / 10000.


@pytest.fixture
def recommender_scores_from_point6() -> np.ndarray:
    rng = np.random.default_rng(seed=1234)

    return rng.integers(
        low=6000,
        high=10000,
        size=_NUM_USERS_TO_EVALUATE,
        dtype=np.int64
    ) / 10000.


class TestStatisticalTests:

    @pytest.mark.parametrize(
        "test_alternative,expected_hypothesis",
        [
            (StatisticalTestAlternative.TWO_SIDED, StatisticalTestHypothesis.ALTERNATIVE),
            (StatisticalTestAlternative.GREATER, StatisticalTestHypothesis.ALTERNATIVE),
            (StatisticalTestAlternative.LESS, StatisticalTestHypothesis.NULL),
        ]
    )
    def test_wilcoxon_and_sign(
        self,
        recommender_scores_up_to_point2: np.ndarray,
        recommender_scores_from_point6: np.ndarray,
        test_alternative: StatisticalTestAlternative,
        expected_hypothesis: StatisticalTestHypothesis,
    ):
        # Arrange
        test_alpha = 0.05
        scores_base = recommender_scores_up_to_point2.astype(
            dtype=np.float64,
        )
        scores_others = np.vstack(
            (
                recommender_scores_from_point6,
                recommender_scores_from_point6,
                recommender_scores_from_point6,
                recommender_scores_from_point6,
            ),
        ).astype(
            dtype=np.float64,
        )

        # Act
        dict_results = compute_statistical_tests_of_base_vs_others(
            scores_base=scores_base,
            scores_others=scores_others,
            alpha=test_alpha,
            alternative=test_alternative,
        )

        # Assert
        for list_results in [
            dict_results["wilcoxon"],
            dict_results["sign"],
        ]:
            for results in list_results:
                p_value = results["p_value"]
                hypothesis = results["hypothesis"]

                assert hypothesis == expected_hypothesis
                assert (
                    StatisticalTestAlternative.LESS == test_alternative
                    and p_value > test_alpha
                ) or (p_value <= test_alpha)

    @pytest.mark.parametrize(
        "test_alternative,expected_hypothesis",
        [
            (StatisticalTestAlternative.TWO_SIDED, StatisticalTestHypothesis.ALTERNATIVE),
            (StatisticalTestAlternative.GREATER, StatisticalTestHypothesis.ALTERNATIVE),
            (StatisticalTestAlternative.LESS, StatisticalTestHypothesis.NULL),
        ]
    )
    def test_bonferroni_corrections_wilcoxon_and_sign(
        self,
        recommender_scores_up_to_point2: np.ndarray,
        recommender_scores_from_point6: np.ndarray,
        test_alternative: StatisticalTestAlternative,
        expected_hypothesis: StatisticalTestHypothesis,
    ):
        # Arrange
        test_alpha = 0.05
        scores_base = recommender_scores_up_to_point2.astype(
            dtype=np.float64,
        )
        scores_others = np.vstack(
            (
                recommender_scores_from_point6,
                recommender_scores_from_point6,
                recommender_scores_from_point6,
                recommender_scores_from_point6,
            ),
        ).astype(
            dtype=np.float64,
        )

        # Act
        dict_results = compute_statistical_tests_of_base_vs_others(
            scores_base=scores_base,
            scores_others=scores_others,
            alpha=test_alpha,
            alternative=test_alternative,
        )

        # Assert
        for dict_results_bonferroni in [
            dict_results["bonferroni_wilcoxon"],
            dict_results["bonferroni_sign"],
        ]:
            num_tested_scores = scores_others.shape[0]
            reject = dict_results_bonferroni["reject"]
            alpha_bonferroni = dict_results_bonferroni["corrected_alphas"]
            p_values_corrected = dict_results_bonferroni["corrected_p_values"]
            hypothesis = dict_results_bonferroni["hypothesis"]

            assert np.isclose(
                alpha_bonferroni,
                test_alpha / num_tested_scores
            )
            assert np.all(
                expected_hypothesis == hypothesis
            )
            assert (
                StatisticalTestAlternative.LESS == test_alternative
                and not np.any(reject)
            ) or np.all(reject)
            assert (
                StatisticalTestAlternative.LESS == test_alternative
                and np.all(p_values_corrected > test_alpha)
            ) or np.all(p_values_corrected <= test_alpha)

    def test_friedman_chi_square(
        self,
        recommender_scores_up_to_point2: np.ndarray,
        recommender_scores_from_point6: np.ndarray,
    ):
        # Arrange
        expected_hypothesis = StatisticalTestHypothesis.ALTERNATIVE

        test_alpha = 0.05
        test_alternative = StatisticalTestAlternative.TWO_SIDED

        scores_base = recommender_scores_up_to_point2.astype(
            dtype=np.float64,
        )
        scores_others = np.vstack(
            (
                recommender_scores_from_point6 - 0.2,
                recommender_scores_from_point6 - 0.1,
                recommender_scores_from_point6,
                recommender_scores_from_point6 + 0.1,
                recommender_scores_from_point6 + 0.2,
                recommender_scores_from_point6 + 0.3,
                recommender_scores_from_point6 + 0.4,
            ),
        ).astype(
            dtype=np.float64,
        )

        # Act
        dict_results = compute_statistical_tests_of_base_vs_others(
            scores_base=scores_base,
            scores_others=scores_others,
            alpha=test_alpha,
            alternative=test_alternative,
        )

        # Assert
        dict_result_friedman_chi_square = dict_results["friedman_chi_square"]
        p_value = dict_result_friedman_chi_square["p_value"]
        p_value_reliable = dict_result_friedman_chi_square["p_value_reliable"]
        hypothesis = dict_result_friedman_chi_square["hypothesis"]

        assert p_value <= test_alpha
        assert p_value_reliable
        assert expected_hypothesis == hypothesis
