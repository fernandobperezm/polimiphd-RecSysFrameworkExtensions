import numpy as np
import pytest

from recsys_framework_extensions.evaluation.statistics_tests import (
    compute_statistical_tests_of_base_vs_others,
    StatisticalTestAlternative,
    StatisticalTestHypothesis,
    ResultsStatisticalTestsBaseVsOthers,
)


_NUM_USERS_TO_EVALUATE = 100
_NUM_ITEMS_TO_EVALUATE = 1000


@pytest.fixture
def recommender_scores_from_zero_to_point2() -> np.ndarray:
    rng = np.random.default_rng(seed=1234)

    return (
        rng.integers(low=0, high=2000, size=_NUM_USERS_TO_EVALUATE, dtype=np.int64)
        / 10000.0
    )


@pytest.fixture
def recommender_scores_from_point1_to_point8() -> np.ndarray:
    rng = np.random.default_rng(seed=1234)

    return (
        rng.integers(low=1000, high=8000, size=_NUM_USERS_TO_EVALUATE, dtype=np.int64)
        / 10000.0
    )


@pytest.fixture
def recommender_scores_from_point6_to_one() -> np.ndarray:
    rng = np.random.default_rng(seed=1234)

    return (
        rng.integers(low=6000, high=10000, size=_NUM_USERS_TO_EVALUATE, dtype=np.int64)
        / 10000.0
    )


class TestStatisticalTests:
    @pytest.mark.parametrize(
        "test_alternative,expected_hypothesis",
        [
            (
                StatisticalTestAlternative.TWO_SIDED,
                StatisticalTestHypothesis.ALTERNATIVE,
            ),
            (
                StatisticalTestAlternative.GREATER,
                StatisticalTestHypothesis.ALTERNATIVE,
            ),
            (
                StatisticalTestAlternative.LESS,
                StatisticalTestHypothesis.NULL,
            ),
        ],
    )
    def test_wilcoxon(
        self,
        recommender_scores_from_zero_to_point2: np.ndarray,
        recommender_scores_from_point6_to_one: np.ndarray,
        test_alternative: StatisticalTestAlternative,
        expected_hypothesis: StatisticalTestHypothesis,
    ):
        # Arrange
        test_alphas = [0.05]
        scores_base = recommender_scores_from_zero_to_point2.astype(
            dtype=np.float64,
        )
        scores_others = np.vstack(
            (
                recommender_scores_from_point6_to_one,
                recommender_scores_from_point6_to_one,
                recommender_scores_from_point6_to_one,
                recommender_scores_from_point6_to_one,
            ),
        ).astype(
            dtype=np.float64,
        )

        # Act
        results_statistical_tests: ResultsStatisticalTestsBaseVsOthers = (
            compute_statistical_tests_of_base_vs_others(
                scores_base=scores_base,
                scores_others=scores_others,
                alphas=test_alphas,
                alternative=test_alternative,
            )
        )

        # Assert
        for results_wilcoxon_test in [
            results_statistical_tests.wilcoxon,
            results_statistical_tests.wilcoxon_zsplit,
        ]:
            for results in results_wilcoxon_test:
                p_value = results.p_value
                hypothesis = results.hypothesis

                assert np.all(np.array(hypothesis) == expected_hypothesis)
                assert (
                    np.all(StatisticalTestHypothesis.NULL == np.array(hypothesis))
                    and np.all(p_value > np.asarray(test_alphas))
                ) or (
                    np.all(
                        StatisticalTestHypothesis.ALTERNATIVE == np.array(hypothesis)
                    )
                    and np.all(p_value <= np.asarray(test_alphas))
                )
                assert (
                    np.all(StatisticalTestHypothesis.NULL == np.array(hypothesis))
                    and StatisticalTestAlternative.LESS == test_alternative
                ) or (
                    np.all(
                        StatisticalTestHypothesis.ALTERNATIVE == np.array(hypothesis)
                    )
                    and test_alternative
                    in [
                        StatisticalTestAlternative.TWO_SIDED,
                        StatisticalTestAlternative.GREATER,
                    ]
                )

    @pytest.mark.parametrize(
        "test_alternative,expected_hypothesis",
        [
            (
                StatisticalTestAlternative.TWO_SIDED,
                StatisticalTestHypothesis.ALTERNATIVE,
            ),
            (
                StatisticalTestAlternative.GREATER,
                StatisticalTestHypothesis.ALTERNATIVE,
            ),
            (
                StatisticalTestAlternative.LESS,
                StatisticalTestHypothesis.NULL,
            ),
        ],
    )
    def test_sign(
        self,
        recommender_scores_from_zero_to_point2: np.ndarray,
        recommender_scores_from_point6_to_one: np.ndarray,
        test_alternative: StatisticalTestAlternative,
        expected_hypothesis: StatisticalTestHypothesis,
    ):
        # Arrange
        test_alphas = [0.05]
        scores_base = recommender_scores_from_zero_to_point2.astype(
            dtype=np.float64,
        )
        scores_others = np.vstack(
            (
                recommender_scores_from_point6_to_one,
                recommender_scores_from_point6_to_one,
                recommender_scores_from_point6_to_one,
                recommender_scores_from_point6_to_one,
            ),
        ).astype(
            dtype=np.float64,
        )

        # Act
        dict_results = compute_statistical_tests_of_base_vs_others(
            scores_base=scores_base,
            scores_others=scores_others,
            alphas=test_alphas,
            alternative=test_alternative,
        )

        # Assert
        for results_rec in dict_results.sign:
            p_value = results_rec.p_value
            hypotheses = results_rec.hypothesis

            assert np.all(np.array(hypotheses) == expected_hypothesis)
            assert (
                np.all(StatisticalTestHypothesis.NULL == np.array(hypotheses))
                and np.all(p_value > np.asarray(test_alphas))
            ) or (
                np.all(StatisticalTestHypothesis.ALTERNATIVE == np.array(hypotheses))
                and np.all(p_value <= np.asarray(test_alphas))
            )
            assert (
                np.all(StatisticalTestHypothesis.NULL == np.array(hypotheses))
                and StatisticalTestAlternative.LESS == test_alternative
            ) or (
                np.all(StatisticalTestHypothesis.ALTERNATIVE == np.array(hypotheses))
                and test_alternative
                in [
                    StatisticalTestAlternative.TWO_SIDED,
                    StatisticalTestAlternative.GREATER,
                ]
            )

    @pytest.mark.parametrize(
        "test_alternative,expected_hypothesis",
        [
            (
                StatisticalTestAlternative.TWO_SIDED,
                StatisticalTestHypothesis.ALTERNATIVE,
            ),
            (StatisticalTestAlternative.GREATER, StatisticalTestHypothesis.ALTERNATIVE),
            (StatisticalTestAlternative.LESS, StatisticalTestHypothesis.NULL),
        ],
    )
    def test_bonferroni_corrections_wilcoxon_and_sign(
        self,
        recommender_scores_from_zero_to_point2: np.ndarray,
        recommender_scores_from_point6_to_one: np.ndarray,
        test_alternative: StatisticalTestAlternative,
        expected_hypothesis: StatisticalTestHypothesis,
    ):
        # Arrange
        test_alphas = [0.05]
        scores_base = recommender_scores_from_zero_to_point2.astype(
            dtype=np.float64,
        )
        scores_others = np.vstack(
            (
                recommender_scores_from_point6_to_one,
                recommender_scores_from_point6_to_one,
                recommender_scores_from_point6_to_one,
                recommender_scores_from_point6_to_one,
            ),
        ).astype(
            dtype=np.float64,
        )

        # Act
        dict_results = compute_statistical_tests_of_base_vs_others(
            scores_base=scores_base,
            scores_others=scores_others,
            alphas=test_alphas,
            alternative=test_alternative,
        )

        # Assert
        for dict_results_bonferroni, dict_results_test in zip(
            [
                dict_results.bonferroni_wilcoxon,
                dict_results.bonferroni_sign,
            ],
            [
                dict_results.wilcoxon,
                dict_results.sign,
            ],
        ):
            num_tested_scores = scores_others.shape[0]
            corrected_alphas = dict_results_bonferroni.corrected_alphas
            corrected_p_values = dict_results_bonferroni.corrected_p_values
            hypothesis = dict_results_bonferroni.hypothesis

            assert np.allclose(
                corrected_alphas, np.asarray(test_alphas) / num_tested_scores
            )
            assert np.allclose(
                corrected_p_values,
                np.asarray(
                    [
                        min(res.p_value * num_tested_scores, 1.0)
                        for res in dict_results_test
                    ]
                ),
            )
            assert np.all(expected_hypothesis == np.array(hypothesis))
            assert (
                np.all(StatisticalTestHypothesis.NULL == np.array(hypothesis))
                and np.all(
                    np.ravel(
                        [
                            p_val > np.asarray(test_alphas)
                            for p_val in corrected_p_values
                        ]
                    )
                )
            ) or (
                np.all(StatisticalTestHypothesis.ALTERNATIVE == np.array(hypothesis))
                and np.all(
                    np.ravel(
                        [
                            p_val <= np.asarray(test_alphas)
                            for p_val in corrected_p_values
                        ]
                    )
                )
            )
            assert (
                np.all(StatisticalTestHypothesis.NULL == np.array(hypothesis))
                and StatisticalTestAlternative.LESS == test_alternative
            ) or (
                np.all(StatisticalTestHypothesis.ALTERNATIVE == np.array(hypothesis))
                and test_alternative
                in [
                    StatisticalTestAlternative.TWO_SIDED,
                    StatisticalTestAlternative.GREATER,
                ]
            )

    def test_friedman_chi_square(
        self,
        recommender_scores_from_zero_to_point2: np.ndarray,
        recommender_scores_from_point6_to_one: np.ndarray,
    ):
        # Arrange
        expected_hypothesis = StatisticalTestHypothesis.ALTERNATIVE

        test_alphas = [0.05]
        test_alternative = StatisticalTestAlternative.TWO_SIDED

        scores_base = recommender_scores_from_zero_to_point2.astype(
            dtype=np.float64,
        )
        scores_others = np.vstack(
            (
                recommender_scores_from_point6_to_one - 0.2,
                recommender_scores_from_point6_to_one - 0.1,
                recommender_scores_from_point6_to_one,
                recommender_scores_from_point6_to_one + 0.1,
                recommender_scores_from_point6_to_one + 0.2,
                recommender_scores_from_point6_to_one + 0.3,
                recommender_scores_from_point6_to_one + 0.4,
            ),
        ).astype(
            dtype=np.float64,
        )

        # Act
        dict_results = compute_statistical_tests_of_base_vs_others(
            scores_base=scores_base,
            scores_others=scores_others,
            alphas=test_alphas,
            alternative=test_alternative,
        )

        # Assert
        p_value = dict_results.friedman.p_value
        p_value_reliable = dict_results.friedman.p_value_reliable
        hypothesis = dict_results.friedman.hypothesis

        assert p_value_reliable
        assert np.all(p_value <= np.array(test_alphas))
        assert np.all(expected_hypothesis == np.array(hypothesis))
