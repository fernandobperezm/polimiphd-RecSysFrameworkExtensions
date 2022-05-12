from enum import Enum
from typing import Any, Optional

import attrs
import numpy as np
import statsmodels.api as sm
from scipy import stats

from recsys_framework_extensions.decorators import timeit
from recsys_framework_extensions.logging import get_logger

logger = get_logger(
    logger_name=__name__,
)


_MAX_P_VALUE = 0.05
_DEFAULT_P_VALUE = 0.05
_CONFIDENCE_INTERVAL_CONFIDENCE = 0.95
_BOOTSTRAPING_NUM_TRIES = 100
_HYPOTHESIS_ALTERNATIVE = "ALTERNATIVE"
_HYPOTHESIS_NULL = "NULL"


@attrs.define
class ConfidenceInterval:
    algorithm: str = attrs.field(default="NA")
    lower: float = attrs.field(default=np.NAN)
    upper: float = attrs.field(default=np.NAN)


@attrs.define
class ComputedConfidenceInterval:
    mean: float = attrs.field()
    std: float = attrs.field()
    var: float = attrs.field()
    p_value: float = attrs.field()
    confidence_intervals: list[ConfidenceInterval] = attrs.field()


def _get_final_hypothesis(
    p_value: float
) -> str:
    return (
        _HYPOTHESIS_ALTERNATIVE
        if p_value <= _MAX_P_VALUE
        else _HYPOTHESIS_NULL
    )


class StatisticalTestAlternative(Enum):
    TWO_SIDED = "two-sided"
    GREATER = "greater"
    LESS = "less"


class SignTestHandleTies(Enum):
    STRICTLY_GREATER = "STRICTLY_GREATER"
    GREATER_OR_EQUAL = "GREATER_OR_EQUAL"


def calculate_pairwise_test_statistics(
    scores_recommender_1: np.ndarray,
    scores_recommender_2: np.ndarray,
) -> Any:
    """Statistically test the pair-wise ranking score between recommender 1 and 2 on all users.

    All these tests suppose that each user is i.i.d sample, and that each recommender is an observation or variable.

    :class:`scipy.stats.wilcoxon`

    Notes
    -----
    As noted by [1]_: "However, using the Wilcoxon signed rank test still requires that differences between the two
    systems are comparable between users."

    References
    ----------
    .. [1] Gunawardana A., Shani G. (2015) Evaluating Recommender Systems. In: Ricci F., Rokach L., Shapira B. (eds)
       Recommender Systems Handbook. Springer, Boston, MA. https://doi.org/10.1007/978-1-4899-7637-6_8

    """
    assert scores_recommender_1.shape == scores_recommender_2.shape

    num_users = scores_recommender_1.shape[0]

    return dict(
        wilcoxon=_wilcoxon_statistic_test(
            scores_recommender_1=scores_recommender_1,
            scores_recommender_2=scores_recommender_2,
            zero_method="pratt",
            alternative=StatisticalTestAlternative.GREATER,
            correction=True,
        ),
        mc_nemar=_mc_nemar_test(
            scores_recommender_1=scores_recommender_1,
            scores_recommender_2=scores_recommender_2,
            num_users=num_users,
            alternative=StatisticalTestAlternative.GREATER,
            handle_ties=SignTestHandleTies.STRICTLY_GREATER,
        )
    )


def calculate_all_test_statistics(
    scores: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """

    Arguments
    ---------
    scores : np.ndarray
        A matrix of dimensions (U, K) where U is the number of users and K is the number of recommenders to
        statistically test.

    """
    num_users, num_recommenders = scores.shape

    return dict(
        friedman_chi_square=_friedman_chi_square_statistical_test(
            scores=scores,
            num_recommenders=num_recommenders,
            num_users=num_users,
        ),
    )


def _wilcoxon_statistic_test(
    scores_recommender_1: np.ndarray,
    scores_recommender_2: np.ndarray,
    alternative: StatisticalTestAlternative,
    correction: bool,
    zero_method: str,
) -> dict[str, Any]:
    """

    Notes
    -----
    alternative="two-sided":
      H0 is ranks are equal,
      Ha is ranks are different, but does not tell you which one is better.
    alternative="greater":
      H0 is ranks are better in R2,
      Ha is ranks are not better in R2.
    alternative="less":
      H0 is ranks are better in R1,
      Ha is ranks are not better in R1.
    zero_method="wilcox"
      This drops the ranks in which we have ties (score(R1, u) == score(R2, u)) for user u.
    mode=auto
      so the test can automatically determine how to calculate the p-value. Per
      default, it changes between "exact" or "approx" if the number of users is higher than 25.
    correction=False
      arbitrary, idk what this means.
    """

    available_hypothesis = {
        StatisticalTestAlternative.TWO_SIDED: {
            _HYPOTHESIS_NULL: "Ranks between recommenders are equivalent.",
            _HYPOTHESIS_ALTERNATIVE: "Ranks between recommenders are not equivalent.",
        },
        StatisticalTestAlternative.GREATER: {
            _HYPOTHESIS_NULL: "R2 ranks better than R1",
            _HYPOTHESIS_ALTERNATIVE: "R2 does not rank better than R1",
        },
        StatisticalTestAlternative.LESS: {
            _HYPOTHESIS_NULL: "R1 ranks better than R2",
            _HYPOTHESIS_ALTERNATIVE: "R1 does not rank better than R2",
        },
    }

    if np.any(
        np.isclose(
            a=scores_recommender_1,
            b=scores_recommender_2,
        )
    ):
        logger.warning(
            f"Found difference in scores close to zero. `zero_method` parameter may change the p-value and statistics."
        )

    test_statistic, p_value = stats.wilcoxon(
        x=scores_recommender_1,
        y=scores_recommender_2,
        zero_method=zero_method,
        alternative=alternative.value,
        mode="auto",
        correction=correction,
    )

    hypothesis = _get_final_hypothesis(
        p_value=p_value,
    )

    textual_hypothesis = available_hypothesis[alternative][hypothesis]

    return dict(
        test_statistic=test_statistic,
        p_value=p_value,
        p_value_reliable=True,
        hypothesis=hypothesis,
        textual_hypothesis=textual_hypothesis,
        kwargs=dict(
            zero_method=zero_method,
            alternative=alternative,
            mode="auto",
            correction=correction,
        )
    )


def _cochrans_q_statistic_test(
    scores_recommender_1: np.ndarray,
    scores_recommender_2: Optional[np.ndarray],
    num_recommenders: int,
    num_users: int,
) -> float:
    """Performs the Cochran's Q Statistical Test

    Arguments
    ---------
    scores_recommender_1
        A vector of size U or a matrix of dimensions (U, K). Where U is the number of users. K is the number of
        recommenders.

    scores_recommender_2
        An optional vector of size U. Where U is the number of users.

    num_users
        An integer, is U, the number of users.

    num_recommenders
        An integer, is K, the number of recommenders.

    Notes
    -----
    Assumptions
        This test assumes that the scores are binary [1]_ [2]_, e.g., 0 or 1. If this is not the case, then do not use
        this test.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Cochran%27s_Q_test
    .. [2] https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.cochrans_q.html?highlight=cochran

    """
    # We should get a (U, K) matrix. U is the number of users. K is the number of recommenders.
    if scores_recommender_2 is None:
        assert len(scores_recommender_1.shape) > 1

        scores = scores_recommender_1

    # We should get two (U) vectors. U is the number of users.
    else:
        assert len(scores_recommender_1.shape) == 1
        assert scores_recommender_1.shape == scores_recommender_2.shape

        scores = np.vstack(
            (scores_recommender_1, scores_recommender_2)
        ).transpose()

    assert scores.shape == (num_users, num_recommenders)
    assert len(np.unique(scores)) == 2

    cochrans_q_results = sm.stats.cochrans_q(
        x=scores,
        return_object=True,
    )
    print(cochrans_q_results, sm.stats.__name__, sm.stats.cochrans_q.__name__, str({}))

    return cochrans_q_results


def _friedman_chi_square_statistical_test(
    scores: np.ndarray,
    num_users: int,
    num_recommenders: int,
) -> Any:
    """
    Notes
    -----
    Definition
        H0 (null hypothesis)
            all recommenders are equivalent.
        Ha (alternative hypothesis):
            At least two recommenders are not equivalent

    The method :class:`statsmodels.api.stats.friedmanchisquare` requires the comparison of at least 7 recommenders
    so the p-value is reliable.

    """
    assert num_users > 10
    assert num_recommenders >= 3

    p_value_reliable = True
    if num_recommenders <= 6:
        p_value_reliable = False
        logger.warning(
            f"The method `stats.friedmanchisquare` requires the comparison of at least 7 recommenders."
        )

    # Notes on the Friedman Chi^2 statistic.
    # H0 (null hypothesis): all recommenders are equivalent.
    # Ha (alternative hypothesis): At least two recommenders are not equivalent.
    test_statistic, p_value = stats.friedmanchisquare(
        *scores
    )

    hypothesis = _get_final_hypothesis(
        p_value=p_value,
    )

    return dict(
        test_statistic=test_statistic,
        p_value=p_value,
        p_value_reliable=p_value_reliable,
        hypothesis=hypothesis,
        kwargs=dict(
            num_users=num_users,
            num_recommenders=num_recommenders,
        )
    )


def _sign_test(
    scores_recommender_1: np.ndarray,
    scores_recommender_2: np.ndarray,
    num_users: int,
    alternative: StatisticalTestAlternative,
    handle_ties: SignTestHandleTies,
) -> dict[str, Any]:
    """Computes a Sign Statistical Test between two recommenders.

    The assumption of this test is that each user in the test set is independent, therefore, the scores obtained for
    each user are i.i.d. To reject or not the null hypothesis, the Sign Test determines statistically if the scores
    given by both algorithms are significantly different or not.

    A two-sided test determines if recommenders are significantly different or not. A one-sided test determines which
    recommender is better than the other.

    if `alternative` ==  `StatisticalTestAlternative.TWO_SIDED`:
        H0 (null hypothesis): R1 1 and R2 are equivalent, i.e., their scores are equivalent.
        Ha (alternative hypothesis): Any of the two are not equivalent, i.e., one is better/worse than the other.

    if `alternative` ==  `StatisticalTestAlternative.GREATER`:
        H0 (null hypothesis): R1 is not better than R2, i.e., scores for R2 are higher than R1.
        Ha (alternative hypothesis): R1 is better than R2. i.e., scores for R1 are higher than R2.

    if `alternative` ==  `StatisticalTestAlternative.LESS`:
        H0 (null hypothesis): R2 is not better than R1, i.e., scores for R1 are higher than R2.
        Ha (alternative hypothesis): R2 is better than R1. i.e., scores for R2 are higher than R1.
    ``

    One of the fundamental components of the test is the number of successes, being a success the number of times
    the score in the recommender 1 is greater than (or greater or equal than) recommender 2. If the test measures
    "strict" improvement, i.e., recommender 1 is strictly better than recommender 2, then use
    `mode=SignTestHandleTies.STRICTLY_GREATER`, on the contrary, if the recommender can perform equally to
    recommender 2, then `mode=SignTestHandleTies.GREATER_OR_EQUAL`

    Practically:

    if `mode` ==  `SignTestHandleTies.STRICTLY_GREATER`:
        successes: Number of times `scores_recommender_1[i]` > `scores_recommender_2[i]`
    else:
        successes: Number of times `scores_recommender_1[i]` >= `scores_recommender_2[i]`

    Notes
    -----
    :class:`SignTestHandleTies.STRICTLY_GREATER` makes draws to be counted towards the null hypothesis. This makes
    *harder* for the test to reject the null hypothesis.

    On the other hand, :class:`SignTestHandleTies.GREATER_OR_EQUAL` counts draws towards the alternative hypothesis,
    thus making it *easier* for the test to reject the null hypothesis.

    Implementation extracted from: https://www.statsmodels.org/stable/_modules/statsmodels/sandbox/stats/runs.html#mcnemar

    References
    ----------
    .. [1] Gunawardana A., Shani G. (2015) Evaluating Recommender Systems. In: Ricci F., Rokach L., Shapira B. (eds)
       Recommender Systems Handbook. Springer, Boston, MA. https://doi.org/10.1007/978-1-4899-7637-6_8

    .. [2] Demsar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. J. Mach. Learn. Res., 7,
       1–30.
    """

    # Ensure these are 1-D arrays and that both arrays have the same shape.
    assert len(scores_recommender_1.shape) == 1
    assert scores_recommender_1.shape[0] == num_users
    assert scores_recommender_1.shape == scores_recommender_2.shape

    num_times_recommender_1_better_than_2 = np.sum(
        scores_recommender_1 > scores_recommender_2,
    )

    num_times_recommender_2_better_than_1 = np.sum(
        scores_recommender_1 < scores_recommender_2,
    )

    num_times_recommenders_are_equal = np.sum(
        scores_recommender_1 == scores_recommender_2,
    )

    assert (
        num_times_recommender_1_better_than_2
        + num_times_recommender_2_better_than_1
        + num_times_recommenders_are_equal
    ) == num_users

    if handle_ties == SignTestHandleTies.STRICTLY_GREATER:
        successes = num_times_recommender_1_better_than_2

    else:
        successes = num_times_recommender_1_better_than_2 + num_times_recommenders_are_equal

    binomial_test_results = stats.binomtest(
        k=successes,
        n=num_users,
        p=_MAX_P_VALUE,
        alternative=alternative.value,
    )

    stat = binomial_test_results.k
    p_value = binomial_test_results.pvalue
    proportion = binomial_test_results.proportion_estimate
    confidence_interval = binomial_test_results.proportion_ci(
        confidence_level=_CONFIDENCE_INTERVAL_CONFIDENCE,
        method="exact",
    )

    hypothesis = _get_final_hypothesis(
        p_value=p_value,
    )

    return dict(
        p_value=p_value,
        statistic=stat,
        hypothesis=hypothesis,
        proportion=proportion,
        confidence_interval=confidence_interval,
        kwargs=dict(
            alternative=alternative,
            handle_ties=handle_ties,
            num_times_recommender_1_better_than_2=num_times_recommender_1_better_than_2,
            num_times_recommender_2_better_than_1=num_times_recommender_2_better_than_1,
            num_times_recommenders_are_equal=num_times_recommenders_are_equal,
            successes=successes,
            non_successes=num_users - successes,
        )
    )


# Provide an alias for the sign_test given that the McNemar and Sign Tests are equivalent.
# https://en.wikipedia.org/wiki/McNemar%27s_test#Related_tests
_mc_nemar_test = _sign_test


def _bonferroni_correction_statistical_test(
    scores: np.ndarray,
    num_users: int,
    num_recommenders: int,
) -> Any:
    """
    Notes
    -----
    Definition
        H0 (null hypothesis)
            all recommenders are equivalent.
        Ha (alternative hypothesis):
            At least two recommenders are not equivalent

    The method :class:`statsmodels.api.stats.friedmanchisquare` requires the comparison of at least 7 recommenders
    so the p-value is reliable.

    """
    pass


def calculate_confidence_intervals_on_scores_mean(
    scores: np.ndarray,
    p_value: float = None,
) -> ComputedConfidenceInterval:
    """

    Arguments
    ---------
    scores : np.ndarray
        A vector of dimensions (U,) where U is the number of users to statistically test.

    p_value : float
        A floating point number between 0 and 1 indicating the percentage of confidence (1 - p_value) to use when
        computing the confidence intervals. Smaller values of `p_value` increase the size of the interval. Bigger
        values of `p_value` decrease the size of the interval.

    """
    assert len(scores.shape) == 1
    assert p_value is None or (0. <= p_value <= 1.)

    if p_value is None:
        p_value = _DEFAULT_P_VALUE

    scores_stats = sm.stats.DescrStatsW(scores)

    # An equivalent interval could be computed using the `scipy.stats.t.interval`,
    # However, implementation-wise using statsmodels is clearer and less error-prone
    # due to the arguments needed by the scipy's function.
    t_ci_lower, t_ci_upper = scores_stats.tconfint_mean(
        alpha=p_value,
        alternative="two-sided",
    )

    # An equivalent interval could be computed using the `scipy.stats.norm.interval`,
    # However, implementation-wise using statsmodels is clearer and less error-prone
    # due to the arguments needed by the scipy's function.
    normal_ci_lower, normal_ci_upper = scores_stats.zconfint_mean(
        alpha=p_value,
        alternative="two-sided",
    )

    return ComputedConfidenceInterval(
        mean=scores.mean(dtype=np.float64),  # type: ignore
        std=scores.std(dtype=np.float64),  # type: ignore
        var=scores.var(dtype=np.float64),  # type: ignore
        p_value=p_value,
        confidence_intervals=[
            ConfidenceInterval(
                algorithm="t-test",
                lower=t_ci_lower,
                upper=t_ci_upper,
            ),
            ConfidenceInterval(
                algorithm="normal",
                lower=normal_ci_lower,
                upper=normal_ci_upper,
            ),
        ]
    )


def _confidence_interval_bootstrapping(
    scores: np.ndarray,
    num_users: int,
    p_value: float,
) -> ConfidenceInterval:
    """Computes the confidence interval for the given score array using bootstrapping.

    Bootstrapping [1]_ calculates the mean of samples with replacement of the scores array several
    times. The interval is then defined as the (2.5, 97.5) percentiles of these means.

    References
    .. [1] https://www.yourdatateacher.com/2021/11/08/how-to-calculate-confidence-intervals-in-python/
    """
    assert scores.shape == (num_users,)

    rng = np.random.default_rng()

    bootstrapped_means: list[np.float64] = [
        rng.choice(
            scores,
            size=num_users,
            replace=True,
        ).mean(
            dtype=np.float64,
        )
        for _ in range(_BOOTSTRAPING_NUM_TRIES)
    ]

    percentiles_to_compute = [
        100 * ((1 - p_value) / 2),
        100 * (p_value / 2),
    ]

    lower_bound, upper_bound = np.percentile(
        a=bootstrapped_means,
        q=percentiles_to_compute,
    )

    return ConfidenceInterval(
        algorithm="bootstrapping",
        lower=lower_bound,
        upper=upper_bound,
    )
