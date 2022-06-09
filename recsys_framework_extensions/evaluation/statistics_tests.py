from enum import Enum
from typing import Optional, Sequence, Iterable, Mapping

import attrs
import numpy as np
import statsmodels.api as sm
from scipy import stats

from recsys_framework_extensions.logging import get_logger

logger = get_logger(
    logger_name=__name__,
)


_MAX_P_VALUE = 0.05
_DEFAULT_ALPHA = 0.05
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
    alpha: float = attrs.field()
    confidence_intervals: list[ConfidenceInterval] = attrs.field()


@attrs.define
class ResultsStatisticalTests:
    mean: float = attrs.field()
    std: float = attrs.field()
    var: float = attrs.field()
    alpha: float = attrs.field()
    confidence_intervals: list[ConfidenceInterval] = attrs.field()


class StatisticalTestAlternative(Enum):
    TWO_SIDED = "two-sided"
    GREATER = "greater"
    LESS = "less"


class StatisticalTestHypothesis(Enum):
    NULL = "NULL"
    ALTERNATIVE = "ALTERNATIVE"


class SignTestHandleTies(Enum):
    STRICTLY_GREATER = "STRICTLY_GREATER"
    GREATER_OR_EQUAL = "GREATER_OR_EQUAL"


@attrs.define
class ResultsFriedmanTest:
    test_statistic: float = attrs.field()
    num_measurements: int = attrs.field()
    p_value: float = attrs.field()
    p_value_reliable: float = attrs.field()
    hypothesis: Sequence[StatisticalTestHypothesis] = attrs.field()
    kwargs: dict = attrs.field()


@attrs.define
class ResultsWilcoxonTest:
    test_statistic: float = attrs.field()
    p_value: float = attrs.field()
    hypothesis: Sequence[StatisticalTestHypothesis] = attrs.field()
    kwargs: dict = attrs.field()


@attrs.define
class ResultsSignTest:
    p_value: float = attrs.field()
    hypothesis: Sequence[StatisticalTestHypothesis] = attrs.field()
    successes: int = attrs.field()
    failures: int = attrs.field()
    num_times_base_better_than_other: int = attrs.field()
    num_times_other_better_than_base: int = attrs.field()
    num_times_both_are_equal: int = attrs.field()
    kwargs: dict = attrs.field()


@attrs.define
class ResultsBonferroniCorrection:
    corrected_p_values: np.ndarray = attrs.field()
    corrected_alphas: np.ndarray = attrs.field()
    hypothesis: Sequence[Sequence[StatisticalTestHypothesis]] = attrs.field()
    kwargs: dict = attrs.field()


@attrs.define
class ResultsStatisticalTestsBaseVsOthers:
    friedman: ResultsFriedmanTest = attrs.field()
    bonferroni_wilcoxon: ResultsBonferroniCorrection = attrs.field()
    bonferroni_sign: ResultsBonferroniCorrection = attrs.field()
    wilcoxon: Sequence[ResultsWilcoxonTest] = attrs.field()
    sign: Sequence[ResultsSignTest] = attrs.field()


def _get_final_hypothesis(
    p_value: float,
    alphas: Iterable[float],
) -> Sequence[StatisticalTestHypothesis]:
    return [
        StatisticalTestHypothesis.ALTERNATIVE
        if p_value <= alpha
        else StatisticalTestHypothesis.NULL

        for alpha in alphas
    ]


def compute_statistical_tests_of_base_vs_others(
    scores_base: np.ndarray,
    scores_others: np.ndarray,
    alphas: Sequence[float],
    alternative: StatisticalTestAlternative,
) -> ResultsStatisticalTestsBaseVsOthers:
    """

    Arguments
    ---------
    alternative : StatisticalTestAlternative
        TODO: fernando-debugger|complete
    scores_base : np.ndarray
        A matrix of dimensions (1, U) where U is the number of users.
    scores_others : np.ndarray
        A matrix of dimensions (K, U) where U is the number of users and K is the number of recommenders to
        statistically test.
    alphas : Sequence[float]
        TODO: fernando-debugger|complete
    """
    assert scores_base.ndim == 1
    assert scores_others.ndim == 2

    num_users_base = scores_base.shape[0]

    num_other_recommenders = scores_others.shape[0]
    num_users_others = scores_others.shape[1]

    assert num_users_base == num_users_others
    assert num_other_recommenders >= 1

    scores = np.vstack(
        (scores_base, scores_others),
    )

    results_friedman = _friedman_chi_square_statistical_test(
        scores=scores,
        alphas=alphas,
    )

    results_wilcoxon = [
        _wilcoxon_statistic_test(
            scores_base=scores_base,
            scores_other=scores_others[idx_recommender, :],
            alphas=alphas,
            alternative=alternative,
        )
        for idx_recommender in range(num_other_recommenders)
    ]
    results_sign = [
        _sign_test(
            scores_base=scores_base,
            scores_other=scores_others[idx_recommender, :],
            alphas=alphas,
            alternative=alternative,
        )
        for idx_recommender in range(num_other_recommenders)
    ]

    p_values_wilcoxon = np.asarray(
        [
            res.p_value
            for res in results_wilcoxon
        ],
        dtype=np.float64,
    )
    p_values_sign = np.asarray(
        [
            res.p_value
            for res in results_sign
        ],
        dtype=np.float64,
    )

    results_bonferroni_wilcoxon = _bonferroni_correction(
        p_values=p_values_wilcoxon,
        alphas=alphas,
    )
    results_bonferroni_sign = _bonferroni_correction(
        p_values=p_values_sign,
        alphas=alphas,
    )

    return ResultsStatisticalTestsBaseVsOthers(
        friedman=results_friedman,
        wilcoxon=results_wilcoxon,
        sign=results_sign,
        bonferroni_wilcoxon=results_bonferroni_wilcoxon,
        bonferroni_sign=results_bonferroni_sign,
    )


def _wilcoxon_statistic_test(
    scores_base: np.ndarray,
    scores_other: np.ndarray,
    alphas: Sequence[float],
    alternative: StatisticalTestAlternative,
) -> ResultsWilcoxonTest:
    """

    Notes
    -----
    alternative="two-sided":
      H0 is ranks are equal,
      Ha is ranks are different, but does not tell you which one is better.
    alternative="greater":
      H0 is ranks are better in RO,
      Ha is ranks are not better in RO.
    alternative="less":
      H0 is ranks are better in RB,
      Ha is ranks are not better in RB.
    zero_method="wilcox"
      This drops the ranks in which we have ties (score(RB, u) == score(RO, u)) for user u.
    mode=auto
      so the test can automatically determine how to calculate the p-value. Per
      default, it changes between "exact" or "approx" if the number of users is higher than 25.
    correction=False
      arbitrary, idk what this means.
    """

    logger.debug(
        f"Wilcoxon Test received the following parameters:"
        f"\n\t* {scores_base=} - {scores_base.ndim=} - {scores_base.shape=}"
        f"\n\t* {scores_other=} - {scores_other.ndim=} - {scores_other.shape=}"
        f"\n\t* {alphas=}"
        f"\n\t* {alternative=}"
    )

    # Definition of Wilcoxon Signed-Rank Test extracted from:
    # https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf, section 3.1.3
    # Zero method is set as `zsplit` as in the paper
    # The mode computes the p-value in an analytical or probabilistic way.
    # The correction is not included as the paper does not correct.
    # In a one-side greater test,
    test_statistic, p_value = stats.wilcoxon(
        x=scores_other,
        y=scores_base,
        alternative=alternative.value,
        zero_method="zsplit",
        mode="auto",
        correction=False,
    )

    hypothesis = _get_final_hypothesis(
        p_value=p_value,
        alphas=alphas,
    )

    logger.debug(
        f"Wilcoxon Test Debug purposes"
        f"\n\t* {hypothesis=}"
        f"\n\t* {test_statistic=}"
        f"\n\t* {p_value=}"
        f"\n\t* {alphas=}"
    )

    return ResultsWilcoxonTest(
        test_statistic=test_statistic,
        p_value=p_value,
        hypothesis=hypothesis,
        kwargs=dict(
            alternative=alternative,
            alphas=alphas,
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
    alphas: Sequence[float],
) -> ResultsFriedmanTest:
    """
    Notes
    -----
    Definition
        H0 (null hypothesis)
            The scores of recommenders are equivalent.
        Ha (alternative hypothesis):
            The scores across recommenders are different

    This method requires the comparison of at least 7 recommenders and 10 users so the p-value is reliable.
    """
    assert scores.ndim == 2
    num_recommenders, num_users = scores.shape

    assert num_recommenders >= 3

    p_value_reliable = True
    if num_recommenders <= 6 or num_users <= 10:
        p_value_reliable = False
        logger.warning(
            f"The method `stats.friedmanchisquare` requires the comparison of at least 7 recommenders "
            f"(got {num_recommenders}) and at least 10 users (got {num_users}). "
        )

    # Implementation note: the function `stats.friedmanchisquare` requires each array of recommender scores (vector
    # of size 1xN, where N is the number of users), to be passed as an argument.
    # Passing a numpy array as *array sends the function each row as a separate argument. Hence, in this case,
    # we are complying with what the function expects, as the `scores` array is a MxN array, where M is the number of
    # recommenders. *scores passes each row (each recommender score) as an argument.
    test_statistic, p_value = stats.friedmanchisquare(
        *scores
    )

    hypothesis = _get_final_hypothesis(
        p_value=p_value,
        alphas=alphas,
    )

    return ResultsFriedmanTest(
        test_statistic=test_statistic,
        num_measurements=num_recommenders,
        p_value=p_value,
        p_value_reliable=p_value_reliable,
        hypothesis=hypothesis,
        kwargs=dict(
            alphas=alphas,
        )
    )


def _sign_test(
    scores_base: np.ndarray,
    scores_other: np.ndarray,
    alphas: Sequence[float],
    alternative: StatisticalTestAlternative,
) -> ResultsSignTest:
    """Computes a Sign Statistical Test between the scores of two recommenders.

    The assumption of this test is that each user in the test set is independent, therefore, the scores obtained for
    each user are i.i.d. To reject or not the null hypothesis, the Sign Test determines statistically if the scores
    given by both algorithms are significantly different or not.

    A two-sided test determines if recommenders are significantly different or not. A one-sided test determines which
    recommender is better than the other.

    if `alternative` ==  `StatisticalTestAlternative.TWO_SIDED`:
        H0 (null hypothesis): RB and RO are equivalent, i.e., their scores are equivalent.
        Ha (alternative hypothesis): Any of the two are not equivalent, i.e., one is better/worse than the other.

    if `alternative` ==  `StatisticalTestAlternative.GREATER`:
        H0 (null hypothesis): RO is not better than RB, i.e., scores for RB are higher than RO.
        Ha (alternative hypothesis): RO is better than RB. i.e., scores for RO are higher than RB.

    if `alternative` ==  `StatisticalTestAlternative.LESS`:
        H0 (null hypothesis): RB is not better than RO, i.e., scores for RO are higher than RB.
        Ha (alternative hypothesis): RB is better than RO. i.e., scores for RB are higher than RO.
    ``

    One of the fundamental components of the test is the number of successes, being a success the number of times
    the score in RO is greater than (or greater or equal than) RB. If the test measures
    "strict" improvement, i.e., RO is strictly better than RB, then use
    `mode=SignTestHandleTies.STRICTLY_GREATER`, on the contrary, if RO may perform equally to
    RB, then `mode=SignTestHandleTies.GREATER_OR_EQUAL`

    Practically:

    if `mode` ==  `SignTestHandleTies.STRICTLY_GREATER`:
        successes: Number of times `scores_other[i]` > `scores_base[i]`
    else:
        successes: Number of times `scores_other[i]` >= `scores_base[i]`

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
    assert scores_base.ndim == 1
    assert scores_other.ndim == 1
    assert scores_base.shape == scores_other.shape

    logger.debug(
        f"Sign Test received the following parameters:"
        f"\n\t* {scores_base=} - {scores_base.ndim=} - {scores_base.shape}"
        f"\n\t* {scores_other=} - {scores_other.ndim=} - {scores_other.shape}"
        f"\n\t* {alphas=}"
        f"\n\t* {alternative=}"
    )

    available_hypothesis = {
        StatisticalTestAlternative.TWO_SIDED: {
            StatisticalTestHypothesis.NULL: (
                "H0 (null hypothesis): RB and RO are equivalent, i.e., their scores are equivalent."
            ),
            StatisticalTestHypothesis.ALTERNATIVE: (
                "Ha (alternative hypothesis): Any of the two are not equivalent, i.e., one is better/worse than the "
                "other."
            ),
        },
        StatisticalTestAlternative.GREATER: {
            StatisticalTestHypothesis.NULL: (
                "H0 (null hypothesis): RO is not better than RB, i.e., scores for RB are higher than RO."
            ),
            StatisticalTestHypothesis.ALTERNATIVE: (
                "Ha (alternative hypothesis): RO is better than RB. i.e., scores for RO are higher than RB."
            ),
        },
        StatisticalTestAlternative.LESS: {
            StatisticalTestHypothesis.NULL: (
                "H0 (null hypothesis): RB is not better than RO, i.e., scores for RO are higher than RB."
            ),
            StatisticalTestHypothesis.ALTERNATIVE: (
                "Ha (alternative hypothesis): RB is better than RO. i.e., scores for RB are higher than RO."
            ),
        },
    }

    num_users = scores_base.shape[0]

    num_times_base_better_than_other = np.sum(
        scores_base > scores_other,
    )

    num_times_other_better_than_base = np.sum(
        scores_base < scores_other,
    )

    num_times_both_are_equal = np.sum(
        scores_base == scores_other,
    )

    successes = num_times_other_better_than_base + num_times_both_are_equal // 2
    failures = num_times_base_better_than_other + num_times_both_are_equal // 2

    assert (
        num_times_both_are_equal % 2 == 0 and successes + failures == num_users
    ) or (
        num_times_both_are_equal % 2 == 1 and successes + failures + 1 == num_users
    )
    assert (
        num_times_base_better_than_other + num_times_other_better_than_base + num_times_both_are_equal
    ) == num_users

    # From v1.8.1 we can use binomtest that returns a set of more comprehensive results.
    # if scipy version is lower, then we have to use binom_test which only returns the computed alpha

    # Sign test as defined by: https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf
    # In particular, both recommenders are considered to win in 50% of the cases.
    # Ties are not discounted but added between the successes and failures, if the num of ties is odd,
    # then 1 is substracted.
    p_value = stats.binom_test(
        x=[successes, failures],
        n=num_users,
        p=0.5,
        alternative=alternative.value,
    )

    hypothesis = _get_final_hypothesis(
        p_value=p_value,
        alphas=alphas,
    )

    return ResultsSignTest(
        p_value=p_value,
        hypothesis=hypothesis,
        successes=successes,
        failures=failures,
        num_times_base_better_than_other=num_times_base_better_than_other,
        num_times_other_better_than_base=num_times_other_better_than_base,
        num_times_both_are_equal=num_times_both_are_equal,
        kwargs=dict(
            alphas=alphas,
            alternative=alternative,
        )
    )


# Provide an alias for the sign given that the McNemar and Sign Tests are equivalent.
# https://en.wikipedia.org/wiki/McNemar%27s_test#Related_tests
_mc_nemar_test = _sign_test


def _bonferroni_correction(
    p_values: np.ndarray,
    alphas: Sequence[float],
) -> ResultsBonferroniCorrection:
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
    assert p_values.ndim == 1

    num_experiments = float(len(p_values))
    corrected_alphas: np.ndarray = np.asarray(alphas).astype(dtype=np.float64) / num_experiments
    corrected_p_values: np.ndarray = np.asarray(p_values).astype(dtype=np.float64) * num_experiments
    corrected_p_values[corrected_p_values > 1.] = 1.

    hypothesis = [
        _get_final_hypothesis(
            p_value=p_value,
            alphas=corrected_alphas,
        )
        for p_value in corrected_p_values
    ]

    return ResultsBonferroniCorrection(
        corrected_p_values=corrected_p_values,
        corrected_alphas=corrected_alphas,
        hypothesis=hypothesis,
        kwargs=dict(
            p_values=p_values,
            alphas=alphas,
        )
    )


def calculate_confidence_intervals_on_scores_mean(
    scores: np.ndarray,
    alpha: float = None,
) -> ComputedConfidenceInterval:
    """

    Arguments
    ---------
    scores : np.ndarray
        A vector of dimensions (U,) where U is the number of users to statistically test.

    alpha : float
        A floating point number between 0 and 1 indicating the percentage of confidence (1 - alpha) to use when
        computing the confidence intervals. Smaller values of `alpha` increase the size of the interval. Bigger
        values of `alpha` decrease the size of the interval.

    """
    assert scores.ndim == 1
    assert alpha is None or (0. <= alpha <= 1.)

    if alpha is None:
        alpha = _DEFAULT_ALPHA

    scores_stats = sm.stats.DescrStatsW(scores)

    # An equivalent interval could be computed using the `scipy.stats.t.interval`,
    # However, implementation-wise using statsmodels is clearer and less error-prone
    # due to the arguments needed by the scipy's function.
    t_ci_lower, t_ci_upper = scores_stats.tconfint_mean(
        alpha=alpha,
        alternative="two-sided",
    )

    # An equivalent interval could be computed using the `scipy.stats.norm.interval`,
    # However, implementation-wise using statsmodels is clearer and less error-prone
    # due to the arguments needed by the scipy's function.
    normal_ci_lower, normal_ci_upper = scores_stats.zconfint_mean(
        alpha=alpha,
        alternative="two-sided",
    )

    return ComputedConfidenceInterval(
        mean=scores.mean(dtype=np.float64),  # type: ignore
        std=scores.std(dtype=np.float64),  # type: ignore
        var=scores.var(dtype=np.float64),  # type: ignore
        alpha=alpha,
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
