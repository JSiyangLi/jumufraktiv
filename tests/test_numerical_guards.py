"""Domain guards, and the three public methods that could not be called.

`post_quantile`, `post_interval` and `post_sample` failed for **every** prior,
and `post_cdf` answered questions below its support with a probability greater
than one or with `RecursionError`. None of that is exotic input: a quantile is
the most ordinary thing to ask a posterior for.

The repairs here are all guards and bracketing. None touches the quadrature
kernel, so no number that was previously correct moves.
"""

import numpy as np
import pytest
from conftest import ALPHA, BETA, POISSON_DATA, POISSON_SCALE

#: The canonical test problem is conjugate, so the posterior is exactly
#: Gamma(ALPHA + sum(y), BETA + n*scale) and scipy gives its quantiles.
POST_SHAPE = ALPHA + sum(POISSON_DATA)
POST_RATE = BETA + len(POISSON_DATA) * POISSON_SCALE


@pytest.fixture
def exact():
    """The conjugate posterior as a `scipy.stats` frozen distribution."""
    from scipy.stats import gamma as gamma_dist

    return gamma_dist(a=POST_SHAPE, scale=1.0 / POST_RATE)


# ==========================================================================
# post_cdf below its support
# ==========================================================================
@pytest.mark.parametrize("u", [-1.0, -0.5, -1e-9, 0.0])
def test_post_cdf_is_zero_below_the_support(poisson_posterior, u):
    """`theta` is strictly positive, so F(u) = 0 for u <= 0.

    This is a legitimate question with a known answer, not a caller error, so
    it is answered rather than refused. Before, `u = -0.5` returned a log-CDF
    of +0.89 -- a probability of 2.43 -- and `u = -1e-9` recursed until
    `RecursionError`.
    """
    assert poisson_posterior.post_cdf(u, log=False) == pytest.approx(0.0, abs=1e-12)
    assert poisson_posterior.post_cdf(u, log=True) == -np.inf


def test_post_cdf_mixes_in_and_out_of_support_in_one_array(poisson_posterior, exact):
    """The guard must be per-element, not all-or-nothing.

    The batch path evaluates one array in a single call, so a single
    out-of-support entry must not disturb its neighbours -- and must not cause
    the whole array to be treated as out of support either.
    """
    points = np.array([-0.5, 0.0, 0.5, 1.0, 2.0])

    got = poisson_posterior.post_cdf(points, log=False)

    expected = np.where(points > 0, exact.cdf(np.maximum(points, 1e-300)), 0.0)
    assert got == pytest.approx(expected, rel=1e-10, abs=1e-15)


def test_the_guard_does_not_disturb_values_inside_the_support(poisson_posterior, exact):
    """Adding a domain check must not move any number that was already right."""
    for u in (0.5, 1.0, 2.0, 5.0):
        assert poisson_posterior.post_cdf(u, log=False) == pytest.approx(
            exact.cdf(u), rel=1e-10
        )


# ==========================================================================
# post_density below its support
# ==========================================================================
@pytest.mark.parametrize("theta", [-1.0, 0.0])
def test_post_density_is_zero_below_the_support(poisson_posterior, theta):
    """The density vanishes off the support rather than producing NaN.

    `post_quantile` evaluates this as the derivative for its Newton step and
    can propose a non-positive point. The old code took `log(theta)` there and
    emitted "invalid value encountered in log", which the suite's
    `filterwarnings = ["error"]` escalates to an exception while ordinary users
    see only a warning and a NaN -- the harness-versus-user divergence that
    CLAUDE.md records as a recurring hazard.
    """
    assert poisson_posterior.post_density(theta, log=False) == pytest.approx(0.0)
    assert poisson_posterior.post_density(theta, log=True) == -np.inf


# ==========================================================================
# post_quantile, post_interval, post_sample
# ==========================================================================
@pytest.mark.parametrize("p", [0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999])
def test_post_quantile_matches_the_conjugate_quantile_function(
    poisson_posterior, exact, p
):
    """Against the exact quantile function, not against a recorded number.

    Every one of these raised before. The bracket ran from 1e-6, and the
    incomplete-MGF derivative is not merely small there but *wrong*: measured
    on this posterior its log comes out as -65.17 where the true value is
    -110.41, with the computed sign flipped negative, which trips the
    non-negativity guard in `post_cdf`. So the failure happened during
    bracketing, before the root finder ran at all.

    The bracket now starts from the posterior's own scale, `(a+1)/b`, which is
    the right order of magnitude for these quantiles and sits well inside the
    region where the CDF is trustworthy.
    """
    assert poisson_posterior.post_quantile(p) == pytest.approx(exact.ppf(p), rel=1e-5)


def test_post_quantile_inverts_the_cdf(poisson_posterior):
    """The round trip is the property that matters, independent of any oracle."""
    for p in (0.025, 0.5, 0.975):
        q = poisson_posterior.post_quantile(p)

        assert poisson_posterior.post_cdf(q, log=False) == pytest.approx(p, abs=1e-6)


def test_post_quantile_agrees_scalar_and_array(poisson_posterior):
    """A probability must give the same answer alone and in a batch.

    Not a style point. `solve_root`'s "auto" mode tries methods in order and
    accepts the first that does not raise, but a diverged Newton step returns a
    non-finite number rather than raising, so it counted as success. That is
    how `post_quantile(0.025)` returned 7.7e+300 while the same probability
    inside an array returned 0.5756. `post_quantile` now verifies its root lies
    in the bracket and falls back to bisection, which cannot fail given a
    bracket and a monotone CDF.

    **The tolerance here is 1e-6, not 1e-8, and that is deliberate.** A small
    residual difference survives and is *not* a defect: the two calls get
    slightly different brackets, because expanding a bracket for three
    probabilities at once widens it more than for one, and the solver then
    stops at different points inside the same tolerance. It tracks `tol`
    exactly, which is what identifies it as convergence rather than error:

    ======  ==================  =================
    `tol`   scalar-vs-array     scalar vs exact
    ======  ==================  =================
    1e-8    4.85e-08            4.83e-08
    1e-10   1.16e-10            3.28e-10
    1e-12   5.16e-11            5.16e-11
    1e-14   9.14e-14            8.79e-14
    ======  ==================  =================

    So both answers sit within the tolerance the caller asked for, and
    tightening `tol` tightens the agreement proportionally. Asserting 1e-8
    equality would be asserting that a 1e-8-tolerance solver is exact.
    """
    probabilities = [0.025, 0.5, 0.975]

    one_at_a_time = np.array(
        [poisson_posterior.post_quantile(p) for p in probabilities]
    )
    batched = np.asarray(poisson_posterior.post_quantile(np.array(probabilities)))

    assert one_at_a_time == pytest.approx(batched, rel=1e-6)


def test_post_interval_brackets_the_requested_mass(poisson_posterior, exact):
    """A 95% interval must contain 95% of the posterior."""
    low, high = poisson_posterior.post_interval(0.95)

    assert low < high
    assert low == pytest.approx(exact.ppf(0.025), rel=1e-5)
    assert high == pytest.approx(exact.ppf(0.975), rel=1e-5)


def test_post_sample_returns_the_requested_size_and_support(poisson_posterior):
    """Samples must be the right count and inside the support.

    Reproducibility is a separate defect, owned by the public-API PR, and is
    still recorded as an expected failure.
    """
    sample = np.asarray(poisson_posterior.post_sample(16))

    assert sample.shape == (16,)
    assert np.all(np.isfinite(sample))
    assert np.all(sample > 0)


def test_post_quantile_rejects_probabilities_outside_the_unit_interval(
    poisson_posterior,
):
    """0 and 1 have no finite quantile here, so they are refused, not guessed."""
    for p in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="between 0 and 1"):
            poisson_posterior.post_quantile(p)


# ==========================================================================
# logminus
# ==========================================================================
@pytest.mark.parametrize("gap", [1e-1, 1e-3, 1e-8, 1e-10, 1e-15, 1e-17])
def test_logminus_is_accurate_at_small_gaps(gap):
    """`log(exp(x) - exp(y))` must survive `x` and `y` being close.

    The function implemented only the `log1p` branch of `log1mexp`, which loses
    the difference once `exp(-d)` rounds to 1. Measured against mpmath at 50
    digits, the absolute error was 1.1e-09 at a gap of 1e-8, 8.3e-08 at 1e-10
    and 8.0e-04 at 1e-15, and at 1e-17 the answer was `-inf` -- with a "divide
    by zero encountered in log1p" warning -- for a quantity that is finite.

    The small-gap regime is the operating regime: this forms differences of
    incomplete MGFs, whose arguments are close by construction.
    """
    mp = pytest.importorskip("mpmath")
    from jumufraktiv.logsum import logminus

    mp.mp.dps = 60
    exact = float(mp.log(mp.e ** mp.mpf(0.0) - mp.e ** mp.mpf(-gap)))

    assert logminus(0.0, -gap) == pytest.approx(exact, rel=1e-14)


def test_logminus_keeps_its_scalar_and_array_contract():
    """A scalar in gives a float out; arrays broadcast and keep their shape."""
    from jumufraktiv.logsum import logminus

    assert isinstance(logminus(0.0, -1.0), float)
    assert np.shape(logminus(np.zeros(3), -1.0)) == (3,)
    assert logminus(0.0, 0.0) == -np.inf
    assert logminus(0.0, 1.0) == -np.inf
