"""The incomplete MGF, and the four methods that cannot work without one.

``post_cdf`` needs ``M(t, u) = int e^{t x} p(x) dx`` truncated at ``u``, and
``post_quantile``, ``post_interval`` and ``post_sample`` are all built on
``post_cdf``. A prior that does not supply one therefore loses four of the
package's public methods at once, on every backend.

The references here are written out by hand and integrated with mpmath. They are
never the package's own answer: for the uniform prior the truncated integral is
elementary, and for the Heaviside prior it is elementary too, so both could be
checked symbolically -- but quadrature on a hand-written integrand is the check
that would survive the closed form being wrong.
"""

import math

import mpmath as mp
import numpy as np
import pytest

from jumufraktiv import registry
from jumufraktiv.MGFDerivative_class import MGFDerivative
from jumufraktiv.MGFPrior_class import MGFPrior

#: Every registry prior, with hyperparameters and the support its density lives
#: on. The support matters: outside it the truncated integral captures no new
#: mass, and the CDF must be flat rather than merely close.
PRIORS = {
    "gamma": (
        {"alpha": 2.0, "beta": 3.0},
        lambda x: 3.0**2 * x ** (2 - 1) * mp.e ** (-3 * x) / mp.gamma(2),
        (mp.mpf(0), mp.inf),
    ),
    "uniform": (
        {"a": 0.5, "b": 2.0},
        lambda x: mp.mpf(1) / mp.mpf("1.5"),
        (mp.mpf("0.5"), mp.mpf(2)),
    ),
    "pareto": (
        {"alpha": 2.0, "xi": 1.0},
        lambda x: mp.mpf(2) * mp.mpf(1) ** 2 / x**3,
        (mp.mpf(1), mp.inf),
    ),
    "heaviside": (
        {"k": 0.5},
        lambda x: mp.mpf(1),
        (mp.mpf("0.5"), mp.inf),
    ),
}

#: Poisson data [1, 2, 3] at scale 1 gives a = sum(y) = 6 and b = sum(s) = 3, so
#: the posterior is proportional to `theta**6 * exp(-3*theta) * prior(theta)`.
DATA, SCALE, ORDER, RATE = [1, 2, 3], 1.0, mp.mpf(6), mp.mpf(3)


@pytest.fixture(autouse=True)
def _dps():
    previous = mp.mp.dps
    mp.mp.dps = 40
    yield
    mp.mp.dps = previous


def _exact_imgf(density, low, t_value, u_value):
    """``int_low^min(u, high) e^{t x} p(x) dx``, by quadrature."""
    if u_value <= float(low):
        return mp.mpf(0)
    return mp.quad(
        lambda x: mp.e ** (mp.mpf(t_value) * x) * density(x),
        [low, mp.mpf(u_value)],
    )


def _exact_posterior_cdf(density, low, high, u_value):
    normaliser = mp.quad(
        lambda x: x**ORDER * mp.e ** (-RATE * x) * density(x), [low, high]
    )
    if u_value <= float(low):
        return mp.mpf(0)
    upper = min(mp.mpf(u_value), high)
    return (
        mp.quad(lambda x: x**ORDER * mp.e ** (-RATE * x) * density(x), [low, upper])
        / normaliser
    )


@pytest.mark.parametrize("name", sorted(PRIORS))
def test_every_registry_prior_supplies_an_incomplete_mgf(name):
    """Two of the four did not, and it was recorded as a fact about the prior.

    "Prior does not support incomplete MGF" reads as a statement about the
    mathematics. For uniform and Heaviside it was a statement about the module:
    both truncated integrals are elementary.
    """
    registry.initialize()
    prior = MGFPrior.from_registry(name, params=PRIORS[name][0])

    assert prior.has_iMGF()


@pytest.mark.parametrize("name", sorted(PRIORS))
@pytest.mark.parametrize("t_value", [-50.0, -3.0, -0.5, -1e-3])
def test_the_incomplete_mgf_matches_quadrature(name, t_value):
    """Against the integral written out here, not against the package."""
    registry.initialize()
    params, density, (low, high) = PRIORS[name]
    prior = MGFPrior.from_registry(name, params=params)

    for u_value in [float(low) - 0.1, float(low), float(low) + 0.25, 1.5, 3.0]:
        if u_value <= 0:
            continue
        expected = _exact_imgf(density, low, t_value, min(u_value, float(high)))
        got = float(prior.imgf(t_value, u_value))

        if expected == 0:
            assert got == pytest.approx(0.0, abs=1e-12)
        else:
            assert got == pytest.approx(float(expected), rel=1e-10)


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(PRIORS))
@pytest.mark.parametrize("method", ["auto", "symbolic", "bell"])
def test_the_posterior_cdf_is_available_and_right_for_every_prior(name, method):
    """The four methods below all route through this one.

    Parametrised over the backend because a prior can supply an incomplete MGF
    that one backend cannot consume, and that is invisible from the default.
    """
    registry.initialize()
    params, density, (low, high) = PRIORS[name]
    prior = MGFPrior.from_registry(name, params=params)
    post = MGFDerivative(
        prior, data=DATA, likelihood="poisson", scale=SCALE, method=method
    )

    for u_value in [float(low) + 0.2, float(low) + 0.6, float(low) + 1.2]:
        expected = float(_exact_posterior_cdf(density, low, high, u_value))
        got = float(post.post_cdf(u_value, log=False))

        assert got == pytest.approx(expected, rel=1e-8), (
            f"{name}/{method} CDF at u={u_value}"
        )


@pytest.mark.slow
@pytest.mark.parametrize("name", ["uniform", "heaviside"])
def test_the_quantile_methods_work_for_the_newly_supported_priors(name):
    """Inverting a CDF that did not exist is what these could not do.

    The reference inverts the exact CDF by bisection rather than by a root
    finder with its own tolerance, so the comparison is not limited by the
    oracle.
    """
    registry.initialize()
    params, density, (low, high) = PRIORS[name]
    prior = MGFPrior.from_registry(name, params=params)
    post = MGFDerivative(prior, data=DATA, likelihood="poisson", scale=SCALE)

    def exact_quantile(probability):
        lower, upper = low, min(high, mp.mpf(30))
        for _ in range(120):
            middle = (lower + upper) / 2
            if _exact_posterior_cdf(density, low, high, float(middle)) < probability:
                lower = middle
            else:
                upper = middle
        return float((lower + upper) / 2)

    for probability in [0.05, 0.5, 0.95]:
        assert post.post_quantile(probability) == pytest.approx(
            exact_quantile(probability), rel=1e-6
        )

    lower, upper = post.post_interval(level=0.9)
    assert float(lower) == pytest.approx(exact_quantile(0.05), rel=1e-6)
    assert float(upper) == pytest.approx(exact_quantile(0.95), rel=1e-6)

    draws = post.post_sample(n=20, rng=np.random.default_rng(0))
    assert np.all(draws >= float(low))
    assert np.all(draws <= float(high))


def test_the_uniform_incomplete_mgf_is_flat_above_its_support():
    """Past `b` there is no more mass, so the value must stop changing.

    A `clip` that is missing or applied to the wrong end shows up here and
    nowhere else: inside the support the wrong expression is still increasing,
    so it looks plausible.
    """
    registry.initialize()
    prior = MGFPrior.from_registry("uniform", params={"a": 0.5, "b": 2.0})

    at_upper = float(prior.imgf(-1.0, 2.0))
    assert float(prior.imgf(-1.0, 5.0)) == pytest.approx(at_upper, rel=1e-14)
    assert float(prior.imgf(-1.0, 1e6)) == pytest.approx(at_upper, rel=1e-14)

    # Below the support it captures nothing, and the log of that is -inf rather
    # than a very negative number that happens to look like one.
    assert float(prior.imgf(-1.0, 0.4)) == 0.0
    assert prior.logimgf(-1.0, 0.4) == -math.inf


def test_the_heaviside_incomplete_mgf_refuses_a_non_negative_t():
    """The prior is improper, so its MGF does not exist at or above the origin.

    Returning a number there would make the posterior CDF a ratio of two
    divergent integrals, which is the kind of answer that comes back looking
    fine.
    """
    registry.initialize()
    prior = MGFPrior.from_registry("heaviside", params={"k": 0.5})

    with pytest.raises(ValueError, match="only for t < 0"):
        prior.logimgf(0.0, 1.0)

    with pytest.raises(ValueError, match="only for t < 0"):
        prior.logimgf(1.0, 1.0)


@pytest.mark.parametrize("name", ["uniform", "heaviside"])
def test_the_new_incomplete_mgfs_broadcast(name):
    """The tuple-vectorisation principle applies to `(t, u)` here as elsewhere."""
    registry.initialize()
    prior = MGFPrior.from_registry(name, params=PRIORS[name][0])

    t_values = np.array([-2.0, -1.0, -0.5])
    u_values = np.array([1.0, 1.5, 1.9])

    batched = np.asarray(prior.logimgf(t_values, u_values))
    assert batched.shape == (3,)

    one_at_a_time = [
        float(prior.logimgf(float(t), float(u)))
        for t, u in zip(t_values, u_values, strict=True)
    ]
    assert batched == pytest.approx(one_at_a_time, rel=1e-14)


# ==========================================================================
# The tuple-vectorisation principle
# ==========================================================================
@pytest.mark.parametrize(
    ("module_name", "function_name", "args"),
    [
        ("uniformMGF", "uniform_logimgf", (0.5, 2.0)),
        ("heavisideMGF", "heaviside_logimgf", (0.5,)),
    ],
)
def test_the_incomplete_mgf_is_batched_not_looped(
    monkeypatch, module_name, function_name, args
):
    """Cost, not only shape. Counted, not timed.

    Both of these were written with a Python loop over `np.argwhere`, which
    returns correctly shaped answers while doing per-element work in the
    innermost function of the package -- the shape half of the principle
    without the cost half, which is the failure PR 9 was written against and
    which no value-based test can see.

    `logminus` broadcasts, so a batched implementation calls it a fixed number
    of times whatever the array length. A wall-clock assertion would vary with
    the machine; a call count is a property of the algorithm.
    """
    import importlib

    module = importlib.import_module(f"jumufraktiv.MGFdictionary.{module_name}")
    calls = []
    original = module.logminus

    def counting_logminus(x, y):
        calls.append(1)
        return original(x, y)

    monkeypatch.setattr(module, "logminus", counting_logminus)
    function = getattr(module, function_name)

    function(np.full(3, -1.0), *args, np.linspace(0.6, 1.9, 3))
    for_three = len(calls)

    calls.clear()
    function(np.full(60, -1.0), *args, np.linspace(0.6, 1.9, 60))
    for_sixty = len(calls)

    assert for_three == for_sixty, (
        f"{function_name} called logminus {for_three} times for 3 points and "
        f"{for_sixty} for 60: it is looping, not broadcasting"
    )


def test_the_jax_and_scipy_incomplete_mgfs_agree_at_the_origin():
    """`t = 0` is reachable, and the JAX path used to return `nan` there.

    `b(y) = 0` whenever every observation sits at the value the likelihood
    subtracts, which puts the evaluation point exactly at the origin. The
    quotient is `0/0` there, so the JAX path has to select the limiting formula
    rather than approach it.
    """
    from jumufraktiv.MGFdictionary.uniformMGF import (
        uniform_logimgf,
        uniform_logimgf_jax,
    )

    for u_value in [0.3, 1.25, 2.0, 5.0]:
        scipy_side = float(uniform_logimgf(0.0, 0.5, 2.0, u_value))
        jax_side = float(uniform_logimgf_jax(0.0, 0.5, 2.0, u_value))

        if scipy_side == -math.inf:
            assert jax_side == -math.inf
        else:
            assert jax_side == pytest.approx(scipy_side, rel=1e-12)
