"""Where the posterior MGF exists, and what happens outside.

`post_mgf(r)` evaluates the prior's MGF at `t = r - b`, so as `r` grows the
evaluation point moves right and eventually leaves the domain where the MGF
converges. An analytic expression evaluated there still returns a number, and
that number is the value of the *formula* rather than of the moment-generating
function -- which does not exist.

For the canonical Gamma(8, 6) posterior the formula is `(6/(6-r))**8`. The
eighth power is even, so past `r = 6` it stays positive and plausible: `25.63`
at `r = 10`, `2.76e-10` at `r = 100`. A caller has nothing to notice.

References here are closed forms written out in the test, or mpmath quadrature
on a density written out separately. Never the package's own answer.
"""

import mpmath as mp
import numpy as np
import pytest
import sympy as sp

from jumufraktiv import registry
from jumufraktiv.MGFDerivative_class import MGFDerivative
from jumufraktiv.mitMGFprior_class import mitMGFprior

#: Prior name -> (hyperparameters, declared MGF bound, density, support).
#: The bound is the supremum of `t` with `M(t) < inf`; it is a property of the
#: prior alone, which is why it is declared on the prior rather than derived
#: from the data.
PRIORS = {
    "gamma": ({"alpha": 2.0, "beta": 3.0}, 3.0),
    "uniform": ({"a": 0.5, "b": 2.0}, float("inf")),
    "pareto": ({"alpha": 2.0, "xi": 1.0}, 0.0),
    "heaviside": ({"k": 0.5}, 0.0),
}

DATA, SCALE, ORDER, RATE = [1, 2, 3], 1.0, 6.0, 3.0


def _posterior(name, method="auto"):
    registry.initialize()
    prior = mitMGFprior.from_registry(name, params=PRIORS[name][0])
    return MGFDerivative(
        prior, data=DATA, likelihood="poisson", scale=SCALE, method=method
    )


@pytest.mark.parametrize("name", sorted(PRIORS))
def test_every_registry_prior_declares_where_its_mgf_converges(name):
    """A new prior must not silently inherit an unexamined infinity.

    Inheriting the default means "no `r` is ever refused", which is right for a
    bounded support and wrong for every heavy tail.
    """
    registry.initialize()
    spec = registry.get_prior(name)(PRIORS[name][0])

    assert "mgf_finite_below" in spec, (
        f"prior '{name}' does not declare mgf_finite_below. Work out the "
        f"supremum of t for which M(t) is finite and state it; use "
        f"float('inf') for a bounded support."
    )
    assert float(spec["mgf_finite_below"]) == PRIORS[name][1]


@pytest.mark.parametrize("r_value", [5.0, 5.9, 5.99])
def test_inside_the_radius_the_mgf_is_exact(r_value):
    """Gamma(8, 6) posterior: `M(r) = (6/(6-r))**8`, in closed form."""
    post = _posterior("gamma")
    expected = (6.0 / (6.0 - r_value)) ** 8

    assert float(post.post_mgf(r_value, log=False)) == pytest.approx(
        expected, rel=1e-12
    )


@pytest.mark.parametrize("r_value", [6.0, 6.1, 10.0, 100.0])
def test_outside_the_radius_it_refuses_rather_than_returning_the_formula(r_value):
    """`E[exp(r*theta)]` is infinite here; the formula is finite and positive."""
    post = _posterior("gamma")

    with pytest.raises(ValueError, match="posterior MGF does not exist"):
        post.post_mgf(r_value, log=False)


def test_the_refusal_names_the_largest_admissible_r():
    """A refusal a caller cannot act on is barely better than a wrong number."""
    post = _posterior("gamma")

    with pytest.raises(ValueError) as excinfo:
        post.post_mgf(10.0)

    message = str(excinfo.value)
    assert "t = r - b" in message
    assert "r above 6" in message


def test_an_array_is_refused_if_any_element_is_outside():
    """One bad element must not be averaged into a plausible array."""
    post = _posterior("gamma")

    with pytest.raises(ValueError, match="posterior MGF does not exist"):
        post.post_mgf(np.array([1.0, 5.0, 7.0]))

    inside = np.asarray(post.post_mgf(np.array([1.0, 2.0, 5.0]), log=False))
    assert inside == pytest.approx(
        [(6.0 / (6.0 - r)) ** 8 for r in (1.0, 2.0, 5.0)], rel=1e-12
    )


def test_a_bounded_support_refuses_nothing():
    """The uniform MGF is entire, so no `r` is out of range."""
    post = _posterior("uniform")

    for r_value in [1.0, 5.0, 50.0]:
        assert np.isfinite(float(post.post_mgf(r_value)))


def test_a_symbolic_r_still_returns_an_expression():
    """The guard applies to numeric points; it must not break the symbolic one."""
    post = _posterior("gamma")

    expression = post.post_mgf(sp.Symbol("r"), log=False)

    assert isinstance(expression, sp.Expr)


# ==========================================================================
# The origin, r = b
# ==========================================================================
# There `t = r - b = 0`, the exponential is 1, and the posterior MGF reduces to
# the raw moment `E[Theta^a]`. Two things went wrong there and both returned
# `nan`: a prior whose moment exists could not be substituted into (the uniform
# MGF has `t` in a denominator, so `subs` sees 0/0), and a prior whose moment
# does not exist was not refused.
def _exact_moment_ratio(density, low, high):
    numerator = mp.quad(lambda x: x**ORDER * density(x), [low, high])
    denominator = mp.quad(
        lambda x: x**ORDER * mp.e ** (-RATE * x) * density(x), [low, high]
    )
    return float(numerator / denominator)


def test_the_origin_is_exact_for_a_prior_whose_moment_exists():
    """`sp.limit` is not the fix: it returns `oo` here, where the value is finite.

    The uniform posterior's value at `r = b` is a perfectly ordinary number.
    Substitution gives `nan` and SymPy's limit gives `oo`, so the origin routes
    through the expectation backend, which integrates `E[Theta^a]` directly.
    """
    mp.mp.dps = 40
    expected = _exact_moment_ratio(
        lambda x: mp.mpf(1) / mp.mpf("1.5"), mp.mpf("0.5"), mp.mpf(2)
    )
    post = _posterior("uniform")

    assert float(post.post_mgf(RATE, log=False)) == pytest.approx(expected, rel=1e-10)
    assert expected == pytest.approx(141.4041138, rel=1e-8)


def test_the_origin_is_exact_for_the_gamma_posterior_too():
    """Gamma(8, 6) at `r = b`: `E[Theta^6] / M(-3)` has a closed form.

    The ratio is `(6/(6-3))**8 / ... ` only in the sense that the posterior MGF
    at `r = 3` is `(6/3)**8 = 256` exactly, which a reader can check by hand.
    """
    post = _posterior("gamma")

    assert float(post.post_mgf(RATE, log=False)) == pytest.approx(256.0, rel=1e-12)


@pytest.mark.parametrize("name", ["pareto", "heaviside"])
def test_the_origin_is_refused_when_the_moment_diverges(name):
    """`E[Theta^6]` is infinite for Pareto(2) and for the improper Heaviside.

    It used to return `nan`, which is neither the value nor a refusal.
    """
    post = _posterior(name)

    with pytest.raises(ValueError, match=r"E\[Theta\^6\]"):
        post.post_mgf(RATE, log=False)


def test_a_moment_below_the_bound_is_still_admitted_at_the_origin():
    """The guard must not reject the orders `max_finite_moment` admits.

    Pareto(alpha=2) has a finite first moment, so a single observation -- which
    makes the derivative order 1 -- is fine at the origin. Refusing it would be
    the over-correction.
    """
    registry.initialize()
    prior = mitMGFprior.from_registry("pareto", params={"alpha": 2.0, "xi": 1.0})
    post = MGFDerivative(prior, data=[1], likelihood="poisson", scale=1.0)

    assert np.isfinite(float(post.post_mgf(1.0, log=False)))
