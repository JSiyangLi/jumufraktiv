"""Executable records of defects that are known, reproduced, and scheduled.

Every test here is marked ``xfail(strict=True)``. That direction is deliberate:

* while the defect exists the suite stays green, so these do not block unrelated
  work;
* the moment a later PR fixes one, the test XPASSes and *fails* the build,
  forcing the fix to be acknowledged here and in ``CLAUDE.md``.

So this file is the to-do list for waves 1-2 of the audit, and it cannot drift
out of sync with the code. Each test names the PR that owns the repair.

The bodies assert the *correct* behaviour, not the broken behaviour, so when the
fix lands the assertion is already the right one.
"""

import numpy as np
import pytest
import sympy as sp

from jumufraktiv.derivativeDispatch import mgfDerivative
from jumufraktiv.MGFDerivative_class import MGFDerivative

# ==========================================================================
# PR 3 — import and registry integrity
# ==========================================================================
# All PR 3 entries (registry initialisation, prior-discovery isolation, the two
# unqualified imports) are fixed. Their tests now live, unmarked, in
# test_registry.py and test_dispatch_imports.py.


# ==========================================================================
# PR 4 — the fractional-order path
# ==========================================================================
@pytest.mark.xfail(
    strict=True,
    reason="PR 4: MGFDerivative._build_derivative calls mgfDerivative(t=None), "
    "but the fractional branch requires t, so any non-integer order raises "
    "at construction",
)
def test_fractional_order_posterior_can_be_constructed(gamma_prior):
    """A non-integer derivative order must not break construction.

    ``normal``, ``halfnormal`` and ``maxwell-boltzmann`` all produce a
    fractional ``a`` whenever the sample size is odd, so this is an ordinary
    use of the package, not an exotic one.
    """
    post = MGFDerivative(
        gamma_prior, data=[0.5, 1.0, 1.5], likelihood="halfnormal", method="auto"
    )

    log_ev, _ = post.evidence()
    assert np.isfinite(log_ev)


@pytest.mark.xfail(
    strict=True,
    reason="PR 4: the array-order branch of mgfDerivative coerces each order "
    "with int(o), silently truncating fractional orders to integers",
)
def test_array_order_does_not_truncate_fractional_orders(gamma_prior):
    """Vectorising over order must agree with looping over scalar orders.

    This path is not hypothetical: ``post_predictive`` always passes an array
    of orders, so a fractional ``a`` yields a silently wrong predictive.
    """
    orders = np.array([1.0, 1.5])
    batch_log, _ = mgfDerivative(orders, gamma_prior, method="auto", t=-1.0, log=True)
    scalar_log = [
        mgfDerivative(float(o), gamma_prior, method="auto", t=-1.0, log=True)[0]
        for o in orders
    ]

    assert batch_log == pytest.approx(np.array(scalar_log), rel=1e-8)


@pytest.mark.xfail(
    strict=True,
    reason="PR 4: symbolic_fractionalDeriv never applies the 1/Gamma(gamma) "
    "prefactor that all five numeric sites apply, so when it does return an "
    "expression it is Gamma(gamma) times too large — 77% at order 0.5. It also "
    "currently returns None for the Gamma prior because SymPy's laplace_transform "
    "raises internally",
)
def test_symbolic_fractional_matches_numeric(gamma_prior):
    """The symbolic and numeric fractional backends must agree."""
    from conftest import gamma_mgf_derivative_log

    expr = mgfDerivative(0.5, gamma_prior, method="symbolic", t=None)

    assert expr is not None, "symbolic fractional backend returned None"
    value = float(expr.subs(sp.Symbol("t", real=True), -1.0).evalf())
    assert np.log(value) == pytest.approx(gamma_mgf_derivative_log(0.5, -1.0), rel=1e-8)


def test_order_below_the_interpolation_threshold_is_accurate(gamma_prior):
    """Order 1.9 is below the interpolation trigger and takes the plain path.

    The dispatcher switches to interpolation only when the fractional part
    exceeds ``max(d_vec) = 0.95``, so this goes straight to quadrature and is
    exact. It is the control for the test below.
    """
    from conftest import gamma_mgf_derivative_log

    log_abs, sign = mgfDerivative(1.9, gamma_prior, method="scipy", t=-1.0, log=True)

    assert sign == 1
    assert log_abs == pytest.approx(gamma_mgf_derivative_log(1.9, -1.0), rel=1e-10)


@pytest.mark.xfail(
    strict=True,
    reason="PR 4: above the interpolation threshold the dispatcher switches to "
    "numeric_fractionalDeriv_interpolation, which fits a 4-point cubic spline in "
    "the order and is markedly LESS accurate than the plain quadrature path just "
    "below the threshold. The underlying difficulty is that gamma -> 0 makes the "
    "result (1/Gamma(gamma)) x (a diverging integral); singularity subtraction "
    "fixes that exactly and would let the interpolation module be retired",
)
@pytest.mark.parametrize("order", [1.99, 1.999])
def test_near_integer_order_is_accurate(gamma_prior, order):
    """Orders just below an integer must not lose accuracy."""
    from conftest import gamma_mgf_derivative_log

    log_abs, sign = mgfDerivative(order, gamma_prior, method="scipy", t=-1.0, log=True)

    assert sign == 1
    assert log_abs == pytest.approx(gamma_mgf_derivative_log(order, -1.0), rel=1e-10)


# ==========================================================================
# PR 5 — symbolic-path correctness
# ==========================================================================
@pytest.mark.xfail(
    strict=True,
    reason="PR 5: integerDeriv_symbolic rejects any order that is not a Python "
    "int, so the symbolic-order row of the backend matrix cannot be reached — "
    "the dispatcher warns that it will return an analytic continuation and "
    "then raises TypeError instead",
)
def test_symbolic_order_returns_expression(gamma_prior):
    """A symbolic order must yield an unevaluated expression, per the matrix."""
    n = sp.Symbol("n", positive=True, integer=True)
    result = mgfDerivative(n, gamma_prior, t=None)

    assert isinstance(result, sp.Expr)


# ==========================================================================
# PR 6 — numerical robustness
# ==========================================================================
@pytest.mark.xfail(
    strict=True,
    reason="PR 6: post_quantile brackets from a lower bound of 1e-6, where the "
    "incomplete-MGF derivative underflows and its computed sign flips negative, "
    "tripping the guard in post_cdf. This makes post_quantile, post_interval "
    "and post_sample unusable for every prior",
)
@pytest.mark.parametrize("p", [0.025, 0.5, 0.975])
def test_post_quantile_inverts_the_cdf(poisson_posterior, p):
    """The quantile function must invert the CDF."""
    q = poisson_posterior.post_quantile(p)

    assert poisson_posterior.post_cdf(q, log=False) == pytest.approx(p, abs=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason="PR 6: post_sample depends on post_quantile, which cannot bracket",
)
def test_post_sample_returns_requested_size(poisson_posterior):
    draws = poisson_posterior.post_sample(16)

    assert np.shape(draws) == (16,)
    assert np.all(draws > 0)


def test_post_cdf_is_zero_at_the_origin(poisson_posterior):
    """At u = 0 the CDF correctly vanishes. This one already works."""
    assert poisson_posterior.post_cdf(0.0, log=False) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.xfail(
    strict=True,
    reason="PR 6: post_cdf has no domain validation on u. Below zero it either "
    "recurses until RecursionError (u = -1e-9) or returns a log-CDF above zero, "
    "i.e. a probability greater than one (u = -0.5), for a parameter that is "
    "constrained positive",
)
@pytest.mark.parametrize("u", [-0.5, -1e-9])
def test_post_cdf_is_zero_below_the_support(poisson_posterior, u):
    """theta is strictly positive, so the CDF must vanish below zero."""
    assert poisson_posterior.post_cdf(u, log=False) == pytest.approx(0.0, abs=1e-12)


# ==========================================================================
# PR 12 — public API surface
# ==========================================================================
@pytest.mark.xfail(
    strict=True,
    reason="PR 12: the log principle says the log argument alone decides the "
    "return shape, but post_raw_moment returns a bare scalar while "
    "post_central_moment returns (log_abs, sign) for the same log=True",
)
def test_moment_methods_share_a_return_convention(poisson_posterior):
    raw = poisson_posterior.post_raw_moment(2)
    central = poisson_posterior.post_central_moment(2)

    assert type(raw) is type(central)


@pytest.mark.xfail(
    strict=True,
    reason="PR 12: post_sample calls the unseeded legacy np.random.rand and "
    "takes no rng argument, so results are not reproducible",
)
def test_post_sample_is_reproducible(poisson_posterior):
    """Two draws under the same seed must agree."""
    first = poisson_posterior.post_sample(8, rng=np.random.default_rng(0))
    second = poisson_posterior.post_sample(8, rng=np.random.default_rng(0))

    assert np.array_equal(first, second)
