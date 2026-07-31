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
# The construction defect (`_build_derivative` passing `t=None` to a backend that
# requires it) is fixed. Its test now lives, unmarked and much expanded, in
# test_deferred_construction.py.


# The array-order coercion (`int(o)` turning a fractional order into a whole
# one, plus the float() coercions of t and u and the flattened reassembly) is
# fixed. Its two records -- the dispatcher-level one and the public-API one
# asserting that the 0th raw moment is exactly 1 -- now live, unmarked, in
# test_array_order.py, along with the sample-size parity coverage that the
# originals lacked.


@pytest.mark.xfail(
    strict=True,
    reason="PR 4b: the numeric backends accumulate the quadrature in linear "
    "space and clamp on overflow, so a large derivative order silently loses "
    "the overflowing contributions instead of raising",
)
@pytest.mark.slow
def test_large_order_does_not_silently_overflow(gamma_prior):
    """A large order must be right or raise, not be quietly wrong.

    Measured against the closed-form Gamma reference: order 150.5 is correct
    to 6e-14 nats, but order 300.5 returns 694.234 where the exact value is
    1006.311 -- wrong by 312 nats, a factor of about 1e135, with no warning.
    The order is `a = sum(y)` for several likelihoods, so it grows with the
    sample and this is reachable from ordinary data.
    """
    from conftest import gamma_mgf_derivative_log

    log_abs, sign = mgfDerivative(300.5, gamma_prior, method="scipy", t=-1.0, log=True)

    assert sign == 1
    assert log_abs == pytest.approx(gamma_mgf_derivative_log(300.5, -1.0), rel=1e-8)


@pytest.mark.xfail(
    strict=True,
    reason="PR 4b: numeric_fractionalDeriv_interpolation binds `result` only "
    "inside its array branch, so the scalar path raises UnboundLocalError. "
    "The module is to be retired rather than repaired",
)
@pytest.mark.slow
def test_near_integer_order_works_without_log(gamma_prior):
    """The `log` argument decides the return shape and nothing else.

    That is the log principle, and here `log=False` does not merely change the
    shape -- it raises `UnboundLocalError: cannot access local variable
    'result'`. The same call with `log=True` returns a value.
    """
    value = mgfDerivative(1.99, gamma_prior, method="scipy", t=-1.0, log=False)

    assert np.isfinite(float(value))


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


@pytest.mark.slow
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
@pytest.mark.slow
def test_near_integer_order_is_accurate(gamma_prior, order):
    """Orders just below an integer must not lose accuracy."""
    from conftest import gamma_mgf_derivative_log

    log_abs, sign = mgfDerivative(order, gamma_prior, method="scipy", t=-1.0, log=True)

    assert sign == 1
    assert log_abs == pytest.approx(gamma_mgf_derivative_log(order, -1.0), rel=1e-10)


@pytest.mark.xfail(
    strict=True,
    reason="PR 6: the L-doubling loop stops widening the integration range too "
    "early at large |t|. It compares consecutive iterates against "
    "tol * max(1.0, |prev|) with tol=1e-6, which is an absolute test whenever "
    "the integral is below 1, and a consecutive-iterate change underestimates "
    "the remaining tail when convergence is slow",
)
@pytest.mark.parametrize("t_value", [-14.0, -30.0])
@pytest.mark.slow
def test_quadrature_reaches_tolerance_at_large_evaluation_points(gamma_prior, t_value):
    """Accuracy must not depend on where `t` happens to land.

    ``maxwell-boltzmann`` on three observations gives ``a = 4.5, b = 14``, an
    ordinary use of the package, and the evidence comes out with relative error
    2.9e-06.

    **Both `t` values are here on purpose, because they do not have the same
    fix.** Measured against the closed-form Gamma reference:

    ==========  ============  ==========  ===========
    setting       t = -14       t = -30
    ==========  ============  ==========  ===========
    tol=1e-6      2.9e-06       1.5e-06     (default)
    tol=1e-9      1.0e-15       2.8e-10
    tol=1e-12     1.0e-15       2.1e-10
    ==========  ============  ==========  ===========

    ``epsabs``, ``epsrel`` and ``limit`` change neither. Tightening ``tol``
    repairs ``t = -14`` outright but leaves ``t = -30`` on a plateau at
    2.1e-10, because it makes the loop widen for longer without making the
    stopping rule correct. So a ``tol`` change would turn the first case green
    and not the second — which is exactly why both are recorded. The real
    repair is the fixed-grid kernel, which takes its range from ``gamma``
    directly instead of discovering it by doubling. See CLAUDE.md,
    "Numerical policy".
    """
    from conftest import gamma_mgf_derivative_log

    log_abs, sign = mgfDerivative(4.5, gamma_prior, method="scipy", t=t_value, log=True)

    assert sign == 1
    assert log_abs == pytest.approx(gamma_mgf_derivative_log(4.5, t_value), rel=1e-10)


@pytest.mark.xfail(
    strict=True,
    reason="PR 6: MGFDerivative.to_prior_object can only build the updated "
    "prior's MGF symbolically. Its numeric route returns a prior with no "
    "mgf_sym/cgf_sym, which no derivative backend can consume, so a posterior "
    "from any numeric backend cannot be updated. Since no symbolic backend "
    "works for fractional orders, fractional posteriors cannot be updated at all",
)
def test_fractional_posterior_can_be_updated_sequentially(gamma_prior):
    """Evidence factorises, so staged conditioning must match one-shot.

    Reachable only since the deferred-construction fix: a fractional posterior
    could not previously be built. `update` now raises rather than returning the
    ``-inf`` that `scipy` and `mpmath` produced, but raising is not the goal.
    """
    staged = MGFDerivative(gamma_prior, data=[1.0, 2.0, 3.0], likelihood="halfnormal")
    updated = staged.update([1.0], likelihood="halfnormal")

    one_shot = MGFDerivative(
        gamma_prior, data=[1.0, 2.0, 3.0, 1.0], likelihood="halfnormal"
    )

    assert staged.evidence()[0] + updated.evidence()[0] == pytest.approx(
        one_shot.evidence()[0], rel=1e-8
    )


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
