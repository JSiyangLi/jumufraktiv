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

import warnings

import numpy as np
import pytest
import sympy as sp

from jumufraktiv.derivativeDispatch import mgfDerivative

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


# The symbolic fractional backend is repaired. It returns a closed form for
# the Gamma prior, exact to 1e-16 with the 1/Gamma(gamma) prefactor applied,
# and raises NotImplementedError naming the prior where SymPy declines --
# rather than returning None, which used to surface as a TypeError far from
# its cause. Its tests now live, unmarked, in test_symbolic_fractional.py.


# ==========================================================================
# PR 5 — symbolic-path correctness
# ==========================================================================
def test_an_integer_valued_sympy_order_is_accepted(gamma_prior):
    """`sp.Integer(2)` must behave exactly like `2`.

    This half was a real defect. The guard was `isinstance(order, int)`, which
    rejects `sympy.Integer` -- an integer by every meaning except Python's type
    check, and one SymPy arithmetic produces routinely. `resolve_backend`
    classifies it as a symbolic order, so it reached the same dead end as a
    free symbol and raised `TypeError`.
    """
    from_python_int = mgfDerivative(2, gamma_prior, method="symbolic", t=None)
    from_sympy_int = mgfDerivative(
        sp.Integer(2), gamma_prior, method="symbolic", t=None
    )

    assert sp.simplify(from_python_int - from_sympy_int) == 0


def test_a_genuinely_symbolic_order_is_refused_clearly(gamma_prior):
    """An order carrying free symbols is refused, and says why.

    This is **not** a defect record. It documents a decision: `sympy.diff`
    needs a concrete number of times to differentiate and cannot return a
    formula in `n`. Closed forms in the order exist for particular priors --
    the Gamma MGF gives a Pochhammer symbol -- but there is no general route,
    and the package has no use for one, since `a(y)` comes from the data and is
    always numeric.

    What was wrong was the *reporting*. The dispatcher warned that the result
    would be "the analytic continuation to non-integer orders", then raised
    `TypeError` blaming SymPy's support for symbolic differentiation -- which
    is not what is missing. The warning fired before the failure, so the last
    thing a caller saw was a claim about a result they never received.
    """
    n = sp.Symbol("n", positive=True, integer=True)

    with pytest.raises(NotImplementedError, match="symbolic number of times"):
        mgfDerivative(n, gamma_prior, method="symbolic", t=None)


def test_refusing_a_symbolic_order_does_not_warn_first(gamma_prior):
    """No warning may precede the refusal.

    Asserted separately because `pytest.raises` alone would pass while the
    misleading warning was still being issued.
    """
    n = sp.Symbol("n", positive=True, integer=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(NotImplementedError):
            mgfDerivative(n, gamma_prior, method="symbolic", t=None)


# ==========================================================================
# PR 6 — numerical robustness
# ==========================================================================
# The kernel defects are fixed and their records are gone: large-order overflow,
# `log=False` raising on the near-integer path, near-integer inaccuracy, and
# truncation that did not adapt to the evaluation point. All four were symptoms
# of one design fault, and all four now assert positively, against the
# closed-form Gamma reference, in test_fixed_grid_kernel.py.
#
# The domain guards, `post_quantile`'s bracketing and `logminus` were repaired
# earlier; those tests live in test_numerical_guards.py.
#
# PR 6c closed the rest. mpmath's dps floor is gone (the range was symmetric
# and the integrand float64); the alternating-CGF cancellation is computed
# rather than flagged, via the direct-expectation route; and sequential
# updating works for numeric and fractional posteriors, because that route
# needs only the prior's density. Those tests live in
# test_sequential_update.py and test_deferred_construction.py.
#
# Nothing from PR 6 remains recorded here.


# ==========================================================================
# Unscheduled — the Pareto prior's numeric incomplete MGF
# ==========================================================================
# Found while clearing PR 8's lint baseline. `F841` flagged a `sign` computed
# and discarded in `pareto_logimgf`; the discarded sign turned out to be the
# least of it. Both numeric routes are unusable and the two fail differently,
# while the symbolic route is exact -- so this is a defect in the numeric
# implementations, not in the mathematics.
#
# It is recorded rather than repaired because it is a numerical fix needing its
# own verification, and PR 8 is a module-layout and dead-code pass. No PR owns
# it yet; see "Known-broken" in CLAUDE.md.
def _pareto_prior():
    from jumufraktiv import registry
    from jumufraktiv.mitMGFprior_class import mitMGFprior

    registry.initialize()
    return mitMGFprior.from_registry("pareto", params={"alpha": 3.0, "xi": 1.0})


#: E[e^{tX} 1{X <= u}] for Pareto(alpha=3, xi=1) at t=-1, u=2, by direct
#: quadrature of the density at 40 digits -- independent of the package.
PARETO_IMGF_AT_MINUS_ONE = 0.24880390851855957


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="unscheduled: pareto_imgf calls scipy's gammaincc(a, z) with "
    "a = -alpha < 0, which is outside its domain, so it returns NaN at every "
    "argument",
)
def test_pareto_numeric_incomplete_mgf_is_finite():
    value = _pareto_prior().imgf(-1.0, 2.0)

    assert value == pytest.approx(PARETO_IMGF_AT_MINUS_ONE, rel=1e-10)


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="unscheduled: pareto_imgf_jax calls jnp.gamma, which does not "
    "exist in jax.numpy, so it raises before reaching the gammaincc domain "
    "error its scipy twin hits",
)
def test_pareto_jax_incomplete_mgf_is_finite():
    value = float(_pareto_prior().imgf_jax(-1.0, 2.0))

    assert value == pytest.approx(PARETO_IMGF_AT_MINUS_ONE, rel=1e-10)


def test_pareto_symbolic_incomplete_mgf_is_correct():
    """Not broken -- asserted here to bound the defect above.

    The symbolic incomplete MGF is exact, so the two failures are in the
    numeric implementations rather than in the expression they implement.
    Without this, "the Pareto incomplete MGF is broken" would be too broad a
    claim, and a repair might start from the wrong end.
    """
    from jumufraktiv.symbols import t as t_sym
    from jumufraktiv.symbols import u as u_sym

    expr = _pareto_prior().imgf_sym.subs({t_sym: -1.0, u_sym: 2.0})

    assert float(sp.re(sp.N(expr))) == pytest.approx(
        PARETO_IMGF_AT_MINUS_ONE, rel=1e-10
    )


# ==========================================================================
# PR 12 — public API surface
# ==========================================================================
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
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
    raises=TypeError,
    reason="PR 12: post_sample calls the unseeded legacy np.random.rand and "
    "takes no rng argument, so results are not reproducible",
)
def test_post_sample_is_reproducible(poisson_posterior):
    """Two draws under the same seed must agree."""
    first = poisson_posterior.post_sample(8, rng=np.random.default_rng(0))
    second = poisson_posterior.post_sample(8, rng=np.random.default_rng(0))

    assert np.array_equal(first, second)
