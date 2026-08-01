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

from conftest import gamma_mgf_derivative_log
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
# Three findings from PR 8's review, all the same shape: a check that exists at
# one entry point and not at the neighbouring one. Recorded rather than fixed,
# because PR 12 owns the interface decisions behind them.


def test_post_predictive_rejects_a_misspelled_parameter(gamma_prior):
    """The constructor catches this typo, and now so does the method.

    `MGFDerivative(..., likelihood="weibull", rh=2.0)` raises with "did you
    mean 'rho'?". `post.post_predictive([2.0], rh=9.0)` used to return
    -1.1053356325054668 -- exactly the value for the default -- where
    `rho=9.0` gives -14.108023936618833. A typo cost 13.003 nats, silently.

    PR 12a fixed the neighbouring half of this: `post_predictive` used to
    ignore the parameters stored at construction. The forwarding was repaired
    in `_likelihood_arguments`; PR 12 added the validation to the same place,
    so both predictive paths are covered by one check.
    """
    post = MGFDerivative(
        gamma_prior, data=[1.0, 2.0, 3.0], likelihood="weibull", rho=2.0
    )

    with pytest.raises(TypeError, match="rh"):
        post.post_predictive([2.0], rh=9.0)

    # The cost of not raising, so the record carries the size of the defect
    # rather than only its shape.
    assert post.post_predictive([2.0]) == pytest.approx(-1.1053356325054668)
    assert post.post_predictive([2.0], rho=9.0) == pytest.approx(
        -14.108023936618833
    )


def test_the_two_fractional_entry_points_name_options_alike():
    """Both spellings were valid layer options, so no guard could catch the mix-up.

    `mgfDerivative_fractional(..., integer_method="bell")` used to reach the
    kernel with `integer_method="symbolic"`, the default. The value was
    discarded because the parameter was spelled `integerDeriv_method` here,
    while `integer_method` was a perfectly good name elsewhere in the layer --
    so it passed the unknown-option guard and landed in `**kwargs`, where
    nothing read it.

    The old spelling now raises, which is the second half of the repair: a
    caller who was using it must see an error rather than a silently different
    backend.
    """
    import inspect

    from jumufraktiv.derivativeDispatch import (
        mgfDerivative,
        mgfDerivative_fractional,
    )

    unified = set(inspect.signature(mgfDerivative).parameters)
    fractional = set(inspect.signature(mgfDerivative_fractional).parameters)

    assert "integer_method" in unified and "integer_method" in fractional
    assert "integerDeriv_method" not in fractional


def test_the_retired_spelling_raises(gamma_prior):
    """`integerDeriv_method` must not fall through to `**kwargs` unread."""
    from jumufraktiv.derivativeDispatch import mgfDerivative_fractional

    with pytest.raises(TypeError, match="integer_method"):
        mgfDerivative_fractional(
            1.5,
            gamma_prior,
            method="scipy",
            t=-1.0,
            integerDeriv_method="bell",
        )


def test_tol_reaches_the_kernel_on_the_default_path(gamma_prior):
    """`tol` is documented as tuning the quadrature. It used not to, under `auto`.

    Measured for a Gamma(2, 3) prior at order 1.5, t = -1, before the repair:

    ==================  =====================
    call                log of the derivative
    ==================  =====================
    auto, tol=1e-14     -1.4538320842478814
    auto, tol=1e-1      -1.4538320842478814
    scipy, tol=1e-14    -1.4538320842363235
    scipy, tol=1e-1     -1.4542925617536986
    ==================  =====================

    The option was real -- it moved the answer by 4.6e-4 through `scipy` --
    and simply unreachable through `auto`, which routes to the expectation
    integral. That route had no tuning parameters at all; it now takes `tol`
    as the quadrature's relative tolerance.

    Asserted against the closed form rather than against a recorded number,
    because "the two calls differ" would also be satisfied by a `tol` that
    made the answer worse.
    """
    from jumufraktiv.derivativeDispatch import mgfDerivative

    exact = gamma_mgf_derivative_log(1.5, -1.0)

    loose = mgfDerivative(1.5, gamma_prior, method="auto", t=-1.0, log=True, tol=1e-1)
    tight = mgfDerivative(1.5, gamma_prior, method="auto", t=-1.0, log=True, tol=1e-14)

    loose_error = abs(loose[0] - exact)
    tight_error = abs(tight[0] - exact)

    assert loose[0] != tight[0]
    assert tight_error < loose_error
    assert tight_error == pytest.approx(0.0, abs=1e-15)


def test_options_the_chosen_route_cannot_read_are_announced(gamma_prior):
    """An option that reaches no route silently is the defect; a warning is the fix.

    `dps` is read by the mpmath kernel and by nothing else, so passing it with
    `method='auto'` -- which routes to the expectation integral -- cannot have
    an effect. It used to be accepted in silence.
    """
    from jumufraktiv.derivativeDispatch import mgfDerivative

    with pytest.warns(UserWarning, match=r"dps.*mpmath"):
        mgfDerivative(1.5, gamma_prior, method="auto", t=-1.0, log=True, dps=60)

    # The option the route *does* read must not warn, or the warning is noise.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mgfDerivative(1.5, gamma_prior, method="auto", t=-1.0, log=True, tol=1e-12)


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


def test_post_sample_is_reproducible(poisson_posterior):
    """Two draws under the same seed must agree.

    `post_sample` used to call `np.random.rand`, the legacy global generator,
    and took no seed, so no draw from it could be reproduced. An integer seed
    must work as well as a `Generator`, since that is how most callers reach
    for one.
    """
    first = poisson_posterior.post_sample(8, rng=np.random.default_rng(0))
    second = poisson_posterior.post_sample(8, rng=np.random.default_rng(0))
    assert np.array_equal(first, second)

    assert np.array_equal(
        poisson_posterior.post_sample(8, rng=0),
        poisson_posterior.post_sample(8, rng=0),
    )

    # And a different seed must give a different draw, or the test above would
    # also pass for a method that ignored `rng` and returned a constant.
    assert not np.array_equal(first, poisson_posterior.post_sample(8, rng=1))
