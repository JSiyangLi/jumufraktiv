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
# The Pareto prior's numeric incomplete MGF — repaired in PR 12
# ==========================================================================
# Both numeric routes were unusable and failed differently, while the symbolic
# route was exact, so the defect was in the implementations rather than in the
# expression. The SciPy side now computes it to 2.1e-15 of direct quadrature;
# the JAX side refuses, because JAX has no real-order exponential integral.
#
# The tests stay here rather than moving out, since this file is where the
# defect was reproduced and the assertions were already the right ones.
def _pareto_prior():
    from jumufraktiv import registry
    from jumufraktiv.mitMGFprior_class import mitMGFprior

    registry.initialize()
    return mitMGFprior.from_registry("pareto", params={"alpha": 3.0, "xi": 1.0})


#: E[e^{tX} 1{X <= u}] for Pareto(alpha=3, xi=1) at t=-1, u=2, by direct
#: quadrature of the density at 40 digits -- independent of the package.
PARETO_IMGF_AT_MINUS_ONE = 0.24880390851855957


def test_pareto_numeric_incomplete_mgf_is_finite():
    """It returned `nan` at every argument, from `gammaincc(-alpha, z)`.

    `scipy.special.gammaincc` is the regularised upper incomplete gamma and
    needs a positive first argument. The value comes instead from the
    generalised exponential integral, `Gamma(a, z) = z**a E_{1-a}(z)`, which
    holds for every real `a`.
    """
    value = _pareto_prior().imgf(-1.0, 2.0)

    assert value == pytest.approx(PARETO_IMGF_AT_MINUS_ONE, rel=1e-10)


@pytest.mark.parametrize(
    ("alpha", "t_value", "u_value"),
    [
        (0.5, -0.1, 1.5),
        (2.0, -1.0, 2.0),
        (3.0, -5.0, 10.0),
        (5.5, -1.0, 10.0),
    ],
)
def test_pareto_numeric_incomplete_mgf_matches_quadrature(alpha, t_value, u_value):
    """Against direct quadrature of the density at 40 digits, not against the
    package's own symbolic route.

    A fractional and an integer `alpha` are both covered, because the obvious
    float64 alternatives fail on exactly one of the two: a downward recurrence
    on the incomplete gamma divides by zero at integer `alpha`, and
    `scipy.special.expn` truncates a real order to an integer, so it answers a
    different question for fractional `alpha` with only a `RuntimeWarning`.
    """
    import mpmath as mp

    from jumufraktiv.mitMGFprior_class import mitMGFprior

    mp.mp.dps = 40

    def density(theta):
        return mp.e ** (mp.mpf(t_value) * theta) * mp.mpf(alpha) / theta ** (alpha + 1)

    expected = float(mp.quad(density, [1.0, u_value]))
    prior = mitMGFprior.from_registry("pareto", params={"alpha": alpha, "xi": 1.0})

    assert prior.imgf(t_value, u_value) == pytest.approx(expected, rel=1e-12)


def test_pareto_jax_incomplete_mgf_refuses_clearly():
    """JAX cannot express this, and now says so instead of failing obscurely.

    It needs `E_{1+alpha}` at real order; `jax.scipy.special` has `expn` at
    integer order only and `gammaincc` for a positive first argument only. The
    old body called `jnp.gamma`, which does not exist, so it raised
    `AttributeError` from inside a traced computation -- naming a missing
    attribute rather than a missing capability.
    """
    for method in ("imgf_jax", "logimgf_jax"):
        with pytest.raises(NotImplementedError, match="integer order only"):
            getattr(_pareto_prior(), method)(-1.0, 2.0)


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
# Unscheduled — accuracy at t = 0 as the order approaches the moment bound
# ==========================================================================
# Found while making `_check_moment_exists_at_origin` reachable from all three
# public entry points. The guard's contract is that an order strictly below the
# prior's `max_finite_moment` is admissible, which is true of the mathematics
# and not of the quadrature: at `t = 0` the integrand `theta^a p(theta)` decays
# only polynomially, like `theta^(a - alpha - 1)`, and the closer `a` gets to
# `alpha` the slower. Away from the origin the exponential restores geometric
# decay and both routes are exact to machine precision, so this is a `t = 0`
# defect specifically and not a heavy-tail defect in general.
#
# Measured for Pareto(alpha=2, xi=1) against the exact E[Theta^a] = 2/(2 - a):
#
#   order   exact       scipy grid   auto (expectation)
#   0.5     1.333       4.3e-13      2.5e-16
#   1.0     2.000       1.4e-17      1.4e-12
#   1.5     4.000       2.0e-04      7.2e-07
#   1.9    20.000       8.2e-02      2.2e-02
#   1.99  200.000       6.1e-01      2.7e-01
#
# At t = -0.01 the same three near-boundary orders are accurate to 1.2e-16.
#
# Unscheduled: it is tail-handling in the quadrature, needing its own
# verification, and PR 12 is an interface pass.
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="unscheduled: at t = 0 the integrand decays only polynomially, so "
    "accuracy collapses as the order approaches the prior's max_finite_moment "
    "-- 61% relative error at order 1.99 against Pareto(alpha=2)",
)
@pytest.mark.parametrize("order", [1.5, 1.9, 1.99])
def test_moments_near_the_bound_are_accurate_at_the_origin(order):
    """An order the guard admits must also be one the kernel can compute."""
    prior = _pareto_prior_alpha_two()
    exact = np.log(2.0 / (2.0 - order))

    log_abs, sign = mgfDerivative(order, prior, method="auto", t=0.0, log=True)

    assert sign == 1
    assert log_abs == pytest.approx(exact, rel=1e-8)


def test_moments_near_the_bound_are_accurate_just_off_the_origin():
    """Not broken -- asserted here to bound the defect above.

    Without this, "heavy-tailed priors lose accuracy at high orders" would be
    far too broad a claim, and a repair might start from the wrong end. The
    same orders at t = -0.01 agree with direct quadrature to 1.2e-16.
    """
    import mpmath as mp

    prior = _pareto_prior_alpha_two()
    mp.mp.dps = 40

    def integrand(theta, a):
        """theta**a e^{t theta} p(theta) for Pareto(2, 1) at t = -0.01."""
        return theta ** mp.mpf(a) * mp.e ** (mp.mpf("-0.01") * theta) * 2 / theta**3

    for order in (1.5, 1.9, 1.99):
        reference = float(
            mp.log(mp.quad(lambda th, a=order: integrand(th, a), [1, mp.inf]))
        )
        log_abs, _ = mgfDerivative(order, prior, method="auto", t=-0.01, log=True)

        assert log_abs == pytest.approx(reference, rel=1e-12)


def _pareto_prior_alpha_two():
    from jumufraktiv import registry
    from jumufraktiv.mitMGFprior_class import mitMGFprior

    registry.initialize()
    return mitMGFprior.from_registry("pareto", params={"alpha": 2.0, "xi": 1.0})


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
    assert post.post_predictive([2.0], rho=9.0) == pytest.approx(-14.108023936618833)


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


def test_a_pair_is_returned_exactly_where_the_quantity_can_be_negative(
    poisson_posterior,
):
    """The rule PR 12 settled: `(log_abs, sign)` iff the quantity is signed.

    Every quantity here is non-negative by construction except a central moment
    of odd order. `Theta > 0`, so `D^a M(t) = E[Theta^a e^{t Theta}] > 0`, and
    an evidence, density, CDF, MGF, predictive or raw moment built from it is
    positive too. A negative value in any of those is a numerical failure, not
    a signed answer, and the package raises rather than returning a flag.

    Asserted together rather than method by method, because the property is
    that they agree on a rule -- the previous state had `evidence` and
    `post_central_moment` returning pairs for two entirely different reasons,
    one of them being no reason at all.
    """
    unsigned = {
        "evidence": poisson_posterior.evidence(),
        "post_density": poisson_posterior.post_density(1.0),
        "post_cdf": poisson_posterior.post_cdf(1.0),
        "post_mgf": poisson_posterior.post_mgf(-1.0),
        "post_predictive": poisson_posterior.post_predictive([2.0]),
        "post_raw_moment": poisson_posterior.post_raw_moment(2),
    }
    for name, value in unsigned.items():
        assert not isinstance(value, tuple), f"{name} returned a pair"
        assert np.isfinite(float(value)), f"{name} was not a finite log"

    log_abs, sign = poisson_posterior.post_central_moment(2)
    assert np.isfinite(log_abs)
    assert sign in (-1, 1)


def test_the_signed_moment_really_can_be_negative():
    """The one case that justifies the pair, so it is not kept out of symmetry.

    A Uniform(0.5, 2) prior with Poisson counts pushes the posterior against
    the upper endpoint, which makes it left-skewed: `mu_3 = -0.0219`. Under a
    Gamma prior the same moment is `+0.0741`.
    """
    from jumufraktiv.mitMGFprior_class import mitMGFprior

    uniform = mitMGFprior.from_registry("uniform", params={"a": 0.5, "b": 2.0})
    post = MGFDerivative(uniform, data=[1, 2, 3], likelihood="poisson", scale=1.0)

    log_abs, sign = post.post_central_moment(3)

    assert sign == -1
    assert -float(np.exp(log_abs)) == pytest.approx(-0.0219, abs=1e-3)


def test_a_negative_evidence_raises_rather_than_returning_a_sign(gamma_prior):
    """Dropping the sign must not mean dropping the check that used it.

    `_store_result` refuses a negative derivative at `t = -b`. That refusal is
    what makes the sign redundant, so it is asserted here rather than left
    implied by the absence of a return value.
    """
    from jumufraktiv.MGFDerivative_class import MGFDerivative as _MGFDerivative

    post = _MGFDerivative(gamma_prior, data=[1, 2, 3], likelihood="poisson", scale=1.0)

    with pytest.raises(ValueError, match="negative"):
        post._store_result((0.5, -1))


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


def test_pareto_complete_mgf_is_finite():
    """The same `gammaincc(-alpha, z)` domain error, in the complete MGF.

    Repairing `pareto_imgf` and `pareto_logimgf` left `pareto_cgf` and
    `pareto_mgf` on the same broken call two functions away, and those are
    wired into the prior spec as its numeric `cgf` and `mgf`, so `prior.mgf(t)`
    returned `nan` at every argument.
    """
    import mpmath as mp

    mp.mp.dps = 40
    prior = _pareto_prior()

    def density(theta):
        return mp.e ** (-theta) * 3 / theta**4

    expected = float(mp.quad(density, [1, mp.inf]))

    assert prior.mgf(-1.0) == pytest.approx(expected, rel=1e-12)
    assert prior.cgf(-1.0) == pytest.approx(float(np.log(expected)), rel=1e-12)
    assert prior.cgf(0.0) == 0.0


def test_pareto_jax_complete_mgf_refuses_clearly():
    """JAX cannot express the complete MGF either, for the same reason."""
    for method in ("mgf_jax", "cgf_jax"):
        with pytest.raises(NotImplementedError, match=r"jax\.scipy\.special"):
            getattr(_pareto_prior(), method)(-1.0)


def test_symbolic_joint_predictive_accepts_an_array():
    """`individual=False` on a symbolic posterior raised for every input.

    The helper wrapped its argument as `[x]`, so the joint branch's array
    became two-dimensional and `_extract_1d` rejected it -- meaning the joint
    symbolic predictive could not be computed at all. Checked against the
    numeric posterior for the same prior, which takes a different code path.
    """
    from test_symbolic_correctness import _gamma_mgf, _gamma_pdf

    from jumufraktiv.mitMGFprior_class import mitMGFprior

    params = {"alpha": 2.0, "beta": 3.0}
    symbolic = mitMGFprior(
        name="gamma_substituted",
        mgf_sym=_gamma_mgf(),
        pdf_sym=_gamma_pdf(),
        params=params,
    ).as_mitMGFprior()
    numeric = mitMGFprior.from_registry("gamma", params=params)

    def posterior(prior, **kwargs):
        return MGFDerivative(
            prior, data=[1, 2, 3], likelihood="poisson", scale=1.0, **kwargs
        )

    joint_symbolic = posterior(symbolic, method="symbolic").post_predictive(
        [2.0, 3.0], individual=False
    )
    joint_numeric = posterior(numeric).post_predictive([2.0, 3.0], individual=False)

    assert float(joint_symbolic) == pytest.approx(float(joint_numeric), rel=1e-10)


def test_uniform_cgf_works_below_the_origin():
    """It raised `ValueError: math domain error` for every `t < 0`.

    `M(t) = (e^{bt} - e^{at}) / (t (b - a))` is positive throughout, but below
    the origin neither factor is: `e^{bt}` is the smaller exponential and `t`
    is negative. Taking the log of each separately asked for the log of a
    negative number twice, so the CGF failed across the whole of this
    package's operating range -- the posterior sits at `t = -b` -- while
    working above the origin, where it is never evaluated.
    """
    import mpmath as mp

    from jumufraktiv.mitMGFprior_class import mitMGFprior

    mp.mp.dps = 40
    prior = mitMGFprior.from_registry("uniform", params={"a": 0.5, "b": 2.0})

    for t_value in (-50.0, -5.0, -1.0, -0.5):
        reference = float(
            mp.log(
                mp.quad(
                    lambda th, tv=t_value: mp.e ** (mp.mpf(tv) * th) / mp.mpf("1.5"),
                    [0.5, 2],
                )
            )
        )
        assert prior.cgf(t_value) == pytest.approx(reference, rel=1e-12)

    assert prior.cgf(0.0) == 0.0
