"""Evaluating at many points at once must agree with doing them one at a time.

Whenever the evaluation point `t` is an array, the scipy backend integrates all
the points together in a single pass. That path used to zero the integrand at
points it had already marked as converged, which corrupted the result: the
convergence flags are assigned *after* the integration returns, so on the next
pass the zero overwrote the converged point's own value, the difference against
its previous value un-converged it, and the loop oscillated while the
integration range doubled.

The consequences were 4-13% errors on three of the four registry priors, an
integration range that grew until `exp` overflowed, and -- because that
overflow is only a warning in an ordinary session -- no indication that
anything had gone wrong.

**These tests compare against the closed form and against the incidence of
disagreement, not against another quadrature path.** Batch-versus-scalar alone
would be a weak check: the scalar loop carries its own truncation defect (see
`test_known_broken.py`), so at some orders the two paths disagree because the
*scalar* one is wrong. Where a closed form exists, use it.
"""

import warnings

import numpy as np
import pytest

from conftest import gamma_mgf_derivative_log
from jumufraktiv.derivativeDispatch import mgfDerivative
from jumufraktiv.MGFPrior_class import MGFPrior

REGISTRY_PARAMS = {
    "gamma": {"alpha": 2.0, "beta": 3.0},
    "uniform": {"a": 0.5, "b": 2.0},
    "pareto": {"alpha": 2.5, "xi": 1.0},
    "heaviside": {"k": 1.0},
}

#: Orders spanning both sides of an integer, none of them near enough to one to
#: trigger the near-integer interpolation path.
ORDERS = [0.5, 1.5, 1.9, 2.5]


def _prior(name):
    return MGFPrior.from_registry(name, params=REGISTRY_PARAMS[name])


# ==========================================================================
# Against the closed form, where one exists
# ==========================================================================
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize(
    "t_values",
    [
        # Kept in the quick pass: this fixture detects the defect at orders
        # 1.9 and 2.5, so the fast path retains real coverage of it.
        np.array([-0.1, -1.0, -30.0]),
        # Five points cost about 31 s of the eight parametrisations. It is the
        # most sensitive fixture and is kept, but only for the full run.
        pytest.param(np.array([-1.0, -2.0, -4.0, -8.0, -16.0]), marks=pytest.mark.slow),
    ],
    ids=["spread-three", "five-points"],
)
def test_batch_matches_the_closed_form(order, t_values):
    """The Gamma MGF's derivatives are known exactly at any order.

    The failing configurations were the ones with points of very different
    magnitudes, because that is what makes some converge long before others --
    so the spread of `t` in these fixtures is doing real work and should not
    be tidied into a uniform grid.

    A third fixture, ``t = [-1, -5]``, was dropped after profiling: it cost
    17 s across the four orders and detected nothing, because two points that
    close together converge at nearly the same rate. Spread, not count, is
    what exercises the defect.
    """
    log_abs, sign = mgfDerivative(
        order, _prior("gamma"), method="scipy", t=t_values, log=True
    )
    exact = np.array([gamma_mgf_derivative_log(order, float(x)) for x in t_values])

    assert np.all(sign == 1)
    assert log_abs == pytest.approx(exact, abs=1e-12)


def test_a_single_point_and_a_batch_of_one_agree(gamma_prior):
    """A one-element array must not take a different answer from a scalar."""
    scalar = mgfDerivative(1.5, gamma_prior, method="scipy", t=-3.0, log=True)[0]
    batch = mgfDerivative(
        1.5, gamma_prior, method="scipy", t=np.array([-3.0]), log=True
    )[0]

    assert float(np.asarray(batch).item()) == pytest.approx(scalar, abs=1e-12)


# ==========================================================================
# Across the registry, where there is no closed form
# ==========================================================================
@pytest.mark.parametrize("prior_name", sorted(REGISTRY_PARAMS))
def test_all_three_routes_to_one_answer_agree(prior_name):
    """Batch, point-by-point, and batch-under-escalated-warnings must agree.

    Three assertions in one test because they share their expensive input --
    profiling showed the separate versions recomputing the same batch three
    times per prior, for about 25 s of the suite. Merged, it is also a
    stronger claim: three-way agreement rather than two pairwise checks.

    **The warning-filter arm is the one that matters most**, and it is the
    check the earlier record could not make. `pyproject.toml` sets
    ``filterwarnings = ["error"]``, so under pytest NumPy's "overflow
    encountered in exp" became an exception; the batch path aborted, fell back
    to the scalar loop, and returned the correct answer -- while an ordinary
    user, whose warnings are not escalated, got the wrong one. The suite was
    therefore structurally unable to see the defect. Asserting that the two
    warning states agree catches that whole class of problem, and keeps
    catching it: any future change that makes a result depend on the caller's
    warning configuration fails here.

    The tolerance is the accuracy the paths' own settings support -- all of
    them integrate with ``epsabs = epsrel = 1e-8``. The defect guarded against
    was a disagreement of order 1e-1, seven orders of magnitude larger.

    **The sign is compared as well as the magnitude**, because the two together
    are the return value: these backends report a result as
    ``(log_abs, sign)``, so checking only ``log_abs`` would let a route flip
    the sign of the derivative and still pass. That is not a hypothetical
    failure mode in this package -- the integer-classification defect recorded
    in `test_known_broken.py` produces a result with the wrong sign for a
    quantity that is provably positive.
    """
    prior = _prior(prior_name)
    t_values = np.array([-1.0, -5.0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        relaxed, relaxed_sign = mgfDerivative(
            1.5, prior, method="scipy", t=t_values, log=True
        )
        point_by_point = [
            mgfDerivative(1.5, prior, method="scipy", t=float(x), log=True)
            for x in t_values
        ]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        strict, strict_sign = mgfDerivative(
            1.5, prior, method="scipy", t=t_values, log=True
        )

    scalar_log = np.array([r[0] for r in point_by_point])
    scalar_sign = np.array([r[1] for r in point_by_point])

    assert relaxed == pytest.approx(strict, rel=1e-8), "depends on the warning filter"
    assert np.array_equal(relaxed_sign, strict_sign), "sign depends on warning filter"

    assert relaxed == pytest.approx(scalar_log, rel=1e-8), "batch != point-by-point"
    assert np.array_equal(relaxed_sign, scalar_sign), "sign differs between routes"

    # D^a M(t) = E[theta^a e^{t theta}] > 0 for theta > 0, so every sign here
    # is positive and any negative one is a defect rather than a disagreement.
    assert np.all(relaxed_sign == 1)


# ==========================================================================
# The integration range must not run away
# ==========================================================================
# `test_batch_does_not_overflow_on_ordinary_input` stood here. It asserted that
# nothing reaches the range where `exp` overflows, under escalated warnings so
# that an overflow would raise.
#
# It did not work. Measured by restoring the masking defect and running it
# alone: `1 passed`. The broad `except Exception` in the scipy backend catches
# the escalated warning, falls back to the point-by-point path, and returns
# finite positives -- so the assertion holds precisely when the defect is
# present. The four `test_batch_matches_the_closed_form[spread-three-*]` arms
# do catch it, so removing this loses nothing.
#
# Recorded rather than silently deleted because the shape recurs: this is the
# third instance in this audit of a check sitting downstream of the property it
# claims to test. See CLAUDE.md, "A testing hazard this repository has already
# hit twice" -- now three times, and the count is the point.


# ==========================================================================
# The tuple-vectorisation principle, asserted on cost rather than on shape
# ==========================================================================
# `CLAUDE.md` states the principle as: evaluation points are the *pair*
# `(t, u)`, broadcast to a common shape and evaluated **as one batch**. Until
# PR 9 the package satisfied that in shape and not in cost -- array `t`, array
# `u` and broadcasting all returned correctly shaped answers, while underneath
# the default route ran one adaptive quadrature per point. The measured
# signature was a cost per point that did not fall as the batch grew:
#
#     points     total    per point
#          1   255.9 ms    255.9 ms
#          5  1277.6 ms    255.5 ms
#         20  5098.2 ms    254.9 ms
#
# These tests assert the property structurally, by counting how many times the
# prior's density is called. A wall-clock assertion would be the obvious
# alternative and a bad one: it varies with the machine, so it is either flaky
# or so loose that it asserts nothing. The call count is a property of the
# algorithm.


class _CountingDensity:
    """Wraps a prior's density and records how often it is called."""

    def __init__(self, prior):
        self._prior = prior
        self.calls = 0
        self._inner = prior.logpdf_func

    def __enter__(self):
        def counted(theta):
            self.calls += 1
            return self._inner(theta)

        self._prior.logpdf_func = counted
        return self

    def __exit__(self, *exc):
        self._prior.logpdf_func = self._inner
        return False


def _density_calls(prior, points):
    from jumufraktiv.numeric_expectation import expectationDeriv

    with _CountingDensity(prior) as counter:
        expectationDeriv(1.5, prior, t=np.asarray(points), log=True)
    return counter.calls


def test_a_batch_does_not_cost_a_multiple_of_the_calls(gamma_prior):
    """Twenty points must not cost twenty times one point's density calls.

    If the implementation loops, the count scales with the number of points.
    If it batches, each quadrature node evaluates every point at once and the
    count barely moves. The threshold is deliberately loose -- a larger batch
    does need somewhat more adaptive subdivision, and bracketing is still
    per-point -- but a loop cannot pass it: the pre-PR-9 code made 20x the
    calls, and this allows 4x.
    """
    one = _density_calls(gamma_prior, [-1.0])
    twenty = _density_calls(gamma_prior, np.linspace(-1.0, -5.0, 20))

    assert twenty < 4 * one, (
        f"{twenty} density calls for 20 points against {one} for 1 point: "
        "the batch is being evaluated one point at a time"
    )


def test_the_counter_would_notice_a_loop(gamma_prior):
    """Guard the guard: the counter must actually count.

    A structural test built on instrumentation can pass because the
    instrumentation is not wired up, and that looks exactly like success. So
    call the density directly and check the count moves.
    """
    with _CountingDensity(gamma_prior) as counter:
        gamma_prior.logpdf_func(np.array([1.0]))
        gamma_prior.logpdf_func(np.array([2.0]))

    assert counter.calls == 2


@pytest.mark.parametrize("order", [0.5, 1.5, 6.0])
def test_batch_and_one_at_a_time_agree_exactly(gamma_prior, order):
    """Sharing one adaptive subdivision must not move any point's answer.

    The batched quadrature maps every point's interval onto a common [0, 1]
    and integrates the resulting vector, so all points share one subdivision.
    That is only safe because each point is scaled by its own peak first; if it
    were not, a point whose integrand is negligible against the others would be
    integrated to the batch's tolerance rather than its own.
    """
    from jumufraktiv.numeric_expectation import expectationDeriv

    points = np.array([-0.5, -1.0, -5.0, -14.0, -30.0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        batched, _ = expectationDeriv(order, gamma_prior, t=points, log=True)
        singly = np.array(
            [
                expectationDeriv(order, gamma_prior, t=float(p), log=True)[0]
                for p in points
            ]
        )

    assert batched == pytest.approx(singly, rel=1e-12, abs=1e-12)


def test_every_prior_density_accepts_an_array(prior_name=None):
    """The batched integrand hands the density a vector of theta.

    `heaviside` could not take one: its density was `1.0 if theta >= k else
    0.0`, a Python conditional, which raises for any array longer than one. It
    passed unnoticed because every caller evaluated a single point at a time.
    """
    from jumufraktiv import registry

    registry.initialize()
    thetas = np.array([0.5, 1.0, 2.0, 5.0])
    for name, params in REGISTRY_PARAMS.items():
        prior = MGFPrior.from_registry(name, params=params)
        for func_name in ("pdf_func", "logpdf_func"):
            values = np.asarray(getattr(prior, func_name)(thetas), dtype=float)
            assert values.shape == thetas.shape, (
                f"{name}.{func_name} did not return one value per theta"
            )


# ==========================================================================
# A caller-supplied density may be written for scalars
# ==========================================================================
# The registry's four priors all take arrays. A density a caller writes need
# not, and the batched integrand hands it a vector. Deciding what to do about
# that per call site produced two failure modes, both measured:
#
#   * the answer depended on the batch size -- a Python-conditional density
#     returned 0.28379634 for one evaluation point and raised ValueError for
#     three, because a one-element array happens to satisfy `if`;
#   * a real failure became a plausible number -- a `math`-module density
#     returned -inf where its vectorised twin returns -1.4481850809269488,
#     because the per-call fallback caught the TypeError and substituted -inf.
#
# `_vectorise` settles it once, at setup, by probing. These tests pin both
# halves: the same density written three ways must give the same answer, and a
# density that is genuinely broken must raise rather than become a number.


def _custom_prior(logpdf):
    import sympy as sp

    from jumufraktiv.symbols import theta

    return MGFPrior(
        name="custom",
        pdf_sym=sp.exp(-theta),
        logpdf_func=logpdf,
        pdf_func=lambda x: np.exp(logpdf(x)),
        params={},
    )


#: One density -- log p(theta) = -theta on theta > 0 -- written three ways.
#: Only the first accepts an array of any length.
DENSITY_WRITINGS = {
    "vectorised": lambda x: -np.asarray(x, dtype=float),
    "python-conditional": lambda x: -float(x) if x >= 0.0 else -np.inf,
    "math-module-scalar": lambda x: -float(x),
}


@pytest.mark.parametrize("writing", sorted(DENSITY_WRITINGS))
@pytest.mark.parametrize("n_points", [1, 3])
def test_a_scalar_only_density_gives_the_same_answer(writing, n_points):
    """Same density, same answer -- whichever way it is written, however many
    points are asked for."""
    from jumufraktiv.numeric_expectation import expectationDeriv

    points = np.linspace(-1.0, -3.0, n_points)
    reference = np.ravel(
        expectationDeriv(
            1.5, _custom_prior(DENSITY_WRITINGS["vectorised"]), t=points, log=True
        )[0]
    )
    got = np.ravel(
        expectationDeriv(
            1.5, _custom_prior(DENSITY_WRITINGS[writing]), t=points, log=True
        )[0]
    )

    assert got == pytest.approx(reference, rel=1e-12, abs=1e-12)


def test_a_broken_density_raises_rather_than_returning_minus_infinity():
    """The failure mode that matters most, because -inf reads as an answer.

    `-inf` is what the route returns when the integrand genuinely has no mass,
    so a density that fails and is reported as `-inf` is indistinguishable from
    one that worked. The caller's own exception must survive.
    """
    from jumufraktiv.numeric_expectation import expectationDeriv

    def broken(theta):
        raise RuntimeError("the caller's density has a bug")

    with pytest.raises(RuntimeError, match="has a bug"):
        expectationDeriv(1.5, _custom_prior(broken), t=-1.0, log=True)


def test_a_density_returning_two_values_for_one_theta_is_refused():
    """The elementwise adapter must not pick one and carry on.

    `np.ravel(...)[0]` would integrate a density that is answering a different
    question, and say nothing about it. The density here refuses arrays, so the
    elementwise path is the one chosen, and then returns two values for a
    single theta.
    """
    import sympy as sp

    from jumufraktiv.numeric_expectation import expectationDeriv
    from jumufraktiv.symbols import theta

    def two_values(x):
        if isinstance(x, np.ndarray) and x.size > 1:
            raise TypeError("this density is scalar-only")
        return np.array([-1.0, -2.0])

    prior = MGFPrior(
        name="two-valued",
        pdf_sym=sp.exp(-theta),
        logpdf_func=two_values,
        pdf_func=lambda x: np.exp(two_values(x)),
        params={},
    )

    with pytest.raises(ValueError, match="returned 2 values"):
        expectationDeriv(1.5, prior, t=-1.0, log=True)


def test_the_batched_quadrature_does_not_depend_on_the_warning_filter(gamma_prior):
    """`filterwarnings = ["error"]` must not change what the code computes.

    `CLAUDE.md` records this as a hazard the repository has already hit:
    NumPy's "overflow encountered in exp" is an exception under pytest and a
    warning everywhere else, so a path that overflows takes one branch in the
    suite and another in a user's session. The suite got the right answer and
    users got a wrong one.

    `batched()` calls `np.exp(exponent - offsets)`, which can overflow when the
    peak search underestimates the offset. Asserting the two filters agree is
    what makes the suite's verdict transferable.
    """
    from jumufraktiv.numeric_expectation import expectationDeriv

    points = np.linspace(-1.0, -8.0, 6)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        strict, _ = expectationDeriv(2.5, gamma_prior, t=points, log=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lenient, _ = expectationDeriv(2.5, gamma_prior, t=points, log=True)

    assert np.array_equal(strict, lenient)
