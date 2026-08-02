"""Asking for several derivative orders at once must match asking one at a time.

`mgfDerivative` accepts an array of orders and dispatches each element
separately. That block used to coerce every element with `int()`, so a
fractional order silently returned the answer for a different derivative --
68% wrong at order 0.5, 35% at 1.5, 61% at 1.9 and 15% at 2.5, measured
against the closed form at `t = -2`.

**The defect was the coercion, not the choice of rounding rule.** `int(2.5)`
and `round(2.5)` are both 2, and the answer is 15% wrong either way.
Truncation versus rounding is only a sub-case, biting near a whole number
where `int(1.9) = 1` but `round(1.9) = 2`.

The same block also forced `t` and `u` through `float()`, so an array order
could not produce a symbolic result, and reassembled its answers as a flat
list, losing the caller's shape.

**Sample-size parity decides which method is affected, so fixtures here must
choose `n` deliberately.** For a Normal likelihood the aggregate order is
`a = n/2` while the per-observation order is `a = 1/2`. So `post_predictive`,
which passes the per-observation orders, is wrong for *even* `n`, while
`post_raw_moment` and `post_central_moment`, which need a fractional aggregate
`a`, are wrong for *odd* `n`. Every sample size is wrong in one of the two,
and a fixture that picks `n` carelessly asserts nothing at all.
"""

import numpy as np
import pytest
import sympy as sp

from conftest import gamma_mgf_derivative_log
from jumufraktiv.derivativeDispatch import mgfDerivative
from jumufraktiv.MGFDerivative_class import MGFDerivative


# ==========================================================================
# The dispatcher
# ==========================================================================
@pytest.mark.parametrize("order", [0.5, 1.5, 1.9, 2.5])
def test_array_order_matches_the_closed_form(gamma_prior, order):
    """Each element must get the derivative it asked for.

    Checked one order per test rather than one array per test, so a failure
    names the order that broke rather than the whole batch.
    """
    orders = np.array([order])

    log_abs, sign = mgfDerivative(orders, gamma_prior, method="auto", t=-2.0, log=True)

    assert sign[0] == 1
    assert log_abs[0] == pytest.approx(gamma_mgf_derivative_log(order, -2.0), rel=1e-10)


def test_array_order_agrees_with_looping_over_scalars(gamma_prior):
    """Vectorising must be a convenience, never a different computation."""
    orders = np.array([1.0, 1.5, 2.0, 2.5])

    batch_log, batch_sign = mgfDerivative(
        orders, gamma_prior, method="auto", t=-1.0, log=True
    )
    scalar = [
        mgfDerivative(float(o), gamma_prior, method="auto", t=-1.0, log=True)
        for o in orders
    ]

    assert batch_log == pytest.approx(np.array([r[0] for r in scalar]), rel=1e-10)
    assert np.array_equal(batch_sign, np.array([r[1] for r in scalar]))


def test_the_shape_of_the_request_is_the_shape_of_the_answer(gamma_prior):
    """Results were reassembled flat, so a 2-D request came back 1-D."""
    orders = np.array([[0.5, 1.5], [1.9, 2.5]])

    log_abs, sign = mgfDerivative(orders, gamma_prior, method="auto", t=-2.0, log=True)

    assert log_abs.shape == (2, 2)
    assert sign.shape == (2, 2)

    expected = np.array(
        [[gamma_mgf_derivative_log(float(o), -2.0) for o in row] for row in orders]
    )
    assert log_abs == pytest.approx(expected, rel=1e-10)


def test_an_array_order_can_still_be_symbolic(gamma_prior):
    """The symbol-numeric principle: `t=None` means an expression, not an error.

    `float(t)` raised `TypeError` here, so the return type depended on how the
    request was spelled rather than on whether unresolved symbols remained.
    """
    result = mgfDerivative(np.array([1.0, 2.0]), gamma_prior, method="symbolic", t=None)

    assert np.shape(result) == (2,)
    assert all(isinstance(x, sp.Basic) for x in np.ravel(result))


def test_array_order_broadcasts_against_array_t(gamma_prior):
    """The tuple-vectorisation principle: order and `t` broadcast together."""
    orders = np.array([0.5, 1.5, 2.5])
    t_values = np.array([-1.0, -2.0, -3.0])

    log_abs, _ = mgfDerivative(orders, gamma_prior, method="auto", t=t_values, log=True)
    expected = np.array(
        [
            gamma_mgf_derivative_log(float(o), float(x))
            for o, x in zip(orders, t_values, strict=True)
        ]
    )

    assert log_abs == pytest.approx(expected, rel=1e-10)


# ==========================================================================
# Through the public interface, where parity decides what breaks
# ==========================================================================
@pytest.mark.parametrize("n", [3, 5])
def test_zeroth_raw_moment_is_one(gamma_prior, n):
    """`E[Theta^0] = 1` for any distribution, so this needs no reference.

    Odd `n` gives a fractional aggregate order (`a = n/2`), which is the case
    the moment methods got wrong. Before the fix these came back as 1.903 at
    `n = 3`; the whole-number case was always correct, which is why the defect
    only surfaced once fractional posteriors became constructible.
    """
    post = MGFDerivative(gamma_prior, data=[1.0] * n, likelihood="halfnormal")

    moments = np.asarray(post.post_raw_moment([0, 1, 2], log=False), dtype=float)

    assert moments[0] == pytest.approx(1.0, rel=1e-10)


@pytest.mark.parametrize("n", [3, 4])
def test_moments_agree_whether_asked_together_or_singly(gamma_prior, n):
    """Both parities, because only one of them exercised the defect."""
    post = MGFDerivative(gamma_prior, data=[1.0] * n, likelihood="halfnormal")

    together = np.asarray(post.post_raw_moment([0, 1, 2], log=False), dtype=float)
    singly = np.array(
        [float(np.ravel(post.post_raw_moment(q, log=False))[0]) for q in (0, 1, 2)]
    )

    assert together == pytest.approx(singly, rel=1e-8)


@pytest.mark.parametrize("n", [2, 3])
def test_posterior_predictive_agrees_with_point_by_point(gamma_prior, n):
    """`post_predictive` passes the per-observation orders, so even `n` broke it.

    Both parities are here for the same reason as above: at `n = 3` the
    predictive was already exact, so a test using only odd sample sizes would
    have asserted nothing about this path.
    """
    post = MGFDerivative(gamma_prior, data=[1.0] * n, likelihood="halfnormal")
    new = [0.5, 1.5]

    together = np.asarray(post.post_predictive(new), dtype=float).ravel()
    singly = np.array([float(np.ravel(post.post_predictive([y]))[0]) for y in new])

    assert together == pytest.approx(singly, rel=1e-8)


# ==========================================================================
# Batching over orders (PR 14c)
# ==========================================================================
def test_an_array_of_orders_is_one_quadrature_not_one_per_order():
    """The cost claim, asserted structurally rather than by wall clock.

    Counting the prior's density calls is a property of the algorithm; a
    timing threshold is a property of the machine, so it would be either
    flaky or too loose to assert anything. This is the same reasoning
    `tests/test_batch_evaluation.py` uses for evaluation points.

    Eight orders dispatched one at a time ran eight independent adaptive
    quadratures, each reaching the density separately -- 4968 `logpdf` calls,
    73% of the runtime, for a term that does not depend on the order at all.
    """
    import warnings

    from jumufraktiv import registry
    from jumufraktiv.derivativeDispatch import mgfDerivative
    from jumufraktiv.MGFPrior_class import MGFPrior

    registry.initialize()
    prior = MGFPrior.from_registry("gamma", params={"alpha": 2.0, "beta": 3.0})

    calls = {"n": 0}
    original = prior.logpdf_func

    def counting(theta):
        calls["n"] += 1
        return original(theta)

    prior.logpdf_func = counting

    orders = np.linspace(0.5, 2.5, 8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mgfDerivative(orders, prior, method="auto", t=-1.0, log=True)
    batched = calls["n"]

    calls["n"] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in orders:
            mgfDerivative(float(order), prior, method="auto", t=-1.0, log=True)
    per_element = calls["n"]

    # Not a factor of eight: the bracketing and peak search stay per order,
    # because both depend on it, and only the quadrature batches. Two is a
    # generous floor that a per-element regression could not sneak under.
    assert batched * 2 < per_element, (
        f"batched {batched} density calls against {per_element} per-element; "
        "the array-order path looks unbatched again"
    )


@pytest.mark.parametrize("t_value", [-0.5, -1.0, -3.0])
def test_batched_orders_agree_with_the_closed_form(t_value):
    """Batching may only change the cost, never the answer.

    The reference is the Gamma MGF's derivative in closed form:
    `E[Theta^a e^{t Theta}] = Gamma(alpha + a)/Gamma(alpha) *
    beta^alpha / (beta - t)^(alpha + a)`, written out here rather than taken
    from the package.
    """
    import warnings

    import mpmath as mp

    from jumufraktiv import registry
    from jumufraktiv.derivativeDispatch import mgfDerivative
    from jumufraktiv.MGFPrior_class import MGFPrior

    registry.initialize()
    alpha, beta = 2.0, 3.0
    prior = MGFPrior.from_registry("gamma", params={"alpha": alpha, "beta": beta})

    orders = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_abs, signs = mgfDerivative(
            orders, prior, method="auto", t=t_value, log=True
        )

    assert log_abs.shape == orders.shape
    assert np.all(signs == 1)

    with mp.workdps(40):
        for index, order in enumerate(orders):
            a, al, be, tt = (
                mp.mpf(float(order)),
                mp.mpf(alpha),
                mp.mpf(beta),
                mp.mpf(t_value),
            )
            exact = mp.gamma(al + a) / mp.gamma(al) * be**al / (be - tt) ** (al + a)
            got = mp.e ** mp.mpf(float(log_abs[index]))
            assert float(abs(got - exact) / exact) < 1e-12, order


def test_a_mixed_integer_and_fractional_array_is_still_one_call():
    """Mixed order types do not mean mixed backends, which is the surprise.

    The condition sending a request to the expectation route does not mention
    the order at all -- with `auto` and a concrete `t` every element takes it,
    whatever its type. So the grouping-by-backend this looked like it needed
    turns out to be unnecessary, and an array of `[1, 1.5, 2, 2.5]` batches as
    readily as a uniform one.
    """
    import warnings

    from jumufraktiv import registry
    from jumufraktiv.derivativeDispatch import mgfDerivative
    from jumufraktiv.MGFPrior_class import MGFPrior

    registry.initialize()
    prior = MGFPrior.from_registry("gamma", params={"alpha": 2.0, "beta": 3.0})

    mixed = np.array([1.0, 1.5, 2.0, 2.5])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_abs, _ = mgfDerivative(mixed, prior, method="auto", t=-1.0, log=True)
        singles = [
            mgfDerivative(float(o), prior, method="auto", t=-1.0, log=True)[0]
            for o in mixed
        ]

    assert log_abs.shape == mixed.shape
    for index in range(mixed.size):
        assert float(log_abs[index]) == pytest.approx(float(singles[index]), rel=1e-9)


@pytest.mark.parametrize(
    ("label", "t_values", "expected"),
    [
        ("numeric t", np.array([-1.0, -2.0]), True),
        ("t is None", np.array([None, None], dtype=object), False),
        ("symbolic t", None, False),  # filled in below; needs the symbol
    ],
)
def test_only_a_fully_numeric_request_is_batched(
    gamma_prior, label, t_values, expected
):
    """The batch path must decline anything it cannot evaluate as a number.

    A SymPy `Symbol` for `t` is not `None`, so a condition testing only
    `t is not None` admits it -- and it then fails inside `np.asarray` as
    `TypeError: Cannot convert expression to float`. A symbolic `t` is
    unsupported package-wide and always has been, a scalar order raising
    identically; the requirement here is only that this fast path not become a
    new way to reach it.

    Asserted on the predicate directly, which is why it was extracted. The
    batched route cannot be observed from outside: the call site passes
    `np.asarray(t_arr, dtype=float)`, and an argument is evaluated before the
    call it belongs to, so a symbolic `t` raises without the route ever being
    entered. Two earlier versions of this test -- one comparing exception
    types, one watching for the route -- passed identically with and without
    the guard, which is how that was found.
    """
    from jumufraktiv.derivativeDispatch import _array_orders_can_batch
    from jumufraktiv.symbols import t as t_sym

    if t_values is None:
        t_values = np.array([t_sym, t_sym], dtype=object)

    orders = np.array([1.0, 2.0])
    assert (
        _array_orders_can_batch(
            orders,
            t_values,
            None,
            complete=True,
            method="auto",
            prior=gamma_prior,
        )
        is expected
    ), label


def test_a_symbolic_order_is_never_batched(gamma_prior):
    """A symbolic order is refused package-wide, so it must not reach here."""
    from jumufraktiv.derivativeDispatch import _array_orders_can_batch

    orders = np.array([sp.Symbol("n"), 2.0], dtype=object)

    assert not _array_orders_can_batch(
        orders,
        np.array([-1.0, -1.0]),
        np.array([None, None], dtype=object),
        complete=True,
        method="auto",
        prior=gamma_prior,
    )


@pytest.mark.parametrize("method", ["symbolic", "scipy", "mpmath", "bell", "jax"])
def test_an_explicit_backend_is_never_batched(gamma_prior, method):
    """Only `auto` and `expectation` reach this route, so only they may batch.

    An explicit `method=` is never reinterpreted anywhere in the package, and
    silently batching one through a different backend would be exactly that.
    """
    from jumufraktiv.derivativeDispatch import _array_orders_can_batch

    assert not _array_orders_can_batch(
        np.array([1.0, 2.0]),
        np.array([-1.0, -1.0]),
        np.array([None, None], dtype=object),
        complete=True,
        method=method,
        prior=gamma_prior,
    )


def test_t_none_still_returns_expressions_for_an_array_order(gamma_prior):
    """The documented symbolic route must survive the batching.

    `t=None` is how a caller asks for an expression, and the fast path has to
    decline it for the symbol-numeric principle to hold.
    """
    from jumufraktiv.derivativeDispatch import mgfDerivative

    result = mgfDerivative(np.array([1.0, 2.0]), gamma_prior, method="symbolic", t=None)

    assert np.shape(result) == (2,)
    assert all(isinstance(x, sp.Basic) for x in np.ravel(result))
