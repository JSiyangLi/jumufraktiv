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
