"""The fixed-grid kernel, measured against the closed form.

Six recorded defects were symptoms of one design fault in the previous
quadrature: an adaptive scheme whose truncation range was discovered by
doubling, whose stopping rule compared consecutive iterates, and which
accumulated in linear space and clamped on overflow. This file measures what
replaced it.

The Gamma prior's MGF has a closed form and so do all of its fractional
derivatives, so every assertion here compares against an exact value rather
than a recorded one.
"""

import numpy as np
import pytest
from conftest import gamma_mgf_derivative_log

from jumufraktiv.derivativeDispatch import mgfDerivative

#: Cases the previous kernel got wrong, with its measured relative error.
#: Keeping the old figure next to each case is what makes the improvement
#: legible rather than a bare tolerance.
PREVIOUSLY_WRONG = [
    pytest.param(2.5, -5.0, 3.6e-10, id="order2.5-t-5"),
    pytest.param(4.5, -14.0, 2.9e-06, id="order4.5-t-14"),
    pytest.param(4.5, -30.0, 1.5e-06, id="order4.5-t-30"),
    pytest.param(1.5, -50.0, 1.2e-06, id="order1.5-t-50"),
    pytest.param(1.99, -1.0, 0.96, id="order1.99"),
    pytest.param(1.999, -1.0, 0.96, id="order1.999"),
]


def _relative_error(order, t):
    log_abs, sign = mgfDerivative(order, _prior(), method="scipy", t=t, log=True)
    exact = gamma_mgf_derivative_log(order, t)
    got = float(np.ravel(log_abs)[0])
    assert int(np.ravel(sign)[0]) == 1, "E[theta^a e^{t theta}] is positive"
    return abs(got - exact) / abs(exact)


def _prior():
    from jumufraktiv import registry
    from jumufraktiv.mitMGFprior_class import mitMGFprior

    registry.initialize()
    return mitMGFprior.from_registry("gamma", params={"alpha": 2.0, "beta": 3.0})


@pytest.mark.parametrize("order,t,previous_error", PREVIOUSLY_WRONG)
@pytest.mark.slow
def test_cases_the_old_kernel_got_wrong(order, t, previous_error):
    """Each of these was a recorded defect; all now reach machine precision.

    `order=4.5, t=-30` is the one to look at. It was recorded as plateauing at
    2.1e-10 *under any* `tol`, because the stopping rule compared consecutive
    iterates and that underestimates the remaining tail when convergence is
    slow. Tightening `tol` made the loop widen for longer without making the
    rule correct. Deriving the range removes the rule instead of tuning it.
    """
    error = _relative_error(order, t)

    assert error < 1e-12
    assert error < previous_error


@pytest.mark.parametrize("order", [0.5, 1.5, 1.9, 2.5])
@pytest.mark.slow
def test_cases_the_old_kernel_got_right_are_unchanged(order):
    """Replacing a kernel must not cost accuracy anywhere it was already fine."""
    assert _relative_error(order, -1.0) < 1e-12


@pytest.mark.parametrize("order", [150.5, 300.5, 500.5])
@pytest.mark.slow
def test_large_orders_do_not_overflow(order):
    """Order 300.5 returned 694.234 against an exact 1006.311 -- 312 nats.

    The recorded defect attributes this to the quadrature accumulating in
    linear space and clamping on overflow. That is only half of it. Fixing the
    accumulation alone still left order 300.5 wrong, because
    `mgfDerivative_integer` returned `inf` for `M^(301)` -- the SymPy value is
    finite and it is the cast to a Python float that overflows. Both had to be
    repaired, and `M^(151)` being exact to 1.4e-16 is what makes the second one
    easy to miss: the inner derivative looks healthy right up until it isn't.

    `a = sum(y)` for several likelihoods, so the order grows with the sample
    and these are reachable from ordinary data.
    """
    assert _relative_error(order, -1.0) < 1e-12


@pytest.mark.slow
def test_the_log_argument_decides_the_return_shape_and_nothing_else():
    """The log principle, on the path that used to break it.

    `numeric_fractionalDeriv_interpolation` bound `result` only inside its
    array branch, so `log=False` raised `UnboundLocalError` on the scalar path
    while `log=True` returned a value. That module is now retired, so the
    near-integer orders that used to reach it go through the same kernel as
    everything else.
    """
    prior = _prior()

    for order in (1.96, 1.99, 1.999):
        value = mgfDerivative(order, prior, method="scipy", t=-1.0, log=False)
        log_abs, sign = mgfDerivative(order, prior, method="scipy", t=-1.0, log=True)

        assert np.isfinite(float(np.ravel(value)[0]))
        assert float(np.ravel(value)[0]) == pytest.approx(
            float(np.ravel(sign)[0]) * np.exp(float(np.ravel(log_abs)[0])), rel=1e-10
        )


@pytest.mark.slow
def test_the_near_integer_threshold_is_no_longer_a_cliff():
    """Accuracy must not jump at the old interpolation trigger.

    The dispatcher used to switch to a spline in the order whenever the
    fractional part exceeded 0.95, and the spline was *less* accurate than the
    quadrature just below the threshold. Straddling it is what makes the
    discontinuity visible; a test on either side alone would not.
    """
    errors = [_relative_error(order, -1.0) for order in (1.94, 1.95, 1.96, 1.97)]

    assert max(errors) < 1e-12


@pytest.mark.slow
def test_batch_and_scalar_evaluation_agree():
    """The tuple-vectorisation principle: one batched call, same answers."""
    prior = _prior()
    points = np.array([-1.0, -5.0, -14.0, -30.0])

    batched, _ = mgfDerivative(2.5, prior, method="scipy", t=points, log=True)
    one_by_one = [
        float(np.ravel(mgfDerivative(2.5, prior, method="scipy", t=p, log=True)[0])[0])
        for p in points
    ]

    assert np.asarray(batched).ravel() == pytest.approx(one_by_one, rel=1e-12)
