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
from jumufraktiv.mitMGFprior_class import mitMGFprior

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
    return mitMGFprior.from_registry(name, params=REGISTRY_PARAMS[name])


# ==========================================================================
# Against the closed form, where one exists
# ==========================================================================
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize(
    "t_values",
    [
        np.array([-0.1, -1.0, -30.0]),
        np.array([-1.0, -2.0, -4.0, -8.0, -16.0]),
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
def test_batch_does_not_overflow_on_ordinary_input(gamma_prior):
    """Nothing here should reach the range where `exp` overflows.

    The overflow was a symptom of the same defect -- the loop never settled, so
    the range kept doubling past the integrand's support. With warnings
    escalated an overflow raises, so this asserts the absence of one.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        log_abs, sign = mgfDerivative(
            1.5,
            gamma_prior,
            method="scipy",
            t=np.array([-1.0, -5.0, -20.0]),
            log=True,
        )

    assert np.all(np.isfinite(log_abs))
    assert np.all(sign == 1)
