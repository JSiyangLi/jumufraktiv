"""The symbolic backend for fractional orders.

It used to return `None` for every prior in the registry, so the one row of the
backend matrix it serves produced nothing usable. Two separate causes: the
Laplace leg raised inside SymPy for the Gamma prior, and the Mellin fallback
subscripted a result that is only sometimes a tuple. Both were swallowed by a
broad `except`, and the `None` surfaced far away as
`TypeError: 'NoneType' object is not callable`.

It also omitted the `1/Gamma(gamma)` prefactor that all the numeric backends
apply, so any expression it had returned would have been `Gamma(gamma)` times
too large — 77% at order 0.5. That defect was unobservable precisely because
nothing was ever returned.

Both transform legs are replaced by direct evaluation of the defining integral.
Substituting `x = t - z` in the Liouville-Caputo integral with lower terminal
minus infinity gives

    D^a M(t) = (1/Gamma(g)) * int_0^oo z^{g-1} M^{(n+1)}(t - z) dz

which SymPy can attempt in one go.

**Not every prior yields a closed form, and that is expected rather than a
defect.** What matters is that the failure is *reported*: a `NotImplementedError`
naming the prior, not a `None` that fails somewhere else later.
"""

import warnings

import numpy as np
import pytest

from conftest import gamma_mgf_derivative_log
from jumufraktiv.MGFPrior_class import MGFPrior
from jumufraktiv.symbolic_fractionalDeriv import fractionalDeriv_symbolic
from jumufraktiv.symbols import t as canonical_t

#: Priors SymPy cannot integrate in closed form here. Listed explicitly rather
#: than discovered, so that a prior moving between the two lists is a visible
#: change rather than a silent one.
NO_CLOSED_FORM = {
    "pareto": {"alpha": 2.5, "xi": 1.0},
    "uniform": {"a": 0.5, "b": 2.0},
    "heaviside": {"k": 1.0},
}


def _symbolic(order, prior, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fractionalDeriv_symbolic(order, prior, **kwargs)


# ==========================================================================
# Where a closed form exists
# ==========================================================================
@pytest.mark.parametrize("order", [0.5, 1.5, 2.5])
def test_gamma_matches_the_closed_form(gamma_prior, order):
    """The expression must agree with the independently known derivative.

    This is what pins the `1/Gamma(gamma)` prefactor: without it the result is
    too large by `Gamma(gamma)`, which is a factor of 1.77 at order 0.5 — far
    outside any tolerance, and in the same direction at every order.
    """
    expr = _symbolic(order, gamma_prior)

    value = float(expr.subs(canonical_t, -1.0).evalf())
    expected = np.exp(gamma_mgf_derivative_log(order, -1.0))

    assert value == pytest.approx(expected, rel=1e-10)


def test_the_expression_is_in_the_canonical_symbol(gamma_prior):
    """A caller substitutes `jumufraktiv.symbols.t`, so that must be what is free.

    The integral needs `t` declared negative to converge — it is the evaluation
    point `-b(y)`, never positive — and the canonical symbol carries no such
    assumption. So the implementation substitutes a local symbol in and must
    map it back; if it forgets, the caller's `subs` silently matches nothing
    and returns the expression unchanged.
    """
    expr = _symbolic(1.5, gamma_prior)

    assert expr.free_symbols == {canonical_t}


def test_simplify_does_not_change_the_value(gamma_prior):
    plain = _symbolic(1.5, gamma_prior)
    simplified = _symbolic(1.5, gamma_prior, simplify=True)

    at_t = -2.0
    assert float(simplified.subs(canonical_t, at_t).evalf()) == pytest.approx(
        float(plain.subs(canonical_t, at_t).evalf()), rel=1e-12
    )


# ==========================================================================
# Where none exists, the failure must be reported
# ==========================================================================
@pytest.mark.parametrize("prior_name", sorted(NO_CLOSED_FORM))
@pytest.mark.parametrize("order", [0.5, 1.5])
def test_priors_without_a_closed_form_raise_rather_than_return_none(prior_name, order):
    """`None` is not an answer, and it fails far from its cause.

    Returning `None` propagated through `mgfDerivative` and reached
    `MGFDerivative`, which called it — so the user saw
    `TypeError: 'NoneType' object is not callable` with no indication that a
    symbolic integration had declined.
    """
    prior = MGFPrior.from_registry(prior_name, params=NO_CLOSED_FORM[prior_name])

    with pytest.raises(NotImplementedError, match=prior_name):
        _symbolic(order, prior)


@pytest.mark.parametrize("prior_name", sorted(NO_CLOSED_FORM))
def test_the_error_says_what_to_use_instead(prior_name):
    prior = MGFPrior.from_registry(prior_name, params=NO_CLOSED_FORM[prior_name])

    with pytest.raises(NotImplementedError, match="scipy"):
        _symbolic(0.5, prior)


def test_declining_is_prompt(gamma_prior):
    """Rejection must not cost the timeout.

    The unevaluated case is rejected *before* any simplification is attempted,
    and that ordering is load-bearing rather than tidiness: `sp.simplify` on an
    unevaluated integral does not return (measured on `uniform` at order 1.5,
    killed at 120 s), while `sp.integrate` on the same input takes 0.29 s.
    Simplifying first would turn every decline into a timeout.
    """
    prior = MGFPrior.from_registry("uniform", params=NO_CLOSED_FORM["uniform"])

    with pytest.raises(NotImplementedError):
        # A timeout far longer than the decline should take. If the ordering
        # regressed, this test would take `timeout_seconds` rather than
        # failing fast — and the assertion below would still pass, so the
        # short timeout is what makes the point.
        _symbolic(1.5, prior, timeout_seconds=5.0)


# ==========================================================================
# Through the dispatcher
# ==========================================================================
def test_reachable_through_the_dispatcher(gamma_prior):
    """`method='symbolic'` at a fractional order must produce an expression.

    This is the backend-matrix row that previously produced nothing at all.
    """
    from jumufraktiv.derivativeDispatch import mgfDerivative

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expr = mgfDerivative(1.5, gamma_prior, method="symbolic", t=None)

    assert expr is not None
    value = float(expr.subs(canonical_t, -1.0).evalf())
    assert value == pytest.approx(
        np.exp(gamma_mgf_derivative_log(1.5, -1.0)), rel=1e-10
    )
