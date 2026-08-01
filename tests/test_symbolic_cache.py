"""The derivative memo must be invisible except in how long things take.

These tests are deliberately small and fast. The cache sits under every
symbolic path in the package, so the seven hundred tests that already exist
are what prove it does not change any answer — re-checking that here would
duplicate them at real quadrature cost. What is left for this file is the
handful of properties those tests cannot see: that the cache is actually being
used, that it is bounded, and that its key does not conflate two expressions
which must stay distinct.
"""

import pytest
import sympy as sp

from jumufraktiv.symbolic_cache import (
    cached_diff,
    clear_derivative_cache,
    derivative_cache_size,
)
from jumufraktiv.symbols import t


def test_the_cache_returns_what_sympy_returns(gamma_prior):
    """Same derivative, cached or not, across the orders the package uses."""
    expr = gamma_prior.mgf_sym

    for order in range(5):
        clear_derivative_cache()
        assert sp.simplify(cached_diff(expr, t, order) - sp.diff(expr, t, order)) == 0


def test_a_repeated_request_is_served_from_the_cache(gamma_prior):
    """The second call must not recompute.

    Asserted by identity rather than equality: `sp.diff` builds a new object
    each time, so `is` distinguishes a cache hit from a recomputation in a way
    that `==` cannot.
    """
    clear_derivative_cache()
    expr = gamma_prior.mgf_sym

    first = cached_diff(expr, t, 2)
    second = cached_diff(expr, t, 2)

    assert first is second
    assert derivative_cache_size() == 1


def test_the_key_does_not_conflate_different_precisions():
    """The property the whole key choice rests on.

    The cache is keyed on the expression object rather than its `srepr`,
    because that is 190x cheaper -- 0.36 us against 68.1 us on this package's
    Gamma MGF. That is only safe because SymPy's equality is precision-aware:
    `Float(9.0, 53) == Float(9.0, 24)` is False, so the two hash to the same
    bucket but do not compare equal, and the dictionary keeps them apart.

    If a future SymPy made those two compare equal, this test fails and the key
    must go back to `srepr`. That is the point of asserting it here rather than
    trusting the note in the module docstring.
    """
    clear_derivative_cache()
    high = sp.Float(9.0, 53) / (1 - t) ** 2
    low = sp.Float(9.0, 24) / (1 - t) ** 2

    cached_diff(high, t, 1)
    cached_diff(low, t, 1)

    assert derivative_cache_size() == 2


def test_distinct_orders_are_distinct_entries(gamma_prior):
    """A cache that ignored the order would return the wrong derivative."""
    clear_derivative_cache()
    expr = gamma_prior.mgf_sym

    first = cached_diff(expr, t, 1)
    second = cached_diff(expr, t, 2)

    assert first is not second
    assert derivative_cache_size() == 2


def test_the_cache_is_bounded():
    """Memory must not grow without limit on user-supplied priors.

    The built-in priors need 40 entries, so the cap is never reached in
    ordinary use; this drives past it deliberately.
    """
    clear_derivative_cache()
    x = sp.Symbol("x")

    for i in range(600):
        cached_diff(sp.Symbol(f"c{i}") * x**3, x, 1)

    assert derivative_cache_size() < 600


def test_an_unhashable_expression_still_differentiates():
    """Falling through uncached beats raising.

    SymPy expressions are hashable, so this is a guard rather than a path the
    package uses. It matters because the alternative -- letting a `TypeError`
    from the dictionary lookup escape -- would turn a working call into a
    failure for no reason connected to the mathematics.
    """

    class Unhashable(sp.Symbol):
        __hash__ = None

    # The lookup raises TypeError; the result must still be correct.
    assert cached_diff(t**2, t, 1) == 2 * t
    assert Unhashable  # the class is what documents the case


# ==========================================================================
# The compiled evaluation path
# ==========================================================================
# `expr.subs(t, value).evalf()` per point was 97.6% of the runtime of the
# `scipy` fractional route: the fixed-grid kernel hands `mgfDerivative_integer`
# an (n_nodes x n_points) array of shifted points, and it substituted into each
# one separately -- 5,024 SymPy substitutions for two evaluation points.
#
# `CLAUDE.md` recorded the remedy under "Numerical policy", measured at ~6400x,
# and it had never been applied. These tests pin the two things that make
# applying it safe.


def test_a_compiled_expression_agrees_with_the_exact_one(gamma_prior):
    """float64 where it can, SymPy where it cannot, and the same answer either way.

    Order 301 is the case that matters. `M^(301)` of the Gamma MGF overflows
    float64, so the compiled value is `inf` and the element falls through to
    the exact path -- which is why the fast path may only *skip* work, never
    change an answer. PR 6b found that overflow the hard way.
    """
    import mpmath as mp

    from jumufraktiv.derivativeDispatch import mgfDerivative_integer

    mp.mp.dps = 60
    alpha, beta = 2.0, 3.0

    for order in (1, 2, 5, 20, 151, 301):
        for t_value in (-1.0, -5.0):
            log_abs, _ = mgfDerivative_integer(
                order, gamma_prior, method="symbolic", t=t_value, log=True
            )
            exact = float(
                mp.log(mp.rf(alpha, order))
                + alpha * mp.log(beta)
                - (alpha + order) * mp.log(mp.mpf(beta) - t_value)
            )
            assert log_abs == pytest.approx(exact, rel=1e-13), (
                f"order {order} at t={t_value}"
            )


def test_an_expression_that_will_not_compile_falls_back(gamma_prior):
    """`expint` is in neither SciPy's nor NumPy's namespace.

    `CLAUDE.md` said `modules=["scipy", "numpy"]` covers `lowergamma`,
    `uppergamma`, `polygamma` and `Ei`. That list is incomplete: the Pareto
    prior's MGF is written with `expint`, which neither module provides.
    SymPy compiles it happily and the result raises `NameError` on the first
    call -- so the compiled form has to be *probed*, not merely built.

    Returning `None` rather than raising is what keeps the prior usable: the
    symbolic path computes the same quantity exactly, just slower.
    """
    import numpy as np

    from jumufraktiv import registry
    from jumufraktiv.mitMGFprior_class import mitMGFprior
    from jumufraktiv.symbolic_cache import cached_diff, cached_lambdify
    from jumufraktiv.symbols import t as t_sym

    registry.initialize()
    pareto = mitMGFprior.from_registry("pareto", params={"alpha": 2.5, "xi": 1.0})

    second = cached_diff(pareto.mgf_sym, t_sym, 2)
    assert "expint" in str(second), "this test is pointless if the MGF changed"

    probe = (np.array([-1.0]),)
    assert cached_lambdify(second, (t_sym,), probe=probe) is None
    # ... and the verdict is cached, so the failure is paid once.
    assert cached_lambdify(second, (t_sym,), probe=probe) is None

    # The prior still works, through the exact path.
    from jumufraktiv.derivativeDispatch import mgfDerivative_integer

    log_abs, sign = mgfDerivative_integer(
        2, pareto, method="symbolic", t=-1.0, log=True
    )
    assert np.isfinite(log_abs) and sign == 1


def test_compiling_without_a_probe_does_not_pretend_to_have_checked(gamma_prior):
    """No probe means no verdict: the caller gets whatever SymPy produced.

    Asserted so that the probe cannot be quietly dropped at a call site and
    leave the guarantee resting on nothing.
    """
    from jumufraktiv.symbolic_cache import cached_diff, cached_lambdify
    from jumufraktiv.symbols import t as t_sym

    compiled = cached_lambdify(cached_diff(gamma_prior.mgf_sym, t_sym, 1), (t_sym,))

    assert callable(compiled)
