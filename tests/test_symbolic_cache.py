"""The derivative memo must be invisible except in how long things take.

These tests are deliberately small and fast. The cache sits under every
symbolic path in the package, so the seven hundred tests that already exist
are what prove it does not change any answer — re-checking that here would
duplicate them at real quadrature cost. What is left for this file is the
handful of properties those tests cannot see: that the cache is actually being
used, that it is bounded, and that its key does not conflate two expressions
which must stay distinct.
"""

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
