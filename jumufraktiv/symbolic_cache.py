"""A memo for repeated symbolic differentiation.

Every backend that differentiates a prior's MGF asks SymPy for the same
derivative over and over. Measured on one quick pass of the test suite:
**97,308 calls to** ``sp.diff`` **for 40 distinct** ``(expression, symbol,
order)`` **triples** — a redundancy factor of about 2,400 — costing 155 s
inside ``sp.diff`` out of a 338 s run. Nearly all of it comes from one line:
95,573 of those calls are ``symbolic_integerDeriv.py``'s single ``sp.diff``.

The reason is structural rather than careless. A prior's symbolic MGF is fixed
once the hyperparameters are substituted, but the quantities built from it —
evidence, density, CDF, moments, the posterior predictive — each re-enter the
dispatcher, and a quadrature evaluates its integrand at every node. The
expression being differentiated does not change across any of that.

Notes
-----
**The key is the expression object, not its** ``srepr``. Both are correct, and
the object is far cheaper: 0.36 µs against 68.1 µs per lookup, measured on this
package's Gamma MGF. Correctness rests on SymPy's ``__eq__`` being structural
and precision-aware — ``Float(9.0, 53) == Float(9.0, 24)`` is ``False``, so two
expressions that agree in value but differ in precision are *not* conflated,
and a derivative computed at one precision is never returned for the other.
That was verified before choosing the key rather than assumed; ``srepr`` was
the obvious conservative choice and turned out not to be necessary.

Returning a cached expression is safe because SymPy expressions are immutable,
so a caller cannot alter what the next caller receives.
"""

import sympy as sp

#: Maps ``(expression, symbol, order)`` to the derivative.
_DERIV_CACHE: dict = {}

#: Above this many entries the cache is emptied rather than evicted one at a
#: time. The built-in priors need 40 entries, so the cap exists only for
#: user-supplied priors and for long sessions that build many of them; it is
#: not a tuning parameter for ordinary use. Clearing wholesale is deliberate:
#: an LRU would add per-call bookkeeping to a path whose entire purpose is to
#: be cheaper than the work it replaces, to manage a dictionary that in
#: practice never fills.
_DERIV_CACHE_MAX = 512


def cached_diff(expr, symbol, order):
    """Return ``d^order/d symbol^order`` of ``expr``, reusing earlier results.

    Parameters
    ----------
    expr : sympy.Expr
        Expression to differentiate.
    symbol : sympy.Symbol
        Variable to differentiate with respect to.
    order : int
        Order of differentiation.

    Returns
    -------
    sympy.Expr
        The derivative. Identical to ``sympy.diff(expr, symbol, order)``.

    Notes
    -----
    An unhashable argument falls through to ``sympy.diff`` uncached rather than
    raising. The ``except`` is narrow by design: only ``TypeError`` from the
    lookup is caught, so a genuine failure inside ``sympy.diff`` still
    propagates.
    """
    key = (expr, symbol, order)
    try:
        hit = _DERIV_CACHE.get(key)
    except TypeError:
        return sp.diff(expr, symbol, order)

    if hit is not None:
        return hit

    result = sp.diff(expr, symbol, order)

    if len(_DERIV_CACHE) >= _DERIV_CACHE_MAX:
        _DERIV_CACHE.clear()
    _DERIV_CACHE[key] = result
    return result


def clear_derivative_cache() -> None:
    """Empty the cache.

    Nothing in the library needs this — the cached values cannot go stale,
    since a SymPy expression is immutable and its derivative is a pure function
    of it. It exists so that tests can measure cache behaviour from a known
    state, and so that a long-lived process can release the memory.
    """
    _DERIV_CACHE.clear()


def derivative_cache_size() -> int:
    """Return the number of cached derivatives, for tests and diagnostics."""
    return len(_DERIV_CACHE)
