"""A memo for repeated symbolic differentiation.

Every backend that differentiates a prior's MGF asks SymPy for the same
derivative over and over, and the redundancy is structural rather than
careless. A prior's symbolic MGF is fixed once the hyperparameters are
substituted, but the quantities built from it — evidence, density, CDF,
moments, the posterior predictive — each re-enter the dispatcher, and a
quadrature evaluates its integrand at every node. The expression being
differentiated does not change across any of that, so a handful of distinct
``(expression, symbol, order)`` triples serve every call the package makes.

Notes
-----
**The key is the expression object, not its** ``srepr``. Both are correct, and
the object is far cheaper: 0.36 μs against 68.1 μs per lookup, measured on this
package's Gamma MGF. Correctness rests on SymPy's ``__eq__`` being structural
and precision-aware — ``Float(9.0, 53) == Float(9.0, 24)`` is ``False``, so two
expressions that agree in value but differ in precision are *not* conflated,
and a derivative computed at one precision is never returned for the other.
Should SymPy's equality ever stop distinguishing precisions, the key must
become ``srepr``; :file:`tests/test_symbolic_cache.py` asserts the property
directly, so that change cannot pass unnoticed.

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


#: Maps ``(expression, argument symbols)`` to a compiled numeric function, or
#: to ``_UNCOMPILABLE`` for an expression SymPy compiles but cannot run.
_LAMBDIFY_CACHE: dict = {}

#: Recorded for an expression whose compiled form raises. Cached like a hit, so
#: the failure is paid once rather than on every evaluation.
_UNCOMPILABLE = object()


def cached_lambdify(expr, args, probe=None):
    """Compile ``expr`` to a NumPy function of ``args``, or return ``None``.

    Parameters
    ----------
    expr : sympy.Expr
        Expression to compile. Must have no free symbols beyond ``args``.
    args : tuple of sympy.Symbol
        The compiled function's parameters, in order.
    probe : tuple, optional
        One in-domain argument value per entry of ``args``. When given, the
        compiled function is called on it once and ``None`` is returned if that
        raises. Without a probe the compiled function is returned unchecked.

    Returns
    -------
    callable or None
        ``None`` means "this expression has no working compiled form"; the
        caller should evaluate it symbolically instead.

    Notes
    -----
    Compiling successfully is not evidence that the compiled function runs. An
    expression containing ``expint``, the generalised exponential integral used
    by the Pareto prior's MGF, compiles without complaint and then raises
    ``NameError`` on the first call. That makes an unprobed compiled function a
    correctness question rather than a performance one, so pass a ``probe``
    whenever the result will be used inside a quadrature: the failure is then
    found once, at setup, on a value the caller knows is in domain.

    ``None`` is returned rather than raised because the symbolic path computes
    the same quantity, exactly; a prior whose expression will not compile
    should be slower, not broken.
    """
    key = (expr, args)
    try:
        hit = _LAMBDIFY_CACHE.get(key)
    except TypeError:
        return None

    if hit is _UNCOMPILABLE:
        return None
    if hit is not None:
        return hit

    try:
        # `scipy` cannot be dropped from `modules`: NumPy alone has no
        # `lowergamma`, `uppergamma`, `polygamma` or `Ei`, all of which appear
        # in this package's priors. Nor is it sufficient -- `expint`, which the
        # Pareto MGF is written with, is in neither, and SymPy compiles it
        # anyway, so the failure surfaces as a `NameError` on the first call.
        # `jumufraktiv.special` supplies it, ahead of both so it wins.
        #
        # The probe below still runs and still matters: it is what catches the
        # next such name rather than this one.
        from jumufraktiv.special import LAMBDIFY_NAMESPACE

        compiled = sp.lambdify(
            args, expr, modules=[LAMBDIFY_NAMESPACE, "scipy", "numpy"]
        )
        if probe is not None:
            compiled(*probe)
    except Exception:
        # Deliberately broad, and deliberately not an error. The question being
        # asked is "does a compiled form of this expression work?", and every
        # answer other than yes means the same thing: use the symbolic path.
        # Nothing is swallowed -- the exact computation still runs.
        _LAMBDIFY_CACHE[key] = _UNCOMPILABLE
        return None

    if len(_LAMBDIFY_CACHE) >= _DERIV_CACHE_MAX:
        _LAMBDIFY_CACHE.clear()
    _LAMBDIFY_CACHE[key] = compiled
    return compiled


#: Maps an expression to a function returning its terms' values, for the
#: cancellation diagnostic. Separate from :data:`_LAMBDIFY_CACHE` because the
#: compiled object is different — a vector of terms rather than their sum.
_TERMS_CACHE: dict = {}


def cached_term_values(expr, symbol, probe=None):
    """Compile the *terms* of an expression, so their cancellation can be seen.

    Parameters
    ----------
    expr : sympy.Expr
        Expression whose terms are wanted, typically a differentiated MGF.
    symbol : sympy.Symbol
        The free symbol to evaluate against.
    probe : array-like, optional
        A value of `symbol` known to be in domain. The compiled function is
        called on it once, and discarded if that raises.

    Returns
    -------
    callable or None
        Maps a value (or array) of `symbol` to a *list* of the terms' values,
        one entry per term. Entries are not all the same shape: a term that
        does not contain `symbol` evaluates to a scalar whatever the input, so
        the caller must broadcast. `None` if the expression carries other free
        symbols, or if it cannot be compiled or called — the same `expint`
        case :func:`cached_lambdify` returns `None` for.

    Notes
    -----
    Summing an expression tells you the answer; summing its terms separately
    tells you whether that answer means anything. ``Σ|term| / |Σ term|`` is the
    factor by which the leading digits cancelled, so ``log10`` of it is the
    number of significant digits lost.

    The expansion and compilation are the expensive part — 276 ms at order 12
    and 1071 ms at order 30 for this package's uniform prior — and they are
    cached here on the same structural key as the derivative itself. Evaluating
    the compiled result costs 0.015 ms to 0.036 ms per point across that range,
    which is why the diagnostic can run on every call rather than on request.
    """
    key = (expr, symbol)
    if key in _TERMS_CACHE:
        return _TERMS_CACHE[key]

    compiled = None
    try:
        terms = sp.expand(expr).as_ordered_terms()
        if not any(term.free_symbols - {symbol} for term in terms):
            # A list rather than a `Matrix`. A term that does not contain
            # `symbol` -- a constant one, which a differentiated MGF can
            # perfectly well have -- evaluates to a scalar while its
            # neighbours evaluate to arrays, and `Matrix` raises outright on
            # that mixture. A list returns them side by side and leaves the
            # broadcasting to the caller.
            compiled = sp.lambdify(symbol, list(terms), modules=["scipy", "numpy"])
    except (TypeError, ValueError, AttributeError, NotImplementedError):
        compiled = None

    # SymPy compiles an expression it has no numeric backend for without
    # complaint, and the result raises NameError on the first call -- Pareto's
    # MGF uses `expint`, which neither scipy nor numpy provides. Calling it
    # once here is the only way to find out, and caching the verdict pays that
    # cost once rather than per evaluation.
    if compiled is not None and probe is not None:
        try:
            compiled(probe)
        except Exception:
            # Deliberately broad: any failure at all means this compiled
            # function is unusable, and the caller falls back to the exact
            # path. Narrowing it would let an unanticipated failure mode
            # through to the evaluation, where it becomes a traceback rather
            # than a fallback.
            compiled = None

    if len(_TERMS_CACHE) >= _DERIV_CACHE_MAX:
        _TERMS_CACHE.clear()
    _TERMS_CACHE[key] = compiled
    return compiled


def clear_derivative_cache() -> None:
    """Empty the cache.

    Nothing in the library needs this — the cached values cannot go stale,
    since a SymPy expression is immutable and its derivative is a pure function
    of it. It exists so that tests can measure cache behaviour from a known
    state, and so that a long-lived process can release the memory.
    """
    _DERIV_CACHE.clear()
    _LAMBDIFY_CACHE.clear()
    _TERMS_CACHE.clear()


def derivative_cache_size() -> int:
    """Return the number of cached derivatives, for tests and diagnostics."""
    return len(_DERIV_CACHE)
