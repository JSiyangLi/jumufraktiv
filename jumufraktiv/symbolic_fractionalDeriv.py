"""
symbolic_fractionalDeriv.py

Symbolic computation of Liouville-Caputo fractional derivatives of MGFs.

This module evaluates the defining integral of the Liouville-Caputo derivative
with lower terminal -∞ directly. Substituting x = t - z gives
    D^α_{(-∞)+} M(t) = 1/Γ(γ) ∫_0^∞ z^{γ-1} M^{(n+1)}(t - z) dz
where n = floor(α) and γ = n+1-α.

The main function `fractionalDeriv_symbolic` hands that single integral to SymPy
under a wall-clock budget (default 30s), and raises NotImplementedError when
SymPy declines it or overruns.

Supports both complete and incomplete MGFs via the `complete` flag.
"""

import concurrent.futures

import sympy as sp

from jumufraktiv.mitMGFprior_class import mitMGFprior
from jumufraktiv.symbolic_cache import cached_diff


class FunctionTimedOut(TimeoutError):
    """Raised when a symbolic integration exceeds its time budget."""


def func_timeout(timeout_seconds, func, args=()):
    """
    Run ``func`` with a wall-clock budget, raising if it overruns.

    Parameters
    ----------
    timeout_seconds : float
        Wall-clock budget in seconds.
    func : callable
        Zero-argument callable (or one accepting ``*args``).
    args : tuple, optional
        Positional arguments for ``func``.

    Returns
    -------
    object
        Whatever ``func`` returns.

    Raises
    ------
    FunctionTimedOut
        If ``func`` has not returned within ``timeout_seconds``.

    Notes
    -----
    This is a local stand-in for the identically named PyPI ``func_timeout``
    package, which is unmaintained and no longer installs on current
    setuptools.

    The worker thread cannot be killed once started, so an overrunning SymPy
    call keeps consuming CPU in the background even after this raises.

    The worker pool is shut down without waiting for that call to finish.
    ``ThreadPoolExecutor`` threads are non-daemon and joined by an ``atexit``
    hook, so a still-running integration can delay interpreter exit even though
    this function has already returned.
    """
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(func, *args)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            raise FunctionTimedOut(
                f"call exceeded its {timeout_seconds}s budget"
            ) from exc
    finally:
        # Never wait: returning promptly on timeout is the whole contract. Do
        # not switch to `with pool:` -- its `__exit__` calls
        # `shutdown(wait=True)`, which joins the runaway worker.
        pool.shutdown(wait=False)


def _is_unevaluated_transform(expr):
    """Check if expression contains an unevaluated Laplace or Mellin transform."""
    from sympy.integrals.transforms import LaplaceTransform, MellinTransform
    return expr.has(LaplaceTransform) or expr.has(MellinTransform)


def fractionalDeriv_symbolic(
    order: float,
    prior: mitMGFprior,
    simplify: bool = False,
    complete: bool = True,
    timeout_seconds: float = 30.0
):
    """
    Compute the Liouville-Caputo fractional derivative in closed form.

    Evaluates the defining integral with SymPy, under the wall-clock budget
    given by ``timeout_seconds``.

    Parameters
    ----------
    order : float
        Fractional order (positive, non-integer).
    prior : mitMGFprior
        Prior object providing the symbolic MGF expression (mgf_sym).
    simplify : bool, optional
        If True, simplify the final expression.
    complete : bool, optional
        If True (default), differentiate the complete MGF (prior.mgf_sym).
        If False, differentiate the incomplete MGF (prior.imgf_sym).
    timeout_seconds : float, optional
        Time budget, in seconds, for the integration and again for the optional
        simplification (default 30). A `simplify=True` call can therefore spend
        it twice.

    Returns
    -------
    sympy.Expr
        The derivative, as an expression in the canonical symbol ``t``.

    Raises
    ------
    NotImplementedError
        If SymPy cannot evaluate the integral in closed form, or does not
        finish within ``timeout_seconds``.
    """
    if order <= 0:
        raise ValueError("Fractional order must be positive.")

    # Get symbolic MGF expression from the prior object
    if complete:
        if not hasattr(prior, "mgf_sym") or prior.mgf_sym is None:
            raise ValueError("Prior does not provide a symbolic MGF (mgf_sym).")
        expr = prior.mgf_sym
    else:
        if not hasattr(prior, "imgf_sym") or prior.imgf_sym is None:
            raise ValueError(
                "Prior does not provide a symbolic incomplete MGF (imgf_sym)."
            )
        expr = prior.imgf_sym

    # If it's a callable, call it to get the expression
    if callable(expr):
        expr = expr()

    if not isinstance(expr, sp.Expr):
        raise TypeError("mgf_sym must be a SymPy expression.")

    # Extract the 't' symbol (should be present)
    t_sym = next((s for s in expr.free_symbols if s.name == 't'), None)
    if t_sym is None:
        raise RuntimeError("No symbol 't' found in the MGF expression.")

    alpha = order
    n = int(sp.floor(alpha))
    gamma_order = (n + 1) - alpha

    # ------------------------------------------------------------------
    # Evaluate the defining integral directly.
    #
    # Substituting x = t - z in the Liouville-Caputo integral with lower
    # terminal -infinity turns it into
    #
    #     D^a M(t) = (1/Gamma(g)) * int_0^oo z^{g-1} M^{(n+1)}(t - z) dz
    #
    # which SymPy can attempt as one integral.
    #
    # `t` must be declared negative for the integral to converge -- it is the
    # evaluation point -b(y), which is never positive -- and the canonical
    # symbol carries no such assumption, so a local one is substituted in and
    # the result mapped back at the end.
    #
    # `meijerg=True` is load-bearing rather than an optimisation. Without it
    # SymPy tries every method in turn and can run indefinitely; with it the
    # Meijer-G route either succeeds quickly or declines quickly.
    # ------------------------------------------------------------------
    t_neg = sp.Symbol("_t_neg", negative=True)
    z = sp.Symbol("_z", positive=True)

    f_n = cached_diff(expr.subs(t_sym, t_neg), t_neg, n + 1)
    integrand = z ** (gamma_order - 1) * f_n.subs(t_neg, t_neg - z)

    def _integral_attempt():
        return sp.integrate(integrand, (z, 0, sp.oo), meijerg=True, conds="none")

    prior_name = getattr(prior, "name", "this prior")

    # The timeout guards a user-supplied prior rather than the registry's four,
    # which all return from the integral in a fraction of a second.
    # `sp.integrate` has no bound in general, so an unguarded call is a hang
    # waiting for a prior nobody has tried.
    #
    # It covers the simplify step as well as the integral, and the unevaluated
    # case is rejected BEFORE any simplification is attempted: `sp.simplify`
    # on an *unevaluated* integral does not return at all (uniform at order
    # 1.5, still running at 120 s), where `sp.integrate` on the same input
    # takes 0.29 s.
    def _guarded(fn):
        try:
            return func_timeout(timeout_seconds, fn, args=())
        except FunctionTimedOut:
            raise NotImplementedError(
                f"Symbolic fractional differentiation of prior '{prior_name}' "
                f"at order {order} did not complete within {timeout_seconds} s. "
                f"Use method='scipy' or method='mpmath' instead."
            ) from None

    raw = _guarded(_integral_attempt)

    # The 1/Gamma(gamma) prefactor, which every backend must apply. Omitting it
    # leaves the result Gamma(gamma) times too large -- 77% at order 0.5.
    frac_expr = raw / sp.gamma(gamma_order)

    if frac_expr.has(sp.Integral) or _is_unevaluated_transform(frac_expr):
        raise NotImplementedError(
            f"SymPy could not evaluate the fractional derivative of prior "
            f"'{prior_name}' at order {order} in closed form. "
            f"Use method='scipy' or method='mpmath' instead."
        )

    # Map back to the canonical symbol, so the caller receives an expression in
    # `jumufraktiv.symbols.t` rather than in this function's local placeholder.
    frac_expr = frac_expr.subs(t_neg, t_sym)

    if simplify:
        frac_expr = _guarded(lambda: sp.simplify(frac_expr))

    return frac_expr
