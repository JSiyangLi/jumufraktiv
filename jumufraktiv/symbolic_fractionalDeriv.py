"""
symbolic_fractionalDeriv.py

Symbolic computation of Liouville‑Caputo fractional derivatives of MGFs.

This module implements the symbolic fractional derivative using the Laplace transform
formula for the Liouville‑Caputo derivative:
    D^α_{(-∞)+} f(x) = I^γ_{(-∞)+} f^{(n+1)}(x)
where n = floor(α), γ = n+1-α, and the integral is represented as a combination
of two Laplace transforms. If the Laplace transform fails or times out, the function
falls back to the Mellin transform approach.

The main function `fractionalDeriv_symbolic` attempts Laplace first, with a timeout
(default 30s). If that fails or returns an unevaluated transform, it falls back to
Mellin. If both fail, it returns None.

Supports both complete and incomplete MGFs via the `complete` flag.
"""

import sympy as sp
from sympy.integrals.transforms import laplace_transform, mellin_transform
from jumufraktiv.mitMGFprior_class import mitMGFprior
from jumufraktiv.symbols import t  # only t is needed

import concurrent.futures


class FunctionTimedOut(TimeoutError):
    """Raised when a symbolic transform exceeds its time budget."""


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
    This replaces the third-party ``func_timeout`` package, which is
    unmaintained and no longer installable: its ``setup.py`` reads the
    ``install_layout`` attribute that setuptools removed, so building it fails
    and the whole module became unimportable — taking the ``symbolic`` backend
    for fractional orders with it.

    The worker thread cannot be killed once started, so an overrunning SymPy
    call keeps consuming CPU in the background even after this raises. That is
    a real limitation, but it matches what the previous dependency provided in
    practice, and it restores control to the caller, which is the point.

    The executor is deliberately **not** used as a context manager. ``__exit__``
    calls ``shutdown(wait=True)``, which joins the worker — so on a timeout the
    wrapper would block until the runaway call finished anyway, defeating the
    entire purpose. Measured before this was corrected: a 0.2s budget against a
    4s call returned after 4.00s.

    A consequence of ``wait=False`` worth knowing: ``ThreadPoolExecutor``
    threads are non-daemon and joined by an ``atexit`` hook, so a still-running
    transform can delay interpreter exit even though this function has already
    returned.
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
        # Never wait: returning promptly on timeout is the whole contract.
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
    Compute the Liouville‑Caputo fractional derivative.

    Tries Laplace transform with a timeout; if that fails, times out,
    or returns unevaluated, falls back to Mellin transform with same timeout.
    If Mellin also fails, returns None.

    Parameters
    ----------
    order : float
        Fractional order (positive, non‑integer).
    prior : mitMGFprior
        Prior object providing the symbolic MGF expression (mgf_sym).
    simplify : bool, optional
        If True, simplify the final expression.
    complete : bool, optional
        If True (default), differentiate the complete MGF (prior.mgf_sym).
        If False, differentiate the incomplete MGF (prior.imgf_sym).
    timeout_seconds : float, optional
        Timeout for each transform (default 30 seconds).

    Returns
    -------
    sympy.Expr or None
        Symbolic expression if successful, otherwise None.
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
            raise ValueError("Prior does not provide a symbolic incomplete MGF (imgf_sym).")
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
    # which SymPy can attempt as one integral. This replaces a Laplace leg and
    # a Mellin fallback that between them returned nothing usable for any prior
    # in the registry: Gamma raised inside `laplace_transform`, Pareto tripped a
    # subscript on an unevaluated `Mul`, and uniform and heaviside did not
    # return at all.
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

    f_n = sp.diff(expr.subs(t_sym, t_neg), t_neg, n + 1)
    integrand = z ** (gamma_order - 1) * f_n.subs(t_neg, t_neg - z)

    def _integral_attempt():
        return sp.integrate(integrand, (z, 0, sp.oo), meijerg=True, conds="none")

    prior_name = getattr(prior, "name", "this prior")

    # The timeout stays, but not for the reason an earlier analysis gave.
    #
    # That analysis concluded every registry prior returns within a fraction of
    # a second, so this machinery could be deleted. The four registry priors do
    # indeed all return from the integral quickly -- measured 0.02-0.29 s. But
    # a prior is user-supplied, and `sp.integrate` has no bound in general, so
    # an unguarded call is a hang waiting for a prior nobody tried.
    #
    # A related measurement is worth recording, because it is what the ordering
    # below is for: `sp.simplify` on an *unevaluated* integral does not return
    # (uniform at order 1.5, killed at 120 s), while `sp.integrate` on the same
    # input takes 0.29 s. So the unevaluated case is rejected BEFORE any
    # simplification is attempted, and the timeout covers the simplify step too
    # rather than only the integral.
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

    # The 1/Gamma(gamma) prefactor. Every numeric backend applies it; this
    # module did not, so its result was Gamma(gamma) times too large -- 77% at
    # order 0.5. That was invisible because nothing ever returned an expression.
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


# ===== Example usage =====
if __name__ == "__main__":
    import jumufraktiv.MGFdictionary  # ensures priors are registered
    from jumufraktiv.mitMGFprior_class import mitMGFprior

    # Create a Gamma prior
    gamma_prior = mitMGFprior.from_registry(
        "gamma",
        params={"alpha": 2.0, "beta": 3.0}
    )

    print("Testing fractional derivative of Gamma MGF (order 0.5):")
    result = fractionalDeriv_symbolic(0.5, gamma_prior, simplify=True)
    if result is not None:
        print("Symbolic result:")
        sp.pprint(result)
    else:
        print("Failed to compute fractional derivative.")

    print("\n" + "-" * 60)

    print("Testing fractional derivative of Gamma MGF (order 3.2):")
    result = fractionalDeriv_symbolic(3.2, gamma_prior, simplify=True)
    if result is not None:
        print("Symbolic result:")
        sp.pprint(result)
    else:
        print("Failed to compute fractional derivative.")