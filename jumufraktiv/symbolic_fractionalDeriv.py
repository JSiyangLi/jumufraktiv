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
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(func, *args)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            raise FunctionTimedOut(
                f"call exceeded its {timeout_seconds}s budget"
            ) from exc
        finally:
            # Do not block interpreter shutdown waiting on a runaway transform.
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
    n = sp.floor(alpha)
    gamma = (n + 1) - alpha

    # ---- Step 2: integer derivative of order n+1 ----
    f_n = sp.diff(expr, t_sym, int(n) + 1)

    # ---- Step 3: Laplace method with timeout ----
    def _laplace_attempt():
        u = sp.Symbol('u', positive=True, real=True)
        w = sp.Symbol('w', positive=True, real=True)
        s = sp.Symbol('s')

        integrand1 = f_n.subs(t_sym, t_sym - sp.exp(u))
        F1 = laplace_transform(integrand1, u, s, noconds=True)
        I1 = F1.subs(s, -gamma)

        integrand2 = f_n.subs(t_sym, t_sym - sp.exp(-w))
        F2 = laplace_transform(integrand2, w, s, noconds=True)
        I2 = F2.subs(s, gamma)

        return I1 + I2

    laplace_success = False
    frac_expr = None

    try:
        frac_expr = func_timeout(timeout_seconds, _laplace_attempt, args=())
        # Check for unevaluated transform
        if _is_unevaluated_transform(frac_expr):
            raise RuntimeError("Laplace returned unevaluated")
        laplace_success = True
    except FunctionTimedOut:
        print(f"⚠️ Laplace transform timed out after {timeout_seconds} seconds.")
    except Exception as e:
        print(f"⚠️ Laplace method failed: {e}")

    # If Laplace succeeded, return result (with optional simplification)
    if laplace_success:
        if simplify:
            frac_expr = sp.simplify(frac_expr)
        return frac_expr

    # ---- Step 4: Mellin transform with timeout ----
    print("   Falling back to Mellin transform...")
    try:
        def _mellin_attempt():
            z = sp.Symbol('z', positive=True)
            g = f_n.subs(t_sym, t_sym - z)
            s_m = sp.Symbol('s')
            return mellin_transform(g, z, s_m)

        mellin_result = func_timeout(timeout_seconds, _mellin_attempt, args=())
        F_s = mellin_result[0]
        frac_expr = F_s.subs(sp.Symbol('s'), gamma)

        if _is_unevaluated_transform(frac_expr):
            print("❌ Mellin transform returned unevaluated.")
            return None

        if simplify:
            frac_expr = sp.simplify(frac_expr)
        return frac_expr

    except FunctionTimedOut:
        print(f"❌ Mellin transform timed out after {timeout_seconds} seconds.")
        return None
    except Exception as e2:
        print(f"❌ Mellin transform failed: {e2}")
        return None


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