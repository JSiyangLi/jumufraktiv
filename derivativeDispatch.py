"""
derivativeDispatch.py

Unified interface to compute integer derivatives of MGFs using symbolic, Bell‑polynomial,
or JAX methods.

Imports:
    - integerDeriv_symbolic from symbolic_integerDeriv.py
    - integerDeriv_numeric_bell from numeric_integerDeriv_Bell.py
    - integerDeriv_numeric_jax from numeric_integerDeriv_JAX.py

Function:
    mgfDerivative_integer(order, prior, method='symbolic', t=nan, params=None,
                          simplify=False, log=True)
"""

import math
import warnings
import sympy as sp
import numpy as np
from jumufraktiv.symbolic_integerDeriv import integerDeriv_symbolic
from jumufraktiv.numeric_integerDeriv_Bell import integerDeriv_numeric_bell
from jumufraktiv.numeric_integerDeriv_JAX import integerDeriv_numeric_jax

from jumufraktiv.symbols import t as t_sym, u as u_sym   # <-- import canonical u

def mgfDerivative_integer(
    order: int | sp.Expr,
    prior,
    method: str = "symbolic",
    t: float = None,
    u: float = None,          # optional for incomplete MGF
    simplify: bool = False,
    log: bool = True,
    complete: bool = True,
    symbolic_timeout: float = 600.0,
    cgf_method: str = "auto",
):
    """
    Compute an integer-order derivative of a prior MGF.

    Parameters
    ----------
    order : int | sp.Expr
        Non-negative derivative order.
    prior : mitMGFprior
        Prior object containing symbolic and/or backend MGF/PDF representations.
    method : {"symbolic", "bell", "jax"}, optional
        Derivative backend.
    t : float, optional
        Evaluation point for the MGF (uses canonical symbol 't').
    u : float, optional
        Truncation point for incomplete MGF (used when complete=False).
        For symbolic method, if provided, substitutes the canonical 'u' symbol.
    simplify : bool, optional
        Whether to simplify symbolic derivatives.
    complete : bool, optional
        If True (default), differentiate the complete MGF.
        If False, differentiate the incomplete MGF.
    log : bool, optional
        If True, numeric methods return (log_abs, sign).
        Otherwise return the ordinary derivative as float.
    symbolic_timeout, cgf_method : passed to Bell backend.

    Returns
    -------
    sympy.Expr, (log_abs, sign), or float
    """
    method = method.lower()
    if method not in {"symbolic", "bell", "jax"}:
        raise ValueError("method must be one of {'symbolic','bell','jax'}.")

    if method == "bell" and not complete:
        raise ValueError("Bell method does not support incomplete MGF (complete=False).")

    # ---------------------------------------------------------
    # Symbolic differentiation
    # ---------------------------------------------------------
    if method == "symbolic":
        expr = integerDeriv_symbolic(
            order=order,
            prior=prior,
            simplify=simplify,
            complete=complete
        )

        # If evaluation point t is provided, substitute it
        if t is not None:
            expr = expr.subs(t_sym, t)

        # If incomplete and u is provided, substitute the canonical u
        if not complete and u is not None:
            expr = expr.subs(u_sym, u)

        # Return symbolic if any free symbols remain
        if expr.free_symbols:
            return expr

        # Fully numeric: evaluate
        val = float(expr.evalf())
        if abs(val) < 1e-300:
            log_abs = -math.inf
            sign = 1
        else:
            log_abs = np.log(abs(val))
            sign = 1 if val > 0 else -1

        return (log_abs, sign) if log else val

    # ---------------------------------------------------------
    # Bell polynomial backend
    # ---------------------------------------------------------
    if method == "bell":
        if t is None:
            raise ValueError("t must be supplied for method='bell'.")
        log_abs, sign = integerDeriv_numeric_bell(
            prior=prior,
            t=t,
            order=order,
            symbolic_timeout=symbolic_timeout,
            cgf_method=cgf_method,
        )
        if log:
            return log_abs, sign
        return 0.0 if log_abs == -math.inf else sign * np.exp(log_abs)

    # ---------------------------------------------------------
    # JAX backend
    # ---------------------------------------------------------
    if t is None:
        raise ValueError("t must be supplied for method='jax'.")
    # Pass u to the JAX function (which expects it for incomplete MGF)
    log_abs, sign = integerDeriv_numeric_jax(
        prior=prior,
        t=t,
        order=order,
        complete=complete,
        u=u,                     # <-- added
    )
    if log:
        return log_abs, sign
    return 0.0 if log_abs == -math.inf else sign * np.exp(log_abs)


def mgfDerivative_fractional(
    order: float,
    prior,
    method: str = "scipy",
    t: float = None,
    simplify: bool = False,
    complete: bool = True,
    log: bool = True,
    integerDeriv_method: str = "symbolic",
    u: float = None,          # truncation point for incomplete MGF
    **kwargs
):
    """
    Unified interface for fractional derivatives of the MGF.

    Parameters
    ----------
    order : float
        Fractional order (positive).
    prior : mitMGFprior
        Prior object.
    method : {'scipy', 'mpmath', 'symbolic'}, default 'scipy'
        Computation backend.
    t : float, optional
        Evaluation point (required for numeric methods).
    simplify : bool, default False
        Simplify symbolic expressions (method='symbolic').
    complete : bool, default True
        If True, differentiate complete MGF; if False, incomplete MGF.
    log : bool, default True
        If True, numeric output is (log_abs, sign); else ordinary float.
    integerDeriv_method : str, default 'symbolic'
        Method for integer derivatives inside integrators.
    u : float, optional
        Truncation point for incomplete MGF (used when complete=False).
    **kwargs : passed to underlying functions.

    Returns
    -------
    sympy.Expr, (log_abs, sign), or float
    """
    # ---- Symbolic path ----
    if method.lower() == "symbolic":
        warnings.warn(
            "Symbolic fractional derivatives can be very slow. "
            "Consider 'scipy' or 'mpmath' instead."
        )
        from symbolic_fractionalDeriv import fractionalDeriv_symbolic

        expr = fractionalDeriv_symbolic(
            order=order,
            prior=prior,
            simplify=simplify,
            complete=complete,
            **kwargs
        )
        if expr is None:
            return None

        # Substitute t and u if provided
        if t is not None:
            expr = expr.subs(t_sym, t)
        if not complete and u is not None:
            expr = expr.subs(u_sym, u)

        # Final resolution
        if expr.free_symbols:
            return expr
        value = float(expr.evalf())
        if abs(value) < 1e-300:
            return (-math.inf, 1) if log else 0.0
        log_abs = math.log(abs(value))
        sign = 1 if value > 0 else -1
        return (log_abs, sign) if log else value

    # ---- Numeric paths require t ----
    if t is None or math.isnan(t):
        raise ValueError(f"For method '{method}', t must be provided.")

    # ---- scipy ----
    if method.lower() == "scipy":
        from jumufraktiv.numeric_fractionalDeriv_scipy import fractionalDeriv_numeric_scipy
        return fractionalDeriv_numeric_scipy(
            order=order,
            prior=prior,
            t=t,
            method=integerDeriv_method,
            simplify=simplify,
            return_log=log,
            complete=complete,
            u=u,
            **kwargs
        )

    # ---- mpmath ----
    if method.lower() == "mpmath":
        from jumufraktiv.numeric_fractionalDeriv_mpmath import fractionalDeriv_numeric_mpmath
        return fractionalDeriv_numeric_mpmath(
            order=order,
            prior=prior,
            t=t,
            method=integerDeriv_method,
            simplify=simplify,
            return_log=log,
            complete=complete,
            u=u,
            **kwargs
        )

    raise ValueError(f"Unknown method: '{method}'. Choose 'scipy', 'mpmath', or 'symbolic'.")

def mgfDerivative(
    order: float,
    prior,
    method: str = "auto",
    t: float = None,
    simplify: bool = False,
    complete: bool = True,
    log: bool = True,
    integer_method: str = "symbolic",
    use_interpolation: bool = True,
    d_vec: tuple = (0.8, 0.9, 0.95),
    int_tol: float = 1e-12,
    u: float = None,          # <-- NEW: truncation point for incomplete MGF
    **kwargs
):
    """
    Unified wrapper for integer or fractional derivatives of the MGF.

    This function automatically detects whether the derivative order is integer
    or fractional and dispatches to the appropriate method (integer or fractional).
    For integer orders, it can use symbolic, Bell-polynomial, or JAX methods.
    For fractional orders, it can use scipy, mpmath, or symbolic methods.

    Parameters
    ----------
    order : float
        Derivative order (integer or fractional). If within `int_tol` of an integer,
        it is treated as integer.
    prior : mitMGFprior
        Prior object containing the MGF and its functions.
    method : str, optional
        For integer order: one of 'symbolic', 'bell', 'jax'.
        For fractional order: one of 'scipy', 'mpmath', 'symbolic'.
        Special value: 'auto' (default) chooses 'symbolic' for integer orders,
        and 'scipy' for fractional orders.
    t : float, optional
        Evaluation point. Required for numeric methods.
    u : float, optional
        Truncation point for incomplete MGF (used when complete=False).
        Required for numeric evaluation when complete=False. Default None.
    simplify : bool, optional
        If True, simplify the symbolic expression (only for 'symbolic' methods).
    complete : bool, optional
        If True (default), differentiate the complete MGF.
        If False, differentiate the incomplete MGF.
    log : bool, optional
        If True and output is numeric, return (log_abs, sign). If False, return the
        ordinary-scale value as a float.
    integer_method : str, optional
        For fractional order only: method to compute the integer derivative inside
        the fractional integrator. One of 'symbolic', 'jax', or 'bell'.
        Default 'symbolic'.
    use_interpolation : bool, optional
        If True, use cubic interpolation for orders near an integer from below.
        Default True.
    d_vec : tuple, optional
        Three values in (0,1) that are complements of deviations. For example,
        d_vec = (0.8, 0.9, 0.95) gives actual deviations (0.2, 0.1, 0.05) via `dev = 1 - d`.
        The interpolation points are n - dev1, n - dev2, n - dev3, n.
        Interpolation is used only for orders in `(n - min_dev, n)` where
        `min_dev = min(dev_i) = 1 - max(d_vec)`.
        Must have exactly 3 elements.
    int_tol : float, optional
        Tolerance for detecting integer order. If |order - round(order)| < int_tol,
        the order is treated as integer. Default 1e-12.
    **kwargs : additional keyword arguments passed to the underlying functions.
        For integer methods (via mgfDerivative_integer):
            symbolic_timeout : float, optional
                Maximum time (seconds) allowed for symbolic CGF derivative computation
                in the 'bell' method. If exceeded, falls back to JAX. Default 600.
            cgf_method : str, optional
                For 'bell' method only. Method to compute CGF derivatives in the numeric path.
                Options: 'auto' (try jet, fallback to grad), 'jet' (force Taylor mode),
                'grad' (force nested grad). Default 'auto'.
        For fractional 'scipy' and 'mpmath' methods: epsabs, epsrel, limit, etc.
        For fractional 'symbolic': timeout_seconds, etc.

    Returns
    -------
    Depending on the method and order:
        - If method='symbolic' and some symbols not substituted:
            sympy.Expr (symbolic expression)
        - If numeric evaluation (log=True): tuple (log_abs, sign)
        - If numeric evaluation (log=False): float (ordinary value)

    Raises
    ------
    ValueError
        If order is invalid, method is invalid for the order type, required arguments missing,
        or d_vec does not have exactly 3 elements or any element >= 1.
    """
    # ---- Extract and pop method‑specific kwargs ----
    cgf_method = kwargs.pop('cgf_method', 'auto')
    symbolic_timeout = kwargs.pop('symbolic_timeout', 600.0)

    # ---- Validate d_vec ----
    if len(d_vec) != 3:
        raise ValueError("d_vec must have exactly 3 elements (e.g., (0.8, 0.9, 0.95)).")
    if any(d >= 1 for d in d_vec):
        raise ValueError("All elements of d_vec must be < 1 (to get positive deviations).")
    min_dev = 1.0 - max(d_vec)

    # ---- Determine order type ----
    if isinstance(order, sp.Basic):
        order_type = "symbolic"
    else:
        if abs(order - round(order)) < int_tol:
            order_type = "integer"
        else:
            order_type = "fractional"

    # ---- Resolve 'auto' method ----
    if method.lower() == 'auto':
        if order_type == "symbolic":
            method = 'symbolic'
        elif order_type == "integer":
            method = 'symbolic'
        else:
            method = 'scipy'

    # ---- Dispatch ----
    if order_type == "symbolic":
        if method.lower() not in {'auto', 'symbolic'}:
            raise ValueError(f"Invalid method '{method}' for symbolic order. Only 'symbolic' is allowed.")
        warnings.warn(
            "Derivative order contains symbolic variables. "
            "Using integerDeriv_symbolic() as a formal symbolic derivative. "
            "The resulting expression is often the analytic continuation "
            "to non-integer orders, but this is not guaranteed.",
            UserWarning,
        )
        return mgfDerivative_integer(
            order=order,
            prior=prior,
            method="symbolic",
            t=t,
            simplify=simplify,
            complete=complete,
            log=log,
            u=u,                     # pass through
            symbolic_timeout=symbolic_timeout,
            cgf_method=cgf_method,
        )

    elif order_type == "integer":
        int_order = int(round(order))
        valid_int_methods = {'symbolic', 'bell', 'jax'}
        if method.lower() not in valid_int_methods:
            raise ValueError(f"Invalid method '{method}' for integer order. Choose from {valid_int_methods}.")
        return mgfDerivative_integer(
            order=int_order,
            prior=prior,
            method=method,
            t=t,
            simplify=simplify,
            complete=complete,
            log=log,
            u=u,                     # pass through
            symbolic_timeout=symbolic_timeout,
            cgf_method=cgf_method,
        )

    else:
        # ---- Fractional order ----
        valid_frac_methods = {'scipy', 'mpmath', 'symbolic'}
        if method.lower() in {'jax', 'bell'}:
            print(f"Note: For fractional order, 'method' should be one of {valid_frac_methods}. "
                  f"Interpreting 'method' as 'integer_method' and using default fractional method 'scipy'.")
            integer_method = method
            method = 'scipy'
        elif method.lower() not in valid_frac_methods:
            raise ValueError(f"Invalid method '{method}' for fractional order. Choose from {valid_frac_methods}.")

        # Validate integer_method
        valid_int_methods = {'symbolic', 'jax', 'bell'}
        if integer_method.lower() not in valid_int_methods:
            raise ValueError(f"Invalid integer_method '{integer_method}'. Choose from {valid_int_methods}.")

        # ---- Check if interpolation should be used ----
        n = int(np.ceil(order))
        if use_interpolation and order > n - min_dev:
            if method.lower() != 'scipy':
                print(f"⚠️ Interpolation triggered for order {order}. "
                      f"Overriding method '{method}' with 'scipy' (interpolation points use scipy).")
            scipy_keys = {'epsabs', 'epsrel', 'limit', 'initial_L', 'max_L', 'tol', 'use_tan'}
            scipy_kwargs = {k: v for k, v in kwargs.items() if k in scipy_keys}
            try:
                from numeric_fractionalDeriv_interpolation import fractionalDeriv_interpolated
            except ImportError as e:
                raise ImportError("Could not import numeric_fractionalDeriv_interpolation") from e
            # Interpolation expects a prior object (no params)
            return fractionalDeriv_interpolated(
                order=order,
                prior=prior,
                t=t,
                d_vec=d_vec,
                return_log=log,
                integer_method=integer_method,
                complete=complete,
                u=u,                     # pass through (requires updating fractionalDeriv_interpolated)
                **scipy_kwargs
            )

        # ---- Otherwise, use standard fractional method ----
        return mgfDerivative_fractional(
            order=order,
            prior=prior,
            method=method,
            t=t,
            simplify=simplify,
            complete=complete,
            log=log,
            integerDeriv_method=integer_method,
            u=u,                     # pass through
            **kwargs
        )


if __name__ == "__main__":
    import math
    import sympy as sp
    import pandas as pd
    import jumufraktiv.MGFdictionary  # registers priors
    from jumufraktiv.mitMGFprior_class import mitMGFprior

    # ---- Create Gamma priors ----
    gamma_prior = mitMGFprior.from_registry(
        "gamma",
        params={"alpha": 2.0, "beta": 3.0}
    )
    gamma_prior_small = mitMGFprior.from_registry(
        "gamma",
        params={"alpha": 1e-5, "beta": 1e-5}
    )
    gamma_prior_exp = mitMGFprior.from_registry(
        "gamma",
        params={"alpha": 1.0, "beta": 0.9}
    )

    print("=" * 60)
    print("Integer derivative examples")
    print("=" * 60)

    # 1. Symbolic expression (no evaluation)
    expr = mgfDerivative_integer(2, gamma_prior, method="symbolic")
    print("Symbolic expression for 2nd derivative of Gamma MGF:")
    sp.pprint(expr)

    # 2. Symbolic evaluation with numeric output (log=True by default)
    log_abs, sign = mgfDerivative_integer(
        2, gamma_prior, method="symbolic", t=-1.0, log=True
    )
    print(f"\nSymbolic evaluated (log scale): log|deriv| = {log_abs:.6f}, sign = {sign}")

    # 3. Symbolic evaluation with ordinary output (log=False)
    val = mgfDerivative_integer(
        2, gamma_prior, method="symbolic", t=-1.0, log=False
    )
    print(f"Symbolic evaluated (ordinary scale): {val:.6f}")

    # 4. Bell method (numeric, log=True default)
    log_abs, sign = mgfDerivative_integer(
        2, gamma_prior, method="bell", t=-1.0
    )
    print(f"\nBell method: log|deriv| = {log_abs:.6f}, sign = {sign}")

    # 5. JAX method with ordinary output
    val = mgfDerivative_integer(
        2, gamma_prior, method="jax", t=-1.0, log=False
    )
    print(f"JAX method (ordinary): {val:.6e}")

    # ===== Fractional derivative examples =====
    print("\n" + "=" * 60)
    print("Fractional derivative examples")
    print("=" * 60)

    frac_order = 1.99
    t_val = -1.0

    # 6. Symbolic fractional derivative (warning will be printed)
    print("\n--- Symbolic fractional derivative ---")
    try:
        expr_frac = mgfDerivative_fractional(
            order=frac_order,
            prior=gamma_prior,
            method='symbolic',
            simplify=True,
            timeout_seconds=10
        )
        print("Symbolic fractional derivative expression:")
        sp.pprint(expr_frac)
    except Exception as e:
        print(f"Symbolic fractional derivative failed: {e}")

    # 7. scipy method (log=True default)
    print("\n--- scipy method (log scale) ---")
    log_abs_frac, sign_frac = mgfDerivative_fractional(
        order=frac_order,
        prior=gamma_prior,
        method='scipy',
        t=t_val,
        integerDeriv_method='symbolic',
        epsrel=1e-10,
        tol=1e-8
    )
    print(f"scipy fractional: log|deriv| = {log_abs_frac:.6f}, sign = {sign_frac}")

    # 8. mpmath method (log=False, high precision)
    print("\n--- mpmath method (ordinary scale, high precision) ---")
    try:
        val_frac_mpmath = mgfDerivative_fractional(
            order=frac_order,
            prior=gamma_prior,
            method='mpmath',
            t=t_val,
            integerDeriv_method='symbolic',
            dps=60,
            tol=1e-10,
            log=False
        )
        print(f"mpmath fractional (ordinary): {val_frac_mpmath:.6e}")
    except Exception as e:
        print(f"mpmath fractional failed: {e}")
        val_frac_mpmath = float('nan')

    # Compare with integer 2nd derivative
    log_abs2, sign2 = mgfDerivative_integer(
        2, gamma_prior, method="symbolic", t=t_val, log=True
    )
    deriv2 = sign2 * np.exp(log_abs2)
    print(f"\nOrdinary 2nd derivative at t={t_val}: {deriv2:.6e}")
    print(f"Difference (scipy vs 2nd) on log scale: {abs(log_abs_frac - log_abs2):.2e}")
    if not math.isnan(val_frac_mpmath):
        print(f"Difference (mpmath vs 2nd) on ordinary scale: {abs(val_frac_mpmath - deriv2):.2e}")

    # ===== Unified wrapper examples =====
    print("\n" + "=" * 60)
    print("Unified wrapper examples")
    print("=" * 60)

    # Integer order via wrapper
    log_abs, sign = mgfDerivative(
        order=2.0,
        prior=gamma_prior,
        method='symbolic',
        t=-1.0,
        log=True
    )
    print(f"Wrapper (integer, log): log|deriv| = {log_abs:.6f}, sign = {sign}")

    # Fractional order via wrapper (auto‑corrected)
    log_abs, sign = mgfDerivative(
        order=1.99,
        prior=gamma_prior,
        method='jax',          # auto‑corrected to integer_method='jax', method='scipy'
        t=-1.0,
        epsrel=1e-10
    )
    print(f"Wrapper (fractional, auto‑corrected): log|deriv| = {log_abs:.6f}, sign = {sign}")

    # Fractional order with explicit scipy method
    log_abs, sign = mgfDerivative(
        order=1.99,
        prior=gamma_prior,
        method='scipy',
        t=-1.0,
        integer_method='symbolic',
        epsrel=1e-10
    )
    print(f"Wrapper (fractional, explicit): log|deriv| = {log_abs:.6f}, sign = {sign}")

    # ===== Interpolation test (near-integer fractional order) =====
    print("\n" + "=" * 60)
    print("Interpolation test (order 1.999, exponential prior)")
    print("=" * 60)

    # Use Gamma likelihood with exponential prior (Gamma(1,0.9)) so analytic formula exists.
    data_interp = pd.DataFrame({'y': [1.0]})
    t_interp = -1.0
    order_interp = 1.999

    # Using interpolation (default)
    log_abs_interp, sign_interp = mgfDerivative(
        order=order_interp,
        prior=gamma_prior_exp,
        method='scipy',
        t=t_interp,
        log=True,
        integer_method='symbolic',
        use_interpolation=True,
        d_vec=(0.8, 0.9, 0.95),
        epsrel=1e-10
    )
    print(f"Interpolated: log|deriv| = {log_abs_interp:.6f}, sign = {sign_interp}")

    # Analytic formula for exponential prior: D^α M(t) = λ * Γ(α+1) * (λ - t)^(-α-1)
    lambda_exp = gamma_prior_exp.params['beta']
    log_analytic = np.log(lambda_exp) + math.lgamma(order_interp + 1) - (order_interp + 1) * np.log(lambda_exp - t_interp)
    print(f"Analytic:     log|deriv| = {log_analytic:.6f}")
    print(f"Difference (interp - analytic): {log_abs_interp - log_analytic:.2e}")
    
    # ===== Incomplete MGF (iMGF) example =====
    print("\n" + "=" * 60)
    print("Incomplete MGF (iMGF) derivative example")
    print("=" * 60)

    # Use Gamma prior (which has numeric iMGF functions registered)
    u_val = 2.0
    t_val_i = -1.0
    order_i = 1.5  # fractional

    # Numeric derivative of incomplete MGF (ordinary scale)
    try:
        val_imgf = mgfDerivative(
            order=order_i,
            prior=gamma_prior,
            method='scipy',
            t=t_val_i,
            complete=False,
            u=u_val,
            log=False,
            epsrel=1e-10,
            integer_method='symbolic'
        )
        print(f"Numeric derivative of incomplete MGF (ordinary): {val_imgf:.6e}")
    except Exception as e:
        print(f"Failed (ordinary): {e}")

    # Numeric derivative (log scale)
    try:
        log_abs_imgf, sign_imgf = mgfDerivative(
            order=order_i,
            prior=gamma_prior,
            method='scipy',
            t=t_val_i,
            complete=False,
            u=u_val,
            log=True,
            epsrel=1e-10,
            integer_method='symbolic'
        )
        print(f"Numeric derivative of incomplete MGF (log): log|val| = {log_abs_imgf:.6f}, sign = {sign_imgf}")
    except Exception as e:
        print(f"Failed (log): {e}")

    # Symbolic integer derivative of incomplete MGF (uses imgf_sym if available)
    try:
        val_imgf_int = mgfDerivative(
            order=2,
            prior=gamma_prior,
            method='symbolic',
            t=t_val_i,
            complete=False,
            u=u_val,
            log=False
        )
        print(f"Symbolic integer derivative of incomplete MGF: {val_imgf_int:.6e}")
    except Exception as e:
        print(f"Symbolic integer failed: {e}")