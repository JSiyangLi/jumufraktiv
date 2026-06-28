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
import sympy as sp
import numpy as np
from symbolic_integerDeriv import integerDeriv_symbolic
from numeric_integerDeriv_Bell import integerDeriv_numeric_bell
from numeric_integerDeriv_JAX import integerDeriv_numeric_jax


def mgfDerivative_integer(
    order: int,
    prior: str,
    method: str = "symbolic",
    t: float = float('nan'),
    params: dict = None,
    simplify: bool = False,
    log: bool = True,
    symbolic_timeout: float = 600.0,
    cgf_method: str = 'auto'
):
    """
    Compute the order‑th integer derivative of the MGF using the specified method.

    Parameters
    ----------
    order : int
        Order of derivative (non‑negative integer).
    prior : str
        'gamma' or 'pareto'.
    method : str, optional
        One of 'symbolic', 'bell', 'jax'. Default 'symbolic'.
    t : float, optional
        Evaluation point. Required for 'bell' and 'jax'.
    params : dict, optional
        Distribution parameters. Required for 'bell' and 'jax'.
    simplify : bool, optional
        If True, simplify the symbolic expression (only for 'symbolic' method).
    log : bool, optional
        If True and output is numeric, return (log_abs, sign).
        If False, return the ordinary‑scale value as a float.
    symbolic_timeout : float, optional
        Maximum time (seconds) allowed for symbolic CGF derivative computation
        in the 'bell' method. If exceeded, falls back to JAX. Default 600.
    cgf_method : str, optional
        For 'bell' method only. Method to compute CGF derivatives in the numeric path.
        Options: 'auto' (try jet, fallback to grad), 'jet' (force Taylor mode),
        'grad' (force nested grad). Default 'auto'.

    Returns
    -------
    For numeric outputs:
        - If log=True: tuple (log_abs, sign)
        - If log=False: float (ordinary value)
    For symbolic method without numeric evaluation:
        - sympy.Expr (symbolic expression)
    """
    if params is None:
        params = {}

    # Dispatch by method
    if method.lower() == "symbolic":
        # Get symbolic expression
        expr = integerDeriv_symbolic(order, prior, simplify=simplify)

        # If numeric evaluation requested (t is not nan and params non‑empty)
        if not math.isnan(t) and params:
            # Extract symbols
            all_syms = expr.free_symbols
            t_sym = next((s for s in all_syms if s.name == 't'), None)
            if t_sym is None:
                raise RuntimeError("No symbol 't' found in expression.")
            # Build substitution dict
            subs_dict = {}
            for sym in all_syms:
                if sym.name == 't':
                    subs_dict[sym] = t
                elif sym.name in params:
                    subs_dict[sym] = params[sym.name]
                # else leave symbolic (should not happen if proper parameters given)
            # Evaluate numerically
            val = expr.subs(subs_dict).evalf()
            val_float = float(val)

            # Compute log_abs and sign
            if abs(val_float) < 1e-300:   # treat as zero
                log_abs = -float('inf')
                sign = 1
            else:
                log_abs = math.log(abs(val_float))
                sign = 1 if val_float > 0 else -1

            if log:
                return (log_abs, sign)
            else:
                return val_float
        else:
            # Return symbolic expression
            return expr

    elif method.lower() == "bell":
        if math.isnan(t):
            raise ValueError("For 'bell' method, t must be provided.")
        if not params:
            raise ValueError("For 'bell' method, params must be provided.")
        log_abs, sign = integerDeriv_numeric_bell(
            t, prior, params, order,
            symbolic_timeout=symbolic_timeout,
            cgf_method=cgf_method
        )
        if log:
            return (log_abs, sign)
        else:
            if log_abs == -float('inf'):
                return 0.0
            else:
                return sign * math.exp(log_abs)

    elif method.lower() == "jax":
        if math.isnan(t):
            raise ValueError("For 'jax' method, t must be provided.")
        if not params:
            raise ValueError("For 'jax' method, params must be provided.")
        log_abs, sign = integerDeriv_numeric_jax(t, prior, params, order)
        if log:
            return (log_abs, sign)
        else:
            if log_abs == -float('inf'):
                return 0.0
            else:
                return sign * math.exp(log_abs)

    else:
        raise ValueError(f"Unknown method: '{method}'. Choose 'symbolic', 'bell', or 'jax'.")

def mgfDerivative_fractional(
    order: float,
    prior: str,
    method: str = "scipy",
    t: float = float('nan'),
    params: dict = None,
    simplify: bool = False,
    log: bool = True,
    integerDeriv_method: str = "symbolic",
    **kwargs
):
    """
    Unified interface for fractional derivatives of the MGF.

    Parameters
    ----------
    order : float
        Fractional order (positive). If integer, it will still work (returns ordinary derivative).
    prior : str
        'gamma' or 'pareto'.
    method : str, optional
        One of:
            - 'scipy'   : uses scipy.integrate.quad (adaptive range, with fallback to tan)
            - 'mpmath'  : uses mpmath.quad (high precision)
            - 'symbolic': returns a symbolic expression (no numerical evaluation)
        Default 'scipy'.
    t : float, optional
        Evaluation point. Required for numeric methods ('scipy', 'mpmath').
    params : dict, optional
        Prior parameters. Required for numeric methods.
    simplify : bool, optional
        If True, simplify the symbolic expression (only for method='symbolic').
    log : bool, optional
        If True and result is numeric, returns (log_abs, sign); else returns ordinary float.
    integerDeriv_method : str, optional
        Method for integer derivatives used inside the numeric fractional integrators.
        One of 'symbolic', 'jax', or 'bell'. Default 'symbolic'.
    **kwargs : additional arguments passed to the underlying fractional function.
        For 'scipy' method: epsabs, epsrel, limit, initial_L, max_L, tol, use_tan.
        For 'mpmath' method: dps, margin, max_u, tol, use_tan.
        For 'symbolic' method: timeout_seconds (to limit computation time).
        (Refer to the docstrings of the respective functions for details.)

    Returns
    -------
    If method == 'symbolic':
        sympy.Expr (symbolic expression for the fractional derivative).
    Else:
        if log=True: (log_abs, sign)
        else: float (ordinary value)
    """
    if params is None:
        params = {}

    # ---- Handle symbolic method ----
    if method.lower() == "symbolic":
        print("⚠️ Warning: Symbolic computation of fractional derivatives can be very slow and inefficient. Consider using 'scipy' or 'mpmath' for numerical evaluation.")
        # Lazy import symbolic_fractionalDeriv
        try:
            from symbolic_fractionalDeriv import fractionalDeriv_symbolic
        except ImportError as e:
            raise ImportError("Could not import symbolic_fractionalDeriv.py") from e
        # Call the symbolic function, passing any kwargs (e.g., timeout_seconds)
        expr = fractionalDeriv_symbolic(order=order, prior=prior, simplify=simplify, **kwargs)
        return expr

    # ---- Numeric methods require t and params ----
    if math.isnan(t):
        raise ValueError(f"For method '{method}', t must be provided.")
    if not params:
        raise ValueError(f"For method '{method}', params must be provided.")

    # ---- Dispatch to scipy or mpmath ----
    if method.lower() == "scipy":
        from numeric_fractionalDeriv_scipy import fractionalDeriv_numeric_scipy
        return fractionalDeriv_numeric_scipy(
            order=order,
            prior=prior,
            params=params,
            t=t,
            method=integerDeriv_method,
            simplify=simplify,
            return_log=log,
            **kwargs
        )

    elif method.lower() == "mpmath":
        from numeric_fractionalDeriv_mpmath import fractionalDeriv_numeric_mpmath
        return fractionalDeriv_numeric_mpmath(
            order=order,
            prior=prior,
            params=params,
            t=t,
            method=integerDeriv_method,
            simplify=simplify,
            return_log=log,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown method: '{method}'. Choose 'scipy', 'mpmath', or 'symbolic'.")

def mgfDerivative(
    order: float,
    prior: str,
    method: str = "symbolic",
    t: float = float('nan'),
    params: dict = None,
    simplify: bool = False,
    log: bool = True,
    integer_method: str = "symbolic",
    use_interpolation: bool = True,
    d_vec: tuple = (0.8, 0.9, 0.95),
    int_tol: float = 1e-12,
    **kwargs
):
    """
    Unified wrapper for integer or fractional derivatives of the MGF.

    This function automatically detects whether the derivative order is integer
    or fractional and dispatches to the appropriate method (integer or fractional).
    For integer orders, it can use symbolic, Bell-polynomial, or JAX methods.
    For fractional orders, it can use scipy, mpmath, or symbolic methods.

    Special handling for Pareto prior:
        - If method='jax' is selected, it is overridden to 'symbolic' with a warning
          because JAX's `jet` does not support the required incomplete gamma function.
        - If method='bell' is selected, a warning is issued about potential slowness
          due to symbolic CGF derivatives, but the method is not overridden.

    Parameters
    ----------
    order : float
        Derivative order (integer or fractional). If within `int_tol` of an integer,
        it is treated as integer.
    prior : str
        Prior name. Must be one of 'gamma', 'pareto' (or others added to the registry).
    method : str, optional
        For integer order: one of 'symbolic', 'bell', 'jax'.
        For fractional order: one of 'scipy', 'mpmath', 'symbolic'.
        Default 'symbolic'.
    t : float, optional
        Evaluation point. Required for numeric methods ('bell', 'jax', 'scipy', 'mpmath').
        For 'symbolic', if provided together with `params`, the symbolic expression is
        evaluated numerically; otherwise, a symbolic expression is returned.
    params : dict, optional
        Prior parameters. Required for numeric methods. For 'symbolic', if provided,
        the symbolic expression is evaluated numerically; otherwise, a symbolic expression
        is returned.
    simplify : bool, optional
        If True, simplify the symbolic expression (only for 'symbolic' methods).
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
        - If method='symbolic' and no numeric evaluation (t is nan or params is None):
            sympy.Expr (symbolic expression)
        - If numeric evaluation (log=True): tuple (log_abs, sign)
        - If numeric evaluation (log=False): float (ordinary value)

    Raises
    ------
    ValueError
        If order is invalid, method is invalid for the order type, required arguments missing,
        or d_vec does not have exactly 3 elements or any element >= 1.
    """
    if params is None:
        params = {}

    # ---- Extract cgf_method and symbolic_timeout from kwargs ----
    cgf_method = kwargs.pop('cgf_method', 'auto')
    symbolic_timeout = kwargs.pop('symbolic_timeout', 600.0)

    # ---- Validate d_vec ----
    if len(d_vec) != 3:
        raise ValueError("d_vec must have exactly 3 elements (e.g., (0.8, 0.9, 0.95)).")
    if any(d >= 1 for d in d_vec):
        raise ValueError("All elements of d_vec must be < 1 (to get positive deviations).")
    min_dev = 1.0 - max(d_vec)

    # ---- Special handling for Pareto prior ----
    if prior.lower() == 'pareto':
        # For integer orders, handle method selection
        # We'll check if order is integer (within tolerance) and warn/override
        is_integer = abs(order - round(order)) < int_tol
        if is_integer:
            if method.lower() == 'jax':
                import warnings
                warnings.warn(
                    "JAX (jet) does not currently support the Pareto prior. "
                    "Overriding method to 'symbolic' for integer order.",
                    UserWarning
                )
                method = 'symbolic'
            elif method.lower() == 'bell':
                import warnings
                warnings.warn(
                    "The Bell method with Pareto prior may be very slow due to symbolic CGF derivatives. "
                    "Consider using 'symbolic' for faster exact results.",
                    UserWarning
                )
                # Do not override; let the user proceed.

    # ---- Determine if order is integer ----
    is_integer = abs(order - round(order)) < int_tol

    if is_integer:
        int_order = int(round(order))
        valid_int_methods = {'symbolic', 'bell', 'jax'}
        if method.lower() not in valid_int_methods:
            raise ValueError(f"Invalid method '{method}' for integer order. Choose from {valid_int_methods}.")
        return mgfDerivative_integer(
            order=int_order,
            prior=prior,
            method=method,
            t=t,
            params=params,
            simplify=simplify,
            log=log,
            symbolic_timeout=symbolic_timeout,
            cgf_method=cgf_method
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
            return fractionalDeriv_interpolated(
                order=order,
                prior=prior,
                params=params,
                t=t,
                d_vec=d_vec,
                return_log=log,
                integer_method=integer_method,
                **scipy_kwargs
            )

        # ---- Otherwise, use standard fractional method ----
        return mgfDerivative_fractional(
            order=order,
            prior=prior,
            method=method,
            t=t,
            params=params,
            simplify=simplify,
            log=log,
            integerDeriv_method=integer_method,
            **kwargs
        )

if __name__ == "__main__":
    import math
    import pandas as pd

    # ===== Integer derivative examples =====
    print("=" * 60)
    print("Integer derivative examples")
    print("=" * 60)

    # 1. Symbolic expression (no evaluation)
    expr = mgfDerivative_integer(2, "gamma", method="symbolic")
    print("Symbolic expression for 2nd derivative of Gamma MGF:")
    sp.pprint(expr)

    # 2. Symbolic evaluation with numeric output (log=True by default)
    log_abs, sign = mgfDerivative_integer(
        2, "gamma", method="symbolic", t=-1.0, params={'alpha': 2.0, 'beta': 3.0}
    )
    print(f"\nSymbolic evaluated (log scale): log|deriv| = {log_abs:.6f}, sign = {sign}")

    # 3. Symbolic evaluation with ordinary output (log=False)
    val = mgfDerivative_integer(
        2, "gamma", method="symbolic", t=-1.0, params={'alpha': 2.0, 'beta': 3.0}, log=False
    )
    print(f"Symbolic evaluated (ordinary scale): {val:.6f}")

    # 4. Bell method (numeric, log=True default)
    log_abs, sign = mgfDerivative_integer(
        2, "gamma", method="bell", t=-1.0, params={'alpha': 2.0, 'beta': 3.0}
    )
    print(f"\nBell method: log|deriv| = {log_abs:.6f}, sign = {sign}")

    # 5. JAX method with ordinary output
    val = mgfDerivative_integer(
        2, "gamma", method="jax", t=-1.0, params={'alpha': 2.0, 'beta': 3.0}, log=False
    )
    print(f"JAX method (ordinary): {val:.6e}")

    # ===== Fractional derivative examples =====
    print("\n" + "=" * 60)
    print("Fractional derivative examples")
    print("=" * 60)

    frac_order = 1.99
    params_gamma = {'alpha': 2.0, 'beta': 3.0}
    t_val = -1.0

    # 6. Symbolic fractional derivative (warning will be printed)
    print("\n--- Symbolic fractional derivative ---")
    try:
        expr_frac = mgfDerivative_fractional(
            order=frac_order,
            prior='gamma',
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
        prior='gamma',
        method='scipy',
        t=t_val,
        params=params_gamma,
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
            prior='gamma',
            method='mpmath',
            t=t_val,
            params=params_gamma,
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
        2, "gamma", method="symbolic", t=t_val, params=params_gamma, log=True
    )
    deriv2 = sign2 * math.exp(log_abs2)
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
        prior='gamma',
        method='symbolic',
        t=-1.0,
        params={'alpha': 2.0, 'beta': 3.0}
    )
    print(f"Wrapper (integer, log): log|deriv| = {log_abs:.6f}, sign = {sign}")

    # Fractional order via wrapper (auto‑corrected)
    log_abs, sign = mgfDerivative(
        order=1.99,
        prior='gamma',
        method='jax',          # auto‑corrected to integer_method='jax', method='scipy'
        t=-1.0,
        params={'alpha': 2.0, 'beta': 3.0},
        epsrel=1e-10
    )
    print(f"Wrapper (fractional, auto‑corrected): log|deriv| = {log_abs:.6f}, sign = {sign}")

    # Fractional order with explicit scipy method
    log_abs, sign = mgfDerivative(
        order=1.99,
        prior='gamma',
        method='scipy',
        t=-1.0,
        params={'alpha': 2.0, 'beta': 3.0},
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
    prior_params_interp = {'alpha': 1.0, 'beta': 0.9}
    t_interp = -1.0
    order_interp = 1.999

    # Using interpolation (default)
    log_abs_interp, sign_interp = mgfDerivative(
        order=order_interp,
        prior='gamma',
        method='scipy',
        t=t_interp,
        params=prior_params_interp,
        log=True,
        integer_method='symbolic',
        use_interpolation=True,
        d_vec=(0.8, 0.9, 0.95),
        epsrel=1e-10
    )
    print(f"Interpolated: log|deriv| = {log_abs_interp:.6f}, sign = {sign_interp}")

    # Analytic formula for exponential prior: D^α M(t) = λ * Γ(α+1) * (λ - t)^(-α-1)
    lambda_exp = prior_params_interp['beta']
    log_analytic = math.log(lambda_exp) + math.lgamma(order_interp + 1) - (order_interp + 1) * math.log(lambda_exp - t_interp)
    print(f"Analytic:     log|deriv| = {log_analytic:.6f}")
    print(f"Difference (interp - analytic): {log_abs_interp - log_analytic:.2e}")