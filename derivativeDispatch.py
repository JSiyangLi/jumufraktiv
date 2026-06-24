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
    log: bool = True
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
        Evaluation point. Required for 'bell' and 'jax'. For 'symbolic', if provided
        together with params, the symbolic expression is evaluated numerically.
    params : dict, optional
        Distribution parameters. Required for 'bell' and 'jax' unless evaluating
        symbolic expression.
    simplify : bool, optional
        If True, simplify the symbolic expression (only for 'symbolic' method).
    log : bool, optional
        If True and output is numeric, return (log_abs, sign).
        If False, return the ordinary‑scale value as a float.

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
        log_abs, sign = integerDeriv_numeric_bell(t, prior, params, order)
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
    **kwargs
):
    if params is None:
        params = {}

    tol = 1e-12
    is_integer = abs(order - round(order)) < tol

    if is_integer:
        int_order = int(round(order))
        valid_int_methods = {'symbolic', 'bell', 'jax'}
        if method.lower() not in valid_int_methods:
            raise ValueError(f"Invalid method '{method}' for integer order. Choose from {valid_int_methods}.")
        return mgfDerivative_integer(
            order=int_order, prior=prior, method=method,
            t=t, params=params, simplify=simplify, log=log
        )
    else:
        # Fractional order
        valid_frac_methods = {'scipy', 'mpmath', 'symbolic'}
        # Auto‑correct if user passed 'jax' or 'bell' as method
        if method.lower() in {'jax', 'bell'}:
            print(f"⚠️ ⚠️ Warning: For fractional order, 'method' should be one of {valid_frac_methods}. ⚠️ ⚠️"
                  f"Interpreting 'method' as 'integer_method' and using default fractional method 'scipy'.")
            integer_method = method
            method = 'scipy'
        elif method.lower() not in valid_frac_methods:
            raise ValueError(f"Invalid method '{method}' for fractional order. Choose from {valid_frac_methods}.")

        # Validate integer_method
        valid_int_methods = {'symbolic', 'jax', 'bell'}
        if integer_method.lower() not in valid_int_methods:
            raise ValueError(f"Invalid integer_method '{integer_method}'. Choose from {valid_int_methods}.")

        return mgfDerivative_fractional(
            order=order, prior=prior, method=method,
            t=t, params=params, simplify=simplify, log=log,
            integerDeriv_method=integer_method, **kwargs
        )
    

if __name__ == "__main__":
    import math

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
    print("\n" + "="*60)
    print("Unified wrapper examples")
    print("="*60)

    # Integer order via wrapper
    log_abs, sign = mgfDerivative(
        order=2.0,
        prior='gamma',
        method='symbolic',
        t=-1.0,
        params={'alpha': 2.0, 'beta': 3.0}
    )
    print(f"Wrapper (integer, log): log|deriv| = {log_abs:.6f}, sign = {sign}")

    # Fractional order via wrapper (with auto‑correction)
    log_abs, sign = mgfDerivative(
        order=1.99,
        prior='gamma',
        method='jax',          # This will be auto‑corrected to integer_method='jax', method='scipy'
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