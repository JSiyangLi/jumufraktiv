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


# ===== Example usage =====
if __name__ == "__main__":
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