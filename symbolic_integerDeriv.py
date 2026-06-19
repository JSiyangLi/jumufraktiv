"""
Integer-order symbolic differentiation of moment generating functions.

This module provides a function to compute symbolic derivatives of MGFs
for Gamma and Pareto priors.
"""

import sympy as sp
from MGFdictionary.gammaMGF import gamma_mgf_symbolic
from MGFdictionary.paretoMGF import pareto_mgf_symbolic


def integerDeriv_symbolic(order: int, prior: str, simplify: bool = False):
    """
    Returns the symbolic derivative of order `order` of the MGF w.r.t. t.

    Parameters
    ----------
    order : int
        Order of differentiation (non‑negative integer).
    prior : str
        One of 'gamma' or 'pareto'.
    simplify : bool, optional
        If True, simplify the resulting expression (default False).

    Returns
    -------
    sympy.Expr
        The derivative expression.

    Raises
    ------
    ValueError
        If prior is not recognised or order is negative.
    """
    if order < 0:
        raise ValueError("Order of derivative must be non‑negative.")

    # Select the MGF expression
    if prior.lower() == "gamma":
        expr = gamma_mgf_symbolic()
    elif prior.lower() == "pareto":
        expr = pareto_mgf_symbolic()
    else:
        raise ValueError("prior must be 'gamma' or 'pareto'")

    # Find the symbol representing t (the differentiation variable)
    # We assume it is named 't' and appears in the expression.
    t_symbols = [sym for sym in expr.free_symbols if sym.name == 't']
    if not t_symbols:
        raise RuntimeError("No symbol 't' found in the MGF expression.")
    t = t_symbols[0]  # assume only one

    # Compute derivative
    derivative = sp.diff(expr, t, order)

    if simplify:
        derivative = sp.simplify(derivative)

    return derivative


# ----------------------------------------------------------------------
# Test: simplified derivatives, orders 0–3
# ----------------------------------------------------------------------
if __name__ == "__main__":
    max_order = 2
    priors = ["gamma", "pareto"]

    for prior in priors:
        print(f"\n{'='*60}")
        print(f"PRIOR: {prior.upper()}")
        print('='*60)
        for order in range(max_order + 1):
            deriv = integerDeriv_symbolic(order, prior, simplify=True)
            print(f"\nOrder {order} derivative (simplified):")
            sp.pprint(deriv, use_unicode=False)

        print("\n" + "-"*60 + "\n")

if __name__ == "__main__":
    import math
    # ---- NEW: 75th derivative of Gamma with simplify=True ----
    print("\n" + "="*60)
    print("Testing 175th derivative of Gamma MGF with simplify=True")
    print("(This may take a very long time due to simplification)")
    print("="*60)

    import time
    order_high = 175

    # This line calls integerDeriv_symbolic with simplify=True
    # ⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️
    deriv_expr = integerDeriv_symbolic(order_high, "gamma", simplify=False)
    # ⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️

    print(f"Simplified derivative expression obtained.")
    print(f"Expression size (operations): {sp.count_ops(deriv_expr)}")

    # Preview the expression (it should be compact after simplification)
    expr_str = str(deriv_expr)
    if len(expr_str) > 500:
        print(f"Preview: {expr_str[:500]}... (truncated)")
    else:
        print("Derivative expression:")
        sp.pprint(deriv_expr, use_unicode=False)

    # Evaluate numerically with small parameters
    alpha_small = 1e-5
    beta_small = 1e-5
    t_small = -1e3

    t_sym = next(s for s in deriv_expr.free_symbols if s.name == 't')
    alpha_sym = next(s for s in deriv_expr.free_symbols if s.name == 'alpha')
    beta_sym = next(s for s in deriv_expr.free_symbols if s.name == 'beta')

    subs_dict = {alpha_sym: alpha_small, beta_sym: beta_small}
    full_expr = deriv_expr.subs(subs_dict).subs({t_sym: t_small})

    print("Attempting numeric evaluation...")
    try:
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("Evaluation timed out")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        numeric_val = full_expr.evalf()
        signal.alarm(0)
        log_abs = math.log(abs(float(numeric_val)))
        sign = 1 if float(numeric_val) > 0 else -1
        print(f" Numeric result: log|deriv| = {log_abs:.6e}, sign = {sign}")
    except TimeoutError:
        print(" Evaluation timed out after 30 seconds.")
        print(" Simplify might still be too heavy; try numeric methods.")
    except Exception as e:
        print(f" Evaluation failed: {e}")