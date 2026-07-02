"""
Integer-order symbolic differentiation of moment generating functions.

This module provides a function to compute symbolic derivatives of MGFs
for any prior that provides a symbolic MGF expression (mgf_sym).
"""

import sympy as sp
from jumufraktiv.mitMGFprior_class import mitMGFprior


def integerDeriv_symbolic(order: int, prior: mitMGFprior, simplify: bool = False):
    """
    Returns the symbolic derivative of order `order` of the MGF w.r.t. t.

    Parameters
    ----------
    order : int
        Order of differentiation (non‑negative integer).
    prior : mitMGFprior
        Prior object providing the symbolic MGF expression (mgf_sym).
    simplify : bool, optional
        If True, simplify the resulting expression (default False).

    Returns
    -------
    sympy.Expr
        The derivative expression.

    Raises
    ------
    ValueError
        If order is negative.
    RuntimeError
        If no 't' symbol is found in the MGF expression.
    """
    if order < 0:
        raise ValueError("Order of derivative must be non‑negative.")

    # Get the symbolic MGF expression from the prior object
    if not hasattr(prior, "mgf_sym") or prior.mgf_sym is None:
        raise ValueError("Prior does not provide a symbolic MGF (mgf_sym).")

    expr = prior.mgf_sym

    # Find the symbol representing t
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
# Test: simplified derivatives, orders 0–3, and high-order check
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import math
    import time
    from jumufraktiv.mitMGFprior_class import mitMGFprior
    import jumufraktiv.MGFdictionary  # necessary to import priors!

    # Create a Gamma prior via registry
    gamma_prior = mitMGFprior.from_registry(
        "gamma",
        params={"alpha": 2.0, "beta": 3.0}
    )

    # ---------- Low-order derivatives ----------
    max_order = 3
    print("Testing low-order derivatives (Gamma prior):")
    print("=" * 60)
    for order in range(max_order + 1):
        deriv = integerDeriv_symbolic(order, gamma_prior, simplify=True)
        print(f"\nOrder {order} derivative (simplified):")
        sp.pprint(deriv, use_unicode=False)

    # ---------- High-order derivative (175th) ----------
    print("\n" + "=" * 60)
    print("Testing 175th derivative of Gamma MGF with simplify=False")
    print("(This may take a very long time due to expression size)")
    print("=" * 60)

    order_high = 175

    # Create a prior with very small parameters for the high-order test
    small_prior = mitMGFprior.from_registry(
        "gamma",
        params={"alpha": 1e-5, "beta": 1e-5}
    )

    deriv_expr = integerDeriv_symbolic(order_high, small_prior, simplify=False)

    print(f"Derivative expression obtained.")
    print(f"Expression size (operations): {sp.count_ops(deriv_expr)}")

    # Preview (truncated if too large)
    expr_str = str(deriv_expr)
    if len(expr_str) > 500:
        print(f"Preview: {expr_str[:500]}... (truncated)")
    else:
        print("Derivative expression:")
        sp.pprint(deriv_expr, use_unicode=False)

    # Evaluate numerically with the small parameters
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