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
    max_order = 3
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