"""
Fractional derivatives of MGFs via Mellin transform.

Implements the Liouville‑Caputo fractional derivative using:
    D^α_{(-∞)+} f(x) = { M[ f^{(⌊α⌋+1)}(x - z) ] }(γ)
where γ = ⌊α⌋+1 - α, and M denotes the Mellin transform.

This script assumes α is non‑integer.
"""

import sympy as sp
from sympy.integrals.transforms import mellin_transform
from MGFdictionary.gammaMGF import gamma_mgf_symbolic
from MGFdictionary.paretoMGF import pareto_mgf_symbolic


def fractionalDeriv_symbolic(
    order: float,
    prior: str,
    simplify: bool = False
):
    """
    Compute the Liouville‑Caputo fractional derivative of order `order`
    of the MGF w.r.t. t using Mellin transform.

    Parameters
    ----------
    order : float
        Order of fractional differentiation (positive, non‑integer).
    prior : str
        One of 'gamma' or 'pareto'.
    simplify : bool, optional
        If True, simplify the resulting expression (default False).

    Returns
    -------
    sympy.Expr
        The symbolic expression for the fractional derivative.

    Raises
    ------
    ValueError
        If prior is not recognised or order ≤ 0.
    RuntimeError
        If Mellin transform fails to produce a closed form.
    """
    if order <= 0:
        raise ValueError("Fractional order must be positive.")

    # Select the MGF expression
    if prior.lower() == "gamma":
        expr = gamma_mgf_symbolic()
    elif prior.lower() == "pareto":
        expr = pareto_mgf_symbolic()
    else:
        raise ValueError("prior must be 'gamma' or 'pareto'")

    # Find the symbol representing t
    t_symbols = [sym for sym in expr.free_symbols if sym.name == 't']
    if not t_symbols:
        raise RuntimeError("No symbol 't' found in the MGF expression.")
    t = t_symbols[0]

    # Use `order` as alpha in the formula
    alpha = order

    # ---- Step 1: determine n = floor(alpha), gamma = n+1 - alpha ----
    n = sp.floor(alpha)
    gamma = (n + 1) - alpha
    # gamma is in (0,1) because alpha is non‑integer

    # ---- Step 2: integer derivative of order n+1 ----
    f_n = sp.diff(expr, t, int(n) + 1)

    # ---- Step 3: substitute z = t - y, so f_n(t - z) ----
    z = sp.Symbol('z', positive=True)
    g = f_n.subs(t, t - z)
    sp.pprint(g)

    # ---- Step 4: Mellin transform of g(z) w.r.t z ----
    s = sp.Symbol('s')
    try:
        mellin_result = mellin_transform(g, z, s)
    except Exception as e:
        raise RuntimeError(f"Mellin transform failed: {e}")

    F_s = mellin_result[0]

    # ---- Step 5: substitute s = gamma to get fractional derivative ----
    frac_expr = F_s.subs(s, gamma)

    if simplify:
        frac_expr = sp.simplify(frac_expr)

    return frac_expr


# ===== Example usage =====
if __name__ == "__main__":
    # Test fractional derivative of Gamma MGF
    print("Testing fractional derivative of Gamma MGF (order 1.1):")
    try:
        result = fractionalDeriv_symbolic(1.1, "gamma", simplify=True)
        print("Symbolic result:")
        sp.pprint(result)
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "-" * 60)

    # Test fractional derivative of Pareto MGF
    print("Testing fractional derivative of Pareto MGF (order 1.1):")
    try:
        result = fractionalDeriv_symbolic(1.1, "pareto", simplify=True)
        print("Symbolic result:")
        sp.pprint(result)
    except Exception as e:
        print(f"Error: {e}")