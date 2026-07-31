"""
symbolic_integerDeriv.py

Symbolic differentiation of moment‑generating functions (MGFs) with respect to t.

This module provides a single function `integerDeriv_symbolic` that computes
the symbolic derivative of order `order` of either the complete MGF or the
incomplete MGF (iMGF) of a given prior. The derivative is returned as a
SymPy expression.

The function respects the **symbol‑numeric principle** in the sense that
it always returns a symbolic expression; numeric evaluation is handled by
the caller (e.g., by substituting numeric values for `t` and evaluating
with `.evalf()`).

Supports:
    - Complete MGF (`prior.mgf_sym`)
    - Incomplete MGF (`prior.imgf_sym`) via the `complete=False` flag.

The module uses the canonical symbol `t` from `jumufraktiv.symbols`.
"""

import numpy as np
import sympy as sp
from jumufraktiv.mitMGFprior_class import mitMGFprior
from jumufraktiv.symbolic_cache import cached_diff
from jumufraktiv.symbols import t  # only t is needed for differentiation


def _as_integer_order(order):
    """Coerce an integer-valued derivative order to a Python ``int``.

    Parameters
    ----------
    order : int, sympy.Integer, numpy integer, or sympy.Expr
        The requested order.

    Returns
    -------
    int
        The same order as a Python integer.

    Raises
    ------
    NotImplementedError
        If the order is not integer-valued, or still contains free symbols.

    Notes
    -----
    The previous guard was ``isinstance(order, int)``, which rejected
    ``sympy.Integer(2)`` -- an integer by every meaning except Python's type
    check. Since the dispatcher hands this function whatever the caller passed,
    and SymPy arithmetic naturally produces ``sympy.Integer``, that made the
    symbolic row of the backend matrix unreachable even for ordinary integer
    orders.

    A genuinely symbolic order is a different matter and is still refused, but
    with an accurate reason. ``sympy.diff(expr, t, n)`` needs a concrete number
    of times to differentiate; it cannot produce a formula in ``n``. The old
    message called this a limitation of SymPy's support for "symbolic
    differentiation", which is not what is missing -- SymPy differentiates
    symbolically, it just cannot do so an unspecified number of times.
    """
    if isinstance(order, bool):
        raise NotImplementedError(
            "Derivative order must be an integer, not a boolean."
        )
    if isinstance(order, (int, np.integer)):
        return int(order)

    if isinstance(order, sp.Basic):
        if order.free_symbols:
            raise NotImplementedError(
                f"Cannot differentiate a symbolic number of times (order={order}). "
                "sympy.diff needs a concrete integer order; there is no closed "
                "form in the order itself. Substitute a value for "
                f"{sorted(map(str, order.free_symbols))} before calling, or use a "
                "fractional backend ('scipy' or 'mpmath') with a numeric order."
            )
        if order.is_Integer:
            return int(order)
        raise NotImplementedError(
            f"The symbolic backend differentiates an integer number of times, so "
            f"it cannot serve order={order}. Use method='scipy' or 'mpmath' for "
            "fractional orders."
        )

    as_float = float(order)
    if as_float.is_integer():
        return int(as_float)
    raise NotImplementedError(
        f"The symbolic backend differentiates an integer number of times, so it "
        f"cannot serve order={order}. Use method='scipy' or 'mpmath' for "
        "fractional orders."
    )


def integerDeriv_symbolic(order: int, prior: mitMGFprior, simplify: bool = False, complete: bool = True):
    """
    Returns the symbolic derivative of order `order` of the MGF (or incomplete MGF)
    with respect to t.

    Parameters
    ----------
    order : int
        Order of differentiation (non‑negative integer).
    prior : mitMGFprior
        Prior object providing the symbolic MGF expression (mgf_sym) and optionally
        the incomplete MGF expression (imgf_sym).
    simplify : bool, optional
        If True, simplify the resulting expression (default False).
    complete : bool, optional
        If True (default), differentiate the complete MGF (prior.mgf_sym).
        If False, differentiate the incomplete MGF (prior.imgf_sym).

    Returns
    -------
    sympy.Expr
        The derivative expression.

    Raises
    ------
    ValueError
        If order is negative, or if `complete=False` and `imgf_sym` is missing.
    RuntimeError
        If no 't' symbol is found in the chosen expression.
    TypeError
        If order is not an integer.
    """
    order = _as_integer_order(order)
    if order < 0:
        raise ValueError("Order of derivative must be non-negative.")

    # Select the expression based on `complete`
    if complete:
        expr = getattr(prior, "mgf_sym", None)
        if expr is None:
            raise ValueError("Prior does not provide a symbolic MGF (mgf_sym).")
    else:
        expr = getattr(prior, "imgf_sym", None)
        if expr is None:
            raise ValueError("Prior does not provide a symbolic incomplete MGF (imgf_sym).")

    if t not in expr.free_symbols:
        raise RuntimeError("Symbol 't' not found in the chosen expression.")

    derivative = cached_diff(expr, t, order)

    if simplify:
        derivative = sp.simplify(derivative)

    return derivative


# ----------------------------------------------------------------------
# Test: simplified derivatives, orders 0–3, and high-order check
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import math
    import signal
    import jumufraktiv.MGFdictionary  # necessary to import priors!
    from jumufraktiv.mitMGFprior_class import mitMGFprior
    from jumufraktiv.symbols import t, param

    alpha = param("alpha")
    beta = param("beta")

    # Build a Gamma prior using the registry
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

    small_prior = mitMGFprior.from_registry(
        "gamma",
        params={"alpha": 1e-5, "beta": 1e-5}
    )

    deriv_expr = integerDeriv_symbolic(order_high, small_prior, simplify=False)

    print(f"Derivative expression obtained.")
    print(f"Expression size (operations): {sp.count_ops(deriv_expr)}")

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

    subs_dict = {alpha: alpha_small, beta: beta_small, t: t_small}
    full_expr = deriv_expr.subs(subs_dict)

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