"""
symbolic_integerDeriv.py

Symbolic differentiation of moment-generating functions (MGFs) with respect to t.

This module provides a single function `integerDeriv_symbolic` that computes
the symbolic derivative of order `order` of either the complete MGF or the
incomplete MGF (iMGF) of a given prior. The derivative is returned as a
SymPy expression.

The function respects the **symbol-numeric principle** in the sense that
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

from jumufraktiv.MGFPrior_class import MGFPrior
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
    An integer-valued order is accepted whatever its type -- Python ``int``,
    NumPy integer, or a SymPy expression that evaluates to an integer and
    carries no free symbols -- and is always returned as a Python ``int``. A
    boolean is refused, as is a genuinely fractional order.

    An order carrying free symbols is refused because ``sympy.diff(expr, t, n)``
    needs a concrete number of times to differentiate; SymPy differentiates
    symbolically, but it cannot do so an unspecified number of times, and there
    is no formula in ``n`` to return.
    """
    if isinstance(order, bool):
        raise NotImplementedError(
            "Derivative order must be an integer, not a boolean."
        )
    # The accepted types are wider than `isinstance(order, int)` on purpose:
    # the dispatcher passes on whatever the caller supplied, and SymPy
    # arithmetic produces `sympy.Integer` routinely, so narrowing the check
    # back to Python's `int` makes the symbolic backend unreachable for
    # ordinary integer orders.
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


def integerDeriv_symbolic(
    order: int,
    prior: MGFPrior,
    simplify: bool = False,
    complete: bool = True,
):
    """
    Returns the symbolic derivative of order `order` of the MGF (or incomplete MGF)
    with respect to t.

    Parameters
    ----------
    order : int
        Order of differentiation: non-negative and integer-valued. A NumPy
        integer, or a SymPy expression that evaluates to an integer and carries
        no free symbols, is accepted as well.
    prior : MGFPrior
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
    NotImplementedError
        If order is not integer-valued, or still carries free symbols.
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
            raise ValueError(
                "Prior does not provide a symbolic incomplete MGF (imgf_sym)."
            )

    if t not in expr.free_symbols:
        raise RuntimeError("Symbol 't' not found in the chosen expression.")

    derivative = cached_diff(expr, t, order)

    if simplify:
        derivative = sp.simplify(derivative)

    return derivative
