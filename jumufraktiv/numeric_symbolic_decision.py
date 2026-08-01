"""Decide between symbolic (SymPy) and numeric differentiation, by measuring.

The approach is "try and measure": differentiate a *low* order symbolically,
time it, count the operations in the result, and recommend from that. The
premise is that a derivative already heavy at order 2 only gets worse.

:func:`~jumufraktiv.numeric_integerDeriv_Bell.integerDeriv_numeric_bell` is
the only caller.

A companion ``suggest_method_Mellin`` used to live here, applying the same
heuristic to a symbolic Mellin transform. Nothing called it. The Mellin side
belongs to the *second* marginalisable family described in the reference --
the one whose operator has lower terminal 0 and acts on ``t^a M(log t)``,
covering Beta, Beta-prime and Dirichlet likelihoods -- and this package does
not implement that family. A helper for choosing how to compute a transform
the package never takes is not a partial feature; it is a leftover.
"""

import logging
import time

import sympy as sp

from jumufraktiv.symbolic_cache import cached_diff

logger = logging.getLogger(__name__)

def suggest_method_integerDeriv(expr, symbol, order, test_order=None, timeout=1.0, return_decision=False):
    """
    Suggest symbolic vs numeric for integer derivatives.

    Parameters:
        expr          : sympy expression
        symbol        : variable
        order         : the actual order we intend to compute
        test_order    : order to test (default: min(order, 2))
        timeout       : max time for test
        return_decision: if True, return dict with recommendation

    Returns:
        if return_decision: dict with keys 'recommend_symbolic', 'elapsed', etc.
        else: logs the recommendation at INFO and returns None
    """
    if test_order is None:
        test_order = min(order, 2)   # test low order
    if test_order <= 0:
        test_order = 1

    logger.debug(
        "Timing a symbolic derivative of order %s (target order %s).",
        test_order, order)
    start = time.time()
    try:
        deriv = cached_diff(expr, symbol, test_order)
        elapsed = time.time() - start
        complexity = sp.count_ops(deriv)
        logger.debug("Succeeded in %.3fs, complexity %s.", elapsed, complexity)

        # If the test derivative is already heavy, high order will be worse
        if elapsed < 0.1 and complexity < 100:
            recommend = True
            msg = "✅ RECOMMEND: Symbolic (SymPy) – fast and simple."
        elif elapsed < timeout and complexity < 500:
            recommend = True  # still okay, but warn
            msg = "⚠️  Symbolic is possible but may be heavy at higher orders."
        else:
            recommend = False
            msg = "❌ NOT RECOMMENDED: Symbolic test is already slow/large. Use numeric (JAX)."
    except Exception as e:
        logger.debug("Symbolic test failed (%s); recommending the numeric path.", e)
        recommend = False
        msg = "❌ Symbolic test failed. Use numeric (JAX)."
        elapsed = timeout
        complexity = -1

    if return_decision:
        return {'recommend_symbolic': recommend, 'elapsed': elapsed,
                'complexity': complexity, 'message': msg, 'test_order': test_order}
    else:
        logger.info("%s", msg)
        return None
