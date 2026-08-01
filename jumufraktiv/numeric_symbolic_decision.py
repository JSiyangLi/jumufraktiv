"""
suggest_methods.py

Heuristic utilities to decide between symbolic (SymPy) and numeric methods for:
- Integer differentiation of a symbolic expression
- Mellin transform of a symbolic expression

Uses "try and measure" approach: attempts symbolic computation with a timeout,
then reports time and expression complexity to guide choice.
"""

import time
import sympy as sp
from sympy.integrals.transforms import mellin_transform
import scipy.integrate as integrate
import numpy as np

from jumufraktiv.symbolic_cache import cached_diff

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
        else: prints and returns None
    """
    if test_order is None:
        test_order = min(order, 2)   # test low order
    if test_order <= 0:
        test_order = 1

    print(f"🔬 Testing symbolic derivative of order {test_order} (target order: {order})...")
    start = time.time()
    try:
        deriv = cached_diff(expr, symbol, test_order)
        elapsed = time.time() - start
        complexity = sp.count_ops(deriv)
        print(f"   ✅ Test succeeded in {elapsed:.3f}s, complexity={complexity}")

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
        print(f"   ❌ Test failed: {e}")
        recommend = False
        msg = "❌ Symbolic test failed. Use numeric (JAX)."
        elapsed = timeout
        complexity = -1

    if return_decision:
        return {'recommend_symbolic': recommend, 'elapsed': elapsed,
                'complexity': complexity, 'message': msg, 'test_order': test_order}
    else:
        print(msg)
        return None

def suggest_method_Mellin(expr, x, s, numeric_params=None, timeout=2.0):
    """
    Suggest whether to use symbolic Mellin transform or numerical integration.

    Parameters:
        expr           : sympy expression (function of x)
        x              : sympy Symbol, integration variable
        s              : sympy Symbol, Mellin parameter
        numeric_params : dict with keys:
                         's_val' : numeric value of s to test numerical integration
                         (optional) other parameters in expr as numeric values
        timeout        : float, max time for symbolic attempt

    Returns:
        None (prints recommendation)
    """
    print("🔬 Testing symbolic Mellin transform...")
    start = time.time()
    try:
        # Try symbolic Mellin transform
        result = mellin_transform(expr, x, s)
        elapsed = time.time() - start
        # result is tuple (F(s), convergence_conditions, auxiliary_conditions)
        F_s, cond, aux = result
        complexity = sp.count_ops(F_s) if F_s else 0
        print(f"   ✅ Symbolic succeeded in {elapsed:.3f}s, complexity={complexity}")
        if elapsed < 0.5 and complexity < 100:
            print("   ✅ RECOMMEND: Symbolic Mellin transform is simple and fast.")
        elif elapsed < timeout and complexity < 500:
            print("   ⚠️  Symbolic transform is possible but may be complex. Consider if closed‑form is needed.")
        else:
            print("   ❌ NOT RECOMMENDED: Symbolic transform is too slow or gives huge expression.")
            print("   💡 Use numerical integration (e.g., scipy.integrate.quad).")
    except Exception as e:
        print(f"   ❌ Symbolic failed: {e}")
        print("   💡 Use numerical integration.")

    # Optionally, test numerical integration for a specific s value
    if numeric_params is not None and 's_val' in numeric_params:
        s_val = numeric_params['s_val']
        # Create a numeric function from expr (substitute any other numeric parameters)
        subs_dict = {k: v for k, v in numeric_params.items() if k != 's_val'}
        expr_num = expr.subs(subs_dict)
        f = sp.lambdify((x, s), expr_num, modules='numpy')
        # Wrapper for quad
        def integrand(x_val):
            return f(x_val, s_val) * x_val**(s_val - 1)
        try:
            res, err = integrate.quad(integrand, 0, np.inf)
            print(f"   🔢 Numerical integration (quad) for s={s_val}: {res:.6f} ± {err:.2e}")
            print("   💡 Use numerical integration if accuracy is acceptable and symbolic is heavy.")
        except Exception as e2:
            print(f"   🔢 Numerical integration failed: {e2}")

# ----------------------------------------------------------------------
# Example usage (when run as a script)
# ----------------------------------------------------------------------
