import sympy as sp
import numpy as np
import scipy.special as sc
from sympy import symbols, exp, lowergamma, gamma

# ---- Define symbols ----
t, u = symbols('t u', real=True)

# ---- Expression 1: contains lowergamma (derivative of log(imgf) for Gamma) ----
alpha = 2.0
beta = 3.0
imgf = (beta / (beta - t))**alpha * lowergamma(alpha, (beta - t)*u) / gamma(alpha)
cgf = sp.log(imgf)
expr_with_lowergamma = sp.diff(cgf, t)

# ---- Expression 2: does NOT contain lowergamma ----
expr_without_lowergamma = (t + u)**2 + sp.exp(t * u)

# ---- Numeric values ----
t_val = -1.0
u_val = 2.0

def evaluate(expr, name):
    print(f"\n{name}")
    print("="*60)
    # 1) SymPy
    val_sym = expr.subs({t: t_val, u: u_val}).evalf()
    print(f"SymPy .subs().evalf():          {repr(float(val_sym))}")

    # 2) Lambdify with mpmath
    try:
        func_mp = sp.lambdify((t, u), expr, modules='mpmath')
        val_mp = func_mp(t_val, u_val)
        print(f"Lambdify (mpmath):             {repr(float(val_mp))}")
    except Exception as e:
        print(f"mpmath lambdify failed: {e}")

    # 3) Lambdify with scipy (using modules=['numpy', 'scipy'])
    try:
        func_scipy = sp.lambdify((t, u), expr, modules=['numpy', 'scipy'])
        val_scipy = func_scipy(t_val, u_val)
        print(f"Lambdify (scipy):              {repr(float(val_scipy))}")
    except Exception as e:
        print(f"scipy lambdify failed: {e}")

    return float(val_sym)

# ---- Evaluate ----
val1 = evaluate(expr_with_lowergamma, "EXPRESSION WITH lowergamma")
val2 = evaluate(expr_without_lowergamma, "EXPRESSION WITHOUT lowergamma")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Expression with lowergamma: SymPy value = {val1}")
print(f"Expression without lowergamma: SymPy value = {val2}")