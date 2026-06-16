import math
import sympy as sp
import scipy.special as sc

import sympy as sp
from sympy.functions.special.error_functions import expint

def pareto_mgf_symbolic():
    """
    Returns a SymPy expression for the Pareto MGF using the exponential integral.
        M(t) = alpha * E_{alpha+1}(-xi*t),   for t <= 0
    """
    t = sp.Symbol('t', real=True, nonpositive=True)
    alpha = sp.Symbol('alpha', positive=True, real=True)
    xi = sp.Symbol('xi', positive=True, real=True)

    z = -xi * t          # z >= 0
    mgf = alpha * expint(alpha + 1, z)
    return mgf

def log_pareto_mgf(t: float, alpha: float, xi: float) -> float:
    """
    Return log M(t) for the Pareto distribution using SciPy's `log_gammaincc`.
    """
    if t > 0:
        raise ValueError("MGF of Pareto distribution exists only for t <= 0")
    if t == 0.0:
        return 0.0
    
    z = -xi * t  # z > 0
    # log(alpha) + alpha * log(z) + log(Γ(-alpha, z))
    # but we compute log(Γ(-alpha, z)) via log_gammaincc
    log_gamma_inc = sc.log_gammaincc(-alpha, z) + sc.gammaln(-alpha)
    return math.log(alpha) + alpha * math.log(z) + log_gamma_inc

def pareto_mgf(t: float, alpha: float, xi: float) -> float:
    """
    Returns the Pareto MGF: M(t) = exp(log_pareto_mgf(t, alpha, xi)).
    """
    return math.exp(log_pareto_mgf(t, alpha, xi))