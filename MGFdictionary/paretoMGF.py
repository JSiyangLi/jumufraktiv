import math
import sympy as sp
import numpy as np
import scipy.special as sc
from sympy.functions.special.error_functions import expint
import jax.numpy as jnp
from jax.scipy.special import gammaln, gammaincc

def pareto_mgf_symbolic():
    """
    Returns a SymPy expression for the Pareto MGF using the exponential integral.
        M(t) = alpha * E_{alpha+1}(-xi*t),   for t <= 0
    """
    t = sp.Symbol('t', real=True, nonpositive=True)
    alpha = sp.Symbol('alpha', positive=True, real=True)
    xi = sp.Symbol('xi', positive=True, real=True)

    z = -xi * t         # z >= 0
    mgf = alpha * expint(alpha + 1, z)
    return mgf

def pareto_cgf_symbolic():
    """
    Returns symbolic expression for the cumulant generating function (CGF)
    of the Pareto(shape=alpha, scale=xi) distribution:
        K(t) = log M(t) = log(alpha) + log(E_{alpha+1}(-xi*t))   for t <= 0
    where E_n(z) is the exponential integral.
    """
    t = sp.Symbol('t', real=True, nonpositive=True)
    alpha = sp.Symbol('alpha', positive=True, real=True)
    xi = sp.Symbol('xi', positive=True, real=True)
    z = -xi * t
    return sp.log(alpha) + sp.log(expint(alpha + 1, z))

def pareto_cgf(t: float, alpha: float, xi: float) -> float:
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
    log_gamma_inc = np.log(sc.gammaincc(-alpha, z)) + sc.gammaln(-alpha) # when sp.log_gammaincc becomes available, replace by log_gamma_inc = sc.log_gammaincc(-alpha, z) + sc.gammaln(-alpha)
    return math.log(alpha) + alpha * math.log(z) + log_gamma_inc

def pareto_mgf(t: float, alpha: float, xi: float) -> float:
    """
    Returns the Pareto MGF: M(t) = exp(pareto_cgf(t, alpha, xi)).
    """
    return math.exp(pareto_cgf(t, alpha, xi))

def pareto_cgf_jax(t, alpha, xi):
    """JAX version of log MGF for Pareto(shape=alpha, scale=xi)."""
    def safe_log(t_val):
        z = -xi * t_val
        a = -alpha
        log_gamma_a = gammaln(a)   # works for negative a
        log_q = jnp.log(gammaincc(a, z))
        log_inc_gamma = log_gamma_a + log_q
        return jnp.log(alpha) + alpha * jnp.log(z) + log_inc_gamma
    return jnp.where(t == 0.0, 0.0, safe_log(t))

def pareto_mgf_jax(t, alpha, xi):
    """JAX version of MGF (ordinary scale)."""
    return jnp.exp(pareto_cgf_jax(t, alpha, xi))