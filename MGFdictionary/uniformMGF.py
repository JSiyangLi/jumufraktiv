"""
uniformMGF.py

Functions for the uniform prior: p(theta) = 1/(b-a) for theta in [a, b], else 0.

The MGF is:
    M(t) = (exp(t*b) - exp(t*a)) / (t*(b-a))   for t != 0
    M(0) = 1

For t < 0, the MGF is finite and positive.
"""

import math
import sympy as sp
import jax.numpy as jnp
import scipy.stats as stats


def uniform_mgf_symbolic():
    """
    Return a SymPy expression for the uniform MGF:
        M(t) = (exp(t*b) - exp(t*a)) / (t*(b-a))
    """
    t, a, b = sp.symbols('t a b', real=True, positive=True)
    return (sp.exp(t * b) - sp.exp(t * a)) / (t * (b - a))


def uniform_cgf_symbolic():
    """
    Returns symbolic expression for the CGF of the uniform prior:
        K(t) = log( (exp(t*b) - exp(t*a)) / (t*(b-a)) )
    """
    t, a, b = sp.symbols('t a b', real=True, positive=True)
    return sp.log(uniform_mgf_symbolic())


def uniform_cgf(t: float, a: float, b: float) -> float:
    """
    Log MGF for uniform prior on [a, b].
    For t != 0: log( (exp(t*b) - exp(t*a)) / (t*(b-a)) )
    For t = 0: returns 0 (since M(0)=1).
    """
    if t == 0.0:
        return 0.0
    if t >= 0:
        # For t > 0, the MGF may be large, but we allow it for completeness.
        pass
    # For t < 0, exp(t*b) and exp(t*a) are positive, denominator negative, ratio positive.
    return math.log(math.exp(t * b) - math.exp(t * a)) - math.log(t * (b - a))


def uniform_mgf(t: float, a: float, b: float) -> float:
    """
    Return the MGF in normal scale.
    """
    return math.exp(uniform_cgf(t, a, b))


def uniform_cgf_jax(t, a, b):
    """JAX version of log M(t)."""
    return jnp.log((jnp.exp(t * b) - jnp.exp(t * a)) / (t * (b - a)))


def uniform_mgf_jax(t, a, b):
    """JAX version of M(t)."""
    return jnp.exp(uniform_cgf_jax(t, a, b))


def uniform_pdf_symbolic():
    """
    Return a SymPy expression for the uniform density:
        p(theta) = 1/(b-a)  for theta in [a, b]
    """
    theta, a, b = sp.symbols('theta a b', real=True, positive=True)
    return 1 / (b - a)


def uniform_pdf_symbolic_sub(params):
    """
    Return the symbolic uniform PDF with parameters substituted.
    params must contain 'a' and 'b'.
    """
    theta = sp.Symbol('theta', real=True)
    a = params['a']
    b = params['b']
    # The density is 1/(b-a) for theta in [a,b], but we return the constant.
    # Support condition is handled by the user.
    return sp.Integer(1) / (b - a)