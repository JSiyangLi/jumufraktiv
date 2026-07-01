"""
heavisideMGF.py

Functions for the improper Heaviside prior: p(theta) ∝ 1 for theta >= k, else 0.

The MGF is defined for t < 0:
    M(t) = ∫_k^∞ e^{tθ} dθ = -e^{k t} / t

The CGF is log M(t) = log(-1/t) + k t, for t < 0.

This prior is improper (integral diverges), but the MGF exists for t < 0.
"""

import math
import sympy as sp
import jax.numpy as jnp
import scipy.stats as stats
from jumufraktiv.registry import register_prior
@register_prior("heaviside")


def heaviside_mgf_symbolic():
    """
    Return a SymPy expression for the Heaviside MGF:
        M(t) = -exp(k*t)/t,  for t < 0
    """
    t, k = sp.symbols('t k', real=True)
    return -sp.exp(k * t) / t


def heaviside_cgf_symbolic():
    """
    Returns symbolic expression for the CGF of the Heaviside prior:
        K(t) = log M(t) = log(-1/t) + k*t, for t < 0
    """
    t, k = sp.symbols('t k', real=True)
    return sp.log(-1 / t) + k * t


def heaviside_cgf(t: float, k: float) -> float:
    """
    Log MGF for Heaviside prior: log(-1/t) + k*t, for t < 0.
    """
    if t >= 0:
        raise ValueError("t must be negative for Heaviside MGF.")
    return math.log(-1.0 / t) + k * t


def heaviside_mgf(t: float, k: float) -> float:
    """
    Return the MGF in normal scale: -exp(k*t)/t, for t < 0.
    """
    return math.exp(heaviside_cgf(t, k))


def heaviside_cgf_jax(t, k):
    """JAX version of log M(t)."""
    return jnp.log(-1.0 / t) + k * t


def heaviside_mgf_jax(t, k):
    """JAX version of M(t)."""
    return jnp.exp(heaviside_cgf_jax(t, k))


def heaviside_pdf_symbolic():
    """
    Return a SymPy expression for the Heaviside density:
        p(theta) = 1,  theta >= k
    """
    theta, k = sp.symbols('theta k', real=True)
    # We return an expression that is 1 for theta >= k, but since it's improper,
    # we can just return 1 (with the support condition handled elsewhere).
    # For symbolic use, we'll just return 1.
    return sp.Integer(1)


def heaviside_pdf_symbolic_sub(params):
    """
    Return the symbolic Heaviside PDF with parameters substituted.
    params must contain 'k'.
    """
    theta = sp.Symbol('theta', real=True)
    k = params['k']
    # The density is 1 for theta >= k, but we return 1 as a constant.
    # The support condition is handled by the user.
    return sp.Integer(1)