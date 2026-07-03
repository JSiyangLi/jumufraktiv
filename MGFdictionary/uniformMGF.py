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
import numpy as np

from jumufraktiv.logsum import logplus, logminus
from jumufraktiv.registry import register_prior, make_prior_spec
from jumufraktiv.symbols import t, theta, param


# ============================================================
# Canonical symbolic parameters
# ============================================================
a = param("a")
b = param("b")


# ============================================================
# Symbolic expressions
# ============================================================

def uniform_mgf_symbolic():
    """
    M(t) = (exp(t*b) - exp(t*a)) / (t*(b-a))
    """
    return (sp.exp(t * b) - sp.exp(t * a)) / (t * (b - a))


def uniform_cgf_symbolic():
    """
    K(t) = log( (exp(t*b) - exp(t*a)) / (t*(b-a)) )
    """
    return sp.log(uniform_mgf_symbolic())


def uniform_pdf_symbolic():
    """
    p(theta) = 1/(b-a),  for a <= theta <= b
             = 0,          otherwise
    """
    return sp.Piecewise((1 / (b - a), (theta >= a) & (theta <= b)), (0, True))


# ============================================================
# Numeric CGF / MGF
# ============================================================

def uniform_cgf(t_val: float, a_val: float, b_val: float) -> float:
    if t_val == 0.0:
        return 0.0
    return math.log(math.exp(t_val * b_val) - math.exp(t_val * a_val)) - math.log(t_val * (b_val - a_val))


def uniform_mgf(t_val: float, a_val: float, b_val: float) -> float:
    return math.exp(uniform_cgf(t_val, a_val, b_val))


# ============================================================
# JAX versions
# ============================================================

def uniform_cgf_jax(t_val, a_val, b_val):
    return jnp.log((jnp.exp(t_val * b_val) - jnp.exp(t_val * a_val)) / (t_val * (b_val - a_val)))


def uniform_mgf_jax(t_val, a_val, b_val):
    return jnp.exp(uniform_cgf_jax(t_val, a_val, b_val))


# ============================================================
# SciPy PDF / logPDF (using scipy.stats.uniform)
# ============================================================

def uniform_pdf(theta_val: float, a_val: float, b_val: float) -> float:
    return stats.uniform(loc=a_val, scale=b_val - a_val).pdf(theta_val)


def uniform_logpdf(theta_val: float, a_val: float, b_val: float) -> float:
    return stats.uniform(loc=a_val, scale=b_val - a_val).logpdf(theta_val)


# ============================================================
# Registry factory
# ============================================================

@register_prior("uniform")
def uniform_factory(params):
    a_val = float(params["a"])
    b_val = float(params["b"])

    # Build symbolic expressions using the global symbols
    mgf_sym = (sp.exp(t * b) - sp.exp(t * a)) / (t * (b - a))
    cgf_sym = sp.log(mgf_sym)
    pdf_sym = sp.Piecewise((1 / (b - a), (theta >= a) & (theta <= b)), (0, True))

    # Substitute numeric parameter values into the symbolic expressions
    subs_map = {a: a_val, b: b_val}
    mgf_sym = mgf_sym.subs(subs_map)
    cgf_sym = cgf_sym.subs(subs_map)
    pdf_sym = pdf_sym.subs(subs_map)

    # Return the spec using make_prior_spec
    return make_prior_spec(
        mgf_sym=mgf_sym,
        cgf_sym=cgf_sym,
        pdf_sym=pdf_sym,

        mgf=lambda t_val: uniform_mgf(t_val, a_val, b_val),
        cgf=lambda t_val: uniform_cgf(t_val, a_val, b_val),

        mgf_jax=lambda t_val: uniform_mgf_jax(t_val, a_val, b_val),
        cgf_jax=lambda t_val: uniform_cgf_jax(t_val, a_val, b_val),

        pdf_func=lambda x: uniform_pdf(x, a_val, b_val),
        logpdf_func=lambda x: uniform_logpdf(x, a_val, b_val),

        params=params,
    )