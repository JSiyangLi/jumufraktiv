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
import numpy as np

from jumufraktiv.logsum import logplus, logminus
from jumufraktiv.registry import register_prior, make_prior_spec
from jumufraktiv.symbols import t, theta, param


# ============================================================
# Canonical symbolic parameters
# ============================================================
k = param("k")


# ============================================================
# Symbolic expressions
# ============================================================

def heaviside_mgf_symbolic():
    """
    M(t) = -exp(k*t)/t,  for t < 0
    """
    return -sp.exp(k * t) / t


def heaviside_cgf_symbolic():
    """
    K(t) = log(-1/t) + k*t, for t < 0
    """
    return sp.log(-1 / t) + k * t


def heaviside_pdf_symbolic():
    """
    p(theta) = 1,  theta >= k
             = 0,  otherwise
    """
    return sp.Piecewise((1, theta >= k), (0, True))


# ============================================================
# Numeric CGF / MGF (log-space stable core)
# ============================================================

def heaviside_cgf(t_val: float, k_val: float) -> float:
    if t_val >= 0:
        raise ValueError("t must be negative for Heaviside MGF.")
    return math.log(-1.0 / t_val) + k_val * t_val


def heaviside_mgf(t_val: float, k_val: float) -> float:
    return math.exp(heaviside_cgf(t_val, k_val))


# ============================================================
# JAX versions
# ============================================================

def heaviside_cgf_jax(t_val, k_val):
    return jnp.log(-1.0 / t_val) + k_val * t_val


def heaviside_mgf_jax(t_val, k_val):
    return jnp.exp(heaviside_cgf_jax(t_val, k_val))


# ============================================================
# SciPy PDF / logPDF (not available for improper Heaviside)
# ============================================================

def heaviside_pdf(theta_val: float, k_val: float) -> float:
    return 1.0 if theta_val >= k_val else 0.0


def heaviside_logpdf(theta_val: float, k_val: float) -> float:
    return 0.0 if theta_val >= k_val else -np.inf


# ============================================================
# Registry factory
# ============================================================

@register_prior("heaviside")
def heaviside_factory(params):
    k_val = float(params["k"])

    # Build symbolic expressions using the global symbols
    mgf_sym = -sp.exp(k * t) / t
    cgf_sym = sp.log(-1 / t) + k * t
    pdf_sym = sp.Piecewise((1, theta >= k), (0, True))

    # Substitute numeric parameter values into the symbolic expressions
    subs_map = {k: k_val}
    mgf_sym = mgf_sym.subs(subs_map)
    cgf_sym = cgf_sym.subs(subs_map)
    pdf_sym = pdf_sym.subs(subs_map)

    # Return the spec using make_prior_spec
    return make_prior_spec(
        mgf_sym=mgf_sym,
        cgf_sym=cgf_sym,
        pdf_sym=pdf_sym,

        mgf=lambda t_val: heaviside_mgf(t_val, k_val),
        cgf=lambda t_val: heaviside_cgf(t_val, k_val),

        mgf_jax=lambda t_val: heaviside_mgf_jax(t_val, k_val),
        cgf_jax=lambda t_val: heaviside_cgf_jax(t_val, k_val),

        pdf_func=lambda x: heaviside_pdf(x, k_val),
        logpdf_func=lambda x: heaviside_logpdf(x, k_val),

        params=params,
    )