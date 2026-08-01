"""
heavisideMGF.py

Functions for the improper Heaviside prior: p(theta) ∝ 1 for theta >= k, else 0.

The MGF is defined for t < 0:
    M(t) = ∫_k^∞ e^{tθ} dθ = -e^{k t} / t

The CGF is log M(t) = log(-1/t) + k t, for t < 0.

This prior is improper (integral diverges), but the MGF exists for t < 0.
"""

import math

import jax.numpy as jnp
import numpy as np
import sympy as sp

from jumufraktiv.registry import make_prior_spec, register_prior
from jumufraktiv.symbols import param, t, theta

# ============================================================
# Canonical symbolic parameters
# ============================================================
k = param("k")


# ============================================================
# Numeric CGF / MGF (log-space stable core)
# ============================================================

def heaviside_cgf(t_val: float, k_val: float) -> float:
    """
    Numeric CGF for the Heaviside prior.

    Parameters
    ----------
    t_val : float
        Evaluation point (must be negative).
    k_val : float
        Threshold parameter.

    Returns
    -------
    float
        log M(t).
    """
    if t_val >= 0:
        raise ValueError("t must be negative for Heaviside MGF.")
    return math.log(-1.0 / t_val) + k_val * t_val


def heaviside_mgf(t_val: float, k_val: float) -> float:
    """
    Numeric MGF for the Heaviside prior.

    Parameters
    ----------
    t_val : float
        Evaluation point (must be negative).
    k_val : float
        Threshold parameter.

    Returns
    -------
    float
        M(t).
    """
    return math.exp(heaviside_cgf(t_val, k_val))


# ============================================================
# JAX versions
# ============================================================

def heaviside_cgf_jax(t_val, k_val):
    """
    JAX‑compatible CGF for the Heaviside prior.

    Parameters
    ----------
    t_val : float or JAX array
        Evaluation point (must be negative).
    k_val : float
        Threshold parameter.

    Returns
    -------
    JAX array
        log M(t).
    """
    return jnp.log(-1.0 / t_val) + k_val * t_val


def heaviside_mgf_jax(t_val, k_val):
    """
    JAX‑compatible MGF for the Heaviside prior.

    Parameters
    ----------
    t_val : float or JAX array
        Evaluation point (must be negative).
    k_val : float
        Threshold parameter.

    Returns
    -------
    JAX array
        M(t).
    """
    return jnp.exp(heaviside_cgf_jax(t_val, k_val))


# ============================================================
# SciPy PDF / logPDF (not available for improper Heaviside)
# ============================================================

def heaviside_pdf(theta_val: float, k_val: float) -> float:
    """
    Numeric PDF for the Heaviside prior.

    Parameters
    ----------
    theta_val : float
        Evaluation point.
    k_val : float
        Threshold parameter.

    Returns
    -------
    float
        1.0 if theta >= k, else 0.0.
    """
    # `np.where`, not a Python conditional. The improper Heaviside prior's
    # density is trivial, and that is exactly why it was written as
    # `1.0 if theta_val >= k_val else 0.0` -- which raises "truth value of an
    # array with more than one element is ambiguous" for any array of length
    # above one. It survived only because every caller evaluated one point at
    # a time; the moment the integrand is handed a vector of theta it fails,
    # and it also failed for anyone calling `prior.pdf_func` on an array.
    return np.where(np.asarray(theta_val) >= k_val, 1.0, 0.0)


def heaviside_logpdf(theta_val: float, k_val: float) -> float:
    """
    Numeric log‑PDF for the Heaviside prior.

    Parameters
    ----------
    theta_val : float
        Evaluation point.
    k_val : float
        Threshold parameter.

    Returns
    -------
    float
        0.0 if theta >= k, else -inf.
    """
    return np.where(np.asarray(theta_val) >= k_val, 0.0, -np.inf)


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

        # Improper prior: int_k^inf theta^a dtheta diverges for every a >= 0,
        # including a = 0. Its MGF exists only for t < 0, so no order is
        # admissible at t = 0.
        max_finite_moment=0.0,

        mgf=lambda t_val: heaviside_mgf(t_val, k_val),
        cgf=lambda t_val: heaviside_cgf(t_val, k_val),

        mgf_jax=lambda t_val: heaviside_mgf_jax(t_val, k_val),
        cgf_jax=lambda t_val: heaviside_cgf_jax(t_val, k_val),

        pdf_func=lambda x: heaviside_pdf(x, k_val),
        logpdf_func=lambda x: heaviside_logpdf(x, k_val),

        params=params,
    )
