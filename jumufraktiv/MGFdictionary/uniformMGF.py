"""
uniformMGF.py

Functions for the uniform prior: p(theta) = 1/(b-a) for theta in [a, b], else 0.

The MGF is:
    M(t) = (exp(t*b) - exp(t*a)) / (t*(b-a))   for t != 0
    M(0) = 1

For t < 0, the MGF is finite and positive. The CGF is defined as log M(t).

Numerical stability notes:
- The expressions involve differences of exponentials (exp(t*b) - exp(t*a)),
  which can suffer from cancellation when t is small or when a and b are close.
- The log-space CGF uses `log(exp(t*b) - exp(t*a))`, which is stable for
  moderate values but may overflow for very large |t|.
- For t=0, the CGF is defined as 0 (and the MGF as 1) by continuity.

Symbolic, numeric (SciPy), and JAX backends are supported.
No incomplete MGF (iMGF) is provided for the uniform prior.
"""

import math

import jax.numpy as jnp
import scipy.stats as stats
import sympy as sp

from jumufraktiv.registry import make_prior_spec, register_prior
from jumufraktiv.symbols import param, t, theta

# ============================================================
# Canonical symbolic parameters
# ============================================================
a = param("a")
b = param("b")


# ============================================================
# Numeric CGF / MGF
# ============================================================

def uniform_cgf(t_val: float, a_val: float, b_val: float) -> float:
    """
    Numeric CGF for the uniform prior.

    Parameters
    ----------
    t_val : float
        Evaluation point.
    a_val : float
        Lower bound.
    b_val : float
        Upper bound.

    Returns
    -------
    float
        log M(t), with M(0)=1.

    Notes
    -----
    For t=0, returns 0.0 by continuity.
    """
    if t_val == 0.0:
        return 0.0
    return math.log(math.exp(t_val * b_val) - math.exp(t_val * a_val)) - math.log(t_val * (b_val - a_val))


def uniform_mgf(t_val: float, a_val: float, b_val: float) -> float:
    """
    Numeric MGF for the uniform prior.

    Parameters
    ----------
    t_val : float
        Evaluation point.
    a_val : float
        Lower bound.
    b_val : float
        Upper bound.

    Returns
    -------
    float
        M(t), with M(0)=1.
    """
    return math.exp(uniform_cgf(t_val, a_val, b_val))


# ============================================================
# JAX versions
# ============================================================

def uniform_cgf_jax(t_val, a_val, b_val):
    """
    JAX‑compatible CGF for the uniform prior.

    Parameters
    ----------
    t_val : float or JAX array
        Evaluation point.
    a_val : float
        Lower bound.
    b_val : float
        Upper bound.

    Returns
    -------
    JAX array
        log M(t).
    """
    return jnp.log((jnp.exp(t_val * b_val) - jnp.exp(t_val * a_val)) / (t_val * (b_val - a_val)))


def uniform_mgf_jax(t_val, a_val, b_val):
    """
    JAX‑compatible MGF for the uniform prior.

    Parameters
    ----------
    t_val : float or JAX array
        Evaluation point.
    a_val : float
        Lower bound.
    b_val : float
        Upper bound.

    Returns
    -------
    JAX array
        M(t).
    """
    return jnp.exp(uniform_cgf_jax(t_val, a_val, b_val))


# ============================================================
# SciPy PDF / logPDF (using scipy.stats.uniform)
# ============================================================

def uniform_pdf(theta_val: float, a_val: float, b_val: float) -> float:
    """
    Numeric PDF for the uniform prior (via SciPy).

    Parameters
    ----------
    theta_val : float
        Evaluation point.
    a_val : float
        Lower bound.
    b_val : float
        Upper bound.

    Returns
    -------
    float
        p(theta).
    """
    return stats.uniform(loc=a_val, scale=b_val - a_val).pdf(theta_val)


def uniform_logpdf(theta_val: float, a_val: float, b_val: float) -> float:
    """
    Numeric log‑PDF for the uniform prior (via SciPy).

    Parameters
    ----------
    theta_val : float
        Evaluation point.
    a_val : float
        Lower bound.
    b_val : float
        Upper bound.

    Returns
    -------
    float
        log p(theta).
    """
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

        # Bounded support, so every moment is finite and no order is
        # inadmissible at t = 0.
        max_finite_moment=float("inf"),

        mgf=lambda t_val: uniform_mgf(t_val, a_val, b_val),
        cgf=lambda t_val: uniform_cgf(t_val, a_val, b_val),

        mgf_jax=lambda t_val: uniform_mgf_jax(t_val, a_val, b_val),
        cgf_jax=lambda t_val: uniform_cgf_jax(t_val, a_val, b_val),

        pdf_func=lambda x: uniform_pdf(x, a_val, b_val),
        logpdf_func=lambda x: uniform_logpdf(x, a_val, b_val),

        params=params,
    )
