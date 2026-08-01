"""
gammaMGF.py

Functions for the Gamma prior in the MGF-marginalisable framework.

The Gamma distribution (in terms of rate β) has density:
    p(θ; α, β) = (β^α / Γ(α)) * θ^{α-1} * exp(-β θ)

The MGF is:
    M(t) = (β / (β - t))^α,  for t < β.

The CGF is:
    K(t) = α (log β - log(β - t)),  for t < β.

This module provides:
- **Incomplete MGF** (iMGF) for the lower-truncated Gamma distribution:
    M_inc(t; α, β, u) = (β/(β−t))^α * γ(α, (β−t)u) / Γ(α),
  with t < β and u > 0, in symbolic, numeric (SciPy) and JAX forms.
- ``gamma_factory``, which the registry calls to build the prior. It writes
  the complete MGF, CGF and PDF out inline, symbolically and as callables, and
  is the single definition of each.

The Gamma prior is numerically stable for all parameter values; no special
caveats apply.
"""

import jax.numpy as jnp
import numpy as np
import sympy as sp
from jax.scipy.special import gammainc as jax_gammainc
from scipy.special import gammainc
from scipy.stats import gamma as scipy_gamma

from jumufraktiv.registry import make_prior_spec, register_prior
from jumufraktiv.symbols import param, t, theta, u

# ============================================================
# Canonical symbolic parameters (shared system)
# ============================================================
alpha = param("alpha")
beta = param("beta")


# ============================================================
# Incomplete MGF (lower truncation at u) for Gamma distribution
# ============================================================

# ---- Symbolic ----

def gamma_imgf_symbolic(u_sym):
    """
    Symbolic expression for the lower-truncated Gamma MGF.

    Parameters
    ----------
    u_sym : sympy.Symbol
        Upper truncation point.

    Returns
    -------
    sympy.Expr
        (β/(β−t))^α * γ(α, (β−t)u) / Γ(α).
    """
    return (beta / (beta - t)) ** alpha * (
        sp.lowergamma(alpha, (beta - t) * u_sym) / sp.gamma(alpha)
    )


# ---- Numeric (SciPy) ----

def gamma_imgf(t_val, alpha_val, beta_val, u_val):
    """
    Numeric incomplete MGF (ordinary scale, vectorised).

    Parameters
    ----------
    t_val : float or array
        Evaluation point (must be < beta_val).
    alpha_val : float
        Shape parameter.
    beta_val : float
        Rate parameter.
    u_val : float or array
        Upper truncation point.

    Returns
    -------
    float or array
        Incomplete MGF.

    Raises
    ------
    ValueError
        If any t_val >= beta_val.
    """
    s = beta_val - t_val
    if np.any(s <= 0):
        raise ValueError("t must be strictly less than beta for all elements")
    reg_gamma = gammainc(alpha_val, s * u_val)   # γ(α, x)/Γ(α)
    return (beta_val / s) ** alpha_val * reg_gamma


def gamma_logimgf(t_val, alpha_val, beta_val, u_val):
    """
    Numeric log-incomplete MGF (vectorised, stable).

    Parameters
    ----------
    t_val : float or array
        Evaluation point (must be < beta_val).
    alpha_val : float
        Shape parameter.
    beta_val : float
        Rate parameter.
    u_val : float or array
        Upper truncation point.

    Returns
    -------
    float or array
        log of the incomplete MGF.

    Raises
    ------
    ValueError
        If any t_val >= beta_val.
    """
    s = beta_val - t_val
    if np.any(s <= 0):
        raise ValueError("t must be strictly less than beta for all elements")
    log_factor = alpha_val * (np.log(beta_val) - np.log(s))
    reg_gamma = gammainc(alpha_val, s * u_val)
    # log(reg_gamma) is -inf where reg_gamma == 0; that is correct.
    return log_factor + np.log(reg_gamma)


# ---- JAX ----

def gamma_imgf_jax(t_val, alpha_val, beta_val, u_val):
    """
    JAX-compatible incomplete MGF (JIT-compatible, vectorised).

    Parameters
    ----------
    t_val : float or JAX array
        Evaluation point (must be < beta_val).
    alpha_val : float
        Shape parameter.
    beta_val : float
        Rate parameter.
    u_val : float or JAX array
        Upper truncation point.

    Returns
    -------
    JAX array
        Incomplete MGF.
    """
    s = beta_val - t_val
    reg_gamma = jax_gammainc(alpha_val, s * u_val)
    return (beta_val / s) ** alpha_val * reg_gamma


def gamma_logimgf_jax(t_val, alpha_val, beta_val, u_val):
    """
    JAX-compatible log-incomplete MGF (JIT-compatible, vectorised).

    Parameters
    ----------
    t_val : float or JAX array
        Evaluation point (must be < beta_val).
    alpha_val : float
        Shape parameter.
    beta_val : float
        Rate parameter.
    u_val : float or JAX array
        Upper truncation point.

    Returns
    -------
    JAX array
        log of the incomplete MGF.
    """
    s = beta_val - t_val
    log_factor = alpha_val * (jnp.log(beta_val) - jnp.log(s))
    reg_gamma = jax_gammainc(alpha_val, s * u_val)
    return log_factor + jnp.log(reg_gamma)


# ============================================================
# Registry factory
# ============================================================

@register_prior("gamma")
def gamma_factory(params):
    alpha_val = float(params["alpha"])
    beta_val = float(params["beta"])

    # Build symbolic expressions using the global symbols
    mgf_sym = (beta / (beta - t)) ** alpha
    cgf_sym = alpha * (sp.log(beta) - sp.log(beta - t))
    pdf_sym = (
        (beta**alpha / sp.gamma(alpha)) * theta**(alpha - 1) * sp.exp(-beta * theta)
    )
    imgf_sym = gamma_imgf_symbolic(u)
    logimgf_sym = sp.log(imgf_sym)

    # Freeze the SciPy distribution ONCE, here rather than inside the lambdas
    # below. `stats.<dist>(params)` builds an `rv_frozen`, and building one runs
    # `_construct_doc`, which formats a docstring -- about 430 us before any
    # density is evaluated. The density is the innermost thing in the package,
    # called at every quadrature node, so it must not be rebuilt per call.
    frozen = scipy_gamma(a=alpha_val, scale=1.0 / beta_val)

    # Substitute numeric parameter values into the symbolic expressions
    subs_map = {alpha: alpha_val, beta: beta_val}
    mgf_sym = mgf_sym.subs(subs_map)
    cgf_sym = cgf_sym.subs(subs_map)
    pdf_sym = pdf_sym.subs(subs_map)
    imgf_sym = imgf_sym.subs(subs_map)
    logimgf_sym = logimgf_sym.subs(subs_map)

    # Return the spec using make_prior_spec
    return make_prior_spec(
        mgf_sym=mgf_sym,
        cgf_sym=cgf_sym,
        pdf_sym=pdf_sym,

        # E[Theta^a] = Gamma(a+alpha)/(beta^a Gamma(alpha)) is finite for every
        # a >= 0, so no order is inadmissible at t = 0.
        max_finite_moment=float("inf"),

        mgf=lambda t_val: (beta_val / (beta_val - t_val)) ** alpha_val,
        cgf=lambda t_val: alpha_val * (np.log(beta_val) - np.log(beta_val - t_val)),

        mgf_jax=lambda t_val: (beta_val / (beta_val - t_val)) ** alpha_val,
        cgf_jax=lambda t_val: (
            alpha_val * (jnp.log(beta_val) - jnp.log(beta_val - t_val))
        ),

        pdf_func=frozen.pdf,
        logpdf_func=frozen.logpdf,

        # ---- Incomplete MGF (truncated at u) ----
        # M(t) = (beta/(beta-t))**alpha is finite exactly for t < beta; at
        # t = beta the denominator vanishes and the integral diverges.
        mgf_finite_below=beta_val,

        imgf_sym=imgf_sym,
        logimgf_sym=logimgf_sym,
        imgf=lambda t_val, u_val: gamma_imgf(t_val, alpha_val, beta_val, u_val),
        logimgf=lambda t_val, u_val: gamma_logimgf(t_val, alpha_val, beta_val, u_val),
        imgf_jax=lambda t_val, u_val: gamma_imgf_jax(t_val, alpha_val, beta_val, u_val),
        logimgf_jax=lambda t_val, u_val: gamma_logimgf_jax(
            t_val, alpha_val, beta_val, u_val
        ),

        params=params,
    )
