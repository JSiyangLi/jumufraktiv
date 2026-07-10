"""
Pareto MGF and related functions for symbolic, numeric, JAX, and Torch.
"""

import math
import sympy as sp
import numpy as np
import scipy.special as sc
import scipy.stats as stats
from sympy.functions.special.error_functions import expint
import jax.numpy as jnp
from jax.scipy.special import gammaln as jax_gammaln, gammaincc as jax_gammaincc
import torch
from torch.special import gammaincc as torch_gammaincc

from jumufraktiv.logsum import logplus, logminus
from jumufraktiv.registry import register_prior, make_prior_spec
from jumufraktiv.symbols import t, theta, param


# ============================================================
# Canonical symbolic parameters
# ============================================================
alpha = param("alpha")
xi = param("xi")


# ============================================================
# Symbolic expressions
# ============================================================

def pareto_mgf_symbolic():
    """
    M(t) = alpha * E_{alpha+1}(-xi*t),   for t <= 0
    """
    return alpha * expint(alpha + 1, -xi * t)


def pareto_cgf_symbolic():
    """
    K(t) = log(alpha) + log(E_{alpha+1}(-xi*t))
    """
    return sp.log(alpha) + sp.log(expint(alpha + 1, -xi * t))


def pareto_pdf_symbolic():
    """
    p(theta) = alpha * xi^alpha / theta^(alpha+1),   theta >= xi
    """
    return alpha * xi**alpha / theta**(alpha + 1)


# ============================================================
# Numeric CGF / MGF (using scipy)
# ============================================================

def pareto_cgf(t_val: float, alpha_val: float, xi_val: float) -> float:
    if t_val > 0:
        raise ValueError("MGF of Pareto distribution exists only for t <= 0")
    if t_val == 0.0:
        return 0.0

    z = -xi_val * t_val
    # log(Γ(-alpha, z)) via gammaincc + gammaln
    log_gamma_inc = np.log(sc.gammaincc(-alpha_val, z)) + sc.gammaln(-alpha_val)
    return math.log(alpha_val) + alpha_val * math.log(z) + log_gamma_inc


def pareto_mgf(t_val: float, alpha_val: float, xi_val: float) -> float:
    return math.exp(pareto_cgf(t_val, alpha_val, xi_val))


# ============================================================
# JAX versions
# ============================================================

def pareto_cgf_jax(t_val, alpha_val, xi_val):
    def safe_log(t):
        z = -xi_val * t
        a = -alpha_val
        log_gamma_a = jax_gammaln(a)
        log_q = jnp.log(jax_gammaincc(a, z))
        log_inc_gamma = log_gamma_a + log_q
        return jnp.log(alpha_val) + alpha_val * jnp.log(z) + log_inc_gamma
    return jnp.where(t_val == 0.0, 0.0, safe_log(t_val))


def pareto_mgf_jax(t_val, alpha_val, xi_val):
    return jnp.exp(pareto_cgf_jax(t_val, alpha_val, xi_val))


# ============================================================
# Torch version (optional)
# ============================================================

def pareto_mgf_torch(t, alpha, xi):
    alpha_t = torch.tensor(alpha, dtype=t.dtype, device=t.device)
    xi_t = torch.tensor(xi, dtype=t.dtype, device=t.device)
    z = -xi_t * t
    a = -alpha_t

    log_gamma_a = torch.lgamma(a)
    log_q = torch.log(torch_gammaincc(a, z))
    log_inc = log_gamma_a + log_q
    log_mgf = torch.log(alpha_t) + alpha_t * torch.log(z) + log_inc

    return torch.where(t == 0.0, 0.0, log_mgf).exp()


# ============================================================
# SciPy PDF / logPDF
# ============================================================

def pareto_pdf(theta_val: float, alpha_val: float, xi_val: float) -> float:
    return stats.pareto(b=alpha_val, scale=xi_val).pdf(theta_val)


def pareto_logpdf(theta_val: float, alpha_val: float, xi_val: float) -> float:
    return stats.pareto(b=alpha_val, scale=xi_val).logpdf(theta_val)


# ============================================================
# Incomplete MGF (upper‑truncated at u)
# ============================================================

def pareto_imgf_symbolic(u_sym):
    s = -t
    return alpha * (s * xi)**alpha * (
        sp.uppergamma(-alpha, s * xi) - sp.uppergamma(-alpha, s * u_sym)
    )

def pareto_logimgf_symbolic(u_sym):
    return sp.log(pareto_imgf_symbolic(u_sym))


def pareto_imgf(t_val, alpha_val, xi_val, u_val):
    if np.any(t_val > 0):
        raise ValueError("t must be ≤ 0")
    if np.isscalar(t_val) and t_val == 0:
        return 1.0 - (xi_val / u_val)**alpha_val
    s = -t_val
    a = -alpha_val
    z1 = s * xi_val
    z2 = s * u_val
    gamma_a = sc.gamma(a)  # signed Γ(a)
    diff = (sc.gammaincc(a, z1) - sc.gammaincc(a, z2)) * gamma_a
    return alpha_val * (s * xi_val)**alpha_val * diff


def pareto_logimgf(t_val, alpha_val, xi_val, u_val):
    if np.any(t_val > 0):
        raise ValueError("t must be ≤ 0")
    if np.isscalar(t_val) and t_val == 0:
        return np.log1p(-(xi_val / u_val)**alpha_val)
    s = -t_val
    a = -alpha_val
    z1 = s * xi_val
    z2 = s * u_val
    reg1 = sc.gammaincc(a, z1)
    reg2 = sc.gammaincc(a, z2)
    diff = reg1 - reg2
    # diff should be positive; use absolute for safety
    sign = np.sign(diff)
    log_val = (np.log(alpha_val) + alpha_val * np.log(s * xi_val) +
               sc.gammaln(a) + np.log(np.abs(diff)))
    return log_val


# ---- JAX ----
def pareto_imgf_jax(t_val, alpha_val, xi_val, u_val):
    def compute(t):
        s = -t
        a = -alpha_val
        z1 = s * xi_val
        z2 = s * u_val
        gamma_a = jnp.exp(jax_gammaln(a)) * jnp.sign(jnp.gamma(a))
        reg1 = jax_gammaincc(a, z1)
        reg2 = jax_gammaincc(a, z2)
        diff = (reg1 - reg2) * gamma_a
        return alpha_val * (s * xi_val)**alpha_val * diff
    return jnp.where(t_val == 0.0,
                     1.0 - (xi_val / u_val)**alpha_val,
                     compute(t_val))

def pareto_logimgf_jax(t_val, alpha_val, xi_val, u_val):
    def compute_log(t):
        s = -t
        a = -alpha_val
        z1 = s * xi_val
        z2 = s * u_val
        reg1 = jax_gammaincc(a, z1)
        reg2 = jax_gammaincc(a, z2)
        diff = reg1 - reg2
        log_abs_diff = jnp.log(jnp.abs(diff), where=diff != 0)
        log_abs_diff = jnp.where(diff == 0, -jnp.inf, log_abs_diff)
        return (jnp.log(alpha_val) + alpha_val * jnp.log(s * xi_val) +
                jax_gammaln(a) + log_abs_diff)
    return jnp.where(t_val == 0.0,
                     jnp.log1p(-(xi_val / u_val)**alpha_val),
                     compute_log(t_val))


# ============================================================
# Registry factory
# ============================================================

@register_prior("pareto")
def pareto_factory(params):
    alpha_val = params["alpha"]
    xi_val = params["xi"]

    # Build symbolic expressions using global symbols
    mgf_sym = alpha * expint(alpha + 1, -xi * t)
    cgf_sym = sp.log(alpha) + sp.log(expint(alpha + 1, -xi * t))
    pdf_sym = alpha * xi**alpha / theta**(alpha + 1)

    # Substitute numeric parameter values
    subs_map = {alpha: alpha_val, xi: xi_val}
    mgf_sym = mgf_sym.subs(subs_map)
    cgf_sym = cgf_sym.subs(subs_map)
    pdf_sym = pdf_sym.subs(subs_map)

    return make_prior_spec(
        mgf_sym=mgf_sym,
        cgf_sym=cgf_sym,
        pdf_sym=pdf_sym,

        mgf=lambda t_val: pareto_mgf(t_val, alpha_val, xi_val),
        cgf=lambda t_val: pareto_cgf(t_val, alpha_val, xi_val),

        mgf_jax=lambda t_val: pareto_mgf_jax(t_val, alpha_val, xi_val),
        cgf_jax=lambda t_val: pareto_cgf_jax(t_val, alpha_val, xi_val),

        pdf_func=lambda x: pareto_pdf(x, alpha_val, xi_val),
        logpdf_func=lambda x: pareto_logpdf(x, alpha_val, xi_val),
        
        # Incomplete MGF (truncated at u)
        imgf=lambda t_val, u_val: pareto_imgf(t_val, alpha_val, xi_val, u_val),
        logimgf=lambda t_val, u_val: pareto_logimgf(t_val, alpha_val, xi_val, u_val),
        imgf_jax=lambda t_val, u_val: pareto_imgf_jax(t_val, alpha_val, xi_val, u_val),
        logimgf_jax=lambda t_val, u_val: pareto_logimgf_jax(t_val, alpha_val, xi_val, u_val),

        params=params,
    )