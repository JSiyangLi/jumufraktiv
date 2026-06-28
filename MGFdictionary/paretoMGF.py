"""
Pareto MGF and related functions for symbolic, numeric, JAX, and Torch.
"""

import math
import sympy as sp
import numpy as np
import scipy.special as sc
from sympy.functions.special.error_functions import expint
import jax.numpy as jnp
from jax.scipy.special import gammaln as jax_gammaln, gammaincc as jax_gammaincc
import torch
from torch.special import gammaincc as torch_gammaincc


def pareto_mgf_symbolic():
    """
    Returns a SymPy expression for the Pareto MGF using the exponential integral.
        M(t) = alpha * E_{alpha+1}(-xi*t),   for t <= 0
    """
    t = sp.Symbol('t', real=True, nonpositive=True)
    alpha = sp.Symbol('alpha', positive=True, real=True)
    xi = sp.Symbol('xi', positive=True, real=True)
    z = -xi * t
    return alpha * expint(alpha + 1, z)


def pareto_cgf_symbolic():
    """
    Returns symbolic expression for the CGF of the Pareto distribution:
        K(t) = log(alpha) + log(E_{alpha+1}(-xi*t))   for t <= 0
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

    z = -xi * t
    # log(Γ(-alpha, z)) via log_gammaincc
    log_gamma_inc = np.log(sc.gammaincc(-alpha, z)) + sc.gammaln(-alpha) # when sp.log_gammaincc becomes available, replace by log_gamma_inc = sc.log_gammaincc(-alpha, z) + sc.gammaln(-alpha)
    return math.log(alpha) + alpha * math.log(z) + log_gamma_inc


def pareto_mgf(t: float, alpha: float, xi: float) -> float:
    """Returns the Pareto MGF in ordinary scale."""
    return math.exp(pareto_cgf(t, alpha, xi))


def pareto_cgf_jax(t, alpha, xi):
    """JAX version of log MGF for Pareto(shape=alpha, scale=xi)."""
    def safe_log(t_val):
        z = -xi * t_val
        a = -alpha
        log_gamma_a = jax_gammaln(a)
        log_q = jnp.log(jax_gammaincc(a, z))
        log_inc_gamma = log_gamma_a + log_q
        return jnp.log(alpha) + alpha * jnp.log(z) + log_inc_gamma
    return jnp.where(t == 0.0, 0.0, safe_log(t))


def pareto_mgf_jax(t, alpha, xi):
    """JAX version of MGF (ordinary scale)."""
    return jnp.exp(pareto_cgf_jax(t, alpha, xi))


def pareto_mgf_torch(t, alpha, xi):
    """Torch version of M(t) for Pareto(shape=alpha, scale=xi)."""
    # Convert alpha, xi to tensors
    alpha_t = torch.tensor(alpha, dtype=t.dtype, device=t.device)
    xi_t = torch.tensor(xi, dtype=t.dtype, device=t.device)

    z = -xi_t * t
    a = -alpha_t

    log_gamma_a = torch.lgamma(a)
    log_q = torch.log(torch_gammaincc(a, z))
    log_inc = log_gamma_a + log_q
    log_mgf = torch.log(alpha_t) + alpha_t * torch.log(z) + log_inc

    return torch.where(t == 0.0, 0.0, log_mgf).exp()


def pareto_pdf_symbolic():
    """
    Return a SymPy expression for the Pareto density:
        p(theta) = alpha * xi^alpha / theta^(alpha+1),  theta >= xi
    """
    theta, alpha, xi = sp.symbols('theta alpha xi', positive=True, real=True)
    return alpha * xi**alpha * theta**(-alpha - 1)


def pareto_pdf_symbolic_sub(params):
    """
    Return the symbolic Pareto PDF with numeric parameters substituted.
    params must contain 'alpha' and 'xi'.
    """
    alpha = params['alpha']
    xi = params['xi']
    return pareto_pdf_symbolic().subs({sp.Symbol('alpha'): alpha, sp.Symbol('xi'): xi})