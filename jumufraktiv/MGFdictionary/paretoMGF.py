"""
Pareto MGF and related functions for symbolic, numeric, JAX, and Torch.

This module provides the Pareto distribution in the MGF‑marginalisable framework.
The MGF is expressed in terms of the exponential integral E_α(z).

The incomplete MGF (upper‑truncated) is also provided, using the upper incomplete
gamma function Γ(a, z). Numerically, this is computed via `scipy.special.gammaincc`
and `scipy.special.gamma`, which can suffer from underflow/overflow for extreme
parameter values. The log‑scale versions attempt to mitigate this, but stability
is not guaranteed for very small or very large arguments.

**Numerical stability notes:**
- `pareto_cgf` and `pareto_cgf_jax` rely on `log(gammaincc(-alpha, z))`.
  Since SciPy and JAX do not provide a log‑scale version of `gammaincc`,
  this term can underflow (→ -inf) or overflow (→ inf) for extreme values
  of `z` or `alpha`. This is a known limitation of the current implementation.
- The incomplete MGF functions (`pareto_imgf`, `pareto_logimgf`, and their JAX
  counterparts) face similar issues because they combine `gammaincc` and `gamma`.

Symbolic, numeric (SciPy), JAX, and optional Torch backends are supported.
"""

import math

import jax.numpy as jnp
import numpy as np
import scipy.special as sc
import scipy.stats as stats
import sympy as sp
from jax.scipy.special import gammaincc as jax_gammaincc
from jax.scipy.special import gammaln as jax_gammaln
from sympy.functions.special.error_functions import expint

from jumufraktiv.registry import make_prior_spec, register_prior
from jumufraktiv.symbols import param, t, theta, u

# ============================================================
# Canonical symbolic parameters
# ============================================================
alpha = param("alpha")
xi = param("xi")


# ============================================================
# Numeric CGF / MGF (using scipy)
# ============================================================

def pareto_cgf(t_val: float, alpha_val: float, xi_val: float) -> float:
    """
    Numeric CGF for the Pareto distribution (log‑space stable).

    Notes
    -----
    The computation uses `scipy.special.gammaincc` and `scipy.special.gammaln`.
    Since SciPy does not provide a log‑scale incomplete gamma function,
    the term `log(gammaincc(...))` is computed directly. For very small or
    very large arguments, this can underflow or overflow, leading to `-inf`
    or `inf`. This is a known limitation of the current implementation.

    Parameters
    ----------
    t_val : float
        Evaluation point (must be <= 0).
    alpha_val : float
        Shape parameter.
    xi_val : float
        Scale parameter.

    Returns
    -------
    float
        log M(t).

    Raises
    ------
    ValueError
        If t > 0.
    """
    if t_val > 0:
        raise ValueError("MGF of Pareto distribution exists only for t <= 0")
    if t_val == 0.0:
        return 0.0

    z = -xi_val * t_val
    # log(Γ(-alpha, z)) via gammaincc + gammaln
    log_gamma_inc = np.log(sc.gammaincc(-alpha_val, z)) + sc.gammaln(-alpha_val)
    return math.log(alpha_val) + alpha_val * math.log(z) + log_gamma_inc


def pareto_mgf(t_val: float, alpha_val: float, xi_val: float) -> float:
    """
    Numeric MGF for the Pareto distribution.

    Parameters
    ----------
    t_val : float
        Evaluation point (must be <= 0).
    alpha_val : float
        Shape parameter.
    xi_val : float
        Scale parameter.

    Returns
    -------
    float
        M(t).
    """
    return math.exp(pareto_cgf(t_val, alpha_val, xi_val))


# ============================================================
# JAX versions
# ============================================================

def pareto_cgf_jax(t_val, alpha_val, xi_val):
    """
    JAX‑compatible CGF for the Pareto distribution.

    Notes
    -----
    The same numerical stability caveat applies as in `pareto_cgf`; JAX's
    `gammaincc` does not have a log‑scale variant, so `log(gammaincc(...))`
    may suffer from overflow/underflow for extreme parameters.

    Parameters
    ----------
    t_val : float or JAX array
        Evaluation point (must be <= 0).
    alpha_val : float
        Shape parameter.
    xi_val : float
        Scale parameter.

    Returns
    -------
    JAX array
        log M(t).
    """
    def safe_log(t):
        z = -xi_val * t
        a = -alpha_val
        log_gamma_a = jax_gammaln(a)
        log_q = jnp.log(jax_gammaincc(a, z))
        log_inc_gamma = log_gamma_a + log_q
        return jnp.log(alpha_val) + alpha_val * jnp.log(z) + log_inc_gamma
    return jnp.where(t_val == 0.0, 0.0, safe_log(t_val))


def pareto_mgf_jax(t_val, alpha_val, xi_val):
    """
    JAX‑compatible MGF for the Pareto distribution.

    Parameters
    ----------
    t_val : float or JAX array
        Evaluation point (must be <= 0).
    alpha_val : float
        Shape parameter.
    xi_val : float
        Scale parameter.

    Returns
    -------
    JAX array
        M(t).
    """
    return jnp.exp(pareto_cgf_jax(t_val, alpha_val, xi_val))


# ============================================================
# SciPy PDF / logPDF
# ============================================================

# ============================================================
# Incomplete MGF (upper‑truncated at u)
# ============================================================

def pareto_imgf_symbolic(u_sym):
    """
    Symbolic expression for the upper‑truncated Pareto MGF.

    Parameters
    ----------
    u_sym : sympy.Symbol
        Upper truncation point.

    Returns
    -------
    sympy.Expr
        ∫_xi^u e^{tθ} p(θ) dθ.
    """
    s = -t
    return alpha * (s * xi)**alpha * (
        sp.uppergamma(-alpha, s * xi) - sp.uppergamma(-alpha, s * u_sym)
    )


def pareto_imgf(t_val, alpha_val, xi_val, u_val):
    """
    Numeric upper‑truncated Pareto MGF (ordinary scale).

    Notes
    -----
    This computation uses `scipy.special.gammaincc` and `scipy.special.gamma`.
    For negative `alpha_val`, `scipy.special.gamma` may be very large or small,
    and the difference of two gammaincc values can suffer from catastrophic
    cancellation. The log‑scale version (`pareto_logimgf`) is recommended for
    small values.

    Parameters
    ----------
    t_val : float or array
        Evaluation point (must be <= 0).
    alpha_val : float
        Shape parameter.
    xi_val : float
        Scale parameter.
    u_val : float
        Upper truncation point.

    Returns
    -------
    float or array
        Incomplete MGF.
    """
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
    """
    Numeric log‑incomplete MGF for the Pareto distribution.

    Notes
    -----
    This function attempts to compute log of the incomplete MGF in a stable way,
    but still relies on `scipy.special.gammaincc` and `scipy.special.gammaln`.
    For very small values, `log(reg1 - reg2)` may be inaccurate. Use with caution
    in extreme tails.

    Parameters
    ----------
    t_val : float or array
        Evaluation point (must be <= 0).
    alpha_val : float
        Shape parameter.
    xi_val : float
        Scale parameter.
    u_val : float
        Upper truncation point.

    Returns
    -------
    float or array
        log of the incomplete MGF.
    """
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
    sign = np.sign(diff)
    log_val = (np.log(alpha_val) + alpha_val * np.log(s * xi_val) +
               sc.gammaln(a) + np.log(np.abs(diff)))
    return log_val


# ---- JAX ----
def pareto_imgf_jax(t_val, alpha_val, xi_val, u_val):
    """
    JAX‑compatible upper‑truncated Pareto MGF.

    Parameters
    ----------
    t_val : float or JAX array
        Evaluation point (must be <= 0).
    alpha_val : float
        Shape parameter.
    xi_val : float
        Scale parameter.
    u_val : float
        Upper truncation point.

    Returns
    -------
    JAX array
        Incomplete MGF.
    """
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
    """
    JAX‑compatible log‑incomplete MGF for the Pareto distribution.

    Parameters
    ----------
    t_val : float or JAX array
        Evaluation point (must be <= 0).
    alpha_val : float
        Shape parameter.
    xi_val : float
        Scale parameter.
    u_val : float
        Upper truncation point.

    Returns
    -------
    JAX array
        log of the incomplete MGF.
    """
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
    imgf_sym = pareto_imgf_symbolic(u)
    logimgf_sym = sp.log(imgf_sym)

    # Freeze the SciPy distribution ONCE. `stats.<dist>(params)` builds an
    # `rv_frozen`, and building one runs `_construct_doc`, which formats a
    # docstring -- 440 us of work, before any density is evaluated. Written
    # inside the lambda it ran on every call, and the density is the innermost
    # thing in the package: every quadrature node calls it. Hoisted, the same
    # call costs 41 us for identical values.
    frozen = stats.pareto(b=alpha_val, scale=xi_val)

    # Substitute numeric parameter values
    subs_map = {alpha: alpha_val, xi: xi_val}
    mgf_sym = mgf_sym.subs(subs_map)
    cgf_sym = cgf_sym.subs(subs_map)
    pdf_sym = pdf_sym.subs(subs_map)
    imgf_sym = imgf_sym.subs(subs_map)
    logimgf_sym = logimgf_sym.subs(subs_map)

    return make_prior_spec(
        mgf_sym=mgf_sym,
        cgf_sym=cgf_sym,
        pdf_sym=pdf_sym,

        mgf=lambda t_val: pareto_mgf(t_val, alpha_val, xi_val),
        cgf=lambda t_val: pareto_cgf(t_val, alpha_val, xi_val),

        mgf_jax=lambda t_val: pareto_mgf_jax(t_val, alpha_val, xi_val),
        cgf_jax=lambda t_val: pareto_cgf_jax(t_val, alpha_val, xi_val),

        pdf_func=frozen.pdf,
        logpdf_func=frozen.logpdf,
        
        # Incomplete MGF (truncated at u)
        imgf_sym=imgf_sym,
        logimgf_sym=logimgf_sym,
        imgf=lambda t_val, u_val: pareto_imgf(t_val, alpha_val, xi_val, u_val),
        logimgf=lambda t_val, u_val: pareto_logimgf(t_val, alpha_val, xi_val, u_val),
        # E[Theta^a] = alpha xi^a / (alpha - a) converges iff a < alpha, so the
        # bound is the tail index itself. Only consulted at t = 0.
        max_finite_moment=float(alpha_val),

        imgf_jax=lambda t_val, u_val: pareto_imgf_jax(t_val, alpha_val, xi_val, u_val),
        logimgf_jax=lambda t_val, u_val: pareto_logimgf_jax(t_val, alpha_val, xi_val, u_val),

        params=params,
    )
