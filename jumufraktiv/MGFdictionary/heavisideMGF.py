"""
heavisideMGF.py

Functions for the improper Heaviside prior: p(theta) ∝ 1 for theta >= k, else 0.

The MGF is defined for t < 0:
    M(t) = ∫_k^∞ e^{tθ} dθ = -e^{k t} / t

The CGF is log M(t) = log(-1/t) + k t, for t < 0.

This prior is improper (integral diverges), but the MGF exists for t < 0.

The incomplete MGF int_k^u e^{t x} dx is finite for every u, so the posterior
CDF -- and the quantile, interval and sampling methods built on it -- are
available even though the prior is not a distribution.
"""

import math

import jax.numpy as jnp
import numpy as np
import sympy as sp

from jumufraktiv.logsum import logminus
from jumufraktiv.registry import make_prior_spec, register_prior
from jumufraktiv.symbols import param, t, theta, u

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
    JAX-compatible CGF for the Heaviside prior.

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
    JAX-compatible MGF for the Heaviside prior.

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

def heaviside_pdf(theta_val, k_val: float):
    """
    Numeric PDF for the Heaviside prior.

    Parameters
    ----------
    theta_val : float or array-like
        Evaluation point(s).
    k_val : float
        Threshold parameter.

    Returns
    -------
    float or numpy.ndarray
        1.0 where theta >= k, else 0.0, with the shape of ``theta_val``.
    """
    # `np.where`, not the Python conditional this trivial density invites.
    # `1.0 if theta_val >= k_val else 0.0` raises "truth value of an array with
    # more than one element is ambiguous" for any array longer than one, and
    # both the quadrature integrand and `prior.pdf_func` are handed vectors of
    # theta.
    return np.where(np.asarray(theta_val) >= k_val, 1.0, 0.0)


def heaviside_logpdf(theta_val, k_val: float):
    """
    Numeric log-PDF for the Heaviside prior.

    Parameters
    ----------
    theta_val : float or array-like
        Evaluation point(s).
    k_val : float
        Threshold parameter.

    Returns
    -------
    float or numpy.ndarray
        0.0 where theta >= k, else -inf, with the shape of ``theta_val``.
    """
    return np.where(np.asarray(theta_val) >= k_val, 0.0, -np.inf)


# ============================================================
# Incomplete MGF (lower truncation at u)
# ============================================================
#
#     M(t, u) = int_k^u e^{t x} dx = (e^{t u} - e^{t k}) / t,   u >= k
#
# which `post_cdf` and the methods built on it need. The prior is improper, but
# the truncated integral is finite for every u, and the ratio M(t, u) / M(t)
# that becomes the posterior CDF is a proper distribution function.
#
# Defined for t < 0 only, like the MGF itself: at t = 0 the denominator
# vanishes and M(t) diverges, so there is no ratio to take.


def heaviside_imgf_symbolic(u_sym):
    """
    Symbolic expression for the lower-truncated Heaviside MGF.

    Parameters
    ----------
    u_sym : sympy.Symbol
        Upper truncation point.

    Returns
    -------
    sympy.Expr
        `(exp(t*Max(u, k)) - exp(t*k)) / t`.
    """
    return (sp.exp(t * sp.Max(u_sym, k)) - sp.exp(t * k)) / t


def heaviside_logimgf(t_val, k_val, u_val):
    """
    Numeric log incomplete MGF (vectorised, stable).

    Parameters
    ----------
    t_val : float or array
        Evaluation point; must be strictly negative.
    k_val : float
        Lower endpoint of the support.
    u_val : float or array
        Upper truncation point.

    Returns
    -------
    float or numpy.ndarray
        `log M(t, u)`, and `-inf` where `u <= k`.

    Raises
    ------
    ValueError
        If any `t_val >= 0`, where the Heaviside MGF does not exist.
    """
    t_arr, u_arr = np.broadcast_arrays(
        np.asarray(t_val, dtype=float), np.asarray(u_val, dtype=float)
    )
    if np.any(t_arr >= 0.0):
        raise ValueError(
            "the Heaviside prior's MGF exists only for t < 0; it is improper, "
            "so the integral diverges at and above the origin"
        )

    out = np.full(t_arr.shape, -np.inf, dtype=float)
    inside = u_arr > k_val
    if not np.any(inside):
        return out if out.ndim else float(out)

    for index in np.argwhere(inside):
        key = tuple(index)
        tv, uv = float(t_arr[key]), float(u_arr[key])
        # t < 0, so e^{t k} is the larger exponential and the numerator's sign
        # cancels with t's.
        out[key] = logminus(tv * k_val, tv * uv) - math.log(-tv)

    return out if out.ndim else float(out)


def heaviside_imgf(t_val, k_val, u_val):
    """Numeric incomplete MGF (ordinary scale, vectorised)."""
    return np.exp(heaviside_logimgf(t_val, k_val, u_val))


def heaviside_logimgf_jax(t_val, k_val, u_val):
    """JAX log incomplete MGF, dividing before the log as the CGF does."""
    upper = jnp.maximum(u_val, k_val)
    return jnp.log((jnp.exp(t_val * upper) - jnp.exp(t_val * k_val)) / t_val)


def heaviside_imgf_jax(t_val, k_val, u_val):
    """JAX incomplete MGF (ordinary scale)."""
    return jnp.exp(heaviside_logimgf_jax(t_val, k_val, u_val))


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
    imgf_sym = heaviside_imgf_symbolic(u)
    logimgf_sym = sp.log(imgf_sym)

    # Substitute numeric parameter values into the symbolic expressions
    subs_map = {k: k_val}
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

        # Improper prior: int_k^inf theta^a dtheta diverges for every a >= 0,
        # including a = 0. Its MGF exists only for t < 0, so no order is
        # admissible at t = 0.
        max_finite_moment=0.0,

        # Improper: int_k^inf e^{t x} dx converges only for t < 0, and the
        # endpoint is not attained.
        mgf_finite_below=0.0,

        mgf=lambda t_val: heaviside_mgf(t_val, k_val),
        cgf=lambda t_val: heaviside_cgf(t_val, k_val),

        mgf_jax=lambda t_val: heaviside_mgf_jax(t_val, k_val),
        cgf_jax=lambda t_val: heaviside_cgf_jax(t_val, k_val),

        imgf_sym=imgf_sym,
        logimgf_sym=logimgf_sym,
        imgf=lambda t_val, u_val: heaviside_imgf(t_val, k_val, u_val),
        logimgf=lambda t_val, u_val: heaviside_logimgf(t_val, k_val, u_val),
        imgf_jax=lambda t_val, u_val: heaviside_imgf_jax(t_val, k_val, u_val),
        logimgf_jax=lambda t_val, u_val: heaviside_logimgf_jax(t_val, k_val, u_val),

        pdf_func=lambda x: heaviside_pdf(x, k_val),
        logpdf_func=lambda x: heaviside_logpdf(x, k_val),

        params=params,
    )
