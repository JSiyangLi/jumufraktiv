"""
Pareto MGF and related functions for symbolic, numeric (SciPy) and JAX use.

This module provides the Pareto distribution in the MGF-marginalisable framework.
The MGF is expressed in terms of the exponential integral E_α(z).

The incomplete MGF (upper-truncated at u) is also provided.

Both the complete and the incomplete MGF need Γ(−α, z), an upper incomplete
gamma of *negative* order — the incomplete one as a difference of two. SciPy's
`gammaincc` is the regularised form and is defined only for a positive first
argument, so it cannot supply it; the value comes from the generalised
exponential integral in arbitrary precision, via
`_log_upper_gamma_negative_order`.

**Numerical notes:**

- The logarithm is taken inside the arbitrary-precision evaluation, so
  `pareto_cgf` and `pareto_logimgf` stay finite where Γ(−α, z) itself
  underflows float64.

- `pareto_logimgf` is a difference of two such terms and forms it with
  `logminus`, the log of a difference, rather than subtracting two logs — the
  two coincide as `u` approaches `xi`.

- JAX has neither an incomplete gamma at negative order nor an exponential
  integral at real order, so every JAX entry point here raises
  `NotImplementedError`. Use a SciPy backend for a Pareto prior.

Symbolic and numeric (SciPy) backends are supported.
"""

import math

import jax.numpy as jnp
import mpmath as mp
import numpy as np
import scipy.stats as stats
import sympy as sp
from sympy.functions.special.error_functions import expint

from jumufraktiv.logsum import logminus
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
    Numeric CGF for the Pareto distribution (log-space stable).

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

    Notes
    -----
    ``log M(t) = log alpha + alpha log z + log Gamma(-alpha, z)`` with
    ``z = -xi t``. The last term is an upper incomplete gamma of negative
    order, which :func:`_log_upper_gamma_negative_order` supplies; see its
    Notes for why ``scipy.special.gammaincc`` cannot.

    Taking the logarithm inside that evaluation keeps the result finite where
    ``Gamma(-alpha, z)`` itself underflows float64.
    """
    if t_val > 0:
        raise ValueError("MGF of Pareto distribution exists only for t <= 0")
    if t_val == 0.0:
        return 0.0

    z = -xi_val * t_val
    return (
        math.log(alpha_val)
        + alpha_val * math.log(z)
        + _log_upper_gamma_negative_order(-alpha_val, z)
    )


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
    JAX-compatible CGF for the Pareto distribution.

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

    Raises
    ------
    NotImplementedError
        Always. The Pareto CGF needs ``Gamma(-alpha, z)``, an upper incomplete
        gamma of negative order, and ``jax.scipy.special`` offers ``gammaincc``
        for a positive first argument only and ``expn`` at integer order only.
        :func:`pareto_cgf` computes it on the SciPy side.
    """
    raise NotImplementedError(
        "The Pareto CGF has no JAX implementation. It requires the upper "
        "incomplete gamma at negative order, equivalently the generalised "
        "exponential integral E_{1+alpha} at real order, and "
        "jax.scipy.special provides neither. Use a non-JAX backend for a "
        "Pareto prior."
    )


def pareto_mgf_jax(t_val, alpha_val, xi_val):
    """
    JAX-compatible MGF for the Pareto distribution.

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
# Incomplete MGF (upper-truncated at u)
# ============================================================


def pareto_imgf_symbolic(u_sym):
    """
    Symbolic expression for the upper-truncated Pareto MGF.

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
    return (
        alpha
        * (s * xi) ** alpha
        * (sp.uppergamma(-alpha, s * xi) - sp.uppergamma(-alpha, s * u_sym))
    )


def _log_upper_gamma_negative_order(a, z):
    """``log Gamma(a, z)`` for ``a < 0``, elementwise over ``z``.

    Parameters
    ----------
    a : float
        First argument of the upper incomplete gamma. Must be negative; for a
        positive one use :func:`scipy.special.gammaincc` directly.
    z : float or array-like
        Second argument. Must be positive.

    Returns
    -------
    float or numpy.ndarray
        The natural logarithm. ``Gamma(a, z)`` is positive for ``a < 0`` and
        ``z > 0``, so no sign accompanies it.

    Notes
    -----
    ``scipy.special.gammaincc`` is the *regularised* upper incomplete gamma and
    is defined only for a positive first argument; at ``a < 0`` it returns
    ``nan``. The generalised exponential integral supplies the value directly,

    .. math:: \\Gamma(a, z) = z^{a} E_{1-a}(z),

    which holds for every real ``a``.

    ``scipy.special.expn`` cannot evaluate it. That function takes an integer
    order and **truncates a real one**, so ``expn(2.5, x)`` silently returns
    ``E_2(x)`` -- a different function -- behind nothing louder than a
    ``RuntimeWarning``. ``mpmath.expint`` accepts a real order and is used
    instead, elementwise, at roughly 120 us per point.

    The logarithm is taken inside the arbitrary-precision evaluation, so the
    result stays finite where ``Gamma(a, z)`` itself underflows float64 --
    which it does readily: ``Gamma(-2, 40)`` is about ``6.2e-23``.
    """
    values = np.asarray(z, dtype=float)
    order = 1.0 - a

    flat = np.empty(values.size, dtype=float)
    for index, value in enumerate(values.ravel()):
        with mp.workdps(30):
            flat[index] = float(
                mp.log(mp.expint(order, mp.mpf(float(value))))
                + a * mp.log(mp.mpf(float(value)))
            )

    result = flat.reshape(values.shape)
    return float(result) if np.ndim(z) == 0 else result


def pareto_imgf(t_val, alpha_val, xi_val, u_val):
    """
    Numeric upper-truncated Pareto MGF (ordinary scale).

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

    Notes
    -----
    Underflows to zero once the truncated mass falls below float64's range;
    :func:`pareto_logimgf` stays finite there and is what the library uses.
    """
    return np.exp(pareto_logimgf(t_val, alpha_val, xi_val, u_val))


def pareto_logimgf(t_val, alpha_val, xi_val, u_val):
    """
    Numeric log-incomplete MGF for the Pareto distribution.

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

    Notes
    -----
    The quantity is

    .. math::

        \\int_\\xi^u e^{t\\theta} p(\\theta)\\,d\\theta
            = \\alpha (s\\xi)^{\\alpha}
              \\left[\\Gamma(-\\alpha, s\\xi) - \\Gamma(-\\alpha, su)\\right],
        \\qquad s = -t,

    a difference of two positive terms, the first the larger because the upper
    incomplete gamma decreases in its second argument and ``xi <= u``. It is
    formed with :func:`jumufraktiv.logsum.logminus`, which computes the log of
    a difference rather than a difference of logs, because the two terms
    approach each other as ``u`` approaches ``xi``.
    """
    if np.any(np.asarray(t_val, dtype=float) > 0):
        raise ValueError("t must be ≤ 0")

    scalar = np.ndim(t_val) == 0 and np.ndim(u_val) == 0
    s, upper = np.broadcast_arrays(
        np.atleast_1d(np.asarray(-t_val, dtype=float)),
        np.atleast_1d(np.asarray(u_val, dtype=float)),
    )

    result = np.empty(s.shape, dtype=float)

    # t = 0 is the untruncated limit, where the exponential is 1 and the
    # integral is just the Pareto CDF at u.
    origin = s == 0.0
    if np.any(origin):
        result[origin] = np.log1p(-((xi_val / upper[origin]) ** alpha_val))

    live = ~origin
    if np.any(live):
        a = -alpha_val
        log_lower = _log_upper_gamma_negative_order(a, s[live] * xi_val)
        log_upper = _log_upper_gamma_negative_order(a, s[live] * upper[live])
        result[live] = (
            np.log(alpha_val)
            + alpha_val * np.log(s[live] * xi_val)
            + logminus(log_lower, log_upper)
        )

    return float(result[0]) if scalar else result


# ---- JAX ----
def pareto_imgf_jax(t_val, alpha_val, xi_val, u_val):
    """
    JAX-compatible upper-truncated Pareto MGF.

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

    Raises
    ------
    NotImplementedError
        Always. See Notes.

    Notes
    -----
    JAX cannot express this quantity. It needs ``Gamma(-alpha, z)``, an upper
    incomplete gamma of negative order, which reduces to the generalised
    exponential integral ``E_{1+alpha}``. ``jax.scipy.special`` offers ``expn``
    at integer order only and ``gammaincc`` for a positive first argument only,
    so neither reaches it for a general ``alpha``.

    :func:`pareto_logimgf` computes it correctly on the SciPy side, so a caller
    who reaches this should use a non-JAX integer backend.
    """
    raise NotImplementedError(
        "The Pareto incomplete MGF has no JAX implementation. It requires the "
        "generalised exponential integral E_{1+alpha} at real order, and "
        "jax.scipy.special provides expn at integer order only. Use "
        "method='symbolic' or method='auto' for an incomplete-MGF derivative "
        "against a Pareto prior; 'bell' reaches this function at order 0 and "
        "whenever cgf_method forces its JAX path."
    )


def pareto_logimgf_jax(t_val, alpha_val, xi_val, u_val):
    """
    JAX-compatible log-incomplete MGF for the Pareto distribution.

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

    Raises
    ------
    NotImplementedError
        Always, for the reason given in :func:`pareto_imgf_jax`, which applies
        equally in log space: ``jax_gammaincc(-alpha, z)`` is outside
        ``gammaincc``'s domain.
    """
    raise NotImplementedError(
        "The Pareto incomplete MGF has no JAX implementation. It requires the "
        "generalised exponential integral E_{1+alpha} at real order, and "
        "jax.scipy.special provides expn at integer order only. Use "
        "method='symbolic' or method='auto' for an incomplete-MGF derivative "
        "against a Pareto prior; 'bell' reaches this function at order 0 and "
        "whenever cgf_method forces its JAX path."
    )


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
    pdf_sym = alpha * xi**alpha / theta ** (alpha + 1)
    imgf_sym = pareto_imgf_symbolic(u)
    logimgf_sym = sp.log(imgf_sym)

    # Freeze the SciPy distribution ONCE, here rather than inside the lambdas
    # below. `stats.<dist>(params)` builds an `rv_frozen`, and building one runs
    # `_construct_doc`, which formats a docstring -- about 440 us before any
    # density is evaluated. The density is the innermost thing in the package,
    # called at every quadrature node, so it must not be rebuilt per call.
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
        # A power-law tail beats no exponential growth: int e^{t x} x^{-alpha-1}
        # diverges for every t > 0. M(0) = 1 exists, so the endpoint is
        # attained; anything strictly above it is refused.
        mgf_finite_below=0.0,
        imgf_sym=imgf_sym,
        logimgf_sym=logimgf_sym,
        imgf=lambda t_val, u_val: pareto_imgf(t_val, alpha_val, xi_val, u_val),
        logimgf=lambda t_val, u_val: pareto_logimgf(t_val, alpha_val, xi_val, u_val),
        # E[Theta^a] = alpha xi^a / (alpha - a) converges iff a < alpha, so the
        # bound is the tail index itself. Only consulted at t = 0.
        max_finite_moment=float(alpha_val),
        imgf_jax=lambda t_val, u_val: pareto_imgf_jax(t_val, alpha_val, xi_val, u_val),
        logimgf_jax=lambda t_val, u_val: pareto_logimgf_jax(
            t_val, alpha_val, xi_val, u_val
        ),
        params=params,
    )
