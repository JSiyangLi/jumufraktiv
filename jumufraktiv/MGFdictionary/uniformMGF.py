"""
uniformMGF.py

Functions for the uniform prior: p(theta) = 1/(b-a) for theta in [a, b], else 0.

The MGF is:
    M(t) = (exp(t*b) - exp(t*a)) / (t*(b-a))   for t != 0
    M(0) = 1

For t < 0, the MGF is finite and positive. The CGF is defined as log M(t).

Numerical stability notes:
- The expressions involve differences of exponentials (exp(t*b) - exp(t*a)),
  which cancel when t is small or when a and b are close. `uniform_cgf` forms
  that difference with `logminus`, the log of a difference, rather than
  subtracting two logs.
- Neither factor of M(t) is positive on its own below the origin: exp(t*b) is
  then the smaller exponential and t is negative. The signs cancel in the
  ratio, so the numerator must be ordered by the sign of t. `uniform_cgf_jax`
  sidesteps this by dividing before taking the log, which is why the two
  implementations differ in shape.
- For t=0, the CGF is defined as 0 (and the MGF as 1) by continuity.

Symbolic, numeric (SciPy), and JAX backends are supported.
The incomplete MGF is the same integral stopped at u; see its section below.
"""

import math

import jax.numpy as jnp
import numpy as np
import scipy.stats as stats
import sympy as sp

from jumufraktiv.logsum import logminus
from jumufraktiv.registry import make_prior_spec, register_prior
from jumufraktiv.symbols import param, t, theta, u

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

    ``M(t) = (e^{bt} - e^{at}) / (t (b - a))`` is positive for every non-zero
    ``t``, but neither of its two factors is: below the origin ``e^{bt}`` is
    the *smaller* exponential and ``t`` is negative, so taking the log of each
    separately asks for the log of a negative number twice. The two signs
    cancel in the ratio and not before it, which is why the numerator is
    ordered by sign here and ``abs`` is taken of the denominator.

    ``t < 0`` is the whole of this package's operating range -- the posterior
    is evaluated at ``t = -b`` -- so the sign convention is not an edge case.
    """
    if t_val == 0.0:
        return 0.0

    # `logminus(x, y)` is log(e^x - e^y) and needs x > y, so the larger
    # exponent leads. Above the origin that is b*t, below it a*t.
    if t_val > 0:
        log_numerator = logminus(t_val * b_val, t_val * a_val)
    else:
        log_numerator = logminus(t_val * a_val, t_val * b_val)

    return log_numerator - math.log(abs(t_val) * (b_val - a_val))


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
    JAX-compatible CGF for the uniform prior.

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
    return jnp.log(
        (jnp.exp(t_val * b_val) - jnp.exp(t_val * a_val)) / (t_val * (b_val - a_val))
    )


def uniform_mgf_jax(t_val, a_val, b_val):
    """
    JAX-compatible MGF for the uniform prior.

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
# Incomplete MGF (lower truncation at u)
# ============================================================
#
# The incomplete MGF is the same integral as the MGF, stopped at u:
#
#     M(t, u) = int_a^min(u, b) e^{t x} / (b - a) dx
#             = (e^{t m} - e^{t a}) / (t (b - a)),   m = clip(u, a, b)
#
# and (m - a) / (b - a) at t = 0, by continuity. It is what `post_cdf` and
# everything built on it -- `post_quantile`, `post_interval`, `post_sample` --
# need from a prior.
#
# The same cancellation the module docstring describes for the MGF applies
# here, for the same reason: below the origin `e^{t m}` is the smaller
# exponential and `t` is negative, so neither factor is positive on its own.
# The numerator is therefore ordered by the sign of `t`, and formed with
# `logminus` rather than by subtracting two logs.


def uniform_imgf_symbolic(u_sym):
    """
    Symbolic expression for the lower-truncated uniform MGF.

    Parameters
    ----------
    u_sym : sympy.Symbol
        Upper truncation point.

    Returns
    -------
    sympy.Expr
        `(exp(t*min(u, b)) - exp(t*a)) / (t*(b - a))`, clipped to the support.
    """
    m = sp.Min(sp.Max(u_sym, a), b)
    return (sp.exp(t * m) - sp.exp(t * a)) / (t * (b - a))


def uniform_logimgf(t_val, a_val, b_val, u_val):
    """
    Numeric log incomplete MGF (vectorised, stable).

    Parameters
    ----------
    t_val : float or array
        Evaluation point.
    a_val, b_val : float
        Support bounds.
    u_val : float or array
        Upper truncation point.

    Returns
    -------
    float or numpy.ndarray
        `log M(t, u)`, and `-inf` where `u <= a`, which is the correct log of
        a truncation that captures no mass.
    """
    t_arr, u_arr = np.broadcast_arrays(
        np.asarray(t_val, dtype=float), np.asarray(u_val, dtype=float)
    )
    m = np.clip(u_arr, a_val, b_val)
    width = math.log(b_val - a_val)

    # Order the difference by the sign of `t` so `logminus` receives the larger
    # exponent first; the sign of the numerator and of `t` then cancel. Where
    # `m == a` the two exponents are equal and `logminus` returns `-inf`, which
    # is the right answer for a truncation that captures no mass, so the
    # below-support case needs no mask of its own.
    above = t_arr > 0.0
    larger = np.where(above, t_arr * m, t_arr * a_val)
    smaller = np.where(above, t_arr * a_val, t_arr * m)

    # `t == 0` is a separate formula, not a limit to be approached: the
    # numerator and `t` both vanish and the value is the plain mass fraction.
    # Both branches are evaluated over the whole array and selected afterwards,
    # so the placeholders below only keep the unused branch from warning.
    nonzero = t_arr != 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        off_origin = (
            logminus(larger, smaller)
            - np.log(np.abs(np.where(nonzero, t_arr, 1.0)))
            - width
        )
        at_origin = np.log(np.where(m > a_val, m - a_val, 1.0)) - width
        at_origin = np.where(m > a_val, at_origin, -np.inf)

    out = np.where(nonzero, off_origin, at_origin)

    return out if out.ndim else float(out)


def uniform_imgf(t_val, a_val, b_val, u_val):
    """Numeric incomplete MGF (ordinary scale, vectorised)."""
    return np.exp(uniform_logimgf(t_val, a_val, b_val, u_val))


def uniform_logimgf_jax(t_val, a_val, b_val, u_val):
    """
    JAX log incomplete MGF.

    Notes
    -----
    Divides before taking the log, as `uniform_cgf_jax` does, so that the two
    negative factors cancel without a signed intermediate. That costs the
    small-`t` stability the SciPy path gets from `logminus`, which is the same
    trade the MGF makes.

    `t = 0` is selected rather than approached. The quotient is `0/0` there and
    evaluates to `nan`, where the value is the plain mass fraction
    `(m - a)/(b - a)` -- and `t = 0` is reachable, since `b(y) = 0` whenever
    every observation sits at the value the likelihood subtracts. Both branches
    are computed and chosen between, because `jnp.where` evaluates both sides.
    """
    m = jnp.clip(u_val, a_val, b_val)
    safe_t = jnp.where(t_val == 0, 1.0, t_val)

    off_origin = (jnp.exp(safe_t * m) - jnp.exp(safe_t * a_val)) / (
        safe_t * (b_val - a_val)
    )
    at_origin = (m - a_val) / (b_val - a_val)

    return jnp.log(jnp.where(t_val == 0, at_origin, off_origin))


def uniform_imgf_jax(t_val, a_val, b_val, u_val):
    """JAX incomplete MGF (ordinary scale)."""
    return jnp.exp(uniform_logimgf_jax(t_val, a_val, b_val, u_val))


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
    imgf_sym = uniform_imgf_symbolic(u)
    logimgf_sym = sp.log(imgf_sym)

    # Freeze the SciPy distribution ONCE, here rather than inside the lambdas
    # below. `stats.<dist>(params)` builds an `rv_frozen`, and building one runs
    # `_construct_doc`, which formats a docstring -- about 430 us before any
    # density is evaluated. The density is the innermost thing in the package,
    # called at every quadrature node, so it must not be rebuilt per call.
    frozen = stats.uniform(loc=a_val, scale=b_val - a_val)

    # Substitute numeric parameter values into the symbolic expressions
    subs_map = {a: a_val, b: b_val}
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

        # Bounded support, so every moment is finite and no order is
        # inadmissible at t = 0.
        max_finite_moment=float("inf"),

        # Bounded support, so M(t) = E[e^{t theta}] <= e^{t b} is finite for
        # every real t: the MGF is entire.
        mgf_finite_below=float("inf"),

        mgf=lambda t_val: uniform_mgf(t_val, a_val, b_val),
        cgf=lambda t_val: uniform_cgf(t_val, a_val, b_val),

        mgf_jax=lambda t_val: uniform_mgf_jax(t_val, a_val, b_val),
        cgf_jax=lambda t_val: uniform_cgf_jax(t_val, a_val, b_val),

        imgf_sym=imgf_sym,
        logimgf_sym=logimgf_sym,
        imgf=lambda t_val, u_val: uniform_imgf(t_val, a_val, b_val, u_val),
        logimgf=lambda t_val, u_val: uniform_logimgf(t_val, a_val, b_val, u_val),
        imgf_jax=lambda t_val, u_val: uniform_imgf_jax(t_val, a_val, b_val, u_val),
        logimgf_jax=lambda t_val, u_val: uniform_logimgf_jax(
            t_val, a_val, b_val, u_val
        ),

        pdf_func=frozen.pdf,
        logpdf_func=frozen.logpdf,

        params=params,
    )
