"""Compute ``D^a M(t)`` as the expectation it is, rather than by differentiating.

The operator's defining property is

.. math::

    D^a M(t) = \\mathbb{E}\\!\\left[\\Theta^a e^{t\\Theta}\\right]
             = \\int_0^\\infty \\theta^a e^{t\\theta} p(\\theta)\\, d\\theta

so a prior that supplies a density can have the quantity computed directly.
Every other backend takes the derivative of the MGF instead, symbolically or by
quadrature over the fractional-integral kernel.

Why this route exists
---------------------
**The integrand here is positive.** It therefore cannot suffer cancellation,
and that is not a marginal improvement — it is the difference between a correct
answer and a wrong one for priors whose CGF derivatives alternate in sign.

Measured against a Uniform(0.5, 2) prior at ``t = -1``, with an mpmath oracle at
80 digits as the reference:

======  ======================  ====================
order   differentiated MGF      this route (float64)
======  ======================  ====================
12      2.0e-10                 2.2e-16
20      1.8e-02                 2.3e-16
30      4.5e+08, wrong sign     8.7e-18
100     --                      1.3e-15
======  ======================  ====================

:file:`CLAUDE.md` recorded the order-30 case as unrecoverable at any precision,
confirmed with exact rationals and ``evalf(80)``. That is true of the
differentiated-MGF route, whose terms cancel through 25--26 digits before the
answer appears. It is not true of the problem: the expectation above never forms
those terms at all.

When it is used
---------------
``method="auto"`` prefers this route for **numeric evaluation**, whenever the
prior supplies a density -- which is always, since :class:`mitMGFprior` refuses
to construct without one in both its symbolic and its backend mode. An explicit
``method=`` is never diverted.

The qualifier "numeric evaluation" is load-bearing. With ``t=None`` the caller
is asking for a *representation*, and only a differentiating backend can build
one before an evaluation point is known, so ``auto`` must not be diverted then.
Routing it unconditionally silently removed the symbolic representation from
every ``auto`` posterior -- ``post_density(theta)`` stopped returning an
expression and ``int_tol`` stopped having any effect.

:func:`expectation_is_available` reports whether a prior can use this route, so
callers can decide rather than fail. It is what the sequential-update guard
consults: the prior built from a numeric posterior carries a density and no
MGF, so this route can consume it and the differentiating backends cannot.
"""

import math

import numpy as np
from scipy import integrate, optimize


def expectation_is_available(prior) -> bool:
    """Return True if ``prior`` exposes a density this route can integrate."""
    return (
        getattr(prior, "logpdf_func", None) is not None
        or getattr(prior, "pdf_func", None) is not None
    )


def _log_density(prior):
    """Return a callable giving ``log p(theta)``, preferring the log form."""
    log_pdf = getattr(prior, "logpdf_func", None)
    if log_pdf is not None:
        return lambda theta: np.asarray(log_pdf(theta), dtype=float)

    pdf = prior.pdf_func

    def from_pdf(theta):
        with np.errstate(divide="ignore"):
            return np.log(np.asarray(pdf(theta), dtype=float))

    return from_pdf


def _bracket(log_integrand, t_value, lower_hint=1e-12, upper_hint=1e12):
    """Bracket where the *integrand* carries its mass, for one evaluation point.

    Two things make this necessary rather than incidental.

    Priors do not declare their support, and it matters: a Uniform(0.5, 2)
    density is zero across almost all of the positive half-line, so a
    quadrature handed ``(0, inf)`` can miss the mass entirely.

    And the bracket has to be found on the integrand, not on the density. An
    earlier version of this walked outward until the *density* stopped being
    finite, which never terminates for a Gamma prior -- its log density is
    finite everywhere, merely tiny -- so the upper limit ran away to about
    1e60 and every Gamma answer came back as ``-inf``. The integrand
    ``theta**a e^{t theta} p(theta)`` is what actually decays, and it is what
    the peak search below already needs.

    The bracket is taken where the log integrand has fallen 700 below its peak,
    which is where its contribution passes under double precision's floor.
    """
    def value_at(theta):
        try:
            out = float(log_integrand(np.array([theta]), t_value)[0])
        except (ValueError, FloatingPointError, ZeroDivisionError):
            return -np.inf
        return out if math.isfinite(out) else -np.inf

    # Find any point with mass, scanning geometrically.
    grid = np.geomspace(lower_hint, upper_hint, 121)
    values = np.array([value_at(theta) for theta in grid])
    if not np.any(np.isfinite(values)):
        return None
    peak_index = int(np.argmax(values))
    peak_value = values[peak_index]

    below = peak_value - 700.0
    inside = np.flatnonzero(values > below)
    low = grid[max(inside[0] - 1, 0)]
    high = grid[min(inside[-1] + 1, len(grid) - 1)]

    # Refine each endpoint by bisection. The coarse grid is enough to *find*
    # the mass but not to bound it: for a Uniform prior the density is
    # discontinuous at its edges, and integrating across a discontinuity costs
    # `quad` several orders of accuracy. Measured on Uniform(0.5, 2), tightening
    # the endpoints onto the support took the relative error from 2.0e-09 to
    # the 1e-13 range.
    def refine(inside_point, outside_point):
        for _ in range(60):
            middle = 0.5 * (inside_point + outside_point)
            if middle in (inside_point, outside_point):
                break
            if value_at(middle) > below:
                inside_point = middle
            else:
                outside_point = middle
        return outside_point

    low = refine(grid[inside[0]], low)
    high = refine(grid[inside[-1]], high)
    return low, high, grid[peak_index]


def expectationDeriv(
    order,
    prior,
    t,
    u=None,
    complete=True,
    log=True,
):
    """Evaluate ``D^order M(t)`` as ``E[Theta^order e^{t Theta}]``.

    Parameters
    ----------
    order : float
        Derivative order. Any non-negative real; integer and fractional orders
        take the same path, since the expectation does not care.
    prior : mitMGFprior
        Prior supplying a density. Check with :func:`expectation_is_available`.
    t : float or array-like
        Evaluation point(s).
    u : float or array-like, optional
        Upper truncation point for the incomplete MGF.
    complete : bool, optional
        If False, integrate only up to ``u``.
    log : bool, optional
        Return ``(log_abs, sign)`` if True, else the plain value.

    Returns
    -------
    tuple or numpy.ndarray or float
        ``(log_abs, sign)`` when ``log`` is True, else the ordinary value.
        A scalar in gives a scalar out.

    Raises
    ------
    ValueError
        If ``order`` is negative, if the prior has no density, or if
        ``complete=False`` without ``u``.

    Notes
    -----
    The integral is accumulated in log space around the integrand's own peak.
    That is what keeps large orders usable: at order 100 the integrand spans
    hundreds of orders of magnitude, and integrating it directly would overflow
    long before the answer appeared. The sign is always ``+1`` -- the integrand
    is positive -- which is itself worth asserting, since the defect this route
    exists to avoid announces itself as a sign flip.
    """
    if order < 0:
        raise ValueError("Derivative order must be non-negative.")
    if not expectation_is_available(prior):
        raise ValueError(
            f"Prior '{getattr(prior, 'name', '?')}' provides no density, so "
            "D^a M(t) cannot be computed as an expectation. Use a "
            "differentiation backend instead."
        )
    if not complete and u is None:
        raise ValueError("u must be provided when complete=False.")

    log_density = _log_density(prior)

    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    if complete:
        u_arr = np.full(t_arr.shape, np.inf)
    else:
        u_arr = np.atleast_1d(np.asarray(u, dtype=float))
        t_arr, u_arr = np.broadcast_arrays(t_arr, u_arr)
    scalar_input = np.ndim(t) == 0 and (complete or np.ndim(u) == 0)

    def log_integrand(theta, t_value):
        with np.errstate(divide="ignore", invalid="ignore"):
            return order * np.log(theta) + t_value * theta + log_density(theta)

    results = np.empty(t_arr.shape, dtype=float)
    for index, (t_value, u_value) in enumerate(
        zip(t_arr.ravel(), u_arr.ravel(), strict=True)
    ):
        found = _bracket(log_integrand, t_value)
        if found is None:
            results.ravel()[index] = -np.inf
            continue
        low, high, peak_guess = found
        if np.isfinite(u_value):
            high = min(high, u_value)
        if high <= low:
            results.ravel()[index] = -np.inf
            continue

        # Scale by the integrand's peak before integrating. Without this,
        # `theta**order` overflows for large orders long before the exponential
        # brings it back down -- at order 100 the integrand spans hundreds of
        # orders of magnitude.
        peak = optimize.minimize_scalar(
            lambda th, tv=t_value: -float(log_integrand(np.array([th]), tv)[0]),
            bracket=None,
            bounds=(low, high),
            method="bounded",
        )
        offset = max(-float(peak.fun), float(log_integrand(np.array([peak_guess]), t_value)[0]))
        if not math.isfinite(offset):
            results.ravel()[index] = -np.inf
            continue

        def scaled(theta, tv=t_value, off=offset):
            with np.errstate(under="ignore"):
                return float(np.exp(log_integrand(np.array([theta]), tv)[0] - off))

        value, _ = integrate.quad(scaled, low, high, limit=200)
        results.ravel()[index] = -np.inf if value <= 0 else math.log(value) + offset

    signs = np.ones(results.shape, dtype=int)
    if scalar_input:
        if log:
            return float(results.reshape(())), 1
        return float(np.exp(results.reshape(())))

    if log:
        return results, signs
    with np.errstate(over="ignore"):
        return np.exp(results)
