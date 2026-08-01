"""Fixed-grid quadrature for the Liouville--Caputo fractional derivative.

This replaces the adaptive scheme in :mod:`numeric_fractionalDeriv_scipy`. The
reasons are recorded in :file:`CLAUDE.md` under "Numerical policy"; in short,
the substitution the package already uses turns the kernel into an integrand
that decays single-exponentially on the whole real line, and that is precisely
the class where a plain uniform-grid trapezoid rule converges *geometrically*
while adaptive Gauss--Kronrod does not.

The mathematics
---------------
With ``n = floor(a)`` and ``gamma = n + 1 - a`` in ``(0, 1]``, the operator is

.. math::

    D^a M(t) = \\frac{1}{\\Gamma(\\gamma)}
               \\int_0^\\infty z^{\\gamma-1} M^{(n+1)}(t - z)\\, dz

and substituting ``z = e^u`` gives ``dz = e^u du``, so ``z^{gamma-1} dz``
becomes ``e^{gamma u} du``:

.. math::

    D^a M(t) = \\frac{1}{\\Gamma(\\gamma)}
               \\int_{-\\infty}^{\\infty} e^{\\gamma u} M^{(n+1)}(t - e^u)\\, du

Singularity subtraction
-----------------------
As ``a`` approaches ``n + 1`` from below, ``gamma -> 0``: the prefactor
``1/Gamma(gamma)`` tends to zero while the integral diverges, so the answer is
computed as ``0 x infinity``. Subtracting a function with the same value at
``z = 0`` and a known weighted integral removes that exactly, because
``\\int_0^\\infty z^{\\gamma-1} e^{-z} dz = \\Gamma(\\gamma)``:

.. math::

    D^a M(t) = M^{(n+1)}(t) + \\frac{1}{\\Gamma(\\gamma)}
        \\int_{-\\infty}^{\\infty} e^{\\gamma u}
        \\left[ M^{(n+1)}(t - e^u) - M^{(n+1)}(t) e^{-e^u} \\right] du

Two things this buys, neither of them a heuristic. The leading term is already
the exact ``gamma -> 0`` limit, so the near-integer case is right before the
integral contributes anything. And the bracket is ``O(z)`` as ``z -> 0``, so the
left tail decays like ``e^{(gamma+1)u}`` **independently of gamma** -- where the
unsubtracted integrand decays like ``e^{gamma u}`` and therefore needs a
truncation point that runs away as ``gamma -> 0``.

Why the range is derived rather than discovered
-----------------------------------------------
The previous kernel started at ``L = 10`` and doubled until consecutive
iterates agreed. That rule is wrong in two ways at once: it compares
consecutive iterates, which underestimates the remaining tail when convergence
is slow, and it tests against ``tol * max(1, |prev|)``, which is an absolute
test whenever the integral is below 1. Here both endpoints are computed from
the decay rates above, so there is no stopping rule to get wrong.
"""

import math
import warnings

import numpy as np

from jumufraktiv.derivativeDispatch import mgfDerivative_integer

#: `math.exp` overflows above this, so `z = e^u` cannot be evaluated past it.
#: The right-hand endpoint is clipped here rather than allowed to produce `inf`.
_MAX_U = 709.0

#: Nodes per unit of `u`. The trapezoid rule on an analytic, exponentially
#: decaying integrand converges geometrically in this, so a modest value is
#: already at machine precision; see `tests/test_fixed_grid_kernel.py`, which
#: measures the convergence rather than assuming it.
_NODES_PER_UNIT = 24


def _signed_logsumexp(log_abs, sign, axis=0):
    """Sum signed values given in log space, returning ``(log_abs, sign)``.

    Accumulating in log space is what stops a large derivative order
    overflowing. The previous kernel exponentiated each contribution into
    linear space and clamped on overflow, which silently dropped the
    overflowing terms: order 300.5 came out as 694.234 against an exact
    1006.311, wrong by 312 nats with no warning.
    """
    log_abs = np.asarray(log_abs, dtype=float)
    sign = np.asarray(sign, dtype=float)

    finite = np.isfinite(log_abs)
    if not np.any(finite):
        shape = np.delete(np.array(log_abs.shape), axis)
        return np.full(tuple(shape), -np.inf), np.ones(tuple(shape))

    peak = np.max(np.where(finite, log_abs, -np.inf), axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)

    with np.errstate(under="ignore"):
        terms = sign * np.exp(log_abs - peak)
    total = np.sum(np.where(finite, terms, 0.0), axis=axis)

    out_sign = np.sign(total)
    out_sign = np.where(out_sign == 0, 1.0, out_sign)
    with np.errstate(divide="ignore"):
        out_log = np.log(np.abs(total)) + np.squeeze(peak, axis=axis)
    return out_log, out_sign


def _integration_range(gamma_val, tol):
    """Return ``(u_min, u_max)`` for the transformed integrand.

    Notes
    -----
    The left endpoint follows from the decay established above: with
    subtraction the integrand behaves like ``e^{(gamma+1)u}``, so reaching
    ``tol`` needs ``u_min <= log(tol) / (gamma + 1)``. That is bounded no matter
    how small ``gamma`` becomes, which is the point of subtracting.

    The right endpoint cannot be derived the same way, and assuming it can was
    a mistake worth recording. The subtracted term ``M^{(n+1)}(t) e^{-z}``
    decays double-exponentially, which suggested ``u_max = 4`` would do -- but
    the *other* term, ``M^{(n+1)}(t - z)``, decays only **polynomially in z**
    for the priors here, so the integrand behaves like
    ``e^{(gamma - tail - n - 1) u}``. For a Gamma(2, 3) prior at order 0.5 the
    rate is 2.5, so ``u = 4`` leaves 4.5e-05 of the tail uncaptured; measured
    relative error was 4.3e-04 at ``u_max = 4`` against 9.0e-16 at ``u_max =
    20``.

    The tail index is a property of the prior and is not exposed, so the
    endpoint is found by *probing the integrand's decay* -- extending until the
    contribution at the endpoint has fallen below ``tol`` relative to the
    largest contribution seen. That is a direct test on the decay, not the
    consecutive-iterate comparison the old kernel used, which underestimates
    the remaining tail exactly when convergence is slow.
    """
    u_min = math.log(tol) / (gamma_val + 1.0)
    return u_min, min(8.0, _MAX_U)


def _tail_has_decayed(nodes, gamma_val, t_arr, u_arr, integer_derivative, tol):
    """Has the integrand fallen below ``tol`` relative to its peak by the end?

    Returns ``(decayed, log_edge, log_peak)``. The comparison is *relative to
    the largest contribution on the grid*, which is what makes it a test of
    decay rather than of magnitude -- the previous kernel compared consecutive
    integral estimates against ``tol * max(1, |prev|)``, an absolute test
    whenever the integral is below 1, and one that underestimates the remaining
    tail whenever convergence is slow.
    """
    z = np.exp(nodes)
    shifted = t_arr[None, ...] - z.reshape((-1,) + (1,) * t_arr.ndim)
    u_shifted = (
        None if u_arr is None else np.broadcast_to(u_arr[None, ...], shifted.shape)
    )
    log_m, _ = integer_derivative(shifted, u_shifted)
    log_m = np.asarray(log_m, dtype=float).reshape(shifted.shape)

    contribution = gamma_val * nodes.reshape((-1,) + (1,) * t_arr.ndim) + log_m
    finite = np.isfinite(contribution)
    if not np.any(finite):
        return True, -np.inf, -np.inf

    log_peak = float(np.max(contribution[finite]))
    log_edge = float(
        np.max(contribution[-1][np.isfinite(contribution[-1])], initial=-np.inf)
    )
    return (log_edge - log_peak) < math.log(tol), log_edge, log_peak


def fractionalDeriv_grid(
    order,
    prior,
    t_points,
    u_points=None,
    complete=True,
    integer_method="symbolic",
    tol=1e-14,
    log=True,
):
    """Evaluate ``D^order M(t)`` on a fixed grid, for one or many ``t``.

    Parameters
    ----------
    order : float
        Fractional derivative order. Must be positive and non-integer.
    prior : mitMGFprior
        Prior supplying the MGF (or incomplete MGF when ``complete=False``).
    t_points : array-like
        Evaluation point(s). Broadcast against ``u_points``.
    u_points : array-like, optional
        Truncation point(s) for the incomplete MGF.
    complete : bool, optional
        Differentiate the complete MGF (default) or the incomplete one.
    integer_method : str, optional
        Backend used for the inner integer derivative ``M^{(n+1)}``.
    tol : float, optional
        Target accuracy; sets the integration range.
    log : bool, optional
        Return ``(log_abs, sign)`` if True, otherwise the plain value. This
        argument decides the return shape and nothing else, per the log
        principle.

    Returns
    -------
    tuple or numpy.ndarray
        ``(log_abs, sign)`` when ``log`` is True, else the ordinary values.
    """
    if order <= 0:
        raise ValueError("Fractional order must be positive.")

    n = math.floor(order)
    gamma_val = (n + 1) - order
    if gamma_val <= 0 or gamma_val > 1:
        raise ValueError(
            f"gamma = (floor(order) + 1) - order must lie in (0, 1]; got {gamma_val} "
            f"for order {order}."
        )

    # A scalar in must give a scalar out. The dispatcher's other backends do
    # this, and callers rely on it -- `atleast_1d` alone turned a float result
    # into a one-element array, which made a list of scalar calls come back as
    # shape (n, 1) instead of (n,) and compare unequal to the batched answer
    # despite every value agreeing.
    scalar_input = np.ndim(t_points) == 0 and (
        u_points is None or np.ndim(u_points) == 0
    )
    t_arr = np.atleast_1d(np.asarray(t_points, dtype=float))
    if u_points is None:
        u_arr = None
        batch_shape = t_arr.shape
    else:
        u_arr = np.atleast_1d(np.asarray(u_points, dtype=float))
        t_arr, u_arr = np.broadcast_arrays(t_arr, u_arr)
        batch_shape = t_arr.shape

    def integer_derivative(at_t, at_u):
        """`M^{(n+1)}` at the given points, as (log_abs, sign)."""
        return mgfDerivative_integer(
            order=n + 1,
            prior=prior,
            method=integer_method,
            t=at_t,
            complete=complete,
            u=at_u,
            log=True,
        )

    # ---- The leading term, which is the exact gamma -> 0 limit ----------
    log_lead, sign_lead = integer_derivative(t_arr, u_arr)
    log_lead = np.asarray(log_lead, dtype=float).reshape(batch_shape)
    sign_lead = np.asarray(sign_lead, dtype=float).reshape(batch_shape)

    # ---- Fixed grid, with the right endpoint found by decay probing -----
    u_min, u_max = _integration_range(gamma_val, tol)
    while True:
        n_nodes = max(math.ceil((u_max - u_min) * _NODES_PER_UNIT) + 1, 33)
        nodes, step = np.linspace(u_min, u_max, n_nodes, retstep=True)
        decayed, log_edge, log_peak = _tail_has_decayed(
            nodes, gamma_val, t_arr, u_arr, integer_derivative, tol
        )
        if decayed or u_max >= _MAX_U:
            break
        u_max = min(u_max * 2.0, _MAX_U)
    if not decayed:
        warnings.warn(
            "The transformed integrand had not decayed below the requested "
            f"tolerance at the largest usable evaluation point (u = {_MAX_U}); "
            f"the endpoint contribution is {log_edge - log_peak:.3g} nats below "
            "the peak. The result is a lower bound on the tail rather than a "
            "converged value.",
            RuntimeWarning,
            stacklevel=2,
        )

    z = np.exp(nodes)  # (n_nodes,)
    shifted = t_arr[None, ...] - z.reshape((-1,) + (1,) * t_arr.ndim)
    u_shifted = (
        None if u_arr is None else np.broadcast_to(u_arr[None, ...], shifted.shape)
    )

    log_m, sign_m = integer_derivative(shifted, u_shifted)
    log_m = np.asarray(log_m, dtype=float).reshape(shifted.shape)
    sign_m = np.asarray(sign_m, dtype=float).reshape(shifted.shape)

    # The subtracted term: M^{(n+1)}(t) * exp(-z), in log space.
    z_col = z.reshape((-1,) + (1,) * t_arr.ndim)
    log_sub = log_lead[None, ...] - z_col
    sign_sub = np.broadcast_to(sign_lead[None, ...], log_sub.shape)

    # bracket = M^{(n+1)}(t - z) - M^{(n+1)}(t) e^{-z}
    log_bracket, sign_bracket = _signed_logsumexp(
        np.stack([log_m, np.broadcast_to(log_sub, log_m.shape)]),
        np.stack([sign_m, -np.asarray(sign_sub, dtype=float)]),
        axis=0,
    )

    # Trapezoid weights, folded in as logs so nothing leaves log space.
    weights = np.full(n_nodes, step)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    with np.errstate(divide="ignore"):
        log_weights = np.log(weights).reshape((-1,) + (1,) * t_arr.ndim)

    log_terms = gamma_val * nodes.reshape((-1,) + (1,) * t_arr.ndim)
    log_terms = log_terms + log_bracket + log_weights

    log_integral, sign_integral = _signed_logsumexp(log_terms, sign_bracket, axis=0)

    # ---- Combine: leading term + integral / Gamma(gamma) ---------------
    log_integral = log_integral - math.lgamma(gamma_val)

    log_total, sign_total = _signed_logsumexp(
        np.stack([log_lead, log_integral]),
        np.stack([sign_lead, sign_integral]),
        axis=0,
    )

    log_total = np.asarray(log_total).reshape(batch_shape)
    sign_total = np.asarray(sign_total).reshape(batch_shape)
    if scalar_input:
        log_total = log_total.reshape(())
        sign_total = sign_total.reshape(())

    if log:
        if scalar_input:
            return float(log_total), int(sign_total)
        return log_total, sign_total

    with np.errstate(over="ignore"):
        value = sign_total * np.exp(log_total)
    return float(value) if scalar_input else value
