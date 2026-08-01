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

The differentiated-MGF route loses these orders to 25--26 digits of
cancellation rather than to rounding, so no increase in working precision
recovers them. Only computing the expectation directly avoids the loss, because
it never forms the cancelling terms at all.

When it is used
---------------
``method="auto"`` prefers this route for **numeric evaluation**, whenever the
prior supplies a density -- which is always, since :class:`mitMGFprior` refuses
to construct without one in both its symbolic and its backend mode. An explicit
``method=`` is never diverted.

The qualifier "numeric evaluation" is load-bearing. With ``t=None`` the caller
is asking for a *representation*, and only a differentiating backend can build
one before an evaluation point is known, so ``auto`` must not be diverted then.

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
    """Return a callable giving ``log p(theta)``, preferring the log form.

    The result is guaranteed to accept an array of ``theta`` and return one
    value per element -- see :func:`_vectorise`, which every caller goes
    through. Downstream code may assume that without checking.
    """
    log_pdf = getattr(prior, "logpdf_func", None)
    if log_pdf is not None:
        return _vectorise(lambda theta: np.asarray(log_pdf(theta), dtype=float))

    pdf = prior.pdf_func

    def from_pdf(theta):
        with np.errstate(divide="ignore"):
            return np.log(np.asarray(pdf(theta), dtype=float))

    return _vectorise(from_pdf)


def _vectorise(density):
    """Return ``density`` if it takes an array, else an elementwise adapter.

    Decided **once**, by probing with a two-element array, rather than guarded
    at each call site. That matters for more than tidiness.

    The registry's priors all take arrays. A caller-supplied one need not: a
    density written as ``0.0 if theta >= k else -inf`` accepts a one-element
    array and raises on anything longer, and one written with ``math`` accepts
    no array at all. Handling that per call site turns a real failure into a
    plausible number, because a per-call fallback catches the ``TypeError`` and
    substitutes ``-inf`` -- turning "this density cannot be called that way"
    into "the integrand has no mass here".

    Probing once removes that: the adapter is chosen before any evaluation, so
    every point sees the same function, and a density that works under neither
    calling convention raises **the caller's own exception** instead of being
    converted into a number.
    """
    # Two elements, deliberately. A one-element array satisfies a Python `if`,
    # so a shorter probe -- or a guard at each call site -- would make the
    # adapter choice, and hence the answer, depend on the batch size.
    probe = np.array([1.0, 2.0])
    try:
        probed = np.asarray(density(probe), dtype=float)
        if probed.shape == probe.shape:
            return density
    except (ValueError, TypeError, FloatingPointError, ZeroDivisionError):
        # Not a failure: the answer to "does this take an array?" is no.
        pass

    def elementwise(theta):
        values = np.asarray(theta, dtype=float)
        flat = []
        for x in values.ravel():
            one = np.ravel(np.asarray(density(float(x)), dtype=float))
            if one.size != 1:
                # Taking `one[0]` here would integrate a density that is
                # answering a different question, and say nothing about it.
                raise ValueError(
                    "A density called with a single theta returned "
                    f"{one.size} values. It must return exactly one; the "
                    "elementwise adapter cannot guess which is meant."
                )
            flat.append(float(one[0]))
        return np.asarray(flat, dtype=float).reshape(values.shape)

    # Confirm the adapter works before promising that it does. If the density
    # cannot be called elementwise either, this raises the caller's exception,
    # which is the honest outcome -- and it happens once, at setup, rather than
    # from inside a quadrature where the traceback says nothing useful.
    elementwise(probe)
    return elementwise


def _add_polynomial_tail(
    log_body, *, order, prior, t_values, u_values, highs, log_density
):
    """Add the mass beyond ``highs`` when the integrand decays only polynomially.

    Parameters
    ----------
    log_body : numpy.ndarray
        Log of the quadrature over ``[low, high]``, one entry per live point.
    order : float
        Derivative order ``a``.
    prior : mitMGFprior
        Consulted for ``max_finite_moment``.
    t_values, u_values : numpy.ndarray
        Evaluation and truncation points for the live entries.
    highs : numpy.ndarray
        Upper end of each point's bracket.
    log_density : callable
        ``log p(theta)``, vectorised.

    Returns
    -------
    numpy.ndarray
        ``log_body`` with the tail added where one applies, unchanged elsewhere.

    Notes
    -----
    At ``t < 0`` the factor ``e^{t theta}`` forces geometric decay and the
    bracket reaches the point where the integrand has fallen 700 below its
    peak, so nothing measurable is left beyond it. **At ``t = 0`` exactly there
    is no such factor**: the integrand is ``theta^a p(theta)``, which for a
    heavy-tailed prior decays like ``theta^{a - alpha - 1}``, and the closer
    ``a`` comes to ``alpha`` the slower.

    No change of variable rescues this, which is why the tail is supplied
    rather than integrated. The remaining mass sits where ``theta`` is not a
    float64: reaching a relative 1e-10 at ``a = 1.99`` against Pareto(2) needs
    the integral carried to ``theta ~ 1e1000``, and even at the representable
    limit of 1.8e308 a thousandth of the answer is still outside.

    So the tail is taken from the prior's own declaration. A finite
    ``max_finite_moment`` says ``E[Theta^a]`` diverges at ``a = alpha``, which
    is the statement that ``p`` has tail index ``alpha``; reading the constant
    off the density at ``T`` gives

    .. math:: \\int_T^\\infty \\theta^a p(\\theta)\\,d\\theta
              = \\frac{p(T)\\,T^{a+1}}{\\alpha - a}.

    For a Pareto prior that is exact rather than asymptotic, since its density
    *is* a power law. For a prior that only approaches one, it is the leading
    term, and still far closer than dropping the tail entirely.

    A prior declaring ``max_finite_moment = inf`` gets nothing added, correctly:
    its tail is lighter than any power law, so the bracket already holds it.
    """
    limit = float(getattr(prior, "max_finite_moment", np.inf))
    if np.isinf(limit) or float(order) >= limit:
        return log_body

    # Only where there is no exponential to force decay, and where the caller
    # has not truncated the integral itself.
    applies = (t_values == 0.0) & ~np.isfinite(u_values)
    if not np.any(applies):
        return log_body

    with np.errstate(divide="ignore", invalid="ignore"):
        log_tail = (
            log_density(highs)
            + (float(order) + 1.0) * np.log(highs)
            - math.log(limit - float(order))
        )

    usable = applies & np.isfinite(log_tail)
    if not np.any(usable):
        return log_body

    return np.where(usable, np.logaddexp(log_body, log_tail), log_body)


def _bracket(log_integrand, t_value, lower_hint=1e-12, upper_hint=1e12):
    """Bracket where the *integrand* carries its mass, for one evaluation point.

    Two things make this necessary rather than incidental.

    Priors do not declare their support, and it matters: a Uniform(0.5, 2)
    density is zero across almost all of the positive half-line, so a
    quadrature handed ``(0, inf)`` can miss the mass entirely.

    And the bracket has to be found on the integrand, not on the density. The
    integrand ``theta**a e^{t theta} p(theta)`` is what actually decays, and it
    is what the peak search below already needs.

    The bracket is taken where the log integrand has fallen 700 below its peak,
    which is where its contribution passes under double precision's floor.
    """
    def values_at(thetas):
        """Log integrand at every theta in one call, non-finite mapped to -inf.

        `log_integrand` is elementwise in theta, so the 121-point scan below
        and the two bisections are each one call rather than one call per
        point. Every such call reaches the prior's density, which is the
        innermost cost on this route.
        """
        # No try/except here, deliberately: `_vectorise` settled at setup that
        # this density can be called this way. Catching would turn a failed
        # call into `-inf`, which reads as "no mass at this theta" rather than
        # "this call did not work".
        out = np.asarray(log_integrand(np.asarray(thetas, dtype=float), t_value))
        return np.where(np.isfinite(out), out, -np.inf)

    # Find any point with mass, scanning geometrically. The scan is on the
    # integrand, not the density: a Gamma log density is finite everywhere,
    # merely tiny, so a density-based outward walk has no termination criterion
    # at all and the upper limit runs away.
    grid = np.geomspace(lower_hint, upper_hint, 121)
    values = values_at(grid)
    if not np.any(np.isfinite(values)):
        return None
    peak_index = int(np.argmax(values))
    peak_value = values[peak_index]

    below = peak_value - 700.0
    inside = np.flatnonzero(values > below)
    low = grid[max(inside[0] - 1, 0)]
    high = grid[min(inside[-1] + 1, len(grid) - 1)]

    # Refine both endpoints by bisection, together. The coarse grid is enough
    # to *find* the mass but not to bound it: for a Uniform prior the density
    # is discontinuous at its edges, and integrating across a discontinuity
    # costs `quad` several orders of accuracy, so the endpoints must be
    # tightened onto the support.
    #
    # The `stuck` mask is what makes doing both endpoints at once equivalent
    # to doing each alone: an element whose midpoint has stopped moving stops
    # being updated, while the other continues.
    interior = np.array([grid[inside[0]], grid[inside[-1]]], dtype=float)
    exterior = np.array([low, high], dtype=float)
    for _ in range(60):
        middle = 0.5 * (interior + exterior)
        stuck = (middle == interior) | (middle == exterior)
        if np.all(stuck):
            break
        keep = values_at(middle) > below
        interior = np.where(stuck | ~keep, interior, middle)
        exterior = np.where(stuck | keep, exterior, middle)
    low, high = float(exterior[0]), float(exterior[1])
    return low, high, grid[peak_index]


#: Relative tolerance handed to :func:`scipy.integrate.quad_vec` when the caller
#: names none. Tight enough that the route's accuracy is set by the integrand
#: rather than by the stopping rule.
DEFAULT_TOL = 1e-10


def expectationDeriv(
    order,
    prior,
    t,
    u=None,
    complete=True,
    log=True,
    tol=DEFAULT_TOL,
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
    tol : float, optional
        Relative tolerance for the quadrature, passed to
        :func:`scipy.integrate.quad_vec` as ``epsrel``. Defaults to
        :data:`DEFAULT_TOL`.

    Returns
    -------
    tuple or numpy.ndarray or float
        ``(log_abs, sign)`` when ``log`` is True, else the ordinary value.
        A scalar in gives a scalar out.

    Raises
    ------
    ValueError
        If ``order`` is negative, if ``tol`` is not positive, if the prior has
        no density, or if ``complete=False`` without ``u``.

    Notes
    -----
    The integral is accumulated in log space around the integrand's own peak.
    That is what keeps large orders usable: at order 100 the integrand spans
    hundreds of orders of magnitude, and integrating it directly would overflow
    long before the answer appeared. The sign is always ``+1`` -- the integrand
    is positive.

    ``tol`` is the quadrature's relative tolerance and nothing else; it does
    not bound the error of the returned logarithm, which also carries the
    accuracy of the peak location used to rescale the integrand.
    """
    if order < 0:
        raise ValueError("Derivative order must be non-negative.")
    if not (tol > 0):
        raise ValueError(f"tol must be positive, got {tol!r}.")
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

    # ---- Per point: locate the mass and its peak -------------------------
    #
    # This part is a loop, and cheaply so: bracketing costs two vectorised
    # calls per point and the peak search a handful more. The quadrature below
    # is the expensive part, and that is the part which batches.
    t_flat = t_arr.ravel()
    u_flat = u_arr.ravel()
    results = np.full(t_flat.shape, -np.inf, dtype=float)

    live, lows, highs, offsets = [], [], [], []
    for index, (t_value, u_value) in enumerate(zip(t_flat, u_flat, strict=True)):
        found = _bracket(log_integrand, t_value)
        if found is None:
            continue
        low, high, peak_guess = found
        if np.isfinite(u_value):
            high = min(high, u_value)
        if high <= low:
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
        offset = max(
            -float(peak.fun),
            float(log_integrand(np.array([peak_guess]), t_value)[0]),
        )
        if not math.isfinite(offset):
            continue

        live.append(index)
        lows.append(low)
        highs.append(high)
        offsets.append(offset)

    # ---- All points at once: one adaptive quadrature ---------------------
    #
    # Each point has its own interval, so they are mapped to a common [0, 1]
    # by theta_i(s) = low_i + s * width_i, which turns N integrals over N
    # intervals into one integral of an N-vector:
    #
    #     int_{low_i}^{high_i} f_i(theta) dtheta
    #         = width_i * int_0^1 f_i(low_i + s width_i) ds
    #
    # `quad_vec` then runs ONE adaptive subdivision for the whole batch,
    # evaluating every point's integrand at each node. Integrating the points
    # one at a time instead would run a separate subdivision per point, and
    # every node of every one of them would reach the prior's density with a
    # single scalar.
    #
    # Sharing the subdivision is safe here *because* of the peak scaling
    # above: every component is O(1) at its own peak, so no component is
    # negligible against the others and the error norm cannot be dominated by
    # one point while another is left inaccurate. Without that scaling this
    # would be a real hazard rather than a bookkeeping detail.
    if live:
        lows = np.asarray(lows, dtype=float)
        widths = np.asarray(highs, dtype=float) - lows
        offsets = np.asarray(offsets, dtype=float)
        t_live = t_flat[live]

        def batched(s):
            theta = lows + s * widths
            # `over` belongs in this list. The suite runs with
            # `filterwarnings = ["error"]`, so NumPy's "overflow encountered
            # in exp" is an exception under pytest and a warning everywhere
            # else, and a path that takes a different branch in the suite than
            # in a user's session is one the suite cannot vouch for. Overflow
            # here needs an underestimated offset -- a multimodal
            # caller-supplied density whose global peak the bounded search
            # missed -- and it then surfaces as `inf`, loudly, rather than as
            # a plausible number.
            with np.errstate(
                divide="ignore", invalid="ignore", under="ignore", over="ignore"
            ):
                exponent = (
                    order * np.log(theta) + t_live * theta + log_density(theta)
                )
                return np.exp(exponent - offsets) * widths

        values, _ = integrate.quad_vec(batched, 0.0, 1.0, epsrel=tol)
        values = np.atleast_1d(np.asarray(values, dtype=float))
        with np.errstate(divide="ignore"):
            results[live] = np.where(
                values > 0, np.log(np.where(values > 0, values, 1.0)) + offsets, -np.inf
            )

        results[live] = _add_polynomial_tail(
            results[live],
            order=order,
            prior=prior,
            t_values=t_flat[live],
            u_values=u_flat[live],
            highs=np.asarray(highs, dtype=float),
            log_density=log_density,
        )

    results = results.reshape(t_arr.shape)

    signs = np.ones(results.shape, dtype=int)
    if scalar_input:
        if log:
            return float(results.reshape(())), 1
        return float(np.exp(results.reshape(())))

    if log:
        return results, signs
    with np.errstate(over="ignore"):
        return np.exp(results)
