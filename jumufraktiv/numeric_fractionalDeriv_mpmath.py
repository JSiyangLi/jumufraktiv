"""
numeric_fractionalDeriv_mpmath.py

Numerical computation of Liouville‑Caputo fractional derivatives of MGFs
using mpmath.quad (arbitrary precision) with the substitution z = e^u.

The main function fractionalDeriv_numeric_mpmath() uses adaptive range expansion
(default). If that fails or if use_tan=True, it uses the tan‑transform method
(fractionalDeriv_numeric_mpmath_tan) which maps (-∞,∞) to (-π/2, π/2).

The formula computed is:
    D^α_{(-∞)+} M(t) = 1/Γ(γ) ∫_{-∞}^{∞} e^{γ u} M^{(n+1)}(t - e^{u}) du,
where n = floor(α), γ = n+1-α.

All arithmetic is performed with the precision specified by mp.dps.
"""
import math
import warnings
import numpy as np
from mpmath import mp, pi, exp, log, tan, gamma, quad, mpf
import sympy as sp
from jumufraktiv.derivativeDispatch import mgfDerivative_integer
from jumufraktiv.mitMGFprior_class import mitMGFprior
from jumufraktiv.symbolic_cache import cached_diff
from jumufraktiv.symbols import t as _t_sym, u as _u_sym


def fractionalDeriv_numeric_mpmath_tan(
    order: float,
    prior: mitMGFprior,
    t: float | np.ndarray | list,
    method: str = "symbolic",
    simplify: bool = False,
    complete: bool = True,
    return_log: bool = False,
    margin: float = 1e-12,
    max_u: float = 20.0,
    dps: int = 50,
    u: float | np.ndarray | list | None = None
):
    """
    Compute fractional derivative using scaled tan‑transform with mpmath.
    Supports tuple‑vectorisation: t and u are broadcast to a common shape.

    Parameters
    ----------
    order : float
        Fractional order (positive).
    prior : mitMGFprior
        Prior object providing the MGF.
    t : float or array-like
        Evaluation point(s) for t.
    method : str, optional
        Method for computing integer derivatives: 'symbolic', 'bell', 'jax'.
    simplify : bool, optional
        Ignored for numeric; kept for interface consistency.
    complete : bool, optional
        If True (default), differentiate the complete MGF.
        If False, differentiate the incomplete MGF.
    return_log : bool, optional
        If True, return (log_abs, sign) instead of ordinary value.
    margin : float
        Offset from the asymptotes to avoid infinities.
    max_u : float
        Maximum absolute value of u after transformation (default 20).
    dps : int
        Number of decimal digits for mpmath (default 50).
    u : float or array-like, optional
        Truncation point(s) for incomplete MGF (used when complete=False).
        If array‑like, broadcast with t to form evaluation points (t, u).

    Returns
    -------
    float or tuple (log_abs, sign)
        If t and u are scalar, returns scalar or tuple.
        If either is array, returns array(s) with the broadcasted shape.
    """
    mp.dps = dps

    if order <= 0:
        raise ValueError("Fractional order must be positive.")

    # ---- Broadcast t and u to a common batch shape ----
    t_arr = np.asarray(t)
    if complete:
        if u is not None:
            raise ValueError("u must be None when complete=True")
        scalar_input = t_arr.ndim == 0
        if scalar_input:
            batch_shape = ()
            t_flat = np.array([float(t_arr)])
            u_flat = None
        else:
            batch_shape = t_arr.shape
            t_flat = t_arr.astype(float).ravel()
            u_flat = None
        n_points = t_flat.size
    else:
        if u is None:
            raise ValueError("u must be provided when complete=False")
        u_arr = np.asarray(u)
        t_broad, u_broad = np.broadcast_arrays(t_arr, u_arr)
        scalar_input = t_broad.ndim == 0
        batch_shape = t_broad.shape
        t_flat = t_broad.astype(float).ravel()
        u_flat = u_broad.astype(float).ravel()
        n_points = t_flat.size

    # ---- Pre-allocate results ----
    if return_log:
        log_abs_vals = np.zeros(n_points)
        sign_vals = np.ones(n_points, dtype=int)
    else:
        val_vals = np.zeros(n_points)

    # ---- Scalar helper for a single evaluation point (t_val, u_val) ----
    def _scalar_eval(t_val, u_val):
        # Integer order
        if order == int(order):
            log_abs, sign = mgfDerivative_integer(
                order=int(order),
                prior=prior,
                method=method,
                t=t_val,
                simplify=simplify,
                log=True,
                complete=complete,
                u=u_val
            )
            if return_log:
                return log_abs, sign
            else:
                if log_abs == -float('inf'):
                    return 0.0
                return sign * float(exp(mpf(log_abs)))

        n = int(mp.floor(order))
        gamma_val = mpf((n + 1) - order)

        def integrand_theta(theta):
            try:
                tan_theta = tan(theta)
                u_var = max_u * tan_theta
                z = exp(u_var)
                y = t_val - z
                log_abs, sign = mgfDerivative_integer(
                    order=n + 1,
                    prior=prior,
                    method=method,
                    t=float(y),
                    simplify=simplify,
                    log=True,
                    complete=complete,
                    u=u_val
                )
                if not math.isfinite(log_abs):
                    return mpf(0.0)
                log_jacobian = log(max_u) + log(1 + tan_theta * tan_theta)
                log_integrand = gamma_val * u_var + mpf(log_abs) + log_jacobian
                return exp(log_integrand) * sign
            except Exception:
                return mpf(0.0)

        a = -pi / 2 + margin
        b = pi / 2 - margin

        try:
            integral = quad(integrand_theta, (a, b), method='tanh-sinh')
        except Exception as e:
            print(f"mpmath tan‑transform integration failed for t={t_val}, u={u_val}: {e}")
            if return_log:
                return float('nan'), 1
            else:
                return float('nan')

        if return_log:
            if integral == 0:
                return -float('inf'), 1
            log_abs = float(log(abs(integral)) - log(gamma(gamma_val)))
            sign = 1 if integral > 0 else -1
            return log_abs, sign
        else:
            result = float((1.0 / gamma(gamma_val)) * integral)
            return result

    # ---- Loop over flattened evaluation points ----
    for idx in range(n_points):
        t_val = t_flat[idx]
        u_val = u_flat[idx] if u_flat is not None else None
        if return_log:
            log_abs, sign = _scalar_eval(t_val, u_val)
            log_abs_vals[idx] = log_abs
            sign_vals[idx] = sign
        else:
            val_vals[idx] = _scalar_eval(t_val, u_val)

    # ---- Reshape to broadcasted shape ----
    if return_log:
        log_abs_vals = log_abs_vals.reshape(batch_shape)
        sign_vals = sign_vals.reshape(batch_shape)
        if scalar_input:
            return float(log_abs_vals.item()), int(sign_vals.item())
        else:
            return log_abs_vals, sign_vals
    else:
        val_vals = val_vals.reshape(batch_shape)
        if scalar_input:
            return float(val_vals.item())
        else:
            return val_vals




def _mp_integer_derivative(prior, n_plus_1, complete):
    """Return a callable giving ``M^{(n+1)}`` at **arbitrary** precision.

    This is what makes ``dps`` mean something. The obvious implementation --
    calling ``mgfDerivative_integer`` per quadrature node -- returns
    ``(log_abs, sign)`` as NumPy ``float64`` and takes ``t`` as a Python float,
    so the mpmath backend was integrating a float-precision function at
    arbitrary precision: the quadrature exact to ``dps`` digits, of a function
    known only to 16. Measured, accuracy stopped improving with ``dps`` at
    around 1e-10 and bounced rather than converging.

    SymPy evaluates to any precision natively, so the derivative expression is
    built once and evaluated per node with ``evalf(dps)``. Substituting the
    prior's hyperparameters up front keeps that per-node work to one
    substitution and one evaluation.

    Returns ``None`` when no symbolic MGF is available, so the caller can fall
    back rather than fail -- the Bell and JAX backends are float by
    construction and cannot be lifted this way.
    """
    attr = "mgf_sym" if complete else "imgf_sym"
    if getattr(prior, attr, None) is None:
        return None

    expr = cached_diff(getattr(prior, attr), _t_sym, n_plus_1)

    params = getattr(prior, "params", None) or {}
    substitutions = {
        sym: params[sym.name] for sym in expr.free_symbols if sym.name in params
    }
    if substitutions:
        expr = expr.subs(substitutions)

    def evaluate(t_value, u_value, dps):
        point = {_t_sym: sp.Float(t_value, dps)}
        if u_value is not None:
            point[_u_sym] = sp.Float(u_value, dps)
        value = expr.subs(point).evalf(dps)
        if value.free_symbols:
            raise ValueError(
                "The prior's symbolic MGF still has free symbols after "
                f"substituting its parameters: {sorted(map(str, value.free_symbols))}."
            )
        return mpf(str(value))

    return evaluate


def _right_endpoint(integrand, gamma_val, tol, start=8.0, cap=709.0):
    """Find where the transformed integrand has decayed, by probing it.

    The left endpoint follows from the ``e^{gamma u}`` decay and can be written
    down. The right cannot: the integrand there is governed by how fast
    ``M^{(n+1)}(t - z)`` falls off, which is a property of the prior -- only
    polynomial in ``z`` for the priors here -- and is not exposed. So it is
    measured rather than assumed, by extending until the value at the endpoint
    is below ``tol`` relative to the largest value seen.

    This is a test of the *integrand's* decay. The rule it replaces compared
    consecutive estimates of the *integral*, which underestimates the remaining
    tail exactly when convergence is slow -- the failure this whole change is
    about.
    """
    u_max = start
    while u_max < cap:
        probes = [u_max * frac for frac in (0.25, 0.5, 0.75, 1.0)]
        values = [abs(float(integrand(mpf(p)))) for p in probes]
        peak = max(values)
        if peak == 0.0 or values[-1] <= tol * peak:
            return u_max
        u_max *= 2.0
    return cap


def fractionalDeriv_numeric_mpmath(
    order: float,
    prior: mitMGFprior,
    t: float | np.ndarray | list,
    method: str = "symbolic",
    complete: bool = True,
    simplify: bool = False,
    return_log: bool = False,
    initial_L: float = 10.0,
    max_L: float = 1e4,
    tol: float = 1e-8,
    use_tan: bool = False,
    dps: int = 50,
    u: float | np.ndarray | list | None = None
):
    """
    Compute the Liouville‑Caputo fractional derivative using mpmath.
    Supports tuple‑vectorisation: t and u are broadcast to a common shape.

    Parameters
    ----------
    order : float
        Fractional order (positive). If integer, returns ordinary derivative.
    prior : mitMGFprior
        Prior object providing the MGF.
    t : float or array-like
        Evaluation point(s) for t.
    method : str, optional
        'symbolic', 'jax', or 'bell' – method for computing the integer derivative.
    simplify : bool, optional
        Ignored for numeric; kept for interface consistency.
    complete : bool, optional
        If True (default), differentiate the complete MGF.
        If False, differentiate the incomplete MGF.
    return_log : bool, optional
        If True, return (log_abs, sign) instead of ordinary value.
    initial_L : float
        Starting half‑width for integration range (adaptive method only).
    max_L : float
        Maximum allowed half‑width.
    tol : float
        Relative tolerance for convergence (default 1e-8).
    use_tan : bool
        If True, directly use the tan‑transform method.
    dps : int
        Number of decimal digits for mpmath (default 50).
    u : float or array-like, optional
        Truncation point(s) for incomplete MGF (used when complete=False).
        If array‑like, broadcast with t to form evaluation points (t, u).

    Returns
    -------
    float or tuple (log_abs, sign)
        If t and u are scalar, returns scalar or tuple.
        If either is array, returns array(s) with the broadcasted shape.
    """
    mp.dps = dps

    if order <= 0:
        raise ValueError("Fractional order must be positive.")

    # ---- If use_tan=True, delegate to vectorized tan version ----
    if use_tan:
        # The tan version already handles broadcasting and tuple‑vectorisation.
        return fractionalDeriv_numeric_mpmath_tan(
            order=order, prior=prior, t=t, method=method,
            simplify=simplify, complete=complete,
            return_log=return_log, dps=dps, u=u
        )

    # ---- Broadcast t and u to a common batch shape ----
    t_arr = np.asarray(t)
    if complete:
        if u is not None:
            raise ValueError("u must be None when complete=True")
        scalar_input = t_arr.ndim == 0
        if scalar_input:
            batch_shape = ()
            t_flat = np.array([float(t_arr)])
            u_flat = None
        else:
            batch_shape = t_arr.shape
            t_flat = t_arr.astype(float).ravel()
            u_flat = None
        n_points = t_flat.size
    else:
        if u is None:
            raise ValueError("u must be provided when complete=False")
        u_arr = np.asarray(u)
        t_broad, u_broad = np.broadcast_arrays(t_arr, u_arr)
        scalar_input = t_broad.ndim == 0
        batch_shape = t_broad.shape
        t_flat = t_broad.astype(float).ravel()
        u_flat = u_broad.astype(float).ravel()
        n_points = t_flat.size

    # ---- Pre-allocate results ----
    if return_log:
        log_abs_vals = np.zeros(n_points)
        sign_vals = np.ones(n_points, dtype=int)
    else:
        val_vals = np.zeros(n_points)

    # ---- Scalar helper for a single evaluation point (t_val, u_val) ----
    def _scalar_eval(t_val, u_val):
        # ---- 1. Integer order ----
        if order == int(order):
            log_abs, sign = mgfDerivative_integer(
                order=int(order),
                prior=prior,
                method=method,
                t=t_val,
                simplify=simplify,
                log=True,
                complete=complete,
                u=u_val
            )
            if return_log:
                return log_abs, sign
            else:
                if log_abs == -float('inf'):
                    return 0.0
                return sign * float(exp(mpf(log_abs)))

        # ---- 2. Adaptive range method ----
        n = int(mp.floor(order))
        gamma_val = mpf((n + 1) - order)

        # `M^(n+1)` at full precision where the prior allows it. Falling back
        # to the float route rather than failing keeps the Bell and JAX integer
        # methods usable; they are float by construction, so `dps` cannot help
        # them and the fallback says so by warning once.
        mp_derivative = _mp_integer_derivative(prior, n + 1, complete)
        if mp_derivative is None:
            warnings.warn(
                f"No symbolic MGF for prior '{getattr(prior, 'name', '?')}', so the "
                "integrand is evaluated in double precision and `dps` cannot "
                "improve the result beyond about 1e-10. Supply `mgf_sym` to get "
                "the precision this backend advertises.",
                RuntimeWarning,
                stacklevel=2,
            )

        def integrand(u_var):
            u_var = mpf(u_var)
            z = exp(u_var)
            y = t_val - z

            if mp_derivative is not None:
                value = mp_derivative(y, u_val, mp.dps)
                if value == 0:
                    return mpf(0.0)
                return exp(gamma_val * u_var) * value

            log_abs, sign = mgfDerivative_integer(
                order=n + 1,
                prior=prior,
                method=method,
                t=float(y),
                simplify=simplify,
                complete=complete,
                log=True,
                u=u_val
            )
            if not math.isfinite(log_abs):
                return mpf(0.0)
            log_integrand = gamma_val * u_var + mpf(log_abs)
            return exp(log_integrand) * sign

        # ---- Derived range, replacing the symmetric doubling loop ---------
        #
        # This used to integrate over (-L, L) with L doubling from 10 until
        # consecutive estimates agreed. Two things were wrong with that, and
        # together they returned a confidently wrong number.
        #
        # The range was SYMMETRIC, but the two tails are nothing alike. The
        # transformed integrand behaves like e^{gamma*u} as u -> -infinity and
        # dies quickly to the right, so the left endpoint needs
        # log(tol)/gamma -- about -64 at gamma = 0.5 -- while the right needs
        # only a small positive u. Doubling both together drove L to 5120,
        # integrating over a range where the integrand is numerically zero
        # across almost all of its width, which is what defeated tanh-sinh.
        #
        # And on failing to converge it printed a warning and returned the last
        # iterate anyway. Measured at order 1.5, t = -1: dps <= 30 returned
        # +1.119329613307 against a closed-form -1.453832084236 -- wrong sign,
        # 177% wrong in magnitude -- and was SLOWER than the dps = 50 run that
        # got the right answer (8.2 s against 5.6 s). `dps` is reachable from
        # the constructor through DERIVATIVE_KWARGS, so an ordinary caller
        # could ask for less precision and receive nonsense.
        #
        # Deriving the endpoints removes the loop, so there is no stopping rule
        # left to be wrong and no non-convergence branch to return garbage from.
        # The truncation target is set by `dps`, not by `tol`. A caller asking
        # this backend for 50 digits is asking for 50 digits, and a range
        # derived from `tol` (default 1e-6) would cap the answer at about 1e-8
        # however high `dps` went -- correct, but silently no better than the
        # scipy backend, which defeats the purpose of using mpmath at all.
        range_tol = mpf(10) ** (-int(dps))
        u_min = log(range_tol) / mpf(gamma_val)
        u_max = mpf(_right_endpoint(integrand, gamma_val, float(range_tol)))

        try:
            # Two details that decide whether `dps` means anything.
            #
            # The result is NOT cast to float here. Doing so discarded every
            # digit past the 16th before the log was taken, which capped the
            # backend at double precision no matter how high `dps` went.
            #
            # And the interval is split at 0. tanh-sinh clusters its nodes near
            # the endpoints, which is right for endpoint singularities but wrong
            # here: this integrand's mass sits near u = 0 while the interval
            # runs from about -69 to 8, so almost all the nodes landed where the
            # integrand is numerically zero. Naming the interior point puts
            # nodes where the mass is.
            interior = [u_min, mpf(0), u_max] if u_min < 0 < u_max else [u_min, u_max]
            integral_valid = quad(integrand, interior, method="tanh-sinh")
        except Exception as exc:
            # A genuine quadrature failure is now an error rather than a print
            # followed by a plausible number.
            raise RuntimeError(
                f"mpmath quadrature failed for order={order}, t={t_val}, "
                f"u={u_val} over the derived range ({float(u_min):.3g}, "
                f"{float(u_max):.3g}): {exc}"
            ) from exc

        # Compute result using log-scale for stability
        if return_log:
            if abs(integral_valid) < mpf(10) ** (-300):
                return -float('inf'), 1
            log_abs = float(log(mpf(abs(integral_valid))) - log(gamma(gamma_val)))
            sign = 1 if integral_valid > 0 else -1
            return log_abs, sign
        else:
            result = float((1.0 / gamma(gamma_val)) * integral_valid)
            return result

    # ---- Loop over flattened evaluation points ----
    for idx in range(n_points):
        t_val = t_flat[idx]
        u_val = u_flat[idx] if u_flat is not None else None
        if return_log:
            log_abs, sign = _scalar_eval(t_val, u_val)
            log_abs_vals[idx] = log_abs
            sign_vals[idx] = sign
        else:
            val_vals[idx] = _scalar_eval(t_val, u_val)

    # ---- Reshape to broadcasted shape ----
    if return_log:
        log_abs_vals = log_abs_vals.reshape(batch_shape)
        sign_vals = sign_vals.reshape(batch_shape)
        if scalar_input:
            return float(log_abs_vals.item()), int(sign_vals.item())
        else:
            return log_abs_vals, sign_vals
    else:
        val_vals = val_vals.reshape(batch_shape)
        if scalar_input:
            return float(val_vals.item())
        else:
            return val_vals
