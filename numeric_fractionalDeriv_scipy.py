"""
numeric_fractionalDeriv_scipy.py

Numerical computation of Liouville‑Caputo fractional derivatives of MGFs
using scipy.integrate.quad with the substitution z = e^u.

The main function fractionalDeriv_numeric_scipy() uses an adaptive range expansion
(default). If that fails or if use_tan=True, it uses the tan‑transform method
(fractionalDeriv_numeric_scipy_tan) which maps (-∞,∞) to (-π/2, π/2).

The formula computed is:
    D^α_{(-∞)+} M(t) = 1/Γ(γ) ∫_{-∞}^{∞} e^{γ u} M^{(n+1)}(t - e^{u}) du,
where n = floor(α), γ = n+1-α.
"""

import math
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma as gamma_func
from derivativeDispatch import mgfDerivative_integer


def fractionalDeriv_numeric_scipy_tan(
    order: float,
    prior: str,
    params: dict,
    t: float,
    method: str = "symbolic",
    simplify: bool = False,
    epsabs: float = 1e-8,
    epsrel: float = 1e-8,
    limit: int = 100,
    return_log: bool = False,
    margin: float = 1e-10,
    max_u: float = 20.0
):
    """
    Compute fractional derivative using a scaled tan‑transform:
        u = max_u * tan(theta)
    with theta in (-pi/2, pi/2). The integration limits are fixed.

    Parameters
    ----------
    max_u : float
        Maximum absolute value of u after transformation (default 20).
    margin : float
        Offset from the asymptotes to avoid infinities.
    """
    if order <= 0:
        raise ValueError("Fractional order must be positive.")

    # Integer order
    if order == int(order):
        log_abs, sign = mgfDerivative_integer(
            order=int(order),
            prior=prior,
            method=method,
            t=t,
            params=params,
            simplify=simplify,
            log=True
        )
        if return_log:
            return log_abs, sign
        else:
            if log_abs == -float('inf'):
                return 0.0
            return sign * math.exp(log_abs)

    n = math.floor(order)
    gamma_val = (n + 1) - order

    def integrand_theta(theta):
        # Compute u = max_u * tan(theta)
        try:
            tan_theta = math.tan(theta)
            u = max_u * tan_theta
            z = math.exp(u)
            y = t - z
            # Get derivative in log scale
            log_abs, sign = mgfDerivative_integer(
                order=n + 1,
                prior=prior,
                method=method,
                t=y,
                params=params,
                simplify=simplify,
                log=True
            )
            if log_abs == -float('inf'):
                return 0.0
            # log Jacobian: du/dtheta = max_u * (1 + tan^2)
            log_jacobian = math.log(max_u) + math.log1p(tan_theta * tan_theta)
            log_integrand = gamma_val * u + log_abs + log_jacobian
            if log_integrand > 700:
                return 0.0
            if log_integrand < -745:
                return 0.0
            return sign * math.exp(log_integrand)
        except Exception:
            # If anything fails (e.g., math range error), treat as zero
            return 0.0

    a = -math.pi/2 + margin
    b = math.pi/2 - margin

    try:
        integral, err = quad(integrand_theta, a, b, epsabs=epsabs, epsrel=epsrel, limit=limit)
    except Exception as e:
        print(f"Scaled tan‑transform integration failed: {e}")
        if return_log:
            return np.nan, 1
        else:
            return np.nan

    factor = 1.0 / gamma_func(gamma_val)
    result = factor * integral

    if return_log:
        if abs(result) < 1e-300:
            return -float('inf'), 1
        else:
            return math.log(abs(result)), 1 if result > 0 else -1
    else:
        return result


def fractionalDeriv_numeric_scipy(
    order: float,
    prior: str,
    params: dict,
    t: float,
    method: str = "symbolic",
    simplify: bool = False,
    epsabs: float = 1e-8,
    epsrel: float = 1e-8,
    limit: int = 100,
    return_log: bool = False,
    initial_L: float = 10.0,
    max_L: float = 1e4,
    tol: float = 1e-6,
    use_tan: bool = False
):
    """
    Compute the Liouville‑Caputo fractional derivative of the MGF.

    Parameters
    ----------
    order : float
        Fractional order (positive). If integer, returns ordinary derivative.
    prior : str
        'gamma' or 'pareto'.
    params : dict
        Distribution parameters (must be numeric).
    t : float
        Evaluation point (must be within MGF domain).
    method : str, optional
        'symbolic', 'jax', or 'bell' – method for computing the integer derivative.
    simplify : bool, optional
        Ignored for numeric; kept for interface consistency.
    epsabs, epsrel : float
        Tolerances for quad.
    limit : int
        Maximum number of subintervals.
    return_log : bool, optional
        If True, return (log_abs, sign) instead of ordinary value.
    initial_L : float
        Starting half‑width for integration range (adaptive method only).
    max_L : float
        Maximum allowed half‑width.
    tol : float
        Relative tolerance for stopping when integral stabilises.
    use_tan : bool
        If True, directly use the tan‑transform method.

    Returns
    -------
    float or tuple (log_abs, sign)
    """
    if order <= 0:
        raise ValueError("Fractional order must be positive.")

    # ---- 1. Handle integer order (common to both methods) ----
    if order == int(order):
        log_abs, sign = mgfDerivative_integer(
            order=int(order),
            prior=prior,
            method=method,
            t=t,
            params=params,
            simplify=simplify,
            log=True
        )
        if return_log:
            return log_abs, sign
        else:
            if log_abs == -float('inf'):
                return 0.0
            return sign * math.exp(log_abs)

    # ---- 2. If use_tan=True, directly call tan version ----
    if use_tan:
        return fractionalDeriv_numeric_scipy_tan(
            order, prior, params, t, method, simplify,
            epsabs, epsrel, limit, return_log
        )

    # ---- 3. Adaptive range method ----
    n = math.floor(order)
    gamma_val = (n + 1) - order

    def integrand(u):
        z = math.exp(u)
        y = t - z
        log_abs, sign = mgfDerivative_integer(
            order=n + 1,
            prior=prior,
            method=method,
            t=y,
            params=params,
            simplify=simplify,
            log=True
        )
        if log_abs == -float('inf'):
            return 0.0
        log_integrand = gamma_val * u + log_abs
        if log_integrand > 700:
            return 0.0
        if log_integrand < -745:
            return 0.0
        return sign * math.exp(log_integrand)

    L = initial_L
    integral_valid = None
    prev_integral = None
    factor = 1.0 / gamma_func(gamma_val)

    while L <= max_L:
        try:
            integral, err = quad(integrand, -L, L, epsabs=epsabs, epsrel=epsrel, limit=limit)
            integral_valid = integral
            if prev_integral is not None:
                if abs(integral - prev_integral) < tol * max(1.0, abs(prev_integral)):
                    break
            prev_integral = integral
            L *= 2
        except Exception as e:
            print(f"Adaptive integration failed at L={L}: {e}")
            # If we have a valid result, use it; otherwise fall back to tan
            if integral_valid is not None:
                print(f"  Using last valid result from L={L/2}.")
                break
            else:
                print("  No valid adaptive result; falling back to tan‑transform...")
                return fractionalDeriv_numeric_scipy_tan(
                    order, prior, params, t, method, simplify,
                    epsabs, epsrel, limit, return_log
                )

    # If we reached max_L without convergence
    if L > max_L and integral_valid is not None:
        print(f"Warning: Adaptive integration did not converge before max_L={max_L}. Using last result.")
    elif L > max_L and integral_valid is None:
        # No valid result at all – fall back to tan
        print("Adaptive method failed to produce a result; falling back to tan‑transform...")
        return fractionalDeriv_numeric_scipy_tan(
            order, prior, params, t, method, simplify,
            epsabs, epsrel, limit, return_log
        )

    result = factor * integral_valid

    if return_log:
        if abs(result) < 1e-300:
            return -float('inf'), 1
        else:
            return math.log(abs(result)), 1 if result > 0 else -1
    else:
        return result


# ===== Example usage =====
if __name__ == "__main__":
    gamma_params = {'alpha': 2.0, 'beta': 3.0}
    t_val = -1.0                     # changed from +1.0
    frac_order = 1.99                # close to 2

    print("Testing fractional derivative of Gamma MGF")
    print(f"  order={frac_order}, t={t_val}, alpha=2, beta=3")
    print("  Using default adaptive method (with fallback to tan)...")
    result_adaptive = fractionalDeriv_numeric_scipy(
        order=frac_order,
        prior='gamma',
        params=gamma_params,
        t=t_val,
        method='symbolic',
        return_log=False
    )
    print(f"  Adaptive result: {result_adaptive:.6e}")

    print("\n  Using explicit tan‑transform method...")
    result_tan = fractionalDeriv_numeric_scipy(
        order=frac_order,
        prior='gamma',
        params=gamma_params,
        t=t_val,
        method='symbolic',
        return_log=False,
        use_tan=True
    )
    print(f"  Tan‑transform result: {result_tan:.6e}")

    # Compare with ordinary 2nd derivative
    log_abs2, sign2 = mgfDerivative_integer(
        order=2,
        prior='gamma',
        method='symbolic',
        t=t_val,
        params=gamma_params,
        log=True
    )
    deriv2 = sign2 * math.exp(log_abs2)
    print(f"\n  Ordinary 2nd derivative at t={t_val}: {deriv2:.6e}")
    print(f"  Relative diff (adaptive vs 2nd): {abs(result_adaptive - deriv2) / abs(deriv2):.2e}")
    print(f"  Relative diff (tan vs 2nd):      {abs(result_tan - deriv2) / abs(deriv2):.2e}")
    print(f"  Relative diff (adaptive vs tan): {abs(result_adaptive - result_tan) / abs(result_tan):.2e}")