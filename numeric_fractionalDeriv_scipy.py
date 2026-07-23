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
from jumufraktiv.derivativeDispatch import mgfDerivative_integer
from jumufraktiv.mitMGFprior_class import mitMGFprior


def fractionalDeriv_numeric_scipy_tan(
    order: float,
    prior: mitMGFprior,
    t: float | np.ndarray | list,
    method: str = "symbolic",
    simplify: bool = False,
    complete: bool = True,
    epsabs: float = 1e-8,
    epsrel: float = 1e-8,
    limit: int = 100,
    return_log: bool = False,
    margin: float = 1e-10,
    max_u: float = 20.0,
    u: float = None
):
    """
    Compute fractional derivative using a scaled tan‑transform.
    Supports vectorized t (returns array of results).

    Parameters
    ----------
    order : float
        Fractional order (positive).
    prior : mitMGFprior
        Prior object providing the MGF.
    t : float or array-like
        Evaluation point(s).
    method : str, optional
        Method for computing integer derivatives: 'symbolic', 'bell', 'jax'.
    simplify : bool, optional
        Ignored for numeric; kept for interface consistency.
    complete : bool, optional
        If True (default), differentiate the complete MGF.
        If False, differentiate the incomplete MGF.
    epsabs, epsrel : float
        Tolerances for quad.
    limit : int
        Maximum number of subintervals.
    return_log : bool, optional
        If True, return (log_abs, sign) instead of ordinary value.
    margin : float
        Offset from the asymptotes to avoid infinities.
    max_u : float
        Maximum absolute value of u after transformation (default 20).
    u : float, optional
        Truncation point for incomplete MGF (used when complete=False). Default None.

    Returns
    -------
    float or tuple (log_abs, sign)
        If t is scalar, returns scalar or tuple.
        If t is array, returns array of same shape (or two arrays for log/sign).
    """
    if order <= 0:
        raise ValueError("Fractional order must be positive.")

    # ---- Detect input type ----
    t_arr = np.asarray(t)
    scalar_input = t_arr.ndim == 0
    if scalar_input:
        t_arr = np.array([t])
    batch = len(t_arr)

    # ---- Pre-allocate results ----
    if return_log:
        log_abs_vals = np.zeros(batch)
        sign_vals = np.ones(batch, dtype=int)
    else:
        val_vals = np.zeros(batch)

    # ---- Scalar helper ----
    def _scalar_eval(t_val):
        # Integer order
        if order == int(order):
            result = mgfDerivative_integer(
                order=int(order),
                prior=prior,
                method=method,
                t=t_val,
                simplify=simplify,
                complete=complete,
                log=return_log,
                u=u
            )
            return result

        n = math.floor(order)
        gamma_val = (n + 1) - order

        def integrand_theta(theta):
            try:
                tan_theta = math.tan(theta)
                u_var = max_u * tan_theta
                z = math.exp(u_var)
                y = t_val - z
                log_abs, sign = mgfDerivative_integer(
                    order=n + 1,
                    prior=prior,
                    method=method,
                    t=y,
                    simplify=simplify,
                    complete=complete,
                    log=True,
                    u=u
                )
                if log_abs == -float('inf'):
                    return 0.0
                log_jacobian = math.log(max_u) + math.log1p(tan_theta * tan_theta)
                log_integrand = gamma_val * u_var + log_abs + log_jacobian
                if log_integrand > 700:
                    return 0.0
                if log_integrand < -745:
                    return 0.0
                return sign * math.exp(log_integrand)
            except Exception:
                return 0.0

        a = -math.pi/2 + margin
        b = math.pi/2 - margin

        try:
            integral, err = quad(integrand_theta, a, b, epsabs=epsabs, epsrel=epsrel, limit=limit)
        except Exception as e:
            print(f"Scaled tan‑transform integration failed for t={t_val}: {e}")
            if return_log:
                return np.nan, 1
            else:
                return np.nan

        # ---- Return result with log-scale stability ----
        if return_log:
            if abs(integral) < 1e-300:  # check integral, not the scaled result
                return -float('inf'), 1
            # log |D^α M| = log |integral| - log Γ(γ)
            log_abs = math.log(abs(integral)) - math.lgamma(gamma_val)
            sign = 1 if integral > 0 else -1
            return log_abs, sign
        else:
            factor = 1.0 / gamma_func(gamma_val)
            result = factor * integral
            return result

    # ---- Loop over t values ----
    for idx, t_val in enumerate(t_arr):
        if return_log:
            log_abs, sign = _scalar_eval(t_val)
            log_abs_vals[idx] = log_abs
            sign_vals[idx] = sign
        else:
            val_vals[idx] = _scalar_eval(t_val)

    # ---- Return ----
    if scalar_input:
        if return_log:
            return log_abs_vals[0], sign_vals[0]
        else:
            return val_vals[0]
    else:
        if return_log:
            return log_abs_vals, sign_vals
        else:
            return val_vals


def fractionalDeriv_numeric_scipy(
    order: float,
    prior: mitMGFprior,
    t: float | np.ndarray | list,
    method: str = "symbolic",
    simplify: bool = False,
    complete: bool = True,
    epsabs: float = 1e-8,
    epsrel: float = 1e-8,
    limit: int = 100,
    return_log: bool = False,
    initial_L: float = 10.0,
    max_L: float = 1e4,
    tol: float = 1e-6,
    use_tan: bool = False,
    u: float = None
):
    """
    Compute the Liouville‑Caputo fractional derivative of the MGF.
    Supports vectorized t (returns array of results).

    Parameters
    ----------
    order : float
        Fractional order (positive). If integer, returns ordinary derivative.
    prior : mitMGFprior
        Prior object providing the MGF.
    t : float or array-like
        Evaluation point(s).
    method : str, optional
        'symbolic', 'jax', or 'bell' – method for computing the integer derivative.
    simplify : bool, optional
        Ignored for numeric; kept for interface consistency.
    complete : bool, optional
        If True (default), differentiate the complete MGF.
        If False, differentiate the incomplete MGF.
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
    u : float, optional
        Truncation point for incomplete MGF (used when complete=False). Default None.

    Returns
    -------
    float or tuple (log_abs, sign)
        If t is scalar, returns scalar or tuple.
        If t is array, returns array of same shape (or two arrays for log/sign).
    """
    if order <= 0:
        raise ValueError("Fractional order must be positive.")

    # ---- Detect input type ----
    t_arr = np.asarray(t)
    scalar_input = t_arr.ndim == 0
    if scalar_input:
        t_arr = np.array([t])
    batch = len(t_arr)

    # ---- Pre-allocate results ----
    if return_log:
        log_abs_vals = np.zeros(batch)
        sign_vals = np.ones(batch, dtype=int)
    else:
        val_vals = np.zeros(batch)

    # ---- Scalar helper (copied from original logic) ----
    def _scalar_eval(t_val):
        # ---- 1. Handle integer order ----
        if order == int(order):
            result = mgfDerivative_integer(
                order=int(order),
                prior=prior,
                method=method,
                t=t_val,
                simplify=simplify,
                complete=complete,
                log=return_log,
                u=u
            )
            return result

        # ---- 2. If use_tan=True, directly call tan version ----
        if use_tan:
            return fractionalDeriv_numeric_scipy_tan(
                order, prior, t_val, method, simplify=simplify,
                epsabs=epsabs, epsrel=epsrel, limit=limit, return_log=return_log,
                complete=complete, u=u
            )

        # ---- 3. Adaptive range method ----
        n = math.floor(order)
        gamma_val = (n + 1) - order

        def integrand(u_var):
            z = math.exp(u_var)
            y = t_val - z
            log_abs, sign = mgfDerivative_integer(
                order=n + 1,
                prior=prior,
                method=method,
                t=y,
                simplify=simplify,
                complete=complete,
                log=True,
                u=u
            )
            if log_abs == -float('inf'):
                return 0.0
            log_integrand = gamma_val * u_var + log_abs
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
                print(f"Adaptive integration failed at L={L} for t={t_val}: {e}")
                if integral_valid is not None:
                    print(f"  Using last valid result from L={L/2}.")
                    break
                else:
                    print("  No valid adaptive result; falling back to tan‑transform...")
                    return fractionalDeriv_numeric_scipy_tan(
                        order, prior, t_val, method, simplify,
                        epsabs=epsabs, epsrel=epsrel, limit=limit,
                        return_log=return_log, complete=complete, u=u
                    )

        # If we reached max_L without convergence
        if L > max_L and integral_valid is not None:
            print(f"Warning: Adaptive integration did not converge before max_L={max_L}. Using last result.")
        elif L > max_L and integral_valid is None:
            print("Adaptive method failed to produce a result; falling back to tan‑transform...")
            return fractionalDeriv_numeric_scipy_tan(
                order, prior, t_val, method, simplify,
                epsabs=epsabs, epsrel=epsrel, limit=limit,
                return_log=return_log, complete=complete, u=u
            )

        result = factor * integral_valid

        if return_log:
            if abs(result) < 1e-300:
                return -float('inf'), 1
            else:
                return math.log(abs(result)), 1 if result > 0 else -1
        else:
            return result

    # ---- Loop over t values ----
    for idx, t_val in enumerate(t_arr):
        if return_log:
            log_abs, sign = _scalar_eval(t_val)
            log_abs_vals[idx] = log_abs
            sign_vals[idx] = sign
        else:
            val_vals[idx] = _scalar_eval(t_val)

    # ---- Return ----
    if scalar_input:
        if return_log:
            return log_abs_vals[0], sign_vals[0]
        else:
            return val_vals[0]
    else:
        if return_log:
            return log_abs_vals, sign_vals
        else:
            return val_vals

# ===== Example usage =====
if __name__ == "__main__":
    import math
    import numpy as np
    import jumufraktiv.MGFdictionary  # ensures priors are registered
    from jumufraktiv.mitMGFprior_class import mitMGFprior

    # Build Gamma prior
    gamma_prior = mitMGFprior.from_registry(
        "gamma",
        params={"alpha": 2.0, "beta": 3.0}
    )

    # ---- Scalar test (original) ----
    t_val = -1.0
    frac_order = 1.99

    print("="*60)
    print("Scalar t test")
    print("="*60)
    print(f"  order={frac_order}, t={t_val}, alpha=2, beta=3")
    print("  Using default adaptive method (with fallback to tan)...")
    result_adaptive = fractionalDeriv_numeric_scipy(
        order=frac_order,
        prior=gamma_prior,
        t=t_val,
        method='symbolic',
        return_log=False
    )
    print(f"  Adaptive result: {result_adaptive:.6e}")

    print("\n  Using explicit tan‑transform method...")
    result_tan = fractionalDeriv_numeric_scipy(
        order=frac_order,
        prior=gamma_prior,
        t=t_val,
        method='symbolic',
        return_log=False,
        use_tan=True
    )
    print(f"  Tan‑transform result: {result_tan:.6e}")

    # Compare with ordinary 2nd derivative
    log_abs2, sign2 = mgfDerivative_integer(
        order=2,
        prior=gamma_prior,
        method='symbolic',
        t=t_val,
        simplify=False,
        log=True,
        complete=True
    )
    deriv2 = sign2 * math.exp(log_abs2)
    print(f"\n  Ordinary 2nd derivative at t={t_val}: {deriv2:.6e}")
    print(f"  Relative diff (adaptive vs 2nd): {abs(result_adaptive - deriv2) / abs(deriv2):.2e}")
    print(f"  Relative diff (tan vs 2nd):      {abs(result_tan - deriv2) / abs(deriv2):.2e}")
    print(f"  Relative diff (adaptive vs tan): {abs(result_adaptive - result_tan) / abs(result_tan):.2e}")

    # ---- Vectorized test (new) ----
    print("\n" + "="*60)
    print("Vectorized t test (multiple evaluation points)")
    print("="*60)

    t_vals = np.linspace(-200.4, -0.5, 5)   # 5 points
    print(f"  t values: {t_vals}")
    print(f"  order={frac_order}, alpha=2, beta=3")

    # Adaptive method
    results_adaptive = fractionalDeriv_numeric_scipy(
        order=frac_order,
        prior=gamma_prior,
        t=t_vals,
        method='symbolic',
        return_log=False
    )
    print("\n  Adaptive results:")
    for t_val, res in zip(t_vals, results_adaptive):
        print(f"    t={t_val:.2f}: {res:.6e}")

    # Tan‑transform method
    results_tan = fractionalDeriv_numeric_scipy(
        order=frac_order,
        prior=gamma_prior,
        t=t_vals,
        method='symbolic',
        return_log=False,
        use_tan=True
    )
    print("\n  Tan‑transform results:")
    for t_val, res in zip(t_vals, results_tan):
        print(f"    t={t_val:.2f}: {res:.6e}")

    # Compare adaptive vs tan for each point
    print("\n  Differences (adaptive - tan):")
    for t_val, res_adapt, res_t in zip(t_vals, results_adaptive, results_tan):
        print(f"    t={t_val:.2f}: {abs(res_adapt - res_t):.2e}")