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
from scipy.integrate import quad, quad_vec
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
    u: float | np.ndarray | list | None = None
):
    """
    Compute fractional derivative using a scaled tan‑transform.
    Supports vectorized t and u via tuple‑vectorisation principle.

    The evaluation point is:
        - complete MGF: (t)
        - incomplete MGF: (t, u)
    If either t or u is array‑like, they are broadcast to a common shape and the
    computation is vectorised over that batch.

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
    u : float or array-like, optional
        Truncation point(s) for incomplete MGF (used when complete=False).
        If array‑like, broadcast with t to form evaluation points (t, u).

    Returns
    -------
    float or tuple (log_abs, sign)
        If t and u are scalar, returns scalar or tuple.
        If either is array, returns array(s) with the broadcasted shape.
    """
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
            result = mgfDerivative_integer(
                order=int(order),
                prior=prior,
                method=method,
                t=t_val,
                simplify=simplify,
                complete=complete,
                log=return_log,
                u=u_val
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
                    u=u_val
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
            print(f"Scaled tan‑transform integration failed for t={t_val}, u={u_val}: {e}")
            if return_log:
                return np.nan, 1
            else:
                return np.nan

        if return_log:
            if abs(integral) < 1e-300:
                return -float('inf'), 1
            log_abs = math.log(abs(integral)) - math.lgamma(gamma_val)
            sign = 1 if integral > 0 else -1
            return log_abs, sign
        else:
            factor = 1.0 / gamma_func(gamma_val)
            result = factor * integral
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

def fractionalDeriv_numeric_scipy_loop(
    order: float,
    prior: mitMGFprior,
    t_flat: np.ndarray,
    u_flat: np.ndarray | None,
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
):
    """
    Scalar‑loop version of fractional derivative (no fallback).
    Raises RuntimeError on failure.
    """
    if order <= 0:
        raise ValueError("Fractional order must be positive.")
    n_points = len(t_flat)

    if return_log:
        log_abs_vals = np.zeros(n_points)
        sign_vals = np.ones(n_points, dtype=int)
    else:
        val_vals = np.zeros(n_points)

    # ---- Scalar helper for a single evaluation point ----
    def _scalar_eval(t_val, u_val):
        if order == int(order):
            result = mgfDerivative_integer(
                order=int(order),
                prior=prior,
                method=method,
                t=t_val,
                simplify=simplify,
                complete=complete,
                log=return_log,
                u=u_val
            )
            return result

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
                u=u_val
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
                raise RuntimeError(f"quad failed at L={L}: {e}")

        if integral_valid is None:
            raise RuntimeError("No valid integral result.")

        # ---- Log‑stable return ----
        if return_log:
            if abs(integral_valid) < 1e-300:
                return -float('inf'), 1
            else:
                # log |D^α M| = log |integral| - log Γ(γ)
                log_abs = math.log(abs(integral_valid)) - math.lgamma(gamma_val)
                sign = 1 if integral_valid > 0 else -1
                return log_abs, sign
        else:
            factor = 1.0 / gamma_func(gamma_val)
            return factor * integral_valid

    # ---- Loop over points ----
    for idx in range(n_points):
        t_val = t_flat[idx]
        u_val = u_flat[idx] if u_flat is not None else None
        if return_log:
            log_abs, sign = _scalar_eval(t_val, u_val)
            log_abs_vals[idx] = log_abs
            sign_vals[idx] = sign
        else:
            val_vals[idx] = _scalar_eval(t_val, u_val)

    if return_log:
        return log_abs_vals, sign_vals
    else:
        return val_vals

def _batch_quad_vec_method(
    order: float,
    prior: mitMGFprior,
    t_flat: np.ndarray,
    u_flat: np.ndarray | None,
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
):
    """
    Batch quadrature using quad_vec with adaptive L expansion.
    Raises RuntimeError on failure.
    """
    if order <= 0:
        raise ValueError("Fractional order must be positive.")
    n_points = len(t_flat)

    # Handle integer order directly
    if order == int(order):
        raise ValueError("Integer order not supported in batch method; use mgfDerivative_integer directly.")

    n = math.floor(order)
    gamma_val = (n + 1) - order

    # ---- Initialise ----
    L = initial_L
    integral_prev = np.full(n_points, np.nan)
    converged = np.full(n_points, False)

    # ---- Vectorized integrand ----
    def integrand_vec(u_var):
        z = np.exp(u_var)
        y = t_flat - z                     # shape (n_points,)
        log_abs, sign = mgfDerivative_integer(
            order=n + 1,
            prior=prior,
            method=method,
            t=y,
            u=u_flat if u_flat is not None else None,
            complete=complete,
            log=True
        )
        # log_integrand = gamma_val * u_var + log_abs
        with np.errstate(over='ignore', invalid='ignore'):
            val = sign * np.exp(gamma_val * u_var + log_abs)
            val[log_abs == -np.inf] = 0.0
        # Mask converged points: integrand = 0
        val[converged] = 0.0
        return val

    # ---- Adaptive L loop ----
    while not np.all(converged):
        integral, err = quad_vec(
            integrand_vec, -L, L,
            epsabs=epsabs, epsrel=epsrel, limit=limit,
            norm='max'
        )
        if np.any(np.isnan(integral)):
            raise RuntimeError(f"quad_vec returned NaN at L={L}")

        if np.all(np.isnan(integral_prev)):
            integral_prev = integral.copy()
        else:
            diff = np.abs(integral - integral_prev)
            converged = diff < tol * np.maximum(1.0, np.abs(integral_prev))
            integral_prev = integral.copy()

        if np.all(converged):
            break

        L *= 2.0
        if L > max_L:
            break

    if not np.all(converged):
        raise RuntimeError(f"Batch method did not converge for {np.sum(~converged)} points.")

    # ---- Log‑stable return ----
    if return_log:
        abs_integral = np.abs(integral)
        log_abs = np.where(
            abs_integral > 1e-300,
            np.log(abs_integral) - math.lgamma(gamma_val),
            -np.inf
        )
        sign = np.sign(integral).astype(int)
        sign[abs_integral <= 1e-300] = 1
        return log_abs, sign
    else:
        factor = 1.0 / gamma_func(gamma_val)
        return integral * factor

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
    u: float | np.ndarray | list | None = None,
    use_loop: bool = False,
):
    """
    Compute the Liouville‑Caputo fractional derivative of the MGF.
    Supports tuple‑vectorisation: t and u are broadcast to a common shape.

    Fallback chain:
        1. If use_loop=True, directly use scalar loop.
        2. Otherwise, try batch method (quad_vec).
        3. If batch fails, fall back to scalar loop.
        4. If scalar loop fails, fall back to tan‑transform.
    """
    if order <= 0:
        raise ValueError("Fractional order must be positive.")

    # ---- If use_tan=True, delegate to vectorized tan (no fallback needed) ----
    if use_tan:
        return fractionalDeriv_numeric_scipy_tan(
            order=order, prior=prior, t=t, method=method,
            simplify=simplify, complete=complete,
            epsabs=epsabs, epsrel=epsrel, limit=limit,
            return_log=return_log, u=u
        )

    # ---- Broadcast t and u to common batch shape ----
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

    # ---- Integer order: handle directly via mgfDerivative_integer ----
    if order == int(order):
        # For integer order, we can call mgfDerivative_integer which already handles batch.
        # But we must pass the full t and u (broadcasted) correctly.
        # We'll pass t_arr and u_arr (original arrays) so it can handle broadcasting.
        return mgfDerivative_integer(
            order=int(order),
            prior=prior,
            method=method,
            t=t,
            u=u,
            complete=complete,
            log=return_log
        )

    # ---- If use_loop, directly call scalar loop ----
    if use_loop:
        result = fractionalDeriv_numeric_scipy_loop(
            order, prior, t_flat, u_flat, method, simplify, complete,
            epsabs, epsrel, limit, return_log, initial_L, max_L, tol
        )
        # reshape and return
        return _reshape_result(result, scalar_input, batch_shape, return_log)

    # ---- Try batch method ----
    try:
        result = _batch_quad_vec_method(
            order, prior, t_flat, u_flat, method, simplify, complete,
            epsabs, epsrel, limit, return_log, initial_L, max_L, tol
        )
        return _reshape_result(result, scalar_input, batch_shape, return_log)
    except Exception as e_batch:
        print(f"Batch method failed: {e_batch}. Falling back to scalar loop.")

        # ---- Fallback to scalar loop ----
        try:
            result = fractionalDeriv_numeric_scipy_loop(
                order, prior, t_flat, u_flat, method, simplify, complete,
                epsabs, epsrel, limit, return_log, initial_L, max_L, tol
            )
            return _reshape_result(result, scalar_input, batch_shape, return_log)
        except Exception as e_loop:
            print(f"Scalar loop failed: {e_loop}. Falling back to tan‑transform.")

            # ---- Final fallback: tan‑transform (vectorized) ----
            # We call the vectorized tan version with the original t and u arrays.
            return fractionalDeriv_numeric_scipy_tan(
                order=order, prior=prior, t=t, method=method,
                simplify=simplify, complete=complete,
                epsabs=epsabs, epsrel=epsrel, limit=limit,
                return_log=return_log, u=u
            )

def _reshape_result(result, scalar_input, batch_shape, return_log):
    """
    Helper to reshape and return scalar/array results.
    """
    if return_log:
        log_abs, sign = result
        log_abs = log_abs.reshape(batch_shape)
        sign = sign.reshape(batch_shape)
        if scalar_input:
            return float(log_abs.item()), int(sign.item())
        else:
            return log_abs, sign
    else:
        vals = result.reshape(batch_shape)
        if scalar_input:
            return float(vals.item())
        else:
            return vals

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