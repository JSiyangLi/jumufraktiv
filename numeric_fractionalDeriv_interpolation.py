"""
numeric_fractionalDeriv_interpolation.py

Cubic interpolation of fractional derivatives for orders approaching an integer from below.

Uses fractionalDeriv_numeric_scipy() to compute values at a set of orders
and interpolates for target orders near the integer.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from jumufraktiv.numeric_fractionalDeriv_scipy import fractionalDeriv_numeric_scipy
from jumufraktiv.mitMGFprior_class import mitMGFprior


def fractionalDeriv_interpolated(
    order: float,
    prior: mitMGFprior,
    t: float | np.ndarray | list,
    d_vec: tuple = (0.8, 0.9, 0.95),
    return_log: bool = True,
    complete: bool = True,
    integer_method: str = "symbolic",
    u: float | np.ndarray | list | None = None,
    **kwargs
):
    """
    Compute fractional derivative using cubic interpolation for orders
    approaching an integer from below. Supports tuple‑vectorisation.

    The evaluation point is:
        - complete MGF: (t)
        - incomplete MGF: (t, u)
    If either t or u is array‑like, they are broadcast to a common shape and the
    computation is vectorised over that batch.

    Parameters
    ----------
    order : float
        Target fractional order (non‑integer, typically just below an integer).
    prior : mitMGFprior
        Prior object providing the MGF and its functions.
    t : float or array-like
        Evaluation point(s) for t.
    d_vec : tuple, optional
        Three complements of deviations (default (0.8, 0.9, 0.95)).
        Actual deviations are 1 - d_i.
    return_log : bool, optional
        If True, return (log_abs, sign); else return ordinary value.
    integer_method : str, optional
        Method for computing integer derivatives inside fractionalDeriv_numeric_scipy.
        Must be one of 'symbolic', 'bell', 'jax'. Default 'symbolic'.
    complete : bool, optional
        If True (default), differentiate the complete MGF.
        If False, differentiate the incomplete MGF.
    u : float or array-like, optional
        Truncation point(s) for incomplete MGF (used when complete=False).
        If array‑like, broadcast with t to form evaluation points (t, u).
    **kwargs : additional arguments passed to fractionalDeriv_numeric_scipy.
        e.g., epsrel, use_tan, epsabs, limit, etc.

    Returns
    -------
    float or tuple (log_abs, sign)
        If t and u are scalar, returns scalar or tuple.
        If either is array, returns array(s) with the broadcasted shape.
    """
    # ---- Validate d_vec ----
    if len(d_vec) != 3:
        raise ValueError("d_vec must have exactly 3 elements.")
    actual_dev = tuple(1.0 - d for d in d_vec)
    if any(dev <= 0 for dev in actual_dev):
        raise ValueError("All elements of d_vec must be < 1 (to get positive deviations).")

    # ---- Convert t to array ----
    t_arr = np.asarray(t)
    scalar_input = t_arr.ndim == 0
    if scalar_input:
        t_arr = np.array([t])
    batch = len(t_arr)

    # ---- Determine interpolation orders ----
    if order == int(order):
        # Exact integer: compute directly for all t (fractionalDeriv_numeric_scipy handles broadcasting)
        return fractionalDeriv_numeric_scipy(
            order=order,
            prior=prior,
            t=t,
            return_log=return_log,
            method=integer_method,
            complete=complete,
            u=u,
            **kwargs
        )

    n = int(np.ceil(order))
    if n <= 0:
        raise ValueError(f"Integer n = {n} must be positive.")
    min_dev = min(actual_dev)
    if order <= n - min_dev:
        raise ValueError(f"order {order} is not within (n - min_dev, n). "
                         f"Consider using direct integration.")

    # ---- Fixed interpolation x‑values ----
    orders_compute = [n - dev for dev in actual_dev] + [n]
    x_vals = np.array(orders_compute)   # length 4

    # ---- Compute log_abs and sign for all interpolation orders ----
    # fractionalDeriv_numeric_scipy now handles tuple‑vectorisation, so we pass t_arr and u as is.
    log_abs_matrix = np.zeros((len(orders_compute), batch))
    sign_matrix = np.zeros((len(orders_compute), batch), dtype=int)

    for idx, alpha in enumerate(orders_compute):
        log_abs_alpha, sign_alpha = fractionalDeriv_numeric_scipy(
            order=alpha,
            prior=prior,
            t=t_arr,
            return_log=True,
            method=integer_method,
            complete=complete,
            u=u,
            **kwargs
        )
        log_abs_matrix[idx, :] = log_abs_alpha
        sign_matrix[idx, :] = sign_alpha

    # ---- For each point, interpolate log_abs at the target order ----
    from scipy.interpolate import CubicSpline
    log_abs_interp = np.zeros(batch)
    for i in range(batch):
        y_vals = log_abs_matrix[:, i]
        spline = CubicSpline(x_vals, y_vals, bc_type='natural')
        log_abs_interp[i] = float(spline(order))

    # ---- Sign: take sign at the highest interpolation order (n) ----
    sign_final = sign_matrix[-1, :]   # sign at order n

    # ---- Reshape to original shape if needed ----
    # If t was originally scalar, we already have scalar; else we preserve shape.
    if scalar_input:
        return (log_abs_interp[0], int(sign_final[0])) if return_log else float(result[0])
    else:
        # For array input, we return arrays with the same shape as t_arr (since u broadcast with t).
        # Note: if t_arr was flattened, we should reshape to original shape.
        original_shape = np.asarray(t).shape
        log_abs_interp = log_abs_interp.reshape(original_shape)
        sign_final = sign_final.reshape(original_shape)
        if return_log:
            return log_abs_interp, sign_final
        else:
            result = sign_final * np.exp(log_abs_interp)
            result[np.isneginf(log_abs_interp)] = 0.0
            return result


# ===== Example usage =====
if __name__ == "__main__":
    import math
    import numpy as np
    import jumufraktiv.MGFdictionary  # ensures priors are registered
    from jumufraktiv.mitMGFprior_class import mitMGFprior

    # ---- Build Gamma prior (Exponential(0.9) is Gamma(1, 0.9)) ----
    gamma_prior = mitMGFprior.from_registry(
        "gamma",
        params={"alpha": 1.0, "beta": 0.9}
    )

    print("=" * 60)
    print("Scalar t test")
    print("=" * 60)

    t_val = -1.0
    order_target = 1.999

    # Interpolated value
    log_abs_interp, sign_interp = fractionalDeriv_interpolated(
        order=order_target,
        prior=gamma_prior,
        t=t_val,
        d_vec=(0.8, 0.9, 0.95),
        integer_method='symbolic',
        epsrel=1e-10
    )
    print(f"Interpolated log|deriv| = {log_abs_interp:.6f}, sign = {sign_interp}")

    # Analytical formula for Exponential prior: D^α M(t) = λ * Γ(α+1) * (λ - t)^(-α-1)
    lambda_exp = gamma_prior.params['beta']
    log_analytic = math.log(lambda_exp) + math.lgamma(order_target + 1) - (order_target + 1) * math.log(lambda_exp - t_val)
    print(f"Analytic log|deriv|      = {log_analytic:.6f}")
    print(f"Difference (interp - analytic) = {log_abs_interp - log_analytic:.2e}")

    # ---- Vectorized t test ----
    print("\n" + "=" * 60)
    print("Vectorized t test (multiple evaluation points)")
    print("=" * 60)

    t_vals = np.linspace(-2.0, -0.5, 5)   # 5 points
    print(f"  t values: {t_vals}")

    # Interpolated values for all t (vectorized)
    log_abs_vec, sign_vec = fractionalDeriv_interpolated(
        order=order_target,
        prior=gamma_prior,
        t=t_vals,
        d_vec=(0.8, 0.9, 0.95),
        integer_method='symbolic',
        epsrel=1e-10
    )

    # Analytical values
    log_analytic_vec = np.log(lambda_exp) + math.lgamma(order_target + 1) - (order_target + 1) * np.log(lambda_exp - t_vals)

    print("\n  Results:")
    print(f"    {'t':>8} {'interp log':>14} {'analytic log':>14} {'diff':>14}")
    for t_val, log_int, log_ana in zip(t_vals, log_abs_vec, log_analytic_vec):
        print(f"    {t_val:8.3f} {log_int:14.6f} {log_ana:14.6f} {log_int - log_ana:14.2e}")