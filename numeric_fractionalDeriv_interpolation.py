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
    t: float,
    d_vec: tuple = (0.8, 0.9, 0.95),
    return_log: bool = True,
    integer_method: str = "symbolic",
    **kwargs
):
    """
    Compute fractional derivative using cubic interpolation for orders
    approaching an integer from below.

    The user supplies `d_vec` as complements of deviations. For example,
    d_vec = (0.8, 0.9, 0.95) means the actual deviations are (0.2, 0.1, 0.05).
    The interpolation points are n - dev1, n - dev2, n - dev3, n, where
    dev_i = 1 - d_i.

    The interpolation is used for orders within (n - min_dev, n),
    where min_dev = min(1 - d_i).

    Parameters
    ----------
    order : float
        Target fractional order (non‑integer, typically just below an integer).
    prior : mitMGFprior
        Prior object providing the MGF and its functions.
    t : float
        Evaluation point.
    d_vec : tuple, optional
        Three complements of deviations (default (0.8, 0.9, 0.95)).
        Actual deviations are 1 - d_i.
    return_log : bool, optional
        If True, return (log_abs, sign); else return ordinary value.
    integer_method : str, optional
        Method for computing integer derivatives inside fractionalDeriv_numeric_scipy.
        Must be one of 'symbolic', 'bell', 'jax'. Default 'symbolic'.
    **kwargs : additional arguments passed to fractionalDeriv_numeric_scipy.
        e.g., epsrel, use_tan, epsabs, limit, etc.

    Returns
    -------
    float or tuple (log_abs, sign)
        The interpolated derivative value (or log‑absolute and sign).
    """
    # ---- Convert d_vec from complements to actual deviations ----
    if len(d_vec) != 3:
        raise ValueError("d_vec must have exactly 3 elements.")
    actual_dev = tuple(1.0 - d for d in d_vec)
    if any(dev <= 0 for dev in actual_dev):
        raise ValueError("All elements of d_vec must be < 1 (to get positive deviations).")

    # Determine the integer n such that order < n
    if order == int(order):
        # Exact integer: compute directly
        return fractionalDeriv_numeric_scipy(
            order=order,
            prior=prior,
            t=t,
            return_log=return_log,
            method=integer_method,
            **kwargs
        )

    n = int(np.ceil(order))
    if n <= 0:
        raise ValueError(f"Integer n = {n} must be positive.")
    # Ensure order > n - max(actual_dev) so that interpolation is valid
    if order <= n - max(actual_dev):
        raise ValueError(f"order {order} is not within (n - max(actual_dev), n). "
                         f"Consider using direct integration.")

    # ---- Compute values at interpolation points ----
    orders_compute = [n - dev for dev in actual_dev] + [n]
    values = []
    for alpha in orders_compute:
        log_abs, sign = fractionalDeriv_numeric_scipy(
            order=alpha,
            prior=prior,
            t=t,
            return_log=True,
            method=integer_method,
            **kwargs
        )
        values.append((log_abs, sign))

    # ---- Sort by order (ascending) ----
    sorted_pairs = sorted(zip(orders_compute, values), key=lambda x: x[0])
    x_vals = np.array([p[0] for p in sorted_pairs])
    y_vals = np.array([p[1][0] for p in sorted_pairs])  # log_abs
    sign_vals = [p[1][1] for p in sorted_pairs]
    sign_final = sign_vals[-1]   # sign at the integer

    # ---- Cubic interpolation of log_abs ----
    spline = CubicSpline(x_vals, y_vals, bc_type='natural')
    log_abs_interp = float(spline(order))

    # ---- Return ----
    if return_log:
        return log_abs_interp, sign_final
    else:
        if log_abs_interp == -float('inf'):
            return 0.0
        return sign_final * np.exp(log_abs_interp)


# ===== Example usage =====
if __name__ == "__main__":
    import math
    import jumufraktiv.MGFdictionary  # ensures priors are registered
    from jumufraktiv.mitMGFprior_class import mitMGFprior

    # Build a Gamma prior (Exponential(0.9) is Gamma(1, 0.9))
    gamma_prior = mitMGFprior.from_registry(
        "gamma",
        params={"alpha": 1.0, "beta": 0.9}
    )

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