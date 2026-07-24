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
import numpy as np
from mpmath import mp, pi, exp, log, tan, gamma, quad, mpf
from jumufraktiv.derivativeDispatch import mgfDerivative_integer
from jumufraktiv.mitMGFprior_class import mitMGFprior


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

        def integrand(u_var):
            u_var = mpf(u_var)
            z = exp(u_var)
            y = t_val - z
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

        L = initial_L
        integral_valid = None
        prev_integral = None
        final_L = None

        while L <= max_L:
            try:
                integral = quad(integrand, (-L, L), method='tanh-sinh')
                integral_valid = float(integral)
                if prev_integral is not None:
                    if abs(integral_valid - prev_integral) < tol * max(1.0, abs(prev_integral)):
                        final_L = L
                        break
                prev_integral = integral_valid
                L *= 2
            except Exception as e:
                print(f"mpmath adaptive integration failed at L={L} for t={t_val}, u={u_val}: {e}")
                if integral_valid is not None:
                    final_L = L / 2
                    print(f"  Using last valid result from L={final_L}.")
                    break
                else:
                    print("  No valid adaptive result; falling back to tan‑transform...")
                    return fractionalDeriv_numeric_mpmath_tan(
                        order, prior, t_val, method, simplify,
                        return_log=return_log, dps=dps, complete=complete, u=u_val
                    )

        # If we reached max_L without convergence
        if L > max_L and integral_valid is not None:
            final_L = L / 2
            print(f"Warning: Adaptive integration did not converge before max_L={max_L} for t={t_val}, u={u_val}. Using last result from L={final_L}.")
        elif L > max_L and integral_valid is None:
            print(f"Adaptive method failed for t={t_val}, u={u_val}; falling back to tan‑transform...")
            return fractionalDeriv_numeric_mpmath_tan(
                order, prior, t_val, method, simplify,
                return_log=return_log, dps=dps, complete=complete, u=u_val
            )

        # Compute result using log-scale for stability
        if return_log:
            if abs(integral_valid) < 1e-300:
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

# ===== Example usage =====
if __name__ == "__main__":
    import math
    import numpy as np
    import jumufraktiv.MGFdictionary  # ensures priors are registered
    from jumufraktiv.mitMGFprior_class import mitMGFprior
    from mpmath import mp, exp

    # ---- Build Gamma prior ----
    gamma_prior = mitMGFprior.from_registry(
        "gamma",
        params={"alpha": 2.0, "beta": 3.0}
    )

    # ============================================================
    # Scalar tests (original)
    # ============================================================
    t_val = -1.0
    frac_order = 1.99

    print("=" * 60)
    print("Testing mpmath fractional derivative of Gamma MGF (scalar t)")
    print("=" * 60)
    print(f"  order={frac_order}, t={t_val}, alpha=2, beta=3")
    print("  Using default adaptive method (with fallback to tan)...")
    result_adaptive = fractionalDeriv_numeric_mpmath(
        order=frac_order,
        prior=gamma_prior,
        t=t_val,
        method='symbolic',
        return_log=False,
        dps=60,          # higher precision
        tol=1e-10         # tighter tolerance
    )
    print(f"  Adaptive result: {result_adaptive:.6e}")

    print("\n  Using explicit tan‑transform method...")
    result_tan = fractionalDeriv_numeric_mpmath(
        order=frac_order,
        prior=gamma_prior,
        t=t_val,
        method='symbolic',
        return_log=False,
        use_tan=True,
        dps=60
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
    deriv2 = sign2 * float(exp(mpf(log_abs2)))
    print(f"\n  Ordinary 2nd derivative at t={t_val}: {deriv2:.6e}")
    print(f"  Relative diff (adaptive vs 2nd): {abs(result_adaptive - deriv2) / abs(deriv2):.2e}")
    print(f"  Relative diff (tan vs 2nd):      {abs(result_tan - deriv2) / abs(deriv2):.2e}")
    print(f"  Relative diff (adaptive vs tan): {abs(result_adaptive - result_tan) / abs(result_tan):.2e}")

    # ============================================================
    # Scalar iMGF test (incomplete MGF)
    # ============================================================
    print("\n" + "=" * 60)
    print("Testing mpmath fractional derivative for iMGF (scalar t)")
    print("=" * 60)

    u_val = 2.0
    t_val_imgf = -1.0
    frac_order_imgf = 1.5

    gamma_prior_imgf = mitMGFprior.from_registry(
        "gamma",
        params={"alpha": 2.0, "beta": 3.0}
    )

    print(f"  order={frac_order_imgf}, t={t_val_imgf}, u={u_val}, alpha=2, beta=3")

    # ---- Adaptive method ----
    try:
        result_adaptive = fractionalDeriv_numeric_mpmath(
            order=frac_order_imgf,
            prior=gamma_prior_imgf,
            t=t_val_imgf,
            method='symbolic',
            return_log=False,
            complete=False,
            u=u_val,
            dps=60,
            tol=1e-10
        )
        print(f"  Adaptive result (ordinary): {result_adaptive:.6e}")
    except Exception as e:
        print(f"  Adaptive failed: {e}")
        result_adaptive = None

    # ---- Tan‑transform method ----
    try:
        result_tan = fractionalDeriv_numeric_mpmath(
            order=frac_order_imgf,
            prior=gamma_prior_imgf,
            t=t_val_imgf,
            method='symbolic',
            return_log=False,
            use_tan=True,
            complete=False,
            u=u_val,
            dps=60
        )
        print(f"  Tan‑transform result (ordinary): {result_tan:.6e}")
    except Exception as e:
        print(f"  Tan‑transform failed: {e}")
        result_tan = None

    # ---- Compare adaptive vs tan ----
    if result_adaptive is not None and result_tan is not None:
        diff_abs = abs(result_adaptive - result_tan)
        rel_diff = diff_abs / max(abs(result_adaptive), abs(result_tan), 1e-300)
        print(f"  Absolute diff (adaptive vs tan): {diff_abs:.2e}")
        print(f"  Relative diff: {rel_diff:.2e}")
        if rel_diff < 1e-6:
            print("  ✅ Good agreement between adaptive and tan.")
        else:
            print("  ⚠️  Significant difference – check precision.")

    # ============================================================
    # Vectorized tests (mpmath)
    # ============================================================
    print("\n" + "=" * 60)
    print("Vectorized t test (mpmath, fractional derivative)")
    print("=" * 60)

    t_vec = np.linspace(-2.0, -0.5, 5)
    print(f"  order={frac_order}, t values: {t_vec}")
    print("  Using tan‑transform method (use_tan=True)...")
    results_vec = fractionalDeriv_numeric_mpmath(
        order=frac_order,
        prior=gamma_prior,
        t=t_vec,
        method='symbolic',
        return_log=False,
        use_tan=True,
        dps=60
    )
    print("  Results:")
    for t_val, res in zip(t_vec, results_vec):
        print(f"    t={t_val:.2f}: {res:.6e}")

    # ---- Vectorized iMGF test ----
    print("\n" + "=" * 60)
    print("Vectorized iMGF test (mpmath, tan‑transform)")
    print("=" * 60)
    t_vec_imgf = np.linspace(-2.0, -0.5, 5)
    print(f"  order={frac_order_imgf}, t values: {t_vec_imgf}, u={u_val}")
    results_vec_imgf = fractionalDeriv_numeric_mpmath(
        order=frac_order_imgf,
        prior=gamma_prior_imgf,
        t=t_vec_imgf,
        method='symbolic',
        return_log=False,
        use_tan=True,
        complete=False,
        u=u_val,
        dps=60
    )
    print("  Results:")
    for t_val, res in zip(t_vec_imgf, results_vec_imgf):
        print(f"    t={t_val:.2f}: {res:.6e}")