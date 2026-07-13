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

from mpmath import mp, pi, exp, log, tan, gamma, quad, mpf
from jumufraktiv.derivativeDispatch import mgfDerivative_integer
from jumufraktiv.mitMGFprior_class import mitMGFprior


def fractionalDeriv_numeric_mpmath_tan(
    order: float,
    prior: mitMGFprior,
    t: float,
    method: str = "symbolic",
    simplify: bool = False,
    complete: bool = True,
    return_log: bool = False,
    margin: float = 1e-12,
    max_u: float = 20.0,
    dps: int = 50,
    u: float = None                     # NEW: truncation point for incomplete MGF
):
    """
    Compute fractional derivative using scaled tan‑transform:
        u = max_u * tan(theta)
    with theta in (-pi/2, pi/2). Uses mpmath.quad.

    Parameters
    ----------
    order : float
        Fractional order (positive).
    prior : mitMGFprior
        Prior object providing the MGF.
    t : float
        Evaluation point.
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
    u : float, optional
        Truncation point for incomplete MGF (used when complete=False). Default None.
    """
    mp.dps = dps

    if order <= 0:
        raise ValueError("Fractional order must be positive.")

    # Integer order: use mgfDerivative_integer (returns Python float)
    if order == int(order):
        log_abs, sign = mgfDerivative_integer(
            order=int(order),
            prior=prior,
            method=method,
            t=t,
            simplify=simplify,
            log=True,
            complete=complete,
            u=u                         # pass u through
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
            y = t - z
            # Get derivative in log scale (Python float)
            log_abs, sign = mgfDerivative_integer(
                order=n + 1,
                prior=prior,
                method=method,
                t=float(y),
                simplify=simplify,
                log=True,
                complete=complete,
                u=u                         # pass u through
            )
            if log_abs == -float('inf'):
                return mpf(0.0)
            # log Jacobian: du/dtheta = max_u * (1 + tan^2)
            log_jacobian = log(max_u) + log(1 + tan_theta * tan_theta)
            log_integrand = gamma_val * u_var + mpf(log_abs) + log_jacobian
            # Return ordinary value
            return exp(log_integrand) * sign
        except Exception:
            return mpf(0.0)

    a = -pi / 2 + margin
    b = pi / 2 - margin

    try:
        integral = quad(integrand_theta, (a, b), method='tanh-sinh')
    except Exception as e:
        print(f"mpmath tan‑transform integration failed: {e}")
        if return_log:
            return float('nan'), 1
        else:
            return float('nan')

    factor = 1.0 / gamma(gamma_val)
    result = float(factor * integral)

    if return_log:
        if abs(result) < 1e-300:
            return -float('inf'), 1
        else:
            return float(log(mpf(abs(result)))), 1 if result > 0 else -1
    else:
        return result


def fractionalDeriv_numeric_mpmath(
    order: float,
    prior: mitMGFprior,
    t: float,
    method: str = "symbolic",
    complete: bool = True,
    simplify: bool = False,
    return_log: bool = False,
    initial_L: float = 10.0,
    max_L: float = 1e4,
    tol: float = 1e-8,           # tightened for better accuracy
    use_tan: bool = False,
    dps: int = 50,
    u: float = None                     # NEW: truncation point for incomplete MGF
):
    """
    Compute the Liouville‑Caputo fractional derivative using mpmath.

    Parameters
    ----------
    order : float
        Fractional order (positive). If integer, returns ordinary derivative.
    prior : mitMGFprior
        Prior object providing the MGF.
    t : float
        Evaluation point (must be within MGF domain).
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
    u : float, optional
        Truncation point for incomplete MGF (used when complete=False). Default None.
    """
    mp.dps = dps

    if order <= 0:
        raise ValueError("Fractional order must be positive.")

    # ---- 1. Integer order ----
    if order == int(order):
        log_abs, sign = mgfDerivative_integer(
            order=int(order),
            prior=prior,
            method=method,
            t=t,
            simplify=simplify,
            log=True,
            complete=complete,
            u=u                         # pass u through
        )
        if return_log:
            return log_abs, sign
        else:
            if log_abs == -float('inf'):
                return 0.0
            return sign * float(exp(mpf(log_abs)))

    # ---- 2. use_tan: direct to tan version ----
    if use_tan:
        return fractionalDeriv_numeric_mpmath_tan(
            order, prior, t, method, simplify,
            return_log=return_log, dps=dps, complete=complete, u=u   # pass u
        )

    # ---- 3. Adaptive range method ----
    n = int(mp.floor(order))
    gamma_val = mpf((n + 1) - order)

    def integrand(u_var):
        u_var = mpf(u_var)
        z = exp(u_var)
        y = t - z
        log_abs, sign = mgfDerivative_integer(
            order=n + 1,
            prior=prior,
            method=method,
            t=float(y),
            simplify=simplify,
            complete=complete,
            log=True,
            u=u                         # pass u through
        )
        if log_abs == -float('inf'):
            return mpf(0.0)
        log_integrand = gamma_val * u_var + mpf(log_abs)
        return exp(log_integrand) * sign

    L = initial_L
    integral_valid = None
    prev_integral = None
    factor = 1.0 / gamma(gamma_val)
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
            print(f"mpmath adaptive integration failed at L={L}: {e}")
            if integral_valid is not None:
                final_L = L / 2
                print(f"  Using last valid result from L={final_L}.")
                break
            else:
                print("  No valid adaptive result; falling back to tan‑transform...")
                return fractionalDeriv_numeric_mpmath_tan(
                    order, prior, t, method, simplify,
                    return_log=return_log, dps=dps, complete=complete, u=u
                )

    # If we reached max_L without breaking, set final_L to last valid
    if L > max_L and integral_valid is not None:
        final_L = L / 2
        print(f"Warning: Adaptive integration did not converge before max_L={max_L}. Using last result from L={final_L}.")
    elif L > max_L and integral_valid is None:
        print("Adaptive method failed to produce a result; falling back to tan‑transform...")
        return fractionalDeriv_numeric_mpmath_tan(
            order, prior, t, method, simplify,
            return_log=return_log, dps=dps, complete=complete, u=u
        )

    # Now final_L should be set
    if final_L is not None:
        print(f"  Adaptive integration used L = {final_L}.")
    else:
        print("  Adaptive integration used L = (unknown).")

    result = factor * integral_valid

    if return_log:
        if abs(result) < 1e-300:
            return -float('inf'), 1
        else:
            return float(log(mpf(abs(result)))), 1 if result > 0 else -1
    else:
        return result


# ===== Example usage =====
if __name__ == "__main__":
    import jumufraktiv.MGFdictionary  # ensures priors are registered
    from jumufraktiv.mitMGFprior_class import mitMGFprior

    # Build Gamma prior
    gamma_prior = mitMGFprior.from_registry(
        "gamma",
        params={"alpha": 2.0, "beta": 3.0}
    )

    t_val = -1.0
    frac_order = 1.99

    print("Testing mpmath fractional derivative of Gamma MGF")
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