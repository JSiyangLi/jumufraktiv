"""
numeric_integerDeriv_JAX.py

Compute integer derivatives of MGFs using JAX's Taylor‑mode AD (jet).
Differentiates the MGF directly – no Bell polynomials needed.
"""

import math
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.experimental import jet

from MGFdictionary.gammaMGF import gamma_mgf_jax
from MGFdictionary.paretoMGF import pareto_mgf_jax

def integerDeriv_numeric_jax(t, prior, params, order):
    """
    Returns
    -------
    log_abs : float
        log(|d^order/dt^order M(t)|)
    sign : int
        sign of the derivative (+1 or -1)
    """

    if order < 0:
        raise ValueError("Order must be non-negative.")

    # Build MGF
    if prior.lower() == "gamma":
        alpha = params["alpha"]
        beta = params["beta"]

        def mgf(x):
            return gamma_mgf_jax(x, alpha, beta)

    elif prior.lower() == "pareto":
        alpha = params["alpha"]
        xi = params["xi"]

        def mgf(x):
            return pareto_mgf_jax(x, alpha, xi)

    else:
        raise ValueError("prior must be 'gamma' or 'pareto'")

    # Zeroth derivative
    if order == 0:
        val = float(mgf(t))

        if abs(val) < 1e-15:
            return -float("inf"), 1

        return math.log(abs(val)), (1 if val > 0 else -1)

    # Taylor series seed:
    # x(t + ε) = t + ε
    series_in = ((1.0,) + (0.0,) * (order - 1),)

    primal_out, series_out = jet.jet(
        mgf,
        (t,),
        series_in,
    )

    # series_out[k] = f^(k+1)(t)/(k+1)!
    coef = float(series_out[order - 1])

    if abs(coef) < 1e-300:
        return -float("inf"), 1

    sign = 1 if coef > 0 else -1

    # log(|f^(n)|) = log(|coef|) + log(n!)
    log_abs = math.log(abs(coef))

    return log_abs, sign


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("Testing integerDeriv_numeric_jax (direct MGF derivatives via jet)")
    print("=" * 60)

    # Gamma prior
    gamma_params = {'alpha': 2.0, 'beta': 3.0}
    t_val = 1.0
    for n in range(4):
        log_abs, sign = integerDeriv_numeric_jax(t_val, 'gamma', gamma_params, n)
        print(f"Gamma M^{{{n}}}({t_val}) : log|.| = {log_abs:.4f}, sign = {sign}")

    # High‑order test: 8th derivative of Gamma with very small parameters
    print("\n" + "=" * 60)
    print("High‑order test: Gamma M^175 with alpha=beta=1e-5, t=-1e3")
    alpha_small = 1e-5
    beta_small = 1e-5
    t_small = -1e3
    order_high = 175
    small_params = {'alpha': alpha_small, 'beta': beta_small}

    start = time.time()
    log_abs, sign = integerDeriv_numeric_jax(t_small, 'gamma', small_params, order_high)
    elapsed = time.time() - start
    print(f"  log|deriv| = {log_abs:.6e}, sign = {sign}")
    print(f"  Time = {elapsed:.4f} seconds")

    # Analytical check
    import math
    log_falling = math.lgamma(alpha_small + order_high) - math.lgamma(alpha_small)
    log_expected = (log_falling
                    + alpha_small * math.log(beta_small)
                    - (alpha_small + order_high) * math.log(beta_small - t_small))
    print(f"  Analytical log|deriv| = {log_expected:.6e}")
    print(f"  Difference = {log_abs - log_expected:.2e}")
    if abs(log_abs - log_expected) < 1e-6:
        print("  ✅ Matches analytical formula.")
    else:
        print("  ⚠️ Difference not negligible – check precision.")