"""
numeric_integerDeriv_Torch.py

Compute integer derivatives of MGFs using PyTorch's automatic differentiation.
Differentiates the MGF directly – no Bell polynomials needed.
"""

import math
import torch

# Import Torch‑compatible MGF functions from the MGF dictionary
from MGFdictionary.gammaMGF import gamma_mgf_torch
from MGFdictionary.paretoMGF import pareto_mgf_torch


def integerDeriv_numeric_torch(t, prior, params, order):
    """
    Compute the order‑th derivative of the MGF at t using PyTorch.

    Returns
    -------
    log_abs : float
        log(|d^order/dt^order M(t)|)
    sign : int
        sign of the derivative (+1 or -1)
    """
    if order < 0:
        raise ValueError("Order must be non-negative.")

    # Convert t to torch scalar with requires_grad
    t_tensor = torch.tensor(t, dtype=torch.float64, requires_grad=True)

    # Build MGF function
    if prior.lower() == "gamma":
        alpha = params["alpha"]
        beta = params["beta"]
        def mgf(x):
            return gamma_mgf_torch(x, alpha, beta)
    elif prior.lower() == "pareto":
        alpha = params["alpha"]
        xi = params["xi"]
        def mgf(x):
            return pareto_mgf_torch(x, alpha, xi)
    else:
        raise ValueError("prior must be 'gamma' or 'pareto'")

    # Zeroth derivative
    if order == 0:
        val = mgf(t_tensor).item()
        if abs(val) < 1e-15:
            return -float("inf"), 1
        return math.log(abs(val)), (1 if val > 0 else -1)

    # Compute derivatives recursively using autograd.grad
    current = mgf(t_tensor)
    for k in range(1, order + 1):
        grad, = torch.autograd.grad(
            current,
            t_tensor,
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )
        if k == order:
            deriv_val = grad.item()
            break
        current = grad

    # Handle near-zero
    if abs(deriv_val) < 1e-300:
        return -float("inf"), 1

    sign = 1 if deriv_val > 0 else -1
    log_abs = math.log(abs(deriv_val))

    return log_abs, sign


# ===== Example usage =====
if __name__ == "__main__":
    import time

    print("=" * 60)
    print("Testing integerDeriv_numeric_torch (direct MGF derivatives via autograd)")
    print("=" * 60)

    # ---- Gamma prior ----
    gamma_params = {'alpha': 2.0, 'beta': 3.0}
    t_val = 1.0
    for n in range(4):
        log_abs, sign = integerDeriv_numeric_torch(t_val, 'gamma', gamma_params, n)
        print(f"Gamma M^{{{n}}}({t_val}) : log|.| = {log_abs:.4f}, sign = {sign}")

    # ---- Pareto prior (small order) ----
    print("\n" + "=" * 60)
    print("Testing Pareto prior (order=3)")
    pareto_params = {'alpha': 3.5, 'xi': 1.0}
    t_val = -0.5
    log_abs, sign = integerDeriv_numeric_torch(t_val, 'pareto', pareto_params, 3)
    print(f"Pareto M^3({t_val}) : log|.| = {log_abs:.6f}, sign = {sign}")

    # ---- High‑order test: Gamma M^5 with very small parameters ----
    print("\n" + "=" * 60)
    print("High‑order test: Gamma M^5 with alpha=beta=1e-5, t=-1e-6")
    alpha_small = 1e-5
    beta_small = 1e-5
    t_small = -1e-6
    order_high = 5
    small_params = {'alpha': alpha_small, 'beta': beta_small}

    start = time.time()
    log_abs, sign = integerDeriv_numeric_torch(t_small, 'gamma', small_params, order_high)
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