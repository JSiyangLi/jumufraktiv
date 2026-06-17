"""
numeric_integerDeriv_Bell.py

Compute integer derivatives of MGFs via Bell polynomials.
Uses suggest_method_integerDeriv() with a low-order test to decide between
symbolic (SymPy) and numeric (JAX). Falls back to JAX if symbolic fails.
"""

import math
import sys
import sympy as sp
from jax import grad

# Local imports – make sure these files are in the same directory
from MGFdictionary.gammaMGF import (
    gamma_cgf_symbolic,
    gamma_cgf,
    gamma_cgf_jax
)
from MGFdictionary.paretoMGF import (
    pareto_cgf_symbolic,
    pareto_cgf,
    pareto_cgf_jax
)
from logsum import logplus, logminus, logplusvec   # needed for Bell polynomial
from numeric_symbolic_decision import suggest_method_integerDeriv


# ===== Bell polynomial (log‑space, sign‑aware) =====
def bell_polynomial_log(n: int, logv: list, vsign: list):
    """
    Compute log |B_n(v1,...,vn)| and sign of B_n.
    Uses recurrence:
        B_0 = 1,
        B_n = sum_{k=1}^n C(n-1, k-1) v_k B_{n-k}
    All operations done in log‑space to avoid overflow/underflow.
    """
    if n == 0:
        return 0.0, 1

    if len(logv) < n or len(vsign) < n:
        raise ValueError("logv and vsign must have length at least n")

    # B_0 = 1
    logB = [0.0] + [-float('inf')] * n   # log|B_i|
    signB = [1] + [1] * n                # sign of B_i

    for i in range(1, n + 1):
        pos_terms = []
        neg_terms = []

        for k in range(1, i + 1):
            # term = C(i-1, k-1) * v_k * B_{i-k}
            log_coeff = math.lgamma(i) - math.lgamma(k) - math.lgamma(i - k + 1)
            log_term = log_coeff + logv[k - 1] + logB[i - k]
            term_sign = vsign[k - 1] * signB[i - k]

            if term_sign > 0:
                pos_terms.append(log_term)
            elif term_sign < 0:
                neg_terms.append(log_term)
            # if term_sign == 0, skip (log_term = -inf)

        # Sum positive and negative terms using logplusvec
        sum_pos = logplusvec(pos_terms) if pos_terms else -float('inf')
        sum_neg = logplusvec(neg_terms) if neg_terms else -float('inf')

        # Combine
        if sum_pos == -float('inf') and sum_neg == -float('inf'):
            logB[i] = -float('inf')
            signB[i] = 1
        elif sum_neg == -float('inf'):
            logB[i] = sum_pos
            signB[i] = 1
        elif sum_pos == -float('inf'):
            logB[i] = sum_neg
            signB[i] = -1
        else:
            # Mixed signs: need to compute log(abs(pos - neg))
            if sum_pos >= sum_neg:
                logB[i] = logminus(sum_pos, sum_neg)  # log(pos - neg)
                signB[i] = 1
            else:
                logB[i] = logminus(sum_neg, sum_pos)  # log(neg - pos)
                signB[i] = -1

    return logB[n], signB[n]


# ===== helper CGF function =====
def get_cgf_func(prior: str, params: dict, use_jax: bool = False):
    """
    Return a callable cgf(t) = log M(t) for the given prior.
    If use_jax=True, return the JAX version; else the math (scalar) version.
    """
    if prior.lower() == "gamma":
        alpha = params['alpha']
        beta = params['beta']
        if use_jax:
            return lambda t: gamma_cgf_jax(t, alpha, beta)
        else:
            return lambda t: gamma_cgf(t, alpha, beta)
    else:  # pareto
        alpha = params['alpha']
        xi = params['xi']
        if use_jax:
            return lambda t: pareto_cgf_jax(t, alpha, xi)
        else:
            return lambda t: pareto_cgf(t, alpha, xi)


# ===== Main function =====
def integerDeriv_numeric_bell(t: float, prior: str, params: dict, order: int):
    """
    Compute the order‑th derivative of M(t) at t using Bell polynomials.
    Returns (log_abs_deriv, sign).
    """
    if order < 0:
        raise ValueError("Order must be non‑negative.")

    # ----- 1. Build symbolic CGF expression (log M) -----
    if prior.lower() == "gamma":
        cgf_sym = gamma_cgf_symbolic()
        # Extract symbols from the expression
        all_syms = cgf_sym.free_symbols
        t_sym = next(s for s in all_syms if s.name == 't')
        alpha_sym = next(s for s in all_syms if s.name == 'alpha')
        beta_sym = next(s for s in all_syms if s.name == 'beta')
        param_syms = (alpha_sym, beta_sym)
        param_names = ('alpha', 'beta')
    else:  # pareto
        cgf_sym = pareto_cgf_symbolic()
        # Extract symbols from the expression
        all_syms = cgf_sym.free_symbols
        t_sym = next(s for s in all_syms if s.name == 't')
        alpha_sym = next(s for s in all_syms if s.name == 'alpha')
        xi_sym = next(s for s in all_syms if s.name == 'xi')
        param_syms = (alpha_sym, xi_sym)
        param_names = ('alpha', 'xi')

    # ----- 2. Decision (test at low order) -----
    decision = suggest_method_integerDeriv(
        cgf_sym, t_sym, order,
        test_order=min(order, 2),
        timeout=1.0,
        return_decision=True
    )
    use_symbolic = decision['recommend_symbolic']
    print(f"Decision: {'Symbolic' if use_symbolic else 'Numeric (JAX)'}")

    # ----- 3. Try symbolic path (with fallback) -----
    if use_symbolic:
        try:
            # Substitute numeric parameters
            subs_dict = {}
            for name, sym in zip(param_names, param_syms):
                subs_dict[sym] = float(params[name])

            # Compute derivatives of CGF (K^(k)(t)) symbolically and evaluate
            kappa_log_abs = []
            kappa_sign = []
            for k in range(1, order + 1):
                deriv_expr = sp.diff(cgf_sym, t_sym, k)
                if t == 0:
                    val = sp.limit(deriv_expr, t_sym, 0, dir='-').subs(subs_dict).evalf()
                else:
                    val = deriv_expr.subs({t_sym: t}).subs(subs_dict).evalf()
                val = float(val)

                if abs(val) < sys.float_info.epsilon:
                    kappa_log_abs.append(-float('inf'))
                    kappa_sign.append(1)
                else:
                    kappa_log_abs.append(math.log(abs(val)))
                    kappa_sign.append(1 if val > 0 else -1)

            # Get CGF(t) using the math version
            cgf_func = get_cgf_func(prior, params, use_jax=False)
            cgf_t = cgf_func(t)

            # Bell polynomial
            log_abs_B, sign_B = bell_polynomial_log(order, kappa_log_abs, kappa_sign)
            # Final: log|M^(n)(t)| = K(t) + log|B_n|
            return cgf_t + log_abs_B, sign_B

        except Exception as e:
            import traceback
            traceback.print_exc()   # prints full stack trace
            print(f"⚠️ Symbolic path failed for full order: {e}. Falling back to JAX.")
            use_symbolic = False

    # ----- 4. Numeric path using JAX (fallback or direct) -----
    cgf_func = get_cgf_func(prior, params, use_jax=True)

    # Build derivative functions K^(0), K^(1), ..., K^(order) via nested grad
    deriv_funcs = [cgf_func]
    for _ in range(order):
        deriv_funcs.append(grad(deriv_funcs[-1]))

    kappa_log_abs = []
    kappa_sign = []
    for k in range(1, order + 1):
        val = float(deriv_funcs[k](t))
        if abs(val) < sys.float_info.epsilon:
            kappa_log_abs.append(-float('inf'))
            kappa_sign.append(1)
        else:
            kappa_log_abs.append(math.log(abs(val)))
            kappa_sign.append(1 if val > 0 else -1)

    cgf_t = float(cgf_func(t))
    log_abs_B, sign_B = bell_polynomial_log(order, kappa_log_abs, kappa_sign)

    return cgf_t + log_abs_B, sign_B


if __name__ == "__main__":
    # ---- Existing tests (Gamma and Pareto, orders 0–3) ----
    print("="*60)
    print("Testing integerDeriv_numeric_bell() for orders 0–3")
    print("="*60)

    # Gamma
    gamma_params = {'alpha': 2.0, 'beta': 3.0}
    t_val = -1.0
    for n in range(4):
        log_abs, sign = integerDeriv_numeric_bell(t_val, 'gamma', gamma_params, n)
        print(f"Gamma M^{{{n}}}({t_val}) : log|.| = {log_abs:.4f}, sign = {sign}")

    # Pareto
    pareto_params = {'alpha': 3.5, 'xi': 1.0}
    t_val = -0.5
    for n in range(4):
        log_abs, sign = integerDeriv_numeric_bell(t_val, 'pareto', pareto_params, n)
        print(f"Pareto M^{{{n}}}({t_val}) : log|.| = {log_abs:.4f}, sign = {sign}")

    # ---- New: 5th derivative of Gamma with very small parameters ----
    print("\n" + "="*60)
    print("Testing 5th derivative of Gamma MGF with small parameters")
    print("(alpha = beta = 1e-5, t = -1e-6)")
    print("="*60)

    import time
    alpha_small = 1e-5
    beta_small = 1e-5
    t_small = -1e-6
    order_test = 5
    small_params = {'alpha': alpha_small, 'beta': beta_small}

    start = time.time()
    log_abs, sign = integerDeriv_numeric_bell(t_small, 'gamma', small_params, order_test)
    elapsed = time.time() - start

    print(f"Gamma M^{{{order_test}}}({t_small}) with alpha={alpha_small:.1e}, beta={beta_small:.1e}")
    print(f"  log|deriv| = {log_abs:.6e}")
    print(f"  sign       = {sign}")
    print(f"  Time       = {elapsed:.3f} seconds")

    # Analytical check for Gamma MGF derivatives
    # M^(n)(t) = (alpha)_n * beta^alpha * (beta - t)^(-alpha - n)
    # log|...| = logΓ(alpha+n) - logΓ(alpha) + alpha*log(beta) - (alpha+n)*log(beta-t)
    import math
    log_falling = math.lgamma(alpha_small + order_test) - math.lgamma(alpha_small)
    log_expected = (log_falling 
                    + alpha_small * math.log(beta_small)
                    - (alpha_small + order_test) * math.log(beta_small - t_small))
    print(f"  Analytical log|deriv| = {log_expected:.6e}")
    print(f"  Difference = {log_abs - log_expected:.2e}")
    if abs(log_abs - log_expected) < 1e-6:
        print("  ✅ Matches analytical formula.")
    else:
        print("  ⚠️  Difference is not negligible – check precision.")