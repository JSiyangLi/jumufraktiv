"""
numeric_integerDeriv_Bell.py

Compute integer derivatives of MGFs via Bell polynomials.
Uses suggest_method_integerDeriv() with a low-order test to decide between
symbolic (SymPy) and numeric (JAX). Falls back to JAX if symbolic fails.
"""

import math
import sympy as sp
import jax
import jax.numpy as jnp
from jax import grad

# Local imports – make sure these files are in the same directory
from MGFdictionary.gammaMGF import (
    gamma_mgf_symbolic,
    log_gamma_mgf as log_gamma_mgf_math,
)
from MGFdictionary.paretoMGF import (
    pareto_mgf_symbolic,
    log_pareto_mgf as log_pareto_mgf_mpmath,
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


# ===== JAX‑compatible log MGF functions =====
def log_gamma_mgf_jax(t, alpha, beta):
    """JAX version of log MGF for Gamma(alpha, rate=beta)."""
    return alpha * (jnp.log(beta) - jnp.log(beta - t))

def log_pareto_mgf_jax(t, alpha, xi):
    """
    JAX version of log MGF for Pareto(shape=alpha, scale=xi).
    Uses: log M(t) = log(alpha) + alpha*log(-xi*t) + log(Γ(-alpha, -xi*t))
    with Γ(a,z) = Γ(a) * Q(a,z) where Q is the regularised upper incomplete gamma.
    Handles t=0 via jnp.where.
    """
    def safe_log(t_val):
        z = -xi * t_val
        a = -alpha
        # log|Γ(a)| + log(Q(a,z))
        log_gamma_a = jax.scipy.special.gammaln(a)   # works for negative a
        log_q = jnp.log(jax.scipy.special.gammaincc(a, z))
        log_inc_gamma = log_gamma_a + log_q
        return jnp.log(alpha) + alpha * jnp.log(z) + log_inc_gamma

    return jnp.where(t == 0.0, 0.0, safe_log(t))

def get_cgf_jax(prior, params):
    """Return a JAX function cgf(t) = log M(t)."""
    if prior.lower() == "gamma":
        alpha = params['alpha']
        beta = params['beta']
        return lambda t: log_gamma_mgf_jax(t, alpha, beta)
    else:  # pareto
        alpha = params['alpha']
        xi = params['xi']
        return lambda t: log_pareto_mgf_jax(t, alpha, xi)


# ===== Main function =====
def integerDeriv_numeric_bell(t, prior, params, order):
    """
    Compute the order‑th derivative of M(t) at t using Bell polynomials.

    Parameters
    ----------
    t : float
        Evaluation point (must satisfy domain restrictions).
    prior : str
        'gamma' or 'pareto'.
    params : dict
        For Gamma: {'alpha': ..., 'beta': ...}
        For Pareto: {'alpha': ..., 'xi': ...}
    order : int
        Order of the derivative (>= 0).

    Returns
    -------
    tuple (log_abs_deriv, sign)
        log_abs_deriv : float, log of absolute value of M^(order)(t)
        sign : int, +1 or -1
    """
    if order < 0:
        raise ValueError("Order must be non‑negative.")

    # ----- 1. Build symbolic log MGF expression -----
    if prior.lower() == "gamma":
        mgf_sym = gamma_mgf_symbolic()
        t_sym, alpha_sym, beta_sym = sp.symbols('t alpha beta', positive=True, real=True)
        log_expr = sp.log(mgf_sym)
        param_syms = (alpha_sym, beta_sym)
        param_names = ('alpha', 'beta')
    else:  # pareto
        mgf_sym = pareto_mgf_symbolic()
        t_sym, alpha_sym, xi_sym = sp.symbols('t alpha xi', positive=True, real=True)
        log_expr = sp.log(mgf_sym)
        param_syms = (alpha_sym, xi_sym)
        param_names = ('alpha', 'xi')

    # ----- 2. Decision (test at low order) -----
    decision = suggest_method_integerDeriv(log_expr, t_sym, order,
                                           test_order=min(order, 2),
                                           timeout=1.0,
                                           return_decision=True)
    use_symbolic = decision['recommend_symbolic']
    print(f"Decision: {'Symbolic' if use_symbolic else 'Numeric (JAX)'} "
          f"({decision['message']})")

    # ----- 3. Try symbolic path (with fallback) -----
    if use_symbolic:
        try:
            # Substitute numeric parameters
            subs_dict = {}
            for name, sym in zip(param_names, param_syms):
                subs_dict[sym] = float(params[name])

            # Compute derivatives K^(k)(t) symbolically and evaluate
            kappa_log_abs = []
            kappa_sign = []
            for k in range(1, order + 1):
                deriv_expr = sp.diff(log_expr, t_sym, k)
                if t == 0:
                    val = sp.limit(deriv_expr, t_sym, 0, dir='-').subs(subs_dict).evalf()
                else:
                    val = deriv_expr.subs({t_sym: t}).subs(subs_dict).evalf()
                val = float(val)
                if abs(val) < 1e-15:
                    log_abs = -float('inf')
                    sign = 1
                else:
                    log_abs = math.log(abs(val))
                    sign = 1 if val > 0 else -1
                kappa_log_abs.append(log_abs)
                kappa_sign.append(sign)

            # Log M(t) – use the numeric log MGF functions (math/mpmath)
            if prior.lower() == "gamma":
                log_mgf_t = log_gamma_mgf_math(t, params['alpha'], params['beta'])
            else:
                log_mgf_t = log_pareto_mgf_mpmath(t, params['alpha'], params['xi'])

            # Bell polynomial
            log_abs_B, sign_B = bell_polynomial_log(order, kappa_log_abs, kappa_sign)
            return log_mgf_t + log_abs_B, sign_B

        except Exception as e:
            print(f"⚠️  Symbolic path failed for full order: {e}. Falling back to JAX.")
            use_symbolic = False  # fall through to numeric

    # ----- 4. Numeric path using JAX (fallback or direct) -----
    # (If use_symbolic was False, or symbolic failed)
    cgf = get_cgf_jax(prior, params)
    # Build derivative functions: K^(0), K^(1), ..., K^(order)
    deriv_funcs = [cgf]
    for _ in range(order):
        deriv_funcs.append(grad(deriv_funcs[-1]))

    kappa_log_abs = []
    kappa_sign = []
    for k in range(1, order + 1):
        val = float(deriv_funcs[k](t))
        if abs(val) < 1e-15:
            log_abs = -float('inf')
            sign = 1
        else:
            log_abs = math.log(abs(val))
            sign = 1 if val > 0 else -1
        kappa_log_abs.append(log_abs)
        kappa_sign.append(sign)

    log_mgf_t = float(cgf(t))
    log_abs_B, sign_B = bell_polynomial_log(order, kappa_log_abs, kappa_sign)
    return log_mgf_t + log_abs_B, sign_B


# ===== Example usage =====
if __name__ == "__main__":
    # Gamma
    gamma_params = {'alpha': 2.0, 'beta': 3.0}
    t_val = 1.0
    for n in range(4):
        log_abs, sign = integerDeriv_numeric_bell(t_val, 'gamma', gamma_params, n)
        print(f"Gamma M^{{{n}}}({t_val}) : log|.| = {log_abs:.4f}, sign = {sign}")

    # Pareto
    pareto_params = {'alpha': 3.5, 'xi': 1.0}
    t_val = -0.5
    for n in range(4):
        log_abs, sign = integerDeriv_numeric_bell(t_val, 'pareto', pareto_params, n)
        print(f"Pareto M^{{{n}}}({t_val}) : log|.| = {log_abs:.4f}, sign = {sign}")