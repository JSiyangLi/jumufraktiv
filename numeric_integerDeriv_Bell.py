"""
numeric_integerDeriv_Bell.py

Compute integer derivatives of MGFs via Bell polynomials.
Uses suggest_method_integerDeriv() with a low-order test to decide between
symbolic (SymPy) and numeric (JAX). Falls back to JAX if symbolic fails
or exceeds a user‑specified timeout.
"""

import math
import time
import sys
import sympy as sp
import jax
jax.config.update("jax_enable_x64", True)
from jax import grad

# New import for mitMGFprior
from jumufraktiv.mitMGFprior_class import mitMGFprior
from jumufraktiv.logsum import logplus, logminus, logplusvec
from jumufraktiv.numeric_symbolic_decision import suggest_method_integerDeriv
from jax.experimental import jet
import jax.numpy as jnp


# ===== Helper: CGF derivatives =====
def cgf_derivatives_jet(cgf_func, t, order):
    """
    Compute derivatives of the CGF using JAX's jet (Taylor mode).
    """
    if order == 0:
        return [], []

    series_in = ((1.0,) + (0.0,) * (order - 1),)
    _, derivs = jet.jet(cgf_func, (t,), series_in)

    log_abs = []
    signs = []
    for d in derivs[:order]:
        val = float(d)
        if abs(val) < sys.float_info.epsilon:
            log_abs.append(-float("inf"))
            signs.append(1)
        else:
            log_abs.append(math.log(abs(val)))
            signs.append(1 if val > 0 else -1)
    return log_abs, signs


def cgf_derivatives_grad(cgf_func, t, order):
    """
    Compute derivatives of the CGF using nested jax.grad (reverse mode).
    """
    log_abs = []
    signs = []
    f = cgf_func
    for k in range(1, order + 1):
        f = grad(f)
        val = float(f(t))
        if abs(val) < sys.float_info.epsilon:
            log_abs.append(-float("inf"))
            signs.append(1)
        else:
            log_abs.append(math.log(abs(val)))
            signs.append(1 if val > 0 else -1)
    return log_abs, signs


def cgf_derivatives_auto(cgf_func, t, order):
    """
    Try jet first; if it fails due to missing rules, fall back to grad.
    """
    try:
        return cgf_derivatives_jet(cgf_func, t, order)
    except Exception as e:
        msg = str(e).lower()
        unsupported = (
            isinstance(e, KeyError)
            or "jet" in msg
            or "primitive" in msg
            or "not implemented" in msg
        )
        if unsupported:
            print(f"⚠️ Jet failed ({type(e).__name__}: {e}). Falling back to grad().")
            return cgf_derivatives_grad(cgf_func, t, order)
        raise


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

    logB = [0.0] + [-float('inf')] * n
    signB = [1] + [1] * n

    for i in range(1, n + 1):
        pos_terms = []
        neg_terms = []

        for k in range(1, i + 1):
            log_coeff = math.lgamma(i) - math.lgamma(k) - math.lgamma(i - k + 1)
            log_term = log_coeff + logv[k - 1] + logB[i - k]
            term_sign = vsign[k - 1] * signB[i - k]

            if term_sign > 0:
                pos_terms.append(log_term)
            elif term_sign < 0:
                neg_terms.append(log_term)

        sum_pos = logplusvec(pos_terms) if pos_terms else -float('inf')
        sum_neg = logplusvec(neg_terms) if neg_terms else -float('inf')

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
            if sum_pos >= sum_neg:
                logB[i] = logminus(sum_pos, sum_neg)
                signB[i] = 1
            else:
                logB[i] = logminus(sum_neg, sum_pos)
                signB[i] = -1

    return logB[n], signB[n]


# ===== main function =====
def integerDeriv_numeric_bell(
    t: float,
    prior: mitMGFprior,
    order: int,
    symbolic_timeout: float = 600.0,
    cgf_method: str = 'auto'
):
    """
    Compute the order‑th derivative of M(t) at t using Bell polynomials.

    Parameters
    ----------
    t : float
        Evaluation point.
    prior : mitMGFprior
        Prior object providing symbolic and numeric CGF functions.
    order : int
        Derivative order (>= 0).
    symbolic_timeout : float, optional
        Maximum time (seconds) allowed for the symbolic CGF derivative
        computation. If exceeded, fall back to JAX. Default 600.
    cgf_method : str, optional
        Method for computing CGF derivatives in the numeric path.
        Options:
            - 'auto': try jet, fall back to grad on failure (default)
            - 'jet': force JAX Taylor mode (jet)
            - 'grad': force nested jax.grad (reverse mode)
        If 'jet' or 'grad', the symbolic path is skipped entirely.

    Returns
    -------
    tuple (log_abs_deriv, sign)
    """
    if order < 0:
        raise ValueError("Order must be non‑negative.")

    # ----- 1. Build symbolic CGF expression -----
    cgf_sym = prior.cgf_sym
    if cgf_sym is None:
        raise ValueError("Prior does not provide a symbolic CGF (cgf_sym).")

    if callable(cgf_sym):
        cgf_sym = cgf_sym()

    if not isinstance(cgf_sym, sp.Expr):
        raise TypeError("cgf_sym must be a SymPy expression.")

    t_sym = next((s for s in cgf_sym.free_symbols if s.name == 't'), None)
    if t_sym is None:
        raise RuntimeError("No symbol 't' found in the CGF expression.")

    params = prior.params or {}

    # ----- 2. Decision: force JAX if user explicitly requested jet/grad -----
    if cgf_method.lower() in ('jet', 'grad'):
        use_symbolic = False
        print(f"Decision: Forced JAX numeric path (cgf_method='{cgf_method}')")
    else:
        # Run the low-order test only if method is 'auto'
        decision = suggest_method_integerDeriv(
            cgf_sym, t_sym, order,
            test_order=min(order, 2),
            timeout=1.0,
            return_decision=True
        )
        use_symbolic = decision['recommend_symbolic']
        print(f"Decision: {'Symbolic' if use_symbolic else 'Numeric (JAX)'}")

    # Flag to track numeric path
    use_jax = not use_symbolic

    # ----- 3. Try symbolic path (with timeout) -----
    if use_symbolic:
        try:
            subs_dict = {}
            for sym in cgf_sym.free_symbols:
                if sym.name in params:
                    subs_dict[sym] = float(params[sym.name])

            kappa_log_abs = []
            kappa_sign = []
            start_time = time.time()

            for k in range(1, order + 1):
                if time.time() - start_time > symbolic_timeout:
                    raise TimeoutError(
                        f"Symbolic computation exceeded {symbolic_timeout:.1f} seconds."
                    )

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

            cgf_t = prior.cgf(t)

            log_abs_B, sign_B = bell_polynomial_log(order, kappa_log_abs, kappa_sign)
            result = cgf_t + log_abs_B

            if math.isnan(result) or math.isinf(result):
                raise ValueError("Symbolic result is NaN or inf – falling back to JAX")

            return result, sign_B

        except (TimeoutError, Exception) as e:
            if isinstance(e, TimeoutError):
                print(f"⚠️ Symbolic route timed out after {symbolic_timeout:.1f}s.")
            else:
                import traceback
                traceback.print_exc()
                print(f"⚠️ Symbolic path failed: {e}. Falling back to JAX.")
            use_jax = True

    # ----- 4. Numeric (JAX) path -----
    if use_jax:
        print("Using JAX numeric path...")
        cgf_func = prior.cgf_jax
        if cgf_func is None:
            raise ValueError("Prior does not provide cgf_jax.")

        if cgf_method == 'jet':
            kappa_log_abs, kappa_sign = cgf_derivatives_jet(cgf_func, t, order)
        elif cgf_method == 'grad':
            kappa_log_abs, kappa_sign = cgf_derivatives_grad(cgf_func, t, order)
        else:  # 'auto'
            kappa_log_abs, kappa_sign = cgf_derivatives_auto(cgf_func, t, order)

        cgf_t = float(cgf_func(t))
        log_abs_B, sign_B = bell_polynomial_log(order, kappa_log_abs, kappa_sign)
        return cgf_t + log_abs_B, sign_B

    raise RuntimeError("No path was executed.")


# ===== Example usage =====
if __name__ == "__main__":
    import time
    import math
    import jumufraktiv.MGFdictionary  # ensures priors are registered
    from jumufraktiv.mitMGFprior_class import mitMGFprior

    print("=" * 60)
    print("Testing integerDeriv_numeric_bell() for Gamma prior (orders 0–3)")
    print("=" * 60)

    # ---- Gamma prior ----
    gamma_params = {'alpha': 2.0, 'beta': 3.0}
    gamma_prior = mitMGFprior.from_registry("gamma", params=gamma_params)
    t_val = -1.0
    for n in range(4):
        log_abs, sign = integerDeriv_numeric_bell(t_val, gamma_prior, n)
        print(f"Gamma M^{{{n}}}({t_val}) : log|.| = {log_abs:.4f}, sign = {sign}")

    # ---- High‑order test: 50th derivative of Gamma with very small parameters ----
    print("\n" + "=" * 60)
    print("Testing 50th derivative of Gamma MGF with small parameters")
    print("(alpha = beta = 1e-5, t = -1e-6)")
    print("=" * 60)

    alpha_small = 1e-5
    beta_small = 1e-5
    t_small = -1e-6
    order_test = 50
    small_params = {'alpha': alpha_small, 'beta': beta_small}
    small_prior = mitMGFprior.from_registry("gamma", params=small_params)

    start = time.time()
    log_abs, sign = integerDeriv_numeric_bell(
        t_small, small_prior, order_test,
        symbolic_timeout=600.0   # 10 minutes
    )
    elapsed = time.time() - start

    print(f"Gamma M^{{{order_test}}}({t_small}) with alpha={alpha_small:.1e}, beta={beta_small:.1e}")
    print(f"  log|deriv| = {log_abs:.6e}")
    print(f"  sign       = {sign}")
    print(f"  Time       = {elapsed:.3f} seconds")

    # Analytical check
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