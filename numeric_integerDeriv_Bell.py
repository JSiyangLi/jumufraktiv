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

from jumufraktiv.mitMGFprior_class import mitMGFprior
from jumufraktiv.logsum import logplus, logminus, logplusvec
from jumufraktiv.numeric_symbolic_decision import suggest_method_integerDeriv
from jax.experimental import jet
import jax.numpy as jnp


# ===== Helper: CGF derivatives (unchanged) =====
def cgf_derivatives_jet(cgf_func, t, order):
    """Compute derivatives of the CGF using JAX's jet (Taylor mode)."""
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
    """Compute derivatives of the CGF using nested jax.grad (reverse mode)."""
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
    """Try jet first; if it fails due to missing rules, fall back to grad."""
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
    """Compute log |B_n(v1,...,vn)| and sign of B_n."""
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
    cgf_method: str = 'auto',
    complete: bool = True,
    u: float = None
):
    """
    Compute the order‑th derivative of M(t) or imgf(t,u) using Bell polynomials.

    Parameters
    ----------
    t : float
        Evaluation point.
    prior : mitMGFprior
        Prior object providing symbolic and numeric CGF/iMGF functions.
    order : int
        Derivative order (>= 0).
    symbolic_timeout : float, optional
        Maximum time (seconds) allowed for the symbolic CGF derivative
        computation. If exceeded, fall back to JAX. Default 600.
    cgf_method : str, optional
        Method for computing CGF derivatives in the numeric path.
        Options: 'auto', 'jet', 'grad'. Default 'auto'.
    complete : bool, optional
        If True (default), differentiate the complete MGF.
        If False, differentiate the incomplete MGF (imgf) at truncation u.
    u : float, optional
        Truncation point for incomplete MGF (required if complete=False).

    Returns
    -------
    tuple (log_abs_deriv, sign)
    """
    if order < 0:
        raise ValueError("Order must be non‑negative.")

    # ------------------------------------------------------------
    # 1. Select symbolic expression and numeric function
    #    based on `complete`
    # ------------------------------------------------------------
    if complete:
        cgf_expr = prior.cgf_sym
        if cgf_expr is None:
            raise ValueError("Prior does not provide a symbolic CGF (cgf_sym).")
        if callable(cgf_expr):
            cgf_expr = cgf_expr()
        if not isinstance(cgf_expr, sp.Expr):
            raise TypeError("cgf_sym must be a SymPy expression.")

        cgf_func = prior.cgf_jax
        if cgf_func is None:
            raise ValueError("Prior does not provide cgf_jax for numeric path.")

    else:
        if u is None:
            raise ValueError("u must be provided when complete=False.")

        # Symbolic log(imgf) if imgf_sym exists
        if prior.imgf_sym is not None:
            cgf_expr = sp.log(prior.imgf_sym)
        else:
            cgf_expr = None
            print("No imgf_sym; symbolic path will be skipped.")

        # Numeric function: prefer logimgf_jax, else log(imgf_jax)
        if prior.logimgf_jax is not None:
            cgf_func = lambda t_val: prior.logimgf_jax(t_val, u)
        elif prior.imgf_jax is not None:
            cgf_func = lambda t_val: jnp.log(prior.imgf_jax(t_val, u))
        else:
            raise ValueError("Prior does not provide imgf_jax or logimgf_jax for iMGF.")

    # ------------------------------------------------------------
    # 2. Extract the symbol 't' from the expression (if symbolic)
    # ------------------------------------------------------------
    t_sym = None
    if cgf_expr is not None:
        for s in cgf_expr.free_symbols:
            if s.name == 't':
                t_sym = s
                break
        if t_sym is None:
            raise RuntimeError("No symbol 't' found in the CGF expression.")

    params = prior.params or {}

    # ------------------------------------------------------------
    # 3. Decision: force JAX if user explicitly requested jet/grad
    # ------------------------------------------------------------
    if cgf_method.lower() in ('jet', 'grad'):
        use_symbolic = False
        print(f"Decision: Forced JAX numeric path (cgf_method='{cgf_method}')")
    else:
        if cgf_expr is not None:
            decision = suggest_method_integerDeriv(
                cgf_expr, t_sym, order,
                test_order=min(order, 2),
                timeout=1.0,
                return_decision=True
            )
            use_symbolic = decision['recommend_symbolic']
            print(f"Decision: {'Symbolic' if use_symbolic else 'Numeric (JAX)'}")
        else:
            use_symbolic = False
            print("No symbolic expression; using numeric (JAX) path.")

    use_jax = not use_symbolic

    # ------------------------------------------------------------
    # 4. Try symbolic path (with timeout)
    # ------------------------------------------------------------
    if use_symbolic:
        try:
            subs_dict = {}
            for sym in cgf_expr.free_symbols:
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

                deriv_expr = sp.diff(cgf_expr, t_sym, k)
                # Substitute t and numeric params
                if t == 0:
                    val = sp.limit(deriv_expr, t_sym, 0, dir='-').subs(subs_dict).evalf()
                else:
                    val = deriv_expr.subs({t_sym: t}).subs(subs_dict).evalf()

                # ---- CRITICAL FIX: substitute u for incomplete MGF ----
                if not complete and u is not None:
                    from jumufraktiv.symbols import u as u_sym
                    if u_sym in val.free_symbols:
                        val = val.subs(u_sym, u)
                # -------------------------------------------------------

                try:
                    val = float(val)
                except Exception:
                    raise ValueError("Symbolic derivative still contains free symbols; falling back to JAX")

                if abs(val) < sys.float_info.epsilon:
                    kappa_log_abs.append(-float('inf'))
                    kappa_sign.append(1)
                else:
                    kappa_log_abs.append(math.log(abs(val)))
                    kappa_sign.append(1 if val > 0 else -1)

            # Get the zeroth cumulant: CGF value
            if complete:
                cgf_t = prior.cgf(t)
            else:
                cgf_t = float(cgf_func(t))

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

    # ------------------------------------------------------------
    # 5. Numeric (JAX) path
    # ------------------------------------------------------------
    if use_jax:
        print("Using JAX numeric path...")
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
    import numpy as np
    import jumufraktiv.MGFdictionary  # ensures priors are registered
    from jumufraktiv.mitMGFprior_class import mitMGFprior
    from jumufraktiv.symbols import t as t_sym, u as u_sym

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

    # ---- JAX branch for complete MGF (auto mode) ----
    print("\n--- JAX branch for complete MGF (auto mode) ---")
    try:
        log_abs_jax_complete, sign_jax_complete = integerDeriv_numeric_bell(
            t=t_val,
            prior=gamma_prior,
            order=3,
            symbolic_timeout=600.0,
            cgf_method='auto',
            complete=True,
            u=None
        )
        val_jax_complete = sign_jax_complete * math.exp(log_abs_jax_complete)
        print(f"  auto mode: log|val| = {log_abs_jax_complete:.6f}, sign = {sign_jax_complete}")
        print(f"    ordinary: {val_jax_complete:.6e}")
    except Exception as e:
        print(f"  JAX auto for complete failed: {e}")
        
    # ---- JAX branch for complete MGF (forced) ----
    print("\n--- JAX branch for complete MGF (forced) ---")
    for cgf_method in ['jet', 'grad']:
        try:
            log_abs_jax_comp, sign_jax_comp = integerDeriv_numeric_bell(
                t=t_val,
                prior=gamma_prior,
                order=3,
                symbolic_timeout=600.0,
                cgf_method=cgf_method,
                complete=True,
                u=None
            )
            val_jax_comp = sign_jax_comp * math.exp(log_abs_jax_comp)
            print(f"  cgf_method={cgf_method}: log|val| = {log_abs_jax_comp:.6f}, sign = {sign_jax_comp}")
            print(f"    ordinary: {val_jax_comp:.6e}")
        except Exception as e:
            print(f"  cgf_method={cgf_method} failed: {e}")

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
        symbolic_timeout=600.0
    )
    elapsed = time.time() - start

    print(f"Gamma M^{{{order_test}}}({t_small}) with alpha={alpha_small:.1e}, beta={beta_small:.1e}")
    print(f"  log|deriv| = {log_abs:.6e}")
    print(f"  sign       = {sign}")
    print(f"  Time       = {elapsed:.3f} seconds")

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

    # ---- Bell method for incomplete MGF (iMGF) ----
    print("\n" + "=" * 60)
    print("Testing Bell method for incomplete MGF (Gamma prior, truncated at u)")
    print("=" * 60)

    u_val = 2.0
    t_val_imgf = -1.0
    order_imgf = 3

    # ----- Symbolic reference (directly from imgf_sym) -----
    try:
        imgf_sym = gamma_prior.imgf_sym
        if imgf_sym is None:
            raise ValueError("imgf_sym not available")

        # Build substitution dict for hyperparameters
        subs_dict = {}
        for sym in imgf_sym.free_symbols:
            if sym.name in gamma_prior.params:
                subs_dict[sym] = float(gamma_prior.params[sym.name])

        # Differentiate symbolically
        deriv_sym = sp.diff(imgf_sym, t_sym, order_imgf)
        # Evaluate at t and u
        val_sym = deriv_sym.subs({t_sym: t_val_imgf, u_sym: u_val}).subs(subs_dict).evalf()
        if val_sym.free_symbols:
            raise ValueError("Symbolic expression still has free symbols")

        val_ref = float(val_sym)
        log_abs_ref = math.log(abs(val_ref))
        sign_ref = 1 if val_ref > 0 else -1
        print(f"Symbolic reference (ordinary): {val_ref:.6e}")
        print(f"Symbolic reference (log): log|val| = {log_abs_ref:.6f}, sign = {sign_ref}")

    except Exception as e:
        print(f"Symbolic reference failed: {e}")
        val_ref = None

    # ----- Bell method (direct call to integerDeriv_numeric_bell) -----
    try:
        log_abs_bell, sign_bell = integerDeriv_numeric_bell(
            t=t_val_imgf,
            prior=gamma_prior,
            order=order_imgf,
            symbolic_timeout=600.0,
            cgf_method='auto',
            complete=False,
            u=u_val
        )
        val_bell = sign_bell * math.exp(log_abs_bell)
        print(f"Bell (log): log|val| = {log_abs_bell:.6f}, sign = {sign_bell}")
        print(f"Bell (ordinary): {val_bell:.6e}")
        if val_ref is not None:
            print(f"Difference (Bell - symbolic): {abs(val_bell - val_ref):.2e}")
    except Exception as e:
        print(f"Bell method failed: {e}")
            
    # ---- JAX branch for incomplete MGF (forced) ----
    print("\n--- JAX branch for iMGF (forced) ---")
    for cgf_method in ['jet', 'grad']:
        try:
            log_abs_jax, sign_jax = integerDeriv_numeric_bell(
                t=t_val_imgf,
                prior=gamma_prior,
                order=order_imgf,
                symbolic_timeout=600.0,
                cgf_method=cgf_method,
                complete=False,
                u=u_val
            )
            val_jax = sign_jax * math.exp(log_abs_jax)
            print(f"  cgf_method={cgf_method}: log|val| = {log_abs_jax:.6f}, sign = {sign_jax}")
            print(f"    ordinary: {val_jax:.6e}")
            if val_ref is not None:
                print(f"    diff vs symbolic: {abs(val_jax - val_ref):.2e}")
        except Exception as e:
            print(f"  cgf_method={cgf_method} failed: {e}")