"""
numeric_integerDeriv_Bell.py

Compute integer derivatives of MGFs via Bell polynomials.
Uses suggest_method_integerDeriv() with a low-order test to decide between
symbolic (SymPy) and numeric (JAX). Falls back to JAX if symbolic fails
or exceeds a user‑specified timeout.
"""

import math
import time
import numpy as np
import sys
import sympy as sp
import scipy.special as sc
import jax
jax.config.update("jax_enable_x64", True)
from jax import grad
from jax.experimental import jet
import jax.numpy as jnp

from jumufraktiv.mitMGFprior_class import mitMGFprior
from jumufraktiv.logsum import logminus
from jumufraktiv.numeric_symbolic_decision import suggest_method_integerDeriv
from jumufraktiv.symbols import t as t_sym, u as u_sym

# ===== Helper: CGF derivatives (unchanged) =====
def cgf_derivatives_jet(cgf_func, t, order):
    """Compute derivatives of the CGF using JAX's jet (Taylor mode)."""
    if order == 0:
        return [], []

    series_in = ((1.0,) + (0.0,) * (order - 1),)
    _, derivs = jet.jet(cgf_func, (t,), series_in)

    log_abs = []
    signs = []
    eps = sys.float_info.epsilon
    for d in derivs[:order]:
        val = d
        is_zero = jnp.abs(val) < eps

        log_abs_i = jnp.where(
            is_zero,
            -jnp.inf,
            jnp.log(jnp.abs(val))
        )

        sign_i = jnp.where(
            is_zero,
            1,
            jnp.where(val >= 0, 1, -1)
        )

        log_abs.append(log_abs_i)
        signs.append(sign_i)
    return log_abs, signs


def cgf_derivatives_grad(cgf_func, t, order):
    """Compute derivatives of the CGF using nested jax.grad (reverse mode)."""
    log_abs = []
    signs = []
    f = cgf_func
    eps = sys.float_info.epsilon
    for k in range(1, order + 1):
        f = grad(f)
        val = f(t)
        is_zero = jnp.abs(val) < eps
        log_abs_i = jnp.where(
            is_zero,
            -jnp.inf,
            jnp.log(jnp.abs(val))
        )
        sign_i = jnp.where(
            is_zero,
            1,
            jnp.where(val >= 0, 1, -1)
        )
        log_abs.append(log_abs_i)
        signs.append(sign_i)
    return log_abs, signs


def _cgf_derivatives_jax_scalar(cgf_func, t, order, cgf_mode):
    """
    Scalar core: compute cumulant derivatives for a single t.
    Returns (log_abs_list, sign_list) as Python lists.
    """
    if cgf_mode == "jet":
        return cgf_derivatives_jet(cgf_func, t, order)
    elif cgf_mode == "grad":
        return cgf_derivatives_grad(cgf_func, t, order)
    elif cgf_mode == "auto":
        try:
            return cgf_derivatives_jet(cgf_func, t, order)
        except Exception as e:
            msg = str(e).lower()
            if any(key in msg for key in ("jet", "primitive", "not implemented", "igamma")):
                print(f"⚠️ Jet failed ({type(e).__name__}: {e}). Falling back to grad().")
                return cgf_derivatives_grad(cgf_func, t, order)
            raise
    else:
        raise ValueError(f"Unknown cgf_mode='{cgf_mode}'. Expected 'jet', 'grad', or 'auto'.")


def cgf_derivatives_jax(cgf_func, t, order, cgf_mode="auto"):
    """
    Vectorized wrapper.

    Parameters
    ----------
    t : scalar or array-like
        Evaluation point(s).

    Returns
    -------
    log_abs : list or np.ndarray of shape (order, batch) if t is array
    sign : list or np.ndarray of same shape
    """
    # Convert to numpy for shape detection
    t_np = np.asarray(t)
    if t_np.ndim == 0:
        # Scalar
        log_abs, sign = _cgf_derivatives_jax_scalar(cgf_func, t_np, order, cgf_mode)
        return log_abs, sign

    # Array: use vmap
    # We need a function that returns a tuple of two JAX arrays
    # We'll convert the lists from the scalar core to JAX arrays inside the vmap
    def scalar_jax(t_val):
        log_abs_list, sign_list = _cgf_derivatives_jax_scalar(cgf_func, t_val, order, cgf_mode)
        # Convert to JAX arrays
        return jnp.array(log_abs_list), jnp.array(sign_list)

    # vmap over the batch
    vmapped = jax.vmap(scalar_jax)
    log_abs, sign = vmapped(jnp.asarray(t))

    # Transpose: vmap returns shape (batch, order) -> we want (order, batch)
    log_abs = jnp.transpose(log_abs)
    sign = jnp.transpose(sign)

    # Convert to numpy arrays for consistency with rest of Bell method
    return np.asarray(log_abs), np.asarray(sign)


# ======================================================================
# Scalar Bell polynomial (single batch)
# ======================================================================
def bell_polynomial_log(n: int, logv: list, vsign: list):
    """
    Compute log |B_n(v1,...,vn)| and sign of B_n for a single set of cumulants.
    This is the original scalar implementation.

    Parameters
    ----------
    n : int
        Order of the Bell polynomial.
    logv : list of length n
        log absolute values of cumulants κ_1..κ_n.
    vsign : list of length n
        signs of cumulants (1 or -1).

    Returns
    -------
    log_abs : float
        log |B_n|.
    sign : int
        sign of B_n.
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

        # Compute sum of positive and negative terms using logaddexp.reduce
        sum_pos = np.logaddexp.reduce(pos_terms) if pos_terms else -float('inf')
        sum_neg = np.logaddexp.reduce(neg_terms) if neg_terms else -float('inf')

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
            # Both finite: need log(exp(sum_pos) - exp(sum_neg))
            # We must ensure sum_pos >= sum_neg for logminus.
            if sum_pos >= sum_neg:
                logB[i] = logminus(sum_pos, sum_neg)
                signB[i] = 1
            else:
                logB[i] = logminus(sum_neg, sum_pos)
                signB[i] = -1

    return logB[n], signB[n]


# ======================================================================
# Batched Bell polynomial (multiple t values)
# ======================================================================
def bell_polynomial_log_batched(logv, vsign):
    """
    Compute log |B_n| and sign for a batch of cumulant derivative vectors.

    Parameters
    ----------
    logv : np.ndarray, shape (n, batch_size)
        log absolute values of cumulants κ_1..κ_n for each batch element.
    vsign : np.ndarray, shape (n, batch_size)
        signs of cumulants (1 or -1).

    Returns
    -------
    log_abs : np.ndarray, shape (batch_size,)
        log |B_n| for each batch element.
    sign : np.ndarray, shape (batch_size,)
        sign of B_n for each batch element.
    """
    n, batch = logv.shape
    if logv.shape != vsign.shape:
        raise ValueError("logv and vsign must have same shape.")

    # Initialize logB[0..n] and signB[0..n]
    logB = np.full((n + 1, batch), -np.inf, dtype=float)
    signB = np.ones((n + 1, batch), dtype=int)
    logB[0, :] = 0.0

    for i in range(1, n + 1):
        # Accumulate positive and negative contributions across k
        pos_sum = np.full(batch, -np.inf)
        neg_sum = np.full(batch, -np.inf)

        for k in range(1, i + 1):
            log_coeff = math.lgamma(i) - math.lgamma(k) - math.lgamma(i - k + 1)
            # term_log = log_coeff + logv[k-1] + logB[i-k]
            term_log = log_coeff + logv[k-1, :] + logB[i-k, :]
            term_sign = vsign[k-1, :] * signB[i-k, :]

            # Split by sign
            pos_mask = term_sign > 0
            neg_mask = term_sign < 0

            # Update pos_sum and neg_sum element-wise using logaddexp
            pos_sum = np.where(pos_mask, np.logaddexp(pos_sum, term_log), pos_sum)
            neg_sum = np.where(neg_mask, np.logaddexp(neg_sum, term_log), neg_sum)

        # Combine pos_sum and neg_sum to get logB[i] and signB[i]
        both_inf = np.isneginf(pos_sum) & np.isneginf(neg_sum)
        pos_only = np.isneginf(neg_sum) & ~np.isneginf(pos_sum)
        neg_only = np.isneginf(pos_sum) & ~np.isneginf(neg_sum)
        both_finite = ~np.isneginf(pos_sum) & ~np.isneginf(neg_sum)

        logB[i, both_inf] = -np.inf
        signB[i, both_inf] = 1

        logB[i, pos_only] = pos_sum[pos_only]
        signB[i, pos_only] = 1

        logB[i, neg_only] = neg_sum[neg_only]
        signB[i, neg_only] = -1

        if np.any(both_finite):
            a = pos_sum[both_finite]
            b = neg_sum[both_finite]
            # Ensure a >= b for logminus; if not, swap
            swap = a < b
            a_swapped = np.where(swap, b, a)
            b_swapped = np.where(swap, a, b)
            # Compute log(exp(a) - exp(b))
            log_diff = logminus(a_swapped, b_swapped)
            sign_val = np.where(swap, -1, 1)
            logB[i, both_finite] = log_diff
            signB[i, both_finite] = sign_val

    return logB[n, :], signB[n, :]


# ===== main function =====
def integerDeriv_numeric_bell(
    t,
    prior: mitMGFprior,
    order: int,
    symbolic_timeout: float = 600.0,
    cgf_mode: str = 'auto',
    complete: bool = True,
    u: float | np.ndarray | None = None,
):
    """
    Compute the order‑th derivative of M(t) or imgf(t,u) using Bell polynomials.

    The evaluation point is:
        - complete MGF: (t)
        - incomplete MGF: (t, u)
    If either t or u is array‑like, they are broadcast to a common shape and the
    computation is vectorised over that batch (tuple‑vectorisation principle).

    Parameters
    ----------
    t : scalar or array-like
        Evaluation point(s) for t.
    prior : mitMGFprior
        Prior object.
    order : int
        Derivative order.
    symbolic_timeout : float, optional
        Timeout for symbolic computation.
    cgf_mode : str, optional
        'auto', 'jet', 'grad', or 'symbolic'.
    complete : bool, optional
        If True, complete MGF; else incomplete.
    u : scalar or array-like, optional
        Upper limit(s) for the incomplete MGF.  Must be broadcastable with t
        when both are arrays.

    Returns
    -------
    If t and u are scalar:
        (log_abs, sign)  # Python floats
    If either is array-like:
        (log_abs_array, sign_array)  # np.ndarray with broadcasted shape
    """
    if order < 0:
        raise ValueError("Order must be non‑negative.")

    valid_modes = {"auto", "symbolic", "jet", "grad"}
    if cgf_mode.lower() not in valid_modes:
        raise ValueError(f"Invalid cgf_mode. Must be one of {valid_modes}.")

    # ------------------------------------------------------------
    # 1. Broadcast t and u to a common batch shape
    # ------------------------------------------------------------
    t_arr = np.asarray(t)
    if complete:
        if u is not None:
            raise ValueError("u must be None when complete=True")
        scalar_input = t_arr.ndim == 0
        if scalar_input:
            batch_shape = ()
            t_flat = np.array([float(t_arr)])
            u_flat = None
            n_points = 1
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

    # ------------------------------------------------------------
    # 2. Select symbolic expression and numeric function
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
        if prior.imgf_sym is not None:
            cgf_expr = sp.log(prior.imgf_sym)
        else:
            cgf_expr = None
            print("No imgf_sym; symbolic path will be skipped.")
        if prior.logimgf_jax is not None:
            cgf_func = prior.logimgf_jax
        elif prior.imgf_jax is not None:
            cgf_func = lambda t_val, u_val: jnp.log(prior.imgf_jax(t_val, u_val))
        else:
            raise ValueError("Prior does not provide imgf_jax or logimgf_jax for iMGF.")

    # ---- Handle order 0 ----
    if order == 0:
        if complete:
            cgf_vals = np.array([cgf_func(t_flat[i]) for i in range(n_points)])
        else:
            cgf_vals = np.array([cgf_func(t_flat[i], u_flat[i]) for i in range(n_points)])
        cgf_vals = cgf_vals.reshape(batch_shape)
        if scalar_input:
            return float(cgf_vals.item()), 1
        else:
            return cgf_vals, np.ones_like(cgf_vals, dtype=int)

    # ------------------------------------------------------------
    # 3. Extract symbol 't' (and 'u' for symbolic)
    # ------------------------------------------------------------
    t_sym = None
    u_sym = None
    if cgf_expr is not None:
        for s in cgf_expr.free_symbols:
            if s.name == 't':
                t_sym = s
            elif s.name == 'u':
                u_sym = s
        if t_sym is None:
            raise RuntimeError("No symbol 't' found in the CGF expression.")

    params = prior.params or {}

    # ------------------------------------------------------------
    # 4. Decision: symbolic vs numeric
    # ------------------------------------------------------------
    if cgf_mode.lower() == "symbolic":
        use_symbolic = True
    elif cgf_mode.lower() in ('jet', 'grad'):
        use_symbolic = False
        print(f"Decision: Forced JAX numeric path (cgf_mode='{cgf_mode}')")
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

    # ------------------------------------------------------------
    # 5. Symbolic path (iterate over points)
    # ------------------------------------------------------------
    if use_symbolic:
        # Pre‑substitute hyperparameters once
        subs_dict = {}
        for sym in cgf_expr.free_symbols:
            if sym.name in params:
                subs_dict[sym] = float(params[sym.name])
        cgf_expr_num = cgf_expr.subs(subs_dict)

        # Build substitution dictionaries for each point once
        subs_list = []
        for idx in range(n_points):
            subs_local = {t_sym: t_flat[idx]}
            if not complete:
                subs_local[u_sym] = u_flat[idx]
            subs_list.append(subs_local)

        logv_list = []
        vsign_list = []
        start_time = time.time()

        for k in range(1, order + 1):
            if time.time() - start_time > symbolic_timeout:
                raise TimeoutError(
                    f"Symbolic computation exceeded {symbolic_timeout:.1f} seconds."
                )

            deriv_expr = sp.diff(cgf_expr_num, t_sym, k)

            vals_k = np.zeros(n_points)
            for idx in range(n_points):
                val_sub = deriv_expr.subs(subs_list[idx]).evalf()
                if val_sub.free_symbols:
                    raise ValueError(f"Free symbols remain for point {idx}: {val_sub.free_symbols}")
                vals_k[idx] = float(val_sub)

            abs_vals = np.abs(vals_k)
            log_abs_k = np.where(abs_vals > sys.float_info.epsilon,
                                 np.log(abs_vals),
                                 -np.inf)
            sign_k = np.where(vals_k >= 0, 1, -1)
            logv_list.append(log_abs_k)
            vsign_list.append(sign_k)

        logv = np.stack(logv_list, axis=0)
        vsign = np.stack(vsign_list, axis=0)

        # Evaluate CGF using pre‑substituted expression
        cgf_t = np.zeros(n_points)
        for idx in range(n_points):
            cgf_val = cgf_expr_num.subs(subs_list[idx]).evalf()
            if cgf_val.free_symbols:
                raise ValueError(f"Free symbols remain for CGF at point {idx}")
            cgf_t[idx] = float(cgf_val)

        log_abs_B, sign_B = bell_polynomial_log_batched(logv, vsign)
        log_abs_deriv = cgf_t + log_abs_B
        log_abs_deriv = log_abs_deriv.reshape(batch_shape)
        sign_B = sign_B.reshape(batch_shape)

        if scalar_input:
            return float(log_abs_deriv.item()), int(sign_B.item())
        else:
            return log_abs_deriv, sign_B

    # ------------------------------------------------------------
    # 6. Numeric (JAX) path – tuple‑vectorised, no Python loops
    # ------------------------------------------------------------
    if not use_symbolic:
        print("Using JAX numeric path (tuple‑vectorised)...")

        # Prepare JAX arrays
        t_vals = jnp.asarray(t_flat)
        if complete:
            # Scalar cumulants for complete MGF (unary)
            def scalar_cumulants(t_val, _):
                log_abs, sign = _cgf_derivatives_jax_scalar(
                    cgf_func, t_val, order, cgf_mode
                )
                return jnp.array(log_abs), jnp.array(sign)
            u_vals = jnp.zeros_like(t_vals)   # dummy, unused
        else:
            u_vals = jnp.asarray(u_flat)
            # Scalar cumulants for incomplete MGF (bind u)
            def scalar_cumulants(t_val, u_val):
                unary = lambda x: cgf_func(x, u_val)
                log_abs, sign = _cgf_derivatives_jax_scalar(
                    unary, t_val, order, cgf_mode
                )
                return jnp.array(log_abs), jnp.array(sign)

        # Vectorise over points
        vmapped = jax.vmap(scalar_cumulants)
        logv, vsign = vmapped(t_vals, u_vals)

        # Transpose to (order, batch)
        logv = jnp.transpose(logv)
        vsign = jnp.transpose(vsign)

        # Batched Bell
        log_abs_B, sign_B = bell_polynomial_log_batched(np.asarray(logv), np.asarray(vsign))

        # Compute CGF values via vmap (no Python loop)
        if complete:
            cgf_vec = jax.vmap(cgf_func)
            cgf_t = cgf_vec(t_vals)
        else:
            cgf_vec = jax.vmap(cgf_func)
            cgf_t = cgf_vec(t_vals, u_vals)

        cgf_t = np.asarray(cgf_t)

        log_abs_deriv = cgf_t + log_abs_B
        log_abs_deriv = log_abs_deriv.reshape(batch_shape)
        sign_B = sign_B.reshape(batch_shape)

        if scalar_input:
            return float(log_abs_deriv.item()), int(sign_B.item())
        else:
            return log_abs_deriv, sign_B

    raise RuntimeError("No path executed.")


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
            cgf_mode='auto',
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
    for cgf_mode in ['jet', 'grad']:
        try:
            log_abs_jax_comp, sign_jax_comp = integerDeriv_numeric_bell(
                t=t_val,
                prior=gamma_prior,
                order=3,
                symbolic_timeout=600.0,
                cgf_mode=cgf_mode,
                complete=True,
                u=None
            )
            val_jax_comp = sign_jax_comp * math.exp(log_abs_jax_comp)
            print(f"  cgf_mode={cgf_mode}: log|val| = {log_abs_jax_comp:.6f}, sign = {sign_jax_comp}")
            print(f"    ordinary: {val_jax_comp:.6e}")
        except Exception as e:
            print(f"  cgf_mode={cgf_mode} failed: {e}")

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
            cgf_mode='auto',
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
    for cgf_mode in ['jet', 'grad']:
        try:
            log_abs_jax, sign_jax = integerDeriv_numeric_bell(
                t=t_val_imgf,
                prior=gamma_prior,
                order=order_imgf,
                symbolic_timeout=600.0,
                cgf_mode=cgf_mode,
                complete=False,
                u=u_val
            )
            val_jax = sign_jax * math.exp(log_abs_jax)
            print(f"  cgf_mode={cgf_mode}: log|val| = {log_abs_jax:.6f}, sign = {sign_jax}")
            print(f"    ordinary: {val_jax:.6e}")
            if val_ref is not None:
                print(f"    diff vs symbolic: {abs(val_jax - val_ref):.2e}")
        except Exception as e:
            print(f"  cgf_mode={cgf_mode} failed: {e}")