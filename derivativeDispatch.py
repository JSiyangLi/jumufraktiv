"""
derivativeDispatch.py

Unified interface to compute integer derivatives of MGFs using symbolic, Bell‑polynomial,
or JAX methods.

Imports:
    - integerDeriv_symbolic from symbolic_integerDeriv.py
    - integerDeriv_numeric_bell from numeric_integerDeriv_Bell.py
    - integerDeriv_numeric_jax from numeric_integerDeriv_JAX.py

Function:
    mgfDerivative_integer(order, prior, method='symbolic', t=nan, params=None,
                          simplify=False, log=True)
"""

import math
import warnings
import sympy as sp
import numpy as np
from jumufraktiv.symbolic_integerDeriv import integerDeriv_symbolic
from jumufraktiv.numeric_integerDeriv_Bell import integerDeriv_numeric_bell
from jumufraktiv.numeric_integerDeriv_JAX import integerDeriv_numeric_jax

from jumufraktiv.symbols import t as t_sym, u as u_sym   # <-- import canonical u

def mgfDerivative_integer(
    order: int | sp.Expr,
    prior,
    method: str = "symbolic",
    t: float | np.ndarray | list | None = None,
    u: float | np.ndarray | list | None = None,
    simplify: bool = False,
    log: bool = True,
    complete: bool = True,
    symbolic_timeout: float = 600.0,
    cgf_method: str = "auto",
):
    """
    Compute an integer-order derivative of a prior MGF.

    Parameters
    ----------
    order : int | sp.Expr
        Non-negative derivative order.
    prior : mitMGFprior
        Prior object containing symbolic and/or backend MGF/PDF representations.
    method : {"symbolic", "bell", "jax"}, optional
        Derivative backend.
    t : float or array-like, optional
        Evaluation point(s) for t.
    u : float or array-like, optional
        Truncation point(s) for incomplete MGF (used when complete=False).
        For symbolic method, substitutes the canonical 'u' symbol. If array‑like,
        it is broadcast with t to form a batch of evaluation points (t, u).
    simplify : bool, optional
        Whether to simplify symbolic derivatives.
    complete : bool, optional
        If True (default), differentiate the complete MGF.
        If False, differentiate the incomplete MGF.
    log : bool, optional
        If True, numeric methods return (log_abs, sign).
        Otherwise return the ordinary derivative as float.
    symbolic_timeout, cgf_method : passed to Bell backend.

    Returns
    -------
    If t and u are scalar (or t is scalar and u is None) and evaluation is numeric:
        If log=True: (log_abs, sign)
        If log=False: float
    If t or u is array-like and evaluation is numeric:
        If log=True: (log_abs_array, sign_array) with broadcasted shape
        If log=False: array with broadcasted shape
    If t is None or symbolic evaluation leaves free symbols:
        sympy.Expr (or list of expressions if multiple points)
    """
    method = method.lower()
    if method not in {"symbolic", "bell", "jax"}:
        raise ValueError("method must be one of {'symbolic','bell','jax'}.")

    # ---------------------------------------------------------
    # Symbolic differentiation
    # ---------------------------------------------------------
    if method == "symbolic":
        expr = integerDeriv_symbolic(
            order=order,
            prior=prior,
            simplify=simplify,
            complete=complete
        )

        # If no t provided, return symbolic expression
        if t is None:
            return expr

        # ---- Broadcast t and u to a common shape ----
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

        # ---- Prepare result arrays ----
        results_log_abs = np.full(n_points, -np.inf, dtype=float)
        results_sign = np.ones(n_points, dtype=int)
        # Also store symbolic results if any free symbols remain
        results_expr = [None] * n_points

        # ---- Build substitution dictionaries for each point ----
        subs_list = []
        for idx in range(n_points):
            subs_local = {t_sym: t_flat[idx]}
            if not complete:
                subs_local[u_sym] = u_flat[idx]
            subs_list.append(subs_local)

        # ---- Evaluate at each point ----
        for idx in range(n_points):
            val_expr = expr.subs(subs_list[idx])
            if val_expr.free_symbols:
                # If any free symbols remain, we keep the expression
                results_expr[idx] = val_expr
            else:
                val = float(val_expr.evalf())
                if abs(val) < 1e-300:
                    results_log_abs[idx] = -np.inf
                    results_sign[idx] = 1
                else:
                    results_log_abs[idx] = np.log(abs(val))
                    results_sign[idx] = 1 if val > 0 else -1

        # ---- Decide return type ----
        # If all results are numeric, return numeric arrays (or scalars)
        if all(r is None for r in results_expr):
            # Reshape to batch_shape
            log_abs_reshaped = results_log_abs.reshape(batch_shape)
            sign_reshaped = results_sign.reshape(batch_shape)
            if scalar_input:
                if log:
                    return float(log_abs_reshaped.item()), int(sign_reshaped.item())
                else:
                    return float(sign_reshaped.item() * np.exp(log_abs_reshaped.item()))
            else:
                if log:
                    return log_abs_reshaped, sign_reshaped
                else:
                    return sign_reshaped * np.exp(log_abs_reshaped)
        else:
            # If any free symbols remain, we cannot return a uniform numeric array.
            # For scalar input, return the expression (first element)
            # For array input, return a list of expressions (some may be numeric, some symbolic)
            # We'll return a list with mixed types.
            if scalar_input:
                return results_expr[0] if log else sp.exp(results_expr[0])
            else:
                # Return list of results (could be mixed numeric/expr)
                # But we want to respect log flag: if log=True, return log_abs or expression; else exponentiate.
                if log:
                    # For numeric points, we have log_abs; for symbolic, we have expr.
                    # We'll return a list of the appropriate items.
                    out = []
                    for idx in range(n_points):
                        if results_expr[idx] is None:
                            out.append(results_log_abs[idx])
                        else:
                            out.append(results_expr[idx])
                    return np.array(out).reshape(batch_shape) if all(isinstance(x, (int, float)) for x in out) else out
                else:
                    out = []
                    for idx in range(n_points):
                        if results_expr[idx] is None:
                            val = results_sign[idx] * np.exp(results_log_abs[idx])
                            out.append(val)
                        else:
                            out.append(sp.exp(results_expr[idx]))
                    return np.array(out).reshape(batch_shape) if all(isinstance(x, (int, float)) for x in out) else out

    # ---------------------------------------------------------
    # Bell polynomial backend (vectorized)
    # ---------------------------------------------------------
    if method == "bell":
        if t is None:
            raise ValueError("t must be supplied for method='bell'.")
        log_abs, sign = integerDeriv_numeric_bell(
            prior=prior,
            t=t,
            order=order,
            symbolic_timeout=symbolic_timeout,
            cgf_mode=cgf_method,
            complete=complete,
            u=u,
        )
        if log:
            return log_abs, sign
        return 0.0 if log_abs == -math.inf else sign * np.exp(log_abs)

    # ---------------------------------------------------------
    # JAX backend (vectorized)
    # ---------------------------------------------------------
    if method == "jax":
        if t is None:
            raise ValueError("t must be supplied for method='jax'.")
        log_abs, sign = integerDeriv_numeric_jax(
            prior=prior,
            t=t,
            order=order,
            complete=complete,
            u=u,
        )
        if log:
            return log_abs, sign
        return 0.0 if log_abs == -math.inf else sign * np.exp(log_abs)


def mgfDerivative_fractional(
    order: float,
    prior,
    method: str = "scipy",
    t: float | np.ndarray | list | None = None,
    simplify: bool = False,
    complete: bool = True,
    log: bool = True,
    integerDeriv_method: str = "symbolic",
    u: float | np.ndarray | list | None = None,
    **kwargs
):
    """
    Unified interface for fractional derivatives of the MGF.
    Supports tuple‑vectorisation: t and u are broadcast to a common shape.

    Parameters
    ----------
    order : float
        Fractional order (positive).
    prior : mitMGFprior
        Prior object.
    method : {'scipy', 'mpmath', 'symbolic'}, default 'scipy'
        Computation backend.
    t : float or array-like, optional
        Evaluation point(s) for t (required for numeric methods).
    simplify : bool, default False
        Simplify symbolic expressions (method='symbolic').
    complete : bool, default True
        If True, differentiate complete MGF; if False, incomplete MGF.
    log : bool, default True
        If True, numeric output is (log_abs, sign); else ordinary float.
    integerDeriv_method : str, default 'symbolic'
        Method for integer derivatives inside integrators.
    u : float or array-like, optional
        Truncation point(s) for incomplete MGF (used when complete=False).
        If array‑like, broadcast with t to form evaluation points (t, u).
    **kwargs : passed to underlying functions.

    Returns
    -------
    sympy.Expr, (log_abs, sign), or float
        If t and u are scalar, returns sympy.Expr or scalar.
        If either is array, returns arrays with the broadcasted shape.
    """
    # ---- Symbolic path ----
    if method.lower() == "symbolic":
        warnings.warn(
            "Symbolic fractional derivatives can be very slow. "
            "Consider 'scipy' or 'mpmath' instead."
        )
        from symbolic_fractionalDeriv import fractionalDeriv_symbolic

        expr = fractionalDeriv_symbolic(
            order=order,
            prior=prior,
            simplify=simplify,
            complete=complete,
            **kwargs
        )
        if expr is None:
            return None

        # If t is not provided, return symbolic expression
        if t is None:
            return expr

        # ---- Broadcast t and u to common shape ----
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
        if log:
            log_abs_vals = np.full(n_points, -np.inf, dtype=float)
            sign_vals = np.ones(n_points, dtype=int)
        else:
            val_vals = np.zeros(n_points)

        # ---- Build substitution dictionaries for each point ----
        subs_list = []
        for idx in range(n_points):
            subs_local = {t_sym: t_flat[idx]}
            if not complete:
                subs_local[u_sym] = u_flat[idx]
            subs_list.append(subs_local)

        # ---- Evaluate for each point ----
        for idx, subs_local in enumerate(subs_list):
            expr_sub = expr.subs(subs_local)
            # If free symbols remain, we cannot proceed for vector input
            if expr_sub.free_symbols:
                if n_points > 1:
                    raise ValueError(
                        f"Symbolic expression still has free symbols at point {idx}: {expr_sub.free_symbols}. "
                        "Cannot vectorize symbolic evaluation."
                    )
                # Scalar input: return expression
                return expr_sub

            value = float(expr_sub.evalf())
            if log:
                if abs(value) < 1e-300:
                    log_abs_vals[idx] = -np.inf
                    sign_vals[idx] = 1
                else:
                    log_abs_vals[idx] = math.log(abs(value))
                    sign_vals[idx] = 1 if value > 0 else -1
            else:
                val_vals[idx] = value

        # ---- Reshape to broadcasted shape ----
        if log:
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

    # ---- Numeric paths require t ----
    if t is None or (isinstance(t, float) and math.isnan(t)):
        raise ValueError(f"For method '{method}', t must be provided.")

    # ---- scipy ----
    if method.lower() == "scipy":
        from jumufraktiv.numeric_fractionalDeriv_scipy import fractionalDeriv_numeric_scipy
        return fractionalDeriv_numeric_scipy(
            order=order,
            prior=prior,
            t=t,
            method=integerDeriv_method,
            simplify=simplify,
            return_log=log,
            complete=complete,
            u=u,
            **kwargs
        )

    # ---- mpmath ----
    if method.lower() == "mpmath":
        from jumufraktiv.numeric_fractionalDeriv_mpmath import fractionalDeriv_numeric_mpmath
        return fractionalDeriv_numeric_mpmath(
            order=order,
            prior=prior,
            t=t,
            method=integerDeriv_method,
            simplify=simplify,
            return_log=log,
            complete=complete,
            u=u,
            **kwargs
        )

    raise ValueError(f"Unknown method: '{method}'. Choose 'scipy', 'mpmath', or 'symbolic'.")

def mgfDerivative(
    order: float | np.ndarray | list | sp.Basic,
    prior,
    method: str = "auto",
    t: float | np.ndarray | list | None = None,
    simplify: bool = False,
    complete: bool = True,
    log: bool = True,
    integer_method: str = "symbolic",
    use_interpolation: bool = True,
    d_vec: tuple = (0.8, 0.9, 0.95),
    int_tol: float = 1e-12,
    u: float | np.ndarray | list | None = None,
    **kwargs
):
    """
    Unified wrapper for integer or fractional derivatives of the MGF.
    Supports tuple‑vectorisation: t and u are broadcast to a common shape.

    If `order` is array-like, each order is processed independently, and the
    results are stacked along a new first axis.

    Parameters
    ----------
    order : float, array-like, or sp.Basic
        Derivative order(s). If array-like, each element is processed separately.
    prior : mitMGFprior
        Prior object.
    method : str, optional
        Derivative backend.
    t : float or array-like, optional
        Evaluation point(s) for t.
    simplify : bool, optional
        If True, simplify symbolic expressions.
    complete : bool, optional
        If True, differentiate complete MGF; else incomplete.
    log : bool, optional
        If True, return (log_abs, sign) for numeric outputs.
    integer_method : str, optional
        For fractional orders: method for integer derivatives.
    use_interpolation : bool, optional
        If True, use cubic interpolation for near‑integer orders.
    d_vec : tuple, optional
        Complements of deviations for interpolation.
    int_tol : float, optional
        Tolerance for treating order as integer.
    u : float or array-like, optional
        Truncation point(s) for incomplete MGF (used when complete=False).
        If array‑like, broadcast with t to form evaluation points (t, u).
    **kwargs : passed to underlying functions.

    Returns
    -------
    If order is scalar:
        - If log=True: (log_abs, sign) where log_abs and sign are scalars or arrays
          (depending on the broadcasted shape of t and u).
        - If log=False: scalar or array.
    If order is array-like:
        - If log=True: (log_abs_array, sign_array) where the arrays have shape
          (len(order), broadcasted_shape).
        - If log=False: array of shape (len(order), broadcasted_shape).
    """
    # ---- Validate d_vec (independent of order) ----
    if len(d_vec) != 3:
        raise ValueError("d_vec must have exactly 3 elements.")
    if any(d >= 1 for d in d_vec):
        raise ValueError("All elements of d_vec must be < 1.")

    # ---- Dispatch for array-like order ----
    if hasattr(order, '__len__') and not isinstance(order, (str, bytes, sp.Basic)):
        order_arr = np.asarray(order)
        # Broadcast all components of one derivative request
        if complete:
            order_arr, t_arr = np.broadcast_arrays(order_arr, t)
            u_arr = [None] * order_arr.size
        else:
            order_arr, t_arr, u_arr = np.broadcast_arrays(order_arr, t, u)

        results = []
        for o, tt, uu in zip(order_arr.flat, t_arr.flat, np.asarray(u_arr).flat):
            results.append(
                mgfDerivative(
                    order=int(o),
                    prior=prior,
                    method=method,
                    t=float(tt),
                    simplify=simplify,
                    complete=complete,
                    log=log,
                    integer_method=integer_method,
                    use_interpolation=use_interpolation,
                    d_vec=d_vec,
                    int_tol=int_tol,
                    u=None if complete else float(uu),
                    **kwargs,
                )
            )

        # Restructure results based on log flag
        if log:
            log_abs_vals = [r[0] for r in results]
            sign_vals = [r[1] for r in results]
            if np.isscalar(log_abs_vals[0]):
                return np.array(log_abs_vals), np.array(sign_vals)
            else:
                # Stack along new axis (axis=0)
                return np.stack(log_abs_vals, axis=0), np.stack(sign_vals, axis=0)
        else:
            if np.isscalar(results[0]):
                return np.array(results)
            else:
                return np.stack(results, axis=0)

    # ---- Continue with scalar order (original logic) ----
    cgf_method = kwargs.pop('cgf_method', 'auto')
    symbolic_timeout = kwargs.pop('symbolic_timeout', 600.0)

    # Determine order type
    if isinstance(order, sp.Basic):
        order_type = "symbolic"
    else:
        if abs(order - round(order)) < int_tol:
            order_type = "integer"
        else:
            order_type = "fractional"

    # Resolve 'auto' method
    if method.lower() == 'auto':
        if order_type == "symbolic":
            method = 'symbolic'
        elif order_type == "integer":
            method = 'symbolic'
        else:
            method = 'scipy'

    # Dispatch
    if order_type == "symbolic":
        if method.lower() not in {'auto', 'symbolic'}:
            raise ValueError(f"Invalid method '{method}' for symbolic order. Only 'symbolic' is allowed.")
        warnings.warn(
            "Derivative order contains symbolic variables. "
            "Using integerDeriv_symbolic() as a formal symbolic derivative. "
            "The resulting expression is often the analytic continuation "
            "to non-integer orders, but this is not guaranteed.",
            UserWarning,
        )
        return mgfDerivative_integer(
            order=order,
            prior=prior,
            method="symbolic",
            t=t,
            simplify=simplify,
            complete=complete,
            log=log,
            u=u,
            symbolic_timeout=symbolic_timeout,
            cgf_method=cgf_method,
        )

    elif order_type == "integer":
        int_order = int(round(order))
        valid_int_methods = {'symbolic', 'bell', 'jax'}
        if method.lower() not in valid_int_methods:
            raise ValueError(f"Invalid method '{method}' for integer order. Choose from {valid_int_methods}.")
        return mgfDerivative_integer(
            order=int_order,
            prior=prior,
            method=method,
            t=t,
            simplify=simplify,
            complete=complete,
            log=log,
            u=u,
            symbolic_timeout=symbolic_timeout,
            cgf_method=cgf_method,
        )

    else:
        # Fractional order
        valid_frac_methods = {'scipy', 'mpmath', 'symbolic'}
        if method.lower() in {'jax', 'bell'}:
            print(f"Note: For fractional order, 'method' should be one of {valid_frac_methods}. "
                  f"Interpreting 'method' as 'integer_method' and using default fractional method 'scipy'.")
            integer_method = method
            method = 'scipy'
        elif method.lower() not in valid_frac_methods:
            raise ValueError(f"Invalid method '{method}' for fractional order. Choose from {valid_frac_methods}.")

        # Validate integer_method
        valid_int_methods = {'symbolic', 'jax', 'bell'}
        if integer_method.lower() not in valid_int_methods:
            raise ValueError(f"Invalid integer_method '{integer_method}'. Choose from {valid_int_methods}.")

        # Check interpolation
        n = int(np.ceil(order))
        if use_interpolation and order > n - (1.0 - max(d_vec)):
            if method.lower() != 'scipy':
                print(f"⚠️ Interpolation triggered for order {order}. "
                      f"Overriding method '{method}' with 'scipy'.")
            scipy_keys = {'epsabs', 'epsrel', 'limit', 'initial_L', 'max_L', 'tol', 'use_tan'}
            scipy_kwargs = {k: v for k, v in kwargs.items() if k in scipy_keys}
            try:
                from numeric_fractionalDeriv_interpolation import fractionalDeriv_interpolated
            except ImportError as e:
                raise ImportError("Could not import numeric_fractionalDeriv_interpolation") from e
            return fractionalDeriv_interpolated(
                order=order,
                prior=prior,
                t=t,
                d_vec=d_vec,
                return_log=log,
                integer_method=integer_method,
                complete=complete,
                u=u,
                **scipy_kwargs
            )

        # Standard fractional method
        return mgfDerivative_fractional(
            order=order,
            prior=prior,
            method=method,
            t=t,
            simplify=simplify,
            complete=complete,
            log=log,
            integerDeriv_method=integer_method,
            u=u,
            **kwargs
        )


if __name__ == "__main__":
    import math
    import numpy as np
    import sympy as sp
    import jumufraktiv.MGFdictionary  # registers priors
    from jumufraktiv.mitMGFprior_class import mitMGFprior
    from scipy.special import gammainc, gamma

    # ------------------------------------------------------------------
    # Exact analytical reference for incomplete Gamma MGF derivatives
    # ------------------------------------------------------------------
    def exact_imgf_deriv(order, t, alpha, beta, u):
        z = beta - t
        lower_gamma_val = gamma(alpha + order) * gammainc(alpha + order, z * u)
        return (beta**alpha / gamma(alpha)) * (z**(-(alpha + order))) * lower_gamma_val

    # ---- Create Gamma priors ----
    gamma_prior = mitMGFprior.from_registry(
        "gamma", params={"alpha": 2.0, "beta": 3.0}
    )
    gamma_prior_exp = mitMGFprior.from_registry(
        "gamma", params={"alpha": 1.0, "beta": 0.9}
    )

    # ============================================================
    # 1. Sanity check: symbolic vs analytical (iMGF)
    # ============================================================
    print("=" * 60)
    print("Sanity check: symbolic vs. analytical (iMGF)")
    print("=" * 60)
    t_chk, u_chk, alpha_chk, beta_chk, order_chk = -1.0, 2.0, 2.0, 3.0, 2
    val_sym = mgfDerivative(
        order=order_chk, prior=gamma_prior, method='symbolic',
        t=t_chk, complete=False, u=u_chk, log=False
    )
    val_ref = exact_imgf_deriv(order_chk, t_chk, alpha_chk, beta_chk, u_chk)
    print(f"Symbolic: {val_sym:.6e}, Analytical: {val_ref:.6e}")
    print(f"Difference: {abs(val_sym - val_ref):.2e}")
    print("✅" if abs(val_sym - val_ref) < 1e-12 else "⚠️ Discrepancy")

    # ============================================================
    # 2. Integer derivatives (complete MGF)
    # ============================================================
    print("\n" + "=" * 60)
    print("Integer derivatives (complete MGF)")
    print("=" * 60)
    t_int = -1.0
    for method in ['symbolic', 'bell', 'jax']:
        log_abs, sign = mgfDerivative_integer(
            2, gamma_prior, method=method, t=t_int, log=True
        )
        print(f"{method:>8}: log|deriv| = {log_abs:.6f}, sign = {sign:+d}")

    # ============================================================
    # 3. Fractional derivatives (complete MGF)
    # ============================================================
    print("\n" + "=" * 60)
    print("Fractional derivatives (complete MGF)")
    print("=" * 60)
    frac_order, t_frac = 1.99, -1.0
    for backend in ['scipy', 'mpmath']:
        try:
            if backend == 'scipy':
                log_abs, sign = mgfDerivative_fractional(
                    order=frac_order, prior=gamma_prior, method=backend,
                    t=t_frac, integerDeriv_method='symbolic', epsrel=1e-10, log=True
                )
            else:  # mpmath
                log_abs, sign = mgfDerivative_fractional(
                    order=frac_order, prior=gamma_prior, method=backend,
                    t=t_frac, integerDeriv_method='symbolic', dps=60, tol=1e-10, log=True
                )
            print(f"{backend:>6}: log|deriv| = {log_abs:.6f}, sign = {sign:+d}")
        except Exception as e:
            print(f"{backend:>6}: failed ({e})")

    # ============================================================
    # 4. Interpolation test (exponential prior)
    # ============================================================
    print("\n" + "=" * 60)
    print("Interpolation test (order 1.999, exponential prior)")
    print("=" * 60)
    t_interp, order_interp = -1.0, 1.999
    log_abs_interp, sign_interp = mgfDerivative(
        order=order_interp, prior=gamma_prior_exp, method='scipy',
        t=t_interp, log=True, integer_method='symbolic',
        use_interpolation=True, d_vec=(0.8, 0.9, 0.95), epsrel=1e-10
    )
    lambda_exp = gamma_prior_exp.params['beta']
    log_analytic = math.log(lambda_exp) + math.lgamma(order_interp + 1) - (order_interp + 1) * math.log(lambda_exp - t_interp)
    print(f"Interpolated: {log_abs_interp:.6f}, Analytic: {log_analytic:.6f}")
    print(f"Difference: {log_abs_interp - log_analytic:.2e}")

    # ============================================================
    # 5. Incomplete MGF (iMGF) derivative tests (integer and fractional)
    # ============================================================
    print("\n" + "=" * 60)
    print("Incomplete MGF (iMGF) derivative tests")
    print("=" * 60)
    u_val, t_val_i, alpha_val, beta_val = 2.0, -1.0, 2.0, 3.0

    # Integer order (exact reference)
    order_int = 2
    ref_int = exact_imgf_deriv(order_int, t_val_i, alpha_val, beta_val, u_val)
    print(f"Integer order {order_int}: exact = {ref_int:.6e}")
    for method in ['bell', 'jax']:
        log_abs, sign = mgfDerivative_integer(
            order_int, gamma_prior, method=method,
            t=t_val_i, complete=False, u=u_val, log=True
        )
        val = sign * np.exp(log_abs)
        print(f"  {method:>4}: {val:.6e} (diff {abs(val - ref_int):.2e})")

    # Fractional order (non‑near‑integer)
    order_frac = 1.5
    ref_frac = exact_imgf_deriv(order_frac, t_val_i, alpha_val, beta_val, u_val)
    print(f"\nFractional order {order_frac}: exact = {ref_frac:.6e}")
    for backend, extra_kwargs in [('scipy', {'epsrel': 1e-12}), ('mpmath', {'dps': 60, 'tol': 1e-12})]:
        try:
            log_abs, sign = mgfDerivative(
                order=order_frac, prior=gamma_prior, method=backend,
                t=t_val_i, complete=False, u=u_val, log=True,
                **extra_kwargs
            )
            val = sign * np.exp(log_abs)
            print(f"  {backend:>6}: {val:.6e} (diff {abs(val - ref_frac):.2e})")
        except Exception as e:
            print(f"  {backend:>6}: failed ({e})")

    # Fractional order near integer (interpolation)
    order_near = 1.999
    ref_near = exact_imgf_deriv(order_near, t_val_i, alpha_val, beta_val, u_val)
    print(f"\nNear‑integer order {order_near}: exact = {ref_near:.6e}")
    try:
        log_abs, sign = mgfDerivative(
            order=order_near, prior=gamma_prior, method='scipy',
            t=t_val_i, complete=False, u=u_val, log=True,
            use_interpolation=True, d_vec=(0.8, 0.9, 0.95), epsrel=1e-10
        )
        val = sign * np.exp(log_abs)
        print(f"  interpolation: {val:.6e} (diff {abs(val - ref_near):.2e})")
    except Exception as e:
        print(f"  interpolation: failed ({e})")

    # ============================================================
    # 6. Vectorized t example (fractional derivative, scipy)
    # ============================================================
    print("\n" + "=" * 60)
    print("Vectorized t example (fractional derivative, scipy)")
    print("=" * 60)
    t_vec = np.linspace(-2.0, -0.5, 5)
    order_vec = 1.99
    print(f"  t values: {t_vec}")
    results_log, results_sign = mgfDerivative(
        order=order_vec,
        prior=gamma_prior,
        method='scipy',
        t=t_vec,
        log=True,
        integer_method='symbolic',
        epsrel=1e-10
    )
    print("  Results (log scale):")
    for t_val, log_abs, sign in zip(t_vec, results_log, results_sign):
        print(f"    t={t_val:.2f}: log|deriv| = {log_abs:.6f}, sign = {sign:+d}")

    # ============================================================
    # 7. Vectorized order example (multiple orders, scipy)
    # ============================================================
    print("\n" + "=" * 60)
    print("Vectorized order example (multiple orders, scipy)")
    print("=" * 60)
    t_order_vec = -1.0
    orders_vec = np.array([1.5, 1.99, 2.5])
    print(f"  t = {t_order_vec}, orders = {orders_vec}")
    log_abs_orders, sign_orders = mgfDerivative(
        order=orders_vec,
        prior=gamma_prior,
        method='scipy',
        t=t_order_vec,
        log=True,
        integer_method='symbolic',
        epsrel=1e-10
    )
    print("  Results (log scale):")
    for o, log_abs, sign in zip(orders_vec, log_abs_orders, sign_orders):
        print(f"    order={o:.2f}: log|deriv| = {log_abs:.6f}, sign = {sign:+d}")