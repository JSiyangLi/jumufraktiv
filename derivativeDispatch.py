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
    u: float | None = None,
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
        Evaluation point(s). For method='symbolic', can be scalar or array.
        For method='bell' or 'jax', can be an array (vectorized).
    u : float, optional
        Truncation point for incomplete MGF (used when complete=False).
        For symbolic method, substitutes the canonical 'u' symbol.
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
    If t is None or symbolic expression has free symbols:
        sympy.Expr
    Else:
        If log=True: (log_abs, sign) or arrays for vector input.
        If log=False: float or array.
    """
    method = method.lower()
    if method not in {"symbolic", "bell", "jax"}:
        raise ValueError("method must be one of {'symbolic','bell','jax'}.")

    # ---------------------------------------------------------
    # Symbolic differentiation (loops over t if array)
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

        # Handle scalar vs array t
        t_arr = np.atleast_1d(t)
        scalar_input = np.isscalar(t)

        # Pre‑allocate results
        results_log_abs = np.zeros_like(t_arr, dtype=float)
        results_sign = np.ones_like(t_arr, dtype=int)

        for idx, t_val in enumerate(t_arr):
            # Substitute t
            val_expr = expr.subs(t_sym, t_val)

            # Substitute u if incomplete
            if not complete and u is not None:
                val_expr = val_expr.subs(u_sym, u)

            # If any free symbols remain, return the expression (not a loop result)
            if val_expr.free_symbols:
                # If we have multiple t values and some expression is still symbolic, raise error
                if len(t_arr) > 1:
                    raise ValueError(
                        f"Symbolic expression still has free symbols at t={t_val}: {val_expr.free_symbols}. "
                        "Cannot vectorize symbolic evaluation."
                    )
                # For scalar, return the expression
                return val_expr

            # Numeric evaluation
            val = float(val_expr.evalf())
            if abs(val) < 1e-300:
                results_log_abs[idx] = -np.inf
                results_sign[idx] = 1
            else:
                results_log_abs[idx] = np.log(abs(val))
                results_sign[idx] = 1 if val > 0 else -1

        # Return scalar or arrays
        if scalar_input:
            log_abs_val = results_log_abs[0]
            sign_val = results_sign[0]
            return (log_abs_val, sign_val) if log else (sign_val * np.exp(log_abs_val))
        else:
            if log:
                return results_log_abs, results_sign
            else:
                return results_sign * np.exp(results_log_abs)

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
    u: float = None,
    **kwargs
):
    """
    Unified interface for fractional derivatives of the MGF.
    Supports vectorized t for numeric methods; symbolic method loops over t.

    Parameters
    ----------
    order : float
        Fractional order (positive).
    prior : mitMGFprior
        Prior object.
    method : {'scipy', 'mpmath', 'symbolic'}, default 'scipy'
        Computation backend.
    t : float or array-like, optional
        Evaluation point(s) (required for numeric methods).
    simplify : bool, default False
        Simplify symbolic expressions (method='symbolic').
    complete : bool, default True
        If True, differentiate complete MGF; if False, incomplete MGF.
    log : bool, default True
        If True, numeric output is (log_abs, sign); else ordinary float.
    integerDeriv_method : str, default 'symbolic'
        Method for integer derivatives inside integrators.
    u : float, optional
        Truncation point for incomplete MGF (used when complete=False).
    **kwargs : passed to underlying functions.

    Returns
    -------
    sympy.Expr, (log_abs, sign), or float
        If t is scalar and method='symbolic', returns sympy.Expr or scalar.
        If t is array and method='symbolic', returns array of results.
        For numeric methods, returns scalar or arrays matching t shape.
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

        # Detect if t is scalar or array
        t_arr = np.asarray(t)
        scalar_input = t_arr.ndim == 0
        if scalar_input:
            t_arr = np.array([t])
        batch = len(t_arr)

        # Pre-allocate results
        if log:
            log_abs_vals = np.zeros(batch)
            sign_vals = np.ones(batch, dtype=int)
        else:
            val_vals = np.zeros(batch)

        # Loop over t values
        for idx, t_val in enumerate(t_arr):
            # Substitute t and u
            expr_sub = expr.subs(t_sym, t_val)
            if not complete and u is not None:
                expr_sub = expr_sub.subs(u_sym, u)

            # If free symbols remain, we cannot proceed for vector input
            if expr_sub.free_symbols:
                if batch > 1:
                    raise ValueError(
                        f"Symbolic expression still has free symbols at t={t_val}: {expr_sub.free_symbols}. "
                        "Cannot vectorize symbolic evaluation."
                    )
                # Scalar input: return expression
                return expr_sub

            # Numeric evaluation
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

        # Return scalar or arrays
        if scalar_input:
            if log:
                return log_abs_vals[0], int(sign_vals[0])
            else:
                return float(val_vals[0])
        else:
            if log:
                return log_abs_vals, sign_vals
            else:
                return val_vals

    # ---- Numeric paths require t ----
    if t is None or (isinstance(t, float) and math.isnan(t)):
        raise ValueError(f"For method '{method}', t must be provided.")

    # Convert to array for consistent handling (underlying functions accept arrays)
    t_arr = np.asarray(t)
    # If scalar, we can pass scalar to underlying functions (they can handle it)
    # The underlying functions already accept arrays.

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
    u: float = None,
    **kwargs
):
    """
    Unified wrapper for integer or fractional derivatives of the MGF.
    Supports vectorized t and vectorized order (array of orders).

    If `order` is array-like, each order is processed independently.
    If `t` is also array-like, the results are stacked: for each order,
    an array over t is produced, and then stacked along a new axis.

    Parameters
    ----------
    order : float or array-like or sp.Basic
        Derivative order(s). If array-like, each element is processed separately.
    prior : mitMGFprior
        Prior object.
    method : str, optional
        Derivative backend.
    t : float or array-like, optional
        Evaluation point(s). If array-like, vectorized over t.
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
    u : float, optional
        Truncation point for incomplete MGF.
    **kwargs : passed to underlying functions.

    Returns
    -------
    If order is scalar:
        - If log=True: (log_abs, sign) where log_abs and sign are scalars or arrays
          (depending on t).
        - If log=False: scalar or array.
    If order is array-like:
        - If log=True: (log_abs_array, sign_array) where the arrays have shape
          (len(order), ...) matching the t shape.
        - If log=False: array of shape (len(order), ...).
    """
    # ---- Validate d_vec (independent of order) ----
    if len(d_vec) != 3:
        raise ValueError("d_vec must have exactly 3 elements.")
    if any(d >= 1 for d in d_vec):
        raise ValueError("All elements of d_vec must be < 1.")

    # ---- Dispatch for array-like order ----
    if hasattr(order, '__len__') and not isinstance(order, (str, bytes, sp.Basic)):
        order_list = list(order)
        results = []
        for o in order_list:
            res = mgfDerivative(
                order=o,
                prior=prior,
                method=method,
                t=t,
                simplify=simplify,
                complete=complete,
                log=log,
                integer_method=integer_method,
                use_interpolation=use_interpolation,
                d_vec=d_vec,
                int_tol=int_tol,
                u=u,
                **kwargs
            )
            results.append(res)

        # Restructure results based on log flag
        if log:
            # results is list of tuples (log_abs, sign)
            # Each log_abs/sign could be scalar or array depending on t
            # We'll stack them along a new first axis
            log_abs_vals = [r[0] for r in results]
            sign_vals = [r[1] for r in results]
            # If first element is scalar, stack into 1D arrays
            if np.isscalar(log_abs_vals[0]):
                return np.array(log_abs_vals), np.array(sign_vals)
            else:
                # Stack along new axis (axis=0)
                return np.stack(log_abs_vals, axis=0), np.stack(sign_vals, axis=0)
        else:
            # results is list of scalars or arrays
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