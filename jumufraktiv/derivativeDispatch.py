"""
derivativeDispatch.py

Unified interface for computing derivatives of moment-generating functions (MGFs).

This module provides the core derivative routines for the package, supporting
both integer and fractional orders, and multiple computational backends.

Main functions:
    - mgfDerivative_integer : integer-order derivatives (symbolic, Bell, JAX)
    - mgfDerivative_fractional : fractional-order derivatives (scipy, mpmath, symbolic)
    - mgfDerivative : unified dispatcher for both integer and fractional orders

Design principles:
    1. **Symbol-numeric principle**: the return type depends only on whether
       unresolved symbols remain; symbolic methods may return `sympy.Expr` or
       numeric values.
    2. **Log principle**: in numeric state, whether to store `(log_abs, sign)`
       or a scalar value depends only on the `log` argument.
    3. **Tuple-vectorisation principle**: evaluation points `(t, u)` are
       broadcast to a common shape, supporting vectorised computation over
       arrays of `t` and/or `u`.

Backends:
    - Symbolic : SymPy differentiation (integer and fractional, via `sp.diff`
      and `sp.Derivative`).
    - Bell : Bell polynomial method (integer orders, JAX-based).
    - JAX : JAX `jet` or `grad` (integer orders).
    - SciPy : adaptive quadrature with fallback to tan‑transform (fractional).
    - Mpmath : high‑precision quadrature (fractional).
    - Interpolation : cubic interpolation for near‑integer fractional orders.

Incomplete MGF (iMGF) derivatives are supported via the `complete=False` flag,
which uses the prior's `imgf_sym` or `imgf_jax` functions.

Imports:
    - integerDeriv_symbolic from symbolic_integerDeriv.py
    - integerDeriv_numeric_bell from numeric_integerDeriv_Bell.py
    - integerDeriv_numeric_jax from numeric_integerDeriv_JAX.py
    - fractionalDeriv_numeric_scipy / _mpmath / _tan / _interpolated

Functions:
    - mgfDerivative_integer(order, prior, method='symbolic', t=None, ...)
    - mgfDerivative_fractional(order, prior, method='scipy', t=None, ...)
    - mgfDerivative(order, prior, method='auto', t=None, ...)
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

    This function respects the **symbol‑numeric principle**: the return type
    depends only on whether unresolved symbols remain.

    - If `t` is `None` or if the substituted expression still contains free
      symbols (e.g., hyperparameters), a symbolic expression is returned.
    - If all symbols are resolved (numeric `t` and `u`), the derivative is
      evaluated numerically.

    The function also supports **tuple‑vectorisation**: if `t` or `u` are
    array‑like, they are broadcast to a common shape and the derivative is
    evaluated for all points simultaneously.

    Parameters
    ----------
    order : int or sympy.Expr
        Non‑negative derivative order. If symbolic, returns an unevaluated
        `Derivative` object.
    prior : mitMGFprior
        Prior object providing symbolic and/or backend MGF/PDF representations.
    method : {"symbolic", "bell", "jax"}, optional
        Derivative backend:
        - `"symbolic"`: uses SymPy differentiation.
        - `"bell"`: uses Bell polynomials (requires JAX).
        - `"jax"`: uses JAX's `jet` or `grad`.
    t : float or array-like, optional
        Evaluation point(s) for the canonical variable `t`.
    u : float or array-like, optional
        Truncation point(s) for the incomplete MGF (used when `complete=False`).
        If array‑like, it is broadcast with `t` to form a batch of evaluation
        points `(t, u)`.
    simplify : bool, optional
        If True, simplify the symbolic derivative expression.
    complete : bool, optional
        If True (default), differentiate the complete MGF (`prior.mgf_sym`).
        If False, differentiate the incomplete MGF (`prior.imgf_sym`).
    log : bool, optional
        If True, numeric methods return `(log_abs, sign)`.
        If False, return the ordinary derivative as float.
    symbolic_timeout : float, optional
        Maximum time (seconds) for symbolic computation in the Bell backend.
    cgf_method : str, optional
        Method for CGF derivatives in the Bell backend (`'auto'`, `'jet'`, `'grad'`).

    Returns
    -------
    sympy.Expr, tuple (log_abs, sign), or float / np.ndarray
        - If `t` is `None` or free symbols remain: `sympy.Expr`.
        - If evaluation is numeric:
            - `log=True`: `(log_abs, sign)` (scalars or arrays).
            - `log=False`: numeric value (scalar or array).

    Notes
    -----
    - The canonical symbols `t` and `u` are imported from `jumufraktiv.symbols`.
    - For the symbolic method, when `t` is an array, each element is evaluated
      individually using `.subs().evalf()` to ensure accuracy (mpmath).

    Examples
    --------
    >>> # Complete MGF, 2nd derivative, symbolic expression
    >>> expr = mgfDerivative_integer(2, prior, method='symbolic')
    >>> # Evaluate at t = -1.0
    >>> log_abs, sign = mgfDerivative_integer(2, prior, method='symbolic', t=-1.0, log=True)

    >>> # Incomplete MGF derivative for multiple t and u
    >>> t_vals = np.linspace(-1.0, 1.0, 10)
    >>> u_vals = 2.0
    >>> log_abs, sign = mgfDerivative_integer(1, prior, method='jax',
    ...                                       t=t_vals, u=u_vals, complete=False)
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

    This function respects the **symbol‑numeric principle**: the return type
    depends only on whether unresolved symbols remain.

    - If `method='symbolic'` and `t` is `None` or the expression still contains
      free symbols, a symbolic expression is returned.
    - If `t` is numeric (scalar or array), the derivative is evaluated numerically
      using either `scipy` or `mpmath` backend.

    The function supports **tuple‑vectorisation**: `t` and `u` (for incomplete MGF)
    are broadcast to a common shape, and the derivative is evaluated for all
    points simultaneously.

    Parameters
    ----------
    order : float
        Fractional order (positive).
    prior : mitMGFprior
        Prior object providing symbolic and/or backend MGF/PDF representations.
    method : {'scipy', 'mpmath', 'symbolic'}, default 'scipy'
        Computation backend:
        - `'scipy'`: uses `scipy.integrate.quad` (adaptive) with fallback.
        - `'mpmath'`: uses `mpmath.quad` (high precision).
        - `'symbolic'`: returns a symbolic expression (may be slow).
    t : float or array-like, optional
        Evaluation point(s) for the canonical variable `t`. Required for numeric
        methods.
    simplify : bool, default False
        If True, simplify the symbolic expression (method='symbolic' only).
    complete : bool, default True
        If True, differentiate the complete MGF (`prior.mgf_sym`).
        If False, differentiate the incomplete MGF (`prior.imgf_sym`).
    log : bool, default True
        If True, numeric output is `(log_abs, sign)` where `log_abs` is the
        natural logarithm of the absolute derivative and `sign` is ±1.
        If False, return the ordinary derivative as a float.
    integerDeriv_method : str, default 'symbolic'
        Method for integer derivatives inside the fractional integrator:
        `'symbolic'`, `'bell'`, or `'jax'`.
    u : float or array-like, optional
        Truncation point(s) for the incomplete MGF (used when `complete=False`).
        If array‑like, it is broadcast with `t` to form a batch of evaluation
        points `(t, u)`.
    **kwargs : additional keyword arguments passed to the underlying backend.
        For `'scipy'`: `epsabs`, `epsrel`, `limit`, `initial_L`, `max_L`, `tol`,
        `use_tan`.
        For `'mpmath'`: `dps`, `tol`, `use_tan`.
        For `'symbolic'`: `timeout_seconds`.

    Returns
    -------
    sympy.Expr, tuple (log_abs, sign), or float / np.ndarray
        - If `method='symbolic'` and `t` is `None` or free symbols remain:
          `sympy.Expr`.
        - If numeric evaluation:
            - `log=True`: `(log_abs, sign)` (scalars or arrays).
            - `log=False`: numeric value (scalar or array).

    Notes
    -----
    - The canonical symbols `t` and `u` are imported from `jumufraktiv.symbols`.
    - For the symbolic path, when `t` is an array, each element is evaluated
      individually using `.subs().evalf()` to preserve accuracy (mpmath).
    - The `scipy` backend uses an adaptive quadrature with range expansion;
      the `mpmath` backend uses `tanh-sinh` quadrature with arbitrary precision.

    Examples
    --------
    >>> # Fractional derivative of order 1.5 at t = -1.0 (scipy)
    >>> log_abs, sign = mgfDerivative_fractional(1.5, prior, method='scipy', t=-1.0, log=True)

    >>> # Vectorised evaluation at multiple t values
    >>> t_vals = np.linspace(-2.0, 0.0, 10)
    >>> log_abs, sign = mgfDerivative_fractional(1.5, prior, method='scipy', t=t_vals, log=True)

    >>> # Incomplete MGF derivative (complete=False)
    >>> log_abs, sign = mgfDerivative_fractional(1.5, prior, method='mpmath',
    ...                                          t=-1.0, u=2.0, complete=False, dps=60)

    >>> # Symbolic fractional derivative (warning: slow)
    >>> expr = mgfDerivative_fractional(1.5, prior, method='symbolic')
    """
    # ---- Symbolic path ----
    if method.lower() == "symbolic":
        warnings.warn(
            "Symbolic fractional derivatives can be very slow. "
            "Consider 'scipy' or 'mpmath' instead."
        )
        from jumufraktiv.symbolic_fractionalDeriv import fractionalDeriv_symbolic

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

def _check_moment_exists_at_origin(order, prior, t) -> None:
    """
    Reject an order whose moment does not exist, when evaluating at ``t = 0``.

    Parameters
    ----------
    order : float or sympy.Basic
        Derivative order. Symbolic orders are not checked.
    prior : mitMGFprior
        Prior, consulted for its ``max_finite_moment``.
    t : float, array-like or None
        Evaluation point(s). Only ``t = 0`` is checked.

    Raises
    ------
    ValueError
        If ``t`` includes 0 and ``order >= prior.max_finite_moment``.

    Notes
    -----
    ``Dᵃ M(t) = E[Θᵃ e^{tΘ}]``. For ``t < 0`` the exponential dominates any
    polynomial, so the moment condition is *not* required and must not be
    enforced — that is what lets heavy-tailed priors such as ``pareto`` work,
    and it is stated as a rule in CLAUDE.md.

    At ``t = 0`` exactly, the exponential is 1 and the identity reduces to
    ``E[Θᵃ]``, so the moment must exist. That boundary is reachable from
    ordinary data: ``b(y) = 0`` whenever every observation sits at the known
    mean (``laplace``, ``normal``), at zero (``halfnormal``), or at the scale
    (``pareto``) — measure-zero for continuous data, but common once data is
    rounded.

    Without this check the failure is silent or misleading. With the registry's
    Pareto(α=2) prior at ``t = 0``: order 1 returns a correct value, order 2
    returns ``inf``, and order 3 raises ``TypeError: Cannot convert complex to
    float`` from inside the quadrature.

    The bound is a property of the *prior*, not of the data, which is why this
    lives here rather than in ``like_stats`` — those modules are pure functions
    of the data and cannot see the prior. The same ``b = 0`` is perfectly valid
    against a Gamma prior at every order.
    """
    if t is None or isinstance(order, sp.Basic):
        return

    limit = float(getattr(prior, "max_finite_moment", np.inf))
    if np.isinf(limit):
        return

    if not np.any(np.asarray(t, dtype=float) == 0.0):
        return

    if float(order) >= limit:
        name = getattr(prior, "name", "this prior")
        raise ValueError(
            f"Cannot evaluate a derivative of order {float(order)} at t = 0 for "
            f"prior '{name}': that requires the moment E[Theta^{float(order)}] "
            f"to be finite, but it exists only for orders strictly below "
            f"{limit}. t = 0 arises when the sufficient statistic b(y) is zero, "
            f"which happens when every observation sits at the boundary the "
            f"likelihood subtracts (for example y == mean, y == 0, or "
            f"y == scale). Away from t = 0 no moment condition applies."
        )


def resolve_backend(
    order,
    method: str = "auto",
    integer_method: str = "symbolic",
    int_tol: float = 1e-12,
):
    """Classify a derivative order and settle which backend will serve it.

    This is the single place where the backend matrix in ``CLAUDE.md`` is
    encoded. :func:`mgfDerivative` calls it to dispatch, and callers that need
    to know *how* a request will be served — before serving it — call it for
    that answer rather than re-deriving the rules.

    Parameters
    ----------
    order : float, int or sympy.Basic
        Derivative order. Must be scalar; array-like orders are resolved
        element by element.
    method : str, optional
        Requested backend, or ``'auto'`` to choose one.
    integer_method : str, optional
        Backend used for the integer derivatives taken *inside* a fractional
        computation. Ignored for integer and symbolic orders.
    int_tol : float, optional
        An order counts as an integer when it lies within ``int_tol`` of one.

    Returns
    -------
    order_type : {'symbolic', 'integer', 'fractional'}
        Which row of the backend matrix applies.
    method : str
        The resolved backend, with ``'auto'`` settled and any reinterpretation
        applied.
    integer_method : str
        The resolved inner integer backend.

    Raises
    ------
    ValueError
        If ``method`` or ``integer_method`` is not valid for ``order_type``.

    Notes
    -----
    Requesting ``'bell'`` or ``'jax'`` for a fractional order is not an error:
    neither can take a fractional derivative, so the argument is reinterpreted
    as ``integer_method`` and the fractional backend falls back to ``'scipy'``.
    A warning records that the argument was not used as written.

    Examples
    --------
    >>> resolve_backend(2, 'auto')
    ('integer', 'symbolic', 'symbolic')
    >>> resolve_backend(1.5, 'auto')
    ('fractional', 'scipy', 'symbolic')
    >>> resolve_backend(1.5, 'bell')     # reinterpreted, with a warning
    ('fractional', 'scipy', 'bell')

    Backend names are matched case-insensitively and returned canonicalised,
    so callers may compare the result directly:

    >>> resolve_backend(2, 'SYMBOLIC')
    ('integer', 'symbolic', 'symbolic')
    """
    # Canonicalise before anything else. The backends have always matched
    # method names case-insensitively, so 'SYMBOLIC' is a legal spelling; a
    # caller comparing the *returned* name against a lowercase literal would
    # otherwise silently take the wrong branch.
    method = method.lower()
    integer_method = integer_method.lower()

    # ---- Determine order type ----
    if isinstance(order, sp.Basic):
        order_type = "symbolic"
    elif abs(order - round(order)) < int_tol:
        order_type = "integer"
    else:
        order_type = "fractional"

    # ---- Resolve 'auto' ----
    if method == "auto":
        method = "scipy" if order_type == "fractional" else "symbolic"

    # ---- Validate against the row ----
    if order_type == "symbolic":
        if method != "symbolic":
            raise ValueError(
                f"Invalid method '{method}' for symbolic order. "
                "Only 'symbolic' is allowed."
            )

    elif order_type == "integer":
        valid_int_methods = {"symbolic", "bell", "jax"}
        if method not in valid_int_methods:
            raise ValueError(
                f"Invalid method '{method}' for integer order. "
                f"Choose from {valid_int_methods}."
            )

    else:
        valid_frac_methods = {"scipy", "mpmath", "symbolic"}
        if method in {"jax", "bell"}:
            warnings.warn(
                f"Method '{method}' cannot take a fractional derivative. "
                f"Interpreting it as integer_method='{method}' and using the "
                f"default fractional method 'scipy'. Pass integer_method="
                f"'{method}' explicitly to silence this warning.",
                UserWarning,
                stacklevel=2,
            )
            integer_method = method
            method = "scipy"
        elif method not in valid_frac_methods:
            raise ValueError(
                f"Invalid method '{method}' for fractional order. "
                f"Choose from {valid_frac_methods}."
            )

        valid_int_methods = {"symbolic", "jax", "bell"}
        if integer_method not in valid_int_methods:
            raise ValueError(
                f"Invalid integer_method '{integer_method}'. "
                f"Choose from {valid_int_methods}."
            )

    return order_type, method, integer_method


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

    This function respects the **symbol‑numeric principle**: the return type
    depends only on whether unresolved symbols remain.

    - If `order` contains symbolic variables, or if `t` is `None` or the
      expression still has free symbols, a symbolic expression is returned.
    - If `order`, `t`, and `u` (if applicable) are fully numeric, the derivative
      is evaluated numerically using the selected backend.

    The function supports **tuple‑vectorisation**: if `t` or `u` are array‑like,
    they are broadcast to a common shape and the derivative is evaluated for
      all points simultaneously.

    If `order` is array‑like, each order is processed independently, and the
    results are stacked along a new first axis.

    Parameters
    ----------
    order : float, array-like, or sympy.Basic
        Derivative order(s). If array-like, each element is processed separately.
        If a SymPy expression, the derivative is treated symbolically.
    prior : mitMGFprior
        Prior object providing symbolic and/or backend MGF/PDF representations.
    method : str, optional
        Derivative backend. For integer orders: `'symbolic'`, `'bell'`, `'jax'`.
        For fractional orders: `'scipy'`, `'mpmath'`, `'symbolic'`.
        Special value `'auto'` chooses the appropriate method automatically.
    t : float or array-like, optional
        Evaluation point(s) for the canonical variable `t`.
    simplify : bool, optional
        If True, simplify symbolic expressions.
    complete : bool, optional
        If True, differentiate the complete MGF (`prior.mgf_sym`).
        If False, differentiate the incomplete MGF (`prior.imgf_sym`).
    log : bool, optional
        If True, numeric outputs return `(log_abs, sign)` where `log_abs` is
        the natural logarithm of the absolute derivative and `sign` is ±1.
        If False, return the ordinary derivative as a float.
    integer_method : str, optional
        For fractional orders: method for integer derivatives inside the
        fractional integrator (`'symbolic'`, `'bell'`, `'jax'`).
    use_interpolation : bool, optional
        If True and the order is near an integer from below, use cubic
        interpolation to speed up the computation.
    d_vec : tuple, optional
        Complements of deviations for interpolation (default `(0.8, 0.9, 0.95)`).
        Actual deviations are `1 - d_i`.
    int_tol : float, optional
        Tolerance for detecting integer order. If `|order - round(order)| < int_tol`,
        the order is treated as an integer.
    u : float or array-like, optional
        Truncation point(s) for the incomplete MGF (used when `complete=False`).
        If array‑like, it is broadcast with `t` to form a batch of evaluation
        points `(t, u)`.
    **kwargs : additional keyword arguments passed to the underlying backend.
        For integer methods: `symbolic_timeout`, `cgf_method`.
        For fractional methods: `epsabs`, `epsrel`, `limit`, `dps`, `tol`, etc.

    Returns
    -------
    sympy.Expr, tuple (log_abs, sign), or float / np.ndarray
        - If `order` is symbolic or `t` is `None` or free symbols remain:
          `sympy.Expr`.
        - If numeric evaluation:
            - `log=True`: `(log_abs, sign)` (scalars or arrays).
            - `log=False`: numeric value (scalar or array).

    Notes
    -----
    - The canonical symbols `t` and `u` are imported from `jumufraktiv.symbols`.
    - The `'auto'` method chooses `'symbolic'` for integer orders (if available)
      and `'scipy'` for fractional orders by default.
    - The interpolation (`use_interpolation=True`) is used when the order is
      within `(n - max_dev, n)` where `n = ceil(order)` and `max_dev = 1 - max(d_vec)`.
    - For `method='symbolic'`, vectorisation over `t` is achieved by looping
      over elements using `.subs().evalf()` to maintain accuracy (mpmath).

    Examples
    --------
    >>> # Integer derivative (2nd order) at t = -1.0 using symbolic method
    >>> log_abs, sign = mgfDerivative(2, prior, method='symbolic', t=-1.0, log=True)

    >>> # Fractional derivative (order 1.5) using scipy backend
    >>> log_abs, sign = mgfDerivative(1.5, prior, method='scipy', t=-1.0, log=True)

    >>> # Vectorised over t values (complete MGF)
    >>> t_vals = np.linspace(-1.0, 1.0, 10)
    >>> log_abs, sign = mgfDerivative(1.5, prior, method='scipy', t=t_vals, log=True)

    >>> # Incomplete MGF (complete=False) with vectorised (t, u)
    >>> t_vals = np.linspace(-1.0, 1.0, 5)
    >>> u_vals = 2.0
    >>> log_abs, sign = mgfDerivative(1.5, prior, method='mpmath',
    ...                               t=t_vals, u=u_vals, complete=False, dps=60)

    >>> # Multiple orders at once (array-like order)
    >>> orders = np.array([1.0, 1.5, 2.0])
    >>> log_abs, sign = mgfDerivative(orders, prior, method='scipy', t=-1.0, log=True)
    # Returns arrays of shape (3,) for log_abs and sign
    """
    # ---- Validate d_vec (independent of order) ----
    if len(d_vec) != 3:
        raise ValueError("d_vec must have exactly 3 elements.")
    if any(d >= 1 for d in d_vec):
        raise ValueError("All elements of d_vec must be < 1.")

    # ---- Dispatch for array-like order ----
    if hasattr(order, '__len__') and not isinstance(order, (str, bytes, sp.Basic)):
        # Each element is dispatched on its own, so this block's only job is to
        # broadcast the request, forward each element unchanged, and reassemble
        # the answers in the caller's shape. Every coercion it used to apply
        # was a defect:
        #
        #   int(o)     turned a fractional order into a whole one, returning
        #              the answer for a different derivative. Not a rounding
        #              question -- at order 2.5 both int() and round() give 2
        #              and the result is 15% wrong either way.
        #   float(tt)  rejected t=None, so an array order could not produce a
        #              symbolic result, violating the symbol-numeric principle.
        #   float(uu)  did the same for the incomplete-MGF truncation point.
        #
        # `order_arr.shape` is the shape of the whole request once broadcast,
        # so reassembling into it preserves the caller's shape instead of
        # flattening it.
        order_arr = np.asarray(order)
        if complete:
            if u is not None:
                raise ValueError("u must be None when complete=True")
            order_arr, t_arr = np.broadcast_arrays(order_arr, np.asarray(t, dtype=object))
            u_flat = [None] * order_arr.size
        else:
            if u is None:
                raise ValueError("u must be provided when complete=False")
            order_arr, t_arr, u_arr = np.broadcast_arrays(
                order_arr, np.asarray(t, dtype=object), np.asarray(u, dtype=object)
            )
            u_flat = list(u_arr.flat)

        batch_shape = order_arr.shape

        results = [
            mgfDerivative(
                order=o.item() if hasattr(o, "item") else o,
                prior=prior,
                method=method,
                t=tt,
                simplify=simplify,
                complete=complete,
                log=log,
                integer_method=integer_method,
                use_interpolation=use_interpolation,
                d_vec=d_vec,
                int_tol=int_tol,
                u=uu,
                **kwargs,
            )
            for o, tt, uu in zip(order_arr.flat, t_arr.flat, u_flat)
        ]

        # A symbolic element makes the whole result symbolic: the
        # symbol-numeric principle keys the return type on whether unresolved
        # symbols remain, not on how the request was spelled.
        if any(isinstance(r, sp.Basic) for r in results):
            return np.array(results, dtype=object).reshape(batch_shape)

        if log:
            log_abs = np.array([r[0] for r in results]).reshape(batch_shape)
            sign = np.array([r[1] for r in results]).reshape(batch_shape)
            return log_abs, sign

        return np.array(results).reshape(batch_shape)

    # ---- Continue with scalar order (original logic) ----
    cgf_method = kwargs.pop('cgf_method', 'auto')
    symbolic_timeout = kwargs.pop('symbolic_timeout', 600.0)

    _check_moment_exists_at_origin(order, prior, t)

    order_type, method, integer_method = resolve_backend(
        order, method, integer_method, int_tol
    )

    # Dispatch
    if order_type == "symbolic":
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
        # Fractional order. The backend and integer_method were settled, and
        # any 'bell'/'jax' reinterpretation applied, by resolve_backend above.

        # Check interpolation
        n = int(np.ceil(order))
        if use_interpolation and order > n - (1.0 - max(d_vec)):
            if method.lower() != 'scipy':
                print(f"⚠️ Interpolation triggered for order {order}. "
                      f"Overriding method '{method}' with 'scipy'.")
            scipy_keys = {'epsabs', 'epsrel', 'limit', 'initial_L', 'max_L', 'tol', 'use_tan'}
            scipy_kwargs = {k: v for k, v in kwargs.items() if k in scipy_keys}
            from jumufraktiv.numeric_fractionalDeriv_interpolation import (
                fractionalDeriv_interpolated,
            )
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