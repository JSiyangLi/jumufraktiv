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
    - Grid : fixed-grid trapezoid on the z = e^u substitution, with exact
      near-integer singularity subtraction (fractional; the `scipy` method).
    - Mpmath : high‑precision quadrature (fractional).

Incomplete MGF (iMGF) derivatives are supported via the `complete=False` flag,
which uses the prior's `imgf_sym` or `imgf_jax` functions.

Imports:
    - integerDeriv_symbolic from symbolic_integerDeriv.py
    - integerDeriv_numeric_bell from numeric_integerDeriv_Bell.py
    - integerDeriv_numeric_jax from numeric_integerDeriv_JAX.py
    - fractionalDeriv_grid from numeric_fractionalDeriv_grid.py
    - fractionalDeriv_numeric_mpmath from numeric_fractionalDeriv_mpmath.py

Functions:
    - mgfDerivative_integer(order, prior, method='symbolic', t=None, ...)
    - mgfDerivative_fractional(order, prior, method='scipy', t=None, ...)
    - mgfDerivative(order, prior, method='auto', t=None, ...)
"""

import difflib
import math
import warnings

import numpy as np
import sympy as sp

from jumufraktiv.numeric_expectation import expectation_is_available
from jumufraktiv.numeric_integerDeriv_Bell import integerDeriv_numeric_bell
from jumufraktiv.numeric_integerDeriv_JAX import integerDeriv_numeric_jax
from jumufraktiv.symbolic_cache import cached_lambdify
from jumufraktiv.symbolic_integerDeriv import integerDeriv_symbolic
from jumufraktiv.symbols import t as t_sym  # <-- import canonical u
from jumufraktiv.symbols import u as u_sym

#: Tuning options the derivative layer understands, and which backend consumes
#: each. This is the authoritative list: :data:`MGFDerivative_class.
#: DERIVATIVE_KWARGS` is built from it rather than restated, so the constructor
#: and the ``mgfDerivative*`` functions cannot disagree about what a valid
#: option is.
#:
#: They used to disagree, and in the direction that hides mistakes. The
#: constructor rejects an unknown keyword argument with a ``TypeError`` and a
#: "did you mean" suggestion (PR 3b), while the functions accepted anything and
#: filtered it away just before the call -- so ``epsrel=1e-14`` raised through
#: one public entry point and was silently ignored through the other, and so
#: was a plain misspelling like ``epsrell``.
BACKEND_OPTIONS = {
    "tol": "the fixed-grid and mpmath fractional kernels",
    "dps": "the mpmath fractional kernel",
    "use_tan": "the mpmath fractional kernel",
    "cgf_method": "the Bell integer backend",
    "symbolic_timeout": "the Bell integer backend",
    "timeout_seconds": "the symbolic fractional backend",
}

#: Every option name the derivative layer accepts, including the named
#: parameters of :func:`mgfDerivative` itself that a caller may reasonably set.
DERIVATIVE_OPTIONS = frozenset(BACKEND_OPTIONS) | {"integer_method", "int_tol"}


def _reject_unknown_options(kwargs, function_name):
    """Raise ``TypeError`` for any keyword argument no backend consumes.

    Parameters
    ----------
    kwargs : dict
        Leftover keyword arguments, after the caller's named parameters bind.
    function_name : str
        Name to quote in the message, so the caller knows which call to fix.

    Raises
    ------
    TypeError
        Naming each unrecognised option, what the layer does accept, and a
        close match where one exists.

    Notes
    -----
    Only names that reach *no* backend are refused here. A name that some
    backend understands but the selected one does not -- ``dps`` under
    ``method="scipy"``, say -- is still accepted and dropped. That is a
    narrower defect with a real design question behind it (reject, warn, or
    honour by switching backends?), it predates this guard, and PR 12 owns it;
    see "Known-broken" in :file:`CLAUDE.md`.
    """
    unknown = sorted(set(kwargs) - DERIVATIVE_OPTIONS)
    if not unknown:
        return

    parts = []
    for name in unknown:
        close = difflib.get_close_matches(name, sorted(DERIVATIVE_OPTIONS), n=1)
        parts.append(f"'{name}'" + (f" (did you mean '{close[0]}'?)" if close else ""))

    raise TypeError(
        f"{function_name}() got unexpected keyword argument(s): "
        + ", ".join(parts)
        + ". Options the derivative layer accepts: "
        + ", ".join(sorted(DERIVATIVE_OPTIONS))
        + "."
    )


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
        # Also store symbolic results if any free symbols remain. The flag is
        # tracked rather than rediscovered: `all(r is None for r in
        # results_expr)` is a Python loop over the whole batch, and the batch
        # here is (n_nodes x n_points) -- 50,244 iterations for a 20-point
        # request through the fixed-grid kernel, to learn something the loop
        # below already knew.
        results_expr = [None] * n_points
        any_symbolic = False

        # ---- Build substitution dictionaries for each point ----
        subs_list = []
        for idx in range(n_points):
            subs_local = {t_sym: t_flat[idx]}
            if not complete:
                subs_local[u_sym] = u_flat[idx]
            subs_list.append(subs_local)

        # ---- Fast path: compile once, evaluate the whole batch -----------
        #
        # `expr.subs(...).evalf()` per point was 97.6% of the runtime of the
        # `scipy` fractional route -- 5,024 SymPy substitutions for two
        # evaluation points, because the fixed-grid kernel hands this function
        # an (n_nodes x n_points) array of shifted points and it took them one
        # at a time. `CLAUDE.md` recorded the remedy under "Numerical policy",
        # measured at ~6400x, and it had never been applied here.
        #
        # `lambdify` compiles the derivative to a NumPy function evaluated on
        # the whole array at once. It works in float64, so it cannot represent
        # `M^(301)`; the exact path below still runs, but only for the elements
        # where float64 overflowed, underflowed, or produced a NaN. Those are
        # rare, so the loop survives for correctness at the extremes while the
        # ordinary case never enters it.
        pending = list(range(n_points))
        symbols_left = expr.free_symbols - ({t_sym} if complete else {t_sym, u_sym})
        if not symbols_left and n_points:
            arg_symbols = (t_sym,) if complete else (t_sym, u_sym)
            # Probe with the caller's own first point, which is in domain by
            # construction. Some priors compile to something that raises --
            # Pareto's MGF uses `expint`, which neither scipy nor numpy
            # provides -- and the only way to find that out is to call it.
            first = (t_flat[:1],) if complete else (t_flat[:1], u_flat[:1])
            compiled = cached_lambdify(expr, arg_symbols, probe=first)
            values = None
            if compiled is not None:
                with np.errstate(all="ignore"):
                    raw = compiled(t_flat) if complete else compiled(t_flat, u_flat)
                values = np.asarray(raw, dtype=float).reshape(n_points)

            if values is not None:
                exact = ~np.isfinite(values) | (values == 0.0)
                finite = ~exact
                with np.errstate(divide="ignore"):
                    results_log_abs[finite] = np.log(np.abs(values[finite]))
                results_sign[finite] = np.where(values[finite] >= 0, 1, -1)
                # A value that underflowed to zero is not necessarily zero, and
                # one that overflowed to inf is not necessarily infinite. Only
                # those go on to the exact path.
                pending = list(np.flatnonzero(exact))

        # ---- Exact path: whatever float64 could not represent -------------
        for idx in pending:
            val_expr = expr.subs(subs_list[idx])
            if val_expr.free_symbols:
                # If any free symbols remain, we keep the expression
                results_expr[idx] = val_expr
                any_symbolic = True
            else:
                evaluated = val_expr.evalf()
                val = float(evaluated)
                if math.isinf(val) or (val == 0.0 and evaluated != 0):
                    # The SymPy value is finite and non-zero; it is the cast to
                    # a Python float that has overflowed or underflowed. Taking
                    # the log first keeps the exponent in range, so a large
                    # derivative order returns its true magnitude instead of
                    # `inf` or `-inf`.
                    #
                    # This is a SECOND source of the recorded large-order
                    # overflow, and the audit attributes that defect entirely to
                    # the outer quadrature accumulating in linear space. Fixing
                    # only the accumulation is not enough: M^(301) of the Gamma
                    # MGF at t = -1 came back as `inf` here, where M^(151) is
                    # exact to 1.4e-16, so the fractional derivative built on it
                    # was wrong before any accumulation happened.
                    results_log_abs[idx] = float(sp.log(sp.Abs(evaluated)).evalf())
                    results_sign[idx] = 1 if evaluated > 0 else -1
                elif abs(val) < 1e-300:
                    results_log_abs[idx] = -np.inf
                    results_sign[idx] = 1
                else:
                    results_log_abs[idx] = np.log(abs(val))
                    results_sign[idx] = 1 if val > 0 else -1

        # ---- Decide return type ----
        # If all results are numeric, return numeric arrays (or scalars)
        if not any_symbolic:
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
        - `'scipy'`: uses the fixed-grid kernel in
          `numeric_fractionalDeriv_grid`.
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
        Only the names in `BACKEND_OPTIONS` are accepted; anything else raises
        `TypeError` naming it. For `'scipy'`: `tol`. For `'mpmath'`: `dps`,
        `tol`, `use_tan`. A name valid for a backend other than the one
        selected is accepted and ignored — see `BACKEND_OPTIONS`.

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
    - The `scipy` backend uses a fixed-grid trapezoid rule on the `z = e^u`
      substitution, with the range derived from `gamma = floor(order)+1-order`;
      the `mpmath` backend uses `tanh-sinh` quadrature at arbitrary precision.

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
    _reject_unknown_options(kwargs, "mgfDerivative_fractional")

    # ---- Symbolic path ----
    if method.lower() == "symbolic":
        warnings.warn(
            "Symbolic fractional derivatives can be very slow. "
            "Consider 'scipy' or 'mpmath' instead.",
            UserWarning,
            stacklevel=2,
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
        # The same fixed-grid kernel `mgfDerivative` uses. Until PR 8 these two
        # entry points reached *different* implementations: `mgfDerivative`
        # was moved onto the grid kernel by PR 6b while this one was left on
        # the adaptive scheme the grid kernel replaced, so the answer depended
        # on which public function the caller happened to reach for.
        from jumufraktiv.numeric_fractionalDeriv_grid import fractionalDeriv_grid

        grid_keys = {"tol"}
        return fractionalDeriv_grid(
            order=order,
            prior=prior,
            t_points=t,
            u_points=u,
            complete=complete,
            integer_method=integerDeriv_method,
            log=log,
            **{k: v for k, v in kwargs.items() if k in grid_keys},
        )

    # ---- mpmath ----
    if method.lower() == "mpmath":
        from jumufraktiv.numeric_fractionalDeriv_mpmath import (
            fractionalDeriv_numeric_mpmath,
        )
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
    prior=None,
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
    #
    # `auto` keeps choosing a DIFFERENTIATING backend here, because this
    # function decides the derivative's *representation* as well as how it is
    # evaluated. Only the symbolic backend can build a representation before an
    # evaluation point is known, and `MGFDerivative` branches on that to offer
    # `post_density(theta)`, `post_mgf`, `post_cdf(u)` and sequential updating
    # symbolically.
    #
    # An earlier version of this sent `auto` straight to the expectation route.
    # That fixed the cancellation but silently removed the symbolic
    # representation from every `auto` posterior -- `_deriv_is_symbolic` became
    # permanently False, `int_tol` stopped affecting anything, and asking for a
    # density as an expression stopped working. The sweep that justified the
    # switch measured numbers, not capabilities, so it could not have caught
    # that; the suite did.
    #
    # The expectation route is preferred at the point of NUMERIC EVALUATION
    # instead, in `mgfDerivative` below, where `t` is known. That gets the
    # accuracy without giving up the representation.
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
        valid_int_methods = {"symbolic", "bell", "jax", "expectation"}
        if method not in valid_int_methods:
            raise ValueError(
                f"Invalid method '{method}' for integer order. "
                f"Choose from {valid_int_methods}."
            )

    else:
        valid_frac_methods = {"scipy", "mpmath", "symbolic", "expectation"}
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

    If `order` is array‑like, each order is dispatched independently and the
    results are returned in the shape of the broadcast request. An `order` of
    shape ``(2, 3)`` against a scalar `t` returns arrays of shape ``(2, 3)``;
    an `order` of shape ``(3,)`` against a `t` of shape ``(3,)`` pairs them
    elementwise and returns shape ``(3,)``.

    Elements are not coerced. A fractional order stays fractional, `t` may be
    `None` — in which case an array of expressions is returned, per the
    symbol‑numeric principle — and `u` is passed through unchanged.

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
    int_tol : float, optional
        Tolerance for detecting integer order. If `|order - round(order)| < int_tol`,
        the order is treated as an integer.
    u : float or array-like, optional
        Truncation point(s) for the incomplete MGF (used when `complete=False`).
        If array‑like, it is broadcast with `t` to form a batch of evaluation
        points `(t, u)`.
    **kwargs : additional keyword arguments passed to the underlying backend.
        Only the names in `BACKEND_OPTIONS` are accepted; anything else raises
        `TypeError`. For integer methods: `symbolic_timeout`, `cgf_method`.
        For fractional methods: `tol`, and `dps` / `use_tan` for `'mpmath'`.

    Returns
    -------
    sympy.Expr, tuple (log_abs, sign), or float / np.ndarray
        - If `order` is symbolic or `t` is `None` or free symbols remain:
          `sympy.Expr`, or — for an array-like `order` — an object array of
          `sympy.Expr` in the shape of the request.
        - If numeric evaluation:
            - `log=True`: `(log_abs, sign)` (scalars or arrays).
            - `log=False`: numeric value (scalar or array).

        Array results carry the shape of the broadcast request, not a flattened
        or stacked one.

    Notes
    -----
    - The canonical symbols `t` and `u` are imported from `jumufraktiv.symbols`.
    - The `'auto'` method chooses `'symbolic'` for integer orders (if available)
      and `'scipy'` for fractional orders by default.
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

    >>> # Multiple orders at once (array-like order). Fractional orders stay
    >>> # fractional; 1.5 gets the order-1.5 derivative, not the order-1 one.
    >>> orders = np.array([1.0, 1.5, 2.0])
    >>> log_abs, sign = mgfDerivative(orders, prior, method='scipy', t=-1.0, log=True)
    # Returns arrays of shape (3,) for log_abs and sign

    >>> # The shape of the request is the shape of the answer
    >>> orders = np.array([[0.5, 1.5], [1.9, 2.5]])
    >>> log_abs, sign = mgfDerivative(orders, prior, method='auto', t=-2.0, log=True)
    # Returns arrays of shape (2, 2), not (4,)
    """
    # Before anything else, so a misspelled option is refused whether the order
    # is scalar or array-valued, and whether the request ends up symbolic.
    _reject_unknown_options(kwargs, "mgfDerivative")

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
                int_tol=int_tol,
                u=uu,
                **kwargs,
            )
            # strict=True is safe and worth having: all three come from the
            # same `np.broadcast_arrays` call above, so unequal lengths
            # would mean the broadcast itself had gone wrong.
            for o, tt, uu in zip(order_arr.flat, t_arr.flat, u_flat, strict=True)
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

    requested_method = method.lower() if isinstance(method, str) else method
    order_type, method, integer_method = resolve_backend(
        order, method, integer_method, int_tol, prior=prior
    )

    # Dispatch
    #
    # The expectation route is preferred for NUMERIC evaluation whenever the
    # prior supplies a density, which every constructible prior does. The
    # condition is `t is not None`: with no evaluation point the caller is
    # asking for a representation, and only a differentiating backend can
    # produce one, so `auto` must not be diverted then.
    #
    # Measured over 240 cases -- four priors, ten orders from 0.5 to 30, six
    # evaluation points from -0.5 to -50, scored against mpmath at 60 digits
    # with each density written out independently:
    #
    #                          differentiating   expectation
    #   unacceptable (>1e-8)   5 of 240          0 of 240
    #   wrong sign             2                 0
    #   worst case             2.46e+00          1.17e-11
    #   median                 6.3e-17           1.0e-16
    #
    # All five failures are the Uniform prior at orders 12-30 with `t` near
    # zero, where its alternating CGF cancels through 25-26 digits; the
    # expectation route cannot cancel because its integrand is positive. It is
    # also 5-8x faster.
    #
    # An explicit `method=` is never diverted -- only `auto` is.
    prefer_expectation = (
        requested_method == "auto"
        and t is not None
        and prior is not None
        and expectation_is_available(prior)
    )
    if method == "expectation" or prefer_expectation:
        from jumufraktiv.numeric_expectation import expectationDeriv

        return expectationDeriv(
            order=float(order),
            prior=prior,
            t=t,
            u=u,
            complete=complete,
            log=log,
        )

    if order_type == "symbolic":
        # No warning here any more. It used to promise that the result "is
        # often the analytic continuation to non-integer orders", which was
        # never true: `sympy.diff` needs a concrete number of times to
        # differentiate, so an order carrying free symbols has always ended in
        # an exception rather than in an expression. Warning about the quality
        # of a result that does not exist told the caller nothing, and it fired
        # *before* the failure, so the last thing they saw was a claim about an
        # analytic continuation they never received.
        #
        # `integerDeriv_symbolic` now raises NotImplementedError naming the
        # free symbols and saying what to do instead. An order that is
        # integer-valued but not a Python `int` -- `sympy.Integer(2)`, which
        # SymPy arithmetic produces routinely -- is classified as symbolic
        # here and now succeeds, where before it hit the same dead end.
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
        # `int(...)` is NOT redundant here, whatever RUF046 says: for a
        # SymPy order `round()` returns a `sympy.Integer`, and
        # `mgfDerivative_integer` needs a Python `int`. PR 5 made
        # `sp.Integer(2)` behave like `2`; dropping this cast undoes it.
        int_order = int(round(order))  # noqa: RUF046
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

        # The near-integer interpolation branch is gone, and so is the module
        # behind it. It fitted a 4-point cubic spline *in the order* whenever
        # the fractional part exceeded max(d_vec) = 0.95, which was less
        # accurate than the plain quadrature just below that threshold, cost
        # four times the work, and took its sign from an endpoint.
        #
        # The difficulty it was working around is real: as the order approaches
        # an integer from below, gamma -> 0 and the answer is computed as
        # (1/Gamma(gamma)) x (a diverging integral), i.e. 0 x infinity. But that
        # has an exact fix rather than an interpolated one -- subtracting a
        # function with the same value at z = 0 and a known weighted integral --
        # which `numeric_fractionalDeriv_grid` applies. Measured relative error
        # at order 1.999: 0.96 through the spline, 2.9e-16 through the kernel.
        if method == "scipy":
            from jumufraktiv.numeric_fractionalDeriv_grid import fractionalDeriv_grid

            grid_keys = {"tol"}
            return fractionalDeriv_grid(
                order=order,
                prior=prior,
                t_points=t,
                u_points=u,
                complete=complete,
                integer_method=integer_method,
                log=log,
                **{k: v for k, v in kwargs.items() if k in grid_keys},
            )

        # `mpmath` and `symbolic` keep their own routes.
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
