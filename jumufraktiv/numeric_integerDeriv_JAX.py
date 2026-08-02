"""
numeric_integerDeriv_JAX.py

Compute integer-order derivatives of MGFs using JAX.

This module provides a vectorised JAX-based backend for integer derivatives
of both complete and incomplete MGFs. It uses JAX's `jet` (Taylor mode) by
default, with a fallback to nested `grad` (reverse mode) when `jet` fails
(e.g., due to unsupported primitives like `igamma`).

The main function, `integerDeriv_numeric_jax`, follows the **tuple-vectorisation
principle**: evaluation points are `(t)` for complete MGFs and `(t, u)` for
incomplete MGFs. If `t` or `u` are array-like, they are broadcast to a common
shape and the derivative is evaluated for all points simultaneously using
`jax.vmap`.

Functions:
    - integerDeriv_numeric_jax : vectorised wrapper for derivative evaluation.
    - _integerDeriv_numeric_jax_scalar : scalar core (internal).
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import jet

logger = logging.getLogger(__name__)

jax.config.update("jax_enable_x64", True)

def _integerDeriv_numeric_jax_scalar(
    t,
    prior,
    order,
    complete: bool = True,
    u=None,
    jax_mode: str = "auto"
):
    """
    Scalar evaluation of an integer-order derivative using JAX.

    This is the core scalar routine used by the vectorised wrapper
    `integerDeriv_numeric_jax`. It computes the derivative for a single
    evaluation point `(t, u)` (or `(t)` for complete MGF) using either
    JAX's `jet` (Taylor mode) or nested `grad` (reverse mode).

    Parameters
    ----------
    t : float
        Evaluation point for the canonical variable `t`.
    prior : mitMGFprior
        Prior object providing JAX-compatible MGF functions.
    order : int
        Non-negative derivative order (must be scalar).
    complete : bool, optional
        If True, differentiate the complete MGF (`prior.mgf_jax`).
        If False, differentiate the incomplete MGF (`prior.imgf_jax`).
    u : float, optional
        Upper truncation point for the incomplete MGF (required when
        `complete=False`).
    jax_mode : str, optional
        Differentiation strategy:

        - `"auto"` (default): try `jet`, fallback to `grad` on failure.
        - `"jet"`: force JAX Taylor mode (`jet`).
        - `"grad"`: force nested reverse-mode `grad`.

    Returns
    -------
    tuple (log_abs, sign)
        `log_abs` is the natural logarithm of the absolute derivative,
        `sign` is ±1 (both are JAX arrays scalars).

    Raises
    ------
    ValueError
        If `order` is negative, or if required functions are missing,
        or if `jax_mode` is unknown.
    """
    if order < 0:
        raise ValueError("Order must be non-negative.")

    # ---------------------------------------------------------
    # Select function
    # ---------------------------------------------------------
    if complete:
        if prior.mgf_jax is None:
            raise ValueError("Prior does not provide a JAX-compatible MGF.")
        expr = prior.mgf_jax

    else:
        if u is None:
            raise ValueError("u must be supplied for incomplete MGF.")
        if prior.imgf_jax is None:
            raise ValueError("Prior does not provide a JAX-compatible incomplete MGF.")

        def expr(t_val):
            return prior.imgf_jax(t_val, u)

    # ---------------------------------------------------------
    # Zeroth derivative
    # ---------------------------------------------------------
    if order == 0:
        val = expr(t)

        if abs(val) < 1e-300:
            return -jnp.inf, 1
        sign = jnp.where(val >= 0, 1, -1)
        return jnp.log(jnp.abs(val)), sign

    # ---------------------------------------------------------
    # Select JAX differentiation strategy
    # ---------------------------------------------------------
    if jax_mode in ("auto", "jet"):

        try:
            series_in = ((1.0,) + (0.0,) * (order - 1),)

            _, series_out = jet.jet(
                expr,
                (t,),
                series_in,
            )

            coef = series_out[order - 1]

        except Exception as e:

            if jax_mode == "jet":
                raise

            # auto mode: fallback to grad only for jet failures
            msg = str(e).lower()

            unsupported = (
                isinstance(e, KeyError)
                or "jet" in msg
                or "primitive" in msg
                or "not implemented" in msg
                or "igamma" in msg
            )

            if not unsupported:
                raise

            logger.debug(
                "jet() failed (%s: %s); using nested grad() instead. "
                "Both compute the same derivatives.", type(e).__name__, e)

            deriv = expr
            for _ in range(order):
                deriv = jax.grad(deriv)

            coef = deriv(t)


    elif jax_mode == "grad":

        deriv = expr
        for _ in range(order):
            deriv = jax.grad(deriv)

        coef = deriv(t)


    else:
        raise ValueError(
            f"Unknown jax_mode='{jax_mode}'. "
            "Expected 'auto', 'jet', or 'grad'."
        )

    # ---------------------------------------------------------
    # Return result
    # ---------------------------------------------------------
    eps = 1e-300

    is_zero = jnp.abs(coef) < eps

    log_abs = jnp.where(
        is_zero,
        -jnp.inf,
        jnp.log(jnp.abs(coef))
    )

    sign = jnp.where(
        is_zero,
        1,
        jnp.where(coef >= 0, 1, -1)
    )

    return log_abs, sign


def integerDeriv_numeric_jax(t, prior, order, complete=True, u=None):
    """
    Evaluate a fixed-order derivative at one or more evaluation points.

    The evaluation point is:
        - complete MGF: (t)
        - incomplete MGF: (t, u)

    If either t or u is array-like, the function vectorises over the
    combined batch of evaluation points (tuple-vectorisation principle).

    Parameters
    ----------
    t : scalar or array-like
        Evaluation point(s) for t.
    prior : mitMGFprior
        The prior whose MGF is differentiated. Supplies `cgf_jax` for the
        complete MGF and `logimgf_jax` for the incomplete one.
    order : int
        Must be scalar.
    complete : bool, optional
        If True, differentiate the complete MGF. If False, differentiate
        the incomplete MGF (requires `u`).
    u : scalar or array-like, optional
        Upper limit(s) for the incomplete MGF.

        Together with t, defines the evaluation point (t,u).
        t and u are broadcast using NumPy broadcasting rules before
        vectorised evaluation.

    Returns
    -------
    (log_abs, sign)
        Scalars if both inputs are scalar.
        Arrays with the broadcasted shape if either input is array-like.

    Notes
    -----
    The scalar core `_integerDeriv_numeric_jax_scalar` uses `jax_mode='auto'`
    by default: it tries `jet` first and falls back to `grad` if `jet` fails
    due to unsupported primitives (e.g., `igamma`).
    """
    if np.ndim(order) != 0:
        raise ValueError(
            "integerDeriv_numeric_jax only accepts a scalar order. "
            "Vectorisation over derivative orders is handled by "
            "mgfDerivative_integer()."
        )

    # ---- Complete MGF: evaluation point is (t) ----
    if complete:
        if np.ndim(t) == 0:
            # Scalar fast path
            return _integerDeriv_numeric_jax_scalar(
                float(t), prior, order, complete=True, u=None
            )
        # Vectorise over t
        t_arr = jnp.asarray(t)
        vmapped = jax.vmap(
            lambda t_val: _integerDeriv_numeric_jax_scalar(
                t_val, prior, order, complete=True, u=None
            )
        )
        log_abs, sign = vmapped(t_arr)
        return np.asarray(log_abs), np.asarray(sign)

    # ---- Incomplete MGF: evaluation point is (t, u) ----
    if u is None:
        raise ValueError("u must be provided for incomplete MGF")

    # Broadcast t and u to a common shape (tuple-vectorisation)
    t_arr = np.asarray(t)
    u_arr = np.asarray(u)
    t_broad, u_broad = np.broadcast_arrays(t_arr, u_arr)

    # Flatten to 1D for vectorisation over points
    t_flat = jnp.asarray(t_broad).reshape(-1)
    u_flat = jnp.asarray(u_broad).reshape(-1)

    vmapped = jax.vmap(
        lambda t_val, u_val: _integerDeriv_numeric_jax_scalar(
            t_val, prior, order, complete=False, u=u_val
        )
    )
    log_abs, sign = vmapped(t_flat, u_flat)

    # Reshape back to the broadcasted shape
    log_abs = np.asarray(log_abs).reshape(t_broad.shape)
    sign = np.asarray(sign).reshape(t_broad.shape)

    return log_abs, sign
