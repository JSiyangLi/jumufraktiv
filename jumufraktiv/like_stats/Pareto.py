"""
Pareto.py

Functions for preparing Pareto likelihood statistics for MGF marginalisation.

For a Pareto distribution with known scale parameter σ (scalar or vector) and
unknown shape θ, the density for y ≥ σ is:

    f(y; θ, σ) = θ * σ^θ / y^(θ+1) = θ * (σ/y)^θ * (1/y)

This can be written as:
    L(θ; y) = c(y) * θ^{a(y)} * exp(-b(y) θ)

with a(y) = 1, b(y) = -log(σ/y) = log(y/σ), c(y) = 1/y.

For a sample of size n:
    a = n
    b = Σ log(y_i/σ_i) = Σ log(y_i) - Σ log(σ_i)
    log_c = -Σ log(y_i)

If σ is a scalar, it is recycled. If σ is a vector, it must have length n.
"""

import numpy as np
import pandas as pd
import sympy as sp

from jumufraktiv.like_stats._common import _extract_1d, _is_1d_dataframe


def readyPareto(
    data: pd.DataFrame | pd.Series | list | np.ndarray,
    scale: float | int | pd.DataFrame | pd.Series | list | np.ndarray,
    **kwargs,
) -> dict[str, float | int]:
    """
    Compute sufficient statistics for a Pareto likelihood with known scale.

    The likelihood (in terms of shape θ) is:
        L(θ; y) = (1/y) * θ * exp(-θ * log(y/σ))

    For a sample of size n:
        a = n
        b = Σ log(y_i/σ_i) = Σ log(y_i) - Σ log(σ_i)
        log_c = -Σ log(y_i)

    Parameters
    ----------
    data : pandas DataFrame (1-column), pandas Series, or array-like
        Observed values (must be >= scale).
    scale : numeric scalar or 1-column pandas DataFrame/Series/array-like
        Known scale parameter(s) σ. If scalar, it is recycled to match length of data.
        If vector, must have same length as data.
    **kwargs : additional arguments (ignored, for compatibility).

    Returns
    -------
    dict
        Keys: 'a', 'b', 'log_c'.

    Raises
    ------
    ValueError
        If inputs are incompatible or contain invalid values.
    """
    # ---- 1. Extract data as 1D array ----
    data_vals = _extract_1d(data)
    n = len(data_vals)
    if n == 0:
        raise ValueError("data must be non-empty")

    # ---- 2. Handle scale ----
    if _is_1d_dataframe(scale):
        scale_vals = _extract_1d(scale, "scale")
        if len(scale_vals) != n:
            raise ValueError("scale must have same length as data or be scalar")
    elif isinstance(scale, (int, float)):
        scale_vals = _extract_1d(np.full(n, float(scale)), "scale")
    else:
        scale_vals = _extract_1d(scale, "scale")
        if len(scale_vals) != n:
            raise ValueError("scale must have same length as data or be scalar")

    # ---- 3. Check support ----
    if np.any(scale_vals <= 0):
        raise ValueError("scale values must be positive.")
    if np.any(data_vals < scale_vals):
        raise ValueError("data values must be >= scale for Pareto likelihood.")

    # ---- 4. Compute sufficient statistics ----
    a = float(n)
    # b = Σ log(y_i) - Σ log(σ_i)
    b = np.sum(np.log(data_vals)) - np.sum(np.log(scale_vals))
    # log_c = -Σ log(y_i)
    log_c = -np.sum(np.log(data_vals))

    return {"a": a, "b": b, "log_c": log_c}


def eachPareto(
    data: pd.DataFrame | pd.Series | list | np.ndarray,
    scale: float | int | pd.DataFrame | pd.Series | list | np.ndarray,
    **kwargs,
) -> dict[str, np.ndarray]:
    """
    Compute per-element sufficient statistics for a Pareto likelihood.

    For each observation y_i and known scale σ_i:
        a_i = 1
        b_i = log(y_i / σ_i)
        log_c_i = -log(y_i)

    Parameters
    ----------
    data : pandas DataFrame (1-column), pandas Series, or array-like
        Observed values (must be >= scale).
    scale : numeric scalar or 1-column pandas DataFrame/Series/array-like
        Known scale parameter(s) σ. If scalar, recycled; if vector, same length as data.
    **kwargs : additional arguments (ignored).

    Returns
    -------
    dict
        Keys: 'a', 'b', 'log_c', each as a numpy array of length n.
    """
    data_vals = _extract_1d(data)
    n = len(data_vals)
    if n == 0:
        raise ValueError("data must be non-empty")

    # ---- Handle scale ----
    if _is_1d_dataframe(scale):
        scale_vals = _extract_1d(scale, "scale")
        if len(scale_vals) != n:
            raise ValueError("scale must have same length as data or be scalar")
    elif isinstance(scale, (int, float)):
        scale_vals = _extract_1d(np.full(n, float(scale)), "scale")
    else:
        scale_vals = _extract_1d(scale, "scale")
        if len(scale_vals) != n:
            raise ValueError("scale must have same length as data or be scalar")

    # ---- Check support ----
    if np.any(scale_vals <= 0):
        raise ValueError("scale values must be positive.")
    if np.any(data_vals < scale_vals):
        raise ValueError("data values must be >= scale for Pareto likelihood.")

    # ---- Per-element statistics ----
    a_vals = np.ones(n, dtype=float)
    b_vals = np.log(data_vals / scale_vals)
    log_c_vals = -np.log(data_vals)

    return {"a": a_vals, "b": b_vals, "log_c": log_c_vals}


def cPareto() -> sp.Expr:
    """
    Return a symbolic expression for the Pareto normalising constant:

        ∏_{i=1}^{n} (1/y_i) = (∏ y_i)^{-1}

    where n is a symbolic integer.

    Returns
    -------
    sympy.Expr
        ∏ 1/y_i
    """
    n = sp.Symbol("n", integer=True, positive=True)
    i = sp.Idx("i")
    y = sp.IndexedBase("y")
    expr = sp.Product(1 / y[i], (i, 1, n))
    return expr
