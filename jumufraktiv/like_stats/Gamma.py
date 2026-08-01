"""
Gamma.py

Functions for preparing Gamma likelihood statistics for MGF marginalisation.

For a Gamma likelihood with known shape α_i (scalar or vector) and unknown rate θ,
the density for y > 0 is:

    f(y; α, θ) = (θ^α / Γ(α)) * y^{α-1} * exp(-θ y)

This can be written as:
    L(θ; y) = C(y) * θ^{a(y)} * exp(-b(y) θ)

with a(y) = α,
    b(y) = y,
    C(y) = y^{α-1} / Γ(α).

For a sample of size n with per-observation shapes α_i:
    a = Σ α_i
    b = Σ y_i
    log_C = Σ ( (α_i - 1) * log(y_i) - log Γ(α_i) )

If shape is a scalar, it is recycled. If shape is a vector, it must have length n.

The user-facing argument is `shape`, which corresponds to the known shape parameter α.

This module provides two statistics functions:
- `readyGamma` : aggregated sufficient statistics (scalars) for the whole sample.
- `bereitGamma` : per-element sufficient statistics (arrays) for vectorised predictive evaluation.

Additionally, `cGamma()` returns a symbolic expression for the normalising constant.
"""


import numpy as np
import pandas as pd
import sympy as sp
from scipy.special import gammaln

from jumufraktiv.like_stats._common import _extract_1d, _is_1d_dataframe


def readyGamma(
    data: pd.DataFrame | pd.Series | list | np.ndarray,
    shape: float | int | pd.DataFrame | pd.Series | list | np.ndarray,
    **kwargs
) -> dict[str, float | int]:
    """
    Compute sufficient statistics for a Gamma likelihood (vectorized).

    For observation i:
        a_i = α_i
        b_i = y_i
        log_c_i = (α_i - 1) * log(y_i) - log Γ(α_i)

    Joint:
        a = Σ a_i
        b = Σ b_i
        log_c = Σ log_c_i

    Parameters
    ----------
    data : pandas DataFrame (1-column), pandas Series, or array-like
        Observed values (must be positive).
    shape : numeric scalar or 1-column pandas DataFrame/Series/array-like
        Shape parameters α_i. If scalar, recycled; if vector, same length as data.

    Returns
    -------
    dict
        Keys: 'a', 'b', 'log_c'.
    """
    data_vals = _extract_1d(data)
    n = len(data_vals)
    if n == 0:
        raise ValueError("data must be non-empty")

    # ---- Handle shape ----
    if _is_1d_dataframe(shape):
        shape_vals = _extract_1d(shape, "shape")
        if len(shape_vals) != n:
            raise ValueError("shape must have same length as data or be scalar")
    elif isinstance(shape, (int, float)):
        shape_vals = _extract_1d(np.full(n, float(shape)), "shape")
    else:
        shape_vals = _extract_1d(shape, "shape")
        if len(shape_vals) != n:
            raise ValueError("shape must have same length as data or be scalar")

    # ---- Positivity checks ----
    if np.any(shape_vals <= 0):
        raise ValueError("shape values must be positive")
    if np.any(data_vals <= 0):
        raise ValueError("data values must be positive (Gamma likelihood requires y > 0)")

    # ---- Vectorized sums ----
    a = np.sum(shape_vals)
    b = np.sum(data_vals)
    log_c = np.sum((shape_vals - 1.0) * np.log(data_vals) - gammaln(shape_vals))

    return {
        'a': float(a),
        'b': float(b),
        'log_c': float(log_c)
    }

def bereitGamma(
    data: pd.DataFrame | pd.Series | list | np.ndarray,
    shape: float | int | pd.DataFrame | pd.Series | list | np.ndarray,
    **kwargs
) -> dict[str, np.ndarray]:
    """
    Compute per-element sufficient statistics for a Gamma likelihood.

    For each observation y_i and shape α_i:
        a_i = α_i
        b_i = y_i
        log_c_i = (α_i - 1) * log(y_i) - log Γ(α_i)

    Parameters
    ----------
    data : pandas DataFrame (1-column), pandas Series, or array-like
        Observed values (must be positive).
    shape : numeric scalar or 1-column pandas DataFrame/Series/array-like
        Shape parameters α_i. If scalar, it is recycled to match length of data.
        If vector, must have same length as data.

    Returns
    -------
    dict
        Keys: 'a', 'b', 'log_c', each as a numpy array of length n.
    """
    data_vals = _extract_1d(data)
    n = len(data_vals)
    if n == 0:
        raise ValueError("data must be non-empty")

    # ---- Handle shape ----
    if _is_1d_dataframe(shape):
        shape_vals = _extract_1d(shape, "shape")
        if len(shape_vals) != n:
            raise ValueError("shape must have same length as data or be scalar")
    elif isinstance(shape, (int, float)):
        shape_vals = _extract_1d(np.full(n, float(shape)), "shape")
    else:
        shape_vals = _extract_1d(shape, "shape")
        if len(shape_vals) != n:
            raise ValueError("shape must have same length as data or be scalar")

    # ---- Positivity checks ----
    if np.any(shape_vals <= 0):
        raise ValueError("shape values must be positive")
    if np.any(data_vals <= 0):
        raise ValueError("data values must be positive (Gamma likelihood requires y > 0)")

    # ---- Per-element statistics ----
    a_vals = shape_vals
    b_vals = data_vals
    # log_c_i = (α_i - 1) * log(y_i) - log Γ(α_i)
    log_c_vals = (shape_vals - 1.0) * np.log(data_vals) - gammaln(shape_vals)

    return {'a': a_vals, 'b': b_vals, 'log_c': log_c_vals}


def cGamma() -> sp.Expr:
    """
    Return a symbolic expression for the Gamma normalising constant:

        ∏_{i=1}^{n} ( y_i^{α_i-1} / Γ(α_i) )

    where y_i and α_i are symbolic variables, and n is a symbolic integer.
    This expression can be used in symbolic MGF marginalisation.

    Returns
    -------
    sympy.Expr
        A product over i of ( y_i^{α_i-1} / Γ(α_i) ).
    """
    # Define symbolic variables
    n = sp.Symbol('n', integer=True, positive=True)
    i = sp.Idx('i')
    y = sp.IndexedBase('y')
    alpha = sp.IndexedBase('alpha')

    # Expression: ∏_{i=1}^{n} y_i^{α_i-1} / Γ(α_i)
    expr = sp.Product(y[i]**(alpha[i] - 1) / sp.gamma(alpha[i]), (i, 1, n))
    return expr
