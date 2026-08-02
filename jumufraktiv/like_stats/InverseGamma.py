"""
InverseGamma.py

Functions for preparing Inverse-Gamma likelihood statistics for MGF marginalisation.

For an Inverse-Gamma distribution with known shape α (scalar or vector) and
unknown rate β, the density for y > 0 is:

    f(y; α, β) = β^α / Γ(α) * y^{-α-1} * exp(-β / y)

This can be written as:
    L(β; y) = c(y) * β^{a(y)} * exp(-b(y) β)

with a(y) = α, b(y) = 1/y, c(y) = y^{-α-1} / Γ(α).

For a sample of size n:
    a = Σ α_i
    b = Σ 1/y_i
    log_c = Σ ( -(α_i+1) log(y_i) - log Γ(α_i) )

If α is a scalar, it is recycled. If α is a vector, it must have length n.
"""


import numpy as np
import pandas as pd
import sympy as sp
from scipy.special import gammaln

from jumufraktiv.like_stats._common import _extract_1d, _is_1d_dataframe


def readyInverseGamma(
    data: pd.DataFrame | pd.Series | list | np.ndarray,
    shape: float | int | pd.DataFrame | pd.Series | list | np.ndarray,
    **kwargs
) -> dict[str, float | int]:
    """
    Compute sufficient statistics for an Inverse-Gamma likelihood with known shape.

    The likelihood (in terms of rate β) is:
        L(β; y) = (y^{-α-1} / Γ(α)) * β^α * exp(-β / y)

    For a sample of size n:
        a = Σ α_i
        b = Σ 1/y_i
        log_c = Σ ( -(α_i+1) log(y_i) - log Γ(α_i) )

    Parameters
    ----------
    data : pandas DataFrame (1-column), pandas Series, or array-like
        Observed values (must be positive).
    shape : numeric scalar or 1-column pandas DataFrame/Series/array-like
        Known shape parameter(s) α. If scalar, it is recycled to match length of data.
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
        raise ValueError("shape values must be positive.")
    if np.any(data_vals <= 0):
        raise ValueError("data values must be positive for Inverse-Gamma likelihood.")

    # ---- Vectorized sums ----
    a = np.sum(shape_vals)
    b = np.sum(1.0 / data_vals)
    log_c = np.sum(-(shape_vals + 1.0) * np.log(data_vals) - gammaln(shape_vals))

    return {
        'a': float(a),
        'b': float(b),
        'log_c': float(log_c)
    }

def eachInverseGamma(
    data: pd.DataFrame | pd.Series | list | np.ndarray,
    shape: float | int | pd.DataFrame | pd.Series | list | np.ndarray,
    **kwargs
) -> dict[str, np.ndarray]:
    """
    Compute per-element sufficient statistics for an Inverse-Gamma likelihood.

    For each observation y_i and known shape α_i:
        a_i = α_i
        b_i = 1 / y_i
        log_c_i = -(α_i + 1) * log(y_i) - log Γ(α_i)

    Parameters
    ----------
    data : pandas DataFrame (1-column), pandas Series, or array-like
        Observed values (must be positive).
    shape : numeric scalar or 1-column pandas DataFrame/Series/array-like
        Known shape parameter(s) α. If scalar, recycled; if vector, same length as data.

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
        raise ValueError("shape values must be positive.")
    if np.any(data_vals <= 0):
        raise ValueError("data values must be positive for Inverse-Gamma likelihood.")

    # ---- Per-element statistics ----
    a_vals = shape_vals
    b_vals = 1.0 / data_vals
    log_c_vals = -(shape_vals + 1.0) * np.log(data_vals) - gammaln(shape_vals)

    return {
        'a': a_vals,
        'b': b_vals,
        'log_c': log_c_vals
    }


def cInverseGamma() -> sp.Expr:
    """
    Return a symbolic expression for the Inverse-Gamma normalising constant:

        ∏_{i=1}^{n} ( y_i^{-α_i-1} / Γ(α_i) )

    where n and α_i are symbolic.

    Returns
    -------
    sympy.Expr
        ∏ ( y_i^{-α_i-1} / Γ(α_i) )
    """
    n = sp.Symbol('n', integer=True, positive=True)
    alpha = sp.IndexedBase('alpha')
    i = sp.Idx('i')
    y = sp.IndexedBase('y')
    expr = sp.Product(y[i]**(-alpha[i] - 1) / sp.gamma(alpha[i]), (i, 1, n))
    return expr
