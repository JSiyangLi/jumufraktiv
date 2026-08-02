"""
Levy.py

Functions for preparing Lévy likelihood statistics for MGF marginalisation.

For a Lévy distribution with known location μ (scalar or vector) and unknown scale θ,
the density for y > μ is:

    f(y; μ, θ) = sqrt(θ / (2π)) * (y - μ)^(-3/2) * exp(-θ / (2 * (y - μ)))

This can be written as:
    L(θ; y) = c(y) * θ^{a(y)} * exp(-b(y) θ)

with a(y) = 1/2, b(y) = 1 / (2*(y - μ)), c(y) = (2π)^(-1/2) * (y - μ)^(-3/2).

For a sample of size n:
    a = n/2
    b = Σ 1 / (2*(y_i - μ_i))
    log_c = -(n/2) * log(2π) - (3/2) * Σ log(y_i - μ_i)

If μ is a scalar, it is recycled. If μ is a vector, it must have length n.
"""

import math

import numpy as np
import pandas as pd
import sympy as sp

from jumufraktiv.like_stats._common import _extract_1d, _is_1d_dataframe


def readyLevy(
    data: pd.DataFrame | pd.Series | list | np.ndarray,
    location: float | int | pd.DataFrame | pd.Series | list | np.ndarray,
    **kwargs
) -> dict[str, float | int]:
    """
    Compute sufficient statistics for a Lévy likelihood with known location.

    The likelihood (in terms of scale θ) is:
        L(θ; y) = (2π)^(-1/2) * (y - μ)^(-3/2) * θ^{1/2} * exp(-θ / (2*(y - μ)))

    For a sample of size n:
        a = n/2
        b = Σ 1 / (2*(y_i - μ_i))
        log_c = -(n/2) * log(2π) - (3/2) * Σ log(y_i - μ_i)

    Parameters
    ----------
    data : pandas DataFrame (1-column), pandas Series, or array-like
        Observed values (must be > location).
    location : numeric scalar or 1-column pandas DataFrame/Series/array-like
        Known location parameter(s) μ. If scalar, it is recycled to match
        length of data. If vector, must have same length as data.
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

    # ---- 2. Handle location ----
    if _is_1d_dataframe(location):
        loc_vals = _extract_1d(location, "location")
        if len(loc_vals) != n:
            raise ValueError("location must have same length as data or be scalar")
    elif isinstance(location, (int, float)):
        loc_vals = _extract_1d(np.full(n, float(location)), "location")
    else:
        loc_vals = _extract_1d(location, "location")
        if len(loc_vals) != n:
            raise ValueError("location must have same length as data or be scalar")

    # ---- 3. Check support ----
    diff = data_vals - loc_vals
    if np.any(diff <= 0):
        raise ValueError(
            "data values must be strictly greater than location for Lévy likelihood."
        )

    # ---- 4. Compute sufficient statistics ----
    a = n / 2.0
    b = np.sum(1.0 / (2.0 * diff))
    # log_c = -(n/2) * log(2π) - (3/2) * Σ log(diff)
    log_c = -n / 2.0 * math.log(2.0 * math.pi) - 1.5 * np.sum(np.log(diff))

    return {
        'a': a,
        'b': b,
        'log_c': log_c
    }

def eachLevy(
    data: pd.DataFrame | pd.Series | list | np.ndarray,
    location: float | int | pd.DataFrame | pd.Series | list | np.ndarray,
    **kwargs
) -> dict[str, np.ndarray]:
    """
    Compute per-element sufficient statistics for a Lévy likelihood.

    For each observation y_i and known location μ_i:
        a_i = 0.5
        b_i = 1 / (2 * (y_i - μ_i))
        log_c_i = -0.5 * log(2π) - 1.5 * log(y_i - μ_i)

    Parameters
    ----------
    data : pandas DataFrame (1-column), pandas Series, or array-like
        Observed values (must be > location).
    location : numeric scalar or 1-column pandas DataFrame/Series/array-like
        Known location parameter(s) μ. If scalar, recycled; if vector, same
        length as data.
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

    # ---- Handle location ----
    if _is_1d_dataframe(location):
        loc_vals = _extract_1d(location, "location")
        if len(loc_vals) != n:
            raise ValueError("location must have same length as data or be scalar")
    elif isinstance(location, (int, float)):
        loc_vals = _extract_1d(np.full(n, float(location)), "location")
    else:
        loc_vals = _extract_1d(location, "location")
        if len(loc_vals) != n:
            raise ValueError("location must have same length as data or be scalar")

    # ---- Check support ----
    diff = data_vals - loc_vals
    if np.any(diff <= 0):
        raise ValueError(
            "data values must be strictly greater than location for Lévy likelihood."
        )

    # ---- Per-element statistics ----
    a_vals = np.full(n, 0.5)
    b_vals = 1.0 / (2.0 * diff)
    log_c_vals = -0.5 * np.log(2.0 * np.pi) - 1.5 * np.log(diff)

    return {
        'a': a_vals,
        'b': b_vals,
        'log_c': log_c_vals
    }


def cLevy() -> sp.Expr:
    """
    Return a symbolic expression for the Lévy normalising constant:

        ∏_{i=1}^{n} ( (2π)^(-1/2) * (y_i - μ_i)^(-3/2) )
        = (2π)^(-n/2) * ∏ (y_i - μ_i)^(-3/2)

    where n and μ_i are symbolic.

    Returns
    -------
    sympy.Expr
        (2π)^(-n/2) * ∏ (y_i - μ_i)^(-3/2)
    """
    n = sp.Symbol('n', integer=True, positive=True)
    mu = sp.IndexedBase('mu')
    i = sp.Idx('i')
    y = sp.IndexedBase('y')
    const = (2 * sp.pi) ** (-n / 2)
    prod = sp.Product((y[i] - mu[i]) ** (-sp.Rational(3, 2)), (i, 1, n))
    return const * prod
