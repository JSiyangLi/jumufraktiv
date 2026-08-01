"""
Normal.py

Functions for preparing Normal likelihood statistics for MGF marginalisation.

For a Normal distribution with known mean μ and precision parameter θ = 1/σ²,
the likelihood in terms of θ is:

    L(θ; y) = sqrt(θ / (2π)) * exp(-θ (y - μ)² / 2)

This can be written as:
    L(θ; y) = c(y) * θ^{a(y)} * exp(-b(y) θ)

with a(y) = 1/2, b(y) = (y - μ)² / 2, c(y) = 1 / sqrt(2π).

For a sample of size n, the joint likelihood is:
    L(θ; y) = (1 / sqrt(2π))^n * θ^{n/2} * exp(-θ * (Σ (y_i - μ)²) / 2)

Thus:
    a = n/2
    b = Σ (y_i - μ)² / 2
    log_c = n * log(1 / sqrt(2π)) = -n/2 * (log(2) + log(π))
"""

import math
from typing import Dict, Union

import numpy as np
import pandas as pd
import sympy as sp

from jumufraktiv.like_stats._common import _extract_1d, _is_1d_dataframe


def readyNormal(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    mean: Union[float, int, pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, Union[float, int]]:
    """
    Compute sufficient statistics for a Normal likelihood with known mean.

    The likelihood (in terms of precision θ = 1/σ²) is:
        L(θ; y) = sqrt(θ / (2π)) * exp(-θ (y - μ)² / 2)

    For a sample of size n:
        a = n/2
        b = Σ (y_i - μ)² / 2
        log_c = -n/2 * (log(2) + log(π))

    Parameters
    ----------
    data : pandas DataFrame (1‑column), pandas Series, or array‑like
        Observed values.
    mean : numeric scalar or 1‑column pandas DataFrame/Series/array‑like
        Known mean μ. If scalar, it is recycled to match length of data.
        If vector, must have same length as data (though mean is typically scalar).
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
        raise ValueError("data must be non‑empty")

    # ---- 2. Handle mean ----
    if _is_1d_dataframe(mean):
        mean_vals = _extract_1d(mean, "mean")
        if len(mean_vals) != n:
            raise ValueError("mean must have same length as data or be scalar")
    elif isinstance(mean, (int, float)):
        mean_vals = _extract_1d(np.full(n, float(mean)), "mean")
    else:
        mean_vals = _extract_1d(mean, "mean")
        if len(mean_vals) != n:
            raise ValueError("mean must have same length as data or be scalar")

    # ---- 3. Compute sufficient statistics ----
    a = n / 2.0
    b = np.sum((data_vals - mean_vals) ** 2) / 2.0
    # log_c = -n/2 * log(2π) = -n/2 * (log(2) + log(π))
    log_c = -n / 2.0 * (math.log(2.0) + math.log(math.pi))

    return {
        'a': a,
        'b': b,
        'log_c': log_c
    }
    
def bereitNormal(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    mean: Union[float, int, pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute per‑element sufficient statistics for a Normal likelihood.

    For each observation y_i and known mean μ_i:
        a_i = 0.5
        b_i = (y_i - μ_i)^2 / 2
        log_c_i = -0.5 * (log(2) + log(π))

    Parameters
    ----------
    data : pandas DataFrame (1‑column), pandas Series, or array‑like
        Observed values.
    mean : numeric scalar or 1‑column pandas DataFrame/Series/array‑like
        Known mean μ. If scalar, recycled; if vector, same length as data.

    Returns
    -------
    dict
        Keys: 'a', 'b', 'log_c', each as a numpy array of length n.
    """
    data_vals = _extract_1d(data)
    n = len(data_vals)
    if n == 0:
        raise ValueError("data must be non‑empty")

    # ---- Handle mean ----
    if _is_1d_dataframe(mean):
        mean_vals = _extract_1d(mean, "mean")
        if len(mean_vals) != n:
            raise ValueError("mean must have same length as data or be scalar")
    elif isinstance(mean, (int, float)):
        mean_vals = _extract_1d(np.full(n, float(mean)), "mean")
    else:
        mean_vals = _extract_1d(mean, "mean")
        if len(mean_vals) != n:
            raise ValueError("mean must have same length as data or be scalar")

    # ---- Per‑element statistics ----
    a_vals = np.full(n, 0.5)
    b_vals = (data_vals - mean_vals) ** 2 / 2.0
    log_c_vals = np.full(n, -0.5 * (np.log(2.0) + np.log(np.pi)))

    return {
        'a': a_vals,
        'b': b_vals,
        'log_c': log_c_vals
    }


def cNormal() -> sp.Expr:
    """
    Return a symbolic expression for the Normal normalising constant:

        ∏_{i=1}^{n} (1 / sqrt(2π)) = (1 / sqrt(2π))^n

    where n is a symbolic integer.

    Returns
    -------
    sympy.Expr
        (1 / sqrt(2π))^n
    """
    n = sp.Symbol('n', integer=True, positive=True)
    return (1 / sp.sqrt(2 * sp.pi)) ** n
