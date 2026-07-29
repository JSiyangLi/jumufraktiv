"""
Laplace.py

Functions for preparing Laplace likelihood statistics for MGF marginalisation.

For a Laplace distribution with known mean μ and scale parameter θ,
the likelihood in terms of the rate parameter λ = 1/θ is:

    L(λ; y) = (λ/2) * exp(-λ |y - μ|) = c(y) * λ^{a(y)} * exp(-b(y) λ)

with a(y) = 1, b(y) = |y - μ|, c(y) = 1/2.

For a sample of size n, the joint likelihood is:
    L(λ; y) = (1/2)^n * λ^n * exp(-λ Σ |y_i - μ|)

Thus:
    a = n
    b = Σ |y_i - μ|
    log_c = n * log(1/2)
"""

import pandas as pd
import numpy as np
import math
import sympy as sp
from typing import Union, Dict, Any


def _is_1d_dataframe(obj: Any) -> bool:
    """Return True if obj is a pandas DataFrame with exactly 1 column."""
    return isinstance(obj, pd.DataFrame) and obj.shape[1] == 1


def _extract_1d(obj: Any) -> np.ndarray:
    """Extract a 1D numpy array from a pandas Series, DataFrame, or array-like."""
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError("DataFrame must have exactly 1 column.")
        return obj.iloc[:, 0].values.astype(float)
    elif isinstance(obj, pd.Series):
        return obj.values.astype(float)
    else:
        return np.asarray(obj, dtype=float)


def readyLaplace(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    mean: Union[float, int, pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, Union[float, int]]:
    """
    Compute sufficient statistics for a Laplace likelihood with known mean.

    The likelihood (in terms of rate parameter λ = 1/θ) is:
        L(λ; y) = (λ/2) * exp(-λ |y - μ|)

    For a sample of size n:
        a = n
        b = Σ |y_i - μ|
        log_c = n * log(1/2)

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
    if data_vals.ndim != 1:
        raise ValueError("data must be 1‑dimensional")
    n = len(data_vals)
    if n == 0:
        raise ValueError("data must be non‑empty")

    # ---- 2. Handle mean ----
    # Check if mean is a 1‑column DataFrame
    if _is_1d_dataframe(mean):
        mean_vals = _extract_1d(mean)
        if len(mean_vals) != n:
            raise ValueError("mean must have same length as data or be scalar")
    elif isinstance(mean, (int, float)):
        mean_vals = np.full(n, float(mean))
    else:
        # Try to treat as array‑like
        try:
            mean_vals = _extract_1d(mean)
            if len(mean_vals) != n:
                raise ValueError("mean must have same length as data or be scalar")
        except Exception:
            raise ValueError("mean must be a numeric scalar or 1‑dimensional array/DataFrame")

    # ---- 3. Compute sufficient statistics ----
    a = float(n)                     # n
    b = np.sum(np.abs(data_vals - mean_vals))   # Σ |y_i - μ|
    log_c = n * math.log(0.5)        # n * log(1/2)

    return {
        'a': a,
        'b': b,
        'log_c': log_c
    }
    
def bereitLaplace(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    mean: Union[float, int, pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute per‑element sufficient statistics for a Laplace likelihood.

    For each observation y_i and known mean μ_i:
        a_i = 1
        b_i = |y_i - μ_i|
        log_c_i = log(1/2) = -log(2)

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
        mean_vals = _extract_1d(mean)
        if len(mean_vals) != n:
            raise ValueError("mean must have same length as data or be scalar")
    elif isinstance(mean, (int, float)):
        mean_vals = np.full(n, float(mean))
    else:
        try:
            mean_vals = _extract_1d(mean)
            if len(mean_vals) != n:
                raise ValueError("mean must have same length as data or be scalar")
        except Exception:
            raise ValueError("mean must be a numeric scalar or 1‑dimensional array/DataFrame")

    # ---- Per‑element statistics ----
    a_vals = np.ones(n, dtype=float)
    b_vals = np.abs(data_vals - mean_vals)
    log_c_vals = np.full(n, -np.log(2.0))   # log(1/2)

    return {
        'a': a_vals,
        'b': b_vals,
        'log_c': log_c_vals
    }


def cLaplace() -> sp.Expr:
    """
    Return a symbolic expression for the Laplace normalising constant:

        ∏_{i=1}^{n} (1/2) = (1/2)^n

    where n is a symbolic integer.

    Returns
    -------
    sympy.Expr
        (1/2)^n
    """
    n = sp.Symbol('n', integer=True, positive=True)
    return (sp.Rational(1, 2)) ** n


# ===== Example usage =====
if __name__ == "__main__":
    # Example with scalar mean
    data_df = pd.DataFrame({'y': [1.0, 2.0, 3.0, 4.0]})
    mean_scalar = 2.5
    stats = readyLaplace(data_df, mean_scalar)
    print("Scalar mean:", stats)

    # Example with vector mean (unusual, but supported)
    mean_vec = pd.DataFrame({'mean': [2.0, 2.5, 3.0, 3.5]})
    stats2 = readyLaplace(data_df, mean_vec)
    print("Vector mean:", stats2)

    # Symbolic constant
    c_expr = cLaplace()
    print("\nSymbolic normalising constant:")
    sp.pprint(c_expr)