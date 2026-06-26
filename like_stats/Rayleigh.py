"""
Rayleigh.py

Functions for preparing Rayleigh likelihood statistics for MGF marginalisation.

For a Rayleigh distribution with scale parameter σ, the density for y ≥ 0 is:

    f(y; σ) = (y / σ²) * exp(-y² / (2σ²))

In terms of rate θ = 1/σ², this becomes:

    f(y; θ) = y * θ * exp(-θ * y² / 2)

This can be written as:
    L(θ; y) = c(y) * θ^{a(y)} * exp(-b(y) θ)

with a(y) = 1, b(y) = y² / 2, c(y) = y.

For a sample of size n, the joint likelihood is:
    L(θ; y) = (∏ y_i) * θ^n * exp(-θ * (Σ y_i²) / 2)

Thus:
    a = n
    b = Σ y_i² / 2
    log_c = Σ log(y_i)
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


def readyRayleigh(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, Union[float, int]]:
    """
    Compute sufficient statistics for a Rayleigh likelihood.

    The likelihood (in terms of rate θ = 1/σ²) is:
        L(θ; y) = y * θ * exp(-θ * y² / 2)

    For a sample of size n:
        a = n
        b = Σ y_i² / 2
        log_c = Σ log(y_i)

    Parameters
    ----------
    data : pandas DataFrame (1‑column), pandas Series, or array‑like
        Observed values (must be positive).
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

    # ---- 2. Check positivity ----
    if np.any(data_vals <= 0):
        raise ValueError("data values must be positive for Rayleigh likelihood.")

    # ---- 3. Compute sufficient statistics ----
    a = float(n)
    b = np.sum(data_vals ** 2) / 2.0
    log_c = np.sum(np.log(data_vals))

    return {
        'a': a,
        'b': b,
        'log_c': log_c
    }


def cRayleigh() -> sp.Expr:
    """
    Return a symbolic expression for the Rayleigh normalising constant:

        ∏_{i=1}^{n} y_i

    where n is a symbolic integer.

    Returns
    -------
    sympy.Expr
        ∏ y_i
    """
    n = sp.Symbol('n', integer=True, positive=True)
    i = sp.Idx('i')
    y = sp.IndexedBase('y')
    expr = sp.Product(y[i], (i, 1, n))
    return expr


# ===== Example usage =====
if __name__ == "__main__":
    # Example data
    data_df = pd.DataFrame({'y': [1.0, 2.0, 3.0]})
    stats = readyRayleigh(data_df)
    print("Statistics for Rayleigh:", stats)

    # Symbolic constant
    c_expr = cRayleigh()
    print("\nSymbolic normalising constant:")
    sp.pprint(c_expr)