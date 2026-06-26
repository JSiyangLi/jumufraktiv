"""
HalfNormal.py

Functions for preparing Half‑Normal likelihood statistics for MGF marginalisation.

For a Half‑Normal distribution with scale parameter σ, the density for y ≥ 0 is:

    f(y; σ) = (√(2) / (σ √π)) * exp(- y² / (2 σ²))

Writing θ = 1/σ², this becomes:
    L(θ; y) = c(y) * θ^{a(y)} * exp(-b(y) θ)

with a(y) = 1/2, b(y) = y² / 2, c(y) = √(2/π).

For a sample of size n:
    a = n/2
    b = Σ y_i² / 2
    log_c = n/2 * (log(2) - log(π))
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


def readyHalfNormal(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, Union[float, int]]:
    """
    Compute sufficient statistics for a Half‑Normal likelihood.

    The likelihood (in terms of precision θ = 1/σ²) is:
        L(θ; y) = √(2/π) * θ^{1/2} * exp(-θ * y² / 2)

    For a sample of size n:
        a = n/2
        b = Σ y_i² / 2
        log_c = n/2 * (log(2) - log(π))

    Parameters
    ----------
    data : pandas DataFrame (1‑column), pandas Series, or array‑like
        Observed values (must be non‑negative).
    **kwargs : additional arguments (ignored, for compatibility).

    Returns
    -------
    dict
        Keys: 'a', 'b', 'log_c'.

    Raises
    ------
    ValueError
        If data contains negative values.
    """
    # ---- 1. Extract data as 1D array ----
    data_vals = _extract_1d(data)
    if data_vals.ndim != 1:
        raise ValueError("data must be 1‑dimensional")
    n = len(data_vals)
    if n == 0:
        raise ValueError("data must be non‑empty")

    # ---- 2. Check non‑negativity ----
    if np.any(data_vals < 0):
        raise ValueError("data values must be non‑negative for Half‑Normal likelihood.")

    # ---- 3. Compute sufficient statistics ----
    a = n / 2.0
    b = np.sum(data_vals ** 2) / 2.0
    log_c = n / 2.0 * (math.log(2.0) - math.log(math.pi))

    return {
        'a': a,
        'b': b,
        'log_c': log_c
    }


def cHalfNormal() -> sp.Expr:
    """
    Return a symbolic expression for the Half‑Normal normalising constant:

        ∏_{i=1}^{n} √(2/π) = (√(2/π))^n

    where n is a symbolic integer.

    Returns
    -------
    sympy.Expr
        (√(2/π))^n
    """
    n = sp.Symbol('n', integer=True, positive=True)
    const = sp.sqrt(2 / sp.pi)
    return const ** n


# ===== Example usage =====
if __name__ == "__main__":
    # Example data
    data_df = pd.DataFrame({'y': [0.5, 1.0, 1.5, 2.0]})
    stats = readyHalfNormal(data_df)
    print("Statistics for Half‑Normal:", stats)

    # Symbolic constant
    c_expr = cHalfNormal()
    print("\nSymbolic normalising constant:")
    sp.pprint(c_expr)