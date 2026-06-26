"""
Pareto.py

Functions for preparing Pareto likelihood statistics for MGF marginalisation.

For a Pareto distribution with known scale parameter σ (scalar or vector) and unknown shape θ,
the density for y ≥ σ is:

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


def readyPareto(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    scale: Union[float, int, pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, Union[float, int]]:
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
    data : pandas DataFrame (1‑column), pandas Series, or array‑like
        Observed values (must be >= scale).
    scale : numeric scalar or 1‑column pandas DataFrame/Series/array‑like
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
    if data_vals.ndim != 1:
        raise ValueError("data must be 1‑dimensional")
    n = len(data_vals)
    if n == 0:
        raise ValueError("data must be non‑empty")

    # ---- 2. Handle scale ----
    if _is_1d_dataframe(scale):
        scale_vals = _extract_1d(scale)
        if len(scale_vals) != n:
            raise ValueError("scale must have same length as data or be scalar")
    elif isinstance(scale, (int, float)):
        scale_vals = np.full(n, float(scale))
    else:
        try:
            scale_vals = _extract_1d(scale)
            if len(scale_vals) != n:
                raise ValueError("scale must have same length as data or be scalar")
        except Exception:
            raise ValueError("scale must be a numeric scalar or 1‑dimensional array/DataFrame")

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

    return {
        'a': a,
        'b': b,
        'log_c': log_c
    }


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
    n = sp.Symbol('n', integer=True, positive=True)
    i = sp.Idx('i')
    y = sp.IndexedBase('y')
    expr = sp.Product(1 / y[i], (i, 1, n))
    return expr


# ===== Example usage =====
if __name__ == "__main__":
    # Scalar scale
    data_df = pd.DataFrame({'y': [2.0, 3.0, 4.0]})
    scale_scalar = 1.0
    stats = readyPareto(data_df, scale_scalar)
    print("Scalar scale (σ=1):", stats)

    # Vector scale
    scale_vec = pd.DataFrame({'scale': [1.0, 2.0, 1.5]})
    stats2 = readyPareto(data_df, scale_vec)
    print("Vector scale:", stats2)

    # Symbolic constant
    c_expr = cPareto()
    print("\nSymbolic normalising constant:")
    sp.pprint(c_expr)