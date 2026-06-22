"""
Poisson.py

Functions for preparing Poisson likelihood statistics for MGF marginalisation.
"""

import pandas as pd
import numpy as np
import math
from typing import Union, Dict, Any
import sympy as sp


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


def readyPoisson(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    scale: Union[float, int, pd.DataFrame, pd.Series, list, np.ndarray] = 1.0
) -> Dict[str, Union[float, int]]:
    """
    Compute sufficient statistics for a Poisson likelihood.

    L(θ; y, s) = ∏ ( (s_i θ)^{y_i} e^{-s_i θ} / y_i! )
               = θ^{Σ y_i} exp(-θ Σ s_i) * ∏ ( s_i^{y_i} / y_i! )

    Returns a dictionary:
        a = Σ y_i
        b = Σ s_i
        log_c = Σ ( y_i log(s_i) - log(y_i!) )

    Parameters
    ----------
    data : pandas DataFrame (1‑column), pandas Series, or array‑like
        Observed counts (must be non‑negative).
    scale : numeric scalar or 1‑column pandas DataFrame/Series/array‑like, default 1.0
        Exposure values. If scalar, it is recycled to match length of data.
        If vector, must have same length as data.

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
    # Check if scale is a 1‑column DataFrame
    if _is_1d_dataframe(scale):
        scale_vals = _extract_1d(scale)
        if len(scale_vals) != n:
            raise ValueError("scale must have same length as data or be scalar")
        b = np.sum(scale_vals)
    # Check if scale is numeric (int or float)
    elif isinstance(scale, (int, float)):
        scale_vals = np.full(n, float(scale))
        b = n * float(scale)
    else:
        # Try to treat as array‑like (maybe Series or list)
        try:
            scale_vals = _extract_1d(scale)
            if len(scale_vals) != n:
                raise ValueError("scale must have same length as data or be scalar")
            b = np.sum(scale_vals)
        except Exception:
            raise ValueError("scale must be a numeric scalar or 1‑dimensional array/DataFrame")

    # ---- 3. Check positivity ----
    if np.any(scale_vals <= 0):
        raise ValueError("scale values must be positive")
    if np.any(data_vals < 0):
        raise ValueError("data values must be non‑negative")

    # ---- 4. Compute sufficient statistics ----
    a = np.sum(data_vals)

    # log_c = Σ ( y_i log(s_i) - log(y_i!) )
    # Use lfactorial = log(y!) = lgamma(y+1)
    def lfactorial(x: float) -> float:
        return math.lgamma(x + 1)  # log Γ(x+1) = log(x!)

    log_c = np.sum(data_vals * np.log(scale_vals) - np.array([lfactorial(y) for y in data_vals]))

    return {
        'a': a,
        'b': b,
        'log_c': log_c
    }


def cPoisson() -> sp.Expr:
    """
    Return a symbolic expression for the Poisson normalising constant:

        ∏_{i=1}^{n} (s_i^{y_i} / y_i!)

    where s_i and y_i are symbolic variables, and n is a symbolic integer.
    This expression can be used in symbolic MGF marginalisation.

    Returns
    -------
    sympy.Expr
        A product over i of (s_i^{y_i} / y_i!).
    """
    # Define symbolic variables
    n = sp.Symbol('n', integer=True, positive=True)
    i = sp.Idx('i')
    y = sp.IndexedBase('y')
    s = sp.IndexedBase('s')

    # Expression: ∏_{i=1}^{n} s_i^{y_i} / y_i!
    expr = sp.Product(s[i]**y[i] / sp.factorial(y[i]), (i, 1, n))
    return expr


# ===== Example usage =====
if __name__ == "__main__":
    # ---- readyPoisson examples ----
    data_df = pd.DataFrame({'counts': [1, 2, 3, 4]})
    stats_default = readyPoisson(data_df)   # scale defaults to 1.0
    print("With default scale (1.0):", stats_default)

    stats_scalar = readyPoisson(data_df, scale=2.0)
    print("With scalar scale 2.0:", stats_scalar)

    scale_df = pd.DataFrame({'exposure': [0.5, 1.0, 1.5, 2.0]})
    stats_vector = readyPoisson(data_df, scale=scale_df)
    print("With vector scale:", stats_vector)

    # ---- cPoisson example ----
    c_expr = cPoisson()
    print("\nSymbolic normalising constant:")
    sp.pprint(c_expr)
    # Optional: show that it can be substituted later