"""
BurrXII.py

Functions for preparing Burr Type XII likelihood statistics for MGF marginalisation.

For a Burr Type XII distribution with known shape parameter c (scalar or vector) and unknown shape k,
the density for y > 0 is:

    f(y; c, k) = c * k * y^{c-1} / (1 + y^c)^{k+1}

This can be written as:
    L(k; y) = C(y) * k^{a(y)} * exp(-b(y) k)

with a(y) = 1,
    b(y) = log(1 + y^c),
    C(y) = c * y^{c-1} / (1 + y^c).

For a sample of size n:
    a = n
    b = Σ log(1 + y_i^{c_i})
    log_C = Σ log(c_i) + Σ (c_i-1) log(y_i) - Σ log(1 + y_i^{c_i})

If c is a scalar, it is recycled. If c is a vector, it must have length n.

The user-facing argument is `known_shape`, which corresponds to the known shape parameter c.
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


def readyBurrXII(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    known_shape: Union[float, int, pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, Union[float, int]]:
    """
    Compute sufficient statistics for a Burr Type XII likelihood with known shape.

    The likelihood (in terms of unknown shape k) is:
        L(k; y) = [c * y^{c-1} / (1 + y^c)] * k * exp(-k * log(1 + y^c))

    Here `known_shape` is the known parameter c.

    For a sample of size n:
        a = n
        b = Σ log(1 + y_i^{c_i})
        log_C = Σ log(c_i) + Σ (c_i-1) log(y_i) - Σ log(1 + y_i^{c_i})

    Parameters
    ----------
    data : pandas DataFrame (1‑column), pandas Series, or array‑like
        Observed values (must be positive).
    known_shape : numeric scalar or 1‑column pandas DataFrame/Series/array‑like
        Known shape parameter(s) c. If scalar, it is recycled to match length of data.
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

    # ---- 2. Handle known_shape ----
    if _is_1d_dataframe(known_shape):
        c_vals = _extract_1d(known_shape)
        if len(c_vals) != n:
            raise ValueError("known_shape must have same length as data or be scalar")
    elif isinstance(known_shape, (int, float)):
        c_vals = np.full(n, float(known_shape))
    else:
        try:
            c_vals = _extract_1d(known_shape)
            if len(c_vals) != n:
                raise ValueError("known_shape must have same length as data or be scalar")
        except Exception:
            raise ValueError("known_shape must be a numeric scalar or 1‑dimensional array/DataFrame")

    # ---- 3. Check positivity ----
    if np.any(c_vals <= 0):
        raise ValueError("known_shape values must be positive.")
    if np.any(data_vals <= 0):
        raise ValueError("data values must be positive for Burr Type XII likelihood.")

    # ---- 4. Compute sufficient statistics ----
    a = float(n)
    # b = Σ log(1 + y_i^c)
    log_term = np.log(1 + data_vals ** c_vals)
    b = np.sum(log_term)
    # log_C = Σ log(c_i) + Σ (c_i-1) log(y_i) - Σ log(1 + y_i^c)
    log_c = np.sum(np.log(c_vals)) + np.sum((c_vals - 1.0) * np.log(data_vals)) - np.sum(log_term)

    return {
        'a': a,
        'b': b,
        'log_c': log_c
    }


def cBurrXII() -> sp.Expr:
    """
    Return a symbolic expression for the Burr XII normalising constant:

        ∏_{i=1}^{n} ( known_shape_i * y_i^{known_shape_i-1} / (1 + y_i^{known_shape_i}) )

    where n, known_shape_i, and y_i are symbolic.

    Returns
    -------
    sympy.Expr
        ∏ ( known_shape_i * y_i^{known_shape_i-1} / (1 + y_i^{known_shape_i}) )
    """
    n = sp.Symbol('n', integer=True, positive=True)
    known_shape = sp.IndexedBase('known_shape')
    i = sp.Idx('i')
    y = sp.IndexedBase('y')
    expr = sp.Product(known_shape[i] * y[i]**(known_shape[i] - 1) / (1 + y[i]**known_shape[i]), (i, 1, n))
    return expr


# ===== Example usage =====
if __name__ == "__main__":
    # Scalar known_shape
    data_df = pd.DataFrame({'y': [0.5, 1.0, 1.5]})
    known_shape_scalar = 2.0
    stats = readyBurrXII(data_df, known_shape=known_shape_scalar)
    print("Scalar known_shape (value=2.0):", stats)

    # Vector known_shape
    known_shape_vec = pd.DataFrame({'known_shape': [2.0, 3.0, 1.5]})
    stats2 = readyBurrXII(data_df, known_shape=known_shape_vec)
    print("Vector known_shape:", stats2)

    # Symbolic constant
    c_expr = cBurrXII()
    print("\nSymbolic normalising constant:")
    sp.pprint(c_expr)