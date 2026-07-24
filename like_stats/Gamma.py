"""
Gamma.py

Functions for preparing Gamma likelihood statistics for MGF marginalisation.
"""

import pandas as pd
import numpy as np
import sympy as sp
from typing import Union, Dict, Any
from scipy.special import gammaln


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

def readyGamma(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    shape: Union[float, int, pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, Union[float, int]]:
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
    data : pandas DataFrame (1‑column), pandas Series, or array‑like
        Observed values (must be positive).
    shape : numeric scalar or 1‑column pandas DataFrame/Series/array‑like
        Shape parameters α_i. If scalar, recycled; if vector, same length as data.

    Returns
    -------
    dict
        Keys: 'a', 'b', 'log_c'.
    """
    data_vals = _extract_1d(data)
    n = len(data_vals)
    if n == 0:
        raise ValueError("data must be non‑empty")

    # ---- Handle shape ----
    if _is_1d_dataframe(shape):
        shape_vals = _extract_1d(shape)
        if len(shape_vals) != n:
            raise ValueError("shape must have same length as data or be scalar")
    elif isinstance(shape, (int, float)):
        shape_vals = np.full(n, float(shape))
    else:
        try:
            shape_vals = _extract_1d(shape)
            if len(shape_vals) != n:
                raise ValueError("shape must have same length as data or be scalar")
        except Exception:
            raise ValueError("shape must be a numeric scalar or 1‑dimensional array/DataFrame")

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
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    shape: Union[float, int, pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute per‑element sufficient statistics for a Gamma likelihood.

    For each observation y_i and shape α_i:
        a_i = α_i
        b_i = y_i
        log_c_i = (α_i - 1) * log(y_i) - log Γ(α_i)

    Parameters
    ----------
    data : pandas DataFrame (1‑column), pandas Series, or array‑like
        Observed values (must be positive).
    shape : numeric scalar or 1‑column pandas DataFrame/Series/array‑like
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
        raise ValueError("data must be non‑empty")

    # ---- Handle shape ----
    if _is_1d_dataframe(shape):
        shape_vals = _extract_1d(shape)
        if len(shape_vals) != n:
            raise ValueError("shape must have same length as data or be scalar")
    elif isinstance(shape, (int, float)):
        shape_vals = np.full(n, float(shape))
    else:
        try:
            shape_vals = _extract_1d(shape)
            if len(shape_vals) != n:
                raise ValueError("shape must have same length as data or be scalar")
        except Exception:
            raise ValueError("shape must be a numeric scalar or 1‑dimensional array/DataFrame")

    # ---- Positivity checks ----
    if np.any(shape_vals <= 0):
        raise ValueError("shape values must be positive")
    if np.any(data_vals <= 0):
        raise ValueError("data values must be positive (Gamma likelihood requires y > 0)")

    # ---- Per‑element statistics ----
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


# ===== Example usage =====
if __name__ == "__main__":
    # Example with scalar shape
    data_df = pd.DataFrame({'y': [1.0, 2.0, 3.0, 4.0]})
    shape_scalar = 2.5
    stats = readyGamma(data_df, shape_scalar)
    print("Scalar shape:", stats)

    # Example with vector shape
    shape_df = pd.DataFrame({'alpha': [1.5, 2.0, 3.0, 2.5]})
    stats2 = readyGamma(data_df, shape_df)
    print("Vector shape:", stats2)

    # Symbolic constant
    c_expr = cGamma()
    print("\nSymbolic normalising constant:")
    sp.pprint(c_expr)