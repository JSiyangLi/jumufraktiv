"""
Gompertz.py

Functions for preparing Gompertz likelihood statistics for MGF marginalisation.

For a Gompertz distribution with known scale parameter β (scalar or vector) and unknown shape θ,
the density for y > 0 is:

    f(y; θ, β) = β * exp(β*y) * θ * exp(-θ * (exp(β*y) - 1))

This can be written as:
    L(θ; y) = c(y) * θ^{a(y)} * exp(-b(y) θ)

with a(y) = 1, b(y) = exp(β*y) - 1, c(y) = β * exp(β*y).

For a sample of size n:
    a = n
    b = Σ (exp(β_i * y_i) - 1)
    log_c = Σ log(β_i) + Σ β_i * y_i

If β is a scalar, it is recycled. If β is a vector, it must have length n.
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


def readyGompertz(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    scale: Union[float, int, pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, Union[float, int]]:
    """
    Compute sufficient statistics for a Gompertz likelihood with known scale.

    The likelihood (in terms of shape θ) is:
        L(θ; y) = β * exp(β*y) * θ * exp(-θ * (exp(β*y) - 1))

    For a sample of size n:
        a = n
        b = Σ (exp(β_i * y_i) - 1)
        log_c = Σ log(β_i) + Σ β_i * y_i

    Parameters
    ----------
    data : pandas DataFrame (1‑column), pandas Series, or array‑like
        Observed values (must be positive).
    scale : numeric scalar or 1‑column pandas DataFrame/Series/array‑like
        Known scale parameter(s) β. If scalar, it is recycled to match length of data.
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

    # ---- 3. Check positivity ----
    if np.any(scale_vals <= 0):
        raise ValueError("scale values must be positive.")
    if np.any(data_vals <= 0):
        raise ValueError("data values must be positive for Gompertz likelihood.")

    # ---- 4. Compute sufficient statistics ----
    a = float(n)
    exp_term = np.exp(scale_vals * data_vals)
    b = np.sum(exp_term - 1.0)
    # log_c = Σ log(β_i) + Σ β_i * y_i
    log_c = np.sum(np.log(scale_vals)) + np.sum(scale_vals * data_vals)

    return {
        'a': a,
        'b': b,
        'log_c': log_c
    }
    
def bereitGompertz(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    scale: Union[float, int, pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute per‑element sufficient statistics for a Gompertz likelihood.

    For each observation y_i and known scale β_i:
        a_i = 1
        b_i = exp(β_i * y_i) - 1
        log_c_i = log(β_i) + β_i * y_i

    Parameters
    ----------
    data : pandas DataFrame (1‑column), pandas Series, or array‑like
        Observed values (must be positive).
    scale : numeric scalar or 1‑column pandas DataFrame/Series/array‑like
        Known scale parameter(s) β. If scalar, recycled; if vector, same length as data.
    **kwargs : additional arguments (ignored).

    Returns
    -------
    dict
        Keys: 'a', 'b', 'log_c', each as a numpy array of length n.
    """
    data_vals = _extract_1d(data)
    n = len(data_vals)
    if n == 0:
        raise ValueError("data must be non‑empty")

    # ---- Handle scale ----
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

    # ---- Positivity checks ----
    if np.any(scale_vals <= 0):
        raise ValueError("scale values must be positive.")
    if np.any(data_vals <= 0):
        raise ValueError("data values must be positive for Gompertz likelihood.")

    # ---- Per‑element statistics ----
    a_vals = np.ones(n, dtype=float)
    exp_term = np.exp(scale_vals * data_vals)
    b_vals = exp_term - 1.0
    log_c_vals = np.log(scale_vals) + scale_vals * data_vals

    return {
        'a': a_vals,
        'b': b_vals,
        'log_c': log_c_vals
    }


def cGompertz() -> sp.Expr:
    """
    Return a symbolic expression for the Gompertz normalising constant:

        ∏_{i=1}^{n} β_i * exp(β_i * y_i)

    where n, β_i, and y_i are symbolic.

    Returns
    -------
    sympy.Expr
        ∏ (β_i * exp(β_i * y_i))
    """
    n = sp.Symbol('n', integer=True, positive=True)
    beta = sp.IndexedBase('beta')
    i = sp.Idx('i')
    y = sp.IndexedBase('y')
    expr = sp.Product(beta[i] * sp.exp(beta[i] * y[i]), (i, 1, n))
    return expr


# ===== Example usage =====
if __name__ == "__main__":
    # Scalar scale
    data_df = pd.DataFrame({'y': [0.5, 1.0, 1.5]})
    scale_scalar = 1.2
    stats = readyGompertz(data_df, scale_scalar)
    print("Scalar scale (β=1.2):", stats)

    # Vector scale
    scale_vec = pd.DataFrame({'beta': [1.0, 1.5, 2.0]})
    stats2 = readyGompertz(data_df, scale_vec)
    print("Vector scale:", stats2)

    # Symbolic constant
    c_expr = cGompertz()
    print("\nSymbolic normalising constant:")
    sp.pprint(c_expr)