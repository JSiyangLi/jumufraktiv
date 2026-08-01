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
from typing import Union, Dict

from jumufraktiv.like_stats._common import _extract_1d, _is_1d_dataframe


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
    n = len(data_vals)
    if n == 0:
        raise ValueError("data must be non‑empty")

    # ---- 2. Handle known_shape ----
    if _is_1d_dataframe(known_shape):
        c_vals = _extract_1d(known_shape, "known_shape")
        if len(c_vals) != n:
            raise ValueError("known_shape must have same length as data or be scalar")
    elif isinstance(known_shape, (int, float)):
        c_vals = _extract_1d(np.full(n, float(known_shape)), "known_shape")
    else:
        try:
            c_vals = _extract_1d(known_shape, "known_shape")
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
    
def bereitBurrXII(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    known_shape: Union[float, int, pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute per‑element sufficient statistics for a Burr Type XII likelihood.

    For each observation y_i and known shape c_i:
        a_i = 1
        b_i = log(1 + y_i^{c_i})
        log_c_i = log(c_i) + (c_i - 1)*log(y_i) - log(1 + y_i^{c_i})

    Parameters
    ----------
    data : pandas DataFrame (1‑column), pandas Series, or array‑like
        Observed values (must be positive).
    known_shape : numeric scalar or 1‑column pandas DataFrame/Series/array‑like
        Known shape parameter(s) c. If scalar, it is recycled to match length of data.
        If vector, must have same length as data.
    **kwargs : additional arguments (ignored).

    Returns
    -------
    dict
        Keys: 'a', 'b', 'log_c', each as a numpy array of length n.
    """
    # ---- Extract data ----
    data_vals = _extract_1d(data)
    n = len(data_vals)
    if n == 0:
        raise ValueError("data must be non‑empty")

    # ---- Handle known_shape ----
    if _is_1d_dataframe(known_shape):
        c_vals = _extract_1d(known_shape, "known_shape")
        if len(c_vals) != n:
            raise ValueError("known_shape must have same length as data or be scalar")
    elif isinstance(known_shape, (int, float)):
        c_vals = _extract_1d(np.full(n, float(known_shape)), "known_shape")
    else:
        try:
            c_vals = _extract_1d(known_shape, "known_shape")
            if len(c_vals) != n:
                raise ValueError("known_shape must have same length as data or be scalar")
        except Exception:
            raise ValueError("known_shape must be a numeric scalar or 1‑dimensional array/DataFrame")

    # ---- Check positivity ----
    if np.any(c_vals <= 0):
        raise ValueError("known_shape values must be positive.")
    if np.any(data_vals <= 0):
        raise ValueError("data values must be positive for Burr Type XII likelihood.")

    # ---- Compute per‑element statistics ----
    log_term = np.log(1 + data_vals ** c_vals)
    a_vals = np.ones(n, dtype=float)
    b_vals = log_term
    log_c_vals = np.log(c_vals) + (c_vals - 1.0) * np.log(data_vals) - log_term

    return {
        'a': a_vals,
        'b': b_vals,
        'log_c': log_c_vals
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
