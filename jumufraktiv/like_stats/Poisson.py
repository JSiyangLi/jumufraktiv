"""
Poisson.py

Functions for preparing Poisson likelihood statistics for MGF marginalisation.

For a Poisson likelihood with rate parameter θ and exposure s_i (scalar or vector),
the density for y_i ≥ 0 is:

    P(Y_i = y_i | θ, s_i) = (s_i θ)^{y_i} e^{-s_i θ} / y_i!

This can be written as:
    L(θ; y, s) = C(y, s) * θ^{a(y)} * exp(-b(y, s) θ)

with a(y) = Σ y_i,
    b(y, s) = Σ s_i,
    log_C(y, s) = Σ ( y_i log(s_i) - log(y_i!) ).

If scale (s_i) is a scalar, it is recycled. If scale is a vector, it must have
length equal to the number of observations.

The user‑facing argument is `scale`, which corresponds to the exposure s_i.

This module provides two statistics functions:
- `readyPoisson` : aggregated sufficient statistics (scalars) for the whole sample.
- `bereitPoisson` : per‑element sufficient statistics (arrays) for vectorised
  predictive evaluation (used in `post_predictive` when `individual=True`).

Additionally, `cPoisson()` returns a symbolic expression for the normalising constant.

Examples
--------
>>> data = [1, 2, 3]
>>> stats = readyPoisson(data, scale=1.0)
>>> stats['a']  # Σ y_i
6.0
>>> stats['b']  # Σ s_i
3.0
"""

import pandas as pd
import numpy as np
import math
from typing import Union, Dict
import sympy as sp
from scipy.special import gammaln

from jumufraktiv.like_stats._common import _extract_1d, _is_1d_dataframe


def readyPoisson(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    scale: Union[float, int, pd.DataFrame, pd.Series, list, np.ndarray] = 1.0,
    **kwargs
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
    data_vals = _extract_1d(data)
    n = len(data_vals)
    if n == 0:
        raise ValueError("data must be non‑empty")

    # ---- Handle scale ----
    if _is_1d_dataframe(scale):
        scale_vals = _extract_1d(scale, "scale")
        if len(scale_vals) != n:
            raise ValueError("scale must have same length as data or be scalar")
    elif isinstance(scale, (int, float)):
        scale_vals = _extract_1d(np.full(n, float(scale)), "scale")
    else:
        try:
            scale_vals = _extract_1d(scale, "scale")
            if len(scale_vals) != n:
                raise ValueError("scale must have same length as data or be scalar")
        except Exception:
            raise ValueError("scale must be a numeric scalar or 1‑dimensional array/DataFrame")

    # ---- Check validity ----
    if np.any(scale_vals <= 0):
        raise ValueError("scale values must be positive")
    if np.any(data_vals < 0):
        raise ValueError("data values must be non‑negative")

    # ---- Vectorized sums ----
    a = np.sum(data_vals)
    b = np.sum(scale_vals)
    log_c = np.sum(data_vals * np.log(scale_vals) - gammaln(data_vals + 1.0))

    return {
        'a': float(a),
        'b': float(b),
        'log_c': float(log_c)
    }

def bereitPoisson(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    scale: Union[float, int, pd.DataFrame, pd.Series, list, np.ndarray] = 1.0,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute per‑element sufficient statistics for a Poisson likelihood.

    For each observation y_i and exposure s_i:
        a_i = y_i
        b_i = s_i
        log_c_i = y_i * log(s_i) - log(y_i!)

    Parameters
    ----------
    data : pandas DataFrame (1‑column), pandas Series, or array‑like
        Observed counts (must be non‑negative).
    scale : numeric scalar or 1‑column pandas DataFrame/Series/array‑like, default 1.0
        Exposure values. If scalar, recycled; if vector, same length as data.
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
        scale_vals = _extract_1d(scale, "scale")
        if len(scale_vals) != n:
            raise ValueError("scale must have same length as data or be scalar")
    elif isinstance(scale, (int, float)):
        scale_vals = _extract_1d(np.full(n, float(scale)), "scale")
    else:
        try:
            scale_vals = _extract_1d(scale, "scale")
            if len(scale_vals) != n:
                raise ValueError("scale must have same length as data or be scalar")
        except Exception:
            raise ValueError("scale must be a numeric scalar or 1‑dimensional array/DataFrame")

    # ---- Check validity ----
    if np.any(scale_vals <= 0):
        raise ValueError("scale values must be positive")
    if np.any(data_vals < 0):
        raise ValueError("data values must be non‑negative")

    # ---- Per‑element statistics ----
    a_vals = data_vals.astype(float)
    b_vals = scale_vals
    # log(y!) = log Γ(y+1)
    log_c_vals = data_vals * np.log(scale_vals) - gammaln(data_vals + 1.0)

    return {
        'a': a_vals,
        'b': b_vals,
        'log_c': log_c_vals
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