"""
InverseGamma.py

Functions for preparing Inverse‑Gamma likelihood statistics for MGF marginalisation.

For an Inverse‑Gamma distribution with known shape α (scalar or vector) and unknown rate β,
the density for y > 0 is:

    f(y; α, β) = β^α / Γ(α) * y^{-α-1} * exp(-β / y)

This can be written as:
    L(β; y) = c(y) * β^{a(y)} * exp(-b(y) β)

with a(y) = α, b(y) = 1/y, c(y) = y^{-α-1} / Γ(α).

For a sample of size n:
    a = Σ α_i
    b = Σ 1/y_i
    log_c = Σ ( -(α_i+1) log(y_i) - log Γ(α_i) )

If α is a scalar, it is recycled. If α is a vector, it must have length n.
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


def readyInverseGamma(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    shape: Union[float, int, pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, Union[float, int]]:
    """
    Compute sufficient statistics for an Inverse‑Gamma likelihood with known shape.

    The likelihood (in terms of rate β) is:
        L(β; y) = (y^{-α-1} / Γ(α)) * β^α * exp(-β / y)

    For a sample of size n:
        a = Σ α_i
        b = Σ 1/y_i
        log_c = Σ ( -(α_i+1) log(y_i) - log Γ(α_i) )

    Parameters
    ----------
    data : pandas DataFrame (1‑column), pandas Series, or array‑like
        Observed values (must be positive).
    shape : numeric scalar or 1‑column pandas DataFrame/Series/array‑like
        Known shape parameter(s) α. If scalar, it is recycled to match length of data.
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

    # ---- 2. Handle shape ----
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

    # ---- 3. Check positivity ----
    if np.any(shape_vals <= 0):
        raise ValueError("shape values must be positive.")
    if np.any(data_vals <= 0):
        raise ValueError("data values must be positive for Inverse‑Gamma likelihood.")

    # ---- 4. Compute sufficient statistics ----
    a = np.sum(shape_vals)
    b = np.sum(1.0 / data_vals)
    # log_c = Σ ( -(α_i+1) log(y_i) - log Γ(α_i) )
    log_c = np.sum(-(shape_vals + 1.0) * np.log(data_vals) - np.array([math.lgamma(alpha) for alpha in shape_vals]))

    return {
        'a': a,
        'b': b,
        'log_c': log_c
    }


def cInverseGamma() -> sp.Expr:
    """
    Return a symbolic expression for the Inverse‑Gamma normalising constant:

        ∏_{i=1}^{n} ( y_i^{-α_i-1} / Γ(α_i) )

    where n and α_i are symbolic.

    Returns
    -------
    sympy.Expr
        ∏ ( y_i^{-α_i-1} / Γ(α_i) )
    """
    n = sp.Symbol('n', integer=True, positive=True)
    alpha = sp.IndexedBase('alpha')
    i = sp.Idx('i')
    y = sp.IndexedBase('y')
    expr = sp.Product(y[i]**(-alpha[i] - 1) / sp.gamma(alpha[i]), (i, 1, n))
    return expr


# ===== Example usage =====
if __name__ == "__main__":
    # Scalar shape
    data_df = pd.DataFrame({'y': [1.0, 2.0, 3.0]})
    shape_scalar = 2.0
    stats = readyInverseGamma(data_df, shape_scalar)
    print("Scalar shape (α=2):", stats)

    # Vector shape
    shape_vec = pd.DataFrame({'alpha': [1.5, 2.0, 2.5]})
    stats2 = readyInverseGamma(data_df, shape_vec)
    print("Vector shape:", stats2)

    # Symbolic constant
    c_expr = cInverseGamma()
    print("\nSymbolic normalising constant:")
    sp.pprint(c_expr)