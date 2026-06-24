"""
Gamma.py

Functions for preparing Gamma likelihood statistics for MGF marginalisation.
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


def readyGamma(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    shape: Union[float, int, pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, Union[float, int]]:
    """
    Compute sufficient statistics for a Gamma likelihood.

    Likelihood for observation i:
        p(y_i | θ, α_i) = (θ^{α_i} / Γ(α_i)) * y_i^{α_i-1} * exp(-θ y_i)

    Joint likelihood for n independent observations:
        L(θ) = θ^{Σ α_i} * exp(-θ Σ y_i) * ∏_{i=1}^n ( y_i^{α_i-1} / Γ(α_i) )

    Returns a dictionary:
        a = Σ α_i
        b = Σ y_i
        log_c = Σ ( (α_i - 1) * log(y_i) - log Γ(α_i) )

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
    # Check if shape is a 1‑column DataFrame
    if _is_1d_dataframe(shape):
        shape_vals = _extract_1d(shape)
        if len(shape_vals) != n:
            raise ValueError("shape must have same length as data or be scalar")
        a = np.sum(shape_vals)
    # Check if shape is numeric (int or float)
    elif isinstance(shape, (int, float)):
        shape_vals = np.full(n, float(shape))
        a = n * float(shape)
    else:
        # Try to treat as array‑like (maybe Series or list)
        try:
            shape_vals = _extract_1d(shape)
            if len(shape_vals) != n:
                raise ValueError("shape must have same length as data or be scalar")
            a = np.sum(shape_vals)
        except Exception:
            raise ValueError("shape must be a numeric scalar or 1‑dimensional array/DataFrame")

    # ---- 3. Check positivity ----
    if np.any(shape_vals <= 0):
        raise ValueError("shape values must be positive")
    if np.any(data_vals <= 0):
        raise ValueError("data values must be positive (Gamma likelihood requires y > 0)")

    # ---- 4. Compute sufficient statistics ----
    b = np.sum(data_vals)

    # log_c = Σ ( (α_i - 1) * log(y_i) - log Γ(α_i) )
    # Use math.lgamma for log Γ(α_i)
    def log_gamma_ratio(alpha, y):
        return (alpha - 1) * math.log(y) - math.lgamma(alpha)

    log_c = np.sum([log_gamma_ratio(alpha, y) for alpha, y in zip(shape_vals, data_vals)])

    return {
        'a': a,
        'b': b,
        'log_c': log_c
    }


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