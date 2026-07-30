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
from scipy.special import gammaln

def _is_1d_dataframe(obj: Any) -> bool:
    """Return True if obj is a pandas DataFrame with exactly 1 column."""
    return isinstance(obj, pd.DataFrame) and obj.shape[1] == 1


def _extract_1d(obj: Any, label: str = "data") -> np.ndarray:
    """Extract a 1D numpy array from a pandas Series, DataFrame, or array-like.

    Parameters
    ----------
    obj : array-like, pandas.Series or 1-column pandas.DataFrame
        Values to extract.
    label : str, optional
        What ``obj`` represents, used in error messages.

    Returns
    -------
    numpy.ndarray
        A 1-D float array.

    Raises
    ------
    ValueError
        If ``obj`` is a DataFrame with more than one column, or if any value is
        NaN or infinite.

    Notes
    -----
    The finiteness check is not decoration. NumPy's ordering comparisons are
    ``False`` for NaN, so ``np.any(values <= 0)`` — the positivity guard every
    likelihood module applies next — passes a NaN straight through. It then
    lands in ``a``, ``b`` or ``log_c`` and surfaces much later as an error that
    names the wrong thing: "Derivative at t=-b is negative" for Rayleigh, "t
    must be provided" for Normal, "cannot convert float NaN to integer" for
    Poisson. None of those mention the data.
    """
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError("DataFrame must have exactly 1 column.")
        values = obj.iloc[:, 0].values.astype(float)
    elif isinstance(obj, pd.Series):
        values = obj.values.astype(float)
    else:
        values = np.asarray(obj, dtype=float)

    if values.size and not np.all(np.isfinite(values)):
        kind = "NaN" if np.any(np.isnan(values)) else "infinite"
        raise ValueError(
            f"{label} contains {kind} values; every entry must be finite."
        )

    return values


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
    data_vals = _extract_1d(data)
    n = len(data_vals)
    if n == 0:
        raise ValueError("data must be non‑empty")

    # ---- Handle shape ----
    if _is_1d_dataframe(shape):
        shape_vals = _extract_1d(shape, "shape")
        if len(shape_vals) != n:
            raise ValueError("shape must have same length as data or be scalar")
    elif isinstance(shape, (int, float)):
        shape_vals = _extract_1d(np.full(n, float(shape)), "shape")
    else:
        try:
            shape_vals = _extract_1d(shape, "shape")
            if len(shape_vals) != n:
                raise ValueError("shape must have same length as data or be scalar")
        except Exception:
            raise ValueError("shape must be a numeric scalar or 1‑dimensional array/DataFrame")

    # ---- Positivity checks ----
    if np.any(shape_vals <= 0):
        raise ValueError("shape values must be positive.")
    if np.any(data_vals <= 0):
        raise ValueError("data values must be positive for Inverse‑Gamma likelihood.")

    # ---- Vectorized sums ----
    a = np.sum(shape_vals)
    b = np.sum(1.0 / data_vals)
    log_c = np.sum(-(shape_vals + 1.0) * np.log(data_vals) - gammaln(shape_vals))

    return {
        'a': float(a),
        'b': float(b),
        'log_c': float(log_c)
    }

def bereitInverseGamma(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    shape: Union[float, int, pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute per‑element sufficient statistics for an Inverse‑Gamma likelihood.

    For each observation y_i and known shape α_i:
        a_i = α_i
        b_i = 1 / y_i
        log_c_i = -(α_i + 1) * log(y_i) - log Γ(α_i)

    Parameters
    ----------
    data : pandas DataFrame (1‑column), pandas Series, or array‑like
        Observed values (must be positive).
    shape : numeric scalar or 1‑column pandas DataFrame/Series/array‑like
        Known shape parameter(s) α. If scalar, recycled; if vector, same length as data.

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
        shape_vals = _extract_1d(shape, "shape")
        if len(shape_vals) != n:
            raise ValueError("shape must have same length as data or be scalar")
    elif isinstance(shape, (int, float)):
        shape_vals = _extract_1d(np.full(n, float(shape)), "shape")
    else:
        try:
            shape_vals = _extract_1d(shape, "shape")
            if len(shape_vals) != n:
                raise ValueError("shape must have same length as data or be scalar")
        except Exception:
            raise ValueError("shape must be a numeric scalar or 1‑dimensional array/DataFrame")

    # ---- Positivity checks ----
    if np.any(shape_vals <= 0):
        raise ValueError("shape values must be positive.")
    if np.any(data_vals <= 0):
        raise ValueError("data values must be positive for Inverse‑Gamma likelihood.")

    # ---- Per‑element statistics ----
    a_vals = shape_vals
    b_vals = 1.0 / data_vals
    log_c_vals = -(shape_vals + 1.0) * np.log(data_vals) - gammaln(shape_vals)

    return {
        'a': a_vals,
        'b': b_vals,
        'log_c': log_c_vals
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