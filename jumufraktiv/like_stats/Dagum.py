"""
Dagum.py

Functions for preparing Dagum likelihood statistics for MGF marginalisation.

For a Dagum distribution with known shape r (scalar or vector) and known scale s (scalar or vector),
and unknown shape q, the density for y > 0 is:

    f(y; r, s, q) = (r * q / y) * (y/s)^(r*q) / ( (y/s)^r + 1 )^(q+1)

This can be written as:
    L(q; y) = C(y) * q^{a(y)} * exp(-b(y) q)

with a(y) = 1,
    b(y) = log(1 + (y/s)^r) - r * log(y/s),
    C(y) = (r / y) / (1 + (y/s)^r).

For a sample of size n:
    a_stat = n
    b_stat = Σ [ log(1 + (y_i/s_i)^{r_i}) - r_i * log(y_i/s_i) ]
    log_C = Σ log(r_i / y_i) - Σ log(1 + (y_i/s_i)^{r_i})

If the known parameters are scalars, they are recycled. If they are vectors,
they must have the same length as data.
"""

import pandas as pd
import numpy as np
import math
import sympy as sp
from typing import Union, Dict, Any


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

def readyDagum(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    r: Union[float, int, pd.DataFrame, pd.Series, list, np.ndarray],
    s: Union[float, int, pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, Union[float, int]]:
    """
    Compute sufficient statistics for a Dagum likelihood with known r and s.

    The likelihood (in terms of unknown shape q) is:
        L(q; y) = C(y) * q * exp(-q * b(y))

    where:
        b(y) = log(1 + (y/s)^r) - r * log(y/s)
        C(y) = (r / y) / (1 + (y/s)^r)

    Here `r` is the known shape parameter, and `s` is the known scale parameter.

    For a sample of size n:
        a_stat = n
        b_stat = Σ [ log(1 + (y_i/s_i)^{r_i}) - r_i * log(y_i/s_i) ]
        log_C = Σ log(r_i / y_i) - Σ log(1 + (y_i/s_i)^{r_i})

    Parameters
    ----------
    data : pandas DataFrame (1‑column), pandas Series, or array‑like
        Observed values (must be positive).
    r : numeric scalar or 1‑column pandas DataFrame/Series/array‑like
        Known shape parameter(s). If scalar, recycled; if vector, same length as data.
    s : numeric scalar or 1‑column pandas DataFrame/Series/array‑like
        Known scale parameter(s). If scalar, recycled; if vector, same length as data.
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

    # ---- Handle r and s (vectorization) ----
    def _handle_param(param, name):
        if _is_1d_dataframe(param):
            vals = _extract_1d(param, name)
            if len(vals) != n:
                raise ValueError(f"{name} must have same length as data or be scalar")
            return vals
        elif isinstance(param, (int, float)):
            return _extract_1d(np.full(n, float(param)), name)
        else:
            try:
                vals = _extract_1d(param, name)
                if len(vals) != n:
                    raise ValueError(f"{name} must have same length as data or be scalar")
                return vals
            except Exception:
                raise ValueError(f"{name} must be a numeric scalar or 1‑dimensional array/DataFrame")

    r_vals = _handle_param(r, 'r')
    s_vals = _handle_param(s, 's')

    # ---- Positivity checks ----
    if np.any(r_vals <= 0):
        raise ValueError("r values must be positive.")
    if np.any(s_vals <= 0):
        raise ValueError("s values must be positive.")
    if np.any(data_vals <= 0):
        raise ValueError("data values must be positive for Dagum likelihood.")

    # ---- Compute log‑stable statistics ----
    # log(1 + (y/s)^r)  – needed for log_c
    ratio = data_vals / s_vals
    log_term = np.log1p(ratio ** r_vals)          # log(1 + (y/s)^r)

    # log(1 + (s/y)^r)  – needed for b (stable version)
    inv_ratio = s_vals / data_vals
    log_term_inv = np.log1p(inv_ratio ** r_vals)   # log(1 + (s/y)^r)

    a_stat = float(n)
    b_stat = np.sum(log_term_inv)
    log_c = np.sum(np.log(r_vals) - np.log(data_vals) - log_term)

    return {'a': a_stat, 'b': b_stat, 'log_c': log_c}
    
def bereitDagum(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    r: Union[float, int, pd.DataFrame, pd.Series, list, np.ndarray],
    s: Union[float, int, pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute per‑element sufficient statistics for a Dagum likelihood.

    For each observation y_i, known shape r_i, known scale s_i:
        a_i = 1
        b_i = log(1 + (y_i/s_i)^{r_i}) - r_i * log(y_i/s_i)
        log_c_i = log(r_i / y_i) - log(1 + (y_i/s_i)^{r_i})

    Parameters
    ----------
    data : pandas DataFrame (1‑column), pandas Series, or array‑like
        Observed values (must be positive).
    r : numeric scalar or 1‑column pandas DataFrame/Series/array‑like
        Known shape parameter(s). If scalar, recycled; if vector, same length as data.
    s : numeric scalar or 1‑column pandas DataFrame/Series/array‑like
        Known scale parameter(s). If scalar, recycled; if vector, same length as data.
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

    # ---- Reuse parameter handling ----
    def _handle_param(param, name):
        if _is_1d_dataframe(param):
            vals = _extract_1d(param, name)
            if len(vals) != n:
                raise ValueError(f"{name} must have same length as data or be scalar")
            return vals
        elif isinstance(param, (int, float)):
            return _extract_1d(np.full(n, float(param)), name)
        else:
            try:
                vals = _extract_1d(param, name)
                if len(vals) != n:
                    raise ValueError(f"{name} must have same length as data or be scalar")
                return vals
            except Exception:
                raise ValueError(f"{name} must be a numeric scalar or 1‑dimensional array/DataFrame")

    r_vals = _handle_param(r, 'r')
    s_vals = _handle_param(s, 's')

    # ---- Positivity checks ----
    if np.any(r_vals <= 0):
        raise ValueError("r values must be positive.")
    if np.any(s_vals <= 0):
        raise ValueError("s values must be positive.")
    if np.any(data_vals <= 0):
        raise ValueError("data values must be positive for Dagum likelihood.")

    # ---- Per‑element computations ----
    ratio = data_vals / s_vals
    log_term = np.log1p(ratio ** r_vals)          # log(1 + (y/s)^r)

    inv_ratio = s_vals / data_vals
    log_term_inv = np.log1p(inv_ratio ** r_vals)   # log(1 + (s/y)^r)

    a_vals = np.ones(n, dtype=float)
    b_vals = log_term_inv
    log_c_vals = np.log(r_vals) - np.log(data_vals) - log_term

    return {'a': a_vals, 'b': b_vals, 'log_c': log_c_vals}


def cDagum() -> sp.Expr:
    """
    Return a symbolic expression for the Dagum normalising constant:

        ∏_{i=1}^{n} ( (r_i / y_i) / (1 + (y_i/s_i)^{r_i}) )

    where n, r_i, s_i, and y_i are symbolic.

    Returns
    -------
    sympy.Expr
        ∏ ( r_i / y_i / (1 + (y_i/s_i)^{r_i}) )
    """
    n = sp.Symbol('n', integer=True, positive=True)
    r = sp.IndexedBase('r')
    s = sp.IndexedBase('s')
    i = sp.Idx('i')
    y = sp.IndexedBase('y')
    expr = sp.Product(
        (r[i] / y[i]) / (1 + (y[i] / s[i])**r[i]),
        (i, 1, n)
    )
    return expr


# ===== Example usage =====
if __name__ == "__main__":
    # Scalar r and s
    data_df = pd.DataFrame({'y': [0.5, 1.0, 1.5]})
    r_scalar = 2.0
    s_scalar = 1.0
    stats = readyDagum(data_df, r_scalar, s_scalar)
    print("Scalar r=2, s=1:", stats)

    # Vector r and s
    r_vec = pd.DataFrame({'r': [2.0, 3.0, 1.5]})
    s_vec = pd.DataFrame({'s': [1.0, 0.8, 1.2]})
    stats2 = readyDagum(data_df, r_vec, s_vec)
    print("Vector r and s:", stats2)

    # Symbolic constant
    c_expr = cDagum()
    print("\nSymbolic normalising constant:")
    sp.pprint(c_expr)