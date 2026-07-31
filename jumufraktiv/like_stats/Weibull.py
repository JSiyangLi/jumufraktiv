"""
Weibull.py

Functions for preparing Weibull likelihood statistics for MGF marginalisation.

For a Weibull distribution with known shape parameter ρ (scalar or vector) and unknown rate λ,
the density for y > 0 is:

    f(y; λ, ρ) = ρ λ y^{ρ-1} exp(-λ y^ρ)

Note the rate convention: λ = scale^{-ρ}, so λ appears to the first power, not
the ρ-th. The alternative convention λ' = 1/scale gives ρ λ'^ρ y^{ρ-1}
exp(-λ'^ρ y^ρ); mixing the prefactor of one with the exponential of the other
yields a function integrating to λ^{ρ-1} rather than 1, and would imply
a(y) = ρ instead of 1.

This can be written as:
    L(λ; y) = c(y) * λ^{a(y)} * exp(-b(y) λ)

with a(y) = 1, b(y) = y^ρ, c(y) = ρ y^{ρ-1}.

For a sample of size n:
    a = n
    b = Σ y_i^{ρ_i}
    log_c = Σ ( log(ρ_i) + (ρ_i-1) log(y_i) )

If ρ is a scalar, it is recycled. If ρ is a vector, it must have length n.
"""

import pandas as pd
import numpy as np
import math
import sympy as sp
from typing import Union, Dict

from jumufraktiv.like_stats._common import _extract_1d, _is_1d_dataframe


def readyWeibull(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    rho: Union[float, int, pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, Union[float, int]]:
    """
    Compute sufficient statistics for a Weibull likelihood with known shape ρ.

    The likelihood (in terms of rate λ) is:
        L(λ; y) = (∏ ρ_i y_i^{ρ_i-1}) * λ^n * exp(-λ Σ y_i^{ρ_i})

    For a sample of size n:
        a = n
        b = Σ y_i^{ρ_i}
        log_c = Σ ( log(ρ_i) + (ρ_i-1) log(y_i) )

    Parameters
    ----------
    data : pandas DataFrame (1‑column), pandas Series, or array‑like
        Observed values (must be positive).
    rho : numeric scalar or 1‑column pandas DataFrame/Series/array‑like
        Known shape parameter(s) ρ. If scalar, it is recycled to match length of data.
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

    # ---- 2. Handle rho ----
    if _is_1d_dataframe(rho):
        rho_vals = _extract_1d(rho, "rho")
        if len(rho_vals) != n:
            raise ValueError("rho must have same length as data or be scalar")
    elif isinstance(rho, (int, float)):
        rho_vals = _extract_1d(np.full(n, float(rho)), "rho")
    else:
        try:
            rho_vals = _extract_1d(rho, "rho")
            if len(rho_vals) != n:
                raise ValueError("rho must have same length as data or be scalar")
        except Exception:
            raise ValueError("rho must be a numeric scalar or 1‑dimensional array/DataFrame")

    # ---- 3. Check positivity ----
    if np.any(rho_vals <= 0):
        raise ValueError("rho values must be positive.")
    if np.any(data_vals <= 0):
        raise ValueError("data values must be positive for Weibull likelihood.")

    # ---- 4. Compute sufficient statistics ----
    a = float(n)
    b = np.sum(data_vals ** rho_vals)
    # log_c = Σ ( log(ρ_i) + (ρ_i-1) log(y_i) )
    log_c = np.sum(np.log(rho_vals) + (rho_vals - 1.0) * np.log(data_vals))

    return {
        'a': a,
        'b': b,
        'log_c': log_c
    }
    
def bereitWeibull(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    rho: Union[float, int, pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute per‑element sufficient statistics for a Weibull likelihood.

    For each observation y_i and known shape ρ_i:
        a_i = 1
        b_i = y_i^{ρ_i}
        log_c_i = log(ρ_i) + (ρ_i - 1) * log(y_i)

    Parameters
    ----------
    data : pandas DataFrame (1‑column), pandas Series, or array‑like
        Observed values (must be positive).
    rho : numeric scalar or 1‑column pandas DataFrame/Series/array‑like
        Known shape parameter(s) ρ. If scalar, recycled; if vector, same length as data.
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

    # ---- Handle rho ----
    if _is_1d_dataframe(rho):
        rho_vals = _extract_1d(rho, "rho")
        if len(rho_vals) != n:
            raise ValueError("rho must have same length as data or be scalar")
    elif isinstance(rho, (int, float)):
        rho_vals = _extract_1d(np.full(n, float(rho)), "rho")
    else:
        try:
            rho_vals = _extract_1d(rho, "rho")
            if len(rho_vals) != n:
                raise ValueError("rho must have same length as data or be scalar")
        except Exception:
            raise ValueError("rho must be a numeric scalar or 1‑dimensional array/DataFrame")

    # ---- Check positivity ----
    if np.any(rho_vals <= 0):
        raise ValueError("rho values must be positive.")
    if np.any(data_vals <= 0):
        raise ValueError("data values must be positive for Weibull likelihood.")

    # ---- Per‑element statistics ----
    a_vals = np.ones(n, dtype=float)
    b_vals = data_vals ** rho_vals
    log_c_vals = np.log(rho_vals) + (rho_vals - 1.0) * np.log(data_vals)

    return {
        'a': a_vals,
        'b': b_vals,
        'log_c': log_c_vals
    }


def cWeibull() -> sp.Expr:
    """
    Return a symbolic expression for the Weibull normalising constant:

        ∏_{i=1}^{n} ρ_i y_i^{ρ_i-1}

    where n is symbolic, and ρ_i and y_i are symbolic indexed variables.

    Returns
    -------
    sympy.Expr
        ∏ ρ_i y_i^{ρ_i-1}
    """
    n = sp.Symbol('n', integer=True, positive=True)
    rho = sp.IndexedBase('rho')
    i = sp.Idx('i')
    y = sp.IndexedBase('y')
    expr = sp.Product(rho[i] * y[i]**(rho[i] - 1), (i, 1, n))
    return expr


# ===== Example usage =====
if __name__ == "__main__":
    # Scalar rho
    data_df = pd.DataFrame({'y': [1.0, 2.0, 3.0]})
    rho_scalar = 2.0
    stats = readyWeibull(data_df, rho_scalar)
    print("Scalar rho (ρ=2):", stats)

    # Vector rho
    rho_vec = pd.DataFrame({'rho': [1.5, 2.0, 2.5]})
    stats2 = readyWeibull(data_df, rho_vec)
    print("Vector rho:", stats2)

    # Symbolic constant
    c_expr = cWeibull()
    print("\nSymbolic normalising constant:")
    sp.pprint(c_expr)