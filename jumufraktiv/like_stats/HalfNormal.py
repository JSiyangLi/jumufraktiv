"""
HalfNormal.py

Functions for preparing Half‑Normal likelihood statistics for MGF marginalisation.

For a Half‑Normal distribution with scale parameter σ, the density for y ≥ 0 is:

    f(y; σ) = (√(2) / (σ √π)) * exp(- y² / (2 σ²))

Writing θ = 1/σ², this becomes:
    L(θ; y) = c(y) * θ^{a(y)} * exp(-b(y) θ)

with a(y) = 1/2, b(y) = y² / 2, c(y) = √(2/π).

For a sample of size n:
    a = n/2
    b = Σ y_i² / 2
    log_c = n/2 * (log(2) - log(π))
"""

import pandas as pd
import numpy as np
import math
import sympy as sp
from typing import Union, Dict

from jumufraktiv.like_stats._common import _extract_1d


def readyHalfNormal(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, Union[float, int]]:
    """
    Compute sufficient statistics for a Half‑Normal likelihood.

    The likelihood (in terms of precision θ = 1/σ²) is:
        L(θ; y) = √(2/π) * θ^{1/2} * exp(-θ * y² / 2)

    For a sample of size n:
        a = n/2
        b = Σ y_i² / 2
        log_c = n/2 * (log(2) - log(π))

    Parameters
    ----------
    data : pandas DataFrame (1‑column), pandas Series, or array‑like
        Observed values (must be non‑negative).
    **kwargs : additional arguments (ignored, for compatibility).

    Returns
    -------
    dict
        Keys: 'a', 'b', 'log_c'.

    Raises
    ------
    ValueError
        If data contains negative values.
    """
    # ---- 1. Extract data as 1D array ----
    data_vals = _extract_1d(data)
    n = len(data_vals)
    if n == 0:
        raise ValueError("data must be non‑empty")

    # ---- 2. Check non‑negativity ----
    if np.any(data_vals < 0):
        raise ValueError("data values must be non‑negative for Half‑Normal likelihood.")

    # ---- 3. Compute sufficient statistics ----
    a = n / 2.0
    b = np.sum(data_vals ** 2) / 2.0
    log_c = n / 2.0 * (math.log(2.0) - math.log(math.pi))

    return {
        'a': a,
        'b': b,
        'log_c': log_c
    }
    
def bereitHalfNormal(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute per‑element sufficient statistics for a Half‑Normal likelihood.

    For each observation y_i:
        a_i = 0.5
        b_i = y_i² / 2
        log_c_i = 0.5 * (log(2) - log(π))

    Parameters
    ----------
    data : pandas DataFrame (1‑column), pandas Series, or array‑like
        Observed values (must be non‑negative).
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

    if np.any(data_vals < 0):
        raise ValueError("data values must be non‑negative for Half‑Normal likelihood.")

    a_vals = np.full(n, 0.5)
    b_vals = data_vals ** 2 / 2.0
    log_c_vals = np.full(n, 0.5 * (np.log(2.0) - np.log(np.pi)))

    return {
        'a': a_vals,
        'b': b_vals,
        'log_c': log_c_vals
    }


def cHalfNormal() -> sp.Expr:
    """
    Return a symbolic expression for the Half‑Normal normalising constant:

        ∏_{i=1}^{n} √(2/π) = (√(2/π))^n

    where n is a symbolic integer.

    Returns
    -------
    sympy.Expr
        (√(2/π))^n
    """
    n = sp.Symbol('n', integer=True, positive=True)
    const = sp.sqrt(2 / sp.pi)
    return const ** n
