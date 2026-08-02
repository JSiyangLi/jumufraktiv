"""
MaxwellBoltzmann.py

Functions for preparing Maxwell-Boltzmann likelihood statistics for MGF marginalisation.

For a Maxwell-Boltzmann distribution with scale parameter a (or rate θ = 1/(2a²)),
the density for a speed y ≥ 0 is:

    f(y; θ) = (4 / √π) * y² * θ^{3/2} * exp(-θ y²)

This can be written as::

    L(θ; y) = c(y) * θ^{a(y)} * exp(-b(y) θ)

with a(y) = 3/2, b(y) = y², c(y) = (4 / √π) * y².

For a sample of size n, the joint likelihood is::

    L(θ; y) = (4 / √π)^n (∏ y_i²) * θ^{3n/2} * exp(-θ Σ y_i²)

Thus::

    a     = 3n/2
    b     = Σ y_i²
    log_c = n * log(4/√π) + 2 Σ log(y_i)
          = n * (log(4) - 0.5 * log(π)) + 2 Σ log(y_i)
"""

import math

import numpy as np
import pandas as pd
import sympy as sp

from jumufraktiv.like_stats._common import _extract_1d


def readyMaxwellBoltzmann(
    data: pd.DataFrame | pd.Series | list | np.ndarray,
    **kwargs
) -> dict[str, float | int]:
    """
    Compute sufficient statistics for a Maxwell-Boltzmann likelihood.

    The likelihood (in terms of rate θ = 1/(2a²)) is:
        L(θ; y) = (4 / √π) * y² * θ^{3/2} * exp(-θ y²)

    For a sample of size n:
        a = 3n/2
        b = Σ y_i²
        log_c = n * (log(4) - 0.5 * log(π)) + 2 Σ log(y_i)

    Parameters
    ----------
    data : pandas DataFrame (1-column), pandas Series, or array-like
        Observed values (must be positive).
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
        raise ValueError("data must be non-empty")

    # ---- 2. Check positivity ----
    if np.any(data_vals <= 0):
        raise ValueError(
            "data values must be positive for Maxwell-Boltzmann likelihood."
        )

    # ---- 3. Compute sufficient statistics ----
    a = 1.5 * n
    b = np.sum(data_vals ** 2)
    # log_c = n * (log(4) - 0.5 * log(π)) + 2 * Σ log(y_i)
    log_c = (
        n * (math.log(4.0) - 0.5 * math.log(math.pi))
        + 2.0 * np.sum(np.log(data_vals))
    )

    return {
        'a': a,
        'b': b,
        'log_c': log_c
    }

def bereitMaxwellBoltzmann(
    data: pd.DataFrame | pd.Series | list | np.ndarray,
    **kwargs
) -> dict[str, np.ndarray]:
    """
    Compute per-element sufficient statistics for a Maxwell-Boltzmann likelihood.

    For each observation y_i:
        a_i = 3/2
        b_i = y_i²
        log_c_i = log(4) - 0.5 * log(π) + 2 * log(y_i)

    Parameters
    ----------
    data : pandas DataFrame (1-column), pandas Series, or array-like
        Observed values (must be positive).
    **kwargs : additional arguments (ignored).

    Returns
    -------
    dict
        Keys: 'a', 'b', 'log_c', each as a numpy array of length n.
    """
    data_vals = _extract_1d(data)
    n = len(data_vals)
    if n == 0:
        raise ValueError("data must be non-empty")

    if np.any(data_vals <= 0):
        raise ValueError(
            "data values must be positive for Maxwell-Boltzmann likelihood."
        )

    a_vals = np.full(n, 1.5)
    b_vals = data_vals ** 2
    log_c_vals = np.full(n, np.log(4.0) - 0.5 * np.log(np.pi)) + 2.0 * np.log(data_vals)

    return {
        'a': a_vals,
        'b': b_vals,
        'log_c': log_c_vals
    }


def cMaxwellBoltzmann() -> sp.Expr:
    """
    Return a symbolic expression for the Maxwell-Boltzmann normalising constant:

        ∏_{i=1}^{n} (4 / √π) * y_i² = (4 / √π)^n * ∏ y_i²

    where n is a symbolic integer.

    Returns
    -------
    sympy.Expr
        (4 / √π)^n * ∏ y_i²
    """
    n = sp.Symbol('n', integer=True, positive=True)
    i = sp.Idx('i')
    y = sp.IndexedBase('y')
    const = 4 / sp.sqrt(sp.pi)
    expr = sp.Product(const * y[i]**2, (i, 1, n))
    return expr
