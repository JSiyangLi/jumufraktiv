"""
MaxwellBoltzmann.py

Functions for preparing Maxwell‑Boltzmann likelihood statistics for MGF marginalisation.

For a Maxwell‑Boltzmann distribution with scale parameter a (or rate θ = 1/(2a²)),
the density for a speed y ≥ 0 is:

    f(y; θ) = (4 / √π) * y² * θ^{3/2} * exp(-θ y²)

This can be written as:
    L(θ; y) = c(y) * θ^{a(y)} * exp(-b(y) θ)

with a(y) = 3/2, b(y) = y², c(y) = (4 / √π) * y².

For a sample of size n, the joint likelihood is:
    L(θ; y) = (4 / √π)^n (∏ y_i²) * θ^{3n/2} * exp(-θ Σ y_i²)

Thus:
    a = 3n/2
    b = Σ y_i²
    log_c = n * log(4/√π) + 2 Σ log(y_i)
         = n * (log(4) - 0.5 * log(π)) + 2 Σ log(y_i)
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


def readyMaxwellBoltzmann(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, Union[float, int]]:
    """
    Compute sufficient statistics for a Maxwell‑Boltzmann likelihood.

    The likelihood (in terms of rate θ = 1/(2a²)) is:
        L(θ; y) = (4 / √π) * y² * θ^{3/2} * exp(-θ y²)

    For a sample of size n:
        a = 3n/2
        b = Σ y_i²
        log_c = n * (log(4) - 0.5 * log(π)) + 2 Σ log(y_i)

    Parameters
    ----------
    data : pandas DataFrame (1‑column), pandas Series, or array‑like
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
    if data_vals.ndim != 1:
        raise ValueError("data must be 1‑dimensional")
    n = len(data_vals)
    if n == 0:
        raise ValueError("data must be non‑empty")

    # ---- 2. Check positivity ----
    if np.any(data_vals <= 0):
        raise ValueError("data values must be positive for Maxwell‑Boltzmann likelihood.")

    # ---- 3. Compute sufficient statistics ----
    a = 1.5 * n
    b = np.sum(data_vals ** 2)
    # log_c = n * (log(4) - 0.5 * log(π)) + 2 * Σ log(y_i)
    log_c = n * (math.log(4.0) - 0.5 * math.log(math.pi)) + 2.0 * np.sum(np.log(data_vals))

    return {
        'a': a,
        'b': b,
        'log_c': log_c
    }
    
def bereitMaxwellBoltzmann(
    data: Union[pd.DataFrame, pd.Series, list, np.ndarray],
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute per‑element sufficient statistics for a Maxwell‑Boltzmann likelihood.

    For each observation y_i:
        a_i = 3/2
        b_i = y_i²
        log_c_i = log(4) - 0.5 * log(π) + 2 * log(y_i)

    Parameters
    ----------
    data : pandas DataFrame (1‑column), pandas Series, or array‑like
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
        raise ValueError("data must be non‑empty")

    if np.any(data_vals <= 0):
        raise ValueError("data values must be positive for Maxwell‑Boltzmann likelihood.")

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
    Return a symbolic expression for the Maxwell‑Boltzmann normalising constant:

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


# ===== Example usage =====
if __name__ == "__main__":
    # Example data
    data_df = pd.DataFrame({'y': [1.0, 2.0, 3.0]})
    stats = readyMaxwellBoltzmann(data_df)
    print("Statistics for Maxwell‑Boltzmann:", stats)

    # Symbolic constant
    c_expr = cMaxwellBoltzmann()
    print("\nSymbolic normalising constant:")
    sp.pprint(c_expr)