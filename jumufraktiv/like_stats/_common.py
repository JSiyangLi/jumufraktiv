"""Input handling shared by the fourteen likelihood modules.

Every ``like_stats`` module used to carry its own byte-identical copy of the
two functions below. Folding them into one place is not only a size saving: it
is what makes their validation *uniform*, which it was not. Ten of the fourteen
``ready*`` functions guarded against two-dimensional data and four did not, and
no ``bereit*`` function guarded at all — so the same input was rejected by one
likelihood and silently mis-computed by another.

The dimensionality check therefore lives in :func:`_extract_1d` rather than in
each caller. Every entry point extracts its data through this function, so
placing the check here is what makes "one guard, applied everywhere" true by
construction instead of by fourteen-fold agreement.
"""

from typing import Any

import numpy as np
import pandas as pd


def _is_1d_dataframe(obj: Any) -> bool:
    """Return True if ``obj`` is a pandas DataFrame with exactly one column."""
    return isinstance(obj, pd.DataFrame) and obj.shape[1] == 1


def _extract_1d(obj: Any, label: str = "data") -> np.ndarray:
    """Extract a 1-D float array from a Series, one-column DataFrame or array-like.

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
        If ``obj`` is a DataFrame with more than one column, if it is not
        one-dimensional, or if any value is NaN or infinite.

    Notes
    -----
    Two of the three checks are load-bearing rather than defensive.

    *Finiteness.* NumPy's ordering comparisons are ``False`` for NaN, so
    ``np.any(values <= 0)`` — the positivity guard every likelihood module
    applies next — passes a NaN straight through. It then lands in ``a``, ``b``
    or ``log_c`` and surfaces much later as an error naming the wrong thing:
    "Derivative at t=-b is negative" for Rayleigh, "t must be provided" for
    Normal, "cannot convert float NaN to integer" for Poisson. None of those
    mention the data.

    *Dimensionality.* These are functions of a one-dimensional sample, so a 2-D
    array is a caller error and not a shape to reinterpret. Accepting one is
    not a cosmetic fault: ``a`` is the *order of differentiation*, and summing
    along one axis of a 2-D array rather than over every element halved it.
    ``readyGamma([[1,2],[3,4]], shape=2.0)`` gave ``a = 4.0`` where the flat
    data gives ``8.0``, and the package went on to compute a derivative of a
    different order without comment.
    """
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError(f"{label} DataFrame must have exactly 1 column.")
        values = obj.iloc[:, 0].values.astype(float)
    elif isinstance(obj, pd.Series):
        values = obj.values.astype(float)
    else:
        values = np.asarray(obj, dtype=float)

    if values.ndim != 1:
        raise ValueError(
            f"{label} must be 1-dimensional; got an array with {values.ndim} "
            f"dimensions and shape {values.shape}."
        )

    if values.size and not np.all(np.isfinite(values)):
        kind = "NaN" if np.any(np.isnan(values)) else "infinite"
        raise ValueError(f"{label} contains {kind} values; every entry must be finite.")

    return values
