"""Input handling shared by the fourteen likelihood modules.

The two functions below are defined here and nowhere else, and no likelihood
module carries its own copy. That is what makes their validation *uniform*: the
dimensionality and finiteness checks live in :func:`_extract_1d`, and every
``ready*`` and ``each*`` entry point passes its data — and every known
parameter — through it, so "one guard, applied everywhere" holds by
construction rather than by fourteen-fold agreement.
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
    *Finiteness.* The returned array is guaranteed finite, and this is the only
    place that guarantee is established. NumPy's ordering comparisons are
    ``False`` for NaN, so ``np.any(values <= 0)`` — the positivity guard every
    likelihood module applies next — passes a NaN straight through, and it
    would then reach ``a``, ``b`` or ``log_c`` and surface much later as an
    error naming something other than the data.

    *Dimensionality.* These are functions of a one-dimensional sample, so a 2-D
    array is a caller error and not a shape to reinterpret. It is refused
    rather than flattened or summed along an axis, because ``a`` is the *order
    of differentiation* and reinterpreting the shape silently changes which
    derivative is taken.
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
