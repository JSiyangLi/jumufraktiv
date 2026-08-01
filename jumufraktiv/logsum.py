import math

import numpy as np

#: The crossover for `log1mexp`. Below it `log(-expm1(-d))` is the accurate
#: branch, above it `log1p(-exp(-d))` is. See `logminus`.
_LOG2 = math.log(2.0)

def logminus(x, y):
    """Compute ``log(exp(x) - exp(y))`` without forming either exponential.

    Parameters
    ----------
    x, y : float or array-like
        Log-space values. Broadcast against each other.

    Returns
    -------
    float or numpy.ndarray
        ``log(exp(x) - exp(y))``, or ``-inf`` where ``x <= y``. A Python float
        if both inputs are scalar, matching the caller's shape otherwise.

    Notes
    -----
    Writing ``d = x - y > 0``, the identity is
    ``log(exp(x) - exp(y)) = x + log1mexp(d)`` with
    ``log1mexp(d) = log(1 - exp(-d))``, and **that inner term needs two
    branches, not one**. This function previously implemented only the
    ``log1p`` branch, which loses the difference entirely once ``exp(-d)``
    rounds to 1:

    ======  =======================  ==========================
    gap     one branch (before)      exact (mpmath, 50 digits)
    ======  =======================  ==========================
    1e-08   -18.420680750029838      -18.420680748952364
    1e-10   -23.025850847200090      -23.025850929990458
    1e-15   -34.539575992340879      -34.538776394910684
    1e-17   -inf                     -39.143946580898778
    ======  =======================  ==========================

    So the absolute error is 8.3e-08 at a gap of 1e-10 -- 3.6e-09 *relative* to
    the value, and both figures are correct, which is exactly the confusion
    :file:`CLAUDE.md` warns about. By 1e-15 it is 8.0e-04 absolute, and at 1e-17
    the answer is ``-inf`` with a "divide by zero encountered in log1p" warning
    for a quantity that is finite.

    The switch point is ``log 2``: below it use ``log(-expm1(-d))``, above it
    ``log1p(-exp(-d))``. Each branch is the one whose rounding error stays
    small on its side of the crossover. This is the standard form (Mächler
    2012, the ``Rmpfr`` ``log1mexp`` vignette).

    The small-gap regime is not hypothetical here: this function forms
    differences of incomplete MGFs, where the two arguments are close by
    construction.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = np.broadcast_arrays(x, y)

    d = x - y
    positive = d > 0.0

    # Evaluate both branches only where the difference is positive, so the
    # `-inf` region cannot raise on `expm1(0)` or `log(0)`.
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_d = np.where(positive, d, 1.0)
        log1mexp = np.where(
            safe_d <= _LOG2,
            np.log(-np.expm1(-safe_d)),
            np.log1p(-np.exp(-safe_d)),
        )

    result = np.where(positive, x + log1mexp, -np.inf)

    if result.shape == ():
        return float(result)
    return result
