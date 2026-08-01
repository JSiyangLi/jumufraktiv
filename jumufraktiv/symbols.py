"""
Symbols for symbolic computations in the jumufraktiv package.

This module defines canonical SymPy symbols used throughout the package
for moment-generating functions, densities, and related quantities.

The main symbols are:
- t   : variable in the MGF domain (Laplace transform variable)
- theta : latent parameter (in the parameter space)
- r   : variable for posterior MGF
- u   : upper limit for CDF evaluation (incomplete MGF)
- q   : moment order for raw moments

These symbols are used in symbolic expressions for priors, likelihoods,
and posterior quantities. They should be imported from this module
rather than redefined elsewhere to ensure consistency.

Additionally, the `param()` function is provided as a factory for
creating hyperparameter symbols (e.g., alpha, beta, xi) with the
appropriate real and positive assumptions.

Examples
--------
>>> from jumufraktiv.symbols import t, theta, r, u, q
>>> from jumufraktiv.symbols import param
>>> alpha = param('alpha')
>>> beta = param('beta')
>>> mgf_expr = (beta / (beta - t)) ** alpha   # Gamma MGF
"""

import sympy as sp

# ============================================================
# Global canonical symbols
# ============================================================
# ---- Canonical variables for MGF marginalisation ----

# transform variable (MGF / CGF domain)
t = sp.Symbol("t", real=True)

# latent variable (parameter space)
theta = sp.Symbol("theta", positive=True, real=True)

# posterior MGF variable
r = sp.Symbol("r", real=True)

# CDF evaluation point
u = sp.Symbol("u", real=True)

# moment order
q = sp.Symbol("q", real=True)

# ============================================================
# Parameter symbol factory (important for extensibility)
# ============================================================

def param(name: str):
    """
    Create a symbolic parameter with real and positive assumptions.

    This function is a factory for creating SymPy symbols that represent
    parameters in prior distributions (e.g., shape, scale, rate). It ensures
    that the symbols are declared as real and positive, which helps SymPy
    with simplification and integration.

    Parameters
    ----------
    name : str
        Name of the parameter (e.g., 'alpha', 'beta', 'xi').

    Returns
    -------
    sympy.Symbol
        A SymPy symbol with the given name, with assumptions ``real=True``
        and ``positive=True``.

    Notes
    -----
    - This function is used in prior factory functions to create symbolic
      representations of hyperparameters.
    - The resulting symbols are later substituted with numeric values when
      a prior is instantiated with concrete parameters.
    - The canonical variables for MGF and CDF are ``t``, ``theta``, ``r``,
      and ``u``, which are defined separately in this module.

    Examples
    --------
    >>> from jumufraktiv.symbols import param
    >>> alpha = param('alpha')
    >>> beta = param('beta')
    >>> expr = (beta / (beta - t)) ** alpha   # symbolic MGF of Gamma
    """
    return sp.Symbol(name, real=True, positive=True)
