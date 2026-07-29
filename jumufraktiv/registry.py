"""
registry.py

Global registry for prior distributions in the MGF‑marginalisable framework.

This module manages the registration and retrieval of prior factories.
Priors are registered using the `@register_prior` decorator, typically
inside the `MGFdictionary` subpackage. The registry is lazily initialised
on first use.

Functions:
    - register_prior : decorator to add a prior factory to the registry.
    - initialize : explicitly load all prior modules.
    - get_prior : retrieve a registered prior by name.
    - list_priors : list all registered prior names.
    - make_prior_spec : helper to create a standard prior specification dict.
"""

import warnings

PRIOR_REGISTRY = {}
_LOADED = False


# ============================================================
# Registration decorator (used by priors)
# ============================================================
def register_prior(name: str):
    """
    Decorator used inside MGFdictionary prior modules to register a prior.

    The decorated function (the factory) must accept a `params` dictionary
    and return a specification dict (as produced by `make_prior_spec`).

    Parameters
    ----------
    name : str
        Unique name of the prior (e.g., 'gamma', 'pareto').

    Returns
    -------
    callable
        The decorator that registers the factory in `PRIOR_REGISTRY`.

    Example
    -------
    >>> @register_prior("gamma")
    ... def gamma_factory(params):
    ...     return make_prior_spec(...)
    """
    def decorator(fn):
        PRIOR_REGISTRY[name] = fn
        return fn
    return decorator


# ============================================================
# Initialization (IMPORTANT: explicit control point)
# ============================================================
def initialize():
    """
    Trigger import‑time discovery of all priors.

    This function imports `jumufraktiv.MGFdictionary`, which in turn
    imports all prior modules and runs their `@register_prior` decorators.
    It is safe to call multiple times.

    Returns
    -------
    None
    """
    global _LOADED

    if _LOADED:
        return

    try:
        import jumufraktiv.MGFdictionary  # triggers auto‑registration
    except Exception as e:
        warnings.warn(f"[registry] Failed to load MGFdictionary: {e}")

    _LOADED = True


# ============================================================
# Public API
# ============================================================
def get_prior(name: str):
    """
    Retrieve a registered prior factory by name.

    Parameters
    ----------
    name : str
        Name of the prior (as registered).

    Returns
    -------
    callable
        The prior factory function.

    Raises
    ------
    KeyError
        If the prior is not found in the registry.

    Examples
    --------
    >>> gamma_factory = get_prior('gamma')
    >>> spec = gamma_factory(params={'alpha':2.0, 'beta':3.0})
    """
    initialize()

    if name not in PRIOR_REGISTRY:
        raise KeyError(
            f"Prior '{name}' not found. "
            f"Available: {list(PRIOR_REGISTRY.keys())}"
        )

    return PRIOR_REGISTRY[name]


def list_priors():
    """
    List all registered prior names.

    Returns
    -------
    list of str
        Names of all available priors.
    """
    initialize()
    return list(PRIOR_REGISTRY.keys())


def make_prior_spec(**kwargs):
    """
    Create a standard prior specification dictionary.

    This function validates that the required fields are present and
    returns the kwargs as a dictionary. It is used by prior factories
    to produce a consistent output format expected by `mitMGFprior`.

    Parameters
    ----------
    **kwargs : dict
        Must contain at least `mgf` and `cgf` callables.
        Typically also includes `mgf_sym`, `pdf_sym`, `mgf_jax`, `cgf_jax`,
        `pdf_func`, `logpdf_func`, `imgf` (for incomplete MGF), etc.

    Returns
    -------
    dict
        The input kwargs, validated.

    Raises
    ------
    ValueError
        If `mgf` or `cgf` are missing.

    Examples
    --------
    >>> spec = make_prior_spec(
    ...     mgf=lambda t: (1 - t)**(-2),
    ...     cgf=lambda t: -2 * np.log(1 - t),
    ...     params={'alpha': 2.0}
    ... )
    """
    required_keys = ["mgf", "cgf"]

    for k in required_keys:
        if k not in kwargs:
            raise ValueError(f"Missing required prior field: {k}")

    return kwargs