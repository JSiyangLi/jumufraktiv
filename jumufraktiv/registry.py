"""
registry.py

Global registry for prior distributions in the MGF-marginalisable framework.

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
    Trigger import-time discovery of all priors.

    This function imports `jumufraktiv.MGFdictionary`, which in turn imports all
    prior modules and runs their `@register_prior` decorators. It is safe to
    call multiple times, and every public function in this module calls it, so
    callers rarely need to.

    Individual prior modules are isolated: one that cannot import is recorded
    and warned about without preventing the others from registering. See
    :func:`failed_prior_modules`.

    Returns
    -------
    None

    Raises
    ------
    ImportError
        If the `MGFdictionary` subpackage itself cannot be imported. This is a
        packaging or installation fault rather than a missing optional backend,
        so it is raised, unlike a failure in an individual prior module, which
        is recorded in `FAILED_MODULES` and only warned about.
    """
    global _LOADED

    if _LOADED:
        return

    # Deliberately not wrapped in try/except. A failure here means the
    # subpackage itself is broken or absent, which no caller can work around
    # and which must not be mistaken for "this prior does not exist".
    import jumufraktiv.MGFdictionary  # noqa: F401  (imported for its side effect)

    _LOADED = True


def failed_prior_modules():
    """
    Return the prior modules that failed to import, and why.

    Returns
    -------
    dict
        Maps module name (e.g. ``'paretoMGF'``) to the exception instance that
        stopped it. Empty when everything imported cleanly.

    Examples
    --------
    >>> from jumufraktiv import registry
    >>> registry.failed_prior_modules()
    {}
    """
    initialize()

    from jumufraktiv.MGFdictionary import FAILED_MODULES

    return dict(FAILED_MODULES)


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
        message = f"Prior '{name}' not found. Available: {sorted(PRIOR_REGISTRY)}"

        # A prior can be absent because its module failed to import rather than
        # because it does not exist. Saying so turns an unactionable "not found"
        # into a fixable one (usually: install an optional extra).
        failed = failed_prior_modules()
        if failed:
            details = "; ".join(
                f"{module} ({type(exc).__name__}: {exc})"
                for module, exc in sorted(failed.items())
            )
            message += (
                f". Note that {len(failed)} prior module(s) failed to import, so "
                f"priors they define are missing from that list: {details}"
            )

        raise KeyError(message)

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
    to produce a consistent output format expected by `MGFPrior`.

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
