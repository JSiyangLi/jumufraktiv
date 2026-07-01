# jumufraktiv/registry.py

import warnings

PRIOR_REGISTRY = {}
_LOADED = False


# ============================================================
# Registration decorator (used by priors)
# ============================================================
def register_prior(name: str):
    """
    Decorator used inside MGFdictionary prior modules.

    Example:
        @register_prior("gamma")
        def gamma_prior(...):
            ...
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
    Safe to call multiple times.
    """
    global _LOADED

    if _LOADED:
        return

    try:
        import jumufraktiv.MGFdictionary  # triggers auto-registration
    except Exception as e:
        warnings.warn(f"[registry] Failed to load MGFdictionary: {e}")

    _LOADED = True


# ============================================================
# Public API
# ============================================================
def get_prior(name: str):
    """
    Retrieve a registered prior by name.
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
    List all registered priors.
    """
    initialize()
    return list(PRIOR_REGISTRY.keys())