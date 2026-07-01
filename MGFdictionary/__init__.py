import pkgutil
import importlib

_DISCOVERED_PRIORS = []


def discover_priors():
    for mod in pkgutil.iter_modules(__path__):
        name = f"{__name__}.{mod.name}"
        importlib.import_module(name)
        _DISCOVERED_PRIORS.append(name)


def list_discovered():
    return _DISCOVERED_PRIORS