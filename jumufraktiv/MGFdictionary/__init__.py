"""Automatic discovery of prior modules.

Importing this subpackage imports every module in it whose filename contains
``MGF``, which runs their ``@register_prior`` decorators and populates
:data:`jumufraktiv.registry.PRIOR_REGISTRY`.

Discovery is **isolated per module**. Each module is imported inside its own
``try``/``except`` so that one module which cannot import — because an optional
backend is missing, or because it has a genuine bug — cannot stop the modules
after it from registering. Before this isolation existed, a single eager
``import torch`` in ``paretoMGF`` aborted the whole loop and silently removed
both ``pareto`` and ``uniform`` from the registry. That import has since been
deleted along with the function it served, so the example is history rather
than something a reader can go and look at; the isolation stays because the
next optional backend will pose the same risk.

The broad ``except`` here is deliberate and is not the "swallow a real failure"
anti-pattern that :file:`CLAUDE.md` prohibits. A prior module is effectively a
plugin, so the exceptions it can raise are not enumerable in advance. The
failure is never discarded: it is recorded in :data:`FAILED_MODULES`, warned
about by name, and reported again by
:func:`jumufraktiv.registry.get_prior` if anyone asks for a prior that is
missing as a result.
"""

import importlib
import pkgutil
import warnings
from pathlib import Path

#: Modules that failed to import, mapped to the exception that stopped them.
#: Inspect with :func:`jumufraktiv.registry.failed_prior_modules`.
FAILED_MODULES: dict[str, Exception] = {}


def _discover() -> None:
    """Import every prior module, isolating failures to the module that caused them."""
    package_dir = Path(__file__).resolve().parent

    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.ispkg:
            continue

        name = module_info.name
        if "MGF" not in name:
            continue

        try:
            importlib.import_module(f"{__name__}.{name}")
        except Exception as exc:
            FAILED_MODULES[name] = exc
            warnings.warn(
                f"Prior module '{name}' could not be imported, so the priors it "
                f"defines are unavailable: {type(exc).__name__}: {exc}. "
                f"Other prior modules are unaffected.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            # A module that previously failed and now imports cleanly (e.g. after
            # an optional dependency was installed) should not stay on the list.
            FAILED_MODULES.pop(name, None)


_discover()
