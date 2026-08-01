"""Bayesian inference for the MGF-marginalisable family of likelihoods.

See :file:`README.md` for the method and :file:`CLAUDE.md` for the internals.

The names below are the supported surface. Three workflows reach for them:

* **Infer.** Build a prior with :class:`mitMGFprior`, hand it and your data to
  :class:`MGFDerivative`, and ask that for the evidence, density, CDF,
  quantiles, moments, predictive or a sequential update.
* **Compute a single derivative.** :func:`mgfDerivative` and its two
  order-specific siblings evaluate ``D^a M(t)`` directly, without a posterior.
* **Add a prior.** Decorate a factory with :func:`register_prior` and have it
  return :func:`make_prior_spec`; :func:`list_priors` reports what is
  available and :func:`failed_prior_modules` reports what failed to import.

Writing a prior also needs the canonical SymPy symbols, which live in
:mod:`jumufraktiv.symbols` and must be imported from there rather than
redefined -- two symbols that print alike but were constructed separately do
not substitute for each other, so a redefined one silently fails to match.
"""

from ._version import __version__
from .derivativeDispatch import (
    mgfDerivative,
    mgfDerivative_fractional,
    mgfDerivative_integer,
)
from .MGFDerivative_class import MGFDerivative
from .mitMGFprior_class import mitMGFprior
from .registry import (
    failed_prior_modules,
    list_priors,
    make_prior_spec,
    register_prior,
)

#: The names re-exported at package level. Listed explicitly so that the
#: re-exports read as intentional rather than as unused imports.
__all__ = [
    "MGFDerivative",
    "__version__",
    "failed_prior_modules",
    "list_priors",
    "make_prior_spec",
    "mgfDerivative",
    "mgfDerivative_fractional",
    "mgfDerivative_integer",
    "mitMGFprior",
    "register_prior",
]
