"""Bayesian inference for the MGF-marginalisable family of likelihoods.

See :file:`README.md` for the method and :file:`CLAUDE.md` for the internals.
"""

import sys

from ._version import __version__
from .derivativeDispatch import (
    mgfDerivative,
    mgfDerivative_fractional,
    mgfDerivative_integer,
)
from .MGFDerivative_class import MGFDerivative
from .mitMGFprior_class import mitMGFprior

#: Backward-compatibility alias from the package's former name. Nothing in the
#: package or the suite imports it; it is here for callers who have not
#: migrated. Scheduled for removal -- see "Deferred decisions" in CLAUDE.md.
#: It sits below the imports rather than above them because it does not need
#: to run first: no module reached during those imports refers to `mgf2post`.
sys.modules["mgf2post"] = sys.modules[__name__]

#: The names re-exported at package level. Listed explicitly so that the
#: re-exports read as intentional rather than as unused imports. PR 12 decides
#: whether this is the *right* set; this records the set as it stands.
__all__ = [
    "MGFDerivative",
    "__version__",
    "mgfDerivative",
    "mgfDerivative_fractional",
    "mgfDerivative_integer",
    "mitMGFprior",
]
