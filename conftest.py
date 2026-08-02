"""Fixtures for the docstring examples.

The examples in `MGFDerivative` and `MGFPrior` are written against a `deriv`
and a `prior` the reader is expected to have built already, which is the
NumPyDoc convention and keeps each example to the one call it illustrates.
Injecting those objects into the doctest namespace is what makes the examples
executable, so ``--doctest-modules`` checks the printed values instead of
trusting them.

This file sits at the repository root rather than in ``tests/`` because
``pytest --doctest-modules jumufraktiv/...`` collects from the package
directory and therefore never loads ``tests/conftest.py``. A conftest here is
an ancestor of both, so one copy of the fixture serves the suite and the
docstring examples alike.
"""

import pytest

# `pythonpath = ["tests"]` in pyproject.toml puts this on the path. It cannot be
# imported from `tests/conftest.py` instead: the rootdir precedes `tests/` on
# sys.path, so `import conftest` from here resolves to this file.
from canonical import ALPHA, BETA, POISSON_DATA, POISSON_SCALE

from jumufraktiv.MGFDerivative_class import MGFDerivative
from jumufraktiv.MGFPrior_class import MGFPrior


@pytest.fixture(scope="session", autouse=True)
def _doctest_namespace(doctest_namespace):
    """Give every ``Examples`` section the objects it refers to.

    Session-scoped because `doctest_namespace` is: the canonical objects are
    built once, not once per test.
    """
    import numpy as _np
    import sympy as _sp

    prior = MGFPrior.from_registry("gamma", params={"alpha": ALPHA, "beta": BETA})
    deriv = MGFDerivative(
        prior, data=POISSON_DATA, likelihood="poisson", scale=POISSON_SCALE
    )

    doctest_namespace["np"] = _np
    doctest_namespace["sp"] = _sp
    doctest_namespace["MGFPrior"] = MGFPrior
    doctest_namespace["MGFDerivative"] = MGFDerivative
    doctest_namespace["prior"] = prior
    doctest_namespace["gamma_prior"] = prior
    doctest_namespace["deriv"] = deriv
