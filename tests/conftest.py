"""Shared fixtures and analytic reference values for the test suite.

The reference values here are *closed forms*, not recorded outputs. For a
Gamma prior the MGF and all of its derivatives are known analytically, and a
Gamma prior against a Poisson likelihood is conjugate, so the whole posterior
is a Gamma with known parameters. That lets most of the suite assert
correctness rather than merely pinning current behaviour.
"""

import numpy as np
import pytest

# The canonical Gamma/Poisson problem lives in its own module so that the
# repository-root conftest.py can share it; see tests/canonical.py. Re-exported
# here because the suite imports it as `from conftest import ALPHA`.
from canonical import ALPHA as ALPHA
from canonical import BETA as BETA
from canonical import POISSON_DATA as POISSON_DATA
from canonical import POISSON_SCALE as POISSON_SCALE
from canonical import POST_RATE as POST_RATE
from canonical import POST_SHAPE as POST_SHAPE
from scipy.special import gammaln

from jumufraktiv import registry
from jumufraktiv.MGFDerivative_class import MGFDerivative
from jumufraktiv.mitMGFprior_class import mitMGFprior


@pytest.fixture(scope="session", autouse=True)
def _initialised_registry():
    """Populate the prior registry once for the whole session.

    ``mitMGFprior.from_registry`` does not initialise the registry itself; see
    ``test_known_broken.py::test_from_registry_initialises_registry``.
    """
    registry.initialize()


@pytest.fixture
def gamma_prior():
    """A Gamma(ALPHA, BETA) prior built through the registry."""
    return mitMGFprior.from_registry("gamma", params={"alpha": ALPHA, "beta": BETA})


@pytest.fixture
def poisson_posterior(gamma_prior):
    """The conjugate Gamma/Poisson posterior for the canonical test problem."""
    return MGFDerivative(
        gamma_prior,
        data=POISSON_DATA,
        likelihood="poisson",
        scale=POISSON_SCALE,
    )


# --------------------------------------------------------------------------
# Analytic references
# --------------------------------------------------------------------------
def gamma_mgf_derivative_log(order, t, alpha=ALPHA, beta=BETA):
    """Exact log of the ``order``-th derivative of a Gamma MGF at ``t``.

    For ``M(t) = (beta / (beta - t)) ** alpha`` the ``a``-th derivative has the
    closed form

    .. math::

       D^{a} M(t) = \\frac{\\Gamma(\\alpha + a)}{\\Gamma(\\alpha)}\\,
                    \\beta^{\\alpha}\\, (\\beta - t)^{-\\alpha - a},

    which is valid for non-integer ``a`` as well — it is the analytic
    continuation the package is meant to compute.

    Parameters
    ----------
    order : float
        Derivative order, integer or fractional.
    t : float or numpy.ndarray
        Evaluation point(s); must satisfy ``t < beta``.
    alpha, beta : float
        Gamma shape and rate.

    Returns
    -------
    float or numpy.ndarray
        Natural log of the derivative. The derivative is strictly positive for
        ``t < beta``, so no sign is returned.
    """
    t = np.asarray(t, dtype=float)
    return (
        gammaln(alpha + order)
        - gammaln(alpha)
        + alpha * np.log(beta)
        - (alpha + order) * np.log(beta - t)
    )


def poisson_log_evidence(data, scale=POISSON_SCALE, alpha=ALPHA, beta=BETA):
    """Exact log marginal likelihood for the Gamma/Poisson model."""
    y = np.asarray(data, dtype=float)
    s = np.full_like(y, float(scale))
    a = y.sum()
    b = s.sum()
    log_c = np.sum(y * np.log(s) - gammaln(y + 1.0))
    return log_c + gamma_mgf_derivative_log(a, -b, alpha=alpha, beta=beta)
