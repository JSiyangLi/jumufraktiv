"""Correctness tests against closed-form references.

Every assertion here compares the package against mathematics, not against a
recorded output, so these tests remain meaningful after a refactor changes how
a number is produced.
"""

import numpy as np
import pytest
from conftest import (
    POISSON_DATA,
    POISSON_SCALE,
    POST_RATE,
    POST_SHAPE,
    gamma_mgf_derivative_log,
    poisson_log_evidence,
)
from scipy.special import gammaln
from scipy.stats import gamma as scipy_gamma

from jumufraktiv.derivativeDispatch import mgfDerivative


# ==========================================================================
# The derivative itself
# ==========================================================================
@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("t", [-5.0, -2.0, -1.0, -0.25, 0.0, 1.0, 2.5])
def test_integer_derivative_matches_closed_form(gamma_prior, order, t):
    """D^n M(t) for a Gamma prior must match the exact formula."""
    log_abs, sign = mgfDerivative(order, gamma_prior, method="symbolic", t=t, log=True)

    assert sign == 1, "the Gamma MGF and all its derivatives are positive for t < beta"
    assert log_abs == pytest.approx(gamma_mgf_derivative_log(order, t), rel=1e-10)


@pytest.mark.parametrize("order", [0, 1, 3])
def test_bell_and_symbolic_backends_agree(gamma_prior, order):
    """Different integer backends must produce the same number."""
    t = -1.5
    sym_log, sym_sign = mgfDerivative(
        order, gamma_prior, method="symbolic", t=t, log=True
    )
    bell_log, bell_sign = mgfDerivative(
        order, gamma_prior, method="bell", t=t, log=True
    )

    assert sym_sign == bell_sign
    assert bell_log == pytest.approx(sym_log, rel=1e-8)


# ==========================================================================
# Conjugate posterior: Gamma prior + Poisson likelihood
# ==========================================================================
def test_evidence_matches_closed_form(poisson_posterior):
    log_ev, sign = poisson_posterior.evidence()

    assert sign == 1
    assert log_ev == pytest.approx(poisson_log_evidence(POISSON_DATA), rel=1e-12)


@pytest.mark.parametrize("theta", [0.1, 0.5, 1.0, 1.3333, 2.0, 5.0])
def test_post_density_matches_conjugate_gamma(poisson_posterior, theta):
    """The posterior density must equal the Gamma(POST_SHAPE, POST_RATE) pdf."""
    expected = scipy_gamma.logpdf(theta, a=POST_SHAPE, scale=1.0 / POST_RATE)

    assert poisson_posterior.post_density(theta) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("u", [0.1, 0.5, 1.0, 2.0, 3.0, 5.0])
def test_post_cdf_matches_conjugate_gamma(poisson_posterior, u):
    """The posterior CDF must equal the Gamma(POST_SHAPE, POST_RATE) cdf."""
    expected = scipy_gamma.logcdf(u, a=POST_SHAPE, scale=1.0 / POST_RATE)

    assert poisson_posterior.post_cdf(u) == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize("r", [-1.0, -0.1, 0.1, 1.0, 3.0])
def test_post_mgf_matches_conjugate_gamma(poisson_posterior, r):
    """The posterior MGF must equal (rate / (rate - r)) ** shape."""
    expected = POST_SHAPE * (np.log(POST_RATE) - np.log(POST_RATE - r))

    assert poisson_posterior.post_mgf(r) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("q", [1, 2, 3])
def test_post_raw_moment_matches_conjugate_gamma(poisson_posterior, q):
    """E[theta^q] = Gamma(shape + q) / Gamma(shape) / rate^q."""
    expected = gammaln(POST_SHAPE + q) - gammaln(POST_SHAPE) - q * np.log(POST_RATE)

    assert poisson_posterior.post_raw_moment(q) == pytest.approx(expected, rel=1e-8)


def test_post_central_moment_matches_gamma_variance(poisson_posterior):
    """The second central moment is the Gamma variance, shape / rate**2."""
    log_var, sign = poisson_posterior.post_central_moment(2)

    assert sign == 1
    assert log_var == pytest.approx(np.log(POST_SHAPE / POST_RATE**2), rel=1e-8)


# ==========================================================================
# Posterior predictive: Gamma/Poisson is negative-binomial
# ==========================================================================
@pytest.mark.parametrize("y_new", [0, 1, 2, 5, 10])
def test_post_predictive_matches_negative_binomial(poisson_posterior, y_new):
    """p(y_new | y) is negative-binomial with the posterior Gamma parameters."""
    s = POISSON_SCALE
    expected = (
        gammaln(POST_SHAPE + y_new)
        - gammaln(POST_SHAPE)
        - gammaln(y_new + 1.0)
        + POST_SHAPE * (np.log(POST_RATE) - np.log(POST_RATE + s))
        + y_new * (np.log(s) - np.log(POST_RATE + s))
    )
    got = poisson_posterior.post_predictive([y_new], scale=s)

    assert float(np.ravel(got)[0]) == pytest.approx(expected, rel=1e-8)


@pytest.mark.slow
def test_predictive_masses_sum_to_one(poisson_posterior):
    """The predictive is a proper pmf over the non-negative integers."""
    y = np.arange(0, 200)
    log_p = poisson_posterior.post_predictive(y, scale=POISSON_SCALE)

    assert np.exp(log_p).sum() == pytest.approx(1.0, abs=1e-6)


# ==========================================================================
# Sequential updating
# ==========================================================================
def test_sequential_update_equals_batch_evidence(gamma_prior):
    """Conditioning on data in two chunks must match conditioning in one.

    Evidence factorises as p(y1, y2) = p(y1) * p(y2 | y1), so the sum of the
    two staged log evidences equals the batch log evidence.
    """
    from jumufraktiv.MGFDerivative_class import MGFDerivative

    first, second = [1, 2], [3, 4]

    batch = MGFDerivative(
        gamma_prior, data=first + second, likelihood="poisson", scale=POISSON_SCALE
    )
    log_batch, _ = batch.evidence()

    staged = MGFDerivative(
        gamma_prior, data=first, likelihood="poisson", scale=POISSON_SCALE
    )
    log_first, _ = staged.evidence()
    log_second, _ = staged.update(second, scale=POISSON_SCALE).evidence()

    assert log_first + log_second == pytest.approx(log_batch, rel=1e-8)
