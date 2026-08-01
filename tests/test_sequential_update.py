"""Sequential updating, including for posteriors that could not update before.

Evidence factorises: conditioning on `y1` and then on `y2` must give the same
total as conditioning on both at once. That identity is the test, and it needs
no reference value — the two sides are computed by different routes through the
package and must agree.

Until the direct-expectation route existed, **no numeric posterior could update
at all**, and since no symbolic backend serves fractional orders, that meant no
fractional posterior could update by any route. The obstacle was that
`to_prior_object`'s numeric route produces a prior carrying a density and no
symbolic MGF, which every *differentiating* backend needs. The expectation route
reads the density and never touches the MGF, so the prior it produces is exactly
what that route consumes.
"""

import numpy as np
import pytest

from conftest import ALPHA, BETA, POISSON_DATA, POISSON_SCALE
from jumufraktiv.MGFDerivative_class import MGFDerivative


def _evidence(posterior):
    """The log evidence, checked finite.

    No sign to check: the evidence is a probability, so a negative value is a
    numerical failure and the constructor refuses it rather than returning a
    flag. See `MGFDerivative.evidence`.
    """
    log_evidence = float(posterior.evidence())
    assert np.isfinite(log_evidence)
    return log_evidence


@pytest.mark.parametrize(
    "likelihood,data,extra,new",
    [
        # a = n/2 = 1.5: a FRACTIONAL posterior, which could not update at all.
        ("halfnormal", [1.0, 2.0, 3.0], {}, [1.0]),
        # a = 1.5n = 4.5: fractional again, and a larger order.
        ("maxwell-boltzmann", [1.0, 2.0, 3.0], {}, [2.0]),
        # a = sum(y) = 6: integer, which used to work only via the symbolic route.
        ("poisson", POISSON_DATA, {"scale": POISSON_SCALE}, [2]),
    ],
)
def test_evidence_factorises_across_a_sequential_update(
    gamma_prior, likelihood, data, extra, new
):
    """p(y1, y2) = p(y1) * p(y2 | y1), whichever order the data arrives in."""
    staged = MGFDerivative(gamma_prior, data=data, likelihood=likelihood, **extra)
    updated = staged.update(new, likelihood=likelihood, **extra)
    one_shot = MGFDerivative(
        gamma_prior, data=list(data) + list(new), likelihood=likelihood, **extra
    )

    assert _evidence(staged) + _evidence(updated) == pytest.approx(
        _evidence(one_shot), rel=1e-8
    )


def test_a_fractional_posterior_can_be_updated(gamma_prior):
    """Named separately because this is the case that was impossible.

    `halfnormal` on three observations gives `a = 1.5`. Fractional orders have
    no working symbolic backend, and the old guard required one, so this raised
    `ValueError` regardless of the method requested.
    """
    posterior = MGFDerivative(
        gamma_prior, data=[1.0, 2.0, 3.0], likelihood="halfnormal"
    )

    assert posterior.a == pytest.approx(1.5)
    assert not posterior._deriv_is_symbolic

    updated = posterior.update([1.0], likelihood="halfnormal")

    assert isinstance(updated, MGFDerivative)
    assert np.isfinite(_evidence(updated))


def test_the_prior_built_from_a_posterior_exposes_a_density_not_a_log_density(
    gamma_prior,
):
    """`pdf_backend` must return p(theta), not log p(theta).

    It passed `log=self.log`, which is `True` by default, so a container
    expecting a density received a log density. Measured at theta = 1.0 it
    returned -3.14 where the density is 0.0432 — and nothing downstream could
    catch that, because a log density is a perfectly well-formed float that
    merely happens to be negative.
    """
    posterior = MGFDerivative(
        gamma_prior, data=[1.0, 2.0, 3.0], likelihood="halfnormal"
    )

    prior = posterior.to_prior_object()
    points = np.array([0.5, 1.0, 2.0])

    density = np.asarray(prior.pdf_func(points), dtype=float)
    expected = np.exp([posterior.post_density(x, log=True) for x in points])

    assert np.all(density > 0), "a density is never negative"
    assert density == pytest.approx(expected, rel=1e-10)


def test_updating_twice_matches_conditioning_on_everything_at_once(gamma_prior):
    """Two updates in a row, to check the result is itself updatable."""
    first = MGFDerivative(
        gamma_prior, data=POISSON_DATA, likelihood="poisson", scale=POISSON_SCALE
    )
    second = first.update([2], likelihood="poisson", scale=POISSON_SCALE)
    third = second.update([3], likelihood="poisson", scale=POISSON_SCALE)

    one_shot = MGFDerivative(
        gamma_prior,
        data=[*POISSON_DATA, 2, 3],
        likelihood="poisson",
        scale=POISSON_SCALE,
    )

    staged = _evidence(first) + _evidence(second) + _evidence(third)
    assert staged == pytest.approx(_evidence(one_shot), rel=1e-8)


def test_the_conjugate_update_lands_on_the_known_posterior(gamma_prior):
    """The Gamma/Poisson case has a closed form, so check the destination too.

    Factorisation alone could be satisfied by two consistently wrong numbers.
    Conditioning on `POISSON_DATA` then on `[2]` must reach the same posterior
    as conditioning on all of it: Gamma(ALPHA + sum(y), BETA + n*scale).
    """
    pytest.importorskip("scipy")
    from scipy.stats import gamma as gamma_dist

    first = MGFDerivative(
        gamma_prior, data=POISSON_DATA, likelihood="poisson", scale=POISSON_SCALE
    )
    updated = first.update([2], likelihood="poisson", scale=POISSON_SCALE)

    all_data = [*POISSON_DATA, 2]
    shape = ALPHA + sum(all_data)
    rate = BETA + len(all_data) * POISSON_SCALE

    for point in (0.5, 1.0, 2.0):
        assert updated.post_density(point, log=True) == pytest.approx(
            float(gamma_dist.logpdf(point, a=shape, scale=1.0 / rate)), rel=1e-6
        )
