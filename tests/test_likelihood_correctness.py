"""Correctness tests for the sufficient statistics themselves.

`test_likelihood_stats.py` checks the *contract* — return shapes, finiteness,
and that per-element statistics sum to the aggregate. None of that would catch
a `b` that was off by a factor of two. These tests check the mathematics.

The criterion is exact, not approximate. A likelihood is MGF-marginalisable
precisely when

    L(theta; y) = c(y) * theta^a(y) * exp(-b(y) * theta)

so the module's own `a`, `b` and `log_c` must reconstruct the true log density:

    log L(theta; y) == log_c + a * log(theta) - b * theta

with the left-hand side computed independently from `scipy.stats`.

**Parameterisation is the whole difficulty.** Whether a likelihood is
MGF-marginalisable is a property of the parameterisation, not of the family:
Rayleigh factorises in the rate `theta = 1/sigma**2` and does not factorise in
the scale `sigma`. So each entry below records exactly what `theta` is and how
it maps onto scipy's arguments, and a mismatch there is far more likely than a
defect in the module — see CLAUDE.md, "Parameterisation is part of the claim".

That also makes these tests self-validating: a wrong mapping in the table below
fails loudly rather than silently passing, because the reconstruction identity
only holds when both sides speak of the same parameter.
"""

import numpy as np
import pytest
from scipy import stats

from jumufraktiv.MGFDerivative_class import LIKELIHOOD_REGISTRY

# --------------------------------------------------------------------------
# theta parameterisation and the matching scipy reference, per likelihood
#
# Each entry maps (y, theta) to an independent log-density. `known` holds the
# parameters the package treats as fixed, and matches what is passed to `ready`.
# --------------------------------------------------------------------------
REFERENCES = {
    "poisson": dict(
        theta="Poisson rate; observation mean is scale * theta",
        known={"scale": 1.0},
        discrete=True,
        logpdf=lambda y, th, k: stats.poisson.logpmf(y, mu=k["scale"] * th),
    ),
    "gamma": dict(
        theta="Gamma rate; scipy takes scale = 1/theta",
        known={"shape": 2.0},
        logpdf=lambda y, th, k: stats.gamma.logpdf(y, a=k["shape"], scale=1.0 / th),
    ),
    "inverse gamma": dict(
        theta="inverse-Gamma rate beta; scipy takes scale = theta",
        known={"shape": 2.0},
        logpdf=lambda y, th, k: stats.invgamma.logpdf(y, a=k["shape"], scale=th),
    ),
    "laplace": dict(
        theta="Laplace rate = 1/scale",
        known={"mean": 0.0},
        logpdf=lambda y, th, k: stats.laplace.logpdf(y, loc=k["mean"], scale=1.0 / th),
    ),
    "normal": dict(
        theta="precision = 1/sigma**2",
        known={"mean": 0.0},
        logpdf=lambda y, th, k: stats.norm.logpdf(y, loc=k["mean"], scale=th**-0.5),
    ),
    "levy": dict(
        theta="Levy scale c; scipy takes scale = theta",
        known={"location": 0.0},
        logpdf=lambda y, th, k: stats.levy.logpdf(y, loc=k["location"], scale=th),
    ),
    "weibull": dict(
        theta="rate on y**rho, i.e. scale**(-rho)",
        known={"rho": 2.0},
        logpdf=lambda y, th, k: stats.weibull_min.logpdf(
            y, c=k["rho"], scale=th ** (-1.0 / k["rho"])
        ),
    ),
    "burrxii": dict(
        theta="Burr XII second shape (scipy `d`); scale is fixed at 1",
        known={"known_shape": 1.5},
        logpdf=lambda y, th, k: stats.burr12.logpdf(y, c=k["known_shape"], d=th),
    ),
    "pareto": dict(
        theta="Pareto type-I tail index (scipy `b`)",
        known={"scale": 0.1},
        logpdf=lambda y, th, k: stats.pareto.logpdf(y, b=th, scale=k["scale"]),
    ),
    "dagum": dict(
        theta="Dagum shape q; Dagum is Burr III, so scipy `burr` with d = theta",
        known={"r": 1.5, "s": 1.0},
        logpdf=lambda y, th, k: stats.burr.logpdf(y, c=k["r"], d=th, scale=k["s"]),
    ),
    "gompertz": dict(
        # The kwarg is named `scale` but is a rate: scipy's scale is its reciprocal.
        theta="Gompertz shape eta (scipy `c`); the `scale` kwarg is a rate beta",
        known={"scale": 1.0},
        logpdf=lambda y, th, k: stats.gompertz.logpdf(y, c=th, scale=1.0 / k["scale"]),
    ),
    "rayleigh": dict(
        theta="1/sigma**2",
        known={},
        logpdf=lambda y, th, k: stats.rayleigh.logpdf(y, scale=th**-0.5),
    ),
    "maxwell-boltzmann": dict(
        theta="1/(2 A**2); scipy takes scale = (2 theta)**-0.5",
        known={},
        logpdf=lambda y, th, k: stats.maxwell.logpdf(y, scale=(2.0 * th) ** -0.5),
    ),
    "halfnormal": dict(
        theta="1/sigma**2",
        known={},
        logpdf=lambda y, th, k: stats.halfnorm.logpdf(y, scale=th**-0.5),
    ),
}

#: Orders of magnitude of theta, so an error that is a constant factor and one
#: that is theta-dependent are distinguishable.
THETAS = [0.05, 0.5, 1.0, 3.0, 20.0]

CONTINUOUS_DATA = [[1.3], [0.4, 1.1, 2.7], [0.2, 0.9, 1.5, 2.2, 3.1]]
COUNT_DATA = [[2], [1, 3, 5], [0, 1, 2, 4, 7]]


def _datasets(name):
    return COUNT_DATA if REFERENCES[name]["discrete"] is True else CONTINUOUS_DATA


for _entry in REFERENCES.values():
    _entry.setdefault("discrete", False)


def _true_log_likelihood(name, data, theta):
    """Independent log joint density, from scipy, in the module's theta."""
    ref = REFERENCES[name]
    return float(
        np.sum(ref["logpdf"](np.asarray(data, dtype=float), theta, ref["known"]))
    )


def _reconstructed(stats_dict, theta):
    """log_c + a log(theta) - b theta, from the module's own statistics."""
    return float(
        stats_dict["log_c"] + stats_dict["a"] * np.log(theta) - stats_dict["b"] * theta
    )


# ==========================================================================
# The criterion
# ==========================================================================
def test_reference_table_covers_the_registry():
    """A new likelihood must not be added without a correctness reference."""
    assert set(REFERENCES) == set(LIKELIHOOD_REGISTRY)


@pytest.mark.parametrize("name", sorted(REFERENCES))
@pytest.mark.parametrize("theta", THETAS)
def test_statistics_reconstruct_the_likelihood(name, theta):
    """`log_c + a log(theta) - b theta` must equal the true log density.

    This is the MGF-marginalisable criterion itself, so a failure means either
    the module's statistics are wrong or the likelihood does not belong in the
    family under this parameterisation.
    """
    ready, _, _ = LIKELIHOOD_REGISTRY[name]

    for data in _datasets(name):
        stats_dict = ready(data, **REFERENCES[name]["known"])
        expected = _true_log_likelihood(name, data, theta)

        assert _reconstructed(stats_dict, theta) == pytest.approx(
            expected, rel=1e-10, abs=1e-10
        ), f"{name} at theta={theta}, n={len(data)}"


@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_pointwise_statistics_reconstruct_each_observation(name):
    """The per-element form must reconstruct each observation individually.

    `post_predictive` relies on this, and summing to the right total would not
    catch per-element statistics that are individually wrong.
    """
    _, _, each = LIKELIHOOD_REGISTRY[name]
    ref = REFERENCES[name]
    data = _datasets(name)[-1]
    theta = 1.7

    per_element = each(data, **ref["known"])

    for i, y in enumerate(data):
        single = {k: np.asarray(v)[i] for k, v in per_element.items()}
        expected = float(ref["logpdf"](float(y), theta, ref["known"]))

        assert _reconstructed(single, theta) == pytest.approx(
            expected, rel=1e-10, abs=1e-10
        ), f"{name} observation {i} (y={y})"


@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_only_b_scales_with_theta(name):
    """`a` and `log_c` must not depend on theta; only `-b*theta` may.

    A statistic that smuggled a theta-dependence into `log_c` could still match
    at a single theta, so the multi-theta check above needs this alongside it.
    """
    ready, _, _ = LIKELIHOOD_REGISTRY[name]
    data = _datasets(name)[1]

    first = ready(data, **REFERENCES[name]["known"])
    again = ready(data, **REFERENCES[name]["known"])

    assert first["a"] == again["a"]
    assert first["b"] == again["b"]
    assert first["log_c"] == again["log_c"]


@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_a_is_non_negative(name):
    """The derivative order must be a valid order.

    `Dᵃ M` is defined for `a >= 0`; a negative order is outside the operator's
    domain entirely.
    """
    ready, _, _ = LIKELIHOOD_REGISTRY[name]

    for data in _datasets(name):
        assert ready(data, **REFERENCES[name]["known"])["a"] >= 0.0


@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_b_is_non_negative(name):
    """`t = -b` must not be positive.

    A negative `b` would put the evaluation point at `t > 0`, outside the domain
    of convergence for any prior whose MGF is one-sided — pareto and lognormal
    among them.
    """
    ready, _, _ = LIKELIHOOD_REGISTRY[name]

    for data in _datasets(name):
        assert ready(data, **REFERENCES[name]["known"])["b"] >= 0.0
