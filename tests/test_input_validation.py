"""Input-validation tests for the likelihood layer and the moment domain.

Two distinct defects, at two different layers, for a reason worth stating.

**NaN and infinity are properties of the data**, so they are rejected in
``like_stats`` — which the layer rule permits, since those modules are pure
functions of the data.

**Whether ``b = 0`` is admissible is a property of the prior**, so it cannot be
decided in ``like_stats`` at all. The same ``b = 0`` is fine against a Gamma
prior at every order and fatal against a Pareto(2) prior at order 2. It is
therefore checked where both the order and the prior are visible.
"""

import numpy as np
import pytest
from conftest import POISSON_DATA
from test_likelihood_stats import COUNTS, DATA, LIKELIHOOD_KWARGS

from jumufraktiv import registry
from jumufraktiv.MGFDerivative_class import LIKELIHOOD_REGISTRY, MGFDerivative
from jumufraktiv.mitMGFprior_class import mitMGFprior

ALL_LIKELIHOODS = sorted(LIKELIHOOD_REGISTRY)


def _data_for(name):
    return COUNTS if name == "poisson" else DATA


# ==========================================================================
# Non-finite data
# ==========================================================================
@pytest.mark.parametrize("name", ALL_LIKELIHOODS)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_data_is_rejected(name, bad):
    """NaN used to pass every guard, because `np.any(x <= 0)` is False for NaN.

    It then landed in `a`, `b` or `log_c` and surfaced much later as an error
    naming the wrong thing — "Derivative at t=-b is negative" for Rayleigh,
    "t must be provided" for Normal, "cannot convert float NaN to integer" for
    Poisson.
    """
    ready, _, _ = LIKELIHOOD_REGISTRY[name]
    data = [*_data_for(name)[:-1], bad]

    with pytest.raises(ValueError, match="finite"):
        ready(data, **LIKELIHOOD_KWARGS[name])


@pytest.mark.parametrize("name", ALL_LIKELIHOODS)
def test_non_finite_data_is_rejected_pointwise_too(name):
    """The per-element form shares the guard, not just the aggregated one."""
    _, _, bereit = LIKELIHOOD_REGISTRY[name]
    data = [*_data_for(name)[:-1], np.nan]

    with pytest.raises(ValueError, match="finite"):
        bereit(data, **LIKELIHOOD_KWARGS[name])


def test_rejection_message_names_the_data():
    ready, _, _ = LIKELIHOOD_REGISTRY["rayleigh"]

    with pytest.raises(ValueError, match="data contains NaN"):
        ready([1.0, np.nan])


def test_infinity_is_reported_as_infinite_not_nan():
    ready, _, _ = LIKELIHOOD_REGISTRY["rayleigh"]

    with pytest.raises(ValueError, match="infinite"):
        ready([1.0, np.inf])


# ==========================================================================
# Non-finite known parameters
# ==========================================================================
@pytest.mark.parametrize(
    ("name", "kwargs", "offender"),
    [
        ("gamma", {"shape": np.nan}, "shape"),
        ("normal", {"mean": np.nan}, "mean"),
        ("weibull", {"rho": np.nan}, "rho"),
        ("poisson", {"scale": np.nan}, "scale"),
        ("levy", {"location": np.nan}, "location"),
        ("burrxii", {"known_shape": np.nan}, "known_shape"),
        ("dagum", {"r": 1.0, "s": np.nan}, "s"),
    ],
)
def test_non_finite_scalar_parameter_names_the_parameter(name, kwargs, offender):
    """A scalar known parameter took a different branch and bypassed the guard.

    `np.full(n, float(x))` skipped the extraction helper entirely, so
    `shape=nan` propagated silently even after the data path was fixed.
    """
    ready, _, _ = LIKELIHOOD_REGISTRY[name]

    with pytest.raises(ValueError, match=offender):
        ready(_data_for(name), **kwargs)


# ==========================================================================
# The guard must not over-reject
# ==========================================================================
@pytest.mark.parametrize("name", ALL_LIKELIHOODS)
def test_valid_data_is_still_accepted(name):
    ready, _, _ = LIKELIHOOD_REGISTRY[name]
    stats = ready(_data_for(name), **LIKELIHOOD_KWARGS[name])

    assert all(np.isfinite(float(v)) for v in stats.values())


def test_vector_known_parameters_still_work():
    pd = pytest.importorskip("pandas")
    ready, _, _ = LIKELIHOOD_REGISTRY["poisson"]

    stats = ready([1, 2, 3], scale=pd.DataFrame({"s": [1.0, 2.0, 3.0]}))

    assert stats["b"] == pytest.approx(6.0)


# ==========================================================================
# The moment domain at t = 0
# ==========================================================================
class TestMomentDomain:
    @pytest.mark.parametrize(
        ("prior_name", "params", "expected"),
        [
            ("gamma", {"alpha": 2.0, "beta": 3.0}, np.inf),
            ("uniform", {"a": 0.5, "b": 2.0}, np.inf),
            ("pareto", {"alpha": 2.0, "xi": 1.0}, 2.0),
            ("heaviside", {"k": 0.5}, 0.0),
        ],
    )
    def test_declared_moment_domains(self, prior_name, params, expected):
        """Verified symbolically: Pareto's is its tail index, heaviside has none.

        `E[Theta^a]` for Pareto(alpha) converges iff `a < alpha`. The heaviside
        prior is improper, so the integral diverges for every `a >= 0`, `a = 0`
        included, and its MGF exists only for `t < 0`.
        """
        prior = mitMGFprior.from_registry(prior_name, params=params)

        assert prior.max_finite_moment == expected

    def test_custom_priors_default_to_no_restriction(self, gamma_prior):
        """The safe default defers to the numerical result rather than guessing."""
        assert mitMGFprior().max_finite_moment == np.inf

    @pytest.mark.parametrize("order", [2, 3])
    def test_order_at_or_above_the_bound_is_rejected_at_the_origin(self, order):
        """b = 0 put the evaluation point at t = 0, where E[Theta^a] must exist.

        Previously this returned `inf` at order 2 and raised
        `TypeError: Cannot convert complex to float` at order 3, neither of
        which named the cause.
        """
        prior = mitMGFprior.from_registry("pareto", params={"alpha": 2.0, "xi": 1.0})

        with pytest.raises(ValueError, match="t = 0"):
            MGFDerivative(prior, data=[0.1] * order, likelihood="pareto", scale=0.1)

    def test_order_below_the_bound_is_accepted_at_the_origin(self):
        prior = mitMGFprior.from_registry("pareto", params={"alpha": 2.0, "xi": 1.0})
        post = MGFDerivative(prior, data=[0.1], likelihood="pareto", scale=0.1)

        assert np.isfinite(post.evidence()[0])

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_a_prior_with_all_moments_is_unaffected_at_the_origin(
        self, gamma_prior, order
    ):
        """The same b = 0 is perfectly valid against a Gamma prior.

        This is why the check cannot live in `like_stats`: admissibility is not
        a property of the data.
        """
        post = MGFDerivative(
            gamma_prior, data=[0.1] * order, likelihood="pareto", scale=0.1
        )

        assert np.isfinite(post.evidence()[0])

    @pytest.mark.parametrize("prior_name", ["pareto", "heaviside"])
    def test_restricted_priors_are_unaffected_away_from_the_origin(self, prior_name):
        """No moment condition applies at t < 0, and none must be imposed.

        Enforcing one there would wrongly reject the heavy-tailed priors the
        operator exists to support.
        """
        params = {"alpha": 2.0, "xi": 1.0} if prior_name == "pareto" else {"k": 0.5}
        prior = mitMGFprior.from_registry(prior_name, params=params)

        post = MGFDerivative(prior, data=POISSON_DATA, likelihood="poisson", scale=1.0)

        assert post.b > 0
        assert np.isfinite(post.evidence()[0])

    def test_improper_prior_is_rejected_at_every_order_at_the_origin(self):
        """The heaviside prior is improper: not even E[Theta^0] is finite."""
        prior = mitMGFprior.from_registry("heaviside", params={"k": 0.5})

        with pytest.raises(ValueError, match="t = 0"):
            MGFDerivative(prior, data=[0.5, 0.5], likelihood="laplace", mean=0.5)


def test_registry_priors_all_declare_a_moment_domain():
    """A new prior must not silently inherit an unexamined default."""
    registry.initialize()

    for name in registry.list_priors():
        params = {
            "gamma": {"alpha": 2.0, "beta": 3.0},
            "pareto": {"alpha": 2.0, "xi": 1.0},
            "uniform": {"a": 0.5, "b": 2.0},
            "heaviside": {"k": 0.5},
        }[name]
        prior = mitMGFprior.from_registry(name, params=params)

        assert prior.max_finite_moment >= 0.0
