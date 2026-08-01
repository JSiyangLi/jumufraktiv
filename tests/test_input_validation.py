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
from test_likelihood_stats import COUNTS, DATA, LIKELIHOOD_KWARGS

from conftest import POISSON_DATA
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


@pytest.mark.parametrize("offender", ["r", "s"])
def test_non_finite_vector_parameter_names_the_parameter(offender):
    """Vector known parameters must name themselves too, not report as "data".

    Dagum routes both `r` and `s` through a shared `_handle_param` helper whose
    array-like branches passed no label, so a NaN in a vector `r` was reported
    as if it were in the data.
    """
    pd = pytest.importorskip("pandas")
    ready, _, _ = LIKELIHOOD_REGISTRY["dagum"]
    kwargs = {"r": 1.0, "s": 1.0}
    kwargs[offender] = pd.DataFrame({"v": [1.0, np.nan, 1.0]})

    with pytest.raises(ValueError, match=f"{offender} contains NaN"):
        ready(DATA, **kwargs)


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

        assert np.isfinite(post.evidence())

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

        assert np.isfinite(post.evidence())

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
        assert np.isfinite(post.evidence())

    def test_improper_prior_is_rejected_at_every_order_at_the_origin(self):
        """The heaviside prior is improper: not even E[Theta^0] is finite."""
        prior = mitMGFprior.from_registry("heaviside", params={"k": 0.5})

        with pytest.raises(ValueError, match="t = 0"):
            MGFDerivative(prior, data=[0.5, 0.5], likelihood="laplace", mean=0.5)

    @pytest.mark.parametrize("order", [2, 3])
    def test_every_public_entry_point_applies_the_bound(self, order):
        """One question, three doors, one answer.

        The guard was reached only through `mgfDerivative` and the constructor.
        `mgfDerivative_integer` and `mgfDerivative_fractional` are advertised
        as main functions and exported at package level, and both skipped it:
        against Pareto(alpha=2) at t = 0 they returned `(inf, 1)` at order 2
        and raised `TypeError: Cannot convert complex to float` at order 3.
        The first of those is the worse one, because it is not an error.
        """
        from jumufraktiv.derivativeDispatch import (
            mgfDerivative,
            mgfDerivative_fractional,
            mgfDerivative_integer,
        )

        prior = mitMGFprior.from_registry("pareto", params={"alpha": 2.0, "xi": 1.0})

        with pytest.raises(ValueError, match="t = 0"):
            mgfDerivative(order, prior, method="symbolic", t=0.0)

        with pytest.raises(ValueError, match="t = 0"):
            mgfDerivative_integer(order, prior, method="symbolic", t=0.0)

        with pytest.raises(ValueError, match="t = 0"):
            mgfDerivative_fractional(order + 0.5, prior, method="scipy", t=0.0)

    def test_an_admissible_fractional_order_still_reaches_the_kernel(self):
        """The bound is about the caller's order, not the integrator's own step.

        The Caputo form of order `a` differentiates the MGF `floor(a) + 1`
        times, so a request for order 1.5 against Pareto(alpha=2) asks the
        integer backend for order 2 -- whose moment does not exist. Checking
        that intermediate would refuse a computable answer: `E[Theta^1.5]` is
        `2 / (2 - 1.5) = 4`, perfectly finite.
        """
        from jumufraktiv.derivativeDispatch import mgfDerivative_fractional

        prior = mitMGFprior.from_registry("pareto", params={"alpha": 2.0, "xi": 1.0})

        log_abs, sign = mgfDerivative_fractional(
            1.5, prior, method="scipy", t=0.0, log=True
        )

        assert sign == 1
        assert np.isfinite(log_abs)
        assert log_abs == pytest.approx(np.log(4.0), rel=1e-3)


REGISTRY_PARAMS = {
    "gamma": {"alpha": 2.0, "beta": 3.0},
    "pareto": {"alpha": 2.0, "xi": 1.0},
    "uniform": {"a": 0.5, "b": 2.0},
    "heaviside": {"k": 0.5},
}


def test_registry_params_table_covers_every_prior():
    registry.initialize()

    assert set(REGISTRY_PARAMS) == set(registry.list_priors())


@pytest.mark.parametrize("name", sorted(REGISTRY_PARAMS))
def test_registry_priors_all_declare_a_moment_domain(name):
    """A new prior must not silently inherit an unexamined default.

    This inspects the **spec returned by the factory**, not the compiled prior.
    Checking the compiled object cannot distinguish "declared infinity after
    working out that all moments exist" from "declared nothing and inherited
    the default", because both produce `max_finite_moment == inf`. An earlier
    version of this test asserted `>= 0.0` on the compiled prior, which every
    possible value satisfies — it enforced nothing at all.
    """
    registry.initialize()
    spec = registry.get_prior(name)(REGISTRY_PARAMS[name])

    assert "max_finite_moment" in spec, (
        f"prior '{name}' does not declare max_finite_moment. Work out the "
        f"supremum of orders a for which E[Theta^a] is finite and state it; "
        f"use float('inf') if all moments exist."
    )
    assert float(spec["max_finite_moment"]) >= 0.0


# ==========================================================================
# Refusals the docstrings advertise
# ==========================================================================
# Both of these were documented as capabilities the code does not have. A
# docstring that promises a return where the code raises is a defect in one of
# the two; these tests fix which one.
def test_a_symbolic_observation_is_refused_by_name():
    """`post_predictive` cannot take a symbol for the observation.

    The predictive density differentiates the posterior MGF `a(y_new)` times,
    so a symbolic `y_new` makes the differentiation *order* symbolic, which
    `sp.diff` cannot use. No backend avoids that, so the refusal belongs at the
    entry point, and it must name the argument the caller passed rather than an
    internal symbol standing in for a statistic.
    """
    import sympy as sp

    registry.initialize()
    prior = mitMGFprior.from_registry("gamma", params={"alpha": 2.0, "beta": 3.0})
    deriv = MGFDerivative(prior, data=POISSON_DATA, likelihood="poisson", scale=1.0)

    with pytest.raises(NotImplementedError) as excinfo:
        deriv.post_predictive(sp.Symbol("y_new", real=True))

    message = str(excinfo.value)
    assert "y_new" in message
    assert "a_new" not in message, "the message names an internal symbol"


def test_the_first_central_moment_is_zero_not_the_mean():
    """`E[Theta - E[Theta]]` is zero by construction.

    Worth asserting because the docstring called order 1 "the mean" for several
    releases, which is the raw moment. A caller who believed it got 0.0 and no
    error.
    """
    registry.initialize()
    prior = mitMGFprior.from_registry("gamma", params={"alpha": 2.0, "beta": 3.0})
    deriv = MGFDerivative(prior, data=POISSON_DATA, likelihood="poisson", scale=1.0)

    assert float(deriv.post_central_moment(order=1, log=False)) == 0.0

    log_abs, sign = deriv.post_central_moment(order=1, log=True)
    assert float(log_abs) == -np.inf
    assert sign == 1

    # The posterior mean is the *raw* first moment: Gamma(8, 6) has mean 8/6.
    assert float(deriv.post_raw_moment(1, log=False)) == pytest.approx(
        8.0 / 6.0, rel=1e-10
    )


def test_a_symbolic_path_does_not_mask_a_deliberate_refusal():
    """The refusal must reach the caller with its own type and message.

    Each symbolic path ends in `except Exception: raise RuntimeError(...)`, to
    say which quantity was being built when SymPy failed unexpectedly. That is
    useful for an unexpected failure and destructive for a deliberate one: it
    costs the caller the type they would catch and buries the message that says
    what to do instead. A symbolic moment order is the reachable example --
    `sp.diff` needs a concrete number of times to differentiate, so the order
    is refused, and the refusal names the free symbol.

    Only `NotImplementedError` is let through. SymPy's own `ValueError` and
    `TypeError` still get the wrapper, because those are the unexpected
    failures it exists to label -- so the second half of this test pins the
    boundary rather than only the repair.
    """
    import sympy as sp

    from jumufraktiv.symbols import param, t, theta

    # A free shape leaves the posterior symbolic, which is what routes the call
    # into the symbolic branch that carries the wrapper.
    alpha = param("alpha")
    free_prior = mitMGFprior(
        name="gamma_free_shape",
        mgf_sym=(3 / (3 - t)) ** alpha,
        pdf_sym=3**alpha * theta ** (alpha - 1) * sp.exp(-3 * theta) / sp.gamma(alpha),
        params={},
    ).as_mitMGFprior()
    deriv = MGFDerivative(
        free_prior,
        data=POISSON_DATA,
        likelihood="poisson",
        scale=1.0,
        method="symbolic",
    )

    with pytest.raises(NotImplementedError) as excinfo:
        deriv.post_raw_moment(sp.Symbol("q", real=True), log=False)

    message = str(excinfo.value)
    assert "q" in message, "the refusal does not name the free symbol"
    assert "computation failed" not in message, "the refusal was wrapped"


# ==========================================================================
# from_registry's hyperparameters
# ==========================================================================
#: A number and its SymPy spellings. `sympy` arithmetic produces `Integer` and
#: `Float` routinely, so a caller can hold one without ever having written
#: `sp.Integer` themselves.
def _sympy_spellings_of_two():
    import sympy as sp

    return {
        "python float": 2.0,
        "python int": 2,
        "sp.Integer": sp.Integer(2),
        "sp.Rational": sp.Rational(4, 2),
        "sp.Float": sp.Float(2.0),
        # An expression that carries symbols but cancels to a number. It has no
        # free symbols, so it is a number in the only sense that matters here.
        "cancelling expr": sp.Symbol("z") - sp.Symbol("z") + sp.Integer(2),
    }


@pytest.mark.parametrize("spelling", sorted(_sympy_spellings_of_two()))
@pytest.mark.parametrize("prior_name", ["gamma", "pareto"])
def test_a_sympy_number_is_a_number(prior_name, spelling):
    """A SymPy object with no free symbols is a hyperparameter, not a symbol.

    Rejecting on `isinstance(value, sp.Basic)` is too strict: it refuses
    `sp.Integer(2)`, which is worth exactly 2. It is also not strict enough in
    the other direction, since passing one *through* builds an object-dtype
    array inside the Pareto factory, which fails as "Cannot cast array data
    from dtype('O')" several frames from the argument at fault. Converting is
    what makes every spelling behave alike, which is what this asserts.
    """
    registry.initialize()

    value = _sympy_spellings_of_two()[spelling]
    other = {"gamma": {"beta": 3.0}, "pareto": {"xi": 1.0}}[prior_name]
    key = "alpha"

    reference = mitMGFprior.from_registry(prior_name, params={key: 2.0, **other})
    got = mitMGFprior.from_registry(prior_name, params={key: value, **other})

    assert float(got.mgf(-1.0)) == pytest.approx(float(reference.mgf(-1.0)), rel=1e-14)
    assert float(got.pdf_func(1.0)) == pytest.approx(
        float(reference.pdf_func(1.0)), rel=1e-14
    )


def test_a_free_symbol_is_refused_by_name():
    """The refusal must name the argument and the route that does work."""
    import sympy as sp

    registry.initialize()
    alpha = sp.Symbol("alpha", positive=True)

    with pytest.raises(TypeError, match=r"alpha, beta are symbolic"):
        mitMGFprior.from_registry(
            "gamma", params={"alpha": alpha, "beta": sp.Symbol("beta", positive=True)}
        )

    # An expression that still carries a symbol is refused for the same reason.
    with pytest.raises(TypeError, match=r"alpha is symbolic"):
        mitMGFprior.from_registry("gamma", params={"alpha": alpha + 1, "beta": 3.0})


@pytest.mark.parametrize(
    "value", [float("inf"), float("-inf"), float("nan")], ids=["inf", "-inf", "nan"]
)
def test_a_non_finite_hyperparameter_is_refused(value):
    """It used to build a prior whose every derived quantity was 0 or nan.

    `alpha=inf` gave `mgf(-1) == 0.0` and `alpha=nan` gave `nan`, with no error
    anywhere -- the failure mode that is worst to debug, because the number
    comes back.
    """
    registry.initialize()

    with pytest.raises(ValueError, match=r"finite hyperparameters"):
        mitMGFprior.from_registry("gamma", params={"alpha": value, "beta": 3.0})


def test_a_non_finite_hyperparameter_is_refused_in_its_sympy_spelling_too():
    """The same rule for both spellings.

    A check that exists in one place and not in the neighbouring one is this
    repository's most-repeated defect, so `sp.oo` and `float('inf')` are held
    to one rule rather than two that agree by inspection.
    """
    import sympy as sp

    registry.initialize()

    with pytest.raises(ValueError, match=r"finite hyperparameters"):
        mitMGFprior.from_registry("gamma", params={"alpha": sp.oo, "beta": 3.0})

    # Complex infinity has no float at all, so it is reported as not a number
    # rather than as a number of the wrong size.
    with pytest.raises(TypeError, match=r"cannot be converted to a float"):
        mitMGFprior.from_registry("gamma", params={"alpha": sp.zoo, "beta": 3.0})


def test_a_hyperparameter_that_is_not_a_number_names_its_type():
    registry.initialize()

    with pytest.raises(TypeError, match=r"alpha \(str\) cannot be converted"):
        mitMGFprior.from_registry("gamma", params={"alpha": "two", "beta": 3.0})
