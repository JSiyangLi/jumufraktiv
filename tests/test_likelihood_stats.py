"""Contract tests for the ``like_stats`` modules.

Every likelihood module exports the same three functions, and the aggregated
and per-element forms must agree. PR 7 will de-duplicate these fourteen modules
(they currently carry byte-identical copies of two private helpers), so pinning
the shared contract first is what makes that refactor safe.
"""

import numpy as np
import pytest
import sympy as sp

from jumufraktiv.MGFDerivative_class import LIKELIHOOD_REGISTRY

#: Positive data, valid for every likelihood in the registry.
DATA = [0.5, 1.0, 1.5]

#: Counts, for the one discrete likelihood.
COUNTS = [1, 2, 3]

#: The known-parameter kwargs each likelihood needs.
LIKELIHOOD_KWARGS = {
    "poisson": {"scale": 1.0},
    "gamma": {"shape": 2.0},
    "inverse gamma": {"shape": 2.0},
    "laplace": {"mean": 0.0},
    "normal": {"mean": 0.0},
    "levy": {"location": 0.0},
    "weibull": {"rho": 1.0},
    "burrxii": {"known_shape": 1.0},
    "pareto": {"scale": 0.1},
    "dagum": {"r": 1.0, "s": 1.0},
    "gompertz": {"scale": 1.0},
    "rayleigh": {},
    "maxwell-boltzmann": {},
    "halfnormal": {},
}

ALL_LIKELIHOODS = sorted(LIKELIHOOD_REGISTRY)


def _call(name, func):
    data = COUNTS if name == "poisson" else DATA
    return func(data, **LIKELIHOOD_KWARGS[name])


def test_kwargs_table_covers_the_registry():
    """Guard against a new likelihood being added without test coverage."""
    assert set(LIKELIHOOD_KWARGS) == set(LIKELIHOOD_REGISTRY)


@pytest.mark.parametrize("name", ALL_LIKELIHOODS)
def test_registry_entry_is_a_ready_c_bereit_triple(name):
    entry = LIKELIHOOD_REGISTRY[name]

    assert len(entry) == 3
    assert all(callable(f) for f in entry)


@pytest.mark.parametrize("name", ALL_LIKELIHOODS)
def test_ready_returns_scalar_statistics(name):
    ready, _, _ = LIKELIHOOD_REGISTRY[name]
    stats = _call(name, ready)

    assert set(stats) == {"a", "b", "log_c"}
    for key, value in stats.items():
        assert np.isscalar(value) or np.ndim(value) == 0, f"{key} is not a scalar"
        assert np.isfinite(float(value)), f"{key} is not finite"


@pytest.mark.parametrize("name", ALL_LIKELIHOODS)
def test_bereit_returns_per_element_statistics(name):
    _, _, bereit = LIKELIHOOD_REGISTRY[name]
    stats = _call(name, bereit)
    n = len(COUNTS if name == "poisson" else DATA)

    assert set(stats) == {"a", "b", "log_c"}
    for key, value in stats.items():
        assert np.shape(np.asarray(value)) == (n,), f"{key} is not length-{n}"
        assert np.all(np.isfinite(value)), f"{key} is not finite"


@pytest.mark.parametrize("name", ALL_LIKELIHOODS)
def test_bereit_aggregates_to_ready(name):
    """The sufficient statistics are additive over independent observations.

    This is the invariant that lets ``post_predictive`` use the per-element
    form and ``evidence`` use the aggregated form and still agree.
    """
    ready, _, bereit = LIKELIHOOD_REGISTRY[name]
    aggregated = _call(name, ready)
    per_element = _call(name, bereit)

    for key in ("a", "b", "log_c"):
        assert np.sum(per_element[key]) == pytest.approx(
            float(aggregated[key]), rel=1e-12
        ), f"sum of per-element {key} does not match the aggregate"


@pytest.mark.parametrize("name", ALL_LIKELIHOODS)
def test_c_returns_a_sympy_expression(name):
    _, c_func, _ = LIKELIHOOD_REGISTRY[name]

    assert isinstance(c_func(), sp.Expr)


@pytest.mark.parametrize("name", ALL_LIKELIHOODS)
def test_empty_data_is_rejected(name):
    ready, _, _ = LIKELIHOOD_REGISTRY[name]

    with pytest.raises(ValueError):
        ready([], **LIKELIHOOD_KWARGS[name])


@pytest.mark.parametrize("name", ALL_LIKELIHOODS)
def test_accepts_list_array_and_series_alike(name):
    """The three accepted input containers must give identical statistics."""
    pd = pytest.importorskip("pandas")
    ready, _, _ = LIKELIHOOD_REGISTRY[name]
    data = COUNTS if name == "poisson" else DATA
    kwargs = LIKELIHOOD_KWARGS[name]

    from_list = ready(list(data), **kwargs)
    from_array = ready(np.asarray(data, dtype=float), **kwargs)
    from_series = ready(pd.Series(data, dtype=float), **kwargs)

    for key in ("a", "b", "log_c"):
        assert float(from_array[key]) == pytest.approx(float(from_list[key]))
        assert float(from_series[key]) == pytest.approx(float(from_list[key]))
