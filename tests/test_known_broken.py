"""Executable records of defects that are known, reproduced, and scheduled.

Every test here is marked ``xfail(strict=True)``. That direction is deliberate:

* while the defect exists the suite stays green, so these do not block unrelated
  work;
* the moment a later PR fixes one, the test XPASSes and *fails* the build,
  forcing the fix to be acknowledged here and in ``CLAUDE.md``.

So this file is the to-do list for waves 1-2 of the audit, and it cannot drift
out of sync with the code. Each test names the PR that owns the repair.

The bodies assert the *correct* behaviour, not the broken behaviour, so when the
fix lands the assertion is already the right one.
"""

import importlib.util
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import sympy as sp

from jumufraktiv import registry
from jumufraktiv.derivativeDispatch import mgfDerivative
from jumufraktiv.MGFDerivative_class import MGFDerivative

HAS_TORCH = importlib.util.find_spec("torch") is not None


# ==========================================================================
# PR 3 — import and registry integrity
# ==========================================================================
@pytest.mark.xfail(
    strict=True,
    reason="PR 3: mitMGFprior.from_registry never calls registry.initialize(), "
    "so it only works if some other registry function ran first",
)
def test_from_registry_initialises_registry():
    """``from_registry`` must work as the first registry call in a fresh process."""
    script = textwrap.dedent(
        """
        import warnings
        warnings.simplefilter("ignore")
        from jumufraktiv.mitMGFprior_class import mitMGFprior
        mitMGFprior.from_registry("gamma", params={"alpha": 2.0, "beta": 3.0})
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


@pytest.mark.skipif(
    HAS_TORCH, reason="the defect only manifests when the torch extra is absent"
)
@pytest.mark.xfail(
    strict=True,
    reason="PR 3: MGFdictionary/paretoMGF.py imports torch eagerly and "
    "MGFdictionary/__init__.py aborts its discovery loop on the first failure, "
    "so a missing optional extra silently removes two unrelated priors",
)
def test_optional_backend_does_not_break_prior_discovery():
    """A missing optional dependency must not cost unrelated priors."""
    assert set(registry.list_priors()) >= {"gamma", "heaviside", "pareto", "uniform"}


@pytest.mark.xfail(
    strict=True,
    reason="PR 3: derivativeDispatch.py imports symbolic_fractionalDeriv without "
    "the package prefix, which never resolves in an installed package",
)
def test_symbolic_fractional_import_resolves(gamma_prior):
    """The symbolic fractional backend must be importable."""
    mgfDerivative(1.5, gamma_prior, method="symbolic", t=None)


# ==========================================================================
# PR 4 — the fractional-order path
# ==========================================================================
@pytest.mark.xfail(
    strict=True,
    reason="PR 4: MGFDerivative._build_derivative calls mgfDerivative(t=None), "
    "but the fractional branch requires t, so any non-integer order raises "
    "at construction",
)
def test_fractional_order_posterior_can_be_constructed(gamma_prior):
    """A non-integer sufficient statistic must not break construction.

    ``normal``, ``halfnormal`` and ``maxwell-boltzmann`` all produce a
    fractional ``a`` whenever the sample size is odd, so this is an ordinary
    use of the package, not an exotic one.
    """
    post = MGFDerivative(
        gamma_prior, data=[0.5, 1.0, 1.5], likelihood="halfnormal", method="auto"
    )

    log_ev, _ = post.evidence()
    assert np.isfinite(log_ev)


@pytest.mark.xfail(
    strict=True,
    reason="PR 4: the array-order branch of mgfDerivative coerces each order "
    "with int(o), silently truncating fractional orders to integers",
)
def test_array_order_does_not_truncate_fractional_orders(gamma_prior):
    """Vectorising over order must agree with looping over scalar orders.

    This path is not hypothetical: ``post_predictive`` always passes an array
    of orders, so a fractional ``a`` yields a silently wrong predictive.
    """
    orders = np.array([1.0, 1.5])
    batch_log, _ = mgfDerivative(orders, gamma_prior, method="auto", t=-1.0, log=True)
    scalar_log = [
        mgfDerivative(float(o), gamma_prior, method="auto", t=-1.0, log=True)[0]
        for o in orders
    ]

    assert batch_log == pytest.approx(np.array(scalar_log), rel=1e-8)


# ==========================================================================
# PR 5 — symbolic-path correctness
# ==========================================================================
@pytest.mark.xfail(
    strict=True,
    reason="PR 5: integerDeriv_symbolic rejects any order that is not a Python "
    "int, so the symbolic-order row of the backend matrix cannot be reached — "
    "the dispatcher warns that it will return an analytic continuation and "
    "then raises TypeError instead",
)
def test_symbolic_order_returns_expression(gamma_prior):
    """A symbolic order must yield an unevaluated expression, per the matrix."""
    n = sp.Symbol("n", positive=True, integer=True)
    result = mgfDerivative(n, gamma_prior, t=None)

    assert isinstance(result, sp.Expr)


# ==========================================================================
# PR 6 — numerical robustness
# ==========================================================================
@pytest.mark.xfail(
    strict=True,
    reason="PR 6: post_quantile brackets from a lower bound of 1e-6, where the "
    "incomplete-MGF derivative underflows and its computed sign flips negative, "
    "tripping the guard in post_cdf. This makes post_quantile, post_interval "
    "and post_sample unusable for every prior",
)
@pytest.mark.parametrize("p", [0.025, 0.5, 0.975])
def test_post_quantile_inverts_the_cdf(poisson_posterior, p):
    """The quantile function must invert the CDF."""
    q = poisson_posterior.post_quantile(p)

    assert poisson_posterior.post_cdf(q, log=False) == pytest.approx(p, abs=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason="PR 6: post_sample depends on post_quantile, which cannot bracket",
)
def test_post_sample_returns_requested_size(poisson_posterior):
    draws = poisson_posterior.post_sample(16)

    assert np.shape(draws) == (16,)
    assert np.all(draws > 0)


def test_post_cdf_is_zero_at_the_origin(poisson_posterior):
    """At u = 0 the CDF correctly vanishes. This one already works."""
    assert poisson_posterior.post_cdf(0.0, log=False) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.xfail(
    strict=True,
    reason="PR 6: post_cdf has no domain validation on u. Below zero it either "
    "recurses until RecursionError (u = -1e-9) or returns a log-CDF above zero, "
    "i.e. a probability greater than one (u = -0.5), for a parameter that is "
    "constrained positive",
)
@pytest.mark.parametrize("u", [-0.5, -1e-9])
def test_post_cdf_is_zero_below_the_support(poisson_posterior, u):
    """theta is strictly positive, so the CDF must vanish below zero."""
    assert poisson_posterior.post_cdf(u, log=False) == pytest.approx(0.0, abs=1e-12)


# ==========================================================================
# PR 12 — public API surface
# ==========================================================================
@pytest.mark.xfail(
    strict=True,
    reason="PR 12: the log principle says the log argument alone decides the "
    "return shape, but post_raw_moment returns a bare scalar while "
    "post_central_moment returns (log_abs, sign) for the same log=True",
)
def test_moment_methods_share_a_return_convention(poisson_posterior):
    raw = poisson_posterior.post_raw_moment(2)
    central = poisson_posterior.post_central_moment(2)

    assert type(raw) is type(central)


@pytest.mark.xfail(
    strict=True,
    reason="PR 12: post_sample calls the unseeded legacy np.random.rand and "
    "takes no rng argument, so results are not reproducible",
)
def test_post_sample_is_reproducible(poisson_posterior):
    """Two draws under the same seed must agree."""
    first = poisson_posterior.post_sample(8, rng=np.random.default_rng(0))
    second = poisson_posterior.post_sample(8, rng=np.random.default_rng(0))

    assert np.array_equal(first, second)
