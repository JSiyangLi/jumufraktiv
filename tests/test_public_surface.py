"""The package must deliver what its interface advertises.

Two unrelated mechanisms share one theme here. The posterior predictive
ignored the known parameters its posterior was built with, and an installation
extra advertised a capability that no code path provided.
"""

import numpy as np
import pytest

from jumufraktiv.MGFDerivative_class import MGFDerivative

#: Known parameters, one entry per likelihood. Eleven of the fourteen take at
#: least one; the three with an empty entry need none. That split is not
#: incidental -- those three, plus Poisson, whose `scale` has a default, are
#: exactly the four for which the posterior predictive did not raise.
KNOWN = {
    "poisson": {"scale": 1.0},
    "gamma": {"shape": 2.0},
    "inverse gamma": {"shape": 2.0},
    "laplace": {"mean": 0.0},
    "normal": {"mean": 0.0},
    "levy": {"location": 0.0},
    "weibull": {"rho": 2.0},
    "burrxii": {"known_shape": 1.5},
    "pareto": {"scale": 0.1},
    "dagum": {"r": 1.5, "s": 1.0},
    "gompertz": {"scale": 1.0},
    "rayleigh": {},
    "maxwell-boltzmann": {},
    "halfnormal": {},
}


def _data(name):
    return [1, 2, 3] if name == "poisson" else [1.0, 2.0, 3.0]


def _new(name):
    return [1] if name == "poisson" else [1.0]


def _posterior(gamma_prior, name, **extra):
    return MGFDerivative(
        gamma_prior, data=_data(name), likelihood=name, **KNOWN[name], **extra
    )


# ==========================================================================
# The posterior predictive must work for every likelihood
# ==========================================================================
@pytest.mark.parametrize("name", sorted(KNOWN))
def test_posterior_predictive_works_for_every_likelihood(gamma_prior, name):
    """A parameter supplied at construction must not have to be supplied again.

    `post_predictive` forwarded only the caller's keyword arguments to the
    likelihood's statistics functions, which take the known parameters as
    *required* arguments. So ten of the fourteen raised

        TypeError: bereitNormal() missing 1 required positional argument: 'mean'

    for a parameter already stored on the object. The four that worked --
    halfnormal, maxwell-boltzmann, poisson, rayleigh -- are exactly the four
    with no *required* known parameter, which is why this was never seen:
    every predictive test in the suite uses Poisson or the Gamma-conjugate
    reference. Poisson did not raise, but it did not work either; see
    `test_poisson_predictive_uses_the_scale_it_was_constructed_with`.
    """
    post = _posterior(gamma_prior, name)

    result = post.post_predictive(_new(name))

    assert np.all(np.isfinite(np.asarray(result, dtype=float)))


@pytest.mark.parametrize("name", sorted(KNOWN))
def test_the_joint_predictive_works_for_every_likelihood(gamma_prior, name):
    """The `individual=False` route goes through a different function.

    `individual=True`, the default, aggregates per observation and calls
    `bereit_func`; `individual=False` treats the new observations as one block
    and calls `ready_func`. Both take the known parameters as required
    arguments, and both were reached with only the caller's keywords, so both
    had to be fixed -- and the two are separate functions whose signatures
    could in principle disagree about what they accept.
    """
    post = _posterior(gamma_prior, name)

    result = post.post_predictive(_data(name), individual=False)

    assert np.all(np.isfinite(np.asarray(result, dtype=float)))


@pytest.mark.parametrize("name", sorted(KNOWN))
def test_predictive_needs_no_repetition_of_the_known_parameters(gamma_prior, name):
    """Passing them again must be permitted, and must change nothing.

    The workaround before the fix was to repeat the parameters at the call
    site. That still has to work, or the fix would break existing code.
    """
    post = _posterior(gamma_prior, name)

    without = np.asarray(post.post_predictive(_new(name)), dtype=float)
    repeated = np.asarray(post.post_predictive(_new(name), **KNOWN[name]), dtype=float)

    assert without == pytest.approx(repeated)


def test_poisson_predictive_uses_the_scale_it_was_constructed_with(gamma_prior):
    """The one likelihood where the defect was silent rather than loud.

    Poisson is the only one of the fourteen whose known parameter carries a
    default (`scale=1.0`). So where the other thirteen raised `TypeError`,
    Poisson quietly fell back to that default and returned a number computed
    against a scale the user never asked for.

    Poisson is also the likelihood every posterior-predictive test in the
    suite used, and `conftest.POISSON_SCALE` is 1.0 -- precisely the default,
    where the wrong answer and the right answer coincide. The suite was
    therefore testing the single likelihood that fails silently, at the single
    value where the failure is invisible.

    Measured at `scale=5.0`, data `[1, 2, 3]`, new observation `2`, against
    this fixture's Gamma(2, 3) prior: the log predictive density is
    -1.4295733328 correctly and was -2.7378967900 before the fix. That is an
    error of 1.308 nats, so the predictive density came out too small by a
    factor of about 3.7.
    """

    def poisson_at(scale):
        return MGFDerivative(
            gamma_prior, data=[1, 2, 3], likelihood="poisson", scale=scale
        )

    at_one, at_five = poisson_at(1.0), poisson_at(5.0)

    honoured = float(np.ravel(at_five.post_predictive([2]))[0])
    # What the old code produced: the caller passed nothing, so the statistics
    # functions fell through to their own default of 1.0.
    defaulted = float(np.ravel(at_five.post_predictive([2], scale=1.0))[0])

    assert honoured != pytest.approx(defaulted, rel=1e-6)

    # And the fix must not have simply pinned everything to some other
    # constant: at scale=1.0 the stored value *is* the default, so the two
    # routes have to agree exactly.
    assert float(np.ravel(at_one.post_predictive([2]))[0]) == pytest.approx(
        float(np.ravel(at_one.post_predictive([2], scale=1.0))[0])
    )


def test_a_caller_can_override_a_known_parameter(gamma_prior):
    """Overrides are honoured rather than silently ignored.

    Merging the stored parameters in could have been done the other way round,
    with the stored values winning. That would quietly discard a caller's
    argument -- the same class of defect as the one being fixed. A new
    observation from a different Weibull shape is a legitimate request.
    """
    post = _posterior(gamma_prior, "weibull")

    stored = float(np.ravel(post.post_predictive([1.0]))[0])
    overridden = float(np.ravel(post.post_predictive([1.0], rho=3.0))[0])

    assert stored != overridden


@pytest.mark.parametrize("name", ["normal", "weibull", "dagum"])
def test_the_predictive_is_a_density_over_the_new_observation(gamma_prior, name):
    """Sanity check that the values mean something, not merely that they exist.

    Reachability was the defect, so a test asserting only "it returns" would
    pass against a version that returned nonsense. The predictive is a log
    density here, so it must decrease as the new observation moves into the
    tail.
    """
    post = _posterior(gamma_prior, name)

    near = float(np.ravel(post.post_predictive([1.0]))[0])
    far = float(np.ravel(post.post_predictive([50.0]))[0])

    assert far < near


# ==========================================================================
# No advertised extra may be inert
# ==========================================================================
def test_no_module_imports_torch():
    """The `[torch]` extra promised a Pareto backend that nothing reached.

    `pareto_mgf_torch` was defined, documented, and referenced by no code
    path: not by `pareto_factory`, not by any other module, not by any test or
    notebook. Installing `jumufraktiv[torch]` bought a heavyweight dependency
    and nothing else.

    The differentiable surface is JAX throughout -- fourteen modules import
    it, `mitMGFprior` has dedicated `mgf_jax` / `cgf_jax` / `imgf_jax` /
    `logimgf_jax` slots, and Pareto already fills all four. So the capability
    the extra advertised was present already, in the framework the rest of the
    package uses.
    """
    import re
    from pathlib import Path

    # Matches a real import, not the word: `MGFdictionary/__init__.py` still
    # records the eager-import incident as history, and that note should
    # survive.
    imports_torch = re.compile(r"^\s*(import torch|from torch[. ])", re.MULTILINE)

    root = Path(__file__).resolve().parent.parent / "jumufraktiv"
    offenders = [
        str(path.relative_to(root.parent))
        for path in root.rglob("*.py")
        if imports_torch.search(path.read_text())
    ]

    assert offenders == [], f"torch is still imported by {offenders}"


@pytest.mark.parametrize("name", sorted(KNOWN))
def test_every_likelihood_still_reaches_the_jax_slots(gamma_prior, name):
    """Removing the torch surface must not disturb the JAX one.

    The Pareto prior fills `mgf_jax`, `cgf_jax`, `imgf_jax` and `logimgf_jax`,
    and those are what the differentiable paths actually use.
    """
    post = _posterior(gamma_prior, name)

    assert np.isfinite(post.evidence())


# ==========================================================================
# What the package exports
# ==========================================================================
def test_every_exported_name_exists_and_is_reachable():
    """`__all__` must not promise a name the package does not bind."""
    import jumufraktiv

    for name in jumufraktiv.__all__:
        assert hasattr(jumufraktiv, name), f"__all__ names '{name}', which is absent"


def test_the_documented_workflows_reach_the_top_level():
    """Each name here is one a documented workflow needs.

    Registering a prior and discovering what is registered are described in
    `CLAUDE.md` as the way to extend the package, and both required a deep
    import into `jumufraktiv.registry` before PR 12. A documented workflow that
    only works through a private path is not really public.
    """
    import jumufraktiv

    expected = {
        "MGFDerivative",  # build a posterior
        "mitMGFprior",  # build a prior
        "mgfDerivative",  # one derivative, no posterior
        "mgfDerivative_integer",
        "mgfDerivative_fractional",
        "register_prior",  # add a prior
        "make_prior_spec",
        "list_priors",  # discover priors
        "failed_prior_modules",  # find out why one is missing
        "__version__",
    }

    assert expected <= set(jumufraktiv.__all__)


def test_the_canonical_symbols_are_importable_from_their_own_module():
    """They are deliberately not re-exported at package level.

    `t` as a bare top-level name would collide with the `t` a caller is
    overwhelmingly likely to have bound already. The module path is the point:
    the symbols must be imported from here rather than redefined, since two
    symbols that print alike but were constructed separately do not substitute
    for each other.
    """
    from jumufraktiv.symbols import param, q, r, t, theta, u

    assert {s.name for s in (t, theta, r, u, q)} == {"t", "theta", "r", "u", "q"}
    assert param("alpha").is_positive


def test_the_former_package_name_no_longer_hijacks_sys_modules():
    """`sys.modules["mgf2post"] = ...` claimed a name this package does not own.

    Importing `jumufraktiv` installed that alias process-wide, so `import
    mgf2post` returned this package even where a genuinely different
    distribution of that name was installed. Nothing in the package or the
    suite used it.
    """
    import sys

    import jumufraktiv  # noqa: F401  -- the import is what used to set the alias

    assert "mgf2post" not in sys.modules

    with pytest.raises(ImportError):
        import mgf2post  # noqa: F401
