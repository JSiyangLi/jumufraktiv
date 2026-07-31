"""The package must deliver what its interface advertises.

Two unrelated mechanisms, one theme. A method that cannot be called for most
likelihoods, and an installation extra that does nothing.
"""

import numpy as np
import pytest

from jumufraktiv.MGFDerivative_class import MGFDerivative

#: Known parameters, one entry per likelihood. Ten of the fourteen take one;
#: those ten are exactly the ones for which the posterior predictive used to
#: raise, so the split matters and is not incidental.
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
    with no required known parameter, which is why this was never seen: every
    predictive test in the suite uses Poisson or the Gamma-conjugate
    reference, and Poisson's one parameter has a default.
    """
    post = _posterior(gamma_prior, name)

    result = post.post_predictive(_new(name))

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

    assert np.isfinite(post.evidence()[0])
