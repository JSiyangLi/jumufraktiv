"""Tests for keyword-argument routing on the ``MGFDerivative`` constructor.

Extra keyword arguments are split between the likelihood's ``ready`` function
and the derivative layer. The split used to match against the *union* of every
likelihood's parameter names and forward everything else to the derivative
layer, where ``**kwargs`` absorbed it — so two ordinary mistakes produced a
confidently wrong number instead of an error.

These tests pin both mistakes as errors, and pin that legitimate arguments still
reach the layer they belong to.
"""

import inspect

import pytest
from conftest import ALPHA, BETA, POISSON_DATA, poisson_log_evidence

from jumufraktiv.MGFDerivative_class import (
    _RESERVED_DERIVATIVE_KWARGS,
    DERIVATIVE_KWARGS,
    LIKELIHOOD_REGISTRY,
    MGFDerivative,
    _likelihood_kwargs,
)


# ==========================================================================
# The two silent failures
# ==========================================================================
def test_misspelled_likelihood_parameter_raises(gamma_prior):
    """A typo must raise rather than silently fall back to the default.

    ``scal=2.0`` used to reach the derivative layer, be absorbed, and leave the
    likelihood on its default ``scale=1.0`` — an evidence wrong by 0.92 nats
    with no error and no warning.
    """
    with pytest.raises(TypeError, match="scal"):
        MGFDerivative(gamma_prior, data=POISSON_DATA, likelihood="poisson", scal=2.0)


def test_misspelling_suggests_the_intended_name(gamma_prior):
    """The error should be actionable, not merely correct."""
    with pytest.raises(TypeError, match="did you mean 'scale'"):
        MGFDerivative(gamma_prior, data=POISSON_DATA, likelihood="poisson", scal=2.0)


def test_parameter_for_a_different_likelihood_raises(gamma_prior):
    """A name valid elsewhere is still invalid here.

    ``rho`` belongs to Weibull. Passing it to a Poisson used to be forwarded
    into ``readyPoisson`` and swallowed by its ``**kwargs``.
    """
    with pytest.raises(TypeError, match="rho"):
        MGFDerivative(
            gamma_prior,
            data=POISSON_DATA,
            likelihood="poisson",
            scale=1.0,
            rho=99.0,
        )


def test_error_names_what_the_likelihood_accepts(gamma_prior):
    with pytest.raises(TypeError) as excinfo:
        MGFDerivative(
            gamma_prior, data=POISSON_DATA, likelihood="poisson", nonsense=1.0
        )

    message = str(excinfo.value)
    assert "'poisson' likelihood accepts" in message
    assert "scale" in message


# ==========================================================================
# Legitimate arguments still work
# ==========================================================================
def test_likelihood_parameter_reaches_the_likelihood(gamma_prior):
    """A correctly spelled parameter must change the answer."""
    default = MGFDerivative(
        gamma_prior, data=POISSON_DATA, likelihood="poisson"
    ).evidence()[0]
    scaled = MGFDerivative(
        gamma_prior, data=POISSON_DATA, likelihood="poisson", scale=2.0
    ).evidence()[0]

    assert default != pytest.approx(scaled)
    assert scaled == pytest.approx(
        poisson_log_evidence(POISSON_DATA, scale=2.0, alpha=ALPHA, beta=BETA),
        rel=1e-10,
    )


@pytest.mark.parametrize(
    "option", [{"int_tol": 1e-10}, {"integer_method": "bell"}, {"use_tan": True}]
)
def test_derivative_options_are_accepted(gamma_prior, option):
    post = MGFDerivative(
        gamma_prior, data=POISSON_DATA, likelihood="poisson", scale=1.0, **option
    )

    assert post.evidence()[0] == pytest.approx(
        poisson_log_evidence(POISSON_DATA), rel=1e-8
    )


def test_no_extra_kwargs_is_fine(gamma_prior):
    post = MGFDerivative(gamma_prior, data=POISSON_DATA, likelihood="poisson")

    assert post.evidence()[0] == pytest.approx(
        poisson_log_evidence(POISSON_DATA), rel=1e-10
    )


# ==========================================================================
# The routing table itself
# ==========================================================================
@pytest.mark.parametrize("name", sorted(LIKELIHOOD_REGISTRY))
def test_accepted_names_come_from_the_signature(name):
    """Each likelihood is checked against its own parameters, not a global list."""
    ready, _, _ = LIKELIHOOD_REGISTRY[name]
    accepted = _likelihood_kwargs(ready)
    signature = inspect.signature(ready).parameters

    for parameter in accepted:
        assert parameter in signature
    assert "data" not in accepted
    # A trailing **kwargs must not widen what is accepted; absorbing unknown
    # names is exactly the defect being fixed.
    assert not any(
        p.kind is p.VAR_KEYWORD and p.name in accepted for p in signature.values()
    )


@pytest.mark.parametrize("name", sorted(LIKELIHOOD_REGISTRY))
def test_no_likelihood_parameter_collides_with_a_derivative_option(name):
    """A name in both groups would be routed ambiguously."""
    ready, _, _ = LIKELIHOOD_REGISTRY[name]

    assert not (_likelihood_kwargs(ready) & DERIVATIVE_KWARGS)


def test_no_derivative_option_collides_with_a_reserved_argument():
    """A user-settable option must not be one the class supplies itself.

    ``complete`` was briefly in the accepted set and produced
    "got multiple values for keyword argument 'complete'", because
    ``_build_derivative`` passes it explicitly — and must, since it is ``True``
    for the evidence and ``False`` for the incomplete-MGF path.
    """
    assert not (DERIVATIVE_KWARGS & _RESERVED_DERIVATIVE_KWARGS)


def test_reserved_arguments_are_rejected_from_callers(gamma_prior):
    """Passing a reserved argument should be an error, not a duplicate-kwarg crash."""
    with pytest.raises(TypeError, match="complete"):
        MGFDerivative(
            gamma_prior,
            data=POISSON_DATA,
            likelihood="poisson",
            scale=1.0,
            complete=False,
        )


def test_every_likelihood_rejects_an_unknown_argument(gamma_prior):
    """The guard applies to all 14 likelihoods, not just the one under test."""
    from test_likelihood_stats import COUNTS, DATA, LIKELIHOOD_KWARGS

    for name in sorted(LIKELIHOOD_REGISTRY):
        data = COUNTS if name == "poisson" else DATA
        with pytest.raises(TypeError, match="definitely_not_a_parameter"):
            MGFDerivative(
                gamma_prior,
                data=data,
                likelihood=name,
                definitely_not_a_parameter=1.0,
                **LIKELIHOOD_KWARGS[name],
            )
