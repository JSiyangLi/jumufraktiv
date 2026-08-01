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
import warnings

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
    # Accepted means "does not raise", which is a different question from
    # "changes the answer". Two of these three are read by no route this
    # posterior takes -- it has an integer order and goes to the expectation
    # integral -- so PR 12 makes them warn. The warning is asserted where it
    # belongs, in `test_known_broken.py`; here it would only mask the TypeError
    # this test exists to rule out, which still propagates.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
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


# ==========================================================================
# The constructor and the dispatch functions must agree
# ==========================================================================
# `MGFDerivative` rejects an unknown keyword argument with a `TypeError` and a
# "did you mean" suggestion. `mgfDerivative` and `mgfDerivative_fractional` --
# also exported at package level, also part of the public surface -- used to
# accept anything and filter it away just before calling the backend.
#
# So the same name behaved two ways depending on which door the caller used:
#
#   name         constructor   mgfDerivative*   reaches any backend?
#   epsrel       TypeError     accepted           no
#   initial_L    TypeError     accepted           no
#   epsrell      TypeError     accepted           no   (a misspelling)
#
# The middle column is the dangerous one. A caller tightening `epsrel` got the
# default and no indication of it, which is precisely the failure mode PR 8
# removed everywhere else.
from jumufraktiv.derivativeDispatch import (  # noqa: E402
    DERIVATIVE_OPTIONS,
    mgfDerivative,
    mgfDerivative_fractional,
)

#: Names that reach no backend: the tuning parameters of the two quadrature
#: implementations PR 6b and PR 8 deleted, plus a plain typo.
DEAD_OPTIONS = [
    "epsabs",
    "epsrel",
    "limit",
    "initial_L",
    "max_L",
    "use_interpolation",
    "d_vec",
    "epsrell",
    "definitely_not_an_option",
]


@pytest.mark.parametrize("name", DEAD_OPTIONS)
def test_dead_options_are_refused_by_every_public_entry_point(gamma_prior, name):
    """One name, three doors, one answer."""
    with pytest.raises(TypeError, match=name):
        MGFDerivative(
            gamma_prior,
            data=POISSON_DATA,
            likelihood="poisson",
            scale=1.0,
            **{name: 1e-9},
        )

    with pytest.raises(TypeError, match=name):
        mgfDerivative(1.5, gamma_prior, method="scipy", t=-1.0, **{name: 1e-9})

    with pytest.raises(TypeError, match=name):
        mgfDerivative_fractional(
            1.5, gamma_prior, method="scipy", t=-1.0, **{name: 1e-9}
        )


@pytest.mark.parametrize("name", sorted(DERIVATIVE_OPTIONS))
def test_live_options_are_accepted_by_every_public_entry_point(gamma_prior, name):
    """The converse, so the guard cannot pass by refusing everything.

    A test that only checks rejections is satisfied by a function that rejects
    its own valid arguments too.
    """
    value = {
        "integer_method": "symbolic",
        "cgf_method": "auto",
        "use_tan": False,
        "dps": 30,
        "symbolic_timeout": 60.0,
    }.get(name, 1e-9)

    # The property is acceptance, so only a `TypeError` may fail this test. A
    # name most of these routes cannot read now warns, which is PR 12's repair
    # of a different defect and is asserted separately; letting it fail here
    # would make one test answer two questions and neither clearly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        MGFDerivative(
            gamma_prior,
            data=POISSON_DATA,
            likelihood="poisson",
            scale=1.0,
            **{name: value},
        )
        mgfDerivative(1.5, gamma_prior, method="scipy", t=-1.0, **{name: value})
        mgfDerivative_fractional(
            1.5, gamma_prior, method="scipy", t=-1.0, **{name: value}
        )


def test_the_constructor_does_not_keep_its_own_list_of_options():
    """The two sets must be one object, not two that happen to match today.

    They did not match: the constructor's literal set still named
    `use_interpolation` and `d_vec` long after PR 6b deleted the module that
    read them. Asserting equality would let the next divergence live until
    someone wrote a test for it; asserting identity makes divergence
    impossible to express.
    """
    assert DERIVATIVE_KWARGS is DERIVATIVE_OPTIONS


def test_the_option_list_matches_the_backends_in_both_directions():
    """Every accepted option reaches a backend, and every backend option is accepted.

    The second half is not decoration. I wrote this guard with only the first
    half and immediately broke `timeout_seconds`, which the symbolic fractional
    backend has always taken: a one-directional check is satisfied by a list
    that is too *short* as well as by one that is right, and too-short means
    refusing an argument that works.

    `symbolic_timeout` and `timeout_seconds` are different options -- the first
    bounds the Bell backend's symbolic attempt, the second the symbolic
    fractional backend's transform -- so neither is a typo for the other and
    both belong in the list.
    """
    from jumufraktiv.derivativeDispatch import BACKEND_OPTIONS, mgfDerivative_integer
    from jumufraktiv.numeric_expectation import expectationDeriv
    from jumufraktiv.numeric_fractionalDeriv_grid import fractionalDeriv_grid
    from jumufraktiv.numeric_fractionalDeriv_mpmath import (
        fractionalDeriv_numeric_mpmath,
    )
    from jumufraktiv.symbolic_fractionalDeriv import fractionalDeriv_symbolic

    #: Parameters that carry the request itself rather than tune how it is
    #: answered. A caller supplies these positionally or by name through the
    #: dispatcher's own signature, never through **kwargs.
    STRUCTURAL = {
        "order",
        "prior",
        "t",
        "u",
        "method",
        "simplify",
        "complete",
        "log",
        "return_log",
        "t_points",
        "u_points",
        "integer_method",
        "int_tol",
        "kwargs",
    }

    declared = set()
    for backend in (
        fractionalDeriv_grid,
        fractionalDeriv_numeric_mpmath,
        fractionalDeriv_symbolic,
        expectationDeriv,
        mgfDerivative_integer,
    ):
        declared |= set(inspect.signature(backend).parameters)
    tunable = declared - STRUCTURAL

    accepted_but_unreachable = sorted(set(BACKEND_OPTIONS) - tunable)
    reachable_but_refused = sorted(tunable - set(BACKEND_OPTIONS))

    assert not accepted_but_unreachable, (
        "accepted but consumed by no backend: " + ", ".join(accepted_but_unreachable)
    )
    assert not reachable_but_refused, (
        "a backend takes these but the dispatcher refuses them: "
        + ", ".join(reachable_but_refused)
    )


def test_the_route_table_agrees_with_the_backend_signatures():
    """`ROUTE_OPTIONS` decides what is announced as discarded, so it must be right.

    `BACKEND_OPTIONS` answers "does *any* backend read this?", which is what
    makes a name valid rather than a misspelling. `ROUTE_OPTIONS` answers "does
    *this* route read it?", which is what makes a value effective rather than
    discarded. A route table that overstates what a route reads would silence
    the warning for an option that is still being dropped -- the exact defect
    the warning exists to end -- so it is checked against the signatures rather
    than trusted.

    Two names are checked by hand because the signature does not carry them.
    `cgf_method` is spelled `cgf_mode` inside the Bell backend, and
    `integer_method` is consumed by the fractional kernels to choose the
    integer backend nested inside them.
    """
    from jumufraktiv.derivativeDispatch import (
        BACKEND_OPTIONS,
        ROUTE_OPTIONS,
        mgfDerivative_integer,
    )
    from jumufraktiv.numeric_expectation import expectationDeriv
    from jumufraktiv.numeric_fractionalDeriv_grid import fractionalDeriv_grid
    from jumufraktiv.numeric_fractionalDeriv_mpmath import (
        fractionalDeriv_numeric_mpmath,
    )
    from jumufraktiv.numeric_integerDeriv_Bell import integerDeriv_numeric_bell
    from jumufraktiv.numeric_integerDeriv_JAX import integerDeriv_numeric_jax
    from jumufraktiv.symbolic_fractionalDeriv import fractionalDeriv_symbolic
    from jumufraktiv.symbolic_integerDeriv import integerDeriv_symbolic

    #: The function each route ultimately reaches, and any option that function
    #: reads under a different name than the dispatcher accepts it under.
    ROUTE_TARGETS = {
        (None, "expectation"): (expectationDeriv, frozenset()),
        ("integer", "symbolic"): (integerDeriv_symbolic, frozenset()),
        ("integer", "bell"): (integerDeriv_numeric_bell, frozenset({"cgf_method"})),
        ("integer", "jax"): (integerDeriv_numeric_jax, frozenset()),
        ("fractional", "scipy"): (fractionalDeriv_grid, frozenset()),
        ("fractional", "mpmath"): (
            fractionalDeriv_numeric_mpmath,
            frozenset({"integer_method"}),
        ),
        ("fractional", "symbolic"): (fractionalDeriv_symbolic, frozenset()),
    }

    assert set(ROUTE_OPTIONS) == set(ROUTE_TARGETS), (
        "a route was added or removed without updating this test"
    )

    tunable_names = set(BACKEND_OPTIONS) | {"integer_method"}
    for route, declared_options in ROUTE_OPTIONS.items():
        target, renamed = ROUTE_TARGETS[route]
        actually_read = (
            set(inspect.signature(target).parameters) & tunable_names
        ) | renamed

        assert declared_options == actually_read, (
            f"ROUTE_OPTIONS{route} says {sorted(declared_options)} but "
            f"{target.__name__} reads {sorted(actually_read)}"
        )

    # And every tunable name must be read by at least one route, or the
    # dispatcher accepts an option nothing can honour.
    covered = set().union(*ROUTE_OPTIONS.values())
    assert tunable_names <= covered, (
        "read by no route: " + ", ".join(sorted(tunable_names - covered))
    )


def test_the_symbolic_fractional_timeout_still_works():
    """The option the one-directional guard broke, asserted on its own.

    A set-comparison test can pass while the behaviour is wrong, if the sets
    are compared with the same mistake in both. This calls the thing.
    """
    import warnings

    from jumufraktiv.derivativeDispatch import mgfDerivative_fractional

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expr = mgfDerivative_fractional(
            1.5, _GAMMA_PRIOR(), method="symbolic", t=None, timeout_seconds=30.0
        )

    assert expr is not None


def _GAMMA_PRIOR():
    from jumufraktiv import registry
    from jumufraktiv.mitMGFprior_class import mitMGFprior

    registry.initialize()
    return mitMGFprior.from_registry("gamma", params={"alpha": ALPHA, "beta": BETA})
