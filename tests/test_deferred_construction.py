"""Construction of the derivative representation, and backend resolution.

Two things were tangled together, and separating them is what this PR does.

**Which backend serves a request** is a property of the order and the requested
method alone. It was decided inline inside `mgfDerivative`, so nothing else
could ask the question without re-deriving the rules. It now lives in
`resolve_backend`, the single encoding of CLAUDE.md's backend matrix.

**When the evaluation point is needed** differs by backend. The `symbolic`
backend differentiates the prior's MGF and returns an expression in `t`, so it
can be built before any `t` is known. Every numeric backend quadratures at a
particular `t` and has nothing to return until one arrives.

`MGFDerivative._build_derivative` assumed the first case universally — it called
`mgfDerivative(..., t=None)` unconditionally — so every numeric backend raised
``ValueError: For method 'scipy', t must be provided.`` at construction. That
took out all six likelihoods with a fractional `a`, and `bell` and `jax` for
integer orders too: two thirds of the backend matrix, none of it reachable
through the public class.
"""

import functools
import warnings

import numpy as np
import pytest
import sympy as sp
from conftest import gamma_mgf_derivative_log

from jumufraktiv.derivativeDispatch import mgfDerivative, resolve_backend
from jumufraktiv.MGFDerivative_class import MGFDerivative

#: Known parameters per likelihood, chosen so that `a` is fractional wherever
#: the likelihood can produce a fractional `a` at all.
KNOWN = {
    "poisson": {"scale": 1.0},
    "gamma": {"shape": 2.5},
    "inverse gamma": {"shape": 1.5},
    "laplace": {"mean": 0.0},
    "normal": {"mean": 0.0},
    "levy": {"location": 0.0},
    "weibull": {"rho": 2.0},
    "burrxii": {"known_shape": 1.5},
    "pareto": {"scale": 0.5},
    "dagum": {"r": 1.5, "s": 1.0},
    "gompertz": {"scale": 1.0},
    "rayleigh": {},
    "maxwell-boltzmann": {},
    "halfnormal": {},
}

DATA = [1.0, 2.0, 3.0]
COUNTS = [1, 2, 3]

#: The six likelihoods whose `a` is not an integer at n = 3. Every one of them
#: raised at construction before this PR.
FRACTIONAL = [
    "gamma",  # a = sum(shape_i) = 7.5
    "halfnormal",  # a = n/2 = 1.5
    "inverse gamma",  # a = sum(shape_i) = 4.5
    "levy",  # a = n/2 = 1.5
    "maxwell-boltzmann",  # a = 1.5n = 4.5
    "normal",  # a = n/2 = 1.5
]


def _data_for(name):
    return COUNTS if name == "poisson" else DATA


#: Constructed posteriors, keyed on the arguments that produced them.
#:
#: Construction runs the quadrature, so building the same posterior afresh in
#: every test is the dominant cost of this file: profiling found roughly twenty
#: separate `halfnormal` builds, and the two `mpmath` ones cost 21 s and 16 s
#: each. Nothing here mutates a posterior -- `update` returns a new object and
#: everything else is a query -- so they are safe to share.
_POSTERIORS = {}


def _build(name, **kwargs):
    key = (name, tuple(sorted(kwargs.items())))
    if key not in _POSTERIORS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _POSTERIORS[key] = MGFDerivative(
                gamma_prior_spec(),
                data=_data_for(name),
                likelihood=name,
                **KNOWN[name],
                **kwargs,
            )
    return _POSTERIORS[key]


@functools.cache
def gamma_prior_spec():
    from jumufraktiv.mitMGFprior_class import mitMGFprior

    return mitMGFprior.from_registry("gamma", params={"alpha": 2.0, "beta": 3.0})


# ==========================================================================
# resolve_backend: the backend matrix, in one place
# ==========================================================================
class TestResolveBackend:
    @pytest.mark.parametrize(
        ("order", "expected"),
        [
            (0, "integer"),
            (2, "integer"),
            (2.0, "integer"),
            (-3, "integer"),
            (1.5, "fractional"),
            (0.5, "fractional"),
            (7.5, "fractional"),
        ],
    )
    def test_order_type(self, order, expected):
        assert resolve_backend(order, "auto")[0] == expected

    def test_symbolic_order_is_classified_symbolic(self):
        assert resolve_backend(sp.Symbol("a"), "auto")[0] == "symbolic"

    def test_integer_test_uses_int_tol_not_exact_equality(self):
        """An order within `int_tol` of an integer counts as that integer."""
        assert resolve_backend(2 + 1e-15, "auto")[0] == "integer"
        assert resolve_backend(2 + 1e-6, "auto")[0] == "fractional"

    def test_int_tol_is_honoured(self):
        assert resolve_backend(2.001, "auto", int_tol=1e-2)[0] == "integer"

    @pytest.mark.parametrize(
        ("order", "method"),
        [(2, "symbolic"), (1.5, "scipy"), (sp.Symbol("a"), "symbolic")],
    )
    def test_auto_resolution(self, order, method):
        assert resolve_backend(order, "auto")[1] == method

    @pytest.mark.parametrize("method", ["symbolic", "bell", "jax"])
    def test_integer_row_accepts_its_methods(self, method):
        assert resolve_backend(2, method)[1] == method

    @pytest.mark.parametrize("method", ["scipy", "mpmath", "symbolic"])
    def test_fractional_row_accepts_its_methods(self, method):
        assert resolve_backend(1.5, method)[1] == method

    @pytest.mark.parametrize(
        ("order", "method"),
        [(2, "scipy"), (2, "mpmath"), (1.5, "nonsense"), (sp.Symbol("a"), "scipy")],
    )
    def test_invalid_combinations_are_rejected(self, order, method):
        with pytest.raises(ValueError, match="Invalid method"):
            resolve_backend(order, method)

    def test_invalid_integer_method_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid integer_method"):
            resolve_backend(1.5, "scipy", integer_method="nonsense")

    @pytest.mark.parametrize("method", ["bell", "jax"])
    def test_integer_backend_requested_for_a_fractional_order_is_reinterpreted(
        self, method
    ):
        """Neither can take a fractional derivative, so it becomes integer_method.

        This is long-standing behaviour and is preserved deliberately; what
        changes is that it now warns rather than printing.
        """
        with pytest.warns(UserWarning, match="cannot take a fractional derivative"):
            order_type, resolved, integer_method = resolve_backend(1.5, method)

        assert (order_type, resolved, integer_method) == ("fractional", "scipy", method)

    def test_reinterpretation_does_not_fire_for_an_integer_order(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert resolve_backend(2, "bell")[1] == "bell"

    @pytest.mark.parametrize(
        ("order", "method"),
        [
            (2, "symbolic"),
            (2, "bell"),
            (2, "jax"),
            (1.5, "scipy"),
            (1.5, "mpmath"),
            (1.5, "symbolic"),
            (2, "auto"),
            (1.5, "auto"),
        ],
    )
    @pytest.mark.parametrize("spelling", [str.upper, str.title, str.lower])
    def test_returned_names_are_canonical_lowercase(self, order, method, spelling):
        """Backend names match case-insensitively, so they must return canonical.

        Every backend has always accepted any casing — each lowercases on
        entry. Callers compare the *returned* name against a lowercase literal,
        so returning the caller's spelling silently sends them down the wrong
        branch: `method='SYMBOLIC'` produced the correct evidence but a numeric
        representation, which disables `update` and every other symbolic path.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, resolved, integer_method = resolve_backend(order, spelling(method))

        assert resolved == resolved.lower()
        assert integer_method == integer_method.lower()
        assert resolved == resolve_backend(order, method.lower())[1]

    @pytest.mark.parametrize("spelling", ["symbolic", "jax", "bell"])
    def test_integer_method_is_canonicalised_too(self, spelling):
        assert resolve_backend(1.5, "scipy", integer_method=spelling.upper())[2] == (
            spelling
        )


# ==========================================================================
# Construction: every likelihood, every backend
# ==========================================================================
class TestConstruction:
    # `test_every_likelihood_constructs` used to sit here, asserting that the
    # evidence is finite for each of the fourteen. `test_evidence_matches_the_
    # closed_form` below asserts that it equals the exact value for the same
    # fourteen, which strictly subsumes it, so it was removed rather than
    # memoised: the cheapest test is the one that does not run.

    @pytest.mark.parametrize("name", FRACTIONAL)
    def test_the_fractional_orders_really_are_fractional(self, name):
        """Guards the premise: if `a` became an integer these tests prove nothing."""
        post = _build(name)

        assert post.a != round(post.a)

    @pytest.mark.parametrize("name", sorted(KNOWN))
    def test_evidence_matches_the_closed_form(self, name):
        """`log p(y) = log_c + log D^a M(-b)`, with the Gamma MGF known exactly.

        `maxwell-boltzmann` gets a looser tolerance because it lands at
        `a = 4.5, t = -14`, where the default truncation `initial_L = 10` stops
        short — see `test_known_broken.py`. That is a quadrature defect this PR
        does not touch, and the loose tolerance here is scoped to the one case
        that hits it rather than blanketing the others.
        """
        post = _build(name)
        expected = post.log_c + gamma_mgf_derivative_log(post.a, -post.b)
        tolerance = 1e-4 if name == "maxwell-boltzmann" else 1e-8

        assert post.evidence()[0] == pytest.approx(expected, rel=tolerance)

    @pytest.mark.parametrize("method", ["auto", "symbolic", "bell", "jax"])
    def test_every_integer_backend_is_reachable_through_the_class(self, method):
        """`bell` and `jax` raised at construction too, for the same reason.

        The backend matrix called the integer row "works today"; that was true
        of `mgfDerivative` and false of the public class.
        """
        post = _build("poisson", method=method)

        assert post.evidence()[0] == pytest.approx(-6.0965964652, rel=1e-8)

    @pytest.mark.parametrize(
        "method",
        [
            "auto",
            "scipy",
            # mpmath at its default precision costs 19 s; jax and bell pay for
            # the reinterpretation path. All three are covered by the full run.
            pytest.param("mpmath", marks=pytest.mark.slow),
            pytest.param("bell", marks=pytest.mark.slow),
            pytest.param("jax", marks=pytest.mark.slow),
        ],
    )
    def test_every_fractional_backend_is_reachable_through_the_class(self, method):
        """`bell`/`jax` here exercise the reinterpretation path end to end."""
        post = _build("halfnormal", method=method)

        assert post.evidence()[0] == pytest.approx(-5.338223705, rel=1e-8)


# ==========================================================================
# The deferred representation
# ==========================================================================
class TestDeferral:
    def test_symbolic_backend_still_builds_an_expression(self):
        """Deferral must not be applied where an expression is available.

        `post_density`, `post_cdf`, `post_mgf` and `update` all branch on
        `_deriv_is_symbolic`, so making the symbolic route defer would silently
        disable every symbolic path in the class.
        """
        post = _build("poisson", method="symbolic")

        assert post._deriv_is_symbolic
        assert isinstance(post._deriv, sp.Expr)

    def test_numeric_backend_defers(self):
        post = _build("halfnormal", method="scipy")

        assert not post._deriv_is_symbolic
        assert callable(post._deriv)

    def test_backend_options_are_captured_not_re_passed(self):
        """The thunk holds the options; passing them again is a TypeError.

        `_evaluate_derivative` used to call `self._deriv(t, **self._deriv_kwargs)`.
        That branch had never run, so the double-pass was invisible; it would
        have raised for any caller who set a backend option.
        """
        post = _build("halfnormal", epsrel=1e-12, epsabs=1e-14, limit=200)

        assert np.isfinite(post.evidence()[0])

    def test_deferred_thunk_respects_the_log_argument(self):
        """`_build_derivative` hardcoded `log=False`, which `_store_result` rejects.

        With `self.log` set, `_store_result` requires a `(log_abs, sign)` tuple
        and raises `TypeError` otherwise, so a thunk built with a fixed
        `log=False` fails for every construction in the default configuration.
        """
        assert isinstance(_build("halfnormal", log=True).evidence(), tuple)

        value = _build("halfnormal", log=False).evidence()
        assert not isinstance(value, tuple)
        assert np.isfinite(float(value))

    @pytest.mark.parametrize("spelling", ["symbolic", "SYMBOLIC", "Symbolic"])
    def test_casing_does_not_change_the_representation(self, spelling):
        """The end-to-end consequence of the canonicalisation above.

        Without it the evidence was still right — `mgfDerivative` lowercases
        internally — so only the *representation* was wrong, and the failure
        surfaced far away, as `update` refusing a posterior that could update.
        """
        post = _build("poisson", method=spelling)

        assert post._deriv_is_symbolic
        assert isinstance(post._deriv, sp.Expr)
        post.update([4], likelihood="poisson", scale=1.0)

    def test_int_tol_reaches_the_deferred_thunk(self):
        """An option that changes backend selection must survive the deferral."""
        post = _build("halfnormal", int_tol=0.6)

        assert post._deriv_is_symbolic, "int_tol=0.6 should make a=1.5 count as integer"

    @pytest.mark.parametrize(
        ("method_name", "args"),
        [
            ("post_density", (1.0,)),
            # post_cdf drives the incomplete-MGF path, which is the most
            # expensive single call in this file at about 10 s.
            pytest.param("post_cdf", (1.0,), marks=pytest.mark.slow),
            ("post_mgf", (0.1,)),
            ("post_raw_moment", (1,)),
            ("post_central_moment", (2,)),
            ("post_predictive", ([1.0],)),
        ],
    )
    def test_the_inference_api_works_on_a_deferred_posterior(self, method_name, args):
        """Construction was only the first gate; every method calls back in."""
        post = _build("halfnormal")

        result = getattr(post, method_name)(*args)
        value = result[0] if isinstance(result, tuple) else result

        assert np.all(np.isfinite(np.asarray(value, dtype=float)))

    @pytest.mark.parametrize(
        "method",
        [
            "auto",
            "scipy",
            # Marked because of the memo above, not because of this test's own
            # cost. Its sibling `test_every_fractional_backend_is_reachable_
            # through_the_class[mpmath]` is already slow-marked, but both build
            # `_build("halfnormal", method="mpmath")` -- the same cache key --
            # so whichever ran first paid the 21 s and the quick pass paid it
            # regardless of the marker. Marking both is what makes either
            # marker mean anything. Measured: file 65.7 s -> 44.4 s, and no
            # extra cost in the full run, since the sibling already pays.
            pytest.param("mpmath", marks=pytest.mark.slow),
            # `bell` and `jax` stay in the quick pass deliberately: they are
            # the only tests there that drive update-refusal through the
            # integer-backend-reinterpretation route at a fractional order.
            # They cost 3.0 s and 5.8 s.
            "bell",
            "jax",
        ],
    )
    def test_sequential_update_refuses_rather_than_returning_minus_infinity(
        self, method
    ):
        """A deferred posterior cannot yet be updated, and must say so.

        `to_prior_object` builds the updated prior's MGF from this posterior's
        and can only do so symbolically; its numeric route returns a prior with
        no `mgf_sym`, which no backend can consume. `bell` and `jax` raised;
        `scipy` and `mpmath` returned `-inf`, because the tan-transform
        integrand's blanket `except Exception: return 0.0` turned the missing
        MGF into a zero at every quadrature node and `log(0)` did the rest.

        Making this reachable is what exposed the silent case: before the
        deferral fix, a fractional posterior could not be constructed at all.
        """
        post = _build("halfnormal", method=method)

        with pytest.raises(ValueError, match="requires a symbolic derivative"):
            post.update([1.0], likelihood="halfnormal")


# ==========================================================================
# Sequential updating: 'auto' and 'symbolic' must not diverge
# ==========================================================================
class TestUpdateGuard:
    @pytest.mark.parametrize("method", ["auto", "symbolic"])
    def test_update_works_wherever_the_derivative_is_symbolic(self, method):
        """`auto` resolves to `symbolic` here, so the two must agree.

        The old guard tested `_is_symbolic` — whether the *result* at `t = -b`
        is an expression — which is False whenever the hyperparameters are
        numeric, i.e. in the ordinary case. So it rejected `method='symbolic'`
        on precisely the posteriors that could update, while `method='auto'`
        went through and returned the right answer.
        """
        staged = _build("poisson", method=method)
        updated = staged.update([4], likelihood="poisson", scale=1.0)

        one_shot = MGFDerivative(
            gamma_prior_spec(), data=[1, 2, 3, 4], likelihood="poisson", scale=1.0
        )

        assert staged.evidence()[0] + updated.evidence()[0] == pytest.approx(
            one_shot.evidence()[0], rel=1e-8
        )

    @pytest.mark.parametrize("method", ["bell", "jax"])
    def test_update_refuses_for_numeric_integer_backends(self, method):
        post = _build("poisson", method=method)

        with pytest.raises(ValueError, match="requires a symbolic derivative"):
            post.update([4], likelihood="poisson", scale=1.0)


# ==========================================================================
# Agreement across backends
# ==========================================================================
@pytest.mark.slow
def test_backends_agree_on_a_fractional_order():
    """`scipy` and `mpmath` are independent implementations of the same integral."""
    prior = gamma_prior_spec()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scipy_log = mgfDerivative(1.5, prior, method="scipy", t=-1.0, log=True)[0]
        mpmath_log = mgfDerivative(1.5, prior, method="mpmath", t=-1.0, log=True)[0]

    assert scipy_log == pytest.approx(mpmath_log, rel=1e-8)
    assert scipy_log == pytest.approx(gamma_mgf_derivative_log(1.5, -1.0), rel=1e-8)
