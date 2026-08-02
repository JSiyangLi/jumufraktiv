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
    from jumufraktiv.MGFPrior_class import MGFPrior

    return MGFPrior.from_registry("gamma", params={"alpha": 2.0, "beta": 3.0})


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

        Every likelihood is held to the same tolerance. `maxwell-boltzmann`
        used to be excepted at `1e-4`, because it lands at `a = 4.5, t = -14`
        and the adaptive kernel's truncation stopped short there. PR 6b
        replaced that kernel and the case now agrees to 1.6e-14, so the
        exception was four orders of magnitude looser than the *worst* result
        it was covering — a test that would no longer have noticed the defect
        coming back, let alone a smaller one.
        """
        post = _build(name)
        expected = post.log_c + gamma_mgf_derivative_log(post.a, -post.b)

        assert post.evidence() == pytest.approx(expected, rel=1e-8)

    @pytest.mark.parametrize("method", ["auto", "symbolic", "bell", "jax"])
    def test_every_integer_backend_is_reachable_through_the_class(self, method):
        """`bell` and `jax` raised at construction too, for the same reason.

        The backend matrix called the integer row "works today"; that was true
        of `mgfDerivative` and false of the public class.
        """
        post = _build("poisson", method=method)

        assert post.evidence() == pytest.approx(-6.0965964652, rel=1e-8)

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

        assert post.evidence() == pytest.approx(-5.338223705, rel=1e-8)


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
        post = _build("halfnormal", tol=1e-12)

        assert np.isfinite(post.evidence())

    def test_deferred_thunk_respects_the_log_argument(self):
        """`_build_derivative` hardcoded `log=False`, which `_store_result` rejects.

        With `self.log` set, `_store_result` requires a `(log_abs, sign)` tuple
        and raises `TypeError` otherwise, so a thunk built with a fixed
        `log=False` fails for every construction in the default configuration.
        """
        logged = _build("halfnormal", log=True)
        assert logged.log_abs is not None and logged.value is None
        assert np.isfinite(float(logged.evidence()))

        plain = _build("halfnormal", log=False)
        assert plain.value is not None and plain.log_abs is None
        assert np.isfinite(float(plain.evidence()))

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
            # `bell` and `jax` are NOT here: they differentiate the prior's
            # symbolic CGF and cannot consume a prior built from a numeric
            # posterior at all. That is a limit of those two backends rather
            # than of updating, and TestUpdateGuard asserts it separately.
        ],
    )
    def test_a_deferred_posterior_can_now_be_updated(self, method):
        """This asserted the opposite until the expectation route existed.

        `to_prior_object`'s numeric route returns a prior with a density and no
        `mgf_sym`, which no *differentiating* backend can consume -- `bell` and
        `jax` raised, `scipy` and `mpmath` returned `-inf`. The old test pinned
        that refusal as correct, which it was only for as long as every route
        needed an MGF.

        The direct-expectation route needs the density instead, so the prior
        that route builds is exactly what it consumes. `halfnormal` on three
        observations gives `a = 1.5`, a fractional posterior, which had no
        working update path by any method.

        The property asserted is that evidence factorises, which needs no
        reference value: staged conditioning must equal one-shot.
        """
        post = _build("halfnormal", method=method)
        # `auto` is what routes to the expectation backend; an explicit
        # differentiating method still cannot consume the prior, which
        # TestUpdateGuard asserts separately.
        updated = post.update([1.0], likelihood="halfnormal", method="auto")

        one_shot = MGFDerivative(
            gamma_prior_spec(), data=[1.0, 2.0, 3.0, 1.0], likelihood="halfnormal"
        )

        assert post.evidence() + updated.evidence() == pytest.approx(
            one_shot.evidence(), rel=1e-8
        )


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

        assert staged.evidence() + updated.evidence() == pytest.approx(
            one_shot.evidence(), rel=1e-8
        )

    @pytest.mark.parametrize(
        "method,likelihood",
        [
            # `scipy` and `mpmath` are only valid for a fractional order, and
            # `bell`/`jax` only for an integer one, so each is paired with a
            # likelihood that produces the right kind: halfnormal gives
            # a = n/2 = 1.5, poisson gives a = sum(y) = 6.
            ("scipy", "halfnormal"),
            pytest.param("mpmath", "halfnormal", marks=pytest.mark.slow),
            ("bell", "poisson"),
            ("jax", "poisson"),
        ],
    )
    def test_an_explicit_differentiating_method_cannot_update_but_says_why(
        self, method, likelihood
    ):
        """A real limit of those backends, not a missing feature.

        Every differentiating backend needs the prior's symbolic MGF or CGF,
        and the prior built from a numeric posterior carries only a density.
        The direct-expectation route needs just the density, which is why
        `auto` and an explicit `expectation` both work and these do not.

        What changed is the reporting. This used to surface as "Prior does not
        provide a symbolic CGF" from deep in the Bell backend, or as a
        `TracerArrayConversionError` from inside JAX, neither of which mentions
        updating. It is now refused up front, naming the backend and saying
        which method does work.
        """
        extra = {"scale": 1.0} if likelihood == "poisson" else {}
        post = _build(likelihood, method=method)

        with pytest.raises(ValueError, match="differentiates the prior's symbolic"):
            post.update([1.0], likelihood=likelihood, **extra)


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


class TestAutoUsesTheExpectationRoute:
    """`auto` must reach the same backend through the class as through the
    dispatcher.

    `_build_derivative` asks `resolve_backend`, which encodes the backend
    matrix and answers "symbolic" for `auto` at an integer order. But
    `mgfDerivative` applies a *second*, separate decision that
    `resolve_backend` does not model: with a concrete `t` it diverts `auto` to
    the direct-expectation route, whose integrand is positive and so cannot
    cancel. The class consulted only the first, so it held a symbolic
    expression and substituted into it -- evaluating the differentiated MGF,
    the route the diversion exists to avoid.

    Invisible to the suite because every class-level test used a Gamma prior,
    whose MGF derivatives have one-signed terms and therefore cannot cancel.
    """

    @staticmethod
    def _uniform_oracle(order, t, lo=0.5, hi=2.0):
        """log E[Theta^order e^{t Theta}] for Uniform(lo, hi), at 50 digits.

        Written out here rather than taken from the package, so it cannot
        agree with the code for the same wrong reason.
        """
        import mpmath as mp

        mp.mp.dps = 50
        width = mp.mpf(hi) - mp.mpf(lo)

        def integrand(theta):
            return theta ** mp.mpf(order) * mp.e ** (mp.mpf(t) * theta) / width

        return float(mp.log(mp.quad(integrand, [lo, hi])))

    @staticmethod
    def _uniform_prior():
        from jumufraktiv.MGFPrior_class import MGFPrior

        return MGFPrior.from_registry("uniform", params={"a": 0.5, "b": 2.0})

    @pytest.mark.parametrize("data", [[1, 2, 3], [8, 9, 10], [20, 20, 20]])
    def test_the_class_matches_the_oracle_for_an_alternating_cgf(self, data):
        """Poisson counts summing to 27 and 60 are ordinary, not contrived.

        Before the repair, `a = 27` gave 5.4e-7 relative error and `a = 60`
        raised `ValueError: Derivative at t=-b is negative` -- the class could
        not construct a posterior the dispatcher computes to 2.4e-16.
        """
        post = MGFDerivative(
            self._uniform_prior(), data=data, likelihood="poisson", scale=1.0
        )
        expected = self._uniform_oracle(float(post.a), -float(post.b))

        assert post.evidence() - post.log_c == pytest.approx(expected, rel=1e-12)

    def test_an_explicit_backend_is_never_diverted(self):
        """`method="symbolic"` must be honoured as asked, warts and all.

        The diversion is what `auto` means, not a correction applied to every
        request. A caller who names a backend gets it, and here that is
        measurably worse -- which is the caller's choice to make.
        """
        post = MGFDerivative(
            self._uniform_prior(),
            data=[8, 9, 10],
            likelihood="poisson",
            scale=1.0,
            method="symbolic",
        )
        expected = self._uniform_oracle(float(post.a), -float(post.b))

        assert post.evidence() - post.log_c != pytest.approx(expected, rel=1e-12)

    def test_the_symbolic_representation_survives_the_diversion(self, gamma_prior):
        """Routing numerically must not cost the expression the class holds.

        `post_density`, `post_cdf`, `post_mgf` and `update` branch on
        `_deriv_is_symbolic`. Swapping the representation for the expectation
        route rather than evaluating through it would take those with it.
        """
        post = MGFDerivative(
            gamma_prior, data=[1, 2, 3], likelihood="poisson", scale=1.0
        )

        assert post._deriv_is_symbolic
        assert isinstance(post.post_density(theta_val=None), sp.Expr)

    def test_an_unbound_hyperparameter_still_answers_symbolically(self):
        """A prior with free hyperparameters has no numeric route to divert to.

        The diversion is refused for a reason distinct from the other two: not
        because the caller named a backend, and not because the point is
        symbolic, but because the *answer* is. `_deriv_is_bound` is what
        records that.
        """
        from test_symbolic_correctness import _gamma_mgf, _gamma_pdf

        from jumufraktiv.MGFPrior_class import MGFPrior

        prior = MGFPrior(
            name="gamma_symbolic",
            mgf_sym=_gamma_mgf(),
            pdf_sym=_gamma_pdf(),
            params={},
        ).as_MGFPrior()
        post = MGFDerivative(
            prior, data=[1, 2, 3], likelihood="poisson", scale=1.0, method="symbolic"
        )

        assert not post._deriv_is_bound
        assert post._is_symbolic
