"""Tests for the prior registry and the ``mitMGFprior`` container."""

import numpy as np
import pytest
import sympy as sp
from conftest import ALPHA, BETA, POISSON_DATA, POISSON_SCALE, poisson_log_evidence

from jumufraktiv import registry
from jumufraktiv.MGFDerivative_class import MGFDerivative
from jumufraktiv.mitMGFprior_class import mitMGFprior
from jumufraktiv.symbols import t, theta


# ==========================================================================
# Registry
# ==========================================================================
class TestRegistry:
    def test_list_priors_returns_names(self):
        priors = registry.list_priors()

        assert isinstance(priors, list)
        assert "gamma" in priors

    def test_get_prior_returns_a_factory(self):
        factory = registry.get_prior("gamma")
        spec = factory({"alpha": ALPHA, "beta": BETA})

        assert callable(factory)
        assert {"mgf", "cgf"} <= set(spec)

    def test_unknown_prior_raises_keyerror(self):
        with pytest.raises(KeyError, match="not found"):
            registry.get_prior("no-such-prior")

    def test_unknown_prior_error_lists_the_alternatives(self):
        """The error must be actionable, not merely correct."""
        with pytest.raises(KeyError) as excinfo:
            registry.get_prior("no-such-prior")

        assert "gamma" in str(excinfo.value)

    def test_initialize_is_idempotent(self):
        before = set(registry.list_priors())
        registry.initialize()
        registry.initialize()

        assert set(registry.list_priors()) == before

    def test_every_dictionary_prior_registers(self):
        """One prior module's optional backend must not cost the others.

        `paretoMGF` used to `import torch` eagerly, and discovery aborted on the
        first module that raised — so a missing optional extra silently removed
        both `pareto` and `uniform`, and the registry came up with half its
        contents and a warning.
        """
        assert set(registry.list_priors()) >= {
            "gamma",
            "heaviside",
            "pareto",
            "uniform",
        }

    def test_no_prior_module_failed_to_import(self):
        assert registry.failed_prior_modules() == {}

    def test_missing_prior_error_mentions_failed_modules(self, monkeypatch):
        """An absent prior must say whether a module failed, not just 'not found'.

        'Prior not found' is unactionable when the real cause is an optional
        dependency; naming the module and the exception makes it fixable.
        """
        import jumufraktiv.MGFdictionary as dictionary

        monkeypatch.setitem(
            dictionary.FAILED_MODULES,
            "brokenMGF",
            ImportError("no module named 'nope'"),
        )

        with pytest.raises(KeyError) as excinfo:
            registry.get_prior("brokenprior")

        message = str(excinfo.value)
        assert "brokenMGF" in message
        assert "failed to import" in message

    @pytest.mark.parametrize("missing", ["mgf", "cgf"])
    def test_make_prior_spec_requires_mgf_and_cgf(self, missing):
        fields = {"mgf": lambda x: x, "cgf": lambda x: x}
        del fields[missing]

        with pytest.raises(ValueError, match=missing):
            registry.make_prior_spec(**fields)

    def test_register_prior_adds_to_the_registry(self):
        name = "_test_only_prior"

        @registry.register_prior(name)
        def _factory(params):
            return registry.make_prior_spec(mgf=lambda x: 1.0, cgf=lambda x: 0.0)

        try:
            assert registry.get_prior(name) is _factory
        finally:
            registry.PRIOR_REGISTRY.pop(name, None)


# ==========================================================================
# mitMGFprior container
# ==========================================================================
class TestMitMGFPrior:
    def test_registry_prior_is_fully_compiled(self, gamma_prior):
        assert mitMGFprior.is_mitMGFprior(gamma_prior)

    def test_gamma_prior_exposes_the_incomplete_mgf(self, gamma_prior):
        assert gamma_prior.has_iMGF()

    def test_unknown_prior_name_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown prior"):
            mitMGFprior.from_registry("no-such-prior")

    def test_from_registry_works_as_the_first_registry_call(self):
        """`from_registry` must not depend on some other call having run first.

        It read `PRIOR_REGISTRY` directly without initialising it, so in a fresh
        process it raised "Unknown prior 'gamma'" — the registry was simply
        empty. A subprocess is the only honest way to test this: once any test
        in this session touches the registry, the module-level cache hides it.
        """
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            """
            import warnings
            warnings.simplefilter("ignore")
            from jumufraktiv.mitMGFprior_class import mitMGFprior
            prior = mitMGFprior.from_registry(
                "gamma", params={"alpha": 2.0, "beta": 3.0}
            )
            assert prior.name == "gamma"
            print("OK")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )

        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_compiled_mgf_matches_the_closed_form(self, gamma_prior):
        """The compiled numeric MGF must equal (beta / (beta - t)) ** alpha."""
        for t_val in (-2.0, -0.5, 0.0, 1.0):
            expected = (BETA / (BETA - t_val)) ** ALPHA
            assert gamma_prior.mgf(t_val) == pytest.approx(expected, rel=1e-12)

    def test_compiled_cgf_is_the_log_of_the_mgf(self, gamma_prior):
        for t_val in (-2.0, -0.5, 0.0, 1.0):
            assert gamma_prior.cgf(t_val) == pytest.approx(
                np.log(gamma_prior.mgf(t_val)), rel=1e-12
            )


# ==========================================================================
# The custom-prior route
# ==========================================================================
class TestCustomPrior:
    @staticmethod
    def _build():
        """A hand-built Gamma prior, bypassing the registry entirely."""
        mgf_sym = (BETA / (BETA - t)) ** ALPHA
        pdf_sym = (
            BETA**ALPHA / sp.gamma(ALPHA) * theta ** (ALPHA - 1) * sp.exp(-BETA * theta)
        )
        return mitMGFprior(
            name="custom_gamma", mgf_sym=mgf_sym, pdf_sym=pdf_sym
        ).as_mitMGFprior()

    def test_symbolic_construction_compiles(self):
        prior = self._build()

        assert mitMGFprior.is_mitMGFprior(prior)
        assert prior.mgf(-1.0) == pytest.approx((BETA / (BETA + 1.0)) ** ALPHA)

    def test_custom_prior_reproduces_the_registry_result(self):
        """A hand-built prior must give the same evidence as the registry one."""
        post = MGFDerivative(
            self._build(),
            data=POISSON_DATA,
            likelihood="poisson",
            scale=POISSON_SCALE,
        )
        log_ev, sign = post.evidence()

        assert sign == 1
        assert log_ev == pytest.approx(poisson_log_evidence(POISSON_DATA), rel=1e-10)

    def test_custom_prior_without_imgf_reports_no_imgf(self):
        assert not self._build().has_iMGF()

    @pytest.mark.parametrize("provided", ["mgf_sym", "pdf_sym"])
    def test_symbolic_mode_requires_both_expressions(self, provided):
        kwargs = {
            provided: (BETA / (BETA - t)) ** ALPHA if provided == "mgf_sym" else theta
        }

        with pytest.raises(ValueError, match="requires both"):
            mitMGFprior(**kwargs).as_mitMGFprior()

    def test_empty_prior_is_rejected(self):
        with pytest.raises(ValueError, match="Must provide either"):
            mitMGFprior().as_mitMGFprior()


# ==========================================================================
# Constructor validation on MGFDerivative
# ==========================================================================
class TestMGFDerivativeValidation:
    def test_non_prior_object_is_rejected(self):
        with pytest.raises(TypeError, match="must be a mitMGFprior"):
            MGFDerivative("gamma", data=POISSON_DATA, likelihood="poisson")

    def test_unknown_likelihood_is_rejected(self, gamma_prior):
        with pytest.raises(ValueError, match="Unknown likelihood"):
            MGFDerivative(gamma_prior, data=POISSON_DATA, likelihood="no-such-thing")

    def test_likelihood_name_is_case_insensitive(self, gamma_prior):
        post = MGFDerivative(
            gamma_prior, data=POISSON_DATA, likelihood="POISSON", scale=POISSON_SCALE
        )

        assert post.likelihood == "poisson"
