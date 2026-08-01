"""The symbolic path must produce the same mathematics as the numeric one.

Three defects lived here, and they shared a cause: **the symbolic path had no
test that evaluated its output.** Nothing substituted hyperparameters into a
returned expression and compared the number against a reference, so an
expression could come back structurally plausible and numerically unusable
without any test noticing.

Two of the three were invisible to `ruff` as well as to the suite, and the
third was visible to `ruff` alone -- `F821`, undefined name -- which is why
that rule is deliberately left blocking in `pyproject.toml`.

Reaching this path at all takes a prior whose hyperparameters are still
symbols. The registry cannot build one: every factory calls `float()` on its
parameters, so `from_registry("gamma", params={"alpha": alpha})` raises
`TypeError: Cannot convert expression to float`. The custom-prior route is the
only way in, which is a large part of why this code was under-exercised.
"""

import numpy as np
import pytest
import sympy as sp

from conftest import ALPHA, BETA, POISSON_DATA, POISSON_SCALE, poisson_log_evidence
from jumufraktiv.MGFDerivative_class import MGFDerivative
from jumufraktiv.mitMGFprior_class import mitMGFprior
from jumufraktiv.symbols import param, t, theta, u

alpha = param("alpha")
beta = param("beta")

#: The conjugate posterior for the canonical test problem is
#: Gamma(ALPHA + sum(y), BETA + n * scale).
POST_SHAPE = ALPHA + sum(POISSON_DATA)
POST_RATE = BETA + len(POISSON_DATA) * POISSON_SCALE


def _gamma_mgf():
    return (beta / (beta - t)) ** alpha


def _gamma_pdf():
    return beta**alpha / sp.gamma(alpha) * theta ** (alpha - 1) * sp.exp(-beta * theta)


@pytest.fixture
def symbolic_prior():
    """A Gamma prior whose hyperparameters are left as free symbols."""
    return mitMGFprior(
        name="gamma_symbolic",
        mgf_sym=_gamma_mgf(),
        pdf_sym=_gamma_pdf(),
        params={},
    ).as_mitMGFprior()


@pytest.fixture
def substituted_prior():
    """Symbolic MGF and PDF, but `params` supplies the values.

    This is the configuration that exposed the discarded substitution: the
    expressions carry `alpha` and `beta`, and `params` says what they are, so
    a correct implementation resolves them and an incorrect one does not.
    """
    return mitMGFprior(
        name="gamma_substituted",
        mgf_sym=_gamma_mgf(),
        pdf_sym=_gamma_pdf(),
        params={"alpha": ALPHA, "beta": BETA},
    ).as_mitMGFprior()


def _with_imgf(prior):
    """Attach the incomplete MGF, which `post_cdf` requires.

    The four numeric slots are never called on the symbolic path; they exist
    because `has_iMGF()` checks that all six are present.
    """
    imgf = (
        (beta / (beta - t)) ** alpha
        * sp.lowergamma(alpha, (beta - t) * u)
        / sp.gamma(alpha)
    )
    prior.imgf_sym = imgf
    prior.logimgf_sym = sp.log(imgf)

    def _unused(*args, **kwargs):  # pragma: no cover - asserts it is not called
        raise AssertionError("the symbolic path must not call a numeric iMGF")

    prior.imgf = prior.logimgf = prior.imgf_jax = prior.logimgf_jax = _unused
    return prior


def _posterior(prior):
    return MGFDerivative(
        prior,
        data=POISSON_DATA,
        likelihood="poisson",
        scale=POISSON_SCALE,
        method="symbolic",
    )


def _numbers(expr):
    """Substitute the hyperparameters and return a float."""
    return float(sp.N(expr.subs({alpha: ALPHA, beta: BETA})))


# ==========================================================================
# post_cdf
# ==========================================================================
def test_post_cdf_symbolic_returns_an_expression(symbolic_prior):
    """It used to raise, because it named two symbols that do not exist.

    The body referenced `t_sym` and `u_sym`; this module imports `t` and `u`.
    Every call raised `NameError`, which a blanket `except Exception` re-raised
    as `RuntimeError: Symbolic posterior CDF computation failed: name 't_sym'
    is not defined` -- reporting a typo as a failed computation.
    """
    post = _posterior(_with_imgf(symbolic_prior))

    expr = post.post_cdf(u, log=True)

    assert isinstance(expr, sp.Expr)
    assert {s.name for s in expr.free_symbols} == {"alpha", "beta", "u"}


def test_post_cdf_symbolic_matches_the_conjugate_closed_form(symbolic_prior):
    """Structure is not enough: the expression must evaluate to the right number.

    The posterior for this problem is Gamma(ALPHA + sum(y), BETA + n*scale), so
    its CDF is known exactly.
    """
    pytest.importorskip("scipy")
    from scipy.stats import gamma as gamma_dist

    post = _posterior(_with_imgf(symbolic_prior))
    expr = post.post_cdf(u, log=True)

    for point in (0.5, 1.0, 2.0, 3.0):
        got = _numbers(expr.subs(u, point))
        expected = float(
            np.log(gamma_dist.cdf(point, a=POST_SHAPE, scale=1.0 / POST_RATE))
        )
        assert got == pytest.approx(expected, rel=1e-10)


def test_post_cdf_says_which_symbols_are_unresolved(symbolic_prior):
    """A numeric threshold against an unresolved prior must explain itself.

    The hyperparameters are still free, so no number exists. The old code
    reported this as a generic failed computation; it now names the symbols
    and says how to supply them.
    """
    post = _posterior(_with_imgf(symbolic_prior))

    with pytest.raises(ValueError, match="alpha"):
        post.post_cdf(0.5)


# ==========================================================================
# post_density
# ==========================================================================
def test_post_density_substitutes_the_known_hyperparameters(substituted_prior):
    """`params` must actually reach the returned expression.

    Two things prevented it. The substitution was applied to `pdf_sym` *after*
    `log_prior` had been formed from it, so the result was computed and thrown
    away; and even in the right order it would have covered only the prior's
    density, never the normalising constant, which carries the same
    hyperparameters. The density came back free in `alpha` and `beta` despite
    `params` giving both.
    """
    post = _posterior(substituted_prior)

    expr = post.post_density(theta, log=True)

    assert {s.name for s in expr.free_symbols} == {"theta"}


def test_post_density_matches_the_conjugate_closed_form(substituted_prior):
    """And the substituted expression must be the right density."""
    pytest.importorskip("scipy")
    from scipy.stats import gamma as gamma_dist

    post = _posterior(substituted_prior)
    expr = post.post_density(theta, log=True)

    for point in (0.5, 1.0, 2.0):
        got = float(sp.N(expr.subs(theta, point)))
        expected = float(gamma_dist.logpdf(point, a=POST_SHAPE, scale=1.0 / POST_RATE))
        assert got == pytest.approx(expected, rel=1e-10)


# ==========================================================================
# evidence
# ==========================================================================
def test_symbolic_evidence_has_no_unresolvable_symbols(symbolic_prior):
    """It used to return an expression the caller could never evaluate.

    The branch multiplied by `c_func()`, the likelihood's normalising constant
    as a *general formula*: for Poisson that is
    `Product(s[i]**y[i]/factorial(y[i]), (i, 1, n))`, over indexed symbols
    never bound to this object's data. So the evidence came back carrying free
    `n`, `s` and `y` and an unevaluated `Product`, and no substitution
    available to the caller could resolve them.

    The constant is already known numerically -- the likelihood's `ready`
    function computed it from the data as `log_c` -- so the only symbols that
    should survive are the prior's own hyperparameters.
    """
    post = _posterior(symbolic_prior)

    evidence = post.evidence()

    assert {s.name for s in evidence.free_symbols} == {"alpha", "beta"}
    assert not evidence.has(sp.Product)


def test_symbolic_evidence_matches_the_closed_form(symbolic_prior):
    """The number it evaluates to must be the conjugate evidence."""
    post = _posterior(symbolic_prior)

    got = _numbers(post.evidence())
    expected = float(
        np.exp(
            poisson_log_evidence(
                POISSON_DATA, scale=POISSON_SCALE, alpha=ALPHA, beta=BETA
            )
        )
    )

    assert got == pytest.approx(expected, rel=1e-10)


def test_symbolic_and_numeric_evidence_agree(symbolic_prior):
    """The two branches must not disagree about what the constant is.

    They took it from different places -- `c_func()` symbolically and `log_c`
    numerically -- which is exactly how they came to disagree.
    """
    from jumufraktiv.mitMGFprior_class import mitMGFprior as Prior

    symbolic = _numbers(_posterior(symbolic_prior).evidence())

    numeric_prior = Prior.from_registry("gamma", params={"alpha": ALPHA, "beta": BETA})
    log_abs = MGFDerivative(
        numeric_prior,
        data=POISSON_DATA,
        likelihood="poisson",
        scale=POISSON_SCALE,
    ).evidence()

    assert symbolic == pytest.approx(float(np.exp(log_abs)), rel=1e-10)
