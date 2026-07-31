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

import warnings

import numpy as np
import pytest
import sympy as sp

from jumufraktiv.derivativeDispatch import mgfDerivative
from jumufraktiv.MGFDerivative_class import MGFDerivative

# ==========================================================================
# PR 3 — import and registry integrity
# ==========================================================================
# All PR 3 entries (registry initialisation, prior-discovery isolation, the two
# unqualified imports) are fixed. Their tests now live, unmarked, in
# test_registry.py and test_dispatch_imports.py.


# ==========================================================================
# PR 4 — the fractional-order path
# ==========================================================================
# The construction defect (`_build_derivative` passing `t=None` to a backend that
# requires it) is fixed. Its test now lives, unmarked and much expanded, in
# test_deferred_construction.py.


# The array-order coercion (`int(o)` turning a fractional order into a whole
# one, plus the float() coercions of t and u and the flattened reassembly) is
# fixed. Its two records -- the dispatcher-level one and the public-API one
# asserting that the 0th raw moment is exactly 1 -- now live, unmarked, in
# test_array_order.py, along with the sample-size parity coverage that the
# originals lacked.


# The symbolic fractional backend is repaired. It returns a closed form for
# the Gamma prior, exact to 1e-16 with the 1/Gamma(gamma) prefactor applied,
# and raises NotImplementedError naming the prior where SymPy declines --
# rather than returning None, which used to surface as a TypeError far from
# its cause. Its tests now live, unmarked, in test_symbolic_fractional.py.


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="PR 6: MGFDerivative.to_prior_object can only build the updated "
    "prior's MGF symbolically. Its numeric route returns a prior with no "
    "mgf_sym/cgf_sym, which no derivative backend can consume, so a posterior "
    "from any numeric backend cannot be updated. Since no symbolic backend "
    "works for fractional orders, fractional posteriors cannot be updated at all",
)
def test_fractional_posterior_can_be_updated_sequentially(gamma_prior):
    """Evidence factorises, so staged conditioning must match one-shot.

    Reachable only since the deferred-construction fix: a fractional posterior
    could not previously be built. `update` now raises rather than returning the
    ``-inf`` that `scipy` and `mpmath` produced, but raising is not the goal.
    """
    staged = MGFDerivative(gamma_prior, data=[1.0, 2.0, 3.0], likelihood="halfnormal")
    updated = staged.update([1.0], likelihood="halfnormal")

    one_shot = MGFDerivative(
        gamma_prior, data=[1.0, 2.0, 3.0, 1.0], likelihood="halfnormal"
    )

    assert staged.evidence()[0] + updated.evidence()[0] == pytest.approx(
        one_shot.evidence()[0], rel=1e-8
    )


# ==========================================================================
# PR 5 — symbolic-path correctness
# ==========================================================================
def test_an_integer_valued_sympy_order_is_accepted(gamma_prior):
    """`sp.Integer(2)` must behave exactly like `2`.

    This half was a real defect. The guard was `isinstance(order, int)`, which
    rejects `sympy.Integer` -- an integer by every meaning except Python's type
    check, and one SymPy arithmetic produces routinely. `resolve_backend`
    classifies it as a symbolic order, so it reached the same dead end as a
    free symbol and raised `TypeError`.
    """
    from_python_int = mgfDerivative(2, gamma_prior, method="symbolic", t=None)
    from_sympy_int = mgfDerivative(
        sp.Integer(2), gamma_prior, method="symbolic", t=None
    )

    assert sp.simplify(from_python_int - from_sympy_int) == 0


def test_a_genuinely_symbolic_order_is_refused_clearly(gamma_prior):
    """An order carrying free symbols is refused, and says why.

    This is **not** a defect record. It documents a decision: `sympy.diff`
    needs a concrete number of times to differentiate and cannot return a
    formula in `n`. Closed forms in the order exist for particular priors --
    the Gamma MGF gives a Pochhammer symbol -- but there is no general route,
    and the package has no use for one, since `a(y)` comes from the data and is
    always numeric.

    What was wrong was the *reporting*. The dispatcher warned that the result
    would be "the analytic continuation to non-integer orders", then raised
    `TypeError` blaming SymPy's support for symbolic differentiation -- which
    is not what is missing. The warning fired before the failure, so the last
    thing a caller saw was a claim about a result they never received.
    """
    n = sp.Symbol("n", positive=True, integer=True)

    with pytest.raises(NotImplementedError, match="symbolic number of times"):
        mgfDerivative(n, gamma_prior, method="symbolic", t=None)


def test_refusing_a_symbolic_order_does_not_warn_first(gamma_prior):
    """No warning may precede the refusal.

    Asserted separately because `pytest.raises` alone would pass while the
    misleading warning was still being issued.
    """
    n = sp.Symbol("n", positive=True, integer=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(NotImplementedError):
            mgfDerivative(n, gamma_prior, method="symbolic", t=None)


# ==========================================================================
# PR 6 — numerical robustness
# ==========================================================================
# The kernel defects are fixed and their records are gone: large-order overflow,
# `log=False` raising on the near-integer path, near-integer inaccuracy, and
# truncation that did not adapt to the evaluation point. All four were symptoms
# of one design fault, and all four now assert positively, against the
# closed-form Gamma reference, in test_fixed_grid_kernel.py.
#
# The domain guards, `post_quantile`'s bracketing and `logminus` were repaired
# earlier; those tests live in test_numerical_guards.py.
#
# What remains for PR 6c: mpmath below dps 30, the alternating-CGF
# cancellation, and the sequential-update record above.


# ==========================================================================
# PR 12 — public API surface
# ==========================================================================
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
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
    raises=TypeError,
    reason="PR 12: post_sample calls the unseeded legacy np.random.rand and "
    "takes no rng argument, so results are not reproducible",
)
def test_post_sample_is_reproducible(poisson_posterior):
    """Two draws under the same seed must agree."""
    first = poisson_posterior.post_sample(8, rng=np.random.default_rng(0))
    second = poisson_posterior.post_sample(8, rng=np.random.default_rng(0))

    assert np.array_equal(first, second)
