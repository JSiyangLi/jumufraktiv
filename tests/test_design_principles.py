"""Property tests for the three normative design principles.

These principles are stated in the module docstrings and in ``CLAUDE.md``. They
are the contract the whole package is built on, so they are asserted directly
rather than being left implicit in behaviour tests.

1. **Symbol-numeric principle** — the return *type* depends only on whether
   unresolved symbols remain, never on which code path ran.
2. **Log principle** — in the numeric state, whether a function returns
   ``(log_abs, sign)`` or a plain scalar depends only on the ``log`` argument.
3. **Tuple-vectorisation principle** — evaluation points are the pair
   ``(t, u)``; both broadcast to a common shape and evaluate as one batch.
"""

import numpy as np
import pytest
import sympy as sp
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from conftest import BETA, gamma_mgf_derivative_log
from jumufraktiv.derivativeDispatch import mgfDerivative
from jumufraktiv.symbols import t as t_sym

# Symbolic differentiation is not fast; keep the example budget small and drop
# the per-example deadline so a slow SymPy call cannot cause a flaky failure.
SLOW = settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

# The Gamma MGF has a pole at t = BETA, so stay strictly below it.
t_values = st.floats(min_value=-20.0, max_value=BETA - 0.1, allow_nan=False)
integer_orders = st.integers(min_value=0, max_value=4)
positive_u = st.floats(min_value=0.05, max_value=20.0, allow_nan=False)


# ==========================================================================
# 1. Symbol-numeric principle
# ==========================================================================
class TestSymbolNumericPrinciple:
    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    def test_none_t_returns_expression(self, gamma_prior, order):
        """``t=None`` leaves the transform variable unresolved, so return an Expr."""
        result = mgfDerivative(order, gamma_prior, method="symbolic", t=None)

        assert isinstance(result, sp.Expr)
        assert t_sym in result.free_symbols

    @SLOW
    @given(order=integer_orders, t=t_values)
    def test_numeric_t_returns_number(self, gamma_prior, order, t):
        """A fully resolved evaluation point must produce numbers, not an Expr."""
        result = mgfDerivative(order, gamma_prior, method="symbolic", t=t, log=True)

        assert not isinstance(result, sp.Expr)
        log_abs, sign = result
        assert isinstance(float(log_abs), float)
        assert int(sign) in (-1, 1)

    # A symbolic *order* should also return an Expr, but that path is currently
    # unreachable; see test_known_broken.py::test_symbolic_order_returns_expression.

    def test_return_type_is_independent_of_backend(self, gamma_prior):
        """Two backends given the same resolved point must agree on return type."""
        sym = mgfDerivative(2, gamma_prior, method="symbolic", t=-1.0, log=True)
        bell = mgfDerivative(2, gamma_prior, method="bell", t=-1.0, log=True)

        assert type(sym) is type(bell) is tuple


# ==========================================================================
# 2. Log principle
# ==========================================================================
class TestLogPrinciple:
    @SLOW
    @given(order=integer_orders, t=t_values)
    def test_log_flag_alone_determines_shape(self, gamma_prior, order, t):
        """``log=True`` yields a 2-tuple; ``log=False`` yields a bare scalar."""
        as_log = mgfDerivative(order, gamma_prior, method="symbolic", t=t, log=True)
        as_plain = mgfDerivative(order, gamma_prior, method="symbolic", t=t, log=False)

        assert isinstance(as_log, tuple) and len(as_log) == 2
        assert not isinstance(as_plain, tuple)

    @SLOW
    @given(order=integer_orders, t=t_values)
    def test_log_and_plain_representations_agree(self, gamma_prior, order, t):
        """``sign * exp(log_abs)`` must reproduce the ordinary value."""
        log_abs, sign = mgfDerivative(
            order, gamma_prior, method="symbolic", t=t, log=True
        )
        plain = mgfDerivative(order, gamma_prior, method="symbolic", t=t, log=False)

        assert sign * np.exp(log_abs) == pytest.approx(plain, rel=1e-9)


# ==========================================================================
# 3. Tuple-vectorisation principle
# ==========================================================================
class TestTupleVectorisationPrinciple:
    @pytest.mark.parametrize("shape", [(1,), (3,), (2, 3)])
    def test_array_t_preserves_shape(self, gamma_prior, shape):
        """An array of ``t`` returns arrays of the same shape."""
        t = np.linspace(-3.0, 1.0, int(np.prod(shape))).reshape(shape)
        log_abs, sign = mgfDerivative(2, gamma_prior, method="symbolic", t=t, log=True)

        assert log_abs.shape == shape
        assert sign.shape == shape

    def test_array_t_agrees_elementwise_with_scalar_calls(self, gamma_prior):
        """Batched evaluation must equal looping over scalars."""
        t = np.array([-3.0, -1.0, 0.0, 2.0])
        batch_log, _ = mgfDerivative(2, gamma_prior, method="symbolic", t=t, log=True)
        one_by_one = [
            mgfDerivative(2, gamma_prior, method="symbolic", t=float(x), log=True)[0]
            for x in t
        ]

        assert batch_log == pytest.approx(np.array(one_by_one), rel=1e-10)

    def test_array_t_matches_closed_form(self, gamma_prior):
        """Batched evaluation must also be *correct*, not merely self-consistent."""
        t = np.array([-4.0, -1.0, 0.5, 2.0])
        log_abs, sign = mgfDerivative(3, gamma_prior, method="symbolic", t=t, log=True)

        assert np.all(sign == 1)
        assert log_abs == pytest.approx(gamma_mgf_derivative_log(3, t), rel=1e-10)

    def test_scalar_t_with_array_u_broadcasts(self, gamma_prior):
        """The evaluation point is the *pair* (t, u): a scalar t broadcasts over u."""
        u = np.array([0.5, 1.0, 2.0, 4.0])
        log_abs, sign = mgfDerivative(
            2, gamma_prior, method="symbolic", t=-1.0, u=u, complete=False, log=True
        )

        assert np.shape(log_abs) == u.shape
        assert np.shape(sign) == u.shape

    @SLOW
    @given(u=positive_u)
    def test_incomplete_never_exceeds_complete(self, gamma_prior, u):
        """The incomplete MGF derivative is a partial integral of the complete one."""
        inc_log, inc_sign = mgfDerivative(
            1, gamma_prior, method="symbolic", t=-1.0, u=u, complete=False, log=True
        )
        comp_log, comp_sign = mgfDerivative(
            1, gamma_prior, method="symbolic", t=-1.0, complete=True, log=True
        )

        assert inc_sign == comp_sign == 1
        assert inc_log <= comp_log + 1e-9

    def test_complete_rejects_u(self, gamma_prior):
        """``u`` is meaningless when integrating over the whole support."""
        with pytest.raises(ValueError, match="u must be None"):
            mgfDerivative(1, gamma_prior, method="symbolic", t=-1.0, u=2.0)

    def test_incomplete_requires_u(self, gamma_prior):
        with pytest.raises(ValueError, match="u must be provided"):
            mgfDerivative(1, gamma_prior, method="symbolic", t=-1.0, complete=False)
