"""SymPy emits names that SciPy and NumPy do not define.

`lambdify` compiles such a name into a bare global rather than failing to
build, so the compiled function raises ``NameError`` on its first call. The
package probes every compiled expression for exactly that reason, and falls
back to symbolic substitution when the probe fails -- correct, and about eight
times slower per node.

`jumufraktiv.special` supplies the missing names so the fallback is not
reached. These tests hold both halves: the values must be right, and the fast
path must actually be taken.
"""

import warnings

import mpmath as mp
import numpy as np
import pytest

from jumufraktiv.derivativeDispatch import mgfDerivative
from jumufraktiv.MGFPrior_class import MGFPrior
from jumufraktiv.special import expint
from jumufraktiv.symbolic_cache import cached_lambdify, clear_derivative_cache

PARETO = {"alpha": 3.0, "xi": 1.0}


@pytest.mark.parametrize("order", [0.5, 1.0, 2.5, 4.0, 4.5])
@pytest.mark.parametrize("argument", [0.05, 0.5, 2.0, 10.0])
def test_expint_matches_mpmath(order, argument):
    """The oracle is mpmath at 40 digits, which is also what backs it.

    Not circular: the implementation works at 30 digits and returns float64,
    so this asserts that the precision it keeps is enough to round correctly,
    which is the property that could plausibly be wrong.
    """
    with mp.workdps(40):
        exact = float(mp.expint(mp.mpf(order), mp.mpf(argument)))

    assert expint(order, argument) == pytest.approx(exact, rel=1e-14)


def test_expint_broadcasts_and_keeps_shape():
    """`lambdify` hands it whole arrays, so scalar-only would be useless."""
    orders = np.array([[1.5], [2.5]])
    arguments = np.array([0.5, 1.0, 2.0])

    out = expint(orders, arguments)

    assert out.shape == (2, 3)
    assert out[1, 2] == pytest.approx(expint(2.5, 2.0), rel=1e-14)
    assert np.isscalar(expint(1.5, 1.0)) or np.ndim(expint(1.5, 1.0)) == 0


def test_the_pareto_mgf_now_compiles():
    """The property that makes the speed difference, asserted directly.

    Timing would be the obvious check and a bad one: it varies with the
    machine, so it would be either flaky or too loose to mean anything. Whether
    `cached_lambdify` returns a callable rather than `None` is a property of
    the code, and it is the one that decides which path runs.
    """
    from jumufraktiv.symbols import t as t_sym

    clear_derivative_cache()
    prior = MGFPrior.from_registry("pareto", params=PARETO)

    compiled = cached_lambdify(prior.mgf_sym_out, (t_sym,), probe=(np.array([-1.0]),))

    assert compiled is not None, (
        "the Pareto MGF still has no compiled form, so every quadrature node "
        "falls back to symbolic substitution"
    )
    assert np.isfinite(np.asarray(compiled(np.array([-1.0])), dtype=float)).all()


@pytest.mark.slow
@pytest.mark.parametrize(("order", "t"), [(1.5, -1.0), (0.5, -1.0), (1.95, -0.5)])
def test_the_compiled_path_did_not_change_the_answer(order, t):
    """A fast path may only skip work, never alter a result.

    The oracle integrates the Pareto density written out here, so it is
    independent of the package: `E[theta^a e^{t theta}]` for Pareto(3, 1).
    """
    prior = MGFPrior.from_registry("pareto", params=PARETO)

    with mp.workdps(50):
        alpha, xi = mp.mpf(PARETO["alpha"]), mp.mpf(PARETO["xi"])
        exact = mp.quad(
            lambda th: (
                th ** mp.mpf(order)
                * mp.e ** (mp.mpf(t) * th)
                * alpha
                * xi**alpha
                / th ** (alpha + 1)
            ),
            [xi, mp.inf],
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_abs, sign = mgfDerivative(order, prior, method="scipy", t=t, log=True)

    got = float(sign) * float(np.exp(log_abs))
    assert got == pytest.approx(float(exact), rel=1e-12)
