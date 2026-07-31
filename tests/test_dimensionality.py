"""Two-dimensional input must be rejected, by every entry point.

**These tests exist so that the de-duplication in this same PR cannot change
this behaviour invisibly.** They are deliberately the first commit: the
fourteen `like_stats` modules carry a byte-identical `_extract_1d`, and folding
those into one shared helper necessarily makes their validation uniform. That
is the point of the refactor — but "uniform" could mean uniformly accepting or
uniformly rejecting, and with no test either outcome ships green.

What the entry points do today, measured:

**`ready*`** aggregates over the sample, and ten of the fourteen reject 2-D
input with `data must be 1-dimensional`. The other four do not, and silently
compute the wrong statistic:

===============  ==========================================================
likelihood       effect of passing [[1,2],[3,4]] instead of [1,2,3,4]
===============  ==========================================================
dagum            ``a`` 4.0 -> 2.0, the order of differentiation, halved
gamma            ``a`` 8.0 -> 4.0, same
inverse gamma    ``a`` 8.0 -> 4.0, same
poisson          ``b`` 4.0 -> 2.0, the evaluation point, halved
===============  ==========================================================

Halving `a` is not a small error. It is the order of the derivative, so the
package computes a different quantity entirely and reports it without comment.

**`bereit*`** returns per-observation statistics, and **none of the fourteen
rejects 2-D input**. Worse, all fourteen return `a` and `b` with *mismatched
shapes* — thirteen give `a` of shape (2,) beside `b` of shape (2, 2), and
poisson gives the mirror image. `post_predictive` consumes exactly these pairs.

The correct behaviour is to reject, at every entry point: these are functions
of a one-dimensional sample, and a 2-D array is a caller error rather than a
shape to be interpreted. The tests below therefore assert rejection
everywhere, and the cases that do not yet reject are marked as expected
failures so the suite stays green until the shared helper lands.
"""

import numpy as np
import pytest

# `pytest.raises` failing to see an exception raises Failed. The xfail markers
# below name it explicitly so that an unrelated error cannot be absorbed as the
# expected failure -- the trap that three near-integer records fell into.
from _pytest.outcomes import Failed

from jumufraktiv.MGFDerivative_class import LIKELIHOOD_REGISTRY

#: Known parameters, one entry per likelihood.
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

#: The four whose `ready*` accepts 2-D today. Named individually rather than
#: computed, so that a module gaining or losing its guard shows up as a diff.
READY_ACCEPTS_2D = {"dagum", "gamma", "inverse gamma", "poisson"}

FLAT = [1.0, 2.0, 3.0, 4.0]
NESTED = [[1.0, 2.0], [3.0, 4.0]]
COUNTS_FLAT = [1, 2, 3, 4]
COUNTS_NESTED = [[1, 2], [3, 4]]


def _data(name, nested):
    if name == "poisson":
        return COUNTS_NESTED if nested else COUNTS_FLAT
    return NESTED if nested else FLAT


def _param(name):
    """Mark a likelihood xfail where the behaviour is not yet correct."""
    if name in READY_ACCEPTS_2D:
        return pytest.param(
            name,
            marks=pytest.mark.xfail(
                strict=True,
                raises=Failed,
                reason=f"PR 7: ready{name} has no `ndim != 1` guard, so 2-D "
                f"input silently halves a statistic instead of raising",
            ),
        )
    return name


# ==========================================================================
# ready*: aggregated statistics
# ==========================================================================
@pytest.mark.parametrize("name", [_param(n) for n in sorted(KNOWN)])
def test_ready_rejects_two_dimensional_data(name):
    """A 2-D sample is a caller error, not a shape to reinterpret."""
    ready, _, _ = LIKELIHOOD_REGISTRY[name]

    with pytest.raises(ValueError, match="1"):
        ready(_data(name, nested=True), **KNOWN[name])


@pytest.mark.parametrize("name", sorted(KNOWN))
def test_ready_accepts_the_flat_equivalent(name):
    """The guard must not over-reject: the same data, flat, is fine.

    Without this, a module could pass the test above by rejecting everything.
    """
    ready, _, _ = LIKELIHOOD_REGISTRY[name]

    stats = ready(_data(name, nested=False), **KNOWN[name])

    assert all(np.isfinite(float(v)) for v in stats.values())


@pytest.mark.parametrize("name", sorted(READY_ACCEPTS_2D))
def test_the_halved_statistic_is_recorded(name):
    """Pins *what* goes wrong, so the repair is visibly a repair.

    This is not asserting that the current behaviour is right — it is
    recording the size of the error so that the commit which fixes it has
    something concrete to point at. It is removed together with the four
    xfail markers above.
    """
    ready, _, _ = LIKELIHOOD_REGISTRY[name]

    flat = ready(_data(name, nested=False), **KNOWN[name])
    nested = ready(_data(name, nested=True), **KNOWN[name])

    # Exactly one of the two statistics is halved, and which one differs by
    # likelihood: poisson sums into `b`, the other three count into `a`.
    if name == "poisson":
        assert nested["b"] == pytest.approx(flat["b"] / 2)
        assert nested["a"] == pytest.approx(flat["a"])
    else:
        assert nested["a"] == pytest.approx(flat["a"] / 2)
        assert nested["b"] == pytest.approx(flat["b"])


# ==========================================================================
# bereit*: per-observation statistics
# ==========================================================================
@pytest.mark.xfail(
    strict=True,
    raises=Failed,
    reason="PR 7: no bereit* function checks dimensionality at all",
)
@pytest.mark.parametrize("name", sorted(KNOWN))
def test_bereit_rejects_two_dimensional_data(name):
    """Not one of the fourteen rejects 2-D input."""
    _, _, bereit = LIKELIHOOD_REGISTRY[name]

    with pytest.raises(ValueError, match="1"):
        bereit(_data(name, nested=True), **KNOWN[name])


@pytest.mark.parametrize("name", sorted(KNOWN))
def test_bereit_returns_consistent_shapes_on_flat_data(name):
    """`a`, `b` and `log_c` must agree in shape. On flat data they do.

    The companion failure is what makes the 2-D case dangerous rather than
    merely wrong: on nested input all fourteen return `a` and `b` with
    *different* shapes — thirteen give `a` of shape (2,) beside `b` of shape
    (2, 2), and poisson the mirror image — and `post_predictive` consumes
    exactly that pair.
    """
    _, _, bereit = LIKELIHOOD_REGISTRY[name]

    stats = bereit(_data(name, nested=False), **KNOWN[name])
    shapes = {k: np.shape(np.asarray(v)) for k, v in stats.items()}

    assert len(set(shapes.values())) == 1, shapes
    assert shapes["a"] == (4,)


@pytest.mark.parametrize("name", sorted(KNOWN))
def test_bereit_shapes_disagree_on_two_dimensional_data(name):
    """Records the mismatch, so the repair has something concrete to remove.

    Removed together with the xfail above, in the commit that gives every
    entry point the same guard.
    """
    _, _, bereit = LIKELIHOOD_REGISTRY[name]

    stats = bereit(_data(name, nested=True), **KNOWN[name])
    a_shape = np.shape(np.asarray(stats["a"]))
    b_shape = np.shape(np.asarray(stats["b"]))

    assert a_shape != b_shape, (
        f"{name} now returns matching shapes {a_shape}; if this fails because "
        f"the module started rejecting 2-D input, delete this test along with "
        f"the xfail records above."
    )
