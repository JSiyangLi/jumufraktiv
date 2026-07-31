"""Two-dimensional input must be rejected, by every entry point.

**These tests were written before the de-duplication in this same PR, so that
it could not change this behaviour invisibly.** The fourteen `like_stats`
modules carried a byte-identical `_extract_1d`, and folding those into one
shared helper necessarily makes their validation uniform. That is the point of
the refactor — but "uniform" could mean uniformly accepting or uniformly
rejecting, and with no test either outcome ships green.

What the entry points did before the shared helper landed, measured:

**`ready*`** aggregates over the sample, and ten of the fourteen rejected 2-D
input. The other four did not, and silently computed the wrong statistic:

===============  ==========================================================
likelihood       effect of passing [[1,2],[3,4]] instead of [1,2,3,4]
===============  ==========================================================
dagum            ``a`` 4.0 -> 2.0, the order of differentiation, halved
gamma            ``a`` 8.0 -> 4.0, same
inverse gamma    ``a`` 8.0 -> 4.0, same
poisson          ``b`` 4.0 -> 2.0, the evaluation point, halved
===============  ==========================================================

Halving `a` is not a small error. It is the order of the derivative, so the
package computed a different quantity entirely and reported it without comment.

**`bereit*`** returns per-observation statistics, and **none of the fourteen
rejected 2-D input**. Worse, all fourteen returned `a` and `b` with *mismatched
shapes* — thirteen gave `a` of shape (2,) beside `b` of shape (2, 2), and
poisson the mirror image. `post_predictive` consumes exactly these pairs.

Both are now fixed by construction rather than by agreement: the check lives
inside the one shared `_extract_1d`, which every entry point routes its data
through, so there is no longer a set of fourteen guards that could drift apart.
"""

import numpy as np
import pytest

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

FLAT = [1.0, 2.0, 3.0, 4.0]
NESTED = [[1.0, 2.0], [3.0, 4.0]]
COUNTS_FLAT = [1, 2, 3, 4]
COUNTS_NESTED = [[1, 2], [3, 4]]


def _data(name, nested):
    if name == "poisson":
        return COUNTS_NESTED if nested else COUNTS_FLAT
    return NESTED if nested else FLAT


# ==========================================================================
# ready*: aggregated statistics
# ==========================================================================
@pytest.mark.parametrize("name", sorted(KNOWN))
def test_ready_rejects_two_dimensional_data(name):
    """A 2-D sample is a caller error, not a shape to reinterpret."""
    ready, _, _ = LIKELIHOOD_REGISTRY[name]

    with pytest.raises(ValueError, match="1-dimensional"):
        ready(_data(name, nested=True), **KNOWN[name])


@pytest.mark.parametrize("name", sorted(KNOWN))
def test_ready_accepts_the_flat_equivalent(name):
    """The guard must not over-reject: the same data, flat, is fine.

    Without this, a module could pass the test above by rejecting everything.
    """
    ready, _, _ = LIKELIHOOD_REGISTRY[name]

    stats = ready(_data(name, nested=False), **KNOWN[name])

    assert all(np.isfinite(float(v)) for v in stats.values())


# ==========================================================================
# bereit*: per-observation statistics
# ==========================================================================
@pytest.mark.parametrize("name", sorted(KNOWN))
def test_bereit_rejects_two_dimensional_data(name):
    """Not one of the fourteen used to reject 2-D input; all fourteen now do."""
    _, _, bereit = LIKELIHOOD_REGISTRY[name]

    with pytest.raises(ValueError, match="1-dimensional"):
        bereit(_data(name, nested=True), **KNOWN[name])


@pytest.mark.parametrize("name", sorted(KNOWN))
def test_bereit_returns_consistent_shapes_on_flat_data(name):
    """`a`, `b` and `log_c` must agree in shape.

    The mismatch this guards against is what made the 2-D case dangerous
    rather than merely wrong: `post_predictive` consumes the `a`/`b` pair, so
    two different shapes reach a computation that assumes one.
    """
    _, _, bereit = LIKELIHOOD_REGISTRY[name]

    stats = bereit(_data(name, nested=False), **KNOWN[name])
    shapes = {k: np.shape(np.asarray(v)) for k, v in stats.items()}

    assert len(set(shapes.values())) == 1, shapes
    assert shapes["a"] == (4,)


# ==========================================================================
# The known parameters go through the same helper
# ==========================================================================
@pytest.mark.parametrize(
    "name", sorted(n for n in KNOWN if KNOWN[n] and n not in {"dagum"})
)
def test_a_two_dimensional_known_parameter_is_rejected(name):
    """`_extract_1d` guards the known parameters too, not only the data.

    Each module extracts its known parameter through the same helper, so the
    dimensionality check applies there as well. The message differs -- most
    modules catch the failure and re-raise their own "must be a numeric scalar
    or 1-dimensional" -- so this asserts only that a `ValueError` is raised and
    mentions dimensionality, not the exact wording.

    Dagum is excluded because it takes two known parameters whose validation
    is interleaved; it is covered by the data tests above.

    The match is on "dimensional" alone rather than "1-dimensional" because the
    modules' own re-raised messages spell it with a non-breaking hyphen, which
    a plain hyphen would not match. That spelling is pre-existing lint debt
    owned by the docstring sweep; it is not this test's to fix, but it is this
    test's to avoid depending on.
    """
    ready, _, _ = LIKELIHOOD_REGISTRY[name]
    key = next(iter(KNOWN[name]))
    bad = dict(KNOWN[name])
    bad[key] = [[1.0, 2.0], [3.0, 4.0]]

    with pytest.raises(ValueError, match="dimensional"):
        ready(_data(name, nested=False), **bad)


# ==========================================================================
# The de-duplication's own invariant
# ==========================================================================
def test_no_module_redefines_the_shared_helpers():
    """The fourteen copies must not come back.

    The helpers were byte-identical across all fourteen modules, which is how
    their validation managed to diverge in the first place: ten callers guarded
    dimensionality and four did not, and nothing made that visible. Keeping the
    definitions in one place is what makes the guard uniform by construction,
    so a module reintroducing its own copy is a regression rather than a style
    question.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "jumufraktiv" / "like_stats"
    offenders = []
    for path in sorted(root.glob("*.py")):
        if path.name == "_common.py":
            continue
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in (
                "_extract_1d",
                "_is_1d_dataframe",
            ):
                offenders.append(f"{path.name}:{node.name}")

    assert offenders == [], f"helpers redefined locally: {offenders}"
