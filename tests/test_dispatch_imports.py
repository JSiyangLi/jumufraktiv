"""Import-integrity tests for the dispatcher's lazily-imported backends.

`derivativeDispatch` imports three backends inside function bodies rather than
at module scope. Two of those imports were written without the package prefix,
so they resolved only when the package directory happened to be on `sys.path`
and raised `ModuleNotFoundError` under any normal installation. Nothing caught
it because the affected paths had no coverage.

These tests exercise each lazy import. They assert reachability, not numerical
correctness — the fractional backends have accuracy defects tracked separately
in `test_known_broken.py`.
"""

import importlib

import pytest

from jumufraktiv.derivativeDispatch import mgfDerivative

#: Modules the dispatcher imports lazily, by fully qualified name.
LAZY_BACKENDS = [
    "jumufraktiv.symbolic_fractionalDeriv",
    "jumufraktiv.numeric_fractionalDeriv_interpolation",
    "jumufraktiv.numeric_fractionalDeriv_scipy",
    "jumufraktiv.numeric_fractionalDeriv_mpmath",
]


@pytest.mark.parametrize("module_name", LAZY_BACKENDS)
def test_lazy_backend_is_importable(module_name):
    """Every backend the dispatcher can reach for must actually import."""
    assert importlib.import_module(module_name) is not None


def test_no_unqualified_intra_package_imports():
    """Intra-package imports must be fully qualified.

    A bare ``from symbolic_fractionalDeriv import ...`` resolves when the package
    directory is on ``sys.path`` — which is true when running from a checkout,
    and false for an installed package. That difference is why these survived.
    """
    import re
    from pathlib import Path

    package_root = Path(__file__).resolve().parent.parent / "jumufraktiv"
    # Module basenames that only resolve unqualified.
    siblings = "|".join(
        p.stem for p in package_root.glob("*.py") if p.stem != "__init__"
    )
    pattern = re.compile(rf"^\s*(?:from|import)\s+({siblings})\b", re.MULTILINE)

    offenders = []
    for path in package_root.rglob("*.py"):
        source = path.read_text()
        for match in pattern.finditer(source):
            line_number = source.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(package_root.parent)}:{line_number}")

    assert not offenders, (
        "unqualified intra-package imports resolve only from a checkout, not an "
        f"installed package: {offenders}"
    )


def test_symbolic_fractional_backend_is_reachable(gamma_prior):
    """The symbolic fractional path must run rather than raise ImportError."""
    with pytest.warns(UserWarning, match="can be very slow"):
        mgfDerivative(1.5, gamma_prior, method="symbolic", t=None)


@pytest.mark.slow
def test_interpolation_backend_is_reachable(gamma_prior):
    """Near-integer orders trigger the interpolation backend; it must import.

    **This is the only test in the suite that goes red if that lazy import
    breaks**, and that makes its assertions load-bearing. The three
    near-integer records in `test_known_broken.py` cannot cover it: they are
    expected failures, so an `ImportError` counts as the failure they expect
    and the suite stays green. Verified — under a simulated broken import they
    give `5 passed, 3 xfailed` while this test gives `1 failed`.

    It used to assert only `sign in (-1, 1)` and `log_abs == log_abs`, both
    trivially true. Measured: inflating the backend's answer by a factor of
    1e6 left the **entire suite** green.

    So it now carries an accuracy floor as well. The backend's relative error
    at this order is 1.094e-05 against the closed form, so `rel=1e-4` passes
    with about 9x headroom while still catching any gross corruption.

    The floor is deliberately loose and deliberately temporary. It is not a
    statement that 1e-05 is acceptable — `test_known_broken.py` records the
    near-integer inaccuracy as a defect, and CLAUDE.md schedules this module
    for retirement. Until then the module is on a live user path (any order
    whose fractional part exceeds 0.95), so it needs *some* floor rather than
    none.
    """
    from conftest import gamma_mgf_derivative_log

    log_abs, sign = mgfDerivative(1.96, gamma_prior, method="scipy", t=-1.0, log=True)

    assert sign == 1
    assert log_abs == pytest.approx(gamma_mgf_derivative_log(1.96, -1.0), rel=1e-4)
