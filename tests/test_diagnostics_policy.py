"""Library code reports through ``logging`` and ``warnings``, never ``print``.

``CLAUDE.md`` has stated this since wave 0, and it was unenforceable as
written. The library made 329 ``print`` calls, but 287 of them lived inside
``if __name__ == "__main__":`` demo blocks -- code that is not library code and
that no import ever runs. A check would have been dominated by them, and the 17
that mattered were invisible in the count.

With the demo blocks gone the rule can be asserted, and the 17 are worth
naming, because they were not one kind of thing:

**Eleven were narration.** ``"Decision: Symbolic"``, ``"Trying method:
newton-jax..."``, ``"Using JAX numeric path (tuple-vectorised)..."`` -- an
account of which branch the library took, on stdout, in the middle of a
numerical routine. These are ``logger.debug`` now, or ``logger.info`` where a
``verbose`` argument already meant the caller had asked to be told.

**Five announced a silent change of algorithm.** ``jet()`` failing and falling
back to nested ``grad()`` is legitimate -- both compute the same derivatives --
but it announced itself only to stdout, so a caller capturing output saw
nothing and a caller not capturing it saw a warning glyph mid-computation.

**One announced a wrong answer.** The mpmath tan-transform route caught every
exception from its quadrature, printed the reason, and returned NaN. A NaN is
not an error: it compares False against everything, so downstream branches take
their ``else`` and the failure resurfaces somewhere unrelated, or not at all.
That one now raises ``RuntimeError`` with the evaluation point in the message
and the original exception chained.

The docstring lines beginning ``>>>`` are documentation, not diagnostics, and
are excluded.
"""

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "jumufraktiv"


def _print_calls(path: Path) -> list[int]:
    """Line numbers of real ``print(...)`` calls, ignoring docstring examples."""
    return [
        node.lineno
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]


def test_no_library_module_calls_print():
    """Parsed, not grepped: a ``>>> print(x)`` in a docstring is just a string."""
    offenders = [
        f"{path.relative_to(PACKAGE_ROOT.parent)}:{line}"
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        for line in _print_calls(path)
    ]

    assert not offenders, (
        "Library code must use `logging` or `warnings`, not `print`:\n  "
        + "\n  ".join(offenders)
    )


def test_no_library_module_has_a_main_block():
    """Demo blocks are how the print calls accumulated unnoticed in the first place."""
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.If) and "__main__" in ast.dump(node.test):
                offenders.append(f"{path.relative_to(PACKAGE_ROOT.parent)}:{node.lineno}")

    assert not offenders, (
        "`if __name__ == '__main__':` blocks are not library code and no test "
        "reaches them:\n  " + "\n  ".join(offenders)
    )


def test_the_scan_sees_a_print_that_is_really_there():
    """Guard the guard: a scan can pass because it is broken, and that looks fine."""
    planted = ast.parse('def f():\n    print("hello")\n')
    found = [
        node.lineno
        for node in ast.walk(planted)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]

    assert found == [2]


def test_a_docstring_example_is_not_counted_as_a_print():
    """The other direction: `>>> print(x)` must not trip the guard."""
    docstring_only = ast.parse('def f():\n    """Example.\n\n    >>> print(1)\n    """\n')
    found = [
        node
        for node in ast.walk(docstring_only)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]

    assert found == []
