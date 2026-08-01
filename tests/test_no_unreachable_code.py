"""Every module-level function in the library must be reachable.

Twenty-five were not, spread across seven files and 454 lines, and they had
accumulated quietly because nothing looks for them. Two kinds mattered more
than the line count:

**Duplicates that had not diverged yet.** Thirteen ``*_symbolic`` wrappers in
the four prior modules returned exactly the expression their ``*_factory``
writes out inline. Checked symbolically before deleting them, every pair
agreed -- so nothing was *wrong*, and that is the whole hazard: an edit to the
factory would have left the wrapper stating the old MGF, with no test able to
tell, because no test could reach the wrapper.

**A duplicate that had.** ``gamma_cgf`` and ``gamma_mgf`` used ``logminus`` as
though it meant "log minus" rather than "log of a difference", and returned
``-inf`` and ``0.0`` at alpha=2, beta=3, t=-1 where the true values are
-0.575 and 0.5625. ``gamma_factory`` wires correct inline lambdas, so the package never
called them -- but ``from ...gammaMGF import gamma_mgf`` was an import away,
and the module's own docstring advertised exactly that import, in an example
whose printed value was also wrong and whose call signature would have raised.
Unreachable code is not harmless code; it is code no one is checking.

This test is a structural guard, not a style rule. It fails on a function that
nothing can reach -- which is either dead code to remove, or a wiring mistake
that left a new function disconnected. Both are worth a red build.
"""

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "jumufraktiv"
TESTS_ROOT = Path(__file__).resolve().parent

#: Functions that are unreachable by static analysis but reachable in fact,
#: each with the mechanism that reaches it. Kept explicit so that adding to it
#: is a decision rather than a side effect.
ALLOWED_UNREACHABLE: dict[str, str] = {
    # Nothing here at present. A prior factory is *not* an entry: it is bound
    # by the `@register_prior` decorator, which the collector below counts.
}


class _NameCollector(ast.NodeVisitor):
    """Every name read, attribute accessed, imported, or bound by a decorator."""

    def __init__(self) -> None:
        self.loaded: set[str] = set()
        self.decorated: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loaded.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # `module.func(...)` reaches `func` without ever binding the bare name.
        self.loaded.add(node.attr)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.loaded.add(alias.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # A decorated function is reached through whatever registry its
        # decorator writes to, so nothing needs to name it.
        if node.decorator_list:
            self.decorated.add(node.name)
        self.generic_visit(node)


def _sources() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py")) + sorted(TESTS_ROOT.rglob("*.py"))


@pytest.fixture(scope="module")
def reachable_names() -> set[str]:
    """Names reachable from anywhere in the package or the suite."""
    collector = _NameCollector()
    for path in _sources():
        collector.visit(ast.parse(path.read_text()))
    return collector.loaded | collector.decorated


def test_no_module_level_function_is_unreachable(reachable_names):
    """A function nothing can name, and no decorator registers, is dead."""
    orphans = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name in reachable_names or node.name in ALLOWED_UNREACHABLE:
                continue
            orphans.append(
                f"{path.relative_to(PACKAGE_ROOT.parent)}:{node.lineno} "
                f"{node.name} ({node.end_lineno - node.lineno + 1} lines)"
            )

    assert not orphans, (
        "Unreachable module-level functions:\n  "
        + "\n  ".join(orphans)
        + "\n\nEither delete them, wire them up, or add them to "
        "ALLOWED_UNREACHABLE with the mechanism that reaches them."
    )


def test_the_guard_can_actually_fail(tmp_path, reachable_names):
    """The guard must detect an orphan, not merely pass over a clean tree.

    A structural test that scans for a condition can pass because its scan is
    broken rather than because the condition is absent, and that failure is
    invisible -- it looks exactly like success. So plant one.
    """
    planted = tmp_path / "orphan_module.py"
    planted.write_text("def a_function_nothing_calls():\n    return 1\n")

    tree = ast.parse(planted.read_text())
    names = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name not in reachable_names
    ]

    assert names == ["a_function_nothing_calls"]
