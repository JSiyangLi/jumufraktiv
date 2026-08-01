"""The documentation must run, not merely render.

Every defect this file guards against shipped, and shipped for one reason: the
suite checked the library and nothing checked the documentation. The README's
quick start -- the PyPI landing page and the Sphinx front page, the first six
lines of code a new user runs -- raised ``TypeError`` for a release, because
``evidence()`` changed shape and the README did not. `twine check` renders the
README but never executes it, so CI was green throughout.

The lesson is the one the audit keeps relearning in different subsystems: a
claim nobody runs is indistinguishable from a claim that is false. These tests
run them.
"""

import ast
import io
import json
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
README = REPO / "README.rst"
NOTEBOOKS = sorted((REPO / "notebooks").glob("*.ipynb"))

#: Names removed from the public API, mapped to what to use instead. A shipped
#: example mentioning one of these is stale by construction, whatever it
#: evaluates to.
RETIRED = {
    r"\w+\s*,\s*\w+\s*=\s*[\w.]+\.evidence\(\)": (
        "evidence() returns the log alone; only post_central_moment returns a pair"
    ),
    r"\bepsrel\s*=": "epsrel was removed with the adaptive kernel it tuned",
    r"\bregister_likelihood\b": "there is no runtime API for registering a likelihood",
    r"\bintegerDeriv_method\s*=": "renamed to integer_method",
    r"\bimport\s+mgf2post\b": "the mgf2post alias was removed",
}


def _readme_python_blocks():
    """Every ``.. code-block:: python`` body in the README, dedented."""
    source = README.read_text(encoding="utf-8")
    blocks = []
    for match in re.finditer(
        r"\.\. code-block:: python\n\n((?:(?: {3,}.*)?\n)+)", source
    ):
        body = match.group(1)
        lines = [
            line[3:] if line.startswith("   ") else line for line in body.split("\n")
        ]
        blocks.append("\n".join(lines))
    return blocks


def test_the_readme_has_a_runnable_example_at_all():
    """A quick start nobody can run is worse than none, and this file assumes one."""
    assert _readme_python_blocks(), "README.rst contains no python code-block"


@pytest.mark.parametrize("index", range(len(_readme_python_blocks())))
def test_every_readme_example_runs(index, tmp_path):
    """Run it in a subprocess from an empty directory, exactly as a reader would.

    In-process would let the test suite's own imports and fixtures stand in for
    setup the reader does not have. The empty working directory matters
    separately: ``python -c`` puts the current directory first on ``sys.path``,
    so running from the repository root would import the source tree even when
    the point of the run is to check an installed wheel.
    """
    code = _readme_python_blocks()[index]
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=600,
    )

    assert result.returncode == 0, (
        f"README example {index + 1} failed:\n{result.stderr[-2000:]}"
    )


@pytest.mark.parametrize("pattern,reason", sorted(RETIRED.items()))
def test_the_readme_does_not_use_a_retired_api(pattern, reason):
    assert not re.search(pattern, README.read_text(encoding="utf-8")), reason


def test_the_readme_renders():
    """PyPI renders reStructuredText strictly, and shows a raw dump on error.

    A malformed table is the easy way to get there: docutils rejects a simple
    table whose cells overrun their column markers, and the failure is invisible
    to anyone reading the source, where the columns still look aligned.
    """
    from docutils.core import publish_doctree

    problems = io.StringIO()
    publish_doctree(
        README.read_text(encoding="utf-8"),
        settings_overrides={
            "report_level": 2,  # warnings and above
            "halt_level": 5,  # collect them all rather than raising on the first
            "warning_stream": problems,
        },
    )

    reported = problems.getvalue()
    assert not reported, f"README.rst does not render:\n{reported}"


def test_the_readme_likelihood_table_matches_the_registry():
    """The table lists every likelihood and its known parameters.

    Both halves have been wrong in shipped prose: the list omitted likelihoods,
    and it named parameters the functions do not take. Neither is visible
    without comparing against the registry, which is what this does.
    """
    from jumufraktiv.MGFDerivative_class import (
        LIKELIHOOD_REGISTRY,
        _likelihood_kwargs,
    )

    source = README.read_text(encoding="utf-8")

    for name, entry in sorted(LIKELIHOOD_REGISTRY.items()):
        cell = f'``"{name}"``'
        assert cell in source, (
            f"likelihood '{name}' is registered but not listed in README.rst"
        )

        row = next(ln for ln in source.splitlines() if ln.strip().startswith(cell))
        ready_func = entry[0]  # (ready, c, bereit)
        for keyword in sorted(_likelihood_kwargs(ready_func)):
            assert f"``{keyword}``" in row, (
                f"README row for '{name}' does not name its known parameter "
                f"'{keyword}':\n{row}"
            )


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
@pytest.mark.parametrize("pattern,reason", sorted(RETIRED.items()))
def test_no_notebook_uses_a_retired_api(notebook, pattern, reason):
    """Static, because executing these takes many minutes.

    It is a weaker check than running them and it is the one that catches the
    failure mode that actually occurred: an API changed shape underneath a
    shipped example. Both notebooks carried seven such calls and died on their
    first substantive cell.
    """
    document = json.loads(notebook.read_text(encoding="utf-8"))
    sources = [
        "".join(cell["source"])
        for cell in document["cells"]
        if cell["cell_type"] == "code"
    ]

    offenders = [s for s in sources if re.search(pattern, s)]
    assert not offenders, f"{notebook.name}: {reason}\n{offenders[0][:400]}"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_sources_parse(notebook):
    """A cell that is not valid Python cannot run, and need not be executed to know."""
    document = json.loads(notebook.read_text(encoding="utf-8"))

    for index, cell in enumerate(document["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # IPython line and cell magics are not Python and are legitimate here.
        if re.match(r"\s*[%!]", source):
            continue
        try:
            ast.parse(source)
        except SyntaxError as error:
            pytest.fail(f"{notebook.name} cell {index} does not parse: {error}")


@pytest.mark.slow
def test_the_docstring_examples_run():
    """This is the only thing that runs them, and the harness must be wired.

    ``--doctest-modules`` collects from ``jumufraktiv/``, so it does not load
    ``tests/conftest.py`` and cannot see the ``deriv`` and ``prior`` the
    examples are written against. The fixture therefore lives in a
    repository-root ``conftest.py``, which is an ancestor of both. Nothing else
    checks that arrangement: the suite passes with the fixture in either place,
    and only this command fails when it is in the wrong one.

    Run in a subprocess rather than by collecting the doctests into this
    session, because the wiring is the thing under test: collected in-process
    they would find the fixture whatever its location.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--doctest-modules",
            "jumufraktiv/MGFDerivative_class.py",
            "jumufraktiv/mitMGFprior_class.py",
            "-q",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=600,
    )

    assert result.returncode == 0, (
        f"the docstring examples do not run as CI invokes them:\n"
        f"{result.stdout[-3000:]}"
    )


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_carry_no_stored_output(notebook):
    """Committed output goes stale silently, and leaks whoever ran it.

    One notebook's stored tracebacks carried a developer's home directory in
    five places, and a warning about "the analytic continuation to non-integer
    orders" that the package stopped emitting several releases ago. Neither is
    visible to anyone reading the source; both are visible to anyone opening
    the notebook.
    """
    document = json.loads(notebook.read_text(encoding="utf-8"))

    with_output = [
        index
        for index, cell in enumerate(document["cells"])
        if cell["cell_type"] == "code" and cell.get("outputs")
    ]

    assert not with_output, (
        f"{notebook.name} has stored output in cells {with_output}; "
        "clear it before committing"
    )


def test_every_api_reference_module_is_importable():
    """`automodule` needs the qualified name, and says nothing when it does not.

    The directives named modules bare -- `derivativeDispatch` rather than
    `jumufraktiv.derivativeDispatch` -- so autodoc could not import them and
    emitted nothing. The build still exited zero, and the rendered API
    reference covered two of the package's modules while appearing to list all
    of them. Importing each target here is the cheap half of what Sphinx does,
    and it is the half that failed.
    """
    import importlib

    source = (REPO / "docs" / "api.rst").read_text(encoding="utf-8")
    targets = re.findall(r"^\.\. automodule:: (\S+)$", source, flags=re.M)

    assert targets, "docs/api.rst declares no automodule targets"

    for target in targets:
        assert target.startswith("jumufraktiv"), (
            f"'{target}' is not qualified; autodoc will silently document nothing"
        )
        importlib.import_module(target)


def test_the_documentation_toctree_resolves():
    """A toctree entry with no document is a dead link in the sidebar."""
    index = (REPO / "docs" / "index.rst").read_text(encoding="utf-8")
    block = re.search(r"\.\. toctree::\n(?:\s+:\w+:.*\n)*\n((?:\s+\S+\n)+)", index)

    assert block, "docs/index.rst has no toctree"

    for name in block.group(1).split():
        assert (REPO / "docs" / f"{name}.rst").exists(), (
            f"toctree references '{name}', which does not exist"
        )
