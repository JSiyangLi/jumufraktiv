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


#: A NumPyDoc section heading plus its underline, at any indentation. Only the
#: two napoleon turns into directives are listed: `Attributes` becomes
#: ``.. attribute::`` and `Methods` becomes ``.. method::``. `Parameters`,
#: `Returns`, `Raises` and the rest become field lists, which register nothing
#: and therefore cannot collide.
_DIRECTIVE_SECTION = re.compile(
    r"^([ \t]*)(Attributes|Methods)[ \t]*\n[ \t]*-{4,}[ \t]*$", re.M
)


def _section_entries(docstring, heading):
    """The names a NumPyDoc `Attributes` or `Methods` section lists.

    An entry is a line at the section's own indentation; `name : type` and
    `name(args)` both reduce to `name`. Lines indented further are the
    descriptions, and a blank line at that indentation ends the section.
    """
    match = _DIRECTIVE_SECTION.search(docstring)
    while match and match.group(2) != heading:
        match = _DIRECTIVE_SECTION.search(docstring, match.end())
    if match is None:
        return []

    indent = len(match.group(1))
    names = []
    for line in docstring[match.end() :].splitlines()[1:]:
        if not line.strip():
            break
        if len(line) - len(line.lstrip()) != indent:
            continue  # a description belonging to the entry above
        names.append(re.split(r"[ (:]", line.strip(), maxsplit=1)[0])
    return names


@pytest.mark.parametrize("heading", ["Attributes", "Methods"])
def test_no_documented_class_lists_a_name_autodoc_can_see(heading):
    """Napoleon's `Attributes` and `Methods` sections register objects too.

    Sphinx builds the API reference from `autoclass` with `:members:`, which
    emits one directive per member. Napoleon renders these two NumPyDoc
    sections into `.. attribute::` and `.. method::` directives of their own,
    so any name appearing in both is described twice and Sphinx warns.

    Twenty-five names did. The rendered page also gained two attributes named
    `Properties` and `----------`, because a heading napoleon does not know
    gets absorbed into the preceding `Attributes` section as further entries.

    A section entry for a name autodoc *cannot* see is fine and is why this
    test asks about `dir()` rather than banning the sections outright:
    `MGFDerivative` assigns seven attributes in `__init__` with no class-level
    declaration, so its `Attributes` section is their only documentation.

    Duplication is not the whole cost. A hand-written `Methods` table repeats
    signatures that autodoc reads from the code, so it drifts silently: six of
    the twelve listed here had, `post_sample` having never learned about the
    `rng` argument that made it reproducible.
    """
    import importlib
    import inspect

    source = (REPO / "docs" / "api.rst").read_text(encoding="utf-8")
    classes = re.findall(r"^\.\. autoclass:: (\S+)$", source, flags=re.M)
    assert classes, "docs/api.rst declares no autoclass targets"

    package = importlib.import_module("jumufraktiv")
    offenders = []
    for name in classes:
        cls = getattr(package, name.rpartition(".")[2])
        doc = inspect.getdoc(cls) or ""
        visible = set(dir(cls))
        offenders += [
            f"{name}.{entry}"
            for entry in _section_entries(doc, heading)
            if entry in visible
        ]

    assert not offenders, (
        f"the '{heading}' section of a class docstring lists "
        f"{len(offenders)} name(s) that autodoc also documents, so Sphinx "
        f"describes each twice: {', '.join(offenders)}. Document these on the "
        "member itself -- a method's own docstring, or a `#:` comment above a "
        "class-level attribute."
    )


#: The NumPyDoc sections that end a `Parameters` block when one follows it.
_NUMPYDOC_SECTIONS = frozenset(
    {
        "Parameters",
        "Returns",
        "Yields",
        "Raises",
        "Warns",
        "Notes",
        "Examples",
        "See Also",
        "References",
        "Attributes",
        "Methods",
        "Other Parameters",
    }
)


def _documented_parameters(docstring):
    """The names a `Parameters` section lists, in the order it lists them.

    Returns `None` when there is no such section, which is different from an
    empty one and must not be treated as a violation.
    """
    lines = docstring.splitlines()
    start = indent = None
    for i, line in enumerate(lines[:-1]):
        if line.strip() == "Parameters" and set(lines[i + 1].strip()) == {"-"}:
            start, indent = i + 2, len(line) - len(line.lstrip())
            break
    if start is None:
        return None

    names = []
    for j in range(start, len(lines)):
        line = lines[j]
        if not line.strip():
            continue  # NumPyDoc allows blank lines between entries
        if (
            line.strip() in _NUMPYDOC_SECTIONS
            and j + 1 < len(lines)
            and set(lines[j + 1].strip()) == {"-"}
        ):
            break
        depth = len(line) - len(line.lstrip())
        if depth < indent:
            break
        if depth != indent:
            continue  # the description of the entry above
        # `a, b : int` documents two parameters on one line; `*args` and
        # `**kwargs` are documented without their stars.
        for name in line.strip().split(":")[0].split(","):
            name = name.strip().lstrip("*")
            if name:
                names.append(name)
    return names


def _public_callables():
    """Every public function and method the package defines, deduplicated."""
    import importlib
    import inspect
    import pkgutil

    import jumufraktiv

    modules = [jumufraktiv]
    for info in pkgutil.walk_packages(jumufraktiv.__path__, "jumufraktiv."):
        try:
            modules.append(importlib.import_module(info.name))
        except Exception:
            # An optional backend may be absent; a module that cannot be
            # imported simply has no docstrings for this test to check.
            continue

    seen = {}
    for module in modules:
        for attr, obj in vars(module).items():
            if attr.startswith("_"):
                continue
            found = []
            if inspect.isfunction(obj) and obj.__module__.startswith("jumufraktiv"):
                found = [(f"{obj.__module__}.{obj.__qualname__}", obj)]
            elif inspect.isclass(obj) and obj.__module__.startswith("jumufraktiv"):
                found = [
                    (f"{obj.__module__}.{obj.__qualname__}.{name}", member)
                    for name, member in vars(obj).items()
                    if inspect.isfunction(member) and not name.startswith("_")
                ]
            for full, fn in found:
                seen.setdefault(full, fn)
    return seen


def test_documented_parameters_are_in_signature_order():
    """A `Parameters` section out of order is how a signature change hides.

    Reading a docstring against its signature is the cheapest check there is,
    and it only works if the two can be read side by side. When the orders
    agree, an added or removed parameter shows up as a single misalignment;
    when they do not, every line has to be matched by name and an omission
    looks like nothing at all.

    Order is compared over the names common to both, so documenting a subset
    is allowed -- this asks whether the documented ones appear in the order
    the signature declares them, not whether all of them appear.
    """
    import inspect

    offenders = []
    for full, fn in sorted(_public_callables().items()):
        documented = _documented_parameters(inspect.getdoc(fn) or "")
        if not documented:
            continue
        try:
            real = [
                p for p in inspect.signature(fn).parameters if p not in ("self", "cls")
            ]
        except (ValueError, TypeError):
            continue  # a builtin or C-implemented callable has no signature
        in_doc_order = [name for name in documented if name in real]
        in_sig_order = [name for name in real if name in documented]
        if in_doc_order != in_sig_order:
            offenders.append(
                f"{full}\n   documented: {in_doc_order}\n   signature:  {in_sig_order}"
            )

    assert not offenders, (
        f"{len(offenders)} callable(s) document their parameters in an order "
        "the signature does not declare them in:\n" + "\n".join(offenders)
    )


def test_the_documentation_toctree_resolves():
    """A toctree entry with no document is a dead link in the sidebar."""
    index = (REPO / "docs" / "index.rst").read_text(encoding="utf-8")
    block = re.search(r"\.\. toctree::\n(?:\s+:\w+:.*\n)*\n((?:\s+\S+\n)+)", index)

    assert block, "docs/index.rst has no toctree"

    for name in block.group(1).split():
        assert (REPO / "docs" / f"{name}.rst").exists(), (
            f"toctree references '{name}', which does not exist"
        )
