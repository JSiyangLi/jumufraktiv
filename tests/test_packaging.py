"""What the built distributions actually contain.

The wheel and the sdist are different artefacts with different jobs, and the
same inclusion rules cannot serve both. The wheel is installed into
site-packages, so `tests` and `docs` must stay out of it. The sdist is what
conda-forge, Debian and anyone building from source download, and the first
thing they do with it is run the test suite.

Getting that backwards is not hypothetical here. With no ``MANIFEST.in``,
setuptools guessed: it collected ``tests/test_*.py`` and left out
``tests/conftest.py`` and ``tests/canonical.py``. That is the worst of the
available outcomes --- the sdist appeared to ship a test suite, and the suite
could not collect a single test. Fourteen collection errors, one per file that
imports a fixture.

Building a distribution takes a few seconds, so these are `slow`-marked and run
in the full suite rather than the quick pass.
"""

import pathlib
import re
import subprocess
import sys
import tarfile
import zipfile

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

#: Files a packager or a citing user reads, which live outside the importable
#: package and so are in neither the wheel nor the sdist unless asked for.
SDIST_METADATA = [
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.rst",
    "pyproject.toml",
]

#: Without these two the other twenty-six test files cannot even be collected.
SDIST_TEST_HARNESS = ["conftest.py", "tests/conftest.py", "tests/canonical.py"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Build both distributions once, into a temporary directory."""
    out = tmp_path_factory.mktemp("dist")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(out)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        pytest.skip(f"`python -m build` unavailable: {result.stderr[-400:]}")

    sdists = list(out.glob("*.tar.gz"))
    wheels = list(out.glob("*.whl"))
    assert len(sdists) == 1 and len(wheels) == 1
    return sdists[0], wheels[0]


def _sdist_names(path):
    with tarfile.open(path) as archive:
        # Strip the `jumufraktiv-0.1.0/` prefix every sdist entry carries.
        return {name.partition("/")[2] for name in archive.getnames()}


@pytest.mark.slow
@pytest.mark.parametrize("name", SDIST_METADATA + SDIST_TEST_HARNESS)
def test_the_sdist_carries_what_a_packager_needs(built, name):
    sdist, _ = built

    assert name in _sdist_names(sdist), (
        f"'{name}' is missing from the sdist. MANIFEST.in decides this; the "
        f"wheel's `packages.find` rules do not apply to it."
    )


@pytest.mark.slow
def test_the_sdist_test_suite_can_be_collected(built, tmp_path):
    """The property the file names cannot express.

    Shipping every test file is not the same as shipping a runnable suite. This
    unpacks the sdist and asks pytest to collect it, which is what a packager's
    build check does.
    """
    sdist, _ = built
    with tarfile.open(sdist) as archive:
        archive.extractall(tmp_path, filter="data")
    root = next(tmp_path.glob("jumufraktiv-*"))

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-header"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=900,
    )

    # pytest exits non-zero on a collection error, which is the whole signal.
    # Searching stdout for "error" would instead match the several tests whose
    # own names contain the word.
    assert result.returncode == 0, (
        f"the sdist's test suite does not collect:\n{result.stdout[-3000:]}"
    )

    collected = re.search(r"(\d+) tests? collected", result.stdout)
    assert collected, f"no collection summary:\n{result.stdout[-2000:]}"
    assert int(collected.group(1)) > 500, (
        f"only {collected.group(1)} tests collected from the sdist; the suite "
        f"is over a thousand, so files are missing"
    )


@pytest.mark.slow
def test_the_wheel_installs_only_the_package(built):
    """`tests` and `docs` in site-packages would be a namespace collision.

    `tests` especially: it is a name many projects use, and installing one at
    the top level of site-packages shadows every other.
    """
    _, wheel = built
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()

    stray = [
        name for name in names if not name.startswith(("jumufraktiv/", "jumufraktiv-"))
    ]
    assert not stray, f"the wheel ships files outside the package: {stray}"

    assert not [n for n in names if n.startswith(("tests/", "docs/"))]


@pytest.mark.slow
def test_the_wheel_ships_exactly_the_source_modules(built):
    """A module deleted from the tree must not survive in the wheel.

    Stale modules in a wheel are invisible locally --- an editable install
    reads the tree --- and reachable for anyone who installs from PyPI.
    """
    _, wheel = built
    with zipfile.ZipFile(wheel) as archive:
        shipped = {
            name[len("jumufraktiv/") :]
            for name in archive.namelist()
            if name.startswith("jumufraktiv/") and name.endswith(".py")
        }

    on_disk = {
        str(path.relative_to(REPO / "jumufraktiv"))
        for path in (REPO / "jumufraktiv").rglob("*.py")
        if "__pycache__" not in path.parts
    }

    assert shipped == on_disk, (
        f"only in the wheel: {sorted(shipped - on_disk)}\n"
        f"only on disk: {sorted(on_disk - shipped)}"
    )


def test_the_citation_file_validates_against_the_cff_schema():
    """GitHub's "Cite this repository" widget parses this, and so does Zenodo.

    `journal: [manuscript in preparation]` is YAML flow-sequence syntax, so the
    field parsed as a one-element list where the schema requires a string. The
    file looked fine and every consumer of it rejected the file.
    """
    cffconvert = pytest.importorskip("cffconvert")

    citation = cffconvert.Citation((REPO / "CITATION.cff").read_text())
    citation.validate()
