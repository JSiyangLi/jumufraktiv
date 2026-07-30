"""Single source of truth for the package version.

`pyproject.toml` reads ``__version__`` from this module via setuptools'
``dynamic.version`` mechanism, and :mod:`jumufraktiv` re-exports it. Keeping the
literal in a dependency-free module means the build backend can read it by
static analysis, without importing NumPy, SymPy or JAX.

Update this value (and ``CHANGELOG.md``) when cutting a release.
"""

__version__ = "0.1.0"
