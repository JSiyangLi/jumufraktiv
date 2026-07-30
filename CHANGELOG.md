# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below `1.0.0` the public API may change in any release
without a deprecation period.

## [Unreleased]

### Added

- `LICENSE` (MIT), `CITATION.cff` and this changelog.
- Complete PyPI distribution metadata in `pyproject.toml`: description, readme,
  `requires-python`, license, authors, keywords, classifiers and project URLs.
- Optional dependency groups: `torch`, `examples`, `docs` and `dev`.
- `jumufraktiv._version`, a dependency-free single source of truth for the
  version, re-exported as `jumufraktiv.__version__` and read by the build
  backend.
- `CLAUDE.md` describing the architecture, design principles and conventions.

### Changed

- `README.rst` is now valid reStructuredText. It was previously Markdown under
  an `.rst` extension, which rendered incorrectly on PyPI and when included by
  Sphinx.
- Corrected the README's likelihood list (fourteen are implemented, not
  thirteen — `HalfNormal` was missing) and its claim of unlimited prior
  support, which now describes the four dictionary priors and the custom-prior
  route.
- Declared the previously missing runtime dependencies `pandas` and `mpmath`,
  and added lower bounds to all runtime dependencies.
- Moved `.gitattributes` from `jumufraktiv/` to the repository root, where it
  applies to the whole tree, and extended it to cover notebooks and binaries.

### Removed

- `jumufraktiv/deprecated/` (four unreferenced modules: `evidence.py`,
  `numeric_integerDeriv_Torch.py`, `temporaty_backend_subs.py`,
  `temporaty_numeric_integerDeriv_Bell2.py`). Recoverable from git history.
- 85 build artefacts from version control that `.gitignore` already claimed to
  ignore: `__pycache__/`, `*.pyc`, `docs/_build/`, `.DS_Store` and
  `jumufraktiv.egg-info/`.
- `jumufraktiv/.vscode/settings.json`, which hardcoded a contributor's local
  conda interpreter path, and an empty `jumufraktiv/.Rhistory`.

## [0.1.0]

Initial development version.
