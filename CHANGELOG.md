# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below `1.0.0` the public API may change in any release
without a deprecation period.

## [Unreleased]

### Added

- A test suite under `tests/`: 213 passing tests plus 14 `xfail(strict=True)`
  records of known defects. Most assertions compare against closed-form
  references (the Gamma MGF and its derivatives, and the conjugate
  Gamma/Poisson posterior) rather than recorded output, and the three normative
  design principles are asserted directly with Hypothesis property tests.
- A `CI` workflow running lint, the test suite on Python 3.10–3.13, and a
  packaging check that builds both distributions, validates them with `twine`,
  and confirms a clean install imports. It runs on pull requests, which no
  previous workflow did.
- `pytest` and `ruff` configuration in `pyproject.toml`, including an itemised
  lint-debt baseline for the pre-audit library code in which every exempted
  rule is annotated with the PR that removes it.
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
- Consolidated the two documentation workflows, which shared a name, both fired
  on every push to `main`, and deployed by different mechanisms. The surviving
  workflow publishes through GitHub Pages' own deployment (the one actually
  serving the site) and additionally builds the docs on pull requests without
  publishing. The now-unused `gh-pages` branch can be deleted.
- Moved the example notebooks out of the installable package to a top-level
  `notebooks/` directory.

### Removed

- `jumufraktiv/tests/test_custom_features.py` and
  `jumufraktiv/tests/mitMGFprior_test.py`. Neither was a test: both were
  print-driven scripts with no assertions that nothing executed, and
  `test_custom_features.py` had been dead since the `MGFDerivative`
  constructor stopped accepting string priors — it could only raise
  `TypeError`. Their coverage is now provided with assertions in `tests/`.

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
