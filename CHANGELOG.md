# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below `1.0.0` the public API may change in any release
without a deprecation period.

## [Unreleased]

### Fixed

- `mitMGFprior.from_registry` now initialises the registry. It previously read
  `PRIOR_REGISTRY` directly, so in a fresh process it raised
  `Unknown prior 'gamma'` — the registry was simply empty — unless some other
  registry function happened to have run first.
- Prior discovery is isolated per module. `MGFdictionary/paretoMGF.py` imported
  `torch` at module scope and the discovery loop aborted on the first module
  that raised, so a missing optional extra silently removed both `pareto` *and*
  `uniform` from the registry, leaving a warning in place of half the priors.
  The `torch` import is now lazy — nothing in the package referenced the
  function it supported — and one module's failure no longer affects the rest.
- Two intra-package imports in `derivativeDispatch.py` were written without the
  package prefix, so they resolved only when the package directory happened to
  be on `sys.path` and raised `ModuleNotFoundError` under a normal install.
  This made the symbolic fractional backend and the near-integer interpolation
  path unreachable.
- `symbolic_fractionalDeriv` no longer depends on `func_timeout`, which is
  unbuildable against current setuptools (its `setup.py` reads the removed
  `install_layout` attribute). The module raised `ImportError` on import as a
  result, making the symbolic fractional backend unreachable for a second,
  independent reason. Replaced with a small `concurrent.futures` equivalent.

### Changed

- Registry errors are now actionable. A prior that is absent because its module
  failed to import says so, naming the module and the exception, rather than
  reporting an unqualified "not found". `registry.failed_prior_modules()`
  exposes the same information programmatically.
- `registry.initialize` no longer downgrades a failure of the `MGFdictionary`
  subpackage itself to a warning. That is an installation fault no caller can
  work around, and swallowing it produced a silently empty registry.

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
- Citation of the paper this package implements — Li, van Dyk & Autenrieth,
  "Using fractional derivatives to derive marginal densities", Biometrika
  (2026), arXiv:2409.11167 — in `CITATION.cff`, the README and `CLAUDE.md`. The
  repository previously cited it nowhere.
- A statement in `CLAUDE.md` of which fractional-derivative operator is correct
  and why the lower terminal at −∞ cannot be changed, together with a
  research-backed numerical policy covering quadrature, near-integer orders,
  log-space arithmetic, symbolic evaluation and CDF inversion.
- `registry.failed_prior_modules()`, reporting prior modules that failed to
  import and the exception that stopped each.

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
