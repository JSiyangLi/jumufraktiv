# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below `1.0.0` the public API may change in any release
without a deprecation period.

## [Unreleased]

### Fixed

- **Two thirds of the backend matrix could not be reached through
  `MGFDerivative`.** `_build_derivative` called `mgfDerivative(..., t=None)`
  unconditionally, but only the `symbolic` backend can build a representation
  before an evaluation point is known — it differentiates the prior's MGF and
  returns an expression in `t`. Every numeric backend quadratures at a
  particular `t` and raised `ValueError: t must be provided` at construction.

  That took out `bell` and `jax` for integer orders, and every backend for
  fractional orders — so six of the fourteen likelihoods could not be used at
  all: `normal`, `halfnormal` and `levy` (`a = n/2`), `maxwell-boltzmann`
  (`a = 1.5n`), and `gamma` and `inverse gamma` (`a = Σ shape_i`, fractional
  for any non-integer shape). The audit's own notes named only three of them.

  Numeric backends now defer: the class holds a thunk that runs the dispatcher
  once an evaluation point arrives. All nine (order type, backend) combinations
  now agree to 1e-8, and the whole inference API — density, CDF, MGF, moments,
  predictive — works on a fractional posterior.

- **`update` rejected the posteriors that could update, and accepted ones that
  could not.** Its guard tested `_is_symbolic`, whether the *result* at
  `t = -b` is an expression, which is `False` whenever the hyperparameters are
  numeric — the ordinary case. So `method='symbolic'` was refused while
  `method='auto'`, which resolves to the same backend, went through and
  returned the right answer. The test is now `_deriv_is_symbolic`, the
  representation, which is what `to_prior_object` actually requires.

  Its advice was wrong too: it suggested "choose a numeric method (jax, bell,
  scipy, mpmath)", and all four fail. `bell` raises "Prior does not provide a
  symbolic CGF", `jax` raises inside its tracer, and `scipy` and `mpmath`
  returned `-inf` — the tan-transform integrand's blanket
  `except Exception: return 0.0` turns a missing MGF into a zero at every
  quadrature node. That silent case became reachable only once fractional
  posteriors could be constructed; it is now refused with an accurate message
  and recorded as `xfail(strict=True)` against the PR that adds the capability.

- Corrected the density in `like_stats/Weibull.py`'s module docstring. It read
  `f(y; λ, ρ) = ρ λ^ρ y^{ρ-1} exp(-λ y^ρ)`, which is not a density — it
  integrates to `λ^{ρ-1}`, giving 2.0 at (ρ=2, λ=2) and 4.0 at (ρ=3, λ=2). It
  mixed the prefactor of one rate convention with the exponential of the other,
  and implied `a(y) = ρ` where the code correctly uses `a(y) = 1`. Only that one
  line was affected; every other statement in the module, and all of its
  arithmetic, used the correct convention. The existing tests exercised only
  `rho=1.0`, the single value at which `λ^ρ = λ` and the error is invisible.

- **Non-finite data and known parameters are now rejected.** `np.any(x <= 0)`
  is `False` for NaN, so a NaN passed every positivity guard in all fourteen
  likelihood modules and landed in `a`, `b` or `log_c`. It never reached the
  user as a NaN — it surfaced much later as an error naming the wrong thing
  ("Derivative at t=-b is negative" for Rayleigh, "t must be provided" for
  Normal, "cannot convert float NaN to integer" for Poisson). Rejection happens
  at the single point where values enter, and the message names the offending
  input. Infinities are covered too, and scalar known parameters — which took a
  separate branch that bypassed the check entirely — now go through the same
  guard.
- **A derivative order whose moment does not exist is rejected at `t = 0`.**
  The evaluation point is `t = -b`, so `b = 0` puts it at the origin, where
  `D^a M(0) = E[Theta^a]` and the moment must be finite. `b = 0` arises from
  ordinary data — every observation at the known mean (`laplace`, `normal`), at
  zero (`halfnormal`), or at the scale (`pareto`) — and is common once data is
  rounded. Previously this returned `inf` at order 2 and raised
  `TypeError: Cannot convert complex to float` at order 3 against a Pareto(2)
  prior, neither naming the cause.

### Added

- `mitMGFprior.max_finite_moment`, the strict supremum of admissible derivative
  orders at `t = 0`: infinite for `gamma` and `uniform`, the tail index for
  `pareto`, and zero for the improper `heaviside` prior, which has no finite
  moments at all. Defaults to infinity for custom priors, deferring to the
  numerical result rather than guessing. Only consulted at `t = 0`; no moment
  condition is imposed anywhere else, since imposing one would wrongly reject
  the heavy-tailed priors the operator exists to support.

- **Unrecognised keyword arguments to `MGFDerivative` are now an error rather
  than a silently wrong answer.** The constructor split `**kwargs` against the
  *union* of every likelihood's parameter names and forwarded everything else to
  the derivative layer, where `**kwargs` absorbed it. Two ordinary mistakes were
  therefore silent:
  - a misspelling — `scal=2.0` instead of `scale=2.0` — left the likelihood on
    its default, giving a log-evidence wrong by 0.92 nats with no error and no
    warning;
  - a parameter valid for a *different* likelihood — `rho=` on a Poisson — was
    forwarded into the `ready` function and swallowed by its `**kwargs`.

  Accepted names are now derived from each likelihood's own signature, so the
  check cannot drift out of step with the likelihood modules and each likelihood
  is validated against its own parameters. Unknown arguments raise `TypeError`
  naming the offending key, suggesting a close match, and listing what is
  accepted.
- Removed a stray `from unittest import result` that shadowed a local name.

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

- `derivativeDispatch.resolve_backend` is now the single encoding of the
  backend matrix — order classification, `auto` resolution, per-row validation
  and the `bell`/`jax` reinterpretation. It was inline inside `mgfDerivative`,
  so nothing else could ask which backend would serve a request without
  re-deriving the rules, and the array-order branch bypassed it entirely.
  Requesting `bell` or `jax` for a fractional order now warns rather than
  printing, and says how to silence it.
- `tests/test_deferred_construction.py`, covering backend resolution, the
  deferred representation, and construction of all fourteen likelihoods against
  the closed-form Gamma reference on every backend.

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
  "Using fractional derivatives to derive marginal densities", manuscript in
  preparation (2026), arXiv:2409.11167 — in `CITATION.cff`, the README and
  `CLAUDE.md`. The repository previously cited it nowhere.
- A statement in `CLAUDE.md` of which fractional-derivative operator is correct
  and why the lower terminal at −∞ cannot be changed, together with a
  research-backed numerical policy covering quadrature, near-integer orders,
  log-space arithmetic, symbolic evaluation and CDF inversion.
- `registry.failed_prior_modules()`, reporting prior modules that failed to
  import and the exception that stopped each.
- `tests/test_likelihood_correctness.py`, checking the MGF-marginalisable
  criterion itself: that each module's own `a`, `b` and `log_c` reconstruct the
  true log density, `log L = log_c + a·log θ − b·θ`, against an independent
  `scipy.stats` reference across five orders of magnitude of `θ` and three
  sample sizes. The existing likelihood tests checked the *contract* — shapes,
  finiteness, additivity — so nothing would have caught a `b` off by a factor
  of two. All fourteen pass.

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
