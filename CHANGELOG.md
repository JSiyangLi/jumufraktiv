# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below `1.0.0` the public API may change in any release
without a deprecation period.

## [Unreleased]

### Changed

- **The Pareto prior's fractional derivative is 4.6x to 6.9x faster**, and no
  number moved. Its MGF is written with `expint`, the generalised exponential
  integral, which neither SciPy nor NumPy defines — SymPy compiles the name
  into a bare global anyway, so the compiled function raised `NameError` on its
  first call and every quadrature node fell back to symbolic substitution:
  **2840 `subs` calls for a single evaluation point**, at roughly 306 us each.

  `jumufraktiv.special` now supplies `expint` to `lambdify`, backed by mpmath
  at about 39 us a point. Measured against an mpmath oracle at 50 digits with
  the Pareto density written out separately, the relative error is 3.3e-16 to
  2.1e-15 across four (order, t) pairs — the fast path skips work rather than
  changing an answer, which is the property it has to have. 1004.6 ms per
  evaluation point becomes 218.7 at a batch of eight, and 870 becomes 126.5 at
  a single point. (PR 14b)

- **The internal names are English.** `mitMGFprior` is now `MGFPrior`, its
  module `jumufraktiv/mitMGFprior_class.py` is `jumufraktiv/MGFPrior_class.py`,
  and its two constructors follow: `as_mitMGFprior` and `is_mitMGFprior` are
  `as_MGFPrior` and `is_MGFPrior`. The fourteen `bereit<Distribution>`
  functions are `each<Distribution>`, keeping them parallel to the `ready*`
  functions they sit beside — `ready*` aggregates over the sample, `each*`
  returns one entry per observation.

  A breaking change with no aliases, which is what pre-release means. The
  package name stays `jumufraktiv`: a distribution name is an identity and is
  expensive to change once published, while an identifier a caller types is
  not. (PR 14a)

### Added

- **Two guards that stop a route returning a number it can prove is wrong.**
  Neither reroutes: an explicit `method=` is still never reinterpreted, and
  each error names the route that does work. (PR 13f)

  `method='symbolic'` and `method='bell'` now measure how far the
  differentiated MGF's terms cancelled — `sum|term| / |sum term|`, so `log10`
  of it is the number of significant digits lost — and refuse below eight
  surviving digits. Against Uniform(0.5, 2) at `t = -1` the route used to
  return `3.60e+16` for a true `6.67e+06` at order 30, and errors of 2.0e-06
  and 1.8e-02 at orders 16 and 20. Orders 6 and 12 retain 13.4 and 9.0 digits
  and still compute. Gamma's ratio is exactly 1.0 at every order, so the check
  never fires for a prior whose CGF has one-signed derivatives.

  The fractional routes now check the derivative their kernel needs,
  `M^(floor(a)+1)`, at `t = 0`. It is infinite for a heavy-tailed prior once
  `floor(a)+1` reaches the moment bound, which the fixed grid cannot resolve —
  against Pareto(alpha=3) it is right to 1.6e-14 at order 1.95 and wrong by
  1.8e-06 at 2.011, a step rather than a slope, which is why the condition is
  exact rather than a tolerance. `mpmath` absorbs that one and is not refused.

### Fixed

- **Both fractional routes returned nonsense at `t = 0` for any prior whose
  MGF has a removable singularity there.** Uniform and heaviside carry `t` in a
  denominator, so `M^(floor(a)+1)` reads 0/0 at the origin even though the
  value is finite. Against Uniform(0.5, 2) at order 1.5 the relative error was
  6.0e+05 for `scipy` and 5.2e+149 for `mpmath`, with no warning. Both now
  refuse and name `method='auto'`, which computes `E[Theta^a]` directly from
  the density and is exact to ~1e-16. Found while testing the guard above; it
  had never been recorded. (PR 13f)

- `cached_term_values` probes a compiled expression before trusting it, the
  same way `cached_lambdify` does — SymPy compiles Pareto's `expint` without
  complaint and the result raises `NameError` on the first call. (PR 13f)

- **The cancellation guard let complete cancellation through**, which is the
  one case it most needed to catch. A sum of non-zero terms reaching exactly
  zero has lost every significant digit, and the true value cannot be zero
  because `D^a M(t) = E[Theta^a e^(t Theta)]` is strictly positive. Skipping
  it disabled the check where it mattered most; scoring only the surviving
  points was worse, because a batch with one fully cancelled point among two
  good ones reported 15.7 surviving digits and passed. (PR 13f)

- The guard blamed a removable singularity for a prior that has no symbolic
  MGF at all — one built from `mgf_backend`/`pdf_backend`, or the one
  `to_prior_object` produces for sequential updating. Such a prior cannot use
  a differentiating route at any `t`, and that route's own failure is the
  accurate report. (PR 13f)

- `cached_term_values` compiles its terms as a list rather than a
  `sympy.Matrix`. A term free of `t` — which a differentiated MGF may
  perfectly well have — evaluates to a scalar while its neighbours evaluate to
  arrays, and `Matrix` raises on that mixture instead of broadcasting.
  (PR 13f)

### Added

- **The posterior CDF, quantile, interval and sampling methods now work for
  every registry prior.** `post_cdf` needs the prior's incomplete MGF, and
  `post_quantile`, `post_interval` and `post_sample` are built on it, so the
  `uniform` and `heaviside` priors were losing four public methods at once,
  on every backend, behind a message that read as a statement about the
  mathematics. Both integrals are elementary. Verified against mpmath at 40
  digits: worst relative error 1.25e-15 over 42 `(t, u)` pairs for uniform and
  9.09e-16 over 24 for heaviside. (PR 13b)

- `MGFPrior.mgf_finite_below`, the supremum of `t` for which the prior's MGF
  is finite. Declared by each registry prior and consulted by `post_mgf`.
  Custom priors default to infinity. (PR 13b)

- `tests/test_documentation_runs.py`, which executes the README's code blocks,
  renders it the way PyPI does, checks its likelihood table against
  `LIKELIHOOD_REGISTRY`, and runs the docstring examples. (PR 13a)

- Docstrings for `MGFDerivative.is_symbolic`, `.value_numeric` and
  `.prior_has_iMGF`, and for the `MGFPrior` fields, which had been described
  only in their class's summary tables. (PR 13e)

### Fixed

- **The API reference described twenty-five members twice, and invented two
  more.** Napoleon renders the NumPyDoc `Attributes` and `Methods` sections
  into `.. attribute::` and `.. method::` directives, which collide with the
  ones `autoclass :members:` already emits. The rendered page also gained
  attributes literally named `Properties` and `----------`, from a heading
  napoleon does not recognise being absorbed into the preceding section.
  Members are now documented on themselves. (PR 13e)

- **Six of the twelve signatures in `MGFDerivative`'s hand-written method
  table had drifted from the code**, `post_sample` never having learned about
  the `rng` argument that made it reproducible and `post_quantile` omitting
  six parameters. autodoc reads signatures from the code, so the table is
  gone rather than corrected. (PR 13e)

- Three inert parameters that the reference advertised as controls:
  `resolve_backend(prior=...)`, which is removed; and `tol` on the mpmath
  fractional backend and on the three JAX root-finders, which are kept for a
  uniform call from `solve_root` and now say plainly that they are not read,
  naming `dps` and `maxiter` as what to adjust instead. (PR 13e)

- `MGFPrior.pdf_sym_func` is removed. It was read from a prior-spec key no
  prior module supplies and no code path consumed, so it was `None` on all
  four registry priors and on every hand-built one. (PR 13e)

- Three docstrings listed their parameters in an order the signature does not
  declare them in, `integerDeriv_numeric_jax` most visibly, whose signature
  begins `(t, prior, order)` while its documentation began with `order`. The
  suite now asserts the ordering across all 94 documented callables. (PR 13e)

- **`post_mgf` returned the value of a formula outside the domain where that
  formula is the MGF.** The Gamma(8, 6) posterior's is `(6/(6-r))**8`, finite
  only for `r < 6`; the even power kept it positive and plausible past the
  radius --- `25.63` at `r = 10` and `2.76e-10` at `r = 100`, where the answer
  is infinite. It now refuses, naming the largest admissible `r`. (PR 13b)

- **`post_mgf` returned `nan` at `r = b` for three of the four priors.** There
  `t = 0` and the value reduces to a raw moment. The uniform prior's MGF has
  `t` in a denominator so substitution is `0/0`; pareto and heaviside diverge
  and nothing refused. Uniform now returns the exact value and the other two
  raise. `sp.limit` is not the fix --- it returns `oo` where the value is
  `12.19` --- so the origin routes through the expectation backend. (PR 13b)

- **The README's quick start raised `TypeError`.** `evidence()` became a bare
  log scalar and the README, both notebooks and ten of the twelve worked
  examples in the API reference were not updated with it. (PR 13a)

- `post_predictive` documented a symbolic-observation route whose branch could
  never succeed; it is now an explicit `NotImplementedError`. (PR 13a)

- `post_central_moment` described order 1 as "the mean"; it is
  `E[Theta - E[Theta]] = 0`. (PR 13a)

- The four symbolic paths wrapped every exception in `RuntimeError`, costing a
  deliberate refusal both its type and its advice. They now let
  `NotImplementedError` through. (PR 13a)

- `from_registry` rejected SymPy *numbers* as well as symbols. It now converts
  them, and refuses non-finite or unconvertible hyperparameters by name --- the
  latter used to build a prior whose every derived quantity was `0` or `nan`.
  (PR 13a)

- **The API reference documented 2 of the package's modules.** The `automodule`
  directives named modules bare rather than `jumufraktiv.`-qualified, so
  autodoc could not import them and silently produced nothing while the build
  exited zero. Now 26 modules, 73 functions, 32 methods and 43 attributes.
  (PR 13c)

- `html_theme` was assigned twice in `docs/conf.py`, so the declared
  `sphinx-rtd-theme` was inert and every build shipped alabaster. (PR 13c)

- Roughly 70 reStructuredText defects in docstrings --- unindented bullet
  continuations, formulae read as definition lists, and mathematical absolute
  bars read as substitution references. These were invisible until the
  `automodule` repair made the modules render at all. (PR 13c)

- Array evaluation points, array-valued derivative orders, the batched
  expectation route, the fixed-grid kernel, symbolic-path correctness, domain
  guards and `logminus`, `like_stats` de-duplication, the diagnostics policy,
  the symbolic-differentiation cache, and the public API surface. These landed
  as PRs 4b through 12b while this file was not being updated; the pull request
  descriptions carry the measurements. (PRs 4b, 4c, 4d, 5, 6a, 6b, 6c, 7, 8, 9,
  10, 12, 12a, 12b)

### Changed

- The two new incomplete MGFs are batched rather than looped: 4.5x faster at 10
  evaluation points and 620x at 10000, and flat rather than linear in between.
  (PR 13b)

- `docs/` gains `installation`, `tutorial` and `examples` pages, which the
  toctree referenced but which did not exist. (PR 13c)

- CI builds the documentation with `-W -E`, so any Sphinx warning fails the
  build. This replaces a counted baseline, which permitted swapping one defect
  for another; `-E` is needed because Sphinx skips unchanged pages and the
  warnings are emitted during the read being skipped. (PR 13e)


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

- `MGFPrior.max_finite_moment`, the strict supremum of admissible derivative
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

- `MGFPrior.from_registry` now initialises the registry. It previously read
  `PRIOR_REGISTRY` directly, so in a fresh process it raised
  `Unknown prior 'gamma'` — the registry was simply empty — unless some other
  registry function happened to have run first.
- Prior discovery is isolated per module. `MGFdictionary/paretoMGF.py` imported
  `torch` at module scope and the discovery loop aborted on the first module
  that raised, so a missing optional extra silently removed both `pareto` *and*
  `uniform` from the registry, leaving a warning in place of half the priors.
  Nothing in the package referenced the function that import supported, so the
  import was first made lazy and the function has since been deleted outright,
  along with the `torch` extra that advertised it. One module's failure no
  longer affects the rest either way.
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
- Optional dependency groups: `examples`, `docs` and `dev`.
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
  `jumufraktiv/tests/MGFPrior_test.py`. Neither was a test: both were
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
