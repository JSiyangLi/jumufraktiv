# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

---

## What this package does

`jumufraktiv` performs Bayesian inference for the **MGF-marginalisable family**
of likelihoods: those whose joint density factorises as

```
L(θ; y) = c(y) · θ^a(y) · exp(−b(y)·θ)
```

where `(a(y), b(y))` is **jointly** sufficient for `θ` and `c` is a normalising
factor. Neither `a` nor `b` is sufficient on its own, and they play different
roles: `a` is the order of differentiation, `b` fixes the evaluation point. Say
"the pair `(a, b)` is jointly sufficient", never "the sufficient statistic `a`".

For such a likelihood the marginal likelihood is an `a`-th derivative of the
prior moment-generating function `M`, evaluated at `t = −b`:

```
p(y) = c(y) · Dᵃ M(t) |_{t = −b}
```

Because `a(y)` need not be an integer, the derivative is generally **fractional**
— that is the package's reason to exist. Every other quantity (posterior
density, CDF, quantiles, moments, predictive density, sequential updates) is
derived from the same object, which is why almost everything routes through one
dispatcher.

Parameters are assumed **strictly positive**. That is not a convenience
assumption: `θ > 0` is exactly the condition that makes the defining integral
converge (see "The operator" below).

## Reference

The package implements the method of

> Li, S.-Y., van Dyk, D. A., & Autenrieth, M. *Using fractional derivatives to
> derive marginal densities.* Manuscript in preparation (2026).
> [arXiv:2409.11167](https://arxiv.org/abs/2409.11167)

Cite it in new docstrings that state the identity, and check it before changing
any of the mathematics. Its Theorem 4.1 gives a characterisation worth knowing:
a likelihood is MGF-marginalisable **iff** it admits a gamma conjugate prior,
i.e. iff it factorises in exactly the form above. That is a testable criterion
for anything added to `LIKELIHOOD_REGISTRY`.

The paper also describes a *second* family, marginalisable by a derivative with
lower terminal **0** acting on `t^a M(log t)` — the Mellin-side analogue,
covering Beta, Beta-prime and Dirichlet likelihoods. That is a different
operator and a different code path; this package does not implement it.

---

## The operator

`Dᵃ` is the **Liouville–Caputo fractional derivative with lower terminal at
−∞**, written `D^a_{(−∞)+}`. With `n = ⌊a⌋` and `γ = n+1−a ∈ (0,1]`:

```
Dᵃ M(t) = (1/Γ(γ)) ∫_{−∞}^{t} (t − x)^{γ−1} M^{(n+1)}(x) dx
```

**The lower terminal is the load-bearing part, and it must not be changed.**
The operator is required to satisfy `Dᵃ e^{tθ} = θᵃ e^{tθ}`, which is what makes
`Dᵃ M(t) = E[θᵃ e^{tθ}]` and hence the whole construction work. That property
depends on the terminal being −∞, and on `θ > 0` (the inner integral is a Gamma
integral, convergent only for positive `θ`).

Three specific ways to get this wrong:

- **Terminal 0 (ordinary Riemann–Liouville or Caputo).** Fails outright. The
  evaluation point is `t = −b(y) < 0`, so `∫₀ᵗ` runs backwards past the origin.
  Even where it is defined it differs from the correct answer by a factor of the
  regularised incomplete gamma `P(1−a, θt)`.
- **The Riemann–Liouville *form* at −∞** (fractionally integrate, then
  differentiate). It has the right eigenfunction property, but it needs the
  prior's shape `c > γ` and **diverges** for priors with density mass piling up
  at `θ → 0`. Worse, adaptive quadrature on the divergent integral returns a
  plausible-looking wrong value without raising. The Caputo form used here
  differentiates first, which improves decay and converges for all `c > 0`.
- **Requiring `E[Θ^{⌊a⌋+1}] < ∞`.** Sufficient but *not* necessary at `t < 0`,
  where the exponential dominates any polynomial. Do not enforce it — it would
  wrongly reject heavy-tailed priors such as `pareto` for which the identity
  holds perfectly well.

**The one exception is `t = 0` exactly.** There the exponential is 1 and the
identity reduces to `Dᵃ M(0) = E[Θᵃ]`, so the moment *does* have to exist. That
boundary is reachable: `b(y) = 0` whenever every observation sits at the value
the likelihood subtracts — `y ≡ mean` (laplace, normal), `y ≡ 0` (halfnormal),
`y ≡ scale` (pareto). Measure-zero for continuous data, ordinary once data is
rounded.

Priors therefore declare `max_finite_moment`, the strict supremum of admissible
orders: `∞` for gamma and uniform, `α` for pareto, and `0` for the improper
heaviside prior, which has no finite moments at all. `mitMGFprior` defaults it
to `∞`, which is the right default for a custom prior — it defers to the
numerical result instead of guessing.

Note where this check lives and why. Admissibility is a joint property of
`(a, prior)`, **not** of the data: the same `b = 0` is fine against a Gamma
prior at every order and fatal against `pareto(α=2)` at order 2. So
`like_stats` cannot decide it — those modules are pure functions of the data
and cannot see the prior. It belongs where both are visible.

A related strength worth preserving: the operator reads `M` only on `(−∞, t]`.
It never needs `M` to the right of the evaluation point, so it works for priors
whose MGF exists only for `t ≤ 0` — Pareto and lognormal among them.

---

## Architecture

Three layers, each with a single responsibility. Respect the boundaries.

```
                     ┌──────────────────────────────┐
  user-facing  ───▶  │  MGFDerivative_class.py      │  inference API
                     │  (evidence, density, cdf,    │
                     │   moments, quantiles,        │
                     │   predictive, update)        │
                     └───────────────┬──────────────┘
                                     │  asks for Dᵃ M(t)
                     ┌───────────────▼──────────────┐
  computation  ───▶  │  derivativeDispatch.py       │  backend selection
                     │    ├─ symbolic_integerDeriv  │
                     │    ├─ numeric_integerDeriv_Bell / _JAX
                     │    ├─ symbolic_fractionalDeriv
                     │    ├─ numeric_fractionalDeriv_grid / _mpmath │
                     │    └─ numeric_expectation (the `auto` default) │
                     └───────────────┬──────────────┘
                                     │  reads MGF/CGF/PDF
                     ┌───────────────▼──────────────┐
  data         ───▶  │  mitMGFprior_class.py        │  prior container
                     │  registry.py + MGFdictionary │  prior registry
                     └──────────────────────────────┘

  like_stats/   — 14 likelihood modules supplying a(y), b(y), c(y)
  root_finding.py — vectorised bisection / Newton (NumPy + JAX)
  symbols.py    — canonical SymPy symbols: t, theta, r, u, q
  logsum.py     — log-space arithmetic helpers
  symbolic_cache.py — memo for repeated sp.diff (see "Caching" below)
```

**Layer rules**

- `MGFDerivative` delegates *all* mathematics to `mgfDerivative`. It must not
  differentiate anything itself.
- `derivativeDispatch` reads priors through the `mitMGFprior` interface only.
  It must not know which distribution it is handling.
- Priors never import from the inference layer.
- `like_stats` modules are pure functions of the data. They know nothing about
  priors or derivatives.

---

## Design principles

These are stated in the module docstrings and are **normative**. New code must
satisfy them; tests should assert them.

1. **Symbol–numeric principle.** The return *type* depends only on whether
   unresolved symbols remain — never on which code path was taken. If `t` is
   `None` or free symbols survive substitution, return a `sympy.Expr`.
   Otherwise return a number.

2. **Log principle.** In the numeric state, whether a function returns
   `(log_abs, sign)` or a plain scalar depends only on the `log` argument.
   Nothing else may change it.

3. **Tuple-vectorisation principle.** Evaluation points are the *pair* `(t, u)`.
   Both are broadcast to a common shape and evaluated as one batch. A function
   that accepts array `t` must accept array `u` and broadcast the two.

---

## Canonical symbols

Import from `jumufraktiv.symbols`; never redefine them locally, or SymPy
substitution will silently fail to match.

| Symbol  | Meaning                                       |
|---------|-----------------------------------------------|
| `t`     | MGF / CGF transform variable                  |
| `theta` | latent parameter (positive)                   |
| `r`     | posterior-MGF variable                        |
| `u`     | truncation point for the incomplete MGF (CDF) |
| `q`     | moment order                                  |

Hyperparameters are created with `symbols.param(name)`, which carries the
`real=True, positive=True` assumptions SymPy needs to simplify.

---

## Backend matrix

Which `method` is valid depends on the *order type*. `resolve_backend` is the
single place this table is encoded; `mgfDerivative` calls it to dispatch, and
anything that needs to know how a request *will* be served asks it rather than
re-deriving the rules.

| Order type   | Valid `method`                  | `auto` resolves to | Works today |
|--------------|---------------------------------|--------------------|-------------|
| symbolic     | —                               | —                  | **not supported**, by design — see below |
| integer      | `symbolic`, `bell`, `jax`       | `symbolic`         | yes         |
| fractional   | `scipy`, `mpmath`, `symbolic`   | `scipy`            | yes         |

**An order carrying free symbols is refused, and that is a decision rather than
a defect.** `sp.diff(expr, t, n)` needs a concrete number of times to
differentiate; it cannot return a formula in `n`. Closed forms in the order do
exist for particular priors — the Gamma MGF differentiates to a Pochhammer
symbol — but there is no general route, and the package has no use for one:
`a(y)` comes from the data through `ready*`, so it is always numeric. A
symbolic order can only arise from calling `mgfDerivative` directly with a
`Symbol`, which is not a supported workflow.

This row was previously listed as broken and scheduled for repair. It now
raises `NotImplementedError` naming the free symbols and saying what to do
instead, and the dispatcher no longer warns that it will return "the analytic
continuation to non-integer orders" — a promise it never kept, issued *before*
the failure so that the last thing a caller saw was a claim about a result they
never received.

**Integer-valued orders that are not Python `int`s do work**, and this is the
part that was a real defect: `sp.Integer(2)` is classified as symbolic by
`resolve_backend` and used to hit the same dead end, even though SymPy
arithmetic produces `sp.Integer` routinely.

The `symbolic` backend for a *fractional* order now works, but only where SymPy
can do the integral. It returns a closed form for the Gamma prior and raises
`NotImplementedError` naming the prior for `pareto`, `uniform` and `heaviside`.
Declining is not a defect — the integral genuinely has no elementary closed
form for those — and the numeric backends serve them exactly.

Requesting `bell` or `jax` for a fractional order is not an error. Neither can
take a fractional derivative, so the argument is reinterpreted as
`integer_method` and the fractional backend falls back to `scipy`, with a
warning that the argument was not used as written.

For fractional orders, `integer_method` separately selects the integer backend
used *inside* the fractional integrator (`symbolic`, `bell`, `jax`).

An order counts as integer when `|order − round(order)| < int_tol`
(default `1e-12`).

**Only the `symbolic` backend can be built before an evaluation point is
known.** It differentiates the prior's MGF and returns an expression in `t`;
every numeric backend quadratures at a particular `t` and has nothing to return
until one arrives. `MGFDerivative` therefore holds one of two representations:
a `sympy.Expr` (`_deriv_is_symbolic`), or a thunk that runs the dispatcher once
an evaluation point is supplied. Methods that need to manipulate the derivative
symbolically — `post_density`, `post_cdf`, `post_mgf`, `update` — branch on
that flag.

---

## Conventions

**Imports.** Always fully qualified within the package:
`from jumufraktiv.symbols import t`. Bare module imports
(`from symbols import t`) resolve only when the package directory happens to be
on `sys.path` and break under normal installation.

**Docstrings.** NumPyDoc style, rendered by Sphinx with `napoleon`. Sections in
order: summary, Parameters, Returns, Raises, Notes, Examples. Docstrings must
describe what the code *does*, not what it is intended to do — if they diverge,
that is a bug in one of the two.

**Pull request descriptions are written for the reviewer, not for the author.**
The people reviewing this repository are experts in the statistics and the
numerical analysis, and are not necessarily fluent in its Python internals. A
description that can only be followed by someone who already knows the module
layout has failed at the one job it has: letting a reviewer judge whether the
change is *correct*.

Length is not the constraint — clarity is. A longer description that can be
read start to finish beats a compressed one that has to be decoded. But most
of the bulk that creeps in is not explanation, it is shorthand, and removing it
usually makes the text both clearer and shorter.

- **Open with what changed for someone using the package**, not with the
  private method that was wrong. "Six of the fourteen likelihoods could not be
  used at all" orients a reader; `_build_derivative` passes `t=None` does not.
- **Introduce every internal term on first use**, including the ones this
  document treats as vocabulary — *backend*, *symbolic versus numeric*,
  *deferred construction*, *strict xfail*. One clause is usually enough.
- **Say what a number means before showing it.** A table of measurements needs
  to state what the reference is and what counts as agreement. Numbers without
  that are decoration.
- **Write in full sentences.** Clipped fragments ("Both fixed." "Two thirds of
  the matrix, unreachable.") read as notes to self.
- **Name the mechanism, not just the symptom** — a reviewer checking
  correctness needs to know *why* it was wrong, and that is the part only the
  author can supply.
- **Flag what you did not fix, and why**, in the same voice as the rest. A
  loosened tolerance or a deferred repair must be visible, never buried.
- Keep the mathematics exactly as precise as the rules below require. Accessible
  prose is not licence to loosen a mathematical statement.

The same applies to commit messages, at proportionate length.

**Mathematical statements are held to the standard of the paper, not of prose.**
This package implements a published method and its authors read the docs, so a
loose statement is a defect even when the code is right. Two rules:

- *Do not weaken a joint statement into a marginal one.* `(a(y), b(y))` is
  jointly sufficient; writing "the sufficient statistic `a(y)`" is wrong, and
  was shipped in the README before an author caught it.
- *Prefer the reference's terminology over the more familiar synonym.* The
  operator is the Liouville–Caputo derivative with lower terminal −∞, because
  that is what the paper calls it; "Weyl" is a defensible name for the same
  object and is still the wrong word to use here.

When stating a result, name what is being claimed about what: which variable is
differentiated, which is integrated, what is held fixed, and over what domain
the statement holds.

**Parameterisation is part of the claim.** Whether a likelihood is
MGF-marginalisable is a property of the *parameterisation*, not of the
distribution family. The factorisation must hold in the particular `θ` the
package treats as unknown. Rayleigh is the clearest case:

```
f(y; σ) = (y/σ²)·exp(−y²/(2σ²))

  in the rate θ = 1/σ² :  y·θ·exp(−θy²/2)      ✓ c(y)·θ^a·exp(−bθ), a=1, b=y²/2
  in the scale σ       :  (y/σ²)·exp(−y²/2σ²)  ✗ the exponent is −b/σ², not −bσ
```

Same distribution, same data; one parameterisation is in the family and the
other is not. So "is the Rayleigh likelihood supported?" is not a well-posed
question — "is it supported in the rate?" is. Every `like_stats` module states
its parameterisation in the module docstring; read it before comparing against
any reference density.

**The `like_stats` modules are author-verified.** All fourteen have been checked
by the package's author. A discrepancy found against an external reference is
therefore **far more likely a parameterisation mismatch on the checker's side
than a defect**, and the burden of proof sits with the checker. Before reporting
one: re-derive the factorisation by hand, confirm which parameter the module
treats as `θ`, and confirm the reference is expressed in that same parameter.
Report only what survives that, and say explicitly which parameterisation was
used on both sides.

**Naming.** Some internals use German (`mitMGFprior`, `bereit*` for per-element
statistics alongside `ready*` for aggregated ones). Anglicising the internals
while keeping the package name is planned for a later wave; until then, follow
the existing convention rather than mixing a third one in.

**Diagnostics.** Library code uses `logging` and `warnings`, never `print`.
This is now enforced — `tests/test_diagnostics_policy.py` fails the build on a
`print` call anywhere under `jumufraktiv/`, and on an `if __name__ ==
"__main__":` block, which is where 287 of the old 329 prints lived.

Which of the two to reach for: `warnings.warn` when the *caller's result* is
affected and they may need to act, `logger.debug` when you are recording which
branch the library took. Always pass `stacklevel=2` to `warnings.warn`, so the
warning names the caller's line rather than the library's — the caller's line
is the only one they can do anything about.

**Errors.** Catch narrowly. Never let a broad `except` turn a real failure into
a warning or a silently wrong number — the registry once dropped half its
priors that way.

**Likelihood modules.** Each `like_stats/X.py` exports exactly three functions:

- `readyX(data, **kwargs) -> {'a', 'b', 'log_c'}` — aggregated over the sample.
- `bereitX(data, **kwargs) -> {'a', 'b', 'log_c'}` — per-element arrays, used
  by the vectorised posterior predictive.
- `cX() -> sympy.Expr` — symbolic normalising constant.

New likelihoods must be added to `LIKELIHOOD_REGISTRY` in
`MGFDerivative_class.py` **and** to the README's list.

Input handling is *not* part of what a likelihood module writes. Both entry
points must take their data — and every known parameter — through
`like_stats/_common.py::_extract_1d`, which converts to a 1-D float array and
rejects non-finite values, multi-column DataFrames and anything that is not
one-dimensional. A module that re-implements any of that is a regression, not a
style choice: fourteen byte-identical copies are precisely how four of them
came to be missing the dimensionality check while ten had it, and
`tests/test_dimensionality.py` asserts that no module defines its own.

**Priors.** A prior module in `MGFdictionary/` registers itself with
`@register_prior("name")` and returns `make_prior_spec(...)`. Discovery is
automatic for any module whose filename contains `MGF`. Optional heavy backends
(e.g. PyTorch) must be imported lazily so that one missing extra cannot prevent
other priors from registering.

---

## Commands

```bash
pip install -e ".[dev]"          # development install
pytest                           # full test suite
pytest -m "not slow" -x -q       # quick pass, for the iteration loop
ruff check .                     # lint
ruff format --check tests/       # formatting (tests/ only, see below)
sphinx-build -b html docs docs/_build/html   # documentation
```

**The quick pass is what you run while working; the full suite is what you run
before committing.** A test earns the `slow` marker by costing more than about
three seconds, which in this package always means real quadrature. Marking is
a scheduling decision and never a statement about importance — several of the
slowest tests are the most valuable in the suite, which is why CI runs the
full set on one Python version rather than dropping them.

Two habits, learned the expensive way. Run long suites in the background
rather than blocking on them, and do not run two at once: the quadrature is
single-threaded but the box is small, so concurrent runs make each other
slower and the timings meaningless. And prefer targeted node IDs while
iterating on one thing — a full run after every edit is the single easiest way
to waste an hour.

---

## Testing

`tests/` holds the suite; run it from the repository root so `conftest.py` is
importable (`pythonpath` is set in `pyproject.toml`).

**Assert mathematics, not recorded output.** A Gamma prior has a closed-form
MGF and all of its derivatives; a Gamma prior against a Poisson likelihood is
conjugate, so the entire posterior is a known Gamma. `tests/conftest.py`
exposes those references (`gamma_mgf_derivative_log`, `poisson_log_evidence`)
and most of the suite compares against them. A test that merely pins today's
number cannot tell a refactor from a regression.

| File | Covers |
|------|--------|
| `conftest.py` | fixtures and closed-form references |
| `test_analytic_reference.py` | evidence, density, CDF, MGF, moments, predictive, sequential update vs. exact values |
| `test_design_principles.py` | the three normative principles, with Hypothesis |
| `test_likelihood_stats.py` | the `ready`/`bereit`/`c` contract across all 14 likelihoods |
| `test_likelihood_correctness.py` | that `a`, `b`, `log_c` reconstruct the true density, vs. `scipy.stats` |
| `test_input_validation.py` | non-finite inputs, and the moment domain at `t = 0` |
| `test_constructor_kwargs.py` | keyword-argument routing on the constructor |
| `test_dispatch_imports.py` | the dispatcher's lazily-imported backends |
| `test_registry.py` | registry, prior container, custom-prior route, constructor validation |
| `test_deferred_construction.py` | `resolve_backend`, the deferred representation, all 14 likelihoods on every backend |
| `test_batch_evaluation.py` | array evaluation points vs. the closed form, and independence from the caller's warning filter |
| `test_array_order.py` | array-valued derivative orders: closed form, shape, symbolic `t`, and the parity cases |
| `test_known_broken.py` | every documented defect, as `xfail(strict=True)` |
| `test_no_unreachable_code.py` | that no module-level function is unreachable |
| `test_diagnostics_policy.py` | no `print`, and no `__main__` block, in library code |

**`test_known_broken.py` is the mechanism that keeps this document honest.**
Each test asserts the *correct* behaviour and is marked `xfail(strict=True)`,
so the suite stays green while the defect exists — but the moment a PR fixes
one, the test XPASSes and *fails* the build. That forces the fix to be recorded
both there and in the "Known-broken" list below. When you repair something,
expect a red build and remove the marker; do not weaken the assertion.

**Lint debt.** `pyproject.toml` carries an itemised `per-file-ignores` baseline
for the pre-audit library code — one entry per rule, each annotated with the PR
that removes it. It is a shrinking list, not a blanket suppression. `tests/`
has no exemptions.

PR 8 removed thirteen rules (`E402`, `E713`, `E731`, `F401`, `F541`, `I001`,
`B028`, `B904` and the five `SIM` codes), plus `RUF059` and `W292`, which its
deletions happened to clear. **`F821` and `F811` now have no exemptions at
all**, package-wide or per-file, which is the state they were always meant to
reach: they are blocking because they find defects rather than style issues,
and an exemption for them is a contradiction carried on sufferance.

What remains is fourteen rules, and the split is worth knowing: eleven are
cosmetic and belong to the documentation waves (`E501`, the three `RUF00x`
ambiguous-unicode codes, `W291`, `W293`, and the four `UP` annotation codes),
while three sit on real work — `B007` and `B905` on the vectorisation PR, and a
single `F841` that is not a lint issue at all but the visible end of the Pareto
incomplete-MGF defect recorded under "Known-broken".

`ruff format` currently runs over `tests/` only. Reformatting 13k lines of
library code would bury the audit's real diffs; that lands as its own PR in
wave 6.

---

## Audit status

The repository is undergoing a staged audit. Work lands one PR at a time.

| Wave | PR | Scope | Status |
|------|----|-------|--------|
| 0 | 1 | Repo hygiene, packaging metadata, project files, this document | **merged** |
| 0 | 2 | pytest + Hypothesis harness, CI, lint config | **merged** |
| 0 | 2b | `slow` marker, CI matrix split, test memoisation | **merged** |
| 1 | 3 | Import and registry integrity | **merged** |
| 1 | 3b | Constructor keyword-argument integrity | **merged** |
| 1 | 3c | Likelihood-statistic correctness tests | **merged** |
| 1 | 4a | Backend resolution and deferred construction | **merged** |
| 1 | 4b | Batch evaluation points (converged-mask) | **merged** |
| 1 | 4c | Array-valued derivative orders | **merged** |
| 1 | 4d | Symbolic fractional backend | **merged** |
| 1 | 12a | Posterior predictive's known parameters; the inert `torch` extra | **merged** |
| 2 | 5 | Symbolic-path correctness | **merged** |
| 2 | 6a | Domain guards; the three unusable posterior methods; `logminus` | **merged** |
| 2 | 6b | The fixed-grid quadrature kernel | **merged** |
| 2 | 6c | mpmath precision, the expectation route, sequential update | **merged** |
| 3 | 7 | De-duplicate `like_stats` | **merged** |
| 3 | 8 | Module layout, dead code, and the diagnostics policy | **in review** |
| 4 | 9 | Vectorisation | planned |
| 4 | 10 | Caching and dispatch | **merged** |
| 5 | 12 | Public API surface (less what 12a already took) | planned |
| 6 | 13 | Documentation infrastructure | planned |
| 6 | 14 | Docstring sweep | planned |

**PR 6 is split into 6a, 6b and 6c by blast radius, not by defect type**, on
the owner's decision. The question that governs verification cost is "does this
change numbers that currently look right?", and the three answer it differently.

- **6a** repairs domain guards, the bracketing of `post_quantile`, and
  `logminus`. None of it touches the quadrature kernel, so no number that was
  already correct moves, and it verifies cheaply. It also makes three
  advertised public methods callable, which they were not for any prior.
- **6b** replaces the quadrature kernel: a fixed grid on the `z = e^u`
  substitution, a range derived from `γ` rather than discovered by doubling,
  the exact near-integer correction, and log-space accumulation. Six recorded
  defects are symptoms of one design fault, so they are repaired together —
  splitting further would mean shipping a half-replaced kernel. This is the
  only one of the three that moves existing numbers, which is why it goes
  second: any surprise is then attributable.
- **6c** was scoped as "what cannot be computed away and must instead become
  loud", and that framing turned out to be wrong about both halves. mpmath's
  `dps` floor was a symmetric integration range and a float64 integrand, not a
  limit of the method; the alternating-CGF cancellation is removed outright by
  computing `E[θᵃe^{tθ}]` directly, whose integrand is positive. What shipped
  is therefore two repairs and one new capability — sequential updating for
  numeric backends — rather than three warnings.

**PR 12a is in wave 1 rather than wave 5 because it is a reachability repair,
not an interface decision.** The rest of PR 12 settles questions of taste —
which options to accept, which names to reserve, what a moment method should
return — and can wait for the waves that reshape those methods. Two of its
findings could not: `post_predictive` raised `TypeError` for ten of the
fourteen likelihoods, and the advertised `torch` extra installed a dependency
that no code path reached. Both are of a piece with wave 1's work on making
declared capability actually reachable, and neither is worth leaving in place
for four more waves.

**A bloat-and-simplification audit ran after wave 1 and is folded into the
table above rather than kept as a separate stream.** Most of what it found
belongs to work already planned, and splitting it out would have meant two
PRs touching the same files for different reasons. Where it lands:

| Finding | Goes to |
|---------|---------|
| `sp.diff` recomputed ~96,000 times per quick pass; a cache removes it | **PR 10** — **done**; re-measured at 97,308 calls for 40 distinct keys, 155 s |
| 2-D input silently accepted by 4 of 14 `ready*`, halving the derivative order; no test covers it | **PR 7**, as a commit *before* the de-duplication — **done** |
| ~900 lines of byte-identical `_extract_1d` / `_is_1d_dataframe` across the 14 modules | **PR 7** — **done**, 758 lines removed |
| 1,293 lines of `__main__` demo blocks holding 282 of the library's 326 `print` calls | **PR 8** — **done**; re-measured at 1,236 lines and 287 of 329, leaving 17 runtime prints to convert |
| 366 lines of never-referenced functions, including 13 stale `*_symbolic` wrappers | **PR 8** — **done**; 454 lines at the transitive fixed point, plus a 704-line superseded kernel |
| `to_prior_object`'s numeric route and the two broad `except Exception` clauses that fabricate finite numbers | **PR 6** |
| `use_loop` rejected by the constructor though it is a real backend option; `return_log` must be reserved | **PR 12** — `use_loop` is moot: it selected a scalar loop inside the adaptive scipy kernel PR 8 deleted. `return_log` still stands |
| Tests that pass against the defect they were written for | **PR 6** |
| `pytest -n 4 --dist loadfile` (2.22×) as a documented opt-in | **PR 13**, with the other developer commands |
| `CHANGELOG.md` has had no entry since PR 4a, leaving 4b, 4c, 4d and 12a unrecorded | **PR 13**, as a catch-up pass |

**Two PRs were folded into their neighbours, on the owner's decision, because
each pair touched the same files for related reasons and splitting them meant
verifying the same code twice.**

- *Tests that do not assert* was to be a PR of its own. Every one of them is a
  test written against a numerical defect that PR 6 repairs, so the repair and
  the test that failed to catch it now land together — which is also the only
  way to show that the strengthened test fails against the old code. (It was
  briefly numbered "PR 6b"; that label now means the fixed-grid kernel, so the
  old number is avoided here to prevent a collision.)
- *The diagnostics policy* was to be PR 11. Its work is replacing `print` with
  `logging`, and 282 of the library's 326 `print` calls live inside the
  `__main__` demo blocks that PR 8 removes. Moving the blocks and deciding what
  the survivors should do is one edit to each file, not two.

The counter-argument, recorded because it was real: "a test that does not test
anything" is a distinct kind of defect from a numerical one, and burying it
risks hiding the finding that most undermines confidence in the rest of the
suite. The mitigation is that PR 6 must call it out explicitly rather than
folding it into a list of numerical fixes.

### Known-broken, scheduled for repair

Do not build on these paths; do not paper over them. Each has a PR assigned and
a matching `xfail(strict=True)` test in `tests/test_known_broken.py`, except
where noted as "no runtime repro".

- **Integer derivatives of an alternating-CGF prior lose all accuracy above
  order ~16, and change sign.** `mgfDerivative(30, uniform(0.5, 2.0),
  method="symbolic", t=-1.0)` returns `−2.97e+15` where the true value of
  `E[θ³⁰e^{−θ}]` is `+6665897.83` (mpmath, 80 digits) — the wrong sign and a
  factor of 4.5e8. Accuracy has already gone by order 16 (1.7e-6) and order 20
  (1.8e-2). The mechanism is 25–26 digits of cancellation against the float
  coefficients in the stored MGF, so **no evaluator at any precision recovers
  it**: confirmed with exact rationals (`+3.09e16`) and with `evalf(80)`
  (`−2.97e15`). `a = Σy` for several likelihoods, so a Poisson sample summing
  to 30 under a uniform prior reaches this in ordinary use.

  Invisible to the suite for a reason worth fixing alongside it: **`bell` and
  `jax` are never run against a non-gamma prior anywhere**, and the Gamma
  MGF's derivatives have one-signed terms, so they cannot cancel. The
  cancellation ratio `Σ|term| / |Σ term|` named under "Numerical policy" is the
  diagnostic.

  **But the conclusion drawn from this — that the case cannot be computed and
  must merely be flagged — is wrong, and that is newly measured.** "No evaluator
  at any precision recovers it" is true of the *differentiated-MGF route* only.
  The defining identity `Dᵃ M(t) = E[θᵃ e^{tθ}]` gives a completely different
  computation whose integrand is **positive**, so it cannot cancel at all.
  Evaluating that expectation by ordinary `scipy.integrate.quad` in plain
  float64, against the same Uniform(0.5, 2) prior at `t = −1`, measured against
  an mpmath oracle at 80 digits:

  | order | differentiated MGF | direct quadrature |
  |-------|--------------------|-------------------|
  | 12    | 2.0e-10            | 2.2e-16           |
  | 20    | 1.8e-02            | 2.3e-16           |
  | 30    | 4.5e+08 (wrong sign) | 8.7e-18         |
  | 100   | —                  | 1.3e-15           |

  So the fix is a route, not a warning: for a prior with a usable density,
  compute the expectation directly instead of differentiating the MGF. That is
  an architectural addition rather than a repair, it needs a decision about when
  to prefer it, and it does not obviously extend to priors given only as an MGF
  — so it is recorded here for the owner rather than assumed.

  *Moved from PR 5 to **PR 6**.* It was filed under symbolic-path correctness
  because the reproduction goes through `method="symbolic"`, but nothing about
  the symbolic path is wrong: `sp.diff` returns the correct derivative and the
  loss happens when float coefficients cancel during evaluation. It is a
  conditioning problem, and it belongs with the other conditioning work rather
  than with the three name-and-substitution defects PR 5 repaired. *(PR 6)*
- **`dps` above about 20 cannot improve the mpmath backend's *return value*,
  because it returns `(log_abs, sign)` as Python floats.** The internals are now
  arbitrary-precision throughout — the integrand is the prior's symbolic
  derivative evaluated with `evalf(dps)`, and the quadrature result is no longer
  cast to float before the log is taken — so the computation carries whatever
  precision is asked of it. Measured relative error against an mpmath oracle at
  60 digits: 7.1e-16 at `dps=15`, then 4.0e-17 from `dps=20` upward, which is
  float64 rounding of an otherwise exact answer.

  So the limit is the return convention, not the mathematics, and lifting it is
  the deferred "replace `(log_abs, sign)` with a small result type" decision
  rather than a numerical repair. Recorded here so the two are not confused.
  *(deferred — see "Deferred decisions")*
- **The Pareto prior's numeric incomplete MGF returns NaN at every argument,
  and its JAX twin raises.** `pareto_imgf` and `pareto_logimgf` call SciPy's
  `gammaincc(a, z)` with `a = −α < 0`, which is outside its domain, so both
  return `nan` — measured at `α=3, ξ=1, t=−1, u=2`, where direct quadrature of
  the density at 40 digits gives `0.24880390851855957`. `pareto_imgf_jax` calls
  `jnp.gamma`, which does not exist in `jax.numpy`, so it raises
  `AttributeError` before it can reach the same domain error.

  **The symbolic route is exact**, matching that reference to 1e-16, so this is
  a defect in the two numeric implementations rather than in the expression
  they implement — which also means a repair should start from `imgf_sym`.
  The consequence for a user is that `post_cdf` and `post_quantile`, which go
  through the incomplete MGF, work for a Pareto prior only on the symbolic
  path.

  Found while clearing PR 8's lint baseline: `F841` flagged a `sign` computed
  and discarded in `pareto_logimgf`, and the discarded sign turned out to be
  the least of it. That one `F841` is the only reason the code remains in the
  baseline. *(unscheduled — a numerical repair needing its own verification,
  so it is recorded rather than folded into a module-layout PR)*
- **The incomplete-MGF derivative is wrong, not merely small, below
  `u ≈ 1e-2`.** Measured against the exact Gamma(8, 6) posterior of the
  canonical test problem, its log comes out as `−65.17` at `u = 1e-6` where the
  true value is `−110.41` — a 45-nat error — with the computed sign flipped
  negative. It is already wrong by 17.8 nats at `u = 1e-4` and becomes
  trustworthy only around `u = 1e-2`.

  The audit previously described this as underflow. It is not: the value is
  badly wrong well before anything underflows, which is cancellation in the
  incomplete gamma. `post_quantile` no longer trips over it, because its
  bracket now starts from the posterior's own scale `(a+1)/b` rather than from
  `1e-6`, but the inaccuracy itself is untouched and belongs to the kernel
  work. It bounds how far into the lower tail any CDF-based method can be
  trusted. *(PR 6b)*
- **`post_raw_moment` and `post_central_moment` disagree on return shape.** With
  the same `log=True`, one returns a bare scalar and the other `(log_abs,
  sign)` — a direct violation of the log principle. *(PR 12)*
- **`post_sample` is not reproducible** — unseeded `np.random.rand`, no `rng`
  argument. *(PR 12)*

### Verified correct

Worth recording, because it bounds where the bugs are. Against closed-form
references, on the paths that do run, the mathematics is right: evidence,
posterior density, CDF, MGF, raw and central moments, posterior predictive, and
sequential updating all match exact values for the conjugate Gamma/Poisson
model to within `1e-8`. Integer derivatives of the Gamma MGF match the
analytic formula for orders 0–5, and the `symbolic` and `bell` backends agree.
The defects above are in dispatch, plumbing and edge handling — not in the
core mathematics.

**Array-valued derivative orders** now match the closed form to 1e-10 at
orders 0.5, 1.5, 1.9 and 2.5, preserve the caller's shape, broadcast against an
array `t`, and return expressions when `t is None`. The public consequence is
visible without any reference value: the 0th raw moment of a fractional
posterior is exactly 1, where it used to be 1.903.

**Array evaluation points**, once the batch path stopped zeroing the integrand
at converged points, agree with the closed form to `1e-12` or better across
orders 0.5, 1.5, 1.9 and 2.5 and over point sets of two, three and five widely
spread `t`. Removing that mask also made the path *faster* — the suite's batch
tests run in 178 s against 475 s — because the mask was what prevented
convergence, so `L` had been doubling far past the integrand's support.

**Dimensionality is now checked in one place.** The `ndim != 1` guard lives
inside the shared `like_stats/_common.py::_extract_1d`, which every entry point
routes its data *and* its known parameters through. All twenty-eight `ready*`
and `bereit*` functions reject 2-D input, where previously ten of twenty-eight
did. That it is one guard rather than fourteen agreeing ones is the point:
byte-identical copies are exactly how the four unguarded modules went
unnoticed.

---

### A testing hazard this repository has already hit three times

Recorded here because each instance cost real time and the shape recurs:
**a check that sits downstream of the property being tested can pass for
reasons unrelated to it.**

- `pyproject.toml` sets `filterwarnings = ["error"]`. NumPy's "overflow
  encountered in exp" therefore becomes an exception *under pytest only*, which
  made a numeric path abort and fall back to a slower but correct route. The
  suite got the right answer; users, whose warnings are not escalated, got a
  wrong one. Any test comparing library output must ask whether the harness has
  changed the code path — `tests/test_batch_evaluation.py` now asserts directly
  that results do not depend on the caller's warning filter.
- A `ruff check --select` invocation appeared to show a rule passing when
  `per-file-ignores` was silently suppressing it; `--isolated` showed the
  violations were all still there.
- **A fixture value that coincides with a default hides every bug that falls
  back to that default.** `post_predictive` forwarded only the caller's keyword
  arguments to the likelihood statistics, ignoring the known parameters stored
  at construction. Thirteen likelihoods raise `TypeError` for that; Poisson
  alone has a default (`scale=1.0`), so it silently computed against a scale
  the user never asked for. Poisson is also the likelihood every predictive
  test used, and `conftest.POISSON_SCALE` is `1.0` — the default exactly, where
  the wrong answer and the right answer are the same number. Off that value the
  error is 1.308 nats at `scale=5.0`. So the suite covered the one likelihood
  that fails quietly, at the one value where the failure cannot be seen.

The general rule: assert the property itself, and confirm a new test fails
against the unfixed code before trusting that it passes against the fixed one.
The third instance adds a corollary — **choose fixture values that are not
defaults**, since a parameter tested only at its default value is a parameter
whose plumbing is untested.

---

## Which route computes the derivative

There are two fundamentally different ways to get `Dᵃ M(t)`, and since PR 6c
the default is the second.

**Differentiate the MGF.** Take `M`, differentiate `a` times — symbolically for
integer orders, or through the fractional-integral kernel otherwise — and
evaluate. Every backend except one does this.

**Compute the expectation.** `Dᵃ M(t) = E[θᵃ e^{tθ}] = ∫ θᵃ e^{tθ} p(θ) dθ`.
Same quantity by definition, entirely different arithmetic, and **the integrand
is positive**, so it cannot cancel.

`resolve_backend` sends `method="auto"` to the expectation route whenever the
prior supplies a density, which is always: `mitMGFprior` refuses to construct
without one in both its symbolic and its backend mode. An explicit `method=` is
never reinterpreted.

The decision is measured. Across 240 cases — four priors, ten orders from 0.5
to 30, six evaluation points from −0.5 to −50 — scored against mpmath at 60
digits with each density written out independently:

| | differentiating | expectation |
|---|---|---|
| unacceptable (rel > 1e-8) | 5 of 240 | **0 of 240** |
| wrong sign | 2 | **0** |
| worst case | 2.46e+00 | 1.17e-11 |
| median | 6.3e-17 | 1.0e-16 |
| cost, order 1.5 | 2087 ms | **270 ms** |

All five failures are the `uniform` prior at orders 12–30 with `t` near zero,
where its alternating CGF cancels through 25–26 digits.

**It is not uniformly better, and that is worth knowing before relying on it.**
Its worst case — 1.17e-11, Gamma at half-integer orders — is about 100× worse
than the differentiated route's worst *non-uniform* case. What it is is never
catastrophic, which is the property a default needs.

Two consequences beyond accuracy. The route needs only `p(θ)`, never the MGF,
which is what makes sequential updating work for numeric backends: the prior
`to_prior_object` builds carries a density and no `mgf_sym`, so no
differentiating backend could consume it. And it is 5–8× faster.

---

## Caching

Symbolic differentiation is memoised in `symbolic_cache.py`, and every library
call site goes through `cached_diff` rather than `sp.diff` directly. Adding a
bare `sp.diff` to library code puts the call back on the uncached path.

The reason is measured, not assumed. One quick pass made **97,308 calls to
`sp.diff` for 40 distinct `(expression, symbol, order)` triples** — about
2,400-fold redundancy, 155 s of runtime — and 95,573 of those came from a single
line in `symbolic_integerDeriv.py`. The redundancy is structural: a prior's MGF
is one fixed expression once its hyperparameters are substituted, but every
quantity derived from it re-enters the dispatcher separately and every
quadrature evaluates its integrand at many nodes.

Removing it took the quick pass from 310 s to 162 s (**1.91×**) and the full
suite from 508 s to 278 s (**1.83×**), measured against an unmodified tree
checked out into a separate worktree. What is left is genuine quadrature, so
further speed comes from PR 6's fixed-grid kernel, not from more `slow` markers.

**The key is the expression object, not its `srepr`.** That is a deliberate
choice with a correctness condition attached. The object key costs 0.36 µs
against `srepr`'s 68.1 µs, and it is safe only because SymPy's equality is
precision-aware: `Float(9.0, 53) == Float(9.0, 24)` is `False`, so the two hash
alike but do not compare equal and stay separate entries. If that ever changes,
a derivative computed at one precision would be returned for another.
`tests/test_symbolic_cache.py` asserts the property directly, so the build fails
rather than the conflation happening silently — at which point the key must go
back to `srepr`.

No invalidation is needed or wanted. A SymPy expression is immutable and its
derivative is a pure function of it, so a cached value cannot go stale;
`clear_derivative_cache()` exists for tests and for releasing memory, never for
correctness.

---

## Numerical policy

Conclusions from a background research pass, recorded so later waves do not
re-litigate them. Each was measured against this repository's own environment.

**The weak singularity is already handled.** The substitution `z = e^u` turns
the kernel into `z^{γ−1} dz = e^{γu} du`, so Gauss–Jacobi, QAWS and
product-integration rules are unnecessary here. The transformed integrand
decays single-exponentially on ℝ, which is precisely the class where a **plain
uniform-grid trapezoid rule converges geometrically** and adaptive
Gauss–Kronrod does not. Measured: trapezoid reaches 2e−15 in 321 evaluations
where the current adaptive scheme needs 1092. A fixed grid is also what makes
the tuple-vectorisation principle achievable — fixed nodes mean one batched
evaluation instead of an independent adaptive loop per point.

**Near-integer orders have an exact fix, not a heuristic one.** As `a → (n+1)⁻`,
`γ → 0`, and the answer is computed as `(1/Γ(γ)) × (a diverging integral)` —
0 × ∞. Subtracting a function with the same value at `z = 0` and a known
weighted integral removes it exactly, because `∫₀^∞ z^{γ−1}e^{−z}dz = Γ(γ)`:

```
Dᵃ M(t) = M^{(n+1)}(t) + (1/Γ(γ)) ∫ e^{γu} [ M^{(n+1)}(t − e^u)
                                            − M^{(n+1)}(t)·e^{−e^u} ] du
```

The bracket is `O(z)` near zero, so the left tail decays like `e^{(γ+1)u}`
*independently of γ*, and the leading term is already the exact `γ → 0` limit.
Measured relative error at order 1.999: 0.96 before, 2e−16 after. **This
superseded `numeric_fractionalDeriv_interpolation.py`, which is now deleted**
— spline-in-the-order was uncontrolled, cost 4× the work, and took its sign
from an endpoint.

**Truncation must scale with γ.** The left tail decays like `e^{γu}`, so
reaching tolerance needs `U ≳ log(1/tol)/γ` — about 55 at `a = 1.5` but 2763 at
`a = 1.99`. The current `initial_L = 10` with a doubling test cannot get there,
and `math.exp(u)` overflows above `u = 709` regardless, so `max_L = 1e4` is
unreachable.

**The `tol` default is the binding constraint before the range is, and a
tighter one is a partial fix — deliberately deferred, not overlooked.** `tol`
governs when the `L`-doubling loop stops *widening*, so despite its name it
controls the truncation range rather than the quadrature precision. Measured
against the closed-form Gamma reference, relative error at each setting:

| order | `t` | `tol=1e-6` (default) | `tol=1e-9` | `tol=1e-12` |
|-------|-----|----------------------|------------|-------------|
| 4.5   | −14 | **2.9e−06**          | 1.0e−15    | 1.0e−15     |
| 2.5   | −5  | 3.6e−10              | 1.7e−15    | 1.7e−15     |
| 1.5   | −50 | 1.2e−06              | 5.5e−11    | 9.3e−15     |
| 4.5   | −30 | 1.5e−06              | 2.8e−10    | **2.1e−10** |
| 1.5   | −1  | 4.6e−16              | unchanged  | unchanged   |
| 0.5   | −1  | 7.9e−16              | unchanged  | unchanged   |

So tightening `tol` alone repairs the `a = 4.5, t = −14` case outright and
improves two others by four to six orders of magnitude, at roughly 2–3× the
runtime on the calls that need it and no cost elsewhere. **It is not
sufficient**: `a = 4.5, t = −30` plateaus at 2.1e−10 however tight `tol` goes,
because the stopping rule compares consecutive iterates and that underestimates
the remaining tail when convergence is slow. Tightening `tol` makes the loop
widen for longer; it does not make the rule correct.

The decision (owner's call, taken deliberately) is **not** to ship the `tol`
change as an interim fix, but to let the fixed-grid kernel supersede it —
that kernel chooses its range from `γ` directly rather than discovering it by
doubling, so it removes the stopping rule instead of tuning it. Both cases are
carried as `xfail(strict=True)` records, parametrised over `t = −14` and
`t = −30`, so the kernel work cannot claim the easy one and quietly leave the
other.

**`logminus` is wrong for small gaps.** It implements only the `log1p` branch.
The standard two-branch form (Mächler 2012, the `Rmpfr` `log1mexp` vignette)
switches at `log 2`: use `log(−expm1(−a))` below and `log1p(−exp(−a))` above.
Measured: the current version returns `−inf` at a gap of 1e−17 and is wrong by
3.6e−09 at 1e−10 — and this function is used to form incomplete-MGF
differences, which is exactly the small-gap regime.

**Use `lambdify(..., modules=["scipy", "numpy"])`, never `"numpy"` alone.**
Measured speedup over `subs().evalf()` in a loop: ~6400×, at a cost of ~3 ulp.
`"numpy"` alone *fails* on `lowergamma`, `uppergamma`, `polygamma` and `Ei`,
all of which appear in this package's priors. Cache the compiled function on a
structural key and pass hyperparameters as arguments rather than substituting.

**Invert the CDF in log space.** Solve `log F(x) = log p`, switching to the
complement above the median. `F(x) − p` is identically zero wherever `F`
underflows, so no bracketing method can converge there; `log F` stays finite to
`e^{−745}`. This mirrors SciPy's own `ilogcdf` design.

**Bell recurrence conditioning.** The recurrence and the explicit partition sum
are numerically equivalent — the recurrence wins on cost, not accuracy. The
useful runtime diagnostic is the cancellation ratio `Σ|term| / |Σ term|`: it is
1.0 for priors whose CGF has one-signed derivatives (gamma, exponential — exact
to machine precision even at order 40) and ~3e6 for alternating ones such as
uniform, where double precision degrades from order ~12.

---

### Deferred decisions

- Make `jax` and `pandas` optional rather than import-time-required.
- Replace the `(log_abs, sign)` return convention with a small result type.
- Remove the `sys.modules["mgf2post"]` alias in `__init__.py`.
- Anglicise internal naming.
