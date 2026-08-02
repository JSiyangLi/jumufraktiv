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
   Nothing else may change it — not which backend ran, not whether the input
   was scalar or an array. Those are what the principle was written against,
   and they are invisible to a caller.

   *Which* of the two shapes a given quantity uses is fixed by whether that
   quantity can be negative, and is stated in its docstring. Every quantity
   this package returns is non-negative by construction — `θ > 0`, so
   `Dᵃ M(t) = E[θᵃe^{tθ}] > 0`, and the evidence, density, CDF, MGF, predictive
   and raw moment inherit it — *except* a central moment of odd order, which
   genuinely is signed: a Uniform(0.5, 2) prior with Poisson counts pushes the
   posterior against the upper endpoint and gives `μ₃ = −0.0219`, where a Gamma
   prior gives `+0.0741`.

   So `post_central_moment` returns a pair and everything else returns the log
   alone, which is the convention of `numpy.linalg.slogdet` and of
   `scipy.special.logsumexp(return_sign=True)`. A negative value anywhere else
   is a numerical failure rather than an answer, and is **raised** rather than
   reported through a sign the caller must remember to check.

3. **Tuple-vectorisation principle.** Evaluation points are the *pair* `(t, u)`.
   Both are broadcast to a common shape and evaluated as one batch. A function
   that accepts array `t` must accept array `u` and broadcast the two.

   *"As one batch" is a claim about cost, not only about shape.* Until PR 9 the
   package satisfied the shape half and not the cost half — correct answers,
   one adaptive quadrature per point — and no test could tell, because every
   test asserted values. `tests/test_batch_evaluation.py` now counts calls to
   the prior's density instead: a loop makes twenty times as many for twenty
   points, and the guard allows four. Wall-clock would have been the obvious
   assertion and a bad one, being either flaky or too loose to mean anything.

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

**Three readers want three different things, and only one of them reads the
docstring.** This is worth stating because the audit repeatedly got it wrong,
in one consistent direction: writing the reviewer's explanation into the API
reference.

| reader | wants | where it goes |
|--------|-------|---------------|
| someone calling `help(f)` or reading the Sphinx page | what it does, what the arguments mean, and any caveat that changes how they would call it | the docstring |
| someone editing the next line | why the code is shaped this way, and what a plausible "simplification" would break | a short inline `#` comment |
| someone reviewing the change | what was wrong, by how much, and what fixed it | the commit message, the PR body, `CHANGELOG.md` |

So a docstring carries the contract and any limitation that is **still true**
— `tol` is the quadrature's relative tolerance; the incomplete-MGF derivative
is untrustworthy below `u ≈ 1e-2`; the sign is always `+1` because the
integrand is positive. It does **not** carry the repository's own history:
no PR numbers, no "this used to…", no before/after tables, no defence of a
decision addressed to a reviewer, no "recorded here for the owner". Those are
facts about an unreleased package's past and tell a caller nothing.

The failure mode is self-compounding, which is why it needs a rule rather than
taste. Every PR that leaves "broken until PR N" in a docstring leaves it there
permanently, so the reference manual becomes an archaeology dig through
defects no user ever saw. Module-level docstrings may carry design rationale,
since documenting a module's reason to exist is their job — but the same ban
on PR bookkeeping applies to them.

Rationale that genuinely prevents a regression is kept, as an inline comment
next to the code it protects. `int(round(order))  # noqa: RUF046` with one
line saying why the cast is not redundant is the model: it is short, it sits
where the mistake would be made, and it is invisible to `help()`.

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
# Documentation. `-W` makes any warning an error, which is what CI enforces,
# and `-E` discards the saved environment so every page is re-read -- Sphinx
# skips unchanged documents otherwise, and a warning is emitted during the
# read it is skipping, so an incremental build reports clean for a page that
# is not. Drop both while iterating; run them before pushing.
sphinx-build -W -E -b html docs docs/_build/html

# The docstring examples. Run from the repository root: the fixture supplying
# `deriv` and `prior` lives in the root conftest.py precisely because this
# command collects from jumufraktiv/ and never loads tests/conftest.py.
pytest --doctest-modules jumufraktiv/MGFDerivative_class.py \
                         jumufraktiv/mitMGFprior_class.py
```

**The notebooks are run by hand, not by the suite.** `pytest` checks them
statically — every code cell parses, none carries stored output, none uses a
retired spelling — because executing `ParetoPumpFailureExample.ipynb` takes
tens of minutes: the Pareto MGF is written with `expint`, which no compiled
backend provides, so it stays on the exact symbolic path over grids of several
hundred points. Execute both before changing anything they exercise:

```bash
python -c "
import nbformat, pathlib
from nbclient import NotebookClient
for path in sorted(pathlib.Path('notebooks').glob('*.ipynb')):
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=1800, allow_errors=True,
                   resources={'metadata': {'path': 'notebooks/'}}).execute()
    bad = [(i, o['ename']) for i, c in enumerate(nb.cells)
           for o in c.get('outputs', []) if o.get('output_type') == 'error']
    print(path.name, bad)
"
```

`allow_errors=True` is deliberate: without it the run stops at the first bad
cell and reports nothing about the rest. Two cells raise *on purpose* — a
symbolic moment order and an out-of-range negative one — and both catch and
print the refusal rather than letting it stop the notebook, so a clean run
reports no error cells at all.

**Timings live outside the suite.** `tests/benchmarks/bench_vectorisation.py`
measures cost per evaluation point and cost per density call, and pytest does
not collect it — the filename does not match `test_*.py`. Run it directly:

```bash
python tests/benchmarks/bench_vectorisation.py
```

A wall-clock assertion inside the suite would vary with the machine, so it
would be either flaky or so loose as to assert nothing. Where a cost property
*must* be asserted, assert it structurally — the vectorisation guard counts
calls to the prior's density, which is a property of the algorithm rather than
of the box it runs on.

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

**There are two conftests, and which one a fixture belongs in is decided by
what collects it.** `pytest --doctest-modules jumufraktiv/...` collects from
the package directory, so it never loads `tests/conftest.py` — a fixture the
docstring examples need is therefore invisible to them from there, and the
examples fail with `NameError` while the whole suite stays green. The
repository-root `conftest.py` is an ancestor of both trees and holds the
doctest namespace for that reason; `tests/canonical.py` holds the Gamma/Poisson
problem both conftests build on, so there is one copy rather than two that
agree by inspection.

| File | Covers |
|------|--------|
| `conftest.py` (root) | the doctest namespace, for `--doctest-modules` |
| `test_incomplete_mgf.py` | the iMGF for every prior, and the four methods built on it |
| `test_mgf_domain.py` | where the posterior MGF converges, and the origin |
| `tests/canonical.py` | the Gamma/Poisson problem, shared by both conftests |
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
| `test_packaging.py` | what the built sdist and wheel contain, and whether the sdist's suite collects |
| `test_documentation_runs.py` | the README executes and renders, the notebooks parse and carry no stored output, the docstring examples run as CI invokes them, and no class docstring lists a member autodoc already documents |
| `test_packaging.py` | what the wheel and the sdist contain, and the citation file against the CFF schema |

**`test_known_broken.py` is the mechanism that keeps this document honest.**
Each test asserts the *correct* behaviour and is marked `xfail(strict=True)`,
so the suite stays green while the defect exists — but the moment a PR fixes
one, the test XPASSes and *fails* the build. That forces the fix to be recorded
both there and in the "Known-broken" list below. When you repair something,
expect a red build and remove the marker; do not weaken the assertion.

**Lint debt: there is none.** `pyproject.toml`'s `per-file-ignores` for
`jumufraktiv/**` is now an empty list, so the library is held to exactly the
rules `tests/` is. An entry added there from now on is a *new* exemption rather
than an inherited one, and should be argued for.

Two of the codes cleared last were filed as cosmetic and were not, which is the
part worth remembering:

- **`RUF001-003` (ambiguous unicode) hid a real defect for the whole audit.**
  Of 520 flagged characters, 336 were `‑` U+2011 NON-BREAKING HYPHEN standing in
  for `-`, and **41 of those sat inside error messages**, where a look-alike
  hyphen silently defeats a grep or a copy-paste search for the message text.
  The rest are Greek letters carrying the mathematics. Blanket suppression
  could not tell the two apart; `lint.allowed-confusables` names the Greek
  letters and the minus sign, so the rule can stay on.

- **`E501` was deferred on a reason that was wrong twice.** It said fixing the
  99 over-length lines meant running `ruff format` over the library, and that
  this would bury the real diffs. The second half is true, which is why the
  formatter is still its own scheduled change. The first half is not — the
  lines were rewrapped directly — and `ruff format` would not have fixed them
  anyway: it leaves 40 of the 99, because it does not reflow strings,
  docstrings or comments, which is where 68 of them lived.

The general lesson is the one this audit keeps relearning: **a suppression
records that nobody has looked, not that there is nothing to find.**

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
| 3 | 8 | Module layout, dead code, and the diagnostics policy | **merged** |
| 4 | 9 | Vectorisation, and the cost of a density call | **merged** |
| 4 | 10 | Caching and dispatch | **merged** |
| 5 | 12 | Public API surface, the prose sweep, and the lint baseline | **merged** |
| 5 | 12b | The moment domain at `t = 0`: the analytic tail | **merged** |
| 6 | 13a | The front door: README, notebooks, and the docstring examples | **merged** |
| 6 | 13b | Quantities that come back wrong without saying so | **merged** |
| 6 | 13c | Documentation infrastructure and the CHANGELOG catch-up | **merged** |
| 6 | 13d | Packaging: what the sdist and wheel actually contain | **merged** |
| 6 | 13e | The API reference read against the code; `-W` in CI | **in review** |
| 6 | 14 | Array-valued orders, the Pareto `expint` path, `ruff format` | planned |

**Wave 6 is split by *who notices the defect*, which is a different axis from
the earlier waves.** They were split by blast radius — does this change numbers
that currently look right? Nothing in wave 6 changes a number that is correct
today, so that question does not separate them. What does is the reader each
defect reaches:

- **13a** is everything a new user meets before they write any code of their
  own: the README, the two example notebooks, and the `Examples` section of
  every public method. These are the claims most likely to be read and least
  likely to be run, so they were the ones stating things the code does not do.
- **13b** is the opposite: quantities a user gets *without* an error, which are
  wrong. They cannot be found by reading, only by computing a reference.
- **13c** and **13d** are read by contributors and by packaging tools rather
  than by callers.
- **13e** is what 13c made visible. Repairing the `automodule` directives took
  the API reference from 2 modules to 26, so roughly twelve times as much
  docstring mathematics rendered for the first time — and prose nobody could
  read was prose nobody had checked.

**13a's charter is that a claim nobody runs is indistinguishable from a claim
that is false**, and the defect that motivated it proves the point: the README
quick start — the PyPI landing page, the Sphinx front page, the first six lines
of code anyone runs — raised `TypeError` for a release, because `evidence()`
changed shape and the README did not. `twine check` renders a README; it never
executes one, so CI was green throughout. `tests/test_documentation_runs.py`
now executes the README's code blocks, renders it the way PyPI does, checks its
likelihood table against `LIKELIHOOD_REGISTRY`, and runs the docstring examples
the way CI invokes them.

**Reading the prose against the code turned up two defects in the code, and
running a notebook turned up a third — the same pattern PR 12 recorded.**

The third has the widest reach. All four symbolic paths — density, MGF, raw
moment, predictive — ended in `except Exception: raise RuntimeError(f"Symbolic
computation failed: {e}")`, which is right for an unexpected SymPy failure and
destructive for a deliberate refusal: the caller loses the type they would
catch, and the message saying what to do instead becomes a suffix. They now let
`NotImplementedError` through unchanged, and **only** that type — `ValueError`
and `TypeError` are what SymPy raises for its own conversion failures ("Cannot
convert expression to float" is a `TypeError`), so those still get the wrapper.
Letting all four through was tried first and the test caught it, turning a
labelled failure into a bare SymPy traceback. One wrapper also said "Falling
back to numeric." after a failure where nothing falls back.

`post_predictive` documented a route for
a symbolic *observation*; the branch existed and could never succeed, because
`a(y_new)` is the differentiation order and a symbolic order is refused
package-wide. It is now an explicit `NotImplementedError` at the entry point
rather than a refusal several frames down naming an internal symbol the caller
never supplied. And `post_central_moment` called order 1 "the mean" when it is
`E[Θ − E[Θ]] = 0`; the value was right and the summary was not.

The API reference's own numbers are worth recording as a measurement rather
than an anecdote. Twelve worked examples carried an expected value and ten did
not match the code: four were placeholders (`1.234567e+00` twice, `0.01234`,
`0.1234`), and six were plausible numbers — `0.8574` for a posterior CDF, a
Gamma(2,3) MGF at `t = −1` given as `0.8888888889` where `(3/4)² = 0.5625` in
three places, and `1.7777777778` where the answer is `0.4444444444` in two. An
eleventh printed a right value in a form NumPy 2 does not produce (`1.0` for
what renders as `np.float64(1.0)`). Every replacement is checked against a
closed form (SciPy's `gamma` and `nbinom`, or mpmath quadrature with the
density written out separately), never against the package.

**13e found that the API reference described twenty-five members twice, and
the mechanism is worth stating because the obvious guess is wrong.** It is not
`autoclass` colliding with an `automodule`, and it is not two directives in
`api.rst`: a nine-line file with one class and one `autoclass` reproduces it.
**Napoleon renders the NumPyDoc `Attributes` and `Methods` sections into
`.. attribute::` and `.. method::` directives of its own**, so a name that
`:members:` also emits is registered twice. That is the whole of it.

Two corollaries decide what to do about it, and both were measured rather
than reasoned:

- *A section entry for a name autodoc cannot see is harmless.* An attribute
  assigned in `__init__` with no class-level declaration is invisible to
  `:members:`, so the section is its only documentation and there is nothing to
  collide with. `MGFDerivative` has seven of those and they never warned, which
  is why its `Attributes` section stays while its `Methods` section goes.
- *A heading napoleon does not know is absorbed into the preceding section.*
  `Properties` is not one of its section names; standing alone it stays plain
  text, but placed after `Attributes` it became two further attributes — the
  published page carried members literally named `Properties` and `----------`.

Duplication was not the worst of it. A hand-written `Methods` table restates
signatures that autodoc reads from the code, so it drifts and nothing catches
it: **six of the twelve had**, `post_sample` never having learned about the
`rng` argument PR 12 added to make it reproducible, and `post_quantile`
listing two parameters of nine. The table is deleted rather than corrected,
because correcting it only resets the clock.

`tests/test_documentation_runs.py` now asserts the invariant directly — no
`Attributes` or `Methods` section may list a name in `dir(cls)` — and it was
confirmed to fail against the unfixed tree, naming all sixteen methods.

**Three docstrings listed their parameters in an order the signature does not
declare them in**, of 94 documented callables — `integerDeriv_numeric_jax`
most visibly, whose signature begins `(t, prior, order)` while its
documentation began with `order`. That is worth a test rather than a fix
because of what the ordering is *for*: reading a docstring against its
signature is the cheapest check available, and it only works side by side.
When the orders agree an added or removed parameter shows up as one
misalignment; when they do not, every line has to be matched by name and an
omission looks like nothing at all. `tests/test_documentation_runs.py` now
asserts it over the names common to both, so documenting a subset stays legal.

**Three parameters were documented as controls and are not read.** Found by
comparing every `Parameters` section against the real signature: of 131
callables, three had drifted, and chasing those turned up the inert ones.
`resolve_backend(prior=...)` is removed — it was left over from when the
choice between differentiating and taking the expectation was to be made
there, and that choice moved to `mgfDerivative`. The other two are kept,
because `solve_root` passes one argument list to every backend and the JAX
loops are `fori_loop`s that cannot exit on a data-dependent condition: `tol`
on the three JAX root-finders, and `tol` on the mpmath backend, where PR 6c
deliberately derived the truncation range from `dps` instead and only the
docstring was not told. All three now say so, and name what to adjust
instead.

**A stale non-editable copy of the package in `dist-packages` cost twenty
minutes and is worth recognising by its signature.** `docs/conf.py` puts the
repository root first on `sys.path`, so a build run from `docs/` reads the
working tree while anything run from elsewhere read the installed copy. The
symptom was a scratch build of a byte-identical copy of `docs/` reporting 24
warnings the real build did not — same command, same files, different answer.
That is the same hazard as the cached-ruff-verdict entry below: **a check that
resolves its subject differently from the thing it is checking is measuring
something else.** `pip install -e .` needs build isolation here, because
`pyproject.toml` uses PEP 639's `license = "MIT"` and the environment's own
setuptools is older than the 77 that requires.

**PR 12 grew well past its charter, and the growth was not scope creep but
consequence.** It was scoped as five recorded interface defects. Two things
enlarged it, both worth recording because neither was on any list.

The first was an owner instruction: docstrings and comments had been
accumulating this repository's own history, which belongs in commit messages.
Sweeping that out is prose work, but reviewing the sweep meant reading every
rewritten passage against the code, and **three of the findings were defects in
the code rather than in the prose** — a sentence stops matching what it
describes when either one is wrong. `pareto_cgf` and `pareto_mgf` returned
`nan` at every argument; the dispatcher's speed claim was backwards; and
`post_predictive(individual=False)` had never worked on a symbolic posterior.

The second was measurement. Explaining the sign convention to the owner needed
a worked example, and the example raised `ValueError` where it should have
returned a number — which is how the `auto` routing defect below was found.

The pattern is the same both times: **prose that had to be justified, and a
number that had to be produced, each turned up a defect that reading the code
alone had not.**

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

**Wave 4's row was one line and should not have been.** "Vectorisation" was
recorded with no itemised findings, and measurement found the largest
user-visible defect left in the package. The default route cost the same per
evaluation point no matter how many were asked for — 255.9 ms at one point,
254.9 ms each at twenty — so a hundred-point posterior density curve, the most
obvious thing anyone does with this package, took 26 seconds.

The **tuple-vectorisation principle** was honoured in *shape* and not in
*cost*: array `t`, array `u` and broadcasting all returned correctly shaped
answers while fifteen per-point Python loops ran underneath. Worth stating
plainly, because a principle the code satisfies only in its return type is
one the tests will keep confirming while the behaviour is absent.

Three causes, and the two that looked smallest were the largest.

*Priors rebuilt a SciPy distribution per call.* Three of the four wrote
`pdf_func=lambda x: stats.<dist>(params).pdf(x)`, which builds a frozen
distribution — and formats its docstring — on **every call**, in the innermost
function in the package: 79% of the route's runtime was inside
`rv_frozen.__init__`. Hoisting it was 9.2× on its own.

*The symbolic backend substituted point by point.* `sp.subs` was 97.6% of the
`scipy` route. Compiling with `lambdify` instead — the remedy this document
had recorded and nobody had applied — took that route from 2054 ms per point
to 2.0. See "Numerical policy" for the part of the recorded advice that turned
out to be incomplete.

*The quadrature ran once per point.* Batching it onto a common interval took
the default route to 5.5 ms per point at a hundred points.

Accuracy improved throughout, from a recorded worst case of 1.17e-11 to
7.85e-15. Measured per-point cost, before and after, at a batch of eight:

| route | before | after |
|-------|--------|-------|
| `scipy` (fixed grid) | 2054.5 ms | 2.0 ms |
| `symbolic` (integer) | 1.1 ms | 0.02 ms |
| `auto` (fractional) | 255.9 ms | 9.7 ms |
| `jax` (integer) | 657.7 ms | 66.6 ms |
| `bell` (integer) | 8.5 ms | 1.7 ms |
| `auto` (integer) | 24.0 ms | 7.8 ms |
| array-valued order | 37.0 ms | 38.1 ms — **still flat** |

The full test suite went from 623 s to 67 s.

**What is left.** The array-valued-order path still dispatches each element
separately, and it is the hard one: different orders may resolve to different
backends, so they cannot share a call. The Pareto prior stays on the exact
symbolic path because its MGF uses `expint`; that is correct but ~1590 ms per
point. And `uniform` and `heaviside` route their far-tail nodes to the exact
path because those underflow to zero in float64 — skipping them is very
probably safe, since a node that underflows contributes nothing to a log-space
accumulation, but "very probably" is not a verification and the check was not
done.

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
| `use_loop` rejected by the constructor though it is a real backend option; `return_log` must be reserved | **PR 12** — `use_loop` is moot: it selected a scalar loop inside the adaptive scipy kernel PR 8 deleted. `return_log` still stands, joined by the wrong-backend option case below |
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

Do not build on these paths; do not paper over them. **Every entry here has a
matching `xfail(strict=True)` test in `tests/test_known_broken.py`, and that
correspondence was re-established by measurement rather than assumed** — it had
drifted to four entries against one marker, three of them citing merged PRs, so
nobody could tell a live defect from a stale record. Reconciled by rerunning
each reproduction:

| entry | recorded | measured | outcome |
|---|---|---|---|
| alternating CGF, `symbolic` | wrong sign at order 30 | 5.4e+09, sign flipped | live; xfail added |
| mpmath `dps` above ~20 | a return-convention limit | unchanged | not a defect; moved to "Deferred decisions" |
| tail at `t = 0`, `scipy` | 6.1e-01 at order 1.99 | — | live; already carried three xfails |
| incomplete MGF below `u ≈ 1e-2` | 45 nats off at `u = 1e-6` | **0.000 nats** | fixed by PR 6b; entry deleted |

The general lesson is the one the mechanism exists to prevent: a list of known
defects that is not executable becomes a list of *claims* about defects, and
ages in both directions at once — it kept an entry that had been fixed and lost
one that had not.

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

  **Re-measured, and it is now reachable only by asking for the backend by
  name.** Against Uniform(0.5, 2) at `t = −1`, with the exact value from mpmath
  at 80 digits:

  | order | `method="symbolic"` | `method="auto"` (the default) |
  |---|---|---|
  | 12 | 2.2e-10 | 2.6e-15 |
  | 16 | 2.0e-06 | 3.1e-15 |
  | 20 | 1.8e-02 | 5.1e-15 |
  | 30 | 5.4e+09, sign flipped | 7.3e-15 |

  So the route PR 6c made the default already computes it correctly, and what
  is left open is an interface question rather than a numerical one: should an
  explicitly requested backend be silently replaced by a better one, or should
  it fail as asked? That is why this stays recorded rather than being closed.
  *(unscheduled — interface decision)*
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
- **The differentiating route loses the tail at `t = 0` near a prior's moment
  bound.** `method="scipy"` gives 2.0e-04 relative error at order 1.5 against
  Pareto(α=2), 8.2e-02 at 1.9 and 6.1e-01 at 1.99, where the exact answer is
  `E[Θᵃ] = 2/(2−a)`. The default `auto` route is exact to 6e-16 across the same
  range; the mechanism differs, so the repair does not carry across. The
  expectation route integrates `θᵃ p(θ)` and can have its tail supplied from
  `max_finite_moment`; the fixed grid integrates `M^{(n+1)}` over the
  fractional-integral kernel, where that correction has no counterpart.
  *(unscheduled)*

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

**Moments near a prior's `max_finite_moment` are exact at `t = 0`.** The guard
admits any order strictly below the bound, which is true of the mathematics and
was not true of the quadrature: at the origin there is no `e^{tθ}` to force
decay, so the integrand is `θᵃp(θ)`, falling off only polynomially for a
heavy-tailed prior. Against Pareto(α=2) at `t = 0`, relative error before and
after:

| order | before | after |
|---|---|---|
| 1.5 | 7.2e-07 | 3.3e-17 |
| 1.9 | 2.2e-02 | 1.0e-16 |
| 1.99 | 2.7e-01 | 6.0e-16 |
| 1.999 | ~0.5 | 5.7e-16 |

**Integrating harder cannot achieve this, which is why the tail is supplied
rather than computed.** Reaching a relative 1e-10 at order 1.99 needs the
integral carried to `θ ~ 1e1000`, and even at float64's limit of 1.8e308 a
thousandth of the answer is still outside — the remaining mass sits where `θ`
cannot be formed at all. A finite `max_finite_moment` is the prior stating that
`E[Θᵃ]` diverges at `a = α`, which is the statement that `p` has tail index
`α`; reading the constant off the density at the bracket's end gives
`∫_T^∞ θᵃp(θ)dθ = p(T)T^{a+1}/(α−a)`. For Pareto that is exact rather than
asymptotic, since its density *is* a power law.

**`MGFDerivative(method="auto")` now reaches the same backend as
`mgfDerivative(method="auto")`.** It did not.
`_build_derivative` consults `resolve_backend`, which encodes the backend
matrix and answers `symbolic` for `auto` at an integer order. But
`mgfDerivative` applies a *second* rule that `resolve_backend` does not
model — with a concrete `t`, prefer the expectation route — so the class held
a symbolic expression and substituted into it, evaluating exactly the route
PR 6c demoted. Measured against mpmath at 50 digits, Uniform(0.5, 2) with
Poisson data:

| data | `a` | `mgfDerivative(auto)` | `MGFDerivative(auto)` |
|---|---|---|---|
| [1, 2, 3] | 6 | 1.3e-16 | 1.3e-16 |
| [8, 9, 10] | 27 | 7.9e-16 | **5.4e-07** |
| [20, 20, 20] | 60 | 2.4e-16 | **`ValueError`, would not construct** |

The last row is the sharpest: the class could not build a posterior the
dispatcher computes to sixteen digits, because `_store_result` correctly
refuses a negative evidence and the substituted route produced one.

Invisible for the whole audit because **every class-level test uses a Gamma
prior**, whose MGF derivatives have one-signed terms and therefore cannot
cancel — the same hazard recorded below under a different name. The 240-case
sweep that established the default measured the *dispatcher*; the class was
assumed to inherit it.

**The incomplete-MGF derivative's lower tail is repaired.** It was recorded as
wrong by 45 nats at `u = 1e-6`, with the sign flipped, and trustworthy only
above `u ≈ 1e-2`. Re-measured against `scipy.stats.gamma.logcdf` for the exact
Gamma(8, 6) posterior of the canonical problem, `post_cdf` now agrees at every
`u` tested:

| `u` | package | exact | nats apart |
|---|---|---|---|
| 1e-6 | −106.7946 | −106.7946 | 0.000 |
| 1e-4 | −69.9538 | −69.9538 | 0.000 |
| 1e-2 | −33.1652 | −33.1652 | 0.000 |
| 1e-1 | −15.2227 | −15.2227 | 0.000 |

PR 6b's fixed-grid kernel closed it and the known-broken list was not updated,
which is how the entry survived three PRs after its own repair.

**The posterior CDF now exists for every registry prior.** `post_cdf` needs the
prior's incomplete MGF `M(t, u) = ∫_{-∞}^{u} e^{tx}p(x)dx`, and
`post_quantile`, `post_interval` and `post_sample` are all built on it, so a
prior without one loses four public methods at once, on every backend. Two of
the four registry priors had none, and the refusal said "Prior does not support
incomplete MGF (iMGF)" — which reads as a statement about the mathematics and
was a statement about the module. Both integrals are elementary:

```
uniform(a, b) :  ∫_a^{min(u,b)} e^{tx}/(b-a) dx = (e^{tm} - e^{ta})/(t(b-a))
heaviside(k)  :  ∫_k^u e^{tx} dx                = (e^{tu} - e^{tk})/t
```

Both are formed with `logminus` and ordered by the sign of `t`, for the reason
`uniform_cgf` already was: below the origin neither factor is positive on its
own and the signs cancel only in the ratio. Measured against mpmath at 40 dps
with the density written out separately, the worst relative error is 1.3e-15
over 42 `(t, u)` pairs for uniform and 9.1e-16 over 24 for heaviside, and the
posterior CDF is right to 1.3e-13 or better on all four backends.

**`post_mgf` returned the value of a formula outside the domain where that
formula is the MGF.** The Gamma(8, 6) posterior's MGF is `(6/(6-r))⁸`, finite
only for `r < 6`. Past that the eighth power keeps the sign positive, so the
package returned plausible numbers where the answer is infinite:

| `r` | returned | true value |
|---|---|---|
| 6.1 | 1.68e+14 | ∞ |
| 10 | **25.63** | ∞ |
| 100 | **2.76e-10** | ∞ |

The repair is a prior-level declaration in the same style as
`max_finite_moment`, because where an MGF converges is a property of the prior
and not of the data: `mgf_finite_below` is the supremum of `t` with `M(t) < ∞`
— `β` for gamma, `∞` for uniform, `0` for pareto and heaviside. `post_mgf`
evaluates at `t = r - b`, so it refuses any `r` that puts the point past the
bound, and the message names the largest admissible `r`.

**`t = 0` needed its own treatment, and got the answer wrong in two different
ways.** At `r = b` the exponential is 1 and the value reduces to `E[Θ^a]`.
Every prior but gamma returned `nan`: uniform because its MGF carries `t` in a
denominator, so `subs` sees 0/0 where the value is finite; pareto and heaviside
because the moment genuinely diverges and nothing refused. Both now behave —
uniform returns 141.404114, exact to 1.4e-15 against mpmath, and the other two
raise naming the order and the bound.

**`sp.limit` is not the fix for the first of those, and that is worth
recording.** For the uniform prior at order 6 it returns `∞` where the value is
`E[Θ⁶] = 12.19`. A limit that returns a wrong number is the same class of
defect being repaired, so the origin routes through the expectation backend
instead, which integrates `E[Θ^a]` directly.

**Dimensionality is now checked in one place.** The `ndim != 1` guard lives
inside the shared `like_stats/_common.py::_extract_1d`, which every entry point
routes its data *and* its known parameters through. All twenty-eight `ready*`
and `bereit*` functions reject 2-D input, where previously ten of twenty-eight
did. That it is one guard rather than fourteen agreeing ones is the point:
byte-identical copies are exactly how the four unguarded modules went
unnoticed.

---

### A testing hazard this repository has already hit four times

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
- **`ruff`'s cache is keyed on file contents, and isort's first-party
  detection is not a function of file contents.** Adding a repository-root
  `conftest.py` reclassifies `conftest` from third-party to first-party in
  every test that imports it, so thirteen files needed their import blocks
  regrouped — and `ruff check .` reported "All checks passed" locally against
  cached verdicts for files whose text had not changed. CI, with a cold cache,
  found all thirteen. **After adding or moving a file, run
  `ruff check --no-cache .` before believing a clean local result.**

The general rule: assert the property itself, and confirm a new test fails
against the unfixed code before trusting that it passes against the fixed one.
The third instance adds a corollary — **choose fixture values that are not
defaults**, since a parameter tested only at its default value is a parameter
whose plumbing is untested. The fourth adds another: **a cached verdict is a
statement about the last run's inputs, not about the current tree.**

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

One consequence beyond accuracy: the route needs only `p(θ)`, never the MGF,
which is what makes sequential updating work for numeric backends. The prior
`to_prior_object` builds carries a density and no `mgf_sym`, so no
differentiating backend could consume it.

**The route is slower, not faster, and the entry that used to say otherwise
was stale rather than wrong when written.** The `cost, order 1.5` row above
was measured before PR 9, which took the fixed-grid kernel from 2054 ms per
evaluation point to 2.0 and inverted the comparison. Nobody re-measured, so
`5–8× faster` survived in this document, in the dispatcher's own comment and
in a docstring, all sourced from the same superseded run. Re-measured against
`method="scipy"` on the current tree, Gamma(2, 3) at order 1.5:

| evaluation points | `auto` (expectation) | `scipy` (fixed grid) | ratio |
|---|---|---|---|
| 1 | 53.98 ms | 2.24 ms | 24.1× slower |
| 8 | 14.68 ms/pt | 3.61 ms/pt | 4.1× slower |
| 20 | 11.32 ms/pt | 3.21 ms/pt | 3.5× slower |

Accuracy is therefore the whole of the case for the default, and it is bought
with time. That is still the right trade for a default — never catastrophically
wrong beats fast — but a caller who knows their prior's CGF has one-signed
derivatives, as gamma and exponential do, can ask for `scipy` and keep the
speed. The gap narrows as the batch grows, so it is worst for a single point.

The general lesson is worth more than the number: **a measured comparison
becomes a claim about two moving things, and speeding one of them up
invalidates it silently.** Nothing failed when PR 9 inverted this; the sentence
simply went on being read.

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

**Applied in PR 9, and the list above turned out to be incomplete.** The
`scipy` fractional route spent 97.6% of its time in `sp.subs` — 5,024
substitutions for two evaluation points, because the fixed-grid kernel hands
`mgfDerivative_integer` an `(n_nodes × n_points)` array and it took the
elements one at a time. Compiling instead took that route from 2054 ms per
evaluation point to 2.0.

But `modules=["scipy", "numpy"]` is necessary and *not sufficient*: the Pareto
prior's MGF is written with `expint`, the generalised exponential integral,
which **neither** module provides. SymPy compiles it without complaint and the
result raises `NameError` on the first call. So a compiled function must be
**probed** — called once, at setup, on a value the caller knows is in domain —
and the expression evaluated symbolically if the probe fails. `cached_lambdify`
returns `None` for such an expression and caches that verdict, so the failure
is paid once rather than per evaluation.

Two other things the compiled path must not do. It works in float64, so it
cannot represent `M^(301)`; elements that overflow, underflow or return NaN
fall through to the exact symbolic path, which is why the fast path may only
*skip* work and never change an answer. And Pareto consequently stays on the
exact path entirely — correct, and still about 1590 ms per point, which is
where the remaining headroom is.

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
  Now needed in one place rather than several: only `post_central_moment`
  returns the pair, since it is the only quantity that can be negative.
- Anglicise internal naming.

The `sys.modules["mgf2post"]` alias is **removed**, and it was more than a
tidiness item. Assigning into `sys.modules` claimed a name this project does
not own: importing `jumufraktiv` made `import mgf2post` return this package
process-wide, shadowing any genuinely different distribution of that name. For
a package heading to public PyPI that is a hazard rather than a courtesy, and
nothing in the package or the suite used it.
