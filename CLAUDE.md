# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

---

## What this package does

`jumufraktiv` performs Bayesian inference for the **MGF-marginalisable family**
of likelihoods: those whose joint density factorises as

```
L(θ; y) = c(y) · θ^a(y) · exp(−b(y)·θ)
```

for sufficient statistics `a`, `b` and a normalising factor `c`. For such a
likelihood the marginal likelihood is an `a`-th derivative of the prior
moment-generating function `M`, evaluated at `t = −b`:

```
p(y) = c(y) · Dᵃ M(t) |_{t = −b}
```

Because `a(y)` need not be an integer, the derivative is generally **fractional**
— that is the package's reason to exist. Every other quantity (posterior
density, CDF, quantiles, moments, predictive density, sequential updates) is
derived from the same object, which is why almost everything routes through one
dispatcher.

Parameters are assumed **strictly positive**.

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
                     │    └─ numeric_fractionalDeriv_scipy / _mpmath
                     │       / _interpolation       │
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

Which `method` is valid depends on the *order type*, and `mgfDerivative`
enforces it:

| Order type   | Valid `method`                  | `auto` resolves to | Works today |
|--------------|---------------------------------|--------------------|-------------|
| symbolic     | `symbolic`                      | `symbolic`         | **no** — see below |
| integer      | `symbolic`, `bell`, `jax`       | `symbolic`         | yes         |
| fractional   | `scipy`, `mpmath`, `symbolic`   | `scipy`            | **no** — see below |

The last column is not decoration. Only the integer row is reachable at
present: the symbolic-order row raises `TypeError`, and the fractional row
cannot be constructed through `MGFDerivative`. Both are listed under
"Known-broken" below with the PR that repairs them. Treat the first two rows as
the intended contract, not as a description of current behaviour.

For fractional orders, `integer_method` separately selects the integer backend
used *inside* the fractional integrator (`symbolic`, `bell`, `jax`).

An order counts as integer when `|order − round(order)| < int_tol`
(default `1e-12`).

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

**Naming.** Some internals use German (`mitMGFprior`, `bereit*` for per-element
statistics alongside `ready*` for aggregated ones). Anglicising the internals
while keeping the package name is planned for a later wave; until then, follow
the existing convention rather than mixing a third one in.

**Diagnostics.** Library code should use `logging` and `warnings`, not `print`.
A large amount of existing code still prints; do not add more.

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
pytest -m "not slow" -x -q       # quick pass
ruff check .                     # lint
ruff format --check tests/       # formatting (tests/ only, see below)
sphinx-build -b html docs docs/_build/html   # documentation
```

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
| `test_registry.py` | registry, prior container, custom-prior route, constructor validation |
| `test_known_broken.py` | every documented defect, as `xfail(strict=True)` |

**`test_known_broken.py` is the mechanism that keeps this document honest.**
Each test asserts the *correct* behaviour and is marked `xfail(strict=True)`,
so the suite stays green while the defect exists — but the moment a PR fixes
one, the test XPASSes and *fails* the build. That forces the fix to be recorded
both there and in the "Known-broken" list below. When you repair something,
expect a red build and remove the marker; do not weaken the assertion.

**Lint debt.** `pyproject.toml` carries an itemised `per-file-ignores` baseline
for the pre-audit library code — one entry per rule, each annotated with the PR
that removes it. It is a shrinking list, not a blanket suppression. `tests/`
has no exemptions. `F821` (undefined-name) and `F811` (redefined-while-unused)
are deliberately left blocking wherever possible, because they found real
defects rather than style issues.

`ruff format` currently runs over `tests/` only. Reformatting 13k lines of
library code would bury the audit's real diffs; that lands as its own PR in
wave 6.

---

## Audit status

The repository is undergoing a staged audit. Work lands one PR at a time.

| Wave | PR | Scope | Status |
|------|----|-------|--------|
| 0 | 1 | Repo hygiene, packaging metadata, project files, this document | **merged** |
| 0 | 2 | pytest + Hypothesis harness, CI, lint config | **in review** |
| 1 | 3 | Import and registry integrity | planned |
| 1 | 4 | Fractional-order path | planned |
| 2 | 5 | Symbolic-path correctness | planned |
| 2 | 6 | Numerical robustness | planned |
| 3 | 7 | De-duplicate `like_stats` | planned |
| 3 | 8 | Module layout and internal boundaries | planned |
| 4 | 9 | Vectorisation | planned |
| 4 | 10 | Caching and dispatch | planned |
| 5 | 11 | Diagnostics policy | planned |
| 5 | 12 | Public API surface | planned |
| 6 | 13 | Documentation infrastructure | planned |
| 6 | 14 | Docstring sweep | planned |

### Known-broken, scheduled for repair

Do not build on these paths; do not paper over them. Each has a PR assigned and
a matching `xfail(strict=True)` test in `tests/test_known_broken.py`, except
where noted as "no runtime repro".

- **Fractional orders cannot be constructed.** `MGFDerivative._build_derivative`
  calls `mgfDerivative(..., t=None)`, but the fractional branch requires `t`.
  Any non-integer `a` raises at construction — which includes `normal`,
  `halfnormal` and `maxwell-boltzmann` whenever *n* is odd. *(PR 4)*
- **Two unqualified imports always fail.** `derivativeDispatch.py:402` and
  `:791`. *(PR 3)*
- **The registry silently drops priors.** `registry.initialize` converts an
  import failure into a warning; `MGFdictionary/paretoMGF.py` imports `torch`
  eagerly, so a missing optional extra removes `pareto` *and* `uniform` from
  the registry. *(PR 3)*
- **`mitMGFprior.from_registry` does not initialise the registry**, so it fails
  in a fresh process unless some other registry function ran first. *(PR 3)*
- **Fractional orders are truncated in the array path.** `int(o)` at
  `derivativeDispatch.py:671` silently rounds; `post_predictive` uses this
  path. *(PR 4)*
- **`post_density` discards its hyperparameter substitution** — `log_prior` is
  formed before the `subs` call. *(PR 5, no runtime repro yet)*
- **The symbolic-order path is dead.** `integerDeriv_symbolic` rejects any order
  that is not a Python `int`, so `mgfDerivative` warns that it will return an
  analytic continuation and then raises `TypeError` — even for `sp.Integer(2)`.
  The symbolic row of the backend matrix is unreachable. *(PR 5)*
- **`post_cdf`'s symbolic branch references undefined names.** It uses `t_sym`
  and `u_sym`, but the module imports `t` and `u`. The resulting `NameError` is
  swallowed by a broad `except` and re-raised as a confusing `RuntimeError`.
  Found by `ruff` `F821`, which is why that rule stays blocking. *(PR 5)*
- **`post_quantile`, `post_interval` and `post_sample` fail for every prior.**
  `post_quantile` brackets from a lower bound of `1e-6`, where the
  incomplete-MGF derivative underflows and its computed sign flips negative,
  tripping the guard in `post_cdf`. The other two call it. *(PR 6)*
- **`post_cdf` has no domain validation on `u`.** `theta` is constrained
  positive, but `u = -1e-9` recurses to `RecursionError` and `u = -0.5` returns
  a probability greater than one. *(PR 6)*
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

### Deferred decisions

- Make `jax` and `pandas` optional rather than import-time-required.
- Replace the `(log_abs, sign)` return convention with a small result type.
- Remove the `sys.modules["mgf2post"]` alias in `__init__.py`.
- Anglicise internal naming.
