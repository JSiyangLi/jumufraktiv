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

| Order type   | Valid `method`                  | `auto` resolves to |
|--------------|---------------------------------|--------------------|
| symbolic     | `symbolic`                      | `symbolic`         |
| integer      | `symbolic`, `bell`, `jax`       | `symbolic`         |
| fractional   | `scipy`, `mpmath`, `symbolic`   | `scipy`            |

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
pip install -e ".[dev]"      # development install
pytest                       # test suite            (added in PR 2)
ruff check . && ruff format --check .   # lint        (added in PR 2)
cd docs && make html         # build documentation
```

---

## Audit status

The repository is undergoing a staged audit. Work lands one PR at a time.

| Wave | PR | Scope | Status |
|------|----|-------|--------|
| 0 | 1 | Repo hygiene, packaging metadata, project files, this document | **in review** |
| 0 | 2 | pytest + Hypothesis harness, CI, lint config | planned |
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

Do not build on these paths; do not paper over them. Each has a PR assigned.

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
  formed before the `subs` call. *(PR 5)*
- **`post_sample` is not reproducible** — unseeded `np.random.rand`, no `rng`
  argument. *(PR 12)*

### Deferred decisions

- Make `jax` and `pandas` optional rather than import-time-required.
- Replace the `(log_abs, sign)` return convention with a small result type.
- Remove the `sys.modules["mgf2post"]` alias in `__init__.py`.
- Anglicise internal naming.
