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
| 0 | 2 | pytest + Hypothesis harness, CI, lint config | **merged** |
| 1 | 3 | Import and registry integrity | **merged** |
| 1 | 3b | Constructor keyword-argument integrity | **in review** |
| 1 | 4a | Fractional-order construction and dispatch | planned |
| 1 | 4b | Fractional-order numerical accuracy | planned |
| 1 | 4c | Symbolic fractional backend | planned |
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
- **Fractional orders are truncated in the array path.** `int(o)` at
  `derivativeDispatch.py:671` silently rounds; `post_predictive` uses this
  path. *(PR 4)*
- **The symbolic fractional backend omits the `1/Γ(γ)` prefactor** that all five
  numeric sites apply, so its result is `Γ(γ)` times too large — 77% at order
  0.5. It also currently returns `None` for the Gamma prior because SymPy's
  `laplace_transform` raises internally. *(PR 4)*
- **Near-integer orders lose accuracy.** Above the interpolation threshold
  (fractional part > 0.95) the dispatcher switches to a 4-point cubic spline in
  the order, which is *less* accurate than the plain quadrature just below the
  threshold. See "Numerical policy" for the exact fix. *(PR 4)*
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
supersedes `numeric_fractionalDeriv_interpolation.py`, which should be retired
rather than repaired** — spline-in-the-order is uncontrolled, costs 4× the work,
and takes its sign from an endpoint.

**Truncation must scale with γ.** The left tail decays like `e^{γu}`, so
reaching tolerance needs `U ≳ log(1/tol)/γ` — about 55 at `a = 1.5` but 2763 at
`a = 1.99`. The current `initial_L = 10` with a doubling test cannot get there,
and `math.exp(u)` overflows above `u = 709` regardless, so `max_L = 1e4` is
unreachable.

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
