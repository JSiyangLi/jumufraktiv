# Contributing to jumufraktiv

Thank you for your interest. This document covers what you need to work on the
package; `CLAUDE.md` covers *why* the code is shaped the way it is, and is worth
reading before changing any of the mathematics.

## Getting set up

```bash
git clone https://github.com/JSiyangLi/jumufraktiv.git
cd jumufraktiv
pip install -e ".[dev]"
```

## The commands

```bash
pytest                           # the full suite
pytest -m "not slow" -x -q       # the quick pass, for the iteration loop
ruff check --no-cache .          # lint
ruff format --check tests/       # formatting (tests/ only, for now)

# The docstring examples. Run from the repository root: the fixture supplying
# `deriv` and `prior` lives in the root conftest.py precisely because this
# command collects from jumufraktiv/ and never loads tests/conftest.py.
pytest --doctest-modules jumufraktiv/MGFDerivative_class.py \
                         jumufraktiv/MGFPrior_class.py

# The documentation.
pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build/html
```

**Run the quick pass while working and the full suite before committing.** A
test earns the `slow` marker by costing more than about three seconds, which in
this package always means real quadrature. Marking is a scheduling decision and
never a statement about importance — several of the slowest tests are the most
valuable in the suite.

Two habits, learned the expensive way: run long suites in the background rather
than blocking on them, and do not run two at once. The quadrature is
single-threaded but the box is usually small, so concurrent runs make each other
slower and the timings meaningless.

**After adding or moving a file, run `ruff check --no-cache .`.** Ruff's cache
is keyed on file contents, and isort's first-party detection is not a function
of file contents — adding a file can reclassify an import in files whose text
did not change, and a cached run will not notice.

## Writing tests

**Assert mathematics, not recorded output.** A Gamma prior has a closed-form
MGF and all of its derivatives; a Gamma prior against a Poisson likelihood is
conjugate, so the entire posterior is a known Gamma. `tests/conftest.py` exposes
those references and most of the suite compares against them. A test that merely
pins today's number cannot tell a refactor from a regression.

**Never validate one of the package's paths against another.** An acceptable
reference is a closed form you write out yourself, or `mpmath` at high precision
with the density written out separately in the test. This matters more than it
sounds: several defects in this package survived for a long time because two
paths agreed with each other and both were wrong.

**Confirm a new test fails against the unfixed code** before trusting that it
passes against the fixed one. This has caught two false records in recent work,
both times an `xfail` that fired on a `NameError` from a missing import rather
than on the defect it claimed to document. After adding an `xfail`, read *why*
it xfailed — do not count `x` characters.

## Known-broken defects

`tests/test_known_broken.py` holds an executable record of every defect that is
known and not yet repaired. Each test asserts the **correct** behaviour and
carries `xfail(strict=True)`, so the suite stays green while the defect exists —
and the moment a change fixes one, the test XPASSes and *fails* the build. That
is deliberate: it forces the fix to be recorded there and in `CLAUDE.md`. If you
repair something, expect a red build and remove the marker. Do not weaken the
assertion.

## Adding a likelihood

Each `like_stats/X.py` exports exactly three functions:

- `readyX(data, **kwargs) -> {'a', 'b', 'log_c'}` — aggregated over the sample
- `eachX(data, **kwargs) -> {'a', 'b', 'log_c'}` — per-element arrays
- `cX() -> sympy.Expr` — the symbolic normalising constant

Take data and every known parameter through `like_stats/_common.py::_extract_1d`
rather than re-implementing the checks; fourteen byte-identical copies are
exactly how four modules came to be missing a dimensionality check while ten had
it. Register the new likelihood in `LIKELIHOOD_REGISTRY` **and** in the README's
table, which a test checks.

Whether a likelihood belongs to the family at all is a property of the
**parameterisation**, not of the distribution: Rayleigh is in the family in the
rate and not in the scale. State the parameterisation in the module docstring.
Theorem 4.1 of the reference gives the criterion — a likelihood is
MGF-marginalisable if and only if it admits a gamma conjugate prior.

## Adding a prior

A module in `MGFdictionary/` registers itself with `@register_prior("name")` and
returns `make_prior_spec(...)`. Discovery is automatic for any filename
containing `MGF`. Declare `max_finite_moment` and `mgf_finite_below` rather than
letting them default: both are properties of the prior that the inference layer
cannot work out for itself, and both exist because a prior that inherited a
silent default returned wrong numbers without an error.

## Style

- **Imports** are fully qualified within the package:
  `from jumufraktiv.symbols import t`.
- **Docstrings** are NumPyDoc, rendered by Sphinx. They carry the contract and
  any limitation that is still true. They do not carry the repository's own
  history — no PR numbers, no "this used to…", no before/after tables. Those
  belong in the commit message. Rationale that prevents a regression goes in a
  short inline comment next to the code it protects.
- **Diagnostics** use `logging` and `warnings`, never `print`. A test enforces
  this. Use `warnings.warn(..., stacklevel=2)` when the caller's result is
  affected, and `logger.debug` when recording which branch was taken.
- **Catch narrowly.** Never let a broad `except` turn a real failure into a
  warning or a silently wrong number.

## Pull requests

Descriptions are written for the reviewer, not the author. The people reviewing
this repository are experts in the statistics and the numerical analysis and are
not necessarily fluent in its Python internals, so:

- open with what changed for someone *using* the package, not with the private
  method that was wrong;
- introduce every internal term on first use;
- say what a number means before showing it — state the reference and what
  counts as agreement;
- name the mechanism, not just the symptom;
- flag what you did **not** fix, and why, in the same voice as the rest.

Mathematical statements are held to the standard of the paper. `(a(y), b(y))` is
*jointly* sufficient — never "the sufficient statistic `a(y)`" — and the
operator is the Liouville–Caputo derivative with lower terminal −∞, because that
is what the reference calls it.

## Reporting a discrepancy in a likelihood

All fourteen `like_stats` modules have been checked by the package's author. A
discrepancy found against an external reference is therefore far more likely a
parameterisation mismatch on your side than a defect, and the burden of proof
sits with the checker. Before reporting one: re-derive the factorisation by
hand, confirm which parameter the module treats as `θ`, and confirm your
reference is expressed in that same parameter. Report only what survives that,
and say explicitly which parameterisation was used on both sides.

## Licence

MIT. By contributing you agree that your contributions are licensed under it.
