"""Reproducible cost measurements for the vectorisation work (audit PR 9).

Run directly. Pytest does not collect it: the filename does not match
`test_*.py`, so it sits under `tests/` without joining the suite. Timings do
not belong in the suite -- they vary with the machine and would either be
flaky or so loose as to assert nothing.

    python tests/benchmarks/bench_vectorisation.py

Two quantities matter and they are different questions.

**Cost per evaluation point.** The tuple-vectorisation principle says a batch
of points is evaluated as one batch. If that holds, cost per point *falls* as
the batch grows. If the implementation loops, it stays flat. Flat is the
symptom this benchmark exists to catch.

**Cost of one density call.** Every quadrature node calls the prior's density,
so a constant overhead there multiplies by thousands. Measuring it separately
is what distinguishes "the loop is the problem" from "each iteration is the
problem" -- and for this package it was the second.
"""

import time

import numpy as np

from jumufraktiv import mgfDerivative, registry
from jumufraktiv.MGFPrior_class import MGFPrior

PRIORS = {
    "gamma": {"alpha": 2.0, "beta": 3.0},
    "uniform": {"a": 0.5, "b": 2.0},
    "pareto": {"alpha": 3.0, "xi": 1.0},
    "heaviside": {"k": 0.1},
}


def _prior(name):
    registry.initialize()
    return MGFPrior.from_registry(name, params=PRIORS[name])


def density_call_cost(repeats=1000):
    """Microseconds per `logpdf_func` call, per prior."""
    print("Cost of one density call")
    print(f"  {'prior':12s} {'logpdf_func':>14s}  {'vectorises?':>12s}")
    print("  " + "-" * 42)
    x1 = np.array([1.0])
    x3 = np.array([0.5, 1.0, 2.0])
    for name in PRIORS:
        prior = _prior(name)
        start = time.perf_counter()
        for _ in range(repeats):
            prior.logpdf_func(x1)
        per_call = (time.perf_counter() - start) / repeats * 1e6
        try:
            prior.logpdf_func(x3)
            vectorises = "yes"
        except Exception as exc:
            vectorises = type(exc).__name__
        print(f"  {name:12s} {per_call:11.1f} us  {vectorises:>12s}")
    print()


def per_point_cost(name="gamma", order=1.5, method="auto", sizes=(1, 5, 20)):
    """Milliseconds per evaluation point, as the batch grows.

    Flat means the implementation loops; falling means it batches.
    """
    prior = _prior(name)
    print(f"Cost per evaluation point ({name}, order {order}, method={method!r})")
    print(f"  {'points':>7s} {'total':>10s} {'per point':>11s}")
    print("  " + "-" * 31)
    for n in sizes:
        points = np.linspace(-1.0, -5.0, n)
        start = time.perf_counter()
        mgfDerivative(order, prior, method=method, t=points, log=True)
        elapsed = time.perf_counter() - start
        print(f"  {n:7d} {elapsed * 1e3:9.1f} ms {elapsed / n * 1e3:9.1f} ms")
    print()


if __name__ == "__main__":
    density_call_cost()
    per_point_cost(method="auto")
