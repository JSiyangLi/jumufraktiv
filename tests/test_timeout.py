"""Tests for the symbolic-transform timeout helper.

These assert *elapsed time*, not just that the exception type is right. An
earlier version raised ``FunctionTimedOut`` on schedule but then blocked inside
the executor's context-manager exit until the runaway call finished — so a test
that only checked the exception passed while the timeout did nothing.
"""

import time

import pytest

from jumufraktiv.symbolic_fractionalDeriv import FunctionTimedOut, func_timeout


def test_fast_call_returns_its_value():
    assert func_timeout(5.0, lambda: 6 * 7) == 42


def test_arguments_are_forwarded():
    assert func_timeout(5.0, lambda a, b: a + b, args=(2, 3)) == 5


def test_slow_call_raises():
    with pytest.raises(FunctionTimedOut, match="budget"):
        func_timeout(0.1, lambda: time.sleep(3.0))


def test_timeout_returns_promptly():
    """The call must return near its budget, not near the worker's runtime.

    This is the assertion that matters. Waiting for the worker would make the
    timeout purely decorative.
    """
    budget, worker_runtime = 0.2, 3.0

    start = time.perf_counter()
    with pytest.raises(FunctionTimedOut):
        func_timeout(budget, lambda: time.sleep(worker_runtime))
    elapsed = time.perf_counter() - start

    assert elapsed < worker_runtime / 2, (
        f"returned after {elapsed:.2f}s for a {budget}s budget against a "
        f"{worker_runtime}s call — the timeout is being defeated by waiting "
        f"on the worker"
    )


def test_exception_from_the_callable_propagates():
    """A genuine failure must surface as itself, not as a timeout."""
    with pytest.raises(ZeroDivisionError):
        func_timeout(5.0, lambda: 1 / 0)
