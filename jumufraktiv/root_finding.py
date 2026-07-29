"""
root_finding.py

Vectorised root-finding methods (NumPy and JAX) for scalar equations.

Functions:
    - bisection_np: pure bisection (NumPy)
    - newton_np: pure Newton (NumPy)
    - bisectioned_newton_np: Newton with bisection fallback (NumPy, recommended)
    - bisection_jax: pure bisection (JAX)
    - newton_jax: pure Newton (JAX)
    - bisectioned_newton_jax: Newton with bisection fallback (JAX)

Dispatcher:
    - solve_root: choose method based on root_method argument.
"""

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


# ======================================================================
# NumPy implementations
# ======================================================================

def bisection_np(
    f,
    lower: np.ndarray,
    upper: np.ndarray,
    maxiter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Vectorised bisection method (NumPy).

    Parameters
    ----------
    f : callable
        Function f(x) whose roots are sought. Must accept and return NumPy arrays.
    lower : np.ndarray
        Lower bounds of the search intervals (must satisfy f(lower) < 0).
    upper : np.ndarray
        Upper bounds of the search intervals (must satisfy f(upper) > 0).
    maxiter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Absolute tolerance for |f(x)|.

    Returns
    -------
    np.ndarray
        Roots such that f(root) ≈ 0.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape:
        raise ValueError("lower and upper must have the same shape.")

    for _ in range(maxiter):
        mid = (lower + upper) / 2.0
        fmid = f(mid)

        # Update brackets
        condition = fmid < 0
        lower = np.where(condition, mid, lower)
        upper = np.where(~condition, mid, upper)

        # Convergence check
        if np.all(np.abs(fmid) < tol):
            break

    return (lower + upper) / 2.0


def newton_np(
    f,
    df,
    x0: np.ndarray,
    maxiter: int = 50,
    tol: float = 1e-8,
    rel_tol: float = 1e-8,
) -> np.ndarray:
    """
    Vectorised pure Newton method (NumPy).

    Parameters
    ----------
    f : callable
        Function f(x) whose roots are sought.
    df : callable
        Derivative f'(x).
    x0 : np.ndarray
        Initial guesses.
    maxiter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Absolute tolerance for |f(x)|.
    rel_tol : float, optional
        Relative tolerance for change in x.

    Returns
    -------
    np.ndarray
        Roots such that f(root) ≈ 0.
    """
    x = np.asarray(x0, dtype=float)
    for _ in range(maxiter):
        fx = f(x)
        dfx = df(x)

        # Avoid division by zero
        dfx_safe = np.where(np.abs(dfx) > 1e-300, dfx, 1e-300)

        dx = fx / dfx_safe
        x_new = x - dx

        # Check convergence
        converged = (np.abs(fx) < tol) | (np.abs(dx) < rel_tol * np.maximum(1.0, np.abs(x)))
        x = x_new
        if np.all(converged):
            break

    return x


def bisectioned_newton_np(
    f,
    df,
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    maxiter: int = 50,
    tol: float = 1e-8,
    rel_tol: float = 1e-8,
) -> np.ndarray:
    """
    Vectorised safeguarded Newton method (NumPy) – Newton with bisection fallback.

    This is the recommended method for general root finding. It maintains
    brackets [lower, upper] and uses Newton steps when they stay within
    the bracket and the derivative is sufficiently large; otherwise, it
    falls back to bisection. This combines the fast local convergence of
    Newton with the robustness of bisection.

    Parameters
    ----------
    f : callable
        Function f(x) whose roots are sought.
    df : callable
        Derivative f'(x).
    x0 : np.ndarray
        Initial guesses (must lie within [lower, upper]).
    lower : np.ndarray
        Lower bounds of the search intervals (must satisfy f(lower) < 0).
    upper : np.ndarray
        Upper bounds of the search intervals (must satisfy f(upper) > 0).
    maxiter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Absolute tolerance for |f(x)|.
    rel_tol : float, optional
        Relative tolerance for change in x.

    Returns
    -------
    np.ndarray
        Roots such that f(root) ≈ 0.
    """
    x = np.asarray(x0, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if not (lower.shape == upper.shape == x.shape):
        raise ValueError("lower, upper, and x0 must have the same shape.")

    for _ in range(maxiter):
        fx = f(x)
        dfx = df(x)

        # Compute Newton step
        dfx_safe = np.where(np.abs(dfx) > 1e-300, dfx, 1e-300)
        dx = fx / dfx_safe
        x_newton = x - dx

        # Fallback to bisection if Newton step is outside bracket or derivative is too small
        outside_bracket = (x_newton <= lower) | (x_newton >= upper)
        derivative_too_small = np.abs(dfx) < 1e-300
        use_bisection = outside_bracket | derivative_too_small
        x_new = np.where(use_bisection, (lower + upper) / 2.0, x_newton)

        # Evaluate f at new point
        fx_new = f(x_new)

        # Update bracket
        condition = fx_new < 0
        lower = np.where(condition, x_new, lower)
        upper = np.where(~condition, x_new, upper)

        # Update x
        x_old = x
        x = x_new

        # Convergence check
        converged = (np.abs(fx_new) < tol) | (np.abs(x - x_old) < rel_tol * np.maximum(1.0, np.abs(x_old)))
        if np.all(converged):
            break

    return x


# ======================================================================
# JAX implementations
# ======================================================================

def bisection_jax(
    f,
    lower,
    upper,
    maxiter=100,
    tol=1e-8,
):
    """
    JAX vectorised bisection method.

    Parameters
    ----------
    f : callable
        JAX-compatible function.
    lower : array
        Lower brackets, f(lower)<0.
    upper : array
        Upper brackets, f(upper)>0.

    Returns
    -------
    array
        Approximate roots.
    """
    lower = jnp.asarray(lower, dtype=jnp.float64)
    upper = jnp.asarray(upper, dtype=jnp.float64)

    def body(_, state):
        lower, upper = state
        mid = (lower + upper) / 2.0
        fmid = f(mid)
        condition = fmid < 0
        lower = jnp.where(condition, mid, lower)
        upper = jnp.where(~condition, mid, upper)
        return lower, upper

    lower, upper = jax.lax.fori_loop(0, maxiter, body, (lower, upper))
    return (lower + upper) / 2.0


def newton_jax(
    f,
    df,
    x0,
    maxiter=50,
    tol=1e-8,
    rel_tol=1e-8,
):
    """
    JAX vectorised pure Newton method.
    """
    x0 = jnp.asarray(x0, dtype=jnp.float64)

    def body(_, x):
        fx = f(x)
        dfx = df(x)
        dfx_safe = jnp.where(jnp.abs(dfx) > 1e-300, dfx, 1e-300)
        dx = fx / dfx_safe
        return x - dx

    return jax.lax.fori_loop(0, maxiter, body, x0)


def bisectioned_newton_jax(
    f,
    df,
    x0,
    lower,
    upper,
    maxiter=50,
    tol=1e-8,
    rel_tol=1e-8,
):
    """
    JAX vectorised safeguarded Newton method.

    Newton steps are accepted only when they remain inside
    the bracket; otherwise bisection is used.
    """
    x = jnp.asarray(x0, dtype=jnp.float64)
    lower = jnp.asarray(lower, dtype=jnp.float64)
    upper = jnp.asarray(upper, dtype=jnp.float64)

    def body(_, state):
        x, lower, upper = state
        fx = f(x)
        dfx = df(x)

        dfx_safe = jnp.where(jnp.abs(dfx) > 1e-300, dfx, 1e-300)
        dx = fx / dfx_safe
        x_newton = x - dx

        outside = (x_newton <= lower) | (x_newton >= upper)
        derivative_small = jnp.abs(dfx) < 1e-300
        use_bisection = outside | derivative_small

        x_new = jnp.where(use_bisection, (lower + upper) / 2.0, x_newton)
        fx_new = f(x_new)

        condition = fx_new < 0
        lower = jnp.where(condition, x_new, lower)
        upper = jnp.where(~condition, x_new, upper)

        return x_new, lower, upper

    x, lower, upper = jax.lax.fori_loop(0, maxiter, body, (x, lower, upper))
    return x


# ======================================================================
# Dispatcher
# ======================================================================
def solve_root(
    f,
    df=None,
    x0=None,
    lower=None,
    upper=None,
    root_method="auto",
    maxiter=100,
    tol=1e-8,
    rel_tol=1e-8,
    verbose=False,
):
    """
    Unified dispatcher for root finding.

    Parameters
    ----------
    f : callable
        Function f(x) whose roots are sought.
    df : callable, optional
        Derivative f'(x). Required for Newton‑based methods.
    x0 : array-like, optional
        Initial guesses. Required for Newton‑based methods.
    lower : array-like, optional
        Lower brackets. Required for bisection‑based methods.
    upper : array-like, optional
        Upper brackets. Required for bisection‑based methods.
    root_method : str, optional
        One of:
            "auto"                      – try methods in order, skipping those with missing args.
            "bisectioned-newton-jax"
            "newton-jax"
            "bisection-jax"
            "bisectioned-newton-np"
            "newton-np"
            "bisection-np"
    maxiter : int, optional
        Maximum iterations.
    tol : float, optional
        Absolute tolerance for |f(x)|.
    rel_tol : float, optional
        Relative tolerance (Newton only).
    verbose : bool, optional
        If True, print messages about which methods are tried and whether they succeed.

    Returns
    -------
    array
        Roots.
    """
    # ---- Validate root_method ----
    allowed_methods = [
        "auto",
        "bisectioned-newton-jax",
        "newton-jax",
        "bisection-jax",
        "bisectioned-newton-np",
        "newton-np",
        "bisection-np",
    ]
    if root_method not in allowed_methods:
        raise ValueError(
            f"root_method must be one of {allowed_methods}, got '{root_method}'."
        )

    # ---- Helper to try a method and return result or None ----
    def try_method(method_name):
        try:
            if method_name == "bisectioned-newton-jax":
                res = bisectioned_newton_jax(f, df, x0, lower, upper, maxiter, tol, rel_tol)
            elif method_name == "newton-jax":
                res = newton_jax(f, df, x0, maxiter, tol, rel_tol)
            elif method_name == "bisection-jax":
                res = bisection_jax(f, lower, upper, maxiter, tol)
            elif method_name == "bisectioned-newton-np":
                res = bisectioned_newton_np(f, df, x0, lower, upper, maxiter, tol, rel_tol)
            elif method_name == "newton-np":
                res = newton_np(f, df, x0, maxiter, tol, rel_tol)
            elif method_name == "bisection-np":
                res = bisection_np(f, lower, upper, maxiter, tol)
            else:
                return None
            # Sanity check: if result equals initial guess (within 1e-12) and |f(res)| > tol*10, treat as failure
            if x0 is not None and np.allclose(res, x0, rtol=1e-12, atol=1e-12):
                f_res = f(res)
                if np.any(np.abs(f_res) > tol * 10):
                    # Not converged; treat as failure
                    return None
            return res
        except Exception:
            return None

    # ---- If explicit method, try it directly ----
    if root_method != "auto":
        if verbose:
            print(f"Trying method: {root_method}...")
        result = try_method(root_method)
        if result is not None:
            if verbose:
                print(f"Method {root_method} succeeded.")
            return result
        else:
            raise RuntimeError(f"Method '{root_method}' failed.")

    # ---- "auto" mode: try methods in order ----
    # Define ordered list
    methods = [
        "bisectioned-newton-jax",
        "newton-jax",
        "bisection-jax",
        "bisectioned-newton-np",
        "newton-np",
        "bisection-np",
    ]
    # Filter out methods requiring df if df is None
    if df is None:
        methods = [m for m in methods if not m in ("bisectioned-newton-jax", "newton-jax", "bisectioned-newton-np", "newton-np")]
    # Filter out methods requiring brackets if brackets are missing
    if lower is None or upper is None:
        methods = [m for m in methods if not m in ("bisectioned-newton-jax", "bisection-jax", "bisectioned-newton-np", "bisection-np")]
    # Filter out methods requiring x0 if x0 is None (Newton methods)
    if x0 is None:
        methods = [m for m in methods if not m in ("bisectioned-newton-jax", "newton-jax", "bisectioned-newton-np", "newton-np")]

    # Try each method in order
    for method_name in methods:
        if verbose:
            print(f"Trying method: {method_name}...")
        result = try_method(method_name)
        if result is not None:
            if verbose:
                print(f"Method {method_name} succeeded.")
            return result
        else:
            if verbose:
                print(f"Method {method_name} failed, trying next.")

    raise RuntimeError("All auto methods failed.")

# ======================================================================
# Example usage (run only when script is executed directly)
# ======================================================================
if __name__ == "__main__":
    import math

    # ---- Test function: f(x) = x - cos(x), root ≈ 0.739085 ----
    def f(x):
        return x - np.cos(x)

    def df(x):
        return 1.0 + np.sin(x)

    # ---- JAX versions for JAX tests ----
    import jax.numpy as jnp
    def f_jax(x):
        return x - jnp.cos(x)
    def df_jax(x):
        return 1.0 + jnp.sin(x)

    # ---- Initial guesses and brackets ----
    x0 = np.array([0.5])      # initial guess
    lower = np.array([0.0])   # f(0) = -1 < 0
    upper = np.array([1.0])   # f(1) = 0.4597 > 0

    # Convert to JAX arrays for JAX methods
    x0_jax = jnp.array([0.5])
    lower_jax = jnp.array([0.0])
    upper_jax = jnp.array([1.0])

    # ---- List of methods to test ----
    methods = [
        "bisectioned-newton-jax",
        "newton-jax",
        "bisection-jax",
        "bisectioned-newton-np",
        "newton-np",
        "bisection-np",
    ]

    print("=" * 60)
    print("Testing root-finding methods on f(x) = x - cos(x)")
    print("True root ≈ 0.7390851332151606")
    print("=" * 60)

    for method in methods:
        # Select the appropriate function variants
        if "jax" in method:
            f_use = f_jax
            df_use = df_jax
            x0_use = x0_jax
            lower_use = lower_jax
            upper_use = upper_jax
        else:
            f_use = f
            df_use = df
            x0_use = x0
            lower_use = lower
            upper_use = upper

        try:
            root = solve_root(
                f=f_use,
                df=df_use if "newton" in method or "bisectioned" in method else None,
                x0=x0_use if "newton" in method or "bisectioned" in method else None,
                lower=lower_use if "bisection" in method or "bisectioned" in method else None,
                upper=upper_use if "bisection" in method or "bisectioned" in method else None,
                root_method=method,
                maxiter=50,
                tol=1e-12,
                rel_tol=1e-12,
            )
            # Convert to float if JAX array
            if hasattr(root, 'item'):
                root = root.item()
            else:
                root = float(root[0])
            error = abs(root - 0.7390851332151606)
            print(f"{method:>25}: root = {root:.12f}, error = {error:.2e}")
        except Exception as e:
            print(f"{method:>25}: FAILED ({e})")

    # ---- Test without derivative: auto should fall back to bisection ----
    print("\n" + "=" * 60)
    print("Test without derivative (auto → bisection methods)")
    print("=" * 60)
    try:
        root_auto = solve_root(
            f=f,
            df=None,
            lower=lower,
            upper=upper,
            root_method="auto",
            maxiter=100,
            tol=1e-12,
        )
        root_auto = float(root_auto[0])
        error = abs(root_auto - 0.7390851332151606)
        print(f"auto (no df): root = {root_auto:.12f}, error = {error:.2e}")
    except Exception as e:
        print(f"auto (no df): FAILED ({e})")