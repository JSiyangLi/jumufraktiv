import jax
import jax.numpy as jnp
from jax.experimental import jet
import math

jax.config.update("jax_enable_x64", True)

def integerDeriv_numeric_jax(
    t,
    prior,
    order,
    complete: bool = True,
    u=None
):

    if order < 0:
        raise ValueError("Order must be non-negative.")

    # ---------------------------------------------------------
    # Select function
    # ---------------------------------------------------------
    if complete:
        if prior.mgf_jax is None:
            raise ValueError("Prior does not provide a JAX-compatible MGF.")
        expr = prior.mgf_jax

    else:
        if u is None:
            raise ValueError("u must be supplied for incomplete MGF.")
        if prior.imgf_jax is None:
            raise ValueError("Prior does not provide a JAX-compatible incomplete MGF.")

        expr = lambda t_val: prior.imgf_jax(t_val, u)

    # ---------------------------------------------------------
    # Zeroth derivative
    # ---------------------------------------------------------
    if order == 0:
        val = float(expr(t))

        if abs(val) < 1e-300:
            return -float("inf"), 1

        return math.log(abs(val)), 1 if val > 0 else -1

    # ---------------------------------------------------------
    # Try Jet first
    # ---------------------------------------------------------
    try:

        series_in = ((1.0,) + (0.0,) * (order - 1),)

        _, series_out = jet.jet(
            expr,
            (t,),
            series_in,
        )

        coef = float(series_out[order - 1])

    except Exception as e:

        msg = str(e).lower()

        unsupported = (
            isinstance(e, KeyError)
            or "jet" in msg
            or "primitive" in msg
            or "not implemented" in msg
            or "igamma" in msg
        )

        if not unsupported:
            raise

        print(f"⚠️ Jet failed ({type(e).__name__}: {e}). Falling back to grad().")

        deriv = expr
        for _ in range(order):
            deriv = jax.grad(deriv)

        coef = float(deriv(t))

    # ---------------------------------------------------------
    # Return result
    # ---------------------------------------------------------
    if abs(coef) < 1e-300:
        return -float("inf"), 1

    return math.log(abs(coef)), (1 if coef > 0 else -1)


if __name__ == "__main__":
    import time
    import math
    import jumufraktiv.MGFdictionary
    from jumufraktiv.mitMGFprior_class import mitMGFprior  # assumed new class

    print("=" * 60)
    print("Testing integerDeriv_numeric_jax (JAX jet on mitMGFprior)")
    print("=" * 60)

    # ---------------------------------------------------------
    # Build Gamma prior object (new architecture)
    # ---------------------------------------------------------
    gamma_prior = mitMGFprior.from_registry(
        "gamma",
        params={'alpha': 2.0, 'beta': 3.0}
    )
    
    t_val = 1.0

    gamma_prior = mitMGFprior.from_registry(
        "gamma",
        params={"alpha": 2.0, "beta": 3.0},
    )

    print(gamma_prior.mgf_jax)
    print(type(gamma_prior.mgf_jax))
    print(callable(gamma_prior.mgf_jax))

    # ---------------------------------------------------------
    # Low-order derivatives
    # ---------------------------------------------------------
    for n in range(4):
        log_abs, sign = integerDeriv_numeric_jax(t_val, gamma_prior, n)
        print(f"Gamma M^{{{n}}}({t_val}) : log|.| = {log_abs:.4f}, sign = {sign}")

    # ---------------------------------------------------------
    # High-order test
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("High-order test: Gamma M^15 with alpha=beta=1e-5, t=-1e3")
    print("=" * 60)

    gamma_small = mitMGFprior.from_registry(
        "gamma",
        params={'alpha': 1e-5, 'beta': 1e-5}
    )

    t_small = -1e3
    order_high = 15

    start = time.time()
    log_abs, sign = integerDeriv_numeric_jax(t_small, gamma_small, order_high)
    elapsed = time.time() - start

    print(f"  log|deriv| = {log_abs:.6e}, sign = {sign}")
    print(f"  Time = {elapsed:.4f} seconds")

    # ---------------------------------------------------------
    # Analytical check (Gamma MGF derivative identity)
    # ---------------------------------------------------------
    alpha = 1e-5
    beta = 1e-5

    log_falling = math.lgamma(alpha + order_high) - math.lgamma(alpha)

    log_expected = (
        log_falling
        + alpha * math.log(beta)
        - (alpha + order_high) * math.log(beta - t_small)
    )

    print(f"  Analytical log|deriv| = {log_expected:.6e}")
    print(f"  Difference = {log_abs - log_expected:.2e}")

    if abs(log_abs - log_expected) < 1e-6:
        print("  ✅ Matches analytical formula.")
    else:
        print("  ⚠️ Difference not negligible – check precision.")

    # ---------------------------------------------------------
    # Special function sanity checks (unchanged)
    # ---------------------------------------------------------
    import jax.scipy.special as jsp
    import scipy.special as sc

    alpha_test = 3.5
    z = 0.5

    print("\nSpecial function comparison:")
    print("JAX:", jsp.gammaincc(-alpha_test, z))
    print("SciPy:", sc.gammaincc(-alpha_test, z))