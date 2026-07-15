import jax
import jax.numpy as jnp
from jax.experimental import jet
import math
import numpy as np

jax.config.update("jax_enable_x64", True)

def _integerDeriv_numeric_jax_scalar(
    t,
    prior,
    order,
    complete: bool = True,
    u=None,
    jax_mode: str = "auto"
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
        val = expr(t)

        if abs(val) < 1e-300:
            return -jnp.inf, 1
        sign = jnp.where(val >= 0, 1, -1)
        return jnp.log(jnp.abs(val)), sign
    
    # ---------------------------------------------------------
    # Select JAX differentiation strategy
    # ---------------------------------------------------------
    if jax_mode in ("auto", "jet"):

        try:
            series_in = ((1.0,) + (0.0,) * (order - 1),)

            _, series_out = jet.jet(
                expr,
                (t,),
                series_in,
            )

            coef = series_out[order - 1]

        except Exception as e:

            if jax_mode == "jet":
                raise

            # auto mode: fallback to grad only for jet failures
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

            coef = deriv(t)


    elif jax_mode == "grad":

        deriv = expr
        for _ in range(order):
            deriv = jax.grad(deriv)

        coef = deriv(t)


    else:
        raise ValueError(
            f"Unknown jax_mode='{jax_mode}'. "
            "Expected 'auto', 'jet', or 'grad'."
        )

    # ---------------------------------------------------------
    # Return result
    # ---------------------------------------------------------
    eps = 1e-300

    is_zero = jnp.abs(coef) < eps

    log_abs = jnp.where(
        is_zero,
        -jnp.inf,
        jnp.log(jnp.abs(coef))
    )

    sign = jnp.where(
        is_zero,
        1,
        jnp.where(coef >= 0, 1, -1)
    )

    return log_abs, sign


def integerDeriv_numeric_jax(t, prior, order, complete=True, u=None):
    """
    Evaluate a fixed-order derivative at one or more t values.

    Parameters
    ----------
    order : int
        Must be scalar.
        
    t : scalar or array-like
        Evaluation point(s).

    Returns
    -------
    (log_abs, sign)
        Scalars if t is scalar.
        Arrays if t is array-like.
    """
    
    # This backend only vectorises over t.
    if np.ndim(order) != 0:
        raise ValueError(
            "integerDeriv_numeric_jax only accepts a scalar order. "
            "Vectorisation over derivative orders is handled by mgfDerivative_integer()."
        )

    # scalar
    if np.ndim(t) == 0:
        return _integerDeriv_numeric_jax_scalar(
            float(t),
            prior,
            order,
            complete=complete,
            u=u
        )

    # vector
    t = jnp.asarray(t)

    vmapped = jax.vmap(
        lambda x: _integerDeriv_numeric_jax_scalar(
            x,
            prior,
            order,
            complete=complete,
            u=u
        )
    )

    log_abs, sign = vmapped(t)

    return np.asarray(log_abs), np.asarray(sign)


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
    
    # ---------------------------------------------------------
    # Vectorised test: evaluate derivative at multiple t values
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Vectorised test: evaluate derivative at an array of t")
    print("=" * 60)

    t_vals = np.linspace(-0.5, 0.5, 5)
    order_vec = 2

    log_abs_vec, sign_vec = integerDeriv_numeric_jax(t_vals, gamma_prior, order_vec)

    print(f"t values: {t_vals}")
    print(f"log|deriv|: {log_abs_vec}")
    print(f"sign: {sign_vec}")

    # Optional: test incomplete MGF if supported
    if gamma_prior.has_iMGF():
        print("\nVectorised incomplete MGF test:")
        u_val = 2.0
        log_abs_imgf, sign_imgf = integerDeriv_numeric_jax(
            t_vals, gamma_prior, order_vec, complete=False, u=u_val
        )
        print(f"log|deriv| (iMGF): {log_abs_imgf}")
        print(f"sign (iMGF): {sign_imgf}")