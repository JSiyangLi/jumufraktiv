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
    Evaluate a fixed-order derivative at one or more evaluation points.

    The evaluation point is:
        - complete MGF: (t)
        - incomplete MGF: (t, u)

    If either t or u is array-like, the function vectorises over the
    combined batch of evaluation points (tuple‑vectorisation principle).

    Parameters
    ----------
    order : int
        Must be scalar.
    t : scalar or array-like
        Evaluation point(s) for t.
    u : scalar or array-like, optional
        Upper limit(s) for the incomplete MGF.

        Together with t, defines the evaluation point (t,u).
        t and u are broadcast using NumPy broadcasting rules before
        vectorised evaluation.

    Returns
    -------
    (log_abs, sign)
        Scalars if both inputs are scalar.
        Arrays with the broadcasted shape if either input is array-like.
    """
    if np.ndim(order) != 0:
        raise ValueError(
            "integerDeriv_numeric_jax only accepts a scalar order. "
            "Vectorisation over derivative orders is handled by mgfDerivative_integer()."
        )

    # ---- Complete MGF: evaluation point is (t) ----
    if complete:
        if np.ndim(t) == 0:
            # Scalar fast path
            return _integerDeriv_numeric_jax_scalar(
                float(t), prior, order, complete=True, u=None
            )
        # Vectorise over t
        t_arr = jnp.asarray(t)
        vmapped = jax.vmap(
            lambda t_val: _integerDeriv_numeric_jax_scalar(
                t_val, prior, order, complete=True, u=None
            )
        )
        log_abs, sign = vmapped(t_arr)
        return np.asarray(log_abs), np.asarray(sign)

    # ---- Incomplete MGF: evaluation point is (t, u) ----
    if u is None:
        raise ValueError("u must be provided for incomplete MGF")

    # Broadcast t and u to a common shape (tuple‑vectorisation)
    t_arr = np.asarray(t)
    u_arr = np.asarray(u)
    t_broad, u_broad = np.broadcast_arrays(t_arr, u_arr)

    # Flatten to 1D for vectorisation over points
    t_flat = jnp.asarray(t_broad).reshape(-1)
    u_flat = jnp.asarray(u_broad).reshape(-1)

    vmapped = jax.vmap(
        lambda t_val, u_val: _integerDeriv_numeric_jax_scalar(
            t_val, prior, order, complete=False, u=u_val
        )
    )
    log_abs, sign = vmapped(t_flat, u_flat)

    # Reshape back to the broadcasted shape
    log_abs = np.asarray(log_abs).reshape(t_broad.shape)
    sign = np.asarray(sign).reshape(t_broad.shape)

    return log_abs, sign


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

    # ---------------------------------------------------------
    # Vectorised tests for u (incomplete MGF)
    # ---------------------------------------------------------
    if gamma_prior.has_iMGF():
        print("\n" + "=" * 60)
        print("Vectorised u tests (incomplete MGF)")
        print("=" * 60)

        order_imgf = 2
        t_scalar = 0.1

        # 1) Scalar t, array u
        print("\n--- Scalar t, array u ---")
        u_arr = np.array([0.5, 1.0, 2.0, 5.0])
        log_abs_sTuA, sign_sTuA = integerDeriv_numeric_jax(
            t_scalar, gamma_prior, order_imgf, complete=False, u=u_arr
        )
        print(f"t = {t_scalar}, u = {u_arr}")
        print(f"log|deriv|: {log_abs_sTuA}")
        print(f"sign: {sign_sTuA}")

        # 2) Array t, scalar u
        print("\n--- Array t, scalar u ---")
        u_scalar = 2.0
        log_abs_aTsU, sign_aTsU = integerDeriv_numeric_jax(
            t_vals, gamma_prior, order_imgf, complete=False, u=u_scalar
        )
        print(f"t = {t_vals}, u = {u_scalar}")
        print(f"log|deriv|: {log_abs_aTsU}")
        print(f"sign: {sign_aTsU}")

        # 3) Array t, array u (elementwise)
        print("\n--- Array t, array u (elementwise) ---")
        t_arr = np.linspace(-0.2, 0.2, 4)
        u_arr_zipped = np.array([0.5, 1.0, 2.0, 5.0])
        log_abs_aTaU, sign_aTaU = integerDeriv_numeric_jax(
            t_arr, gamma_prior, order_imgf, complete=False, u=u_arr_zipped
        )
        print(f"t = {t_arr}")
        print(f"u = {u_arr_zipped}")
        print(f"log|deriv|: {log_abs_aTaU}")
        print(f"sign: {sign_aTaU}")

        # 4) Consistency check: vectorised result == loop of scalar calls
        print("\n--- Consistency check (vectorised vs scalar loop) ---")
        log_abs_loop = []
        sign_loop = []
        for tt, uu in zip(t_arr, u_arr_zipped):
            la, s = integerDeriv_numeric_jax(
                tt, gamma_prior, order_imgf, complete=False, u=uu
            )
            log_abs_loop.append(float(la))
            sign_loop.append(int(s))

        log_abs_loop = np.array(log_abs_loop)
        sign_loop = np.array(sign_loop)

        max_diff_log = np.max(np.abs(log_abs_aTaU - log_abs_loop))
        signs_match = np.array_equal(sign_aTaU, sign_loop)

        print(f"Max log-abs difference: {max_diff_log:.2e}")
        print(f"Signs match: {signs_match}")
        if max_diff_log < 1e-10 and signs_match:
            print("  ✅ Vectorised u matches scalar loop.")
        else:
            print("  ⚠️ Mismatch between vectorised and scalar loop.")

    else:
        print("\nSkipping incomplete-MGF tests: prior does not provide imgf_jax.")