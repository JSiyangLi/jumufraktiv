import math
import sympy as sp
import jax
import jax.numpy as jnp
from jumufraktiv.mitMGFprior_class import mitMGFprior
import jumufraktiv.MGFdictionary # necessary to import priors!

def main():

    print("=" * 60)
    print("Example 1: Symbolic construction")
    print("=" * 60)

    t = sp.Symbol("t")
    theta = sp.Symbol("theta")

    alpha = 2.0
    beta = 3.0

    mgf_sym = (beta / (beta - t)) ** alpha

    pdf_sym = (
        beta**alpha
        / sp.gamma(alpha)
        * theta ** (alpha - 1)
        * sp.exp(-beta * theta)
    )

    prior1 = (
        mitMGFprior(
            name="custom_gamma_symbolic",
            mgf_sym=mgf_sym,
            pdf_sym=pdf_sym,
            params={"alpha": alpha, "beta": beta},
        )
        .as_mitMGFprior()
    )

    print("MGF(1) =", prior1.mgf(1.0))
    print("CGF(1) =", prior1.cgf(1.0))
    print("PDF(0.5) =", prior1.pdf_func(0.5))

    grad_cgf = jax.grad(prior1.cgf_jax)
    print("CGF'(1) =", grad_cgf(1.0))

    print("Object is mitMGFprior:", mitMGFprior.is_mitMGFprior(prior1))

    print("\n" + "=" * 60)
    print("Example 2: Backend construction")
    print("=" * 60)

    def gamma_backend(t, xp=math, special=None, alpha=2.0, beta=3.0):
        return (beta / (beta - t)) ** alpha

    def gamma_pdf_backend(theta, xp=math, special=None, alpha=2.0, beta=3.0):

        if special is None:
            import math as special

        gamma_alpha = special.gamma(alpha)

        return (beta**alpha / gamma_alpha) * xp.exp(
            (alpha - 1) * xp.log(theta) - beta * theta
        )

    prior2 = (
        mitMGFprior(
            name="custom_gamma_backend",
            mgf_backend=gamma_backend,
            pdf_backend=gamma_pdf_backend,
            params={"alpha": alpha, "beta": beta},
        )
        .as_mitMGFprior()
    )

    print("MGF(1) =", prior2.mgf(1.0))
    print("CGF(1) =", prior2.cgf(1.0))
    print("PDF(0.8) =", prior2.pdf_func(0.8))
    grad_cgf = jax.grad(prior2.cgf_jax)
    print("CGF'(1) =", grad_cgf(1.0))

    print("Object is mitMGFprior:", mitMGFprior.is_mitMGFprior(prior2))

    print("\n" + "=" * 60)
    print("Example 3: Registry construction")
    print("=" * 60)

    prior3 = mitMGFprior.from_registry(
        "gamma",
        params={"alpha": alpha, "beta": beta},
    )

    print("MGF(1) =", prior3.mgf(1.0))
    print("CGF(1) =", prior3.cgf(1.0))
    print("MGF_JAX(1) =", prior3.mgf_jax(1.0))
    print("PDF(0.5) =", prior3.pdf_func(0.5))

    print("\n" + "=" * 60)
    print("Availability checks")
    print("=" * 60)

    print("Has symbolic MGF:",
          prior3.mgf_sym is not None)
    print("Has JAX MGF:",
          prior3.mgf_jax is not None)
    print("Has PDF:",
          prior3.pdf_func is not None)

    print("Object is mitMGFprior:",
          mitMGFprior.is_mitMGFprior(prior3))


if __name__ == "__main__":
    main()