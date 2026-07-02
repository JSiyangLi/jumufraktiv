import math
import sympy as sp
import jax.numpy as jnp
from scipy.stats import gamma as scipy_gamma

from jumufraktiv.logsum import logplus, logminus
from jumufraktiv.registry import register_prior, make_prior_spec
from jumufraktiv.symbols import t, theta, param


# ============================================================
# Canonical symbolic parameters (shared system)
# ============================================================
alpha = param("alpha")
beta = param("beta")


# ============================================================
# Symbolic expressions (used for symbolic mode)
# ============================================================

def gamma_mgf_symbolic():
    return (beta / (beta - t)) ** alpha

def gamma_cgf_symbolic():
    return alpha * (sp.log(beta) - sp.log(beta - t))

def gamma_pdf_symbolic():
    return (beta**alpha / sp.gamma(alpha)) * theta**(alpha - 1) * sp.exp(-beta * theta)


# ============================================================
# Numeric CGF / MGF (log-space stable core)
# ============================================================

def gamma_cgf(t_val: float, alpha_val: float, beta_val: float) -> float:
    if t_val >= beta_val:
        raise ValueError(f"t must be < beta ({beta_val})")
    log_beta = math.log(beta_val)
    log_beta_minus_t = math.log(beta_val - t_val)
    log_ratio = logminus(log_beta, log_beta_minus_t)
    return alpha_val * log_ratio

def gamma_mgf(t_val: float, alpha_val: float, beta_val: float) -> float:
    return math.exp(gamma_cgf(t_val, alpha_val, beta_val))


# ============================================================
# JAX versions
# ============================================================

def gamma_cgf_jax(t_val, alpha_val, beta_val):
    return alpha_val * (jnp.log(beta_val) - jnp.log(beta_val - t_val))

def gamma_mgf_jax(t_val, alpha_val, beta_val):
    return jnp.exp(gamma_cgf_jax(t_val, alpha_val, beta_val))


# ============================================================
# SciPy PDF / logPDF
# ============================================================

def gamma_pdf(theta_val: float, alpha_val: float, beta_val: float) -> float:
    return scipy_gamma(a=alpha_val, scale=1.0 / beta_val).pdf(theta_val)

def gamma_logpdf(theta_val: float, alpha_val: float, beta_val: float) -> float:
    return scipy_gamma(a=alpha_val, scale=1.0 / beta_val).logpdf(theta_val)


# ============================================================
# Registry factory
# ============================================================

@register_prior("gamma")
def gamma_factory(params):
    alpha_val = float(params["alpha"])
    beta_val = float(params["beta"])

    # Build symbolic expressions using the global symbols
    mgf_sym = (beta / (beta - t)) ** alpha
    cgf_sym = alpha * (sp.log(beta) - sp.log(beta - t))
    pdf_sym = (beta**alpha / sp.gamma(alpha)) * theta**(alpha - 1) * sp.exp(-beta * theta)

    # Substitute numeric parameter values into the symbolic expressions
    subs_map = {alpha: alpha_val, beta: beta_val}
    mgf_sym = mgf_sym.subs(subs_map)
    cgf_sym = cgf_sym.subs(subs_map)
    pdf_sym = pdf_sym.subs(subs_map)

    # Return the spec using make_prior_spec
    return make_prior_spec(
        mgf_sym=mgf_sym,
        cgf_sym=cgf_sym,
        pdf_sym=pdf_sym,

        mgf=lambda t_val: (beta_val / (beta_val - t_val)) ** alpha_val,
        cgf=lambda t_val: alpha_val * (math.log(beta_val) - math.log(beta_val - t_val)),

        mgf_jax=lambda t_val: (beta_val / (beta_val - t_val)) ** alpha_val,
        cgf_jax=lambda t_val: alpha_val * (jnp.log(beta_val) - jnp.log(beta_val - t_val)),

        pdf_func=lambda x: scipy_gamma(a=alpha_val, scale=1/beta_val).pdf(x),
        logpdf_func=lambda x: scipy_gamma(a=alpha_val, scale=1/beta_val).logpdf(x),

        params=params,
    )