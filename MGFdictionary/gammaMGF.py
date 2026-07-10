import math
import sympy as sp
import jax.numpy as jnp
import numpy as np
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
    log_beta = np.log(beta_val)
    log_beta_minus_t = np.log(beta_val - t_val)
    log_ratio = logminus(log_beta, log_beta_minus_t)
    return alpha_val * log_ratio

def gamma_mgf(t_val: float, alpha_val: float, beta_val: float) -> float:
    return np.exp(gamma_cgf(t_val, alpha_val, beta_val))


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
# Incomplete MGF (lower truncation at u) for Gamma distribution
# ============================================================
#   M(t; α, β, u) = ∫_0^u e^(tθ) p(θ) dθ
#                 = (β/(β−t))^α * γ(α, (β−t)u) / Γ(α)
#   where t < β.
# ============================================================

# ---- Symbolic ----
def gamma_imgf_symbolic(u_sym):
    """Symbolic expression for the incomplete MGF."""
    return (beta / (beta - t)) ** alpha * (
        sp.lowergamma(alpha, (beta - t) * u_sym) / sp.gamma(alpha)
    )

def gamma_logimgf_symbolic(u_sym):
    """Symbolic log-incomplete MGF."""
    return sp.log(gamma_imgf_symbolic(u_sym))


# ---- Numeric (SciPy) ----
from scipy.special import gammainc

def gamma_imgf(t_val, alpha_val, beta_val, u_val):
    """
    Numeric incomplete MGF (vectorised).
    Returns the ordinary-scale value.
    """
    s = beta_val - t_val
    if np.any(s <= 0):
        raise ValueError("t must be strictly less than beta for all elements")
    reg_gamma = gammainc(alpha_val, s * u_val)   # γ(α, x)/Γ(α)
    return (beta_val / s) ** alpha_val * reg_gamma

def gamma_logimgf(t_val, alpha_val, beta_val, u_val):
    """
    Numeric log-incomplete MGF (vectorised).
    Returns log of the incomplete MGF.
    """
    s = beta_val - t_val
    if np.any(s <= 0):
        raise ValueError("t must be strictly less than beta for all elements")
    log_factor = alpha_val * (np.log(beta_val) - np.log(s))
    reg_gamma = gammainc(alpha_val, s * u_val)
    # log(reg_gamma) is -inf where reg_gamma == 0; that is correct.
    return log_factor + np.log(reg_gamma)


# ---- JAX ----
import jax.numpy as jnp
from jax.scipy.special import gammainc as jax_gammainc

def gamma_imgf_jax(t_val, alpha_val, beta_val, u_val):
    """JAX version of the incomplete MGF (JIT‑compatible, vectorised)."""
    s = beta_val - t_val
    reg_gamma = jax_gammainc(alpha_val, s * u_val)
    return (beta_val / s) ** alpha_val * reg_gamma

def gamma_logimgf_jax(t_val, alpha_val, beta_val, u_val):
    """JAX version of the log-incomplete MGF (JIT‑compatible, vectorised)."""
    s = beta_val - t_val
    log_factor = alpha_val * (jnp.log(beta_val) - jnp.log(s))
    reg_gamma = jax_gammainc(alpha_val, s * u_val)
    return log_factor + jnp.log(reg_gamma)


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
        cgf=lambda t_val: alpha_val * (np.log(beta_val) - np.log(beta_val - t_val)),

        mgf_jax=lambda t_val: (beta_val / (beta_val - t_val)) ** alpha_val,
        cgf_jax=lambda t_val: alpha_val * (jnp.log(beta_val) - jnp.log(beta_val - t_val)),

        pdf_func=lambda x: scipy_gamma(a=alpha_val, scale=1/beta_val).pdf(x),
        logpdf_func=lambda x: scipy_gamma(a=alpha_val, scale=1/beta_val).logpdf(x),
        
        # ---- Incomplete MGF (truncated at u) ----
        # These are extra callables; they require both t and u.
        imgf=lambda t_val, u_val: gamma_imgf(t_val, alpha_val, beta_val, u_val),
        logimgf=lambda t_val, u_val: gamma_logimgf(t_val, alpha_val, beta_val, u_val),
        imgf_jax=lambda t_val, u_val: gamma_imgf_jax(t_val, alpha_val, beta_val, u_val),
        logimgf_jax=lambda t_val, u_val: gamma_logimgf_jax(t_val, alpha_val, beta_val, u_val),

        params=params,
    )