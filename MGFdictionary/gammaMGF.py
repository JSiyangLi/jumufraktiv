import math
from logsum import logplus, logminus
import sympy as sp
import jax.numpy as jnp   # optional, but we'll add it
import torch
from jumufraktiv.registry import register_prior
@register_prior("gamma")

def gamma_mgf_symbolic():
    """
    Return a SymPy expression for the gamma MGF:
        M(t) = (beta / (beta - t)) ** alpha
    """
    t, alpha, beta = sp.symbols('t alpha beta', positive=True, real=True)
    return (beta / (beta - t)) ** alpha

def gamma_cgf_symbolic():
    """
    Returns symbolic expression for the cumulant generating function (CGF)
    of the Gamma(alpha, rate=beta) distribution:
        K(t) = log M(t) = alpha * (log(beta) - log(beta - t))
    """
    t, alpha, beta = sp.symbols('t alpha beta', positive=True, real=True)
    return alpha * (sp.log(beta) - sp.log(beta - t))

def gamma_cgf(t: float, alpha: float, beta: float) -> float:
    """
    Log moment generating function of Gamma(alpha, rate=beta).
    log M(t) = alpha * log(beta) - alpha * log(beta - t),  for t < beta.
    
    This version uses logminus() to compute log(beta - t) in log space.
    """
    if t >= beta:
        raise ValueError(f"t must be less than beta ({beta})")
    
    # Convert inputs to log scale
    log_beta = math.log(beta)
    log_nt = math.log(-t)
    
    # Compute log(beta - t) = log(exp(log_beta) - exp(log_t))
    log_diff = logplus(log_beta, log_nt)   # logminus returns NaN if log_beta <= log_t (t>=beta)
    
    if math.isnan(log_diff):
        # This shouldn't happen because we already checked t < beta,
        # but guard against numerical issues.
        raise ValueError("Numerical error in logminus")
    
    # log(M) = alpha * (log_beta - log_diff)
    log_mgf = alpha * (log_beta - log_diff)
    
    return log_mgf

def gamma_mgf(t: float, alpha: float, beta: float) -> float:
    """
    Wrapper to return the MGF in normal scale.
    """
    log_mgf = gamma_cgf(t, alpha, beta)
    return math.exp(log_mgf)

def gamma_cgf_jax(t, alpha, beta):
    """JAX version of log M(t)."""
    return alpha * (jnp.log(beta) - jnp.log(beta - t))

def gamma_mgf_jax(t, alpha, beta):
    return jnp.exp(gamma_cgf_jax(t, alpha, beta))

def gamma_mgf_torch(t, alpha, beta):
    """Torch version of M(t) for Gamma(alpha, rate=beta)."""
    # Ensure alpha and beta are tensors with matching dtype/device
    alpha_t = torch.tensor(alpha, dtype=t.dtype, device=t.device)
    beta_t = torch.tensor(beta, dtype=t.dtype, device=t.device)
    return torch.exp(alpha_t * (torch.log(beta_t) - torch.log(beta_t - t)))

def gamma_pdf_symbolic():
    """
    Return a SymPy expression for the Gamma(alpha, beta) density at theta:
        p(theta) = beta^alpha / Gamma(alpha) * theta^(alpha-1) * exp(-beta*theta)
    """
    theta, alpha, beta = sp.symbols('theta alpha beta', positive=True, real=True)
    return (beta**alpha / sp.gamma(alpha)) * theta**(alpha - 1) * sp.exp(-beta * theta)

def gamma_pdf_symbolic_sub(params):
    """
    Return the symbolic Gamma PDF with parameters substituted.
    params must contain 'alpha' and 'beta'.
    """
    alpha = params['alpha']
    beta = params['beta']
    # We could just call gamma_pdf_symbolic and substitute, but we can also build directly.
    # Using the base function ensures consistency.
    return gamma_pdf_symbolic().subs({sp.Symbol('alpha'): alpha, sp.Symbol('beta'): beta})