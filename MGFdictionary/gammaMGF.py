import math
from logsum import logminus
import sympy as sp

def gamma_mgf_symbolic():
    """
    Return a SymPy expression for the gamma MGF:
        M(t) = (beta / (beta - t)) ** alpha
    """
    t, alpha, beta = sp.symbols('t alpha beta', positive=True, real=True)
    return (beta / (beta - t)) ** alpha

def log_gamma_mgf(t: float, alpha: float, beta: float) -> float:
    """
    Log moment generating function of Gamma(alpha, rate=beta).
    log M(t) = alpha * log(beta) - alpha * log(beta - t),  for t < beta.
    
    This version uses logminus() to compute log(beta - t) in log space.
    """
    if t >= beta:
        raise ValueError(f"t must be less than beta ({beta})")
    
    # Convert inputs to log scale
    log_beta = math.log(beta)
    log_t = math.log(t)
    
    # Compute log(beta - t) = log(exp(log_beta) - exp(log_t))
    log_diff = logminus(log_beta, log_t)   # logminus returns NaN if log_beta <= log_t (t>=beta)
    
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
    log_mgf = log_gamma_mgf(t, alpha, beta)
    return math.exp(log_mgf)